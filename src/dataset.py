"""
Patient Sequence Dataset for ATLAS-EHR.

Loads pre-processed .pt files with dual-chain structure:
    state_tokens, state_values, state_times
    inter_tokens, inter_values, inter_times
    labels (dict)

Temporal logic:
    Phase 1 (SSL): timestep-indexed state sequence over unique state times.
"""

import csv
import json
import os
import glob
import hashlib
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning.pytorch as pl
from typing import Dict, Any, List, Optional, Tuple, Sequence




# ── Constants ────────────────────────────────────────────────────────────────



class LengthBucketBatchSampler(Sampler[List[int]]):
    """Fixed-size, reproducible length buckets with strict DDP partitioning."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int = 42,
        bucket_multiplier: int = 50,
        costs: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_multiplier = int(bucket_multiplier)
        if costs is not None and len(costs) != len(self.lengths):
            raise ValueError("costs must have one entry per sequence length")
        self.costs = [tuple(map(int, item)) for item in costs] if costs is not None else None
        self.epoch = 0
    @staticmethod
    def _world_info() -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        rank, world_size = self._world_info()
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        order.sort(key=lambda index: self.costs[index] if self.costs is not None else self.lengths[index])
        bucket_size = max(self.batch_size, self.batch_size * self.bucket_multiplier)
        buckets = [order[start:start + bucket_size] for start in range(0, len(order), bucket_size)]
        shuffled: List[List[int]] = []
        for bucket in buckets:
            perm = torch.randperm(len(bucket), generator=generator).tolist()
            shuffled.extend([bucket[i] for i in perm])
        batches = [shuffled[start:start + self.batch_size] for start in range(0, len(shuffled), self.batch_size)]
        # Bucketing controls padding *within* a batch; it must not impose a
        # short-to-long curriculum across an epoch.  Shuffle complete batches
        # before rank partitioning so DDP remains disjoint and balanced.
        batch_perm = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[index] for index in batch_perm]
        if world_size > 1:
            if len(batches) % world_size:
                raise RuntimeError(
                    "Length bucketing requires an equal number of full batches per DDP rank; "
                    "adjust dataset size or batch_size rather than duplicating/overlapping samples."
                )
            batches = batches[rank::world_size]
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        n = int(math.ceil(len(self.lengths) / self.batch_size))
        _, world_size = self._world_info()
        if world_size > 1 and n % world_size:
            raise RuntimeError("Unequal DDP bucket batch count; see LengthBucketBatchSampler.")
        return n // world_size


class ExactDistributedEvalSampler(Sampler[int]):
    """Non-overlapping DDP evaluation sampler without padding duplicates."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @staticmethod
    def _world_info() -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self):
        rank, world_size = self._world_info()
        yield from range(rank, len(self.dataset), world_size)

    def __len__(self) -> int:
        rank, world_size = self._world_info()
        return max(0, (len(self.dataset) - rank + world_size - 1) // world_size)


# ── PatientSequenceDataset ───────────────────────────────────────────────────

class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        schema_dir: str,
        split: str,
        max_seq_len_limit: Optional[int] = None,
        max_set_size_limit: int = 64,
        cutoff_min: Optional[float] = None,
        cutoff_relative_to_study_start: bool = False,
        dedup_policy: str = "none",
        # Phase-1 numeric frequency-guided masking.
        mask_target_ratio: float = 0.15,
        mask_rarity_strength: float = 0.40,
        mask_min_prob: float = 0.05,
        mask_max_prob: float = 0.50,
        static_mask_prob: float = 0.15,
        static_numeric_vids: Optional[Sequence[int]] = (5921,),
        # Normalization (Safety net only — values arrive pre-normalized)
        value_limit: float = 10.0,
        # Filtering
        target_ids: List[str] = None,
        max_samples: int = None,
        train_pct: float = 1.0,
        fraction: float = 1.0,
        deterministic: bool = False,
        return_legacy_nested: bool = True,
        shape_manifest_dir: Optional[str] = None,
        # Task
        phase: int = 1,
        target_task: str = "mortality",
    ):
        super().__init__()
        self.split_dir = os.path.join(data_dir, split)
        self.schema_dir = schema_dir
        self.split = split
        self.max_seq_len_limit = max_seq_len_limit
        self.max_set_size_limit = max_set_size_limit
        self.cutoff_min = None if cutoff_min is None else float(cutoff_min)
        self.cutoff_relative_to_study_start = bool(cutoff_relative_to_study_start)
        if str(dedup_policy) not in {"none", "exact", "vid_time_mean"}:
            raise ValueError(f"Unsupported dedup_policy: {dedup_policy}")
        self.dedup_policy = str(dedup_policy)
        self.mask_target_ratio = float(mask_target_ratio)
        self.mask_rarity_strength = float(mask_rarity_strength)
        self.mask_min_prob = float(mask_min_prob)
        self.mask_max_prob = float(mask_max_prob)
        self.static_mask_prob = static_mask_prob
        self.static_numeric_vids = tuple(int(vid) for vid in (static_numeric_vids or ()))
        if not 0.0 <= self.mask_target_ratio <= 1.0:
            raise ValueError("mask_target_ratio must lie in [0, 1].")
        if not 0.0 <= self.mask_min_prob <= self.mask_max_prob <= 1.0:
            raise ValueError("Require 0 <= mask_min_prob <= mask_max_prob <= 1.")
        if not 0.0 <= float(self.static_mask_prob) <= 1.0:
            raise ValueError("static_mask_prob must lie in [0, 1].")
        if self.mask_rarity_strength < 0.0:
            raise ValueError("mask_rarity_strength must be non-negative.")
        # Normalization (safety net)
        self.value_limit = value_limit
        self.deterministic = bool(deterministic)
        self.return_legacy_nested = bool(return_legacy_nested)
        self.shape_manifest_dir = shape_manifest_dir or os.path.join(data_dir, ".atlas_phase1_shape_manifest")
        self.fraction = float(fraction)
        if int(phase) != 1:
            raise ValueError("PatientSequenceDataset only supports phase=1.")
        self.phase = 1
        self.target_task = target_task

        # ── File Discovery (Flat: {split}/{hadm_id}.pt) ─────────────────
        print(f"[{split.upper()}] Scanning {self.split_dir} for patients...")
        all_files = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))

        # 1. ID Filter (Optional)
        requested_target_set: Optional[set[str]] = None
        if target_ids is not None:
            target_set = set(str(tid) for tid in target_ids)
            requested_target_set = target_set
            self.files = [fp for fp in all_files
                          if os.path.basename(fp).replace(".pt", "") in target_set]
            found_ids = {os.path.basename(fp).replace(".pt", "") for fp in self.files}
            missing_ids = sorted(target_set - found_ids)
            if missing_ids:
                raise FileNotFoundError(
                    f"[{split.upper()}] Requested target IDs missing from {self.split_dir}: "
                    f"{missing_ids[:20]}{' ...' if len(missing_ids) > 20 else ''}. "
                    "Cross-split lookup is forbidden."
                )
        else:
            self.files = all_files

        # 2. Native Capping
        if max_samples is not None and max_samples > 0:
            self.files = self.files[:max_samples]

        # 3. Percentage Capping
        if train_pct < 1.0:
            num_to_keep = max(1, int(len(self.files) * train_pct))
            print(f"[{split.upper()}] Subsetting to {train_pct*100:.1f}% "
                  f"({num_to_keep}/{len(self.files)})")
            self.files = self.files[:num_to_keep]

        # 4. Uniform split fraction capping
        if self.fraction < 1.0:
            num_to_keep = max(1, int(len(self.files) * self.fraction))
            print(f"[{split.upper()}] Fraction subsetting to {self.fraction*100:.1f}% "
                  f"({num_to_keep}/{len(self.files)})")
            self.files = self.files[:num_to_keep]

        if requested_target_set is not None:
            final_ids = {os.path.basename(fp).replace(".pt", "") for fp in self.files}
            if final_ids != requested_target_set:
                missing_ids = sorted(requested_target_set - final_ids)
                raise RuntimeError(
                    "target_ids must be preserved exactly; capping/filtering removed requested IDs: "
                    f"{missing_ids[:20]}{' ...' if len(missing_ids) > 20 else ''}."
                )

        if not self.files:
            print(f"[WARN] No .pt files found in {self.split_dir}")
        else:
            print(f"[{split.upper()}] Found {len(self.files)} samples "
                  f"(filtered from {len(all_files)} total).")

        # ── Schema Loading ───────────────────────────────────────────────
        schema_csv_path = os.path.join(schema_dir, "schema_processed.csv")
        if not os.path.exists(schema_csv_path):
            raise FileNotFoundError(
                f"Expected schema_processed.csv at {schema_csv_path}. "
                "Legacy schema.json support has been removed."
            )
        self._build_stats_tables(schema_csv_path)

    # ── Schema Lookup Tables ─────────────────────────────────────────────

    def _build_stats_tables(self, schema_csv_path: str):
        """
        Builds numpy lookup tables from schema_processed.csv for:
        - precomputed numeric frequency-guided masking probabilities
        - Value verification (type)
        """
        MAX_ID = 66000
        self.mask_probability_by_vid = np.zeros(MAX_ID, dtype=np.float64)
        self.type_table = np.full(MAX_ID, 1, dtype=np.int8)  # Default: 1=categorical
        state_rows: List[Dict[str, str]] = []
        with open(schema_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("feature_category") == "state":
                    state_rows.append(row)

        type_map = {"numeric": 0, "categorical": 1}
        loaded = 0
        numeric_counts: Dict[int, float] = {}
        for row in state_rows:
            vid_str = row.get("vid", "")
            try:
                fid = int(vid_str)
            except ValueError:
                continue
            if fid < 0 or fid >= MAX_ID:
                continue

            sample_count_str = row.get("sample_count", "") or "0"
            try:
                sample_count = float(sample_count_str)
            except ValueError:
                sample_count = 0.0
            model_type = str(row.get("model_type", "categorical")).lower()
            self.type_table[fid] = type_map.get(model_type, 1)
            if model_type == "numeric":
                numeric_counts[fid] = max(0.0, sample_count)
            loaded += 1

        unknown_static = sorted(set(self.static_numeric_vids) - set(numeric_counts))
        if unknown_static:
            raise ValueError(
                "static_numeric_vids must refer to numeric state VIDs in schema_processed.csv; "
                f"invalid: {unknown_static}"
            )
        self._initialize_mask_probability_table(numeric_counts)

        print(
            f"[{self.split.upper()}] Schema: {loaded} state features loaded from schema_processed.csv."
        )

    def _initialize_mask_probability_table(self, numeric_counts: Dict[int, float]) -> None:
        """Solve the event-weighted numeric masking contract once at construction."""
        if not numeric_counts:
            raise ValueError("schema_processed.csv has no numeric state rows for Phase-1 masking.")
        vids = np.asarray(sorted(numeric_counts), dtype=np.int64)
        counts = np.asarray([numeric_counts[int(vid)] for vid in vids], dtype=np.float64)
        total_count = float(counts.sum())
        if total_count <= 0.0:
            raise ValueError("Numeric state sample_count values must have positive total mass.")

        static_mask = np.isin(vids, np.asarray(self.static_numeric_vids, dtype=np.int64))
        nonstatic_mask = ~static_mask
        probabilities = np.zeros_like(counts)
        probabilities[static_mask] = float(self.static_mask_prob)
        static_mass = float(counts[static_mask].sum())
        nonstatic_mass = float(counts[nonstatic_mask].sum())
        target_total = float(self.mask_target_ratio) * total_count

        if nonstatic_mass == 0.0:
            achieved = float((counts * probabilities).sum() / total_count)
            if abs(achieved - self.mask_target_ratio) > 1e-6:
                raise ValueError("Static numeric VIDs alone cannot satisfy mask_target_ratio.")
        else:
            required_nonstatic_mean = (
                target_total - static_mass * float(self.static_mask_prob)
            ) / nonstatic_mass
            if not self.mask_min_prob - 1e-12 <= required_nonstatic_mean <= self.mask_max_prob + 1e-12:
                raise ValueError(
                    "mask_target_ratio is infeasible with static_mask_prob and mask probability bounds: "
                    f"required non-static mean={required_nonstatic_mean:.8f}."
                )
            log_counts = np.log1p(counts[nonstatic_mask])
            spread = max(float(log_counts.max() - log_counts.min()), np.finfo(np.float64).eps)
            rarity = 1.0 - (log_counts - log_counts.min()) / spread
            weights = 1.0 + float(self.mask_rarity_strength) * rarity

            def weighted_mean(scale: float) -> float:
                probs = np.clip(scale * weights, self.mask_min_prob, self.mask_max_prob)
                return float((counts[nonstatic_mask] * probs).sum() / nonstatic_mass)

            lo, hi = 0.0, max(1.0, float(self.mask_max_prob) / float(weights.min()))
            while weighted_mean(hi) < required_nonstatic_mean:
                hi *= 2.0
                if hi > 1e12:
                    raise RuntimeError("Unable to bracket numeric masking scale.")
            for _ in range(100):
                mid = (lo + hi) / 2.0
                if weighted_mean(mid) < required_nonstatic_mean:
                    lo = mid
                else:
                    hi = mid
            probabilities[nonstatic_mask] = np.clip(
                hi * weights, self.mask_min_prob, self.mask_max_prob
            )

        achieved = float((counts * probabilities).sum() / total_count)
        if abs(achieved - self.mask_target_ratio) > 1e-6:
            raise RuntimeError(
                f"Numeric event-weighted mask probability contract failed: {achieved} != {self.mask_target_ratio}."
            )
        self.mask_probability_by_vid[vids] = probabilities
        self.numeric_mask_probability_summary = {
            "min": float(probabilities.min()),
            "median": float(np.median(probabilities)),
            "max": float(probabilities.max()),
            "event_weighted_mean": achieved,
        }

    # ── Value Verification (Safety Net) ──────────────────────────────────

    def _verify_values(self, vals: torch.Tensor, name: str) -> None:
        """Assert no NaN; clamp extremes as safety net."""
        has_nan = torch.isnan(vals).any()
        if has_nan:
            raise ValueError(f"[DATASET] NaN in {name}! File may be corrupted.")
        if (vals.abs() > self.value_limit).any():
            vals.clamp_(-self.value_limit, self.value_limit)

    # ── Main Loading Logic ───────────────────────────────────────────────

    def __len__(self):
        return len(self.files)

    def sequence_length_summary(self) -> Dict[str, Any]:
        """Load or atomically build the task-independent Phase-1 shape manifest."""
        file_meta = [
            (os.path.basename(path), os.stat(path).st_size, os.stat(path).st_mtime_ns)
            for path in self.files
        ]
        signature_payload = {
            "split": self.split,
            "data_root": os.path.abspath(self.split_dir),
            "files": file_meta,
            "cutoff_min": self.cutoff_min,
            "cutoff_relative_to_study_start": self.cutoff_relative_to_study_start,
            "max_set_size_limit": self.max_set_size_limit,
            "schema_size": int(len(self.type_table)),
            "version": 1,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest_path = os.path.join(self.shape_manifest_dir, f"{self.split}.pt")
        complete_path = manifest_path + ".complete"

        def load_manifest() -> Optional[Dict[str, Any]]:
            if not (os.path.isfile(manifest_path) and os.path.isfile(complete_path)):
                return None
            manifest = torch.load(manifest_path, weights_only=False)
            if manifest.get("signature") != signature:
                return None
            if len(manifest.get("raw_lengths", [])) != len(self.files):
                return None
            return manifest

        manifest = load_manifest()
        if manifest is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            if rank == 0:
                raw_lengths, max_set_sizes, numeric_events, total_events = [], [], [], []
                for path in self.files:
                    payload = torch.load(path, weights_only=False)
                    times = payload["state_times"]
                    tokens = payload["state_tokens"]
                    if self.cutoff_min is not None:
                        if self.cutoff_relative_to_study_start:
                            start=float(payload.get("study_start_rel_min", 0.0))
                            keep=(times >= start) & (times <= start + float(self.cutoff_min))
                        else:
                            keep = times <= float(self.cutoff_min)
                        times, tokens = times[keep], tokens[keep]
                    if times.numel():
                        _, counts = torch.unique(times, return_counts=True)
                        raw_lengths.append(int(counts.numel()))
                        max_set_sizes.append(int(counts.max()))
                    else:
                        raw_lengths.append(0)
                        max_set_sizes.append(0)
                    safe_tokens = tokens.clamp(0, len(self.type_table) - 1).numpy()
                    numeric_events.append(int((self.type_table[safe_tokens] == 0).sum()))
                    total_events.append(int(tokens.numel()))
                manifest = {
                    "signature": signature,
                    "raw_lengths": raw_lengths,
                    "max_set_sizes": max_set_sizes,
                    "numeric_events": numeric_events,
                    "total_events": total_events,
                }
                os.makedirs(self.shape_manifest_dir, exist_ok=True)
                temp_path = manifest_path + f".tmp.{os.getpid()}"
                torch.save(manifest, temp_path)
                os.replace(temp_path, manifest_path)
                with open(complete_path, "w", encoding="utf-8") as handle:
                    handle.write(signature + "\n")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            manifest = load_manifest()
            if manifest is None:
                raise RuntimeError("Phase-1 shape manifest build completed without a valid manifest.")

        raw_lengths = list(manifest["raw_lengths"])
        self._shape_records = {
            "max_set_sizes": list(manifest["max_set_sizes"]),
            "numeric_events": list(manifest["numeric_events"]),
            "total_events": list(manifest["total_events"]),
            "signature": signature,
            "path": manifest_path,
        }
        raw = np.asarray(raw_lengths, dtype=np.int64)
        self._raw_sequence_lengths = raw_lengths
        cap = self.max_seq_len_limit
        effective = raw if cap is None or int(cap) <= 0 else np.minimum(raw, int(cap))
        summary = {
            "split": self.split,
            "n": int(raw.size),
            "configured_max_seq_len_limit": None if cap is None else int(cap),
            "raw_unique_timestamps": {
                "min": int(raw.min()) if raw.size else 0,
                "median": float(np.median(raw)) if raw.size else 0.0,
                "max": int(raw.max()) if raw.size else 0,
            },
            "effective_unique_timestamps": {
                "min": int(effective.min()) if effective.size else 0,
                "median": float(np.median(effective)) if effective.size else 0.0,
                "max": int(effective.max()) if effective.size else 0,
            },
        }
        print(f"[{self.split.upper()}] sequence cap summary: {summary}")
        return summary

    def cache_extraction_shapes(self) -> Tuple[List[int], List[int]]:
        """Return cached (timestamp count, maximum set size) per admission."""
        self.sequence_length_summary()
        lengths = list(getattr(self, "_raw_sequence_lengths", ()))
        max_set_sizes = list(getattr(self, "_shape_records", {}).get("max_set_sizes", ()))
        if len(lengths) != len(self.files) or len(max_set_sizes) != len(self.files):
            raise RuntimeError("Phase-1 shape manifest is incomplete for cache extraction.")
        return lengths, max_set_sizes

    def _rng_for_file(self, fp: str):
        if not self.deterministic:
            return np.random
        payload = f"{self.split}|{self.phase}|{os.path.basename(fp)}"
        seed = int(hashlib.md5(payload.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(seed)

    @staticmethod
    def _rng_integers(rng, low: int, high: int) -> int:
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))
        return int(rng.randint(low, high))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a .pt file and constructs Phase-1 temporal sequences.
        """
        fp = self.files[idx]
        data = torch.load(fp, weights_only=False)
        rng = self._rng_for_file(fp)

        # ── Raw disk tensors ─────────────────────────────────────────────
        raw_s_tokens = data["state_tokens"]     # (N_s,) Long
        raw_s_values = data["state_values"]     # (N_s,) Float32
        raw_s_times  = data["state_times"]      # (N_s,) Float32
        raw_i_tokens = data["inter_tokens"]     # (N_i,) Long
        raw_i_values = data["inter_values"]     # (N_i,) Float32
        raw_i_times  = data["inter_times"]      # (N_i,) Float32

        if self.dedup_policy != "none" and raw_s_tokens.numel() > 0:
            if self.dedup_policy == "exact":
                seen = set()
                keep = []
                for i, (vid, val, tm) in enumerate(zip(raw_s_tokens.tolist(), raw_s_values.tolist(), raw_s_times.tolist())):
                    key = (int(vid), float(tm), float(val))
                    if key not in seen:
                        seen.add(key)
                        keep.append(i)
                idx = torch.tensor(keep, dtype=torch.long)
                raw_s_tokens, raw_s_values, raw_s_times = raw_s_tokens[idx], raw_s_values[idx], raw_s_times[idx]
            else:
                grouped: Dict[Tuple[int, float], List[float]] = {}
                order: List[Tuple[int, float]] = []
                for vid, val, tm in zip(raw_s_tokens.tolist(), raw_s_values.tolist(), raw_s_times.tolist()):
                    key = (int(vid), float(tm))
                    if key not in grouped:
                        grouped[key] = []
                        order.append(key)
                    grouped[key].append(float(val))
                raw_s_tokens = torch.tensor([x[0] for x in order], dtype=torch.long)
                raw_s_times = torch.tensor([x[1] for x in order], dtype=torch.float32)
                raw_s_values = torch.tensor([float(np.mean(grouped[x])) for x in order], dtype=torch.float32)

        # ── Timeline Construction ───────────────────────────────────────
        # Cutoff is applied before unique-time construction and before the
        # timestamp cap.  This keeps the cap in timestamp space and matches
        # the retrospective MIMIC extraction contract.
        if self.cutoff_min is not None:
            if self.cutoff_relative_to_study_start:
                study_start=float(data.get("study_start_rel_min", 0.0))
                raw_s_keep=(raw_s_times >= study_start) & (raw_s_times <= study_start + float(self.cutoff_min))
            else:
                raw_s_keep = raw_s_times <= float(self.cutoff_min)
            raw_s_tokens = raw_s_tokens[raw_s_keep]
            raw_s_values = raw_s_values[raw_s_keep]
            raw_s_times = raw_s_times[raw_s_keep]
        # Sort once and use consecutive-group offsets below. This preserves the
        # old within-timestamp event order (stable sort) without repeatedly
        # scanning every state event for each timestamp.
        state_order = torch.argsort(raw_s_times, stable=True)
        raw_s_tokens = raw_s_tokens[state_order]
        raw_s_values = raw_s_values[state_order]
        raw_s_times = raw_s_times[state_order]
        unique_times, state_counts = torch.unique_consecutive(raw_s_times, return_counts=True)
        state_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long), state_counts.cumsum(dim=0)
        ])
        T = len(unique_times)

        # ── Delta Times (in-memory computation) ──────────────────────────
        delta_times = torch.zeros(T, dtype=torch.float32)
        if T > 1:
            delta_times[1:] = unique_times[1:] - unique_times[:-1]

        # ── Sequence Cap ─────────────────────────────────────────────────
        # max_seq_len_limit <= 0 means "no cap" (used by ablation extraction to
        # avoid accidental truncation and keep parity with benchmark windows).
        start = 0
        if self.max_seq_len_limit and self.max_seq_len_limit > 0 and T > self.max_seq_len_limit:
            if self.split == "train":
                start = self._rng_integers(rng, 0, T - self.max_seq_len_limit + 1)
            T = self.max_seq_len_limit

        times_window = unique_times[start:start + T]
        deltas_window = delta_times[start:start + T]
        state_offsets_window = state_offsets[start:start + T + 1]

        # Vectorized Phase-1 masking over the complete cropped event window.
        # A single RNG draw preserves the exact timestamp-major draw order of
        # the former per-timestamp implementation, including draws consumed by
        # events later removed by the set-size cap.
        full_counts_window = state_offsets_window[1:] - state_offsets_window[:-1]
        window_lo = int(state_offsets_window[0]) if state_offsets_window.numel() else 0
        window_hi = int(state_offsets_window[-1]) if state_offsets_window.numel() else 0
        window_tokens = raw_s_tokens[window_lo:window_hi]
        window_values = raw_s_values[window_lo:window_hi]
        if window_tokens.numel():
            vid_np = window_tokens.numpy().astype(int)
            safe_vid = np.clip(vid_np, 0, len(self.type_table) - 1)
            is_numeric = self.type_table[safe_vid] == 0
            p_mask = np.zeros(window_tokens.numel(), dtype=np.float32)
            p_mask[is_numeric] = self.mask_probability_by_vid[safe_vid[is_numeric]]
            window_masks = (rng.random(window_tokens.numel()) < p_mask).astype(np.float32)
            window_masks[~is_numeric] = 0.0
            assert not np.any((window_masks > 0.5) & (~is_numeric)), (
                "Phase 1 categorical state events must never be masked"
            )
        else:
            window_masks = np.empty((0,), dtype=np.float32)

        if T:
            group_ids = torch.repeat_interleave(torch.arange(T, dtype=torch.long), full_counts_window)
            group_starts = torch.cumsum(full_counts_window, dim=0) - full_counts_window
            within_set_slot = torch.arange(window_tokens.numel(), dtype=torch.long) - torch.repeat_interleave(
                group_starts, full_counts_window
            )
            keep_event = within_set_slot < self.max_set_size_limit
            selected_counts = torch.bincount(group_ids[keep_event], minlength=T)
        else:
            keep_event = torch.empty((0,), dtype=torch.bool)
            selected_counts = torch.empty((0,), dtype=torch.long)
        selected_tokens = window_tokens[keep_event]
        selected_values = window_values[keep_event]
        selected_masks = torch.from_numpy(window_masks)[keep_event]
        if selected_values.numel():
            self._verify_values(selected_values, "state_window")
        selected_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long), selected_counts.cumsum(dim=0)
        ])

        # ── Per-Timestep Assembly ────────────────────────────────────────
        state_seq = [] if self.return_legacy_nested else None
        # Phase 1 never consumes intervention tensors.  Keep its legacy empty
        # fields for collate compatibility, but do not sort/search/pack every
        # intervention event in the training data path.
        empty_intervention = {
            "tokens": raw_i_tokens.new_empty((0,)),
            "values": raw_i_values.new_empty((0,)),
            "mask": torch.empty((0,), dtype=torch.float32),
            "times": raw_i_times.new_empty((0,)),
            "offset_from_left": torch.empty((0,), dtype=torch.float32),
            "offset_to_right": torch.empty((0,), dtype=torch.float32),
            "interval_dt": torch.empty((0,), dtype=torch.float32),
        }
        interv_seq = [empty_intervention] * T if self.return_legacy_nested else []

        if state_seq is not None:
            for t in range(T):
                s_lo = int(selected_offsets[t])
                s_hi = int(selected_offsets[t + 1])
                state_seq.append({
                    "tokens": selected_tokens[s_lo:s_hi],
                    "values": selected_values[s_lo:s_hi],
                    "mask": selected_masks[s_lo:s_hi],
                })

        # ── Label Extraction ─────────────────────────────────────────────
        labels_dict = data["labels"]
        raw_label = labels_dict[self.target_task]
        if raw_label is None:
            # Phase-1 cache extraction does not consume labels. Preserve an
            # explicit missing sentinel instead of crashing or encoding it as
            # a negative target; formal probe target maps filter missing values.
            label_tensor = torch.tensor(-1, dtype=torch.long if self.target_task in ("mortality", "readmission") else torch.float32)
        elif self.target_task in ("mortality", "readmission"):
            label_tensor = torch.tensor(raw_label, dtype=torch.long)
        else:
            label_tensor = torch.tensor(raw_label, dtype=torch.float32)

        pid = str(data.get("hadm_id",
                           os.path.basename(fp).replace(".pt", "")))

        study_start = data.get("study_start_rel_min", 0.0)

        # Flat timestamp-major views let the Phase-1 collate path scatter once
        # without nested timestamp/slot Python assignment loops.
        state_counts_window = selected_counts
        state_offsets_flat = torch.cat([
            torch.zeros(1, dtype=torch.long), state_counts_window.cumsum(dim=0)
        ])
        state_tokens_flat = selected_tokens
        state_values_flat = selected_values
        state_masks_flat = selected_masks
        inter_counts_window = torch.zeros(T, dtype=torch.long)
        inter_tokens_flat = raw_i_tokens.new_empty((0,))
        inter_values_flat = raw_i_values.new_empty((0,))
        inter_masks_flat = torch.empty((0,), dtype=torch.float32)
        inter_times_flat = raw_i_times.new_empty((0,))
        inter_from_left_flat = torch.empty((0,), dtype=torch.float32)
        inter_to_right_flat = torch.empty((0,), dtype=torch.float32)
        inter_interval_flat = torch.empty((0,), dtype=torch.float32)

        return {
            "state": state_seq if state_seq is not None else [],
            "interv": interv_seq if self.return_legacy_nested else [],
            "state_times": times_window.view(-1, 1),
            "delta_times": deltas_window.view(-1, 1),
            "study_start": study_start,
            "labels": label_tensor,
            "hadm_id": pid,
            "state_counts": state_counts_window,
            "state_offsets": state_offsets_flat,
            "state_tokens_flat": state_tokens_flat,
            "state_values_flat": state_values_flat,
            "state_masks_flat": state_masks_flat,
            "inter_counts": inter_counts_window,
            "inter_tokens_flat": inter_tokens_flat,
            "inter_values_flat": inter_values_flat,
            "inter_masks_flat": inter_masks_flat,
            "inter_times_flat": inter_times_flat,
            "inter_offset_from_left_flat": inter_from_left_flat,
            "inter_offset_to_right_flat": inter_to_right_flat,
            "inter_interval_dt_flat": inter_interval_flat,
        }


# ── Collate Function ─────────────────────────────────────────────────────────

def _nested_padding_collate_reference(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Vectorized collate for dual-chain ATLAS-EHR.

    State chain:   (B, T, S) with per-timestep sets.
    Intervention:  Sparse flat packing (B, Total_I) with temporal anchors.
    """
    if not batch:
        return {}
    if "event_type_id" in batch[0]:
        B = len(batch)
        max_L = max(int(x["event_abs_time"].size(0)) for x in batch)
        event_type_id = torch.zeros((B, max_L), dtype=torch.long)
        event_feature_id = torch.zeros((B, max_L), dtype=torch.long)
        event_value = torch.zeros((B, max_L, 1), dtype=torch.float32)
        event_abs_time = torch.zeros((B, max_L, 1), dtype=torch.float32)
        event_is_state = torch.zeros((B, max_L), dtype=torch.bool)
        event_is_intervention = torch.zeros((B, max_L), dtype=torch.bool)
        event_pad_mask = torch.ones((B, max_L), dtype=torch.bool)
        labels = torch.stack([x["labels"] for x in batch])
        hadm_ids = [str(x["hadm_id"]) for x in batch]
        for b, item in enumerate(batch):
            L = int(item["event_abs_time"].size(0))
            event_type_id[b, :L] = item["event_type_id"].long()
            event_feature_id[b, :L] = item["event_feature_id"].long()
            event_value[b, :L] = item["event_value"].float()
            event_abs_time[b, :L] = item["event_abs_time"].float()
            event_is_state[b, :L] = item["event_is_state"].bool()
            event_is_intervention[b, :L] = item["event_is_intervention"].bool()
            event_pad_mask[b, :L] = False
        return {
            "event_type_id": event_type_id,
            "event_feature_id": event_feature_id,
            "event_value": event_value,
            "event_abs_time": event_abs_time,
            "event_is_state": event_is_state,
            "event_is_intervention": event_is_intervention,
            "event_pad_mask": event_pad_mask,
            "labels": labels,
            "hadm_id": hadm_ids,
        }
    B = len(batch)

    # ── 1. Dynamic Dimensionality ────────────────────────────────────────
    seq_lens = [len(x["state"]) for x in batch]
    max_T = max(seq_lens)

    max_S_s = 0
    pat_interv_counts = []

    for x in batch:
        for t_item in x["state"]:
            max_S_s = max(max_S_s, len(t_item["tokens"]))
        total_i = sum(len(it["tokens"]) for it in x["interv"])
        pat_interv_counts.append(total_i)

    max_S_s = max(max_S_s, 1) + 1  # +1 for Dummy ID at slot 0
    max_Total_I = max(max(pat_interv_counts), 1)

    # ── 2. Allocate Tensors ──────────────────────────────────────────────

    # [A] State Tensors
    state_tokens   = torch.zeros((B, max_T, max_S_s), dtype=torch.long) # Use 0 for [PAD]
    state_values   = torch.zeros(B, max_T, max_S_s, 1)
    state_masks    = torch.zeros(B, max_T, max_S_s, 1)
    state_pad_mask = torch.ones(B, max_T, max_S_s, dtype=torch.bool)

    # [B] Intervention Tensors (Sparse Flat)
    inter_tokens   = torch.zeros((B, max_Total_I), dtype=torch.long) # Use 0 for [PAD]
    inter_values   = torch.zeros(B, max_Total_I, 1)
    inter_masks    = torch.zeros(B, max_Total_I, 1)
    inter_t_step   = torch.zeros(B, max_Total_I, dtype=torch.long)
    inter_order    = torch.zeros(B, max_Total_I, dtype=torch.long)
    inter_pad_mask = torch.ones(B, max_Total_I, dtype=torch.bool)
    inter_times = torch.zeros(B, max_Total_I, 1)
    inter_offset_from_left = torch.zeros(B, max_Total_I, 1)
    inter_offset_to_right = torch.zeros(B, max_Total_I, 1)
    inter_interval_dt = torch.zeros(B, max_Total_I, 1)

    # [C] Time & Metadata
    state_times_out = torch.zeros(B, max_T, 1)
    delta_times_out = torch.zeros(B, max_T, 1)
    study_starts    = torch.zeros(B)
    seq_pad_mask    = torch.ones(B, max_T, dtype=torch.bool)

    labels_out = torch.stack([x["labels"] for x in batch])
    ids_out = [x.get("hadm_id", f"P_unk_{idx}") for idx, x in enumerate(batch)]

    # ── 3. Pack ──────────────────────────────────────────────────────────
    for b, pat in enumerate(batch):
        T_curr = len(pat["state"])
        state_times_out[b, :T_curr] = pat["state_times"]
        delta_times_out[b, :T_curr] = pat["delta_times"]
        study_starts[b] = pat["study_start"]
        seq_pad_mask[b, :T_curr] = False

        # [A] States
        for t in range(T_curr):
            st = pat["state"][t]
            n_st = len(st["tokens"])
            if n_st > 0:
                # Slot 0 is reserved for [PAD] (vid=0)
                state_tokens[b, t, 1:n_st + 1]    = st["tokens"]
                state_values[b, t, 1:n_st + 1, 0]  = st["values"]
                state_masks[b, t, 1:n_st + 1, 0]   = st["mask"]
                state_pad_mask[b, t, 1:n_st + 1]    = False

        # [B] Interventions (Sparse)
        curr_i_idx = 0
        for t in range(T_curr):
            it = pat["interv"][t]
            n_it = len(it["tokens"])
            if n_it > 0:
                room = max_Total_I - curr_i_idx
                if room <= 0:
                    break
                keep_n = min(n_it, room)
                end_idx = curr_i_idx + keep_n
                inter_tokens[b, curr_i_idx:end_idx]      = it["tokens"][:keep_n]
                inter_values[b, curr_i_idx:end_idx, 0]    = it["values"][:keep_n]
                inter_masks[b, curr_i_idx:end_idx, 0]     = it["mask"][:keep_n]
                inter_t_step[b, curr_i_idx:end_idx]       = t
                inter_order[b, curr_i_idx:end_idx]        = torch.arange(1, keep_n + 1)
                inter_pad_mask[b, curr_i_idx:end_idx]     = False
                inter_times[b, curr_i_idx:end_idx, 0]     = it["times"][:keep_n]
                inter_offset_from_left[b, curr_i_idx:end_idx, 0] = it["offset_from_left"][:keep_n]
                inter_offset_to_right[b, curr_i_idx:end_idx, 0] = it["offset_to_right"][:keep_n]
                inter_interval_dt[b, curr_i_idx:end_idx, 0] = it["interval_dt"][:keep_n]
                curr_i_idx = end_idx

    return {
        # State chain
        "state_tokens":      state_tokens,
        "state_values":      state_values,
        "state_mask_labels": state_masks,
        "state_pad_mask":    state_pad_mask,
        # Intervention chain
        "inter_tokens":      inter_tokens,
        "inter_values":      inter_values,
        "inter_mask_labels": inter_masks,
        "inter_t_step":      inter_t_step,
        "inter_order":       inter_order,
        "inter_pad_mask":    inter_pad_mask,
        "inter_times":       inter_times,
        "inter_offset_from_left": inter_offset_from_left,
        "inter_offset_to_right": inter_offset_to_right,
        "inter_interval_dt": inter_interval_dt,
        "hist_inter_tokens": inter_tokens,
        "hist_inter_values": inter_values,
        "hist_inter_t_step": inter_t_step,
        "hist_inter_pad_mask": inter_pad_mask,
        "hist_inter_times": inter_times,
        "hist_inter_offset_from_left": inter_offset_from_left,
        "hist_inter_offset_to_right": inter_offset_to_right,
        "hist_inter_interval_dt": inter_interval_dt,
        # Dynamics
        "state_times":       state_times_out,
        "delta_times":       delta_times_out,
        "study_start":       study_starts,
        "seq_pad_mask":      seq_pad_mask,
        # Labels & ID
        "labels":            labels_out,
        "hadm_id":           ids_out,
    }


def nested_padding_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Vectorized Phase-1 collate preserving the legacy output contract.

    Dataset items expose timestamp-major flat views. The production Phase-1
    path concatenates those views and scatters once; the nested reference
    collate remains above for semantic parity tests.
    """
    if not batch or "event_type_id" in batch[0] or "state_tokens_flat" not in batch[0]:
        return _nested_padding_collate_reference(batch)
    B = len(batch)
    seq_lens = torch.tensor([int(x["state_counts"].numel()) for x in batch], dtype=torch.long)
    max_t = max(int(seq_lens.max()), 1)
    max_s = max(max((int(x["state_counts"].max()) if x["state_counts"].numel() else 0) for x in batch), 1) + 1
    inter_lens = torch.tensor([int(x["inter_tokens_flat"].numel()) for x in batch], dtype=torch.long)
    max_i = max(int(inter_lens.max()), 1)

    state_tokens = torch.zeros((B, max_t, max_s), dtype=torch.long)
    state_values = torch.zeros((B, max_t, max_s, 1), dtype=torch.float32)
    state_masks = torch.zeros((B, max_t, max_s, 1), dtype=torch.float32)
    state_pad_mask = torch.ones((B, max_t, max_s), dtype=torch.bool)
    state_times = pad_sequence([x["state_times"].float() for x in batch], batch_first=True)
    delta_times = pad_sequence([x["delta_times"].float() for x in batch], batch_first=True)
    seq_pad_mask = torch.arange(max_t).unsqueeze(0) >= seq_lens.unsqueeze(1)

    # Patient-level loops only collect flat tensor references and index vectors.
    flat_tokens, flat_values, flat_masks, flat_b, flat_t, flat_s = [], [], [], [], [], []
    for b, item in enumerate(batch):
        counts = item["state_counts"].long()
        n = int(counts.sum())
        if n == 0:
            continue
        offsets = item["state_offsets"].long()[:-1]
        flat_tokens.append(item["state_tokens_flat"].long())
        flat_values.append(item["state_values_flat"].float())
        flat_masks.append(item["state_masks_flat"].float())
        flat_b.append(torch.full((n,), b, dtype=torch.long))
        flat_t.append(torch.repeat_interleave(torch.arange(counts.numel()), counts))
        flat_s.append(torch.arange(n) - torch.repeat_interleave(offsets, counts) + 1)
    if flat_tokens:
        b_idx, t_idx, s_idx = torch.cat(flat_b), torch.cat(flat_t), torch.cat(flat_s)
        state_tokens[b_idx, t_idx, s_idx] = torch.cat(flat_tokens)
        state_values[b_idx, t_idx, s_idx, 0] = torch.cat(flat_values)
        state_masks[b_idx, t_idx, s_idx, 0] = torch.cat(flat_masks)
        state_pad_mask[b_idx, t_idx, s_idx] = False

    # Phase-1 masked-query packing is intentionally performed on CPU while
    # data is still in pinned memory.  This preserves batch/timestamp/slot
    # ordering and removes GPU-side nonzero/bincount/max/scatter setup from
    # every training step.
    masked_idx = (state_masks[..., 0] > 0.5).nonzero(as_tuple=False)
    masked_per_patient = torch.bincount(masked_idx[:, 0], minlength=B) if masked_idx.numel() else torch.zeros(B, dtype=torch.long)
    max_masked = int(masked_per_patient.max()) if masked_per_patient.numel() else 0
    packed_query_vids = torch.zeros((B, max_masked), dtype=torch.long)
    packed_query_t = torch.zeros((B, max_masked), dtype=torch.long)
    packed_query_s = torch.zeros((B, max_masked), dtype=torch.long)
    packed_query_times = torch.zeros((B, max_masked), dtype=torch.float32)
    packed_query_true_values = torch.zeros((B, max_masked), dtype=torch.float32)
    packed_query_pad_mask = torch.ones((B, max_masked), dtype=torch.bool)
    packed_query_indices = masked_idx
    packed_query_ranks = torch.zeros((masked_idx.size(0),), dtype=torch.long)
    if masked_idx.numel():
        starts = torch.cumsum(masked_per_patient, dim=0) - masked_per_patient
        packed_query_ranks = torch.arange(masked_idx.size(0)) - torch.repeat_interleave(starts, masked_per_patient)
        mb, mt, ms = masked_idx.unbind(dim=1)
        packed_query_vids[mb, packed_query_ranks] = state_tokens[mb, mt, ms]
        packed_query_t[mb, packed_query_ranks] = mt
        packed_query_s[mb, packed_query_ranks] = ms
        packed_query_times[mb, packed_query_ranks] = state_times[mb, mt, 0]
        packed_query_true_values[mb, packed_query_ranks] = state_values[mb, mt, ms, 0]
        packed_query_pad_mask[mb, packed_query_ranks] = False

    inter_tokens = torch.zeros((B, max_i), dtype=torch.long)
    inter_values = torch.zeros((B, max_i, 1), dtype=torch.float32)
    inter_masks = torch.zeros((B, max_i, 1), dtype=torch.float32)
    inter_pad_mask = torch.ones((B, max_i), dtype=torch.bool)
    inter_t_step = torch.zeros((B, max_i), dtype=torch.long)
    inter_order = torch.zeros((B, max_i), dtype=torch.long)
    inter_times = torch.zeros((B, max_i, 1), dtype=torch.float32)
    inter_from_left = torch.zeros((B, max_i, 1), dtype=torch.float32)
    inter_to_right = torch.zeros((B, max_i, 1), dtype=torch.float32)
    inter_interval = torch.zeros((B, max_i, 1), dtype=torch.float32)
    for b, item in enumerate(batch):
        n = int(item["inter_tokens_flat"].numel())
        if n == 0:
            continue
        sl = slice(0, n)
        inter_tokens[b, sl] = item["inter_tokens_flat"].long()
        inter_values[b, sl, 0] = item["inter_values_flat"].float()
        inter_masks[b, sl, 0] = item["inter_masks_flat"].float()
        inter_pad_mask[b, sl] = False
        counts = item["inter_counts"].long()
        inter_t_step[b, sl] = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0)[:-1]])
        inter_order[b, sl] = torch.arange(n) - torch.repeat_interleave(offsets, counts) + 1
        inter_times[b, sl, 0] = item["inter_times_flat"].float()
        inter_from_left[b, sl, 0] = item["inter_offset_from_left_flat"].float()
        inter_to_right[b, sl, 0] = item["inter_offset_to_right_flat"].float()
        inter_interval[b, sl, 0] = item["inter_interval_dt_flat"].float()

    labels = torch.stack([x["labels"] for x in batch])
    ids = [x.get("hadm_id", f"P_unk_{idx}") for idx, x in enumerate(batch)]
    return {
        "state_tokens": state_tokens, "state_values": state_values,
        "state_mask_labels": state_masks, "state_pad_mask": state_pad_mask,
        "packed_query_vids": packed_query_vids,
        "packed_query_timestamp_indices": packed_query_t,
        "packed_query_slot_indices": packed_query_s,
        "packed_query_times": packed_query_times,
        "packed_query_true_values": packed_query_true_values,
        "packed_query_pad_mask": packed_query_pad_mask,
        "packed_query_indices": packed_query_indices,
        "packed_query_ranks": packed_query_ranks,
        "masked_targets_per_patient": masked_per_patient,
        "max_masked_targets_in_batch": max_masked,
        "masked_targets_in_batch": int(masked_idx.size(0)),
        "inter_tokens": inter_tokens, "inter_values": inter_values,
        "inter_mask_labels": inter_masks, "inter_t_step": inter_t_step,
        "inter_order": inter_order, "inter_pad_mask": inter_pad_mask,
        "inter_times": inter_times, "inter_offset_from_left": inter_from_left,
        "inter_offset_to_right": inter_to_right, "inter_interval_dt": inter_interval,
        "hist_inter_tokens": inter_tokens, "hist_inter_values": inter_values,
        "hist_inter_t_step": inter_t_step, "hist_inter_pad_mask": inter_pad_mask,
        "hist_inter_times": inter_times, "hist_inter_offset_from_left": inter_from_left,
        "hist_inter_offset_to_right": inter_to_right, "hist_inter_interval_dt": inter_interval,
        "state_times": state_times, "delta_times": delta_times,
        "study_start": torch.tensor([x["study_start"] for x in batch], dtype=torch.float32),
        "seq_pad_mask": seq_pad_mask, "labels": labels, "hadm_id": ids,
    }


# ── LightningDataModule ─────────────────────────────────────────────────────

class JEPADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        schema_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        max_samples: int = None,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        train_pct: float = 1.0,
        fraction: float = 1.0,
        target_ids: List[str] = None,
        length_bucketing: bool = False,
        length_bucket_seed: int = 42,
        length_bucket_multiplier: int = 50,
        shape_manifest_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.schema_dir = schema_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        self.max_samples = max_samples
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.train_pct = train_pct
        self.fraction = float(fraction)
        self.target_ids = target_ids
        self.length_bucketing = bool(length_bucketing)
        self.length_bucket_seed = int(length_bucket_seed)
        self.length_bucket_multiplier = int(length_bucket_multiplier)
        self.shape_manifest_dir = shape_manifest_dir
        self._train_batch_sampler: Optional[LengthBucketBatchSampler] = None
        self.kwargs = kwargs
        allowed_mask_keys = {
            "mask_target_ratio", "mask_rarity_strength", "mask_min_prob",
            "mask_max_prob", "static_mask_prob", "static_numeric_vids",
        }
        unknown_mask_keys = sorted(
            key for key in self.kwargs
            if (key.startswith("mask_") and key not in allowed_mask_keys) or key.startswith("block_")
        )
        if unknown_mask_keys:
            raise ValueError(
                "Unsupported Phase-1 masking parameters: " + ", ".join(unknown_mask_keys)
            )

    def setup(self, stage=None):
        # ``main`` calls setup once to build the model/preflight batch, then
        # Lightning calls it again at fit start. Rebuilding here rescans every
        # split and recomputes the full length manifest without changing any
        # dataset semantics.
        if all(hasattr(self, name) for name in ("train_ds", "val_ds", "test_ds")):
            return
        allowed_mask_keys = {
            "mask_target_ratio", "mask_rarity_strength", "mask_min_prob",
            "mask_max_prob", "static_mask_prob", "static_numeric_vids",
        }
        ds_keys = {"max_seq_len_limit", "max_set_size_limit", "value_limit", *allowed_mask_keys}
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k in ds_keys}
        phase = self.kwargs.get("phase", 1)
        target_task = self.kwargs.get("target_task", "mortality")

        common = dict(
            fraction=self.fraction,
            target_ids=self.target_ids,
            phase=phase,
            target_task=target_task,
            return_legacy_nested=(phase != 1),
            shape_manifest_dir=self.shape_manifest_dir,
            **filtered_kwargs,
        )
        eval_common = dict(common)
        eval_common["deterministic"] = True
        self.train_ds = PatientSequenceDataset(
            self.data_dir, self.schema_dir, "train",
            max_samples=(self.max_train_samples if self.max_train_samples is not None else self.max_samples),
            train_pct=self.train_pct, deterministic=False, **common)
        self.val_ds = PatientSequenceDataset(
            self.data_dir, self.schema_dir, "val",
            max_samples=(self.max_val_samples if self.max_val_samples is not None else self.max_samples), **eval_common)
        self.test_ds = PatientSequenceDataset(
            self.data_dir, self.schema_dir, "test",
            max_samples=(self.max_test_samples if self.max_test_samples is not None else self.max_samples), **eval_common)
        self.sequence_length_summaries = {
            split: dataset.sequence_length_summary()
            for split, dataset in (("train", self.train_ds), ("val", self.val_ds), ("test", self.test_ds))
        }
        if self.length_bucketing:
            records = getattr(self.train_ds, "_shape_records", {})
            raw_lengths = self.train_ds._raw_sequence_lengths
            cap = self.train_ds.max_seq_len_limit
            effective_t = [min(int(length), int(cap)) if cap and int(cap) > 0 else int(length) for length in raw_lengths]
            max_s = records.get("max_set_sizes", [0] * len(raw_lengths))
            numeric_events = records.get("numeric_events", [0] * len(raw_lengths))
            target_ratio = float(filtered_kwargs.get("mask_target_ratio", 0.15))
            costs = [
                (int(t), min(int(s), int(self.train_ds.max_set_size_limit)), int(math.ceil(target_ratio * numeric)))
                for t, s, numeric in zip(effective_t, max_s, numeric_events)
            ]
            self._train_batch_sampler = LengthBucketBatchSampler(
                self.train_ds._raw_sequence_lengths,
                self.batch_size,
                seed=self.length_bucket_seed,
                bucket_multiplier=self.length_bucket_multiplier,
                costs=costs,
            )

    def _loader_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": nested_padding_collate,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(self.num_workers > 0 and self.persistent_workers),
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        """Move tensor fields once via the DataModule's pinned-memory H2D path."""
        del dataloader_idx
        if not isinstance(batch, dict):
            return batch
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_dataloader(self):
        if self._train_batch_sampler is not None:
            return DataLoader(self.train_ds, batch_sampler=self._train_batch_sampler,
                              num_workers=self.num_workers, collate_fn=nested_padding_collate,
                              pin_memory=self.pin_memory,
                              persistent_workers=bool(self.num_workers > 0 and self.persistent_workers),
                              **({"prefetch_factor": self.prefetch_factor} if self.num_workers > 0 and self.prefetch_factor is not None else {}))
        return DataLoader(self.train_ds, shuffle=True, **self._loader_kwargs())

    def on_train_epoch_start(self) -> None:
        if self._train_batch_sampler is not None and self.trainer is not None:
            self._train_batch_sampler.set_epoch(int(self.trainer.current_epoch))

    def val_dataloader(self):
        if self.length_bucketing and torch.distributed.is_available() and torch.distributed.is_initialized():
            return DataLoader(self.val_ds, sampler=ExactDistributedEvalSampler(self.val_ds), **self._loader_kwargs())
        return DataLoader(self.val_ds, shuffle=False, **self._loader_kwargs())

    def test_dataloader(self):
        if self.length_bucketing and torch.distributed.is_available() and torch.distributed.is_initialized():
            return DataLoader(self.test_ds, sampler=ExactDistributedEvalSampler(self.test_ds), **self._loader_kwargs())
        return DataLoader(self.test_ds, shuffle=False, **self._loader_kwargs())
