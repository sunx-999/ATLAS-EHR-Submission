import bisect
import hashlib
import json
import logging
import os
import time
import subprocess
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.utils import get_autocast_dtype, validate_phase1_masking_config


FULL_VISIBLE = "full_visible"
RETROSPECTIVE_CUTOFF = "retrospective_cutoff"
READOUT_TRAINING_CACHE = "readout_training_cache"
# Probe caches are deliberately a separate, strict contract. Older shards
# exposed internal Phase-1 names; never reinterpret those shards for exact-time probes.
CACHE_SCHEMA_VERSION = "phase1_probe_exact_time_v4"
CACHE_CONTRACT_VERSION = "phase1_probe_canonical_contract_v2_packed"

DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_CUTOFFS = (1440, 2880)
_CUTOFF_STAGE_NAMES = {1440: "cutoff_24h", 2880: "cutoff_48h"}

# A completed stage is immutable for the lifetime of a probe process.  Keep
# its structural audit in-process so ``--probe_task all --probe_source all``
# does not reopen every shard for every task/source pair.
_VALIDATED_PROBE_SEQUENCE_STAGES: set = set()
_PROBE_SHARD_SIZE_CACHE: Dict[Tuple[str, int, int, int], int] = {}
_PROBE_DATASET_REGISTRY: Dict[Tuple[str, str, str], "ShardDataset"] = {}

READOUT_CACHE_SCHEMA_VERSION = "phase1_target_manifold_query_v3"
READOUT_CACHE_CONTRACT_VERSION = "phase1_target_manifold_query_contract_v3_identity_locf"
READOUT_QUERY_FIELDS = ["z_pred", "z_target", "target_value", "feature_id", "hadm_id", "t_idx", "slot_idx", "query_key", "locf_value", "locf_available"]
READOUT_QUERY_KEY = ["hadm_id", "t_idx", "slot_idx", "feature_id"]


@dataclass
class BuildStats:
    samples: int = 0
    shards: int = 0
    io_sec: float = 0.0
    io_bytes: int = 0
    build_sec: float = 0.0
    gpu_util_percent: Any = "unavailable"

    def as_dict(self) -> Dict[str, Any]:
        build_sec = max(float(self.build_sec), 1.0e-12)
        io_sec = max(float(self.io_sec), 1.0e-12)
        return {
            "samples": int(self.samples),
            "shards": int(self.shards),
            "io_sec": float(self.io_sec),
            "io_bytes": int(self.io_bytes),
            "build_sec": float(self.build_sec),
            "samples_per_sec": float(self.samples / build_sec),
            "bytes_per_sec": float(self.io_bytes / io_sec),
            "gpu_util_percent": self.gpu_util_percent,
        }


def _gpu_util_percent() -> Any:
    if not torch.cuda.is_available():
        return "unavailable"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        values = [float(line.strip()) for line in output.splitlines() if line.strip()]
        if values:
            return float(np.mean(values))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return "unavailable"


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()




def build_readout_cache_signature(*, data_dir: str, schema_dir: str, cfg_dict: Dict[str, Any], phase1_ckpt_path: str,
                                  teacher: Any, split: str, fraction: float = 1.0,
                                  max_seq_len_limit: Optional[int] = None) -> Dict[str, Any]:
    schema_path = os.path.join(schema_dir, "schema_processed.csv")
    ds_kwargs = _phase1_dataset_kwargs(cfg_dict, max_seq_len_limit=max_seq_len_limit)
    readout_cfg = cfg_dict.get("readout", {})
    masking = {k: ds_kwargs.get(k) for k in ("mask_target_ratio", "mask_rarity_strength", "mask_min_prob", "mask_max_prob", "static_mask_prob", "static_numeric_vids")}
    policy = {k: ds_kwargs.get(k) for k in ("max_seq_len_limit", "max_set_size_limit", "value_limit", "phase", "target_task")}
    if not str(split).strip():
        raise ValueError("readout cache signature split must be non-empty")
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("readout cache signature fraction must satisfy 0 < fraction <= 1")
    if not os.path.isfile(phase1_ckpt_path) or not os.path.isfile(schema_path):
        raise FileNotFoundError("readout cache signature requires existing checkpoint and schema files")
    architecture = getattr(teacher, "phase1_architecture_version", None)
    if not architecture:
        raise RuntimeError("readout cache signature requires a non-empty Phase-1 architecture version")
    signature = {
        "readout_cache_schema_version": READOUT_CACHE_SCHEMA_VERSION,
        "readout_cache_contract_version": READOUT_CACHE_CONTRACT_VERSION,
        "phase1_checkpoint_realpath": os.path.realpath(phase1_ckpt_path),
        "phase1_checkpoint_sha256": _sha256_file(phase1_ckpt_path),
        "phase1_architecture_version": architecture,
        "schema_realpath": os.path.realpath(schema_path),
        "schema_sha256": _sha256_file(schema_path),
        "dataset_content_signature": _dataset_content_signature(data_dir),
        "config_signature": _stable_json_sha256(cfg_dict),
        "readout_masking_signature": _stable_json_sha256(masking),
        "dataset_policy_signature": _stable_json_sha256(policy),
        "fraction": float(fraction), "split": str(split),
        "query_fraction": float(fraction),
        "builder_source_realpath": os.path.realpath(__file__),
        "builder_source_sha256": _sha256_file(__file__),
        "model_source_realpath": os.path.realpath(__import__("src.model", fromlist=["__file__"]).__file__),
        "model_source_sha256": _sha256_file(__import__("src.model", fromlist=["__file__"]).__file__),
        "dataset_source_realpath": os.path.realpath(__import__("src.dataset", fromlist=["__file__"]).__file__),
        "dataset_source_sha256": _sha256_file(__import__("src.dataset", fromlist=["__file__"]).__file__),
        "latent_dimension": int(getattr(teacher, "embed_dim")),
        "masking_configuration": masking,
        "dataset_policy": policy,
        "fields": READOUT_QUERY_FIELDS,
        "query_key": READOUT_QUERY_KEY,
    }
    required = {"readout_cache_schema_version", "readout_cache_contract_version", "phase1_checkpoint_realpath", "phase1_checkpoint_sha256", "phase1_architecture_version", "schema_realpath", "schema_sha256", "dataset_content_signature", "config_signature", "readout_masking_signature", "dataset_policy_signature", "split", "fraction", "builder_source_sha256", "model_source_sha256", "dataset_source_sha256", "latent_dimension"}
    if not isinstance(signature, dict) or not required.issubset(signature):
        raise RuntimeError("readout cache signature construction produced an incomplete payload")
    return signature


def _validate_readout_shard(path: str) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("cache_kind") != READOUT_TRAINING_CACHE:
        raise RuntimeError(f"Invalid readout cache shard header: {path}")
    n = int(payload.get("n", -1))
    if n < 0 or any(key not in payload for key in READOUT_QUERY_FIELDS):
        raise RuntimeError(f"Invalid readout cache shard fields: {path}")
    for key in ("z_pred", "z_target", "target_value", "feature_id", "t_idx", "slot_idx", "locf_value", "locf_available"):
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor) or tensor.size(0) != n:
            raise RuntimeError(f"Invalid readout cache shard shape for {key}: {path}")
        if key in ("z_pred", "z_target", "target_value") and not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"Non-finite readout cache shard values: {path}")
    if len(payload["hadm_id"]) != n or (payload["feature_id"] <= 0).any() or (payload["t_idx"] < 0).any() or (payload["slot_idx"] < 0).any():
        raise RuntimeError(f"Invalid readout cache shard identity fields: {path}")
    if len(payload.get("query_key", [])) != n or len(payload.get("locf_available", [])) != n:
        raise RuntimeError(f"Invalid readout cache identity fields: {path}")
    keys = list(zip(payload["hadm_id"], payload["t_idx"].tolist(), payload["slot_idx"].tolist(), payload["feature_id"].tolist()))
    if [tuple(k) if isinstance(k, (list, tuple)) else k for k in payload["query_key"]] != keys:
        raise RuntimeError(f"Readout query_key does not match identity fields: {path}")
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate readout query key in shard: {path}")
    return n


def validate_readout_query_cache(cache_root: str, split: str, expected_signature: Mapping[str, Any]) -> Dict[str, Any]:
    stage = resolve_path(cache_root, READOUT_TRAINING_CACHE, split=split)
    if not _complete_marker_matches(os.path.join(stage, ".complete")):
        raise RuntimeError(f"Readout cache is incomplete; rebuild the readout cache explicitly: {stage}")
    metadata_path = os.path.join(stage, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise RuntimeError(f"Readout cache metadata is missing; rebuild the readout cache explicitly: {stage}")
    metadata = _read_json(metadata_path)
    provenance_only = {"builder_source_realpath", "builder_source_sha256", "model_source_realpath", "model_source_sha256", "dataset_source_realpath", "dataset_source_sha256", "config_signature"}
    for key, value in expected_signature.items():
        if key in provenance_only:
            continue
        if metadata.get(key) != value:
            raise RuntimeError(f"Readout cache provenance mismatch for {key}; rebuild the readout cache explicitly: {stage}")
    shard_paths = sorted(os.path.join(stage, name) for name in os.listdir(stage) if name.endswith(".pt"))
    total = 0; global_keys = set()
    for path in shard_paths:
        _validate_readout_shard(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        keys = set(zip(payload["hadm_id"], payload["t_idx"].tolist(), payload["slot_idx"].tolist(), payload["feature_id"].tolist()))
        if global_keys.intersection(keys):
            raise RuntimeError(f"Duplicate readout query key across shards; rebuild the readout cache explicitly: {stage}")
        global_keys.update(keys); total += len(keys)
    if int(metadata.get("shard_count", len(shard_paths))) != len(shard_paths) or int(metadata.get("n_queries", -1)) != total:
        raise RuntimeError(f"Readout cache shard/query count mismatch; rebuild the readout cache explicitly: {stage}")
    return metadata


def build_readout_query_cache(
    *,
    cache_root: str,
    data_dir: str,
    schema_dir: str,
    cfg_dict: Dict[str, Any],
    teacher: Any,
    phase1_ckpt_path: str,
    split: str,
    labels_csv: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    shard_size: int = 4096,
    fraction: float = 1.0,
    max_seq_len_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build sparse query-level z_pred/teacher-event cache in one Phase-1 pass.

    The cache intentionally contains no feature type IDs: VID type is fixed by
    the Phase-1 schema and the readout manifold bank validates numeric support.
    """
    root = os.path.abspath(cache_root)
    final_dir = resolve_path(root, READOUT_TRAINING_CACHE, split=split)
    parent_dir = os.path.dirname(final_dir)
    os.makedirs(parent_dir, exist_ok=True)
    out_dir = os.path.join(parent_dir, f".{split}.tmp.{os.getpid()}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ds_kwargs = _phase1_dataset_kwargs(cfg_dict, max_seq_len_limit=max_seq_len_limit)
    signature = build_readout_cache_signature(
        data_dir=data_dir,
        schema_dir=schema_dir,
        cfg_dict=cfg_dict,
        phase1_ckpt_path=phase1_ckpt_path,
        teacher=teacher,
        split=split,
        fraction=fraction,
        max_seq_len_limit=max_seq_len_limit,
    )
    ds = PatientSequenceDataset(
        data_dir=data_dir, schema_dir=schema_dir, split=str(split), train_pct=1.0,
        fraction=float(fraction), deterministic=True, **ds_kwargs,
    )
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size), "shuffle": False, "num_workers": int(num_workers),
        "collate_fn": nested_padding_collate,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": bool(int(num_workers) > 0),
    }
    loader = DataLoader(ds, **kwargs)
    device = next(teacher.parameters()).device
    records: List[Dict[str, Any]] = []
    shard_idx = 0
    total = 0
    seen = set()
    for batch in tqdm(loader, desc=f"readout_query_cache/{split}", ascii=True, mininterval=5.0):
        batch_dev = _batch_to_device(batch, device)
        with torch.inference_mode():
            out = teacher._forward_phase1_numeric_jepa(batch_dev, output_mode="readout_cache")
        valid = (~out["packed_query_pad_mask"]).detach()
        pred = out["z_pred_packed"].detach()[valid].float().cpu()
        target_z = out["target_z_event_packed"].detach()[valid].float().cpu()
        vals = out["packed_true_values"].detach()[valid].float().cpu()
        packed_idx = out["packed_indices"].detach()
        fids = batch_dev["state_tokens"][packed_idx[:, 0], packed_idx[:, 1], packed_idx[:, 2]].long().cpu()
        tids = out["packed_timestamp_indices"].detach()[valid].long().cpu()
        bidx = out["packed_batch_indices"].detach().long().cpu()
        slot = out["packed_slot_indices"].detach()[valid].long().cpu()
        if not (pred.size(0) == target_z.size(0) == vals.size(0) == fids.size(0) == tids.size(0) == bidx.size(0) == slot.size(0)):
            raise RuntimeError(f"readout packed query tensors are misaligned in split={split}")
        if (bidx < 0).any() or (bidx >= len(batch.get("hadm_id", []))).any():
            raise RuntimeError(f"readout packed batch indices are out of range in split={split}")
        # ``packed_indices`` is already compacted to one row per query;
        # unlike the padded packed tensors it must not be indexed by the
        # [B,Q] valid mask a second time.
        packed_valid = packed_idx.long().cpu()
        if not torch.equal(tids, packed_valid[:, 1]) or not torch.equal(slot, packed_valid[:, 2]):
            raise RuntimeError(f"readout packed query identity disagrees with packed_indices in split={split}")
        hadm = [str(x) for x in batch.get("hadm_id", [])]
        # Strict-past LOCF: only observations at an earlier timestamp index
        # from the same admission and VID are eligible.  The target slot and
        # same-timestamp observations are never used.
        locf_values: List[float] = []
        locf_available: List[bool] = []
        state_tokens = batch["state_tokens"].detach().cpu()
        state_values = batch["state_values"].detach().cpu().float()
        state_pad = batch["state_pad_mask"].detach().cpu().bool()
        for bi, ti, fi in zip(bidx.tolist(), tids.tolist(), fids.tolist()):
            prior = state_tokens[int(bi), :int(ti)]
            prior_values = state_values[int(bi), :int(ti)]
            prior_pad = state_pad[int(bi), :int(ti)]
            mask = (prior == int(fi)) & (~prior_pad)
            if bool(mask.any()):
                latest_t = int(mask.nonzero(as_tuple=False)[-1, 0])
                locf_values.append(float(prior_values[latest_t][prior[latest_t] == int(fi)][-1]))
                locf_available.append(True)
            else:
                locf_values.append(float("nan")); locf_available.append(False)
        for i in range(int(pred.size(0))):
            key = (hadm[int(bidx[i])], int(tids[i]), int(slot[i]), int(fids[i]))
            if key in seen:
                raise RuntimeError(f"duplicate readout query key in split={split}: {key}")
            seen.add(key)
            records.append({
                "z_pred": pred[i], "z_target": target_z[i], "target_value": vals[i],
                "feature_id": fids[i], "hadm_id": hadm[int(bidx[i])], "t_idx": tids[i], "slot_idx": slot[i],
                "query_key": (hadm[int(bidx[i])], int(tids[i]), int(slot[i]), int(fids[i])),
                "locf_value": torch.tensor(locf_values[i], dtype=torch.float32),
                "locf_available": torch.tensor(locf_available[i], dtype=torch.bool),
            })
            total += 1
            if len(records) >= int(shard_size):
                payload = {
                    "cache_kind": READOUT_TRAINING_CACHE, "n": len(records),
                    "z_pred": torch.stack([r["z_pred"] for r in records]),
                    "z_target": torch.stack([r["z_target"] for r in records]),
                    "target_value": torch.stack([r["target_value"] for r in records]),
                    "feature_id": torch.stack([r["feature_id"] for r in records]),
                    "slot_idx": torch.stack([r["slot_idx"] for r in records]),
                    "t_idx": torch.stack([r["t_idx"] for r in records]),
                    "hadm_id": [r["hadm_id"] for r in records],
                    "query_key": [r["query_key"] for r in records],
                    "locf_value": torch.stack([r["locf_value"] for r in records]),
                    "locf_available": torch.stack([r["locf_available"] for r in records]),
                }
                _write_pt_shard(os.path.join(out_dir, f"shard_{shard_idx:05d}.pt"), payload)
                shard_idx += 1
                records = []
    if records:
        payload = {
            "cache_kind": READOUT_TRAINING_CACHE, "n": len(records),
            "z_pred": torch.stack([r["z_pred"] for r in records]),
            "z_target": torch.stack([r["z_target"] for r in records]),
            "target_value": torch.stack([r["target_value"] for r in records]),
            "feature_id": torch.stack([r["feature_id"] for r in records]),
            "slot_idx": torch.stack([r["slot_idx"] for r in records]),
            "t_idx": torch.stack([r["t_idx"] for r in records]),
            "hadm_id": [r["hadm_id"] for r in records],
            "query_key": [r["query_key"] for r in records],
            "locf_value": torch.stack([r["locf_value"] for r in records]),
            "locf_available": torch.stack([r["locf_available"] for r in records]),
        }
        _write_pt_shard(os.path.join(out_dir, f"shard_{shard_idx:05d}.pt"), payload)
        shard_idx += 1
    metadata = {
        "cache_kind": READOUT_TRAINING_CACHE,
        "schema": READOUT_CACHE_SCHEMA_VERSION,
        "split": str(split), "n_queries": int(total),
        **signature,
        "phase1_checkpoint": os.path.realpath(phase1_ckpt_path),
        "fields": READOUT_QUERY_FIELDS,
        "shard_count": int(shard_idx),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if sum(_validate_readout_shard(os.path.join(out_dir, f"shard_{i:05d}.pt")) for i in range(shard_idx)) != total:
        raise RuntimeError("Readout cache shard validation failed before completion")
    _write_complete(os.path.join(out_dir, ".complete"))
    backup = f"{final_dir}.old.{os.getpid()}"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(final_dir):
        os.replace(final_dir, backup)
    os.replace(out_dir, final_dir)
    if os.path.exists(backup):
        shutil.rmtree(backup)
    return metadata


def _probe_shard_state(path: str) -> Tuple[str, int, int, int]:
    stat = os.stat(path)
    return (os.path.realpath(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)


class CacheExtractionBatchSampler(Sampler[List[int]]):
    """Length/set-aware batches for clean Phase-1 cache extraction."""

    def __init__(self, lengths: Sequence[int], max_set_sizes: Sequence[int], *,
                 max_patients: int, max_context_pairwise_elements: int,
                 max_set_pairwise_elements: int, seed: int = 42) -> None:
        if len(lengths) != len(max_set_sizes):
            raise ValueError("lengths and max_set_sizes must have equal length.")
        if min(int(max_patients), int(max_context_pairwise_elements), int(max_set_pairwise_elements)) < 1:
            raise ValueError("Cache extraction budgets must all be positive.")
        self.lengths = [int(value) for value in lengths]
        self.max_set_sizes = [int(value) for value in max_set_sizes]
        if any(length < 1 or size < 1 for length, size in zip(self.lengths, self.max_set_sizes)):
            raise ValueError("Cache extraction requires non-empty timestamp and set shapes.")
        self.max_patients = int(max_patients)
        self.max_context_pairwise_elements = int(max_context_pairwise_elements)
        self.max_set_pairwise_elements = int(max_set_pairwise_elements)
        self.seed = int(seed)

    def _batches(self) -> List[List[int]]:
        generator = torch.Generator().manual_seed(self.seed)
        order = list(range(len(self.lengths)))
        order.sort(key=lambda index: (self.lengths[index], self.max_set_sizes[index], index))
        bucket_size = max(32, min(512, self.max_patients * 16))
        bucketed: List[int] = []
        for start in range(0, len(order), bucket_size):
            bucket = order[start:start + bucket_size]
            permutation = torch.randperm(len(bucket), generator=generator).tolist()
            bucketed.extend(bucket[index] for index in permutation)
        batches: List[List[int]] = []
        current: List[int] = []
        max_t = max_s = 0
        for index in bucketed:
            candidate_t = max(max_t, self.lengths[index])
            candidate_s = max(max_s, self.max_set_sizes[index])
            candidate_n = len(current) + 1
            exceeds = (
                candidate_n > self.max_patients
                or candidate_n * candidate_t * candidate_t > self.max_context_pairwise_elements
                or candidate_n * candidate_t * candidate_s * candidate_s > self.max_set_pairwise_elements
            )
            if current and exceeds:
                batches.append(current)
                current, max_t, max_s = [], 0, 0
            current.append(index)
            max_t = max(max_t, self.lengths[index])
            max_s = max(max_s, self.max_set_sizes[index])
        if current:
            batches.append(current)
        permutation = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[index] for index in permutation]

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_content_signature(data_dir: str) -> str:
    """Hash only split patient files, excluding generated auxiliary manifests."""
    root = os.path.abspath(data_dir)
    files = sorted(
        os.path.join(root, split, name)
        for split in DEFAULT_SPLITS
        if os.path.isdir(os.path.join(root, split))
        for name in os.listdir(os.path.join(root, split))
        if name.endswith(".pt") and os.path.isfile(os.path.join(root, split, name))
    )
    if not files:
        raise FileNotFoundError(f"No patient .pt files found under cache data_dir: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(os.path.relpath(path, root).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def build_phase1_cache_signature(
    *,
    data_dir: str,
    schema_dir: str,
    cfg_dict: Dict[str, Any],
    teacher: Any,
    phase1_ckpt_path: str,
) -> Dict[str, Any]:
    schema_path = os.path.abspath(os.path.join(schema_dir, "schema_processed.csv"))
    checkpoint_path = os.path.abspath(phase1_ckpt_path)
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"cache_bank requires exact schema_processed.csv: {schema_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"cache_bank requires the exact Phase-1 checkpoint: {checkpoint_path}")
    architecture_version = getattr(teacher, "phase1_architecture_version", None)
    if not architecture_version:
        raise RuntimeError("Phase-1 cache build refused: model has no architecture version.")
    source_root = os.path.dirname(os.path.abspath(__file__))
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(source_root), text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=os.path.dirname(source_root), text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "cache_builder_sha256": _sha256_file(__file__),
        "model_sha256": _sha256_file(os.path.join(source_root, "model.py")),
        "dataset_sha256": _sha256_file(os.path.join(source_root, "dataset.py")),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "phase1_architecture_version": str(architecture_version),
        "phase1_checkpoint_sha256": _sha256_file(checkpoint_path),
        "schema_sha256": _sha256_file(schema_path),
        "config_signature": hashlib.sha256(
            json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "dataset_signature": _dataset_content_signature(data_dir),
        "latent_dimension": int(getattr(teacher, "embed_dim")),
    }


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _signature_matches(found: Any, expected: Dict[str, Any]) -> bool:
    return isinstance(found, dict) and all(found.get(key) == value for key, value in expected.items())


_CACHE_PROVENANCE_FIELDS = {"cache_builder_sha256", "git_commit", "git_dirty"}


def _cache_signature_differences(found: Any, expected: Dict[str, Any]) -> List[str]:
    if not isinstance(found, dict):
        return sorted(expected)
    return sorted(key for key, value in expected.items() if found.get(key) != value)


def _probe_cache_payload_matches(found: Any, expected: Dict[str, Any]) -> bool:
    """Compare fields that determine extracted latent payloads, not provenance."""
    return isinstance(found, dict) and all(
        found.get(key) == value
        for key, value in expected.items()
        if key not in _CACHE_PROVENANCE_FIELDS
    )


def _write_complete(path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(CACHE_SCHEMA_VERSION + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _complete_marker_matches(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip() == CACHE_SCHEMA_VERSION
    except OSError:
        return False


def _validate_probe_sequence_header(payload: Dict[str, Any], shard_path: str) -> None:
    """Validate packed shard structure without paging all latent values."""
    required = {"cache_kind", "n", "offsets", "lengths", "z_set", "z_context", "time_min", "hadm_id"}
    missing = required.difference(payload)
    if missing or payload.get("cache_kind") != "probe_exact_time_sequence_v3_packed":
        raise RuntimeError(f"Invalid exact-time probe shard {shard_path}: missing={sorted(missing)}")
    n = int(payload["n"])
    offsets, lengths, z_set, z_context, time_min, hadm = (payload[k] for k in ("offsets", "lengths", "z_set", "z_context", "time_min", "hadm_id"))
    if (not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.long or tuple(offsets.shape) != (n + 1,)
            or not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or tuple(lengths.shape) != (n,)):
        raise RuntimeError(f"Invalid packed offsets/lengths in {shard_path}")
    if not all(isinstance(x, torch.Tensor) for x in (z_set, z_context, time_min)):
        raise RuntimeError(f"Probe shard tensors are missing in {shard_path}")
    if z_set.ndim != 2 or z_context.shape != z_set.shape or time_min.shape != (z_set.size(0),):
        raise RuntimeError(f"Inconsistent probe tensor shapes in {shard_path}")
    if z_set.dtype != torch.float16 or z_context.dtype != torch.float16 or time_min.dtype != torch.float32:
        raise RuntimeError(f"Probe cache tensors must be FP16 latent and FP32 time in {shard_path}")
    if len(hadm) != n or len(set(map(str, hadm))) != n:
        raise RuntimeError(f"Probe cache hadm_id values must be unique within shard {shard_path}")
    if offsets[0].item() != 0 or offsets[-1].item() != z_set.size(0) or not torch.equal(offsets[1:] - offsets[:-1], lengths):
        raise RuntimeError(f"Invalid packed offsets in {shard_path}")
    if (lengths < 1).any() or not torch.all(offsets[1:] > offsets[:-1]):
        raise RuntimeError(f"Invalid packed sequence lengths in {shard_path}")
    # Full numerical validation happens before a shard is atomically committed
    # during cache construction.  Loading a completed cache must stay mmap
    # metadata-only; otherwise every probe startup rereads all FP16 latents.


def _validate_probe_sequence_payload(payload: Dict[str, Any], shard_path: str) -> None:
    """Full build-time validation for packed exact-time probe cache shards."""
    _validate_probe_sequence_header(payload, shard_path)
    offsets, z_set, z_context, time_min = (payload[key] for key in ("offsets", "z_set", "z_context", "time_min"))
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        if not torch.isfinite(z_set[start:end]).all() or not torch.isfinite(z_context[start:end]).all() or not torch.isfinite(time_min[start:end]).all():
            raise RuntimeError(f"Non-finite valid probe cache value in {shard_path}")
        if end - start > 1 and not torch.all(time_min[start + 1:end] > time_min[start:end - 1]):
            raise RuntimeError(f"Non-increasing time_min in {shard_path}")


def _stage_is_complete(
    root: str,
    kind: str,
    signature: Dict[str, Any],
    splits: Sequence[str],
    cutoffs: Optional[Sequence[int]] = None,
) -> bool:
    if kind == RETROSPECTIVE_CUTOFF:
        if cutoffs is None:
            return False
        stage_roots = [resolve_path(root, kind, cutoff_min=int(c)) for c in cutoffs]
    else:
        stage_roots = [resolve_path(root, kind)]
    for stage_root in stage_roots:
        metadata_path = os.path.join(stage_root, "metadata.json")
        if not (os.path.isfile(metadata_path) and _complete_marker_matches(os.path.join(stage_root, ".complete"))):
            return False
        try:
            if not _signature_matches(_read_json(metadata_path).get("cache_signature"), signature):
                return False
        except (OSError, json.JSONDecodeError):
            return False
        for split in splits:
            split_dir = os.path.join(stage_root, str(split))
            if not (
                os.path.isdir(split_dir)
                and any(name.endswith(".pt") for name in os.listdir(split_dir))
                and _complete_marker_matches(os.path.join(split_dir, ".complete"))
            ):
                return False
    return True


def resolve_path(
    cache_root: str,
    cache_kind: str,
    *,
    split: Optional[str] = None,
    cutoff_min: Optional[int] = None,
) -> str:
    root = os.path.abspath(cache_root)
    if cache_kind == FULL_VISIBLE:
        if split is None:
            return os.path.join(root, FULL_VISIBLE)
        return os.path.join(root, FULL_VISIBLE, str(split))
    if cache_kind == RETROSPECTIVE_CUTOFF:
        if cutoff_min is None:
            return root
        cutoff_dir = _CUTOFF_STAGE_NAMES.get(int(cutoff_min), f"cutoff_{int(cutoff_min)}min")
        if split is None:
            return os.path.join(root, cutoff_dir)
        return os.path.join(root, cutoff_dir, str(split))
    if cache_kind == READOUT_TRAINING_CACHE:
        if split is None:
            return os.path.join(root, READOUT_TRAINING_CACHE)
        return os.path.join(root, READOUT_TRAINING_CACHE, str(split))
    raise ValueError(f"Unknown cache_kind: {cache_kind}")


def _choose_storage_backend(estimated_dense_bytes: int, prefer_mmap_threshold_bytes: int = 32 * (1024**3)) -> str:
    # Default is batched .pt shards. Switch to mmap only when size is huge.
    if int(estimated_dense_bytes) >= int(prefer_mmap_threshold_bytes):
        return "mmap"
    return "pt_shard"


def _phase1_dataset_kwargs(cfg_dict: Dict[str, Any], max_seq_len_limit: Optional[int] = None) -> Dict[str, Any]:
    validate_phase1_masking_config(cfg_dict)
    p1_cfg = cfg_dict.get("phase1", {})
    ds_keys = {
        "max_seq_len_limit",
        "max_set_size_limit",
        "mask_target_ratio",
        "mask_rarity_strength",
        "mask_min_prob",
        "mask_max_prob",
        "static_mask_prob",
        "static_numeric_vids",
        "value_limit",
    }
    merged: Dict[str, Any] = {}
    merged.update(p1_cfg.get("data", {}))
    merged.update(p1_cfg.get("recipe", {}))
    merged.update(p1_cfg.get("normalization", {}))
    ds_kwargs = {k: v for k, v in merged.items() if k in ds_keys}
    ds_kwargs["phase"] = 1
    ds_kwargs["target_task"] = "mortality"
    if max_seq_len_limit is not None:
        ds_kwargs["max_seq_len_limit"] = int(max_seq_len_limit)
    return ds_kwargs


def _resolve_loader_cfg(
    cfg_dict: Dict[str, Any],
    override_batch_size: Optional[int] = None,
    override_num_workers: Optional[int] = None,
    safe_mode: bool = False,
) -> Dict[str, Any]:
    probe_cache = cfg_dict.get("probe", {}).get("cache_build", {})
    p1_train = cfg_dict.get("phase1", {}).get("training", {})
    batch_size = int(
        override_batch_size
        if override_batch_size is not None
        else probe_cache.get("batch_size", p1_train.get("batch_size", 32))
    )
    num_workers = int(
        override_num_workers
        if override_num_workers is not None
        else probe_cache.get("num_workers", p1_train.get("num_workers", 4))
    )
    pin_memory = bool(probe_cache.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(probe_cache.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = probe_cache.get("prefetch_factor", 4)
    if safe_mode:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _apply_cutoff_to_batch(batch_dev: Dict[str, Any], cutoff_min: Optional[float]) -> Dict[str, Any]:
    if cutoff_min is None:
        return batch_dev
    cutoff = float(cutoff_min)
    out: Dict[str, Any] = dict(batch_dev)
    if "state_times" not in out:
        return out
    state_times = out["state_times"].float().squeeze(-1)
    seq_pad = out["seq_pad_mask"].bool()
    cutoff_mask = state_times > cutoff
    seq_pad_new = seq_pad | cutoff_mask
    out["seq_pad_mask"] = seq_pad_new

    if "state_pad_mask" in out:
        sp = out["state_pad_mask"].bool()
        sp = sp | seq_pad_new.unsqueeze(-1)
        out["state_pad_mask"] = sp
    if "state_tokens" in out:
        out["state_tokens"] = out["state_tokens"].masked_fill(seq_pad_new.unsqueeze(-1), 0)
    if "state_values" in out:
        out["state_values"] = out["state_values"].masked_fill(seq_pad_new.unsqueeze(-1).unsqueeze(-1), 0.0)
    if "state_mask_labels" in out:
        out["state_mask_labels"] = out["state_mask_labels"].masked_fill(seq_pad_new.unsqueeze(-1).unsqueeze(-1), 0.0)

    if "inter_times" in out and "inter_pad_mask" in out:
        inter_times = out["inter_times"].float().squeeze(-1)
        inter_pad = out["inter_pad_mask"].bool()
        inter_cutoff = inter_times > cutoff
        if "inter_t_step" in out:
            inter_t_step = out["inter_t_step"].long()
            inter_cutoff = inter_cutoff | seq_pad_new.gather(1, inter_t_step.clamp(min=0, max=seq_pad_new.size(1) - 1))
        inter_pad_new = inter_pad | inter_cutoff
        out["inter_pad_mask"] = inter_pad_new
        for k in ["inter_tokens", "inter_values", "inter_mask_labels", "inter_times", "inter_offset_from_left", "inter_offset_to_right", "inter_interval_dt"]:
            if k in out:
                fill_val = 0 if out[k].dtype in (torch.int32, torch.int64, torch.long) else 0.0
                out[k] = out[k].masked_fill(inter_pad_new.unsqueeze(-1) if out[k].dim() == 3 else inter_pad_new, fill_val)
        if "hist_inter_pad_mask" in out:
            out["hist_inter_pad_mask"] = inter_pad_new
    return out




def _profile_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    warmup_batches: int = 8,
    measure_batches: int = 32,
    collate_fn: Optional[Any] = None,
) -> Dict[str, float]:
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and int(num_workers) > 0),
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if int(num_workers) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(dataset, **kwargs)
    it = iter(loader)
    for _ in range(int(warmup_batches)):
        try:
            next(it)
        except StopIteration:
            break
    t0 = time.perf_counter()
    seen = 0
    for _ in range(int(measure_batches)):
        try:
            b = next(it)
        except StopIteration:
            break
        if isinstance(b, dict) and "event_feature_id" in b:
            seen += int(b["event_feature_id"].size(0))
        elif isinstance(b, dict) and "visit_embedding" in b:
            seen += int(b["visit_embedding"].size(0))
        else:
            seen += int(batch_size)
    dt = float(time.perf_counter() - t0)
    bps = (float(measure_batches) / dt) if dt > 0 else 0.0
    return {
        "samples_seen": float(seen),
        "batches_measured": float(measure_batches),
        "elapsed_sec": float(dt),
        "batches_per_sec": float(bps),
    }


def _probe_sequence_profile_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep variable-length canonical probe records intact for throughput profiling."""
    return batch


def _io_bytes_from_tensor_dict(payload: Dict[str, Any]) -> int:
    n = 0
    for v in payload.values():
        if isinstance(v, torch.Tensor):
            n += int(v.numel()) * int(v.element_size())
    return n


def _write_pt_shard(path: str, payload: Dict[str, Any]) -> float:
    t0 = time.perf_counter()
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return float(time.perf_counter() - t0)


def _flush_sequence_shard(
    records: List[Dict[str, Any]],
    out_dir: str,
    shard_idx: int,
) -> Tuple[int, float, int]:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"shard_{int(shard_idx):05d}.pt")

    lengths = torch.tensor([int(x["z_set"].size(0)) for x in records], dtype=torch.long)
    if not records or (lengths < 1).any() or any(
        x["z_set"].ndim != 2 or x["z_context"].shape != x["z_set"].shape
        or x["time_min"].numel() != x["z_set"].size(0) for x in records
    ):
        raise RuntimeError("Probe sequence shard records must be nonempty aligned sequences.")
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0)])
    z_set = torch.cat([x["z_set"].detach().to(dtype=torch.float16).contiguous() for x in records], dim=0).contiguous()
    z_context = torch.cat([x["z_context"].detach().to(dtype=torch.float16).contiguous() for x in records], dim=0).contiguous()
    time_min = torch.cat([x["time_min"].detach().reshape(-1).to(dtype=torch.float32).contiguous() for x in records], dim=0).contiguous()
    hadm = [str(x["hadm_id"]) for x in records]

    payload = {
        "cache_kind": "probe_exact_time_sequence_v3_packed",
        "n": int(len(records)),
        "offsets": offsets,
        "lengths": lengths,
        "z_set": z_set,
        "z_context": z_context,
        "time_min": time_min,
        "hadm_id": hadm,
    }
    _validate_probe_sequence_payload(payload, p)
    io_sec = _write_pt_shard(p, payload)
    io_bytes = _io_bytes_from_tensor_dict(payload)
    return len(records), io_sec, io_bytes


def build_full_visible(
    *,
    cache_root: str,
    data_dir: str,
    schema_dir: str,
    cfg_dict: Dict[str, Any],
    teacher: Any,
    device: torch.device,
    splits: Sequence[str] = DEFAULT_SPLITS,
    shard_size: int = 2048,
    max_seq_len_limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    safe_mode: bool = False,
    fraction: float = 1.0,
    task_cutoff_min: Optional[float] = None,
    output_root: Optional[str] = None,
    cache_signature: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if cache_signature is None:
        raise RuntimeError("build_full_visible requires a validated Phase-1 cache signature.")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    out_root = os.path.abspath(output_root) if output_root is not None else resolve_path(cache_root, FULL_VISIBLE)
    os.makedirs(out_root, exist_ok=True)

    ds_kwargs = _phase1_dataset_kwargs(cfg_dict, max_seq_len_limit=max_seq_len_limit)
    # Probe extraction is clean inference: no Phase-1 masking and no training
    # sequence cap.  Long admissions are handled by downstream attention-cost
    # batching, never silently cropped here.
    ds_kwargs.pop("max_seq_len_limit", None)
    ds_kwargs.update({"mask_target_ratio": 0.0, "mask_min_prob": 0.0, "mask_max_prob": 0.0,
                      "static_mask_prob": 0.0, "static_numeric_vids": []})
    loader_cfg = _resolve_loader_cfg(
        cfg_dict,
        override_batch_size=batch_size,
        override_num_workers=num_workers,
        safe_mode=safe_mode,
    )
    cache_cfg = dict(cfg_dict.get("probe", {}).get("cache_build", {}))
    max_patients = int(cache_cfg.get("max_patients_per_batch", loader_cfg["batch_size"]))
    max_context_cost = int(cache_cfg.get("max_context_pairwise_elements", 4_000_000))
    max_set_cost = int(cache_cfg.get("max_set_pairwise_elements", 8_000_000))
    if safe_mode:
        max_patients = min(max_patients, 4)
        max_context_cost = min(max_context_cost, 16_384)
        max_set_cost = min(max_set_cost, 32_768)
    autocast_dtype = get_autocast_dtype(device.type)
    build_stats = BuildStats()
    start = time.perf_counter()

    length_profiles: Dict[str, Any] = {}
    batch_profiles: Dict[str, Any] = {}
    for split in splits:
        split_dir = os.path.join(out_root, str(split))
        if os.path.isdir(split_dir) and any(f.endswith(".pt") for f in os.listdir(split_dir)):
            logging.info(f"build_full_visible: split={split} already done, skipping.")
            continue
        os.makedirs(split_dir, exist_ok=True)
        ds = PatientSequenceDataset(
            data_dir=data_dir,
            schema_dir=schema_dir,
            split=str(split),
            train_pct=1.0,
            fraction=float(fraction),
            deterministic=True,
            **ds_kwargs,
        )
        shape = ds.sequence_length_summary()
        raw_lengths, max_set_sizes = ds.cache_extraction_shapes()
        if len(raw_lengths) != len(ds) or any(int(length) < 1 for length in raw_lengths):
            raise RuntimeError(f"Phase-1 probe cache split={split} has an invalid empty timestamp sequence.")
        length_profiles[str(split)] = shape
        lkwargs: Dict[str, Any] = {
            "batch_sampler": CacheExtractionBatchSampler(
                raw_lengths, max_set_sizes, max_patients=max_patients,
                max_context_pairwise_elements=max_context_cost,
                max_set_pairwise_elements=max_set_cost, seed=42,
            ),
            "num_workers": int(loader_cfg["num_workers"]),
            "collate_fn": nested_padding_collate,
            "pin_memory": bool(loader_cfg["pin_memory"]),
            "persistent_workers": bool(loader_cfg["persistent_workers"]),
        }
        if int(loader_cfg["num_workers"]) > 0 and loader_cfg["prefetch_factor"] is not None:
            lkwargs["prefetch_factor"] = int(loader_cfg["prefetch_factor"])
        loader = DataLoader(ds, **lkwargs)

        shard_buf: List[Dict[str, Any]] = []
        shard_idx = 0
        observed_batch_patients: List[int] = []
        observed_max_t: List[int] = []
        observed_context_cost: List[int] = []
        observed_set_cost: List[int] = []
        for batch in tqdm(loader, desc=f"cache_bank/full_visible/{split}", ascii=True, mininterval=5.0):
            batch_dev = _batch_to_device(batch, device)
            batch_dev = _apply_cutoff_to_batch(batch_dev, cutoff_min=task_cutoff_min)
            with torch.inference_mode():
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    seq_pad = batch_dev["seq_pad_mask"].bool()
                    frontend = teacher.encode_teacher_sets(
                        batch_dev["state_tokens"], batch_dev["state_values"], batch_dev["state_pad_mask"], seq_pad
                    )
                    visit = frontend["z_set"].detach()
                    times = batch_dev["state_times"].float().squeeze(-1)
                    z_context = teacher.contextualize_teacher_z_set(visit, seq_pad, times).detach()

            actual_sizes = (~batch_dev["state_pad_mask"].bool()).sum(dim=-1)
            current_b = int(seq_pad.size(0))
            current_t = int((~seq_pad).sum(dim=1).max().item())
            current_s = int(actual_sizes.max().item())
            observed_batch_patients.append(current_b)
            observed_max_t.append(current_t)
            observed_context_cost.append(current_b * current_t * current_t)
            observed_set_cost.append(current_b * current_t * current_s * current_s)

            # ``batch`` remains CPU-resident after non-blocking H2D transfer.
            # Retain only tensors written to this packed cache; do not create
            # a second CPU dictionary for all raw event fields.
            visit_cpu = visit.detach().to(device="cpu", dtype=torch.float16)
            z_context_cpu = z_context.detach().to(device="cpu", dtype=torch.float16)
            state_times = batch["state_times"].float().squeeze(-1)
            seq_lengths_cpu = (~batch["seq_pad_mask"].bool()).sum(dim=1).tolist()
            hadm_ids = [str(x) for x in batch["hadm_id"]]
            for b, (hid, length) in enumerate(zip(hadm_ids, seq_lengths_cpu)):
                if int(length) <= 0:
                    continue
                n = int(length)
                v = visit_cpu[b, :n].contiguous()
                z = z_context_cpu[b, :n].contiguous()
                t = state_times[b, :n].to(torch.float32).contiguous()
                shard_buf.append(
                    {
                        "hadm_id": hid,
                        "z_set": v,
                        "z_context": z,
                        "time_min": t,
                    }
                )
                if len(shard_buf) >= int(shard_size):
                    n, io_sec, io_bytes = _flush_sequence_shard(shard_buf, split_dir, shard_idx)
                    build_stats.samples += n
                    build_stats.shards += 1
                    build_stats.io_sec += io_sec
                    build_stats.io_bytes += io_bytes
                    shard_buf = []
                    shard_idx += 1

        if shard_buf:
            n, io_sec, io_bytes = _flush_sequence_shard(shard_buf, split_dir, shard_idx)
            build_stats.samples += n
            build_stats.shards += 1
            build_stats.io_sec += io_sec
            build_stats.io_bytes += io_bytes

        _write_complete(os.path.join(split_dir, ".complete"))
        def _summary(values: List[int]) -> Dict[str, float]:
            array = np.asarray(values, dtype=np.float64)
            return {"min": float(array.min()), "median": float(np.median(array)), "max": float(array.max())} if array.size else {"min": 0.0, "median": 0.0, "max": 0.0}
        elapsed_split = max(time.perf_counter() - start, 1e-12)
        batch_profiles[str(split)] = {
            "batch_count": len(observed_batch_patients),
            "batch_patients": _summary(observed_batch_patients),
            "batch_max_t": _summary(observed_max_t),
            "observed_context_cost": _summary(observed_context_cost),
            "observed_set_cost": _summary(observed_set_cost),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "admissions_per_sec": float(sum(observed_batch_patients) / elapsed_split),
            "timestamps_per_sec": float(sum(p * t for p, t in zip(observed_batch_patients, observed_max_t)) / elapsed_split),
        }

    build_stats.build_sec = float(time.perf_counter() - start)
    build_stats.gpu_util_percent = _gpu_util_percent()
    summary = {
        "cache_kind": FULL_VISIBLE,
        "phase1_architecture_version": getattr(teacher, "phase1_architecture_version", None),
        "cache_signature": cache_signature,
        "splits": [str(x) for x in splits],
        "fields": ["z_set", "z_context", "time_min", "offsets", "lengths", "hadm_id"],
        "storage_backend": _choose_storage_backend(build_stats.io_bytes),
        "build_throughput": build_stats.as_dict(),
        "sequence_length_profiles": length_profiles,
        "cache_extraction_batch_profiles": batch_profiles,
        "cache_extraction_budgets": {"max_patients_per_batch": max_patients,
                                      "max_context_pairwise_elements": max_context_cost,
                                      "max_set_pairwise_elements": max_set_cost},
    }
    with open(os.path.join(out_root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_complete(os.path.join(out_root, ".complete"))
    return summary


def build_retrospective_cutoff(
    *,
    cache_root: str,
    data_dir: str,
    schema_dir: str,
    cfg_dict: Dict[str, Any],
    teacher: Any,
    device: torch.device,
    cutoffs_min: Sequence[int] = DEFAULT_CUTOFFS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    shard_size: int = 2048,
    max_seq_len_limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    safe_mode: bool = False,
    fraction: float = 1.0,
    cache_signature: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build retrospective_cutoff from pre-built full_visible shards.

    Re-run the EMA context helper on a cutoff-masked z_set sequence.  Both
    cache kinds expose the same exact-time probe record contract.
    """
    if cache_signature is None:
        raise RuntimeError("build_retrospective_cutoff requires a validated Phase-1 cache signature.")
    return _build_retrospective_cutoff_combined(
        cache_root=cache_root, cfg_dict=cfg_dict, teacher=teacher, device=device,
        cutoffs_min=cutoffs_min, splits=splits, shard_size=shard_size,
        batch_size=batch_size, cache_signature=cache_signature,
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast_dtype = get_autocast_dtype(device.type)
    root = resolve_path(cache_root, RETROSPECTIVE_CUTOFF)
    os.makedirs(root, exist_ok=True)
    cache_cfg = dict(cfg_dict.get("probe", {}).get("cache_build", {}))
    context_budget = int(cache_cfg.get("max_context_pairwise_elements", 4_000_000))
    max_patients = int(cache_cfg.get("max_patients_per_batch", batch_size or 32))
    if context_budget < 1 or max_patients < 1:
        raise ValueError("cutoff cache batching budgets must be positive.")
    cutoff_reports: Dict[str, Any] = {}

    # Build all requested cutoff stages from each full shard in one pass.  A
    # stage may still be reused independently when it is already complete.
    pending_cutoffs = [int(cutoff) for cutoff in cutoffs_min if not _stage_is_complete(
        cache_root, RETROSPECTIVE_CUTOFF, cache_signature, [str(split) for split in splits], [int(cutoff)]
    )]
    for cutoff in pending_cutoffs:
        cutoff_root = resolve_path(cache_root, RETROSPECTIVE_CUTOFF, cutoff_min=int(cutoff))
        os.makedirs(cutoff_root, exist_ok=True)
        n_total = 0
        excluded: List[Dict[str, str]] = []
        build_start = time.perf_counter()

        for split in splits:
            out_split_dir = os.path.join(cutoff_root, str(split))
            if _stage_is_complete(
                cache_root, RETROSPECTIVE_CUTOFF, cache_signature, [str(split)], [int(cutoff)]
            ):
                logging.info(f"build_retrospective_cutoff: cutoff={cutoff} split={split} already done, skipping.")
                continue

            # Require full_visible to exist
            fv_split_dir = resolve_path(cache_root, FULL_VISIBLE, split=split)
            if not _stage_is_complete(cache_root, FULL_VISIBLE, cache_signature, [str(split)]):
                raise RuntimeError(
                    f"A signed, complete full_visible/{split} cache is required before retrospective_cutoff. "
                    f"Expected: {fv_split_dir}"
                )
            os.makedirs(out_split_dir, exist_ok=True)

            # Load full_visible shards directly
            fv_shard_paths = sorted(
                os.path.join(fv_split_dir, fn)
                for fn in os.listdir(fv_split_dir)
                if fn.endswith(".pt")
            )

            shard_buf: List[Dict[str, Any]] = []
            shard_idx = 0

            for shard_path in tqdm(fv_shard_paths, desc=f"cache_bank/retro_{int(cutoff)}/{split}", ascii=True, mininterval=5.0):
                payload = torch.load(shard_path, map_location="cpu", weights_only=False)
                B = int(payload["n"])
                seq_lens = payload["lengths"]
                offsets = payload["offsets"]
                packed_z_set = payload["z_set"]
                packed_time = payload["time_min"]
                hadm_ids_list = payload["hadm_id"]

                # Full-visible is packed on disk.  This temporary batch-local
                # view is the only dense allocation needed for the context
                # forward and is released before the next shard.
                T_max, D = int(seq_lens.max().item()), int(packed_z_set.size(1))
                z_set_full = torch.zeros((B, T_max, D), dtype=torch.float32)
                time_min_full = torch.zeros((B, T_max), dtype=torch.float32)
                for row, (start, end) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
                    length = int(end - start)
                    z_set_full[row, :length] = packed_z_set[start:end].float()
                    time_min_full[row, :length] = packed_time[start:end]

                # Batch backbone pass for z_context at cutoff
                # Build masked visit_embedding and seq_pad for each patient
                T_max = int(z_set_full.size(1))

                positions = torch.arange(T_max).unsqueeze(0)
                valid_position = positions < seq_lens.unsqueeze(1)
                cutoff_valid = valid_position & (time_min_full <= float(cutoff))
                for b_off in range(0, B, int(batch_size or 32)):
                    b_end = min(b_off + int(batch_size or 32), B)
                    bsz = b_end - b_off

                    v_slice = z_set_full[b_off:b_end].to(device).float()
                    cut_mask = cutoff_valid[b_off:b_end].to(device)
                    cut_lengths = cut_mask.sum(dim=1).long()
                    keep = cut_lengths > 0
                    for local in torch.where(~keep)[0].tolist():
                        excluded.append({"hadm_id": str(hadm_ids_list[b_off + int(local)]),
                                         "reason": "no_timestamp_at_or_before_cutoff"})
                    if not keep.any():
                        continue
                    kept_rows = torch.where(keep)[0]
                    v_slice = v_slice[kept_rows]
                    cut_lengths = cut_lengths[kept_rows]
                    # Timestamp order is canonical and cutoff membership is a
                    # prefix.  Compact to the batch maximum so the backbone
                    # never receives a future z_set, time or padding column.
                    prefix_t = int(cut_lengths.max().item())
                    v_masked = v_slice[:, :prefix_t].clone()
                    t_masked = time_min_full[b_off:b_end, :prefix_t].to(device).float()[kept_rows].clone()
                    positions = torch.arange(prefix_t, device=device).unsqueeze(0)
                    seq_pad_cut = positions >= cut_lengths.unsqueeze(1)
                    v_masked = v_masked.masked_fill(seq_pad_cut.unsqueeze(-1), 0.0)
                    t_masked = t_masked.masked_fill(seq_pad_cut, 0.0)

                    # Run frozen teacher backbone only (no feature projector)
                    with torch.inference_mode():
                        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                            z_context_cut = teacher.contextualize_teacher_z_set(
                                v_masked,
                                seq_pad_cut,
                                t_masked,
                                causal_mask=None,
                            ).detach().float().cpu()  # (bsz, T, D)

                    for bi, local in enumerate(kept_rows.tolist()):
                        b_abs = b_off + int(local)
                        length = int(cut_lengths[bi].item())
                        v_out = z_set_full[b_abs, :length].clone()
                        z_out = z_context_cut[bi, :length].clone()
                        t_out = time_min_full[b_abs, :length].clone()

                        rec: Dict[str, Any] = {
                            "hadm_id": str(hadm_ids_list[b_abs]),
                            "z_set": v_out,
                            "z_context": z_out,
                            "time_min": t_out,
                        }

                        shard_buf.append(rec)
                        n_total += 1
                        if len(shard_buf) >= int(shard_size):
                            _flush_sequence_shard(shard_buf, out_split_dir, shard_idx)
                            shard_buf = []
                            shard_idx += 1

            if shard_buf:
                _flush_sequence_shard(shard_buf, out_split_dir, shard_idx)
                shard_buf = []
            _write_complete(os.path.join(out_split_dir, ".complete"))

        elapsed = float(time.perf_counter() - build_start)
        report: Dict[str, Any] = {
            "cache_kind": RETROSPECTIVE_CUTOFF,
            "phase1_architecture_version": getattr(teacher, "phase1_architecture_version", None),
            "cache_signature": cache_signature,
            "cutoff_min": int(cutoff),
            "splits": [str(x) for x in splits],
            "n_samples": n_total,
            "excluded_admissions": excluded,
            "excluded_admission_count": len(excluded),
            "build_sec": elapsed,
            "fields": ["z_set", "z_context", "time_min", "offsets", "lengths", "hadm_id"],
        }
        with open(os.path.join(cutoff_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        _write_complete(os.path.join(cutoff_root, ".complete"))
        cutoff_reports[f"cutoff_{int(cutoff)}"] = report

    summary = {
        "cache_kind": RETROSPECTIVE_CUTOFF,
        "cache_signature": cache_signature,
        "cutoffs": [int(x) for x in cutoffs_min],
        "reports": cutoff_reports,
    }
    with open(os.path.join(root, "cutoff_cache_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _build_retrospective_cutoff_combined(
    *, cache_root: str, cfg_dict: Dict[str, Any], teacher: Any, device: torch.device,
    cutoffs_min: Sequence[int], splits: Sequence[str], shard_size: int,
    batch_size: Optional[int], cache_signature: Dict[str, Any],
) -> Dict[str, Any]:
    """Build all cutoff stages while each full-visible shard is resident once."""
    requested = tuple(sorted({int(value) for value in cutoffs_min}))
    if not requested:
        raise ValueError("At least one retrospective cutoff is required.")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast_dtype = get_autocast_dtype(device.type)
    root = resolve_path(cache_root, RETROSPECTIVE_CUTOFF)
    os.makedirs(root, exist_ok=True)
    cache_cfg = dict(cfg_dict.get("probe", {}).get("cache_build", {}))
    context_budget = int(cache_cfg.get("max_context_pairwise_elements", 4_000_000))
    max_patients = int(cache_cfg.get("max_patients_per_batch", batch_size or 32))
    if context_budget < 1 or max_patients < 1:
        raise ValueError("cutoff cache batching budgets must be positive.")
    complete_cutoffs = {
        cutoff for cutoff in requested
        if _stage_is_complete(cache_root, RETROSPECTIVE_CUTOFF, cache_signature, [str(value) for value in splits], [cutoff])
    }
    reports: Dict[int, Dict[str, Any]] = {
        cutoff: {
            "cache_kind": RETROSPECTIVE_CUTOFF,
            "phase1_architecture_version": getattr(teacher, "phase1_architecture_version", None),
            "cache_signature": cache_signature, "cutoff_min": cutoff,
            "splits": [str(value) for value in splits], "n_samples": 0,
            "excluded_admissions": [], "reused_full_for_24h_count": 0,
            "reused_full_for_48h_count": 0, "recomputed_24h_count": 0,
            "recomputed_48h_count": 0, "dynamic_context_batches": 0,
            "fields": ["z_set", "z_context", "time_min", "offsets", "lengths", "hadm_id"],
        } for cutoff in requested if cutoff not in complete_cutoffs
    }
    for cutoff in complete_cutoffs:
        reports[cutoff] = _read_json(os.path.join(resolve_path(cache_root, RETROSPECTIVE_CUTOFF, cutoff_min=cutoff), "metadata.json"))

    def group_rows(rows: List[int], prefix_lengths: List[int]) -> List[List[int]]:
        ordered = sorted(rows, key=lambda row: (prefix_lengths[row], row))
        groups: List[List[int]] = []
        current: List[int] = []
        max_t = 0
        for row in ordered:
            candidate_t = max(max_t, prefix_lengths[row])
            if current and (len(current) + 1 > max_patients or (len(current) + 1) * candidate_t ** 2 > context_budget):
                groups.append(current)
                current, max_t = [], 0
            current.append(row)
            max_t = max(max_t, prefix_lengths[row])
        if current:
            groups.append(current)
        return groups

    started = time.perf_counter()
    for split_value in splits:
        split = str(split_value)
        if not _stage_is_complete(cache_root, FULL_VISIBLE, cache_signature, [split]):
            raise RuntimeError(f"A signed, complete full_visible/{split} cache is required before cutoff construction.")
        pending = [cutoff for cutoff in requested if cutoff not in complete_cutoffs and not _stage_is_complete(
            cache_root, RETROSPECTIVE_CUTOFF, cache_signature, [split], [cutoff]
        )]
        if not pending:
            continue
        output_dirs = {cutoff: resolve_path(cache_root, RETROSPECTIVE_CUTOFF, split=split, cutoff_min=cutoff) for cutoff in pending}
        for directory in output_dirs.values():
            if os.path.isdir(directory) and any(name.endswith(".pt") for name in os.listdir(directory)):
                raise RuntimeError(f"Refusing to append to incomplete cutoff cache split: {directory}")
            os.makedirs(directory, exist_ok=True)
        buffers: Dict[int, List[Dict[str, Any]]] = {cutoff: [] for cutoff in pending}
        shard_indices = {cutoff: 0 for cutoff in pending}
        full_dir = resolve_path(cache_root, FULL_VISIBLE, split=split)
        paths = sorted(os.path.join(full_dir, name) for name in os.listdir(full_dir) if name.endswith(".pt"))
        for path in tqdm(paths, desc=f"cache_bank/cutoffs/{split}", ascii=True, mininterval=5.0):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _validate_probe_sequence_payload(payload, path)
            offsets = [int(value) for value in payload["offsets"].tolist()]
            lengths = [int(value) for value in payload["lengths"].tolist()]
            hadm_ids = [str(value) for value in payload["hadm_id"]]
            z_set, z_context, time_min = payload["z_set"], payload["z_context"], payload["time_min"]
            for cutoff in pending:
                prefix = [int((time_min[offsets[row]:offsets[row + 1]] <= float(cutoff)).sum()) for row in range(len(lengths))]
                records: List[Optional[Dict[str, Any]]] = [None] * len(lengths)
                recompute: List[int] = []
                for row, (full_length, prefix_length) in enumerate(zip(lengths, prefix)):
                    start, end = offsets[row], offsets[row + 1]
                    if prefix_length == 0:
                        reports[cutoff]["excluded_admissions"].append({"hadm_id": hadm_ids[row], "reason": "no_timestamp_at_or_before_cutoff"})
                    elif prefix_length == full_length:
                        records[row] = {"hadm_id": hadm_ids[row], "z_set": z_set[start:end],
                                        "z_context": z_context[start:end], "time_min": time_min[start:end]}
                        reports[cutoff]["reused_full_for_24h_count" if cutoff == 1440 else "reused_full_for_48h_count"] += 1
                    else:
                        recompute.append(row)
                for group in group_rows(recompute, prefix):
                    max_t = max(prefix[row] for row in group)
                    dim = int(z_set.size(1))
                    dense_z = torch.zeros((len(group), max_t, dim), dtype=torch.float32, device=device)
                    dense_t = torch.zeros((len(group), max_t), dtype=torch.float32, device=device)
                    group_lengths = torch.tensor([prefix[row] for row in group], dtype=torch.long, device=device)
                    for local, row in enumerate(group):
                        start, length = offsets[row], prefix[row]
                        dense_z[local, :length] = z_set[start:start + length].to(device=device, dtype=torch.float32, non_blocking=True)
                        dense_t[local, :length] = time_min[start:start + length].to(device=device, dtype=torch.float32, non_blocking=True)
                    pad = torch.arange(max_t, device=device).unsqueeze(0) >= group_lengths.unsqueeze(1)
                    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                        context = teacher.contextualize_teacher_z_set(dense_z, pad, dense_t, causal_mask=None).detach().to(device="cpu", dtype=torch.float16)
                    reports[cutoff]["dynamic_context_batches"] += 1
                    reports[cutoff]["recomputed_24h_count" if cutoff == 1440 else "recomputed_48h_count"] += len(group)
                    for local, row in enumerate(group):
                        start, length = offsets[row], prefix[row]
                        records[row] = {"hadm_id": hadm_ids[row], "z_set": z_set[start:start + length],
                                        "z_context": context[local, :length].contiguous(), "time_min": time_min[start:start + length]}
                for record in records:
                    if record is None:
                        continue
                    buffers[cutoff].append(record)
                    reports[cutoff]["n_samples"] += 1
                    if len(buffers[cutoff]) >= int(shard_size):
                        _flush_sequence_shard(buffers[cutoff], output_dirs[cutoff], shard_indices[cutoff])
                        buffers[cutoff] = []
                        shard_indices[cutoff] += 1
            del payload
        for cutoff in pending:
            if buffers[cutoff]:
                _flush_sequence_shard(buffers[cutoff], output_dirs[cutoff], shard_indices[cutoff])
            _write_complete(os.path.join(output_dirs[cutoff], ".complete"))
    for cutoff in requested:
        if cutoff in complete_cutoffs:
            continue
        report = reports[cutoff]
        report["excluded_admission_count"] = len(report["excluded_admissions"])
        report["build_sec"] = float(time.perf_counter() - started)
        stage_root = resolve_path(cache_root, RETROSPECTIVE_CUTOFF, cutoff_min=cutoff)
        with open(os.path.join(stage_root, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        _write_complete(os.path.join(stage_root, ".complete"))
    summary = {"cache_kind": RETROSPECTIVE_CUTOFF, "cache_signature": cache_signature,
               "cutoffs": list(requested), "reports": {f"cutoff_{cutoff}": reports[cutoff] for cutoff in requested}}
    with open(os.path.join(root, "cutoff_cache_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary



def build_cache_bank(
    *,
    cache_root: str,
    data_dir: str,
    schema_dir: str,
    cfg_dict: Dict[str, Any],
    teacher: Any,
    phase1_ckpt_path: str,
    device: torch.device,
    rebuild: bool = False,
    build_full_visible_cache: bool = True,
    build_retrospective_cutoff_cache: bool = True,
    cutoffs_min: Sequence[int] = DEFAULT_CUTOFFS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    shard_size: int = 2048,
    max_seq_len_limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    safe_mode: bool = False,
    fraction: float = 1.0,
) -> Dict[str, Any]:
    root = os.path.abspath(cache_root)
    signature = build_phase1_cache_signature(
        data_dir=data_dir,
        schema_dir=schema_dir,
        cfg_dict=cfg_dict,
        teacher=teacher,
        phase1_ckpt_path=phase1_ckpt_path,
    )
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, "cache_manifest.json")
    root_complete = os.path.join(root, ".complete")
    existing_entries = [name for name in os.listdir(root) if name not in {".DS_Store"}]
    if existing_entries and not rebuild:
        if not (os.path.isfile(manifest_path) and os.path.isfile(root_complete)):
            raise RuntimeError(
                "Refusing to reuse legacy or incomplete Phase-1 cache bank. "
                "Rebuild explicitly with rebuild=True."
            )
        try:
            stored_manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Refusing to reuse unreadable Phase-1 cache manifest; rebuild required.") from exc
        stored_signature = stored_manifest.get("cache_signature")
        if not _signature_matches(stored_signature, signature):
            differing = _cache_signature_differences(stored_signature, signature)
            if not _probe_cache_payload_matches(stored_signature, signature):
                raise RuntimeError(
                    "Phase-1 cache payload signature mismatch; reuse is forbidden. "
                    f"Differing fields={differing}. Rebuild the cache with rebuild=True."
                )
            logging.info(
                "Reusing Phase-1 probe cache despite provenance-only differences: %s",
                differing,
            )
            signature = stored_signature
    if rebuild:
        rebuild_paths = [resolve_path(root, FULL_VISIBLE)]
        rebuild_paths.extend(resolve_path(root, RETROSPECTIVE_CUTOFF, cutoff_min=int(c)) for c in cutoffs_min)
        for p in rebuild_paths:
            if os.path.isdir(p):
                for dirpath, _, filenames in os.walk(p, topdown=False):
                    for fn in filenames:
                        os.remove(os.path.join(dirpath, fn))
                    try:
                        os.rmdir(dirpath)
                    except OSError:
                        pass

    report: Dict[str, Any] = {
        "root": root,
        "cache_signature": signature,
        "cache_kinds": [],
        "profile": {},
    }
    def _stage_done(kind: str, check_splits: Optional[List[str]] = None) -> bool:
        """Reuse only a fully signed cache stage with all split completion markers."""
        if rebuild:
            return False
        targets = check_splits if check_splits is not None else splits
        return _stage_is_complete(root, kind, signature, targets, cutoffs_min if kind == RETROSPECTIVE_CUTOFF else None)

    def _reject_partial_stage(kind: str, check_splits: Sequence[str], cutoffs: Optional[Sequence[int]] = None) -> None:
        if rebuild or _stage_is_complete(root, kind, signature, check_splits, cutoffs):
            return
        stage_roots = ([resolve_path(root, kind, cutoff_min=int(c)) for c in (cutoffs or ())]
                       if kind == RETROSPECTIVE_CUTOFF else [resolve_path(root, kind)])
        if any(os.path.isdir(stage_root) and any(os.scandir(stage_root)) for stage_root in stage_roots):
            raise RuntimeError(
                f"Refusing to reuse incomplete or unsigned {kind} cache stage. "
                "Rebuild explicitly with rebuild=True."
            )

    if build_full_visible_cache:
        _reject_partial_stage(FULL_VISIBLE, splits)
        if _stage_done(FULL_VISIBLE):
            logging.info("build_cache_bank: full_visible already complete, skipping.")
            report["cache_kinds"].append(FULL_VISIBLE)
            report[FULL_VISIBLE] = {"skipped": True}
        else:
            full_report = build_full_visible(
                cache_root=root,
                data_dir=data_dir,
                schema_dir=schema_dir,
                cfg_dict=cfg_dict,
                teacher=teacher,
                device=device,
                splits=splits,
                shard_size=shard_size,
                max_seq_len_limit=max_seq_len_limit,
                batch_size=batch_size,
                num_workers=num_workers,
                safe_mode=safe_mode,
                fraction=fraction,
                cache_signature=signature,
            )
            report["cache_kinds"].append(FULL_VISIBLE)
            report[FULL_VISIBLE] = full_report
    if build_retrospective_cutoff_cache:
        _reject_partial_stage(RETROSPECTIVE_CUTOFF, splits, cutoffs_min)
        retro_all_done = (not rebuild and _stage_is_complete(
            root, RETROSPECTIVE_CUTOFF, signature, splits, cutoffs_min
        ))
        if retro_all_done:
            logging.info("build_cache_bank: retrospective_cutoff already complete, skipping.")
            report["cache_kinds"].append(RETROSPECTIVE_CUTOFF)
            report[RETROSPECTIVE_CUTOFF] = {"skipped": True}
        else:
            retro_report = build_retrospective_cutoff(
                cache_root=root,
                data_dir=data_dir,
                schema_dir=schema_dir,
                cfg_dict=cfg_dict,
                teacher=teacher,
                device=device,
                cutoffs_min=cutoffs_min,
                splits=splits,
                shard_size=shard_size,
                max_seq_len_limit=max_seq_len_limit,
                batch_size=batch_size,
                num_workers=num_workers,
                safe_mode=safe_mode,
                fraction=fraction,
                cache_signature=signature,
            )
            report["cache_kinds"].append(RETROSPECTIVE_CUTOFF)
            report[RETROSPECTIVE_CUTOFF] = retro_report
    # All cache payload stages are complete at this point. Publish the root
    # manifest before optional profiling, whose loader enforces this contract.
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"cache_signature": signature}, f, indent=2)
    _write_complete(root_complete)
    # Profile only stages requested by this invocation.
    loader_cfg = _resolve_loader_cfg(
        cfg_dict, override_batch_size=batch_size, override_num_workers=num_workers, safe_mode=safe_mode,
    )
    profile_kwargs = {
        "batch_size": int(loader_cfg["batch_size"]), "num_workers": int(loader_cfg["num_workers"]),
        "pin_memory": bool(loader_cfg["pin_memory"]), "persistent_workers": bool(loader_cfg["persistent_workers"]),
        "prefetch_factor": loader_cfg["prefetch_factor"],
    }
    profile_split = str(splits[0])
    if build_full_visible_cache:
        report["profile"][FULL_VISIBLE] = _profile_loader(
            load_dataset(root, FULL_VISIBLE, split=profile_split),
            collate_fn=_probe_sequence_profile_collate, **profile_kwargs,
        )
    if build_retrospective_cutoff_cache:
        report["profile"][RETROSPECTIVE_CUTOFF] = {
            str(cutoff): _profile_loader(
                load_dataset(root, RETROSPECTIVE_CUTOFF, split=profile_split, cutoff_min=int(cutoff)),
                collate_fn=_probe_sequence_profile_collate, **profile_kwargs,
            ) for cutoff in cutoffs_min
        }


    with open(os.path.join(root, "cache_bank_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"cache_signature": signature}, f, indent=2)
    _write_complete(root_complete)
    return report


def build_phase1_probe_cache(
    *, cache_root: str, data_dir: str, schema_dir: str, cfg_dict: Dict[str, Any], teacher: Any,
    phase1_ckpt_path: str, device: torch.device, labels_csv: Optional[str] = None,
    rebuild: bool = False, batch_size: Optional[int] = None, num_workers: Optional[int] = None,
    fraction: float = 1.0,
) -> Dict[str, Any]:
    """Build the probe-only cache from frozen Phase-1 EMA modules alone."""
    # External eICU caches must never be built from the legacy ICU-stay
    # representation.  Keep this check at the canonical cache boundary so
    # every caller (not only external_validation.py) fails closed.
    prep_summary_path = os.path.join(os.path.abspath(data_dir), "prep_summary.json")
    if os.path.isfile(prep_summary_path):
        with open(prep_summary_path, encoding="utf-8") as handle:
            prep_summary = json.load(handle)
        if prep_summary.get("sample_unit") == "patientunitstayid":
            raise RuntimeError("Refusing to build probe cache from legacy ICU-stay samples.")
        if prep_summary.get("sample_unit") == "patienthealthsystemstayid" and prep_summary.get("validation_protocol") != "hospital_admission_level_external_validation":
            raise RuntimeError("Hospital-admission processed data has an incompatible validation protocol.")
    # Probe cache identity is strict: a changed builder/config/checkpoint
    # requires rebuild.
    report = build_cache_bank(
        cache_root=cache_root, data_dir=data_dir, schema_dir=schema_dir, cfg_dict=cfg_dict, teacher=teacher,
        phase1_ckpt_path=phase1_ckpt_path, device=device, rebuild=rebuild,
        build_full_visible_cache=True, build_retrospective_cutoff_cache=True,
        cutoffs_min=(1440, 2880), splits=DEFAULT_SPLITS, batch_size=batch_size, num_workers=num_workers,
        fraction=fraction,
    )
    # Preserve the external sample-unit contract in the canonical cache
    # metadata; old ICU-stay caches are rejected before this function runs.
    prep_summary_path = os.path.join(os.path.abspath(data_dir), "prep_summary.json")
    if os.path.isfile(prep_summary_path):
        with open(prep_summary_path, encoding="utf-8") as handle:
            prep_summary = json.load(handle)
        report["sample_unit"] = prep_summary.get("sample_unit")
        report["validation_protocol"] = prep_summary.get("validation_protocol")
        metadata_path = os.path.join(os.path.abspath(cache_root), "cache_bank_metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata.update({"sample_unit": report["sample_unit"], "validation_protocol": report["validation_protocol"]})
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
    if labels_csv is not None:
        labels = pd.read_csv(labels_csv)
        cache_ids_by_split: Dict[str, List[str]] = {}
        for split in DEFAULT_SPLITS:
            cache_ds = load_dataset(cache_root, FULL_VISIBLE, split=split)
            cache_ids_by_split[split] = [str(cache_ds[index]["hadm_id"]) for index in range(len(cache_ds))]
        report["label_alignment"] = _validate_probe_label_alignment(labels, cache_ids_by_split)
        cohort_rows: List[Dict[str, Any]] = []
        normalized = labels.loc[:, ["hadm_id", "split"]].copy()
        normalized["hadm_id"] = normalized["hadm_id"].astype(str)
        normalized["split"] = normalized["split"].astype(str)
        cache_sets = {split: set(values) for split, values in cache_ids_by_split.items()}
        for row in normalized.itertuples(index=False):
            split, hadm_id = str(row.split), str(row.hadm_id)
            in_cache = hadm_id in cache_sets.get(split, set())
            cohort_rows.append({"hadm_id": hadm_id, "split": split, "in_full_labels": True,
                                "in_processed_data": True if in_cache else None, "in_cache": in_cache, "included": in_cache,
                                "exclusion_reason": "" if in_cache else "unknown_preprocessing_exclusion",
                                "exclusion_reason_note": "" if in_cache else "reason unavailable from preprocessing artifacts"})
        cohort_frame = pd.DataFrame(cohort_rows)
        cohort_frame.to_csv(os.path.join(cache_root, "probe_cohort_manifest.csv"), index=False)
        report["cohort_exclusion_reason_counts"] = cohort_frame.loc[~cohort_frame["included"], "exclusion_reason"].value_counts().to_dict()
        # ``build_cache_bank`` wrote this before labels were checked.  Persist
        # the explicit full-cohort exclusions for auditability.
        with open(os.path.join(cache_root, "cache_bank_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def _validate_probe_label_alignment(labels: pd.DataFrame,
                                    cache_ids_by_split: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """Permit a complete-cohort label table while requiring all cache IDs to label-match.

    Processed Phase-1 datasets commonly exclude admissions during preprocessing.
    Those label-only admissions are recorded as exclusions; a cached admission
    without exactly one label in its own split remains a hard error.
    """
    if not {"hadm_id", "split"}.issubset(labels.columns):
        raise RuntimeError("labels.csv must provide hadm_id and split for probe-cache alignment.")
    normalized = labels.loc[:, ["hadm_id", "split"]].copy()
    normalized["hadm_id"] = normalized["hadm_id"].astype(str)
    normalized["split"] = normalized["split"].astype(str)
    if normalized["hadm_id"].duplicated().any():
        duplicate = normalized.loc[normalized["hadm_id"].duplicated(), "hadm_id"].iloc[0]
        raise RuntimeError(f"labels.csv has duplicate or cross-split admission ID {duplicate!r}.")
    seen: set = set()
    split_report: Dict[str, Any] = {}
    for split in DEFAULT_SPLITS:
        cache_ids = [str(value) for value in cache_ids_by_split.get(split, ())]
        cache_set, label_set = set(cache_ids), set(normalized.loc[normalized["split"] == split, "hadm_id"])
        if len(cache_ids) != len(cache_set):
            raise RuntimeError(f"Duplicate cache admission IDs in split={split}.")
        overlap = seen.intersection(cache_set)
        if overlap:
            raise RuntimeError(f"Probe cache admission appears across splits: {sorted(overlap)[:20]}")
        missing_labels = sorted(cache_set - label_set)
        if missing_labels:
            raise RuntimeError(f"Probe cache/labels split mismatch {split}: cache IDs without labels={missing_labels[:20]}")
        excluded = sorted(label_set - cache_set)
        split_report[split] = {
            "cache_admission_count": len(cache_set), "full_label_admission_count": len(label_set),
            "excluded_label_only_count": len(excluded), "excluded_label_only_examples": excluded[:20],
        }
        seen.update(cache_set)
    return {"mode": "full_cohort_labels_filtered_to_processed_cache", "splits": split_report}


class ShardDataset(Dataset):
    def __init__(self, shard_dir: str, mode: str, payload_cache_size: int = 2):
        self.shard_dir = os.path.abspath(shard_dir)
        self.mode = str(mode)
        self.shards = sorted(
            os.path.join(self.shard_dir, fn)
            for fn in os.listdir(self.shard_dir)
            if fn.endswith(".pt")
        ) if os.path.isdir(self.shard_dir) else []
        if not self.shards:
            raise RuntimeError(f"No shard files found: {self.shard_dir}")
        self._shard_sizes: List[int] = []
        self._meta_cache: Dict[int, int] = {}
        self._payload_cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self._payload_cache_size = max(1, int(payload_cache_size))
        self._shard_load_count = 0
        self._shard_cache_hit_count = 0
        self._shard_cache_miss_count = 0
        self._shard_switch_count = 0
        self._last_payload_shard: Optional[int] = None
        self._sequence_lengths_cache: Optional[List[int]] = None
        self._admission_ids_cache: Optional[List[str]] = None
        self._row_offsets: List[int] = [0]
        for sidx, p in enumerate(self.shards):
            n = self._infer_shard_size(sidx)
            self._shard_sizes.append(n)
            self._row_offsets.append(int(self._row_offsets[-1] + int(n)))

    def __len__(self) -> int:
        return int(self._row_offsets[-1])

    def set_payload_cache_size(self, size: int) -> None:
        self._payload_cache_size = max(1, int(size))
        while len(self._payload_cache) > self._payload_cache_size:
            self._payload_cache.popitem(last=False)

    def shard_index(self, row_index: int) -> int:
        i = int(row_index)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        return int(bisect.bisect_right(self._row_offsets, i) - 1)

    def shard_stats(self) -> Dict[str, int]:
        return {
            "shard_load_count": int(self._shard_load_count),
            "shard_switch_count": int(self._shard_switch_count),
            "shard_cache_hit_count": int(self._shard_cache_hit_count),
            "shard_cache_miss_count": int(self._shard_cache_miss_count),
            "shard_cache_size": int(self._payload_cache_size),
        }

    def sequence_lengths(self) -> List[int]:
        """Read only per-row scalar lengths for downstream batching."""
        if self.mode != "sequence":
            raise RuntimeError("sequence_lengths is only defined for sequence shards.")
        if self._sequence_lengths_cache is not None:
            return list(self._sequence_lengths_cache)
        lengths: List[int] = []
        for shard_idx in range(len(self.shards)):
            payload = self._torch_load_mmap(self.shards[shard_idx])
            values = payload.get("lengths")
            if not isinstance(values, torch.Tensor):
                raise RuntimeError(f"Missing seq_len in shard {self.shards[shard_idx]}")
            lengths.extend(int(x) for x in values.tolist())
        if len(lengths) != len(self):
            raise RuntimeError("Sequence-length index does not match shard dataset size.")
        self._sequence_lengths_cache = lengths
        return list(lengths)

    def admission_ids(self) -> List[str]:
        if self.mode != "sequence":
            raise RuntimeError("admission_ids is only defined for sequence shards.")
        if self._admission_ids_cache is not None:
            return list(self._admission_ids_cache)
        ids: List[str] = []
        for shard_idx in range(len(self.shards)):
            payload = self._torch_load_mmap(self.shards[shard_idx])
            current = payload.get("hadm_id")
            if not isinstance(current, list):
                raise RuntimeError(f"Missing hadm_id in shard {self.shards[shard_idx]}")
            ids.extend(str(x) for x in current)
        if len(ids) != len(self) or len(ids) != len(set(ids)):
            raise RuntimeError("Invalid or duplicate sequence shard admission IDs.")
        self._admission_ids_cache = ids
        return list(ids)

    def _infer_shard_size(self, shard_idx: int) -> int:
        sid = int(shard_idx)
        if sid in self._meta_cache:
            return int(self._meta_cache[sid])
        path = self.shards[sid]
        state = _probe_shard_state(path)
        if state in _PROBE_SHARD_SIZE_CACHE:
            n = int(_PROBE_SHARD_SIZE_CACHE[state])
        else:
            payload = self._torch_load_mmap(path)
            n = int(payload.get("n", payload.get("lengths", torch.zeros((0,))).numel()))
            _PROBE_SHARD_SIZE_CACHE[state] = n
        self._meta_cache[sid] = n
        return n

    @staticmethod
    def _torch_load_mmap(path: str) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        sid = int(shard_idx)
        if sid in self._payload_cache:
            self._shard_cache_hit_count += 1
            payload = self._payload_cache.pop(sid)
            self._payload_cache[sid] = payload
            return payload
        self._shard_cache_miss_count += 1
        if self._last_payload_shard is not None and self._last_payload_shard != sid:
            self._shard_switch_count += 1
        payload = self._torch_load_mmap(self.shards[sid])
        self._payload_cache[sid] = payload
        self._shard_load_count += 1
        while len(self._payload_cache) > self._payload_cache_size:
            self._payload_cache.popitem(last=False)
        self._last_payload_shard = sid
        return payload

    def sequence_record(self, idx: int, source_name: str) -> Dict[str, Any]:
        """Read one canonical latent source without materialising the other."""
        if self.mode != "sequence" or source_name not in {"z_set", "z_context"}:
            raise ValueError("sequence_record requires a canonical sequence source.")
        i = int(idx)
        if i < 0 or i >= int(self._row_offsets[-1]):
            raise IndexError(i)
        sidx = int(bisect.bisect_right(self._row_offsets, i) - 1)
        ridx = int(i - int(self._row_offsets[sidx]))
        payload = self._load_shard(sidx)
        start, end = int(payload["offsets"][ridx].item()), int(payload["offsets"][ridx + 1].item())
        return {
            source_name: payload[source_name][start:end], "time_min": payload["time_min"][start:end],
            "seq_len": int(payload["lengths"][ridx].item()), "hadm_id": str(payload["hadm_id"][ridx]),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = int(idx)
        if i < 0 or i >= int(self._row_offsets[-1]):
            raise IndexError(i)
        sidx = int(bisect.bisect_right(self._row_offsets, i) - 1)
        ridx = int(i - int(self._row_offsets[sidx]))
        p = self._load_shard(sidx)
        if self.mode == "sequence":
            start, end = int(p["offsets"][ridx].item()), int(p["offsets"][ridx + 1].item())
            t = int(p["lengths"][ridx].item())
            out: Dict[str, Any] = {
                "z_set": p["z_set"][start:end],
                "z_context": p["z_context"][start:end],
                "time_min": p["time_min"][start:end],
                "seq_len": t,
                "hadm_id": str(p["hadm_id"][ridx]),
            }
            return out
        if self.mode == "readout_query":
            out = {
                "z_pred": p["z_pred"][ridx].clone(),
                "z_target": p["z_target"][ridx].clone(),
                "target_value": p["target_value"][ridx].clone(),
                "feature_id": p["feature_id"][ridx].clone(),
                "hadm_id": str(p["hadm_id"][ridx]),
                "t_idx": p["t_idx"][ridx].clone(),
                "slot_idx": p["slot_idx"][ridx].clone(),
                "query_key": tuple(p["query_key"][ridx]),
                "locf_value": p["locf_value"][ridx].clone(),
                "locf_available": p["locf_available"][ridx].clone(),
            }
            if "target_class" in p:
                out["target_class"] = p["target_class"][ridx].clone()
            return out
        raise ValueError(f"Unknown dataset mode: {self.mode}")





def load_dataset(
    cache_root: str,
    cache_kind: str,
    *,
    split: str,
    cutoff_min: Optional[int] = None,
    expected_phase1_ckpt: Optional[str] = None,
    expected_schema: Optional[str] = None,
    waive_source_provenance: bool = False,
) -> Dataset:
    root = os.path.abspath(cache_root)
    if cache_kind == READOUT_TRAINING_CACHE:
        stage = resolve_path(root, READOUT_TRAINING_CACHE, split=split)
        metadata = os.path.join(stage, "metadata.json")
        shards = [name for name in os.listdir(stage)] if os.path.isdir(stage) else []
        if not (os.path.isfile(metadata) and _complete_marker_matches(os.path.join(stage, ".complete"))
                and any(name.endswith(".pt") for name in shards)):
            raise RuntimeError(f"Refusing to load incomplete readout query cache: {stage}")
        meta = _read_json(metadata)
        required_fields = set(READOUT_QUERY_FIELDS)
        if meta.get("readout_cache_schema_version") != READOUT_CACHE_SCHEMA_VERSION or meta.get("readout_cache_contract_version") != READOUT_CACHE_CONTRACT_VERSION or set(meta.get("fields", [])) != required_fields:
            raise RuntimeError(f"Refusing to load legacy readout query cache: {stage}")
        if expected_phase1_ckpt is not None and meta.get("phase1_checkpoint_sha256") != _sha256_file(expected_phase1_ckpt):
            raise RuntimeError(f"Refusing to load readout cache for a different Phase-1 checkpoint: {stage}")
        if expected_schema is not None and meta.get("schema_sha256") != _sha256_file(expected_schema):
            raise RuntimeError(f"Refusing to load readout cache for a different schema: {stage}")
        key = (os.path.realpath(stage), "readout_query", _stable_json_sha256(meta))
        dataset = _PROBE_DATASET_REGISTRY.get(key)
        if dataset is None:
            dataset = ShardDataset(stage, mode="readout_query")
            _PROBE_DATASET_REGISTRY[key] = dataset
        return dataset
    manifest_path = os.path.join(root, "cache_manifest.json")
    if not (os.path.isfile(manifest_path) and _complete_marker_matches(os.path.join(root, ".complete"))):
        raise RuntimeError("Refusing to load legacy or incomplete Phase-1 cache bank; rebuild required.")
    try:
        root_signature = _read_json(manifest_path)["cache_signature"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError("Refusing to load cache without a valid immutable signature.") from exc

    def _checked_stage(stage_root: str, shard_dir: str, mode: str) -> Dataset:
        metadata_path = os.path.join(stage_root, "metadata.json")
        if not (
            os.path.isfile(metadata_path)
            and _complete_marker_matches(os.path.join(stage_root, ".complete"))
            and _complete_marker_matches(os.path.join(shard_dir, ".complete"))
        ):
            raise RuntimeError(f"Refusing to load incomplete cache stage: {shard_dir}")
        try:
            if not waive_source_provenance and not _signature_matches(_read_json(metadata_path).get("cache_signature"), root_signature):
                raise RuntimeError(f"Refusing to load cache with signature mismatch: {stage_root}")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Refusing to load unreadable cache metadata: {stage_root}") from exc
        if mode == "sequence":
            shards = sorted(
                os.path.join(shard_dir, name) for name in os.listdir(shard_dir) if name.endswith(".pt")
            )
            if not shards:
                raise RuntimeError(f"Refusing to load probe cache with no shards: {shard_dir}")
            shard_state_rows = [_probe_shard_state(path) for path in shards]
            shard_state = tuple(shard_state_rows)
            stage_key = (
                os.path.realpath(stage_root),
                json.dumps(root_signature, sort_keys=True, default=str),
                shard_state,
            )
            if stage_key not in _VALIDATED_PROBE_SEQUENCE_STAGES:
                seen: set = set()
                for path in shards:
                    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                    _validate_probe_sequence_header(payload, path)
                    _PROBE_SHARD_SIZE_CACHE[_probe_shard_state(path)] = int(payload["n"])
                    ids = [str(x) for x in payload["hadm_id"]]
                    duplicate = seen.intersection(ids)
                    if duplicate:
                        raise RuntimeError(f"Duplicate hadm_id across probe cache shards: {sorted(duplicate)[:5]}")
                    seen.update(ids)
                _VALIDATED_PROBE_SEQUENCE_STAGES.add(stage_key)
        key = (os.path.realpath(shard_dir), mode, _stable_json_sha256(root_signature))
        dataset = _PROBE_DATASET_REGISTRY.get(key)
        if dataset is None:
            dataset = ShardDataset(shard_dir, mode=mode)
            _PROBE_DATASET_REGISTRY[key] = dataset
        return dataset

    if cache_kind == FULL_VISIBLE:
        stage_root = resolve_path(cache_root, FULL_VISIBLE)
        return _checked_stage(stage_root, resolve_path(cache_root, FULL_VISIBLE, split=split), "sequence")
    if cache_kind == RETROSPECTIVE_CUTOFF:
        if cutoff_min is None:
            raise ValueError("cutoff_min is required for retrospective_cutoff")
        shard_dir = resolve_path(cache_root, RETROSPECTIVE_CUTOFF, split=split, cutoff_min=cutoff_min)
        if not os.path.isdir(shard_dir):
            retro_root = resolve_path(cache_root, RETROSPECTIVE_CUTOFF)
            available: List[str] = []
            if os.path.isdir(retro_root):
                available = sorted(
                    d for d in os.listdir(retro_root)
                    if d.startswith("cutoff_") and os.path.isdir(os.path.join(retro_root, d))
                )
            raise RuntimeError(
                f"retrospective_cutoff/{cutoff_min} not found: {shard_dir}\n"
                f"Available cutoff dirs: {available or '(none)'}\n"
                f"Rebuild with build_cache_bank(cutoffs_min=[..., {cutoff_min}, ...])"
            )
        return _checked_stage(resolve_path(cache_root, RETROSPECTIVE_CUTOFF, cutoff_min=cutoff_min), shard_dir, "sequence")
    raise ValueError(f"Unknown cache_kind: {cache_kind}")




def validate(cache_root: str, expected_signature: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    root = os.path.abspath(cache_root)
    result: Dict[str, Any] = {
        "root": root,
        "ok": True,
        "checks": {},
    }
    required = [FULL_VISIBLE, RETROSPECTIVE_CUTOFF]
    manifest_path = os.path.join(root, "cache_manifest.json")
    complete_path = os.path.join(root, ".complete")
    manifest: Dict[str, Any] = {}
    try:
        manifest = _read_json(manifest_path)
        signature_ok = _signature_matches(
            manifest.get("cache_signature"), manifest.get("cache_signature", {})
        ) and bool(manifest.get("cache_signature"))
    except (OSError, json.JSONDecodeError):
        signature_ok = False
    signature_matches_expected = (
        expected_signature is None
        or (signature_ok and _signature_matches(manifest.get("cache_signature"), expected_signature))
    )
    result["checks"]["root_manifest"] = {
        "exists": os.path.isfile(manifest_path),
        "complete": os.path.isfile(complete_path),
        "signature_present": signature_ok,
        "matches_expected_signature": bool(signature_matches_expected),
    }
    result["ok"] = result["ok"] and os.path.isfile(complete_path) and signature_ok and signature_matches_expected
    for kind in required:
        p = resolve_path(root, kind)
        metadata_path = os.path.join(p, "metadata.json")
        ok = os.path.isdir(p) and os.path.isfile(metadata_path) and os.path.isfile(os.path.join(p, ".complete"))
        if ok:
            try:
                stage_metadata = _read_json(metadata_path)
                ok = _signature_matches(stage_metadata.get("cache_signature"), manifest["cache_signature"])
                if expected_signature is not None:
                    ok = ok and _signature_matches(stage_metadata.get("cache_signature"), expected_signature)
            except (OSError, json.JSONDecodeError, KeyError, UnboundLocalError):
                ok = False
        result["checks"][kind] = {"exists": bool(ok), "path": p}
        result["ok"] = result["ok"] and ok

    # Field-level contract checks.
    try:
        ds = load_dataset(root, FULL_VISIBLE, split="train")
        x = ds[0]
        req = ["z_set", "z_context", "time_min", "seq_len", "hadm_id"]
        miss = [k for k in req if k not in x]
        if miss:
            raise RuntimeError(f"full_visible missing fields: {miss}")
        result["checks"]["full_visible_fields"] = {"ok": True}
    except Exception as exc:
        result["checks"]["full_visible_fields"] = {"ok": False, "error": str(exc)}
        result["ok"] = False



    return result
