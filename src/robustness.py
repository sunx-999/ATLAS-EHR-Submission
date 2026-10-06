#!/usr/bin/env python3
"""Raw-event MCAR robustness evaluation for canonical exact-time probes."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache_bank import (
    CACHE_SCHEMA_VERSION,
    FULL_VISIBLE,
    RETROSPECTIVE_CUTOFF,
    CacheExtractionBatchSampler,
    _batch_to_device,
    _phase1_dataset_kwargs,
    _resolve_loader_cfg,
    build_phase1_cache_signature,
    load_dataset,
)
from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.eval import FORMAL_SOURCE_ORDER, FORMAL_TASK_ORDER, _validate_multilabel_threshold_artifact, load_exact_time_probe
from src.model import JEPAModel
from src.probes import CompetitiveDRGProbe, TASK_CACHE_KIND, _task_metric, _targets_for_task, build_delta_time_min, canonical_probe_collate, multilabel_threshold_metrics
from src.utils import compute_binary_metrics, load_config_dict

ROBUSTNESS_VERSION = "raw_event_mcar_robustness_v3"
WINDOW_BY_KIND = {"retrospective_cutoff_1440": 1440, "retrospective_cutoff_2880": 2880, FULL_VISIBLE: None}
WINDOW_ORDER = ("retrospective_cutoff_1440", "retrospective_cutoff_2880", FULL_VISIBLE)


class _IndexedRawDataset(Dataset):
    def __init__(self, base: PatientSequenceDataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = [int(x) for x in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.base[self.indices[int(index)]]

    def cache_extraction_shapes(self) -> Tuple[list[int], list[int]]:
        lengths, set_sizes = self.base.cache_extraction_shapes()
        return [int(lengths[i]) for i in self.indices], [int(set_sizes[i]) for i in self.indices]

    def admission_ids(self) -> list[str]:
        ids = _dataset_ids_from_files(self.base)
        return [str(ids[i]) for i in self.indices]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _atomic_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})
    os.replace(tmp, path)


def _selector(value: str, allowed: Sequence[str], order: Sequence[str]) -> Tuple[str, ...]:
    requested = list(order) if str(value).strip().lower() == "all" else [x.strip() for x in str(value).split(",") if x.strip()]
    unknown = sorted(set(requested).difference(allowed))
    if unknown or not requested:
        raise ValueError(f"Invalid selection: {unknown or value!r}")
    if len(set(requested)) != len(requested):
        raise ValueError("Selections must be unique.")
    return tuple(x for x in order if x in requested)


def _rates(value: str) -> Tuple[float, ...]:
    rates = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if not rates:
        raise ValueError("--drop_rates cannot be empty.")
    if any(rate <= 0.0 or rate >= 1.0 for rate in rates):
        raise ValueError("--drop_rates must all satisfy 0 < rate < 1.")
    return tuple(sorted(set(rates)))


def _resolve_checkpoint(probe_root: str, task: str, source: str) -> str:
    path = os.path.join(probe_root, task, source, "checkpoints", "best.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing formal best.pt probe for {task}/{source}: {path}")
    return path


def _primary_metric_name(task: str) -> str:
    if task == "icd":
        return "micro_auprc"
    if task == "hcup":
        return "macro_auprc"
    if task == "drg":
        return "micro_f1"
    return "auprc"


def _load_cache_signature(cache_root: str) -> Dict[str, Any]:
    root = os.path.abspath(cache_root)
    metadata_path = os.path.join(root, "cache_bank_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata_text = json.dumps(metadata, sort_keys=True, default=str).lower()
        if any(token in metadata_text for token in ("hospital_admission_level_external_validation", "patienthealthsystemstayid")):
            raise RuntimeError("Robustness is MIMIC-only and refuses external-validation/eICU cache metadata.")
    path = os.path.join(root, "cache_manifest.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload_text = json.dumps(payload, sort_keys=True, default=str).lower()
    if any(token in payload_text for token in ("external_validation", "hospital_admission", "patienthealthsystemstayid")):
        raise RuntimeError("Robustness is MIMIC-only and refuses external-validation/eICU cache metadata.")
    signature = payload.get("cache_signature")
    if not isinstance(signature, dict):
        raise RuntimeError("Canonical cache manifest lacks cache_signature.")
    if signature.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError("Cache schema version mismatch; rebuild canonical cache.")
    if signature.get("validation_protocol") or str(signature.get("sample_unit", "")).lower() in {"patienthealthsystemstayid", "hospital_admission"}:
        raise RuntimeError("Robustness is MIMIC-only and refuses external-validation cache metadata.")
    return signature


def _assert_cache_signature(cache_signature: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if cache_signature.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError("Cache schema version mismatch; rebuild canonical cache.")
    for key in ("cache_contract_version", "phase1_architecture_version", "phase1_checkpoint_sha256",
                "schema_sha256", "dataset_signature", "latent_dimension"):
        if cache_signature.get(key) != expected.get(key):
            raise RuntimeError(f"Canonical cache provenance mismatch for {key}.")


def _cache_ids(cache_root: str, cache_kind: str = FULL_VISIBLE, cutoff_min: Optional[int] = None) -> list[str]:
    ds = load_dataset(cache_root, cache_kind, split="test", cutoff_min=cutoff_min)
    ids = [str(x) for x in ds.admission_ids()]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Canonical {cache_kind} test cache contains duplicate admission IDs.")
    return ids


def _task_cache_ids(cache_root: str, tasks: Sequence[str]) -> Dict[str, list[str]]:
    by_stage: Dict[Tuple[str, Optional[int]], list[str]] = {}
    out: Dict[str, list[str]] = {}
    for task in tasks:
        kind, cutoff = TASK_CACHE_KIND[task]
        key = (kind, None if kind == FULL_VISIBLE else int(cutoff))
        if key not in by_stage:
            by_stage[key] = _cache_ids(cache_root, FULL_VISIBLE if kind == FULL_VISIBLE else RETROSPECTIVE_CUTOFF, key[1])
        out[task] = by_stage[key]
    return out


def _dataset_ids_from_files(ds: PatientSequenceDataset) -> list[str]:
    files = getattr(ds, "files", None)
    if files is None:
        return [str(ds[i]["hadm_id"]) for i in range(len(ds))]
    return [os.path.basename(str(path)).replace(".pt", "") for path in files]


def _raw_dataset(args: argparse.Namespace, cfg: Mapping[str, Any], target_ids: Sequence[str]) -> PatientSequenceDataset:
    ds_kwargs = _phase1_dataset_kwargs(dict(cfg))
    ds_kwargs.pop("max_seq_len_limit", None)
    ds_kwargs.update({"mask_target_ratio": 0.0, "mask_min_prob": 0.0, "mask_max_prob": 0.0,
                      "static_mask_prob": 0.0, "static_numeric_vids": []})
    return PatientSequenceDataset(
        data_dir=str(args.data_dir), schema_dir=str(args.schema_dir), split="test",
        target_ids=[str(x) for x in target_ids], fraction=1.0, deterministic=True,
        return_legacy_nested=False, **ds_kwargs,
    )


def _loader(args: argparse.Namespace, cfg: Mapping[str, Any], ds: PatientSequenceDataset, device: torch.device) -> DataLoader:
    loader_cfg = _resolve_loader_cfg(dict(cfg), override_batch_size=None, override_num_workers=int(args.num_workers), safe_mode=False)
    raw_lengths, max_set_sizes = ds.cache_extraction_shapes()
    cache_cfg = dict(cfg.get("probe", {}).get("cache_build", {}))
    max_patients = int(cache_cfg.get("max_patients_per_batch", loader_cfg["batch_size"]))
    max_context_cost = int(cache_cfg.get("max_context_pairwise_elements", 4_000_000))
    max_set_cost = int(cache_cfg.get("max_set_pairwise_elements", 8_000_000))
    kwargs: Dict[str, Any] = {
        "batch_sampler": CacheExtractionBatchSampler(
            raw_lengths, max_set_sizes, max_patients=max_patients,
            max_context_pairwise_elements=max_context_cost,
            max_set_pairwise_elements=max_set_cost, seed=42,
        ),
        "collate_fn": nested_padding_collate,
        "num_workers": int(loader_cfg["num_workers"]),
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(loader_cfg["persistent_workers"]),
    }
    if int(loader_cfg["num_workers"]) > 0 and loader_cfg.get("prefetch_factor") is not None:
        kwargs["prefetch_factor"] = int(loader_cfg["prefetch_factor"])
    return DataLoader(ds, **kwargs)


def _stable_scores(shape: torch.Size, hadm_ids: Sequence[str], seed: int, repeat: int, device: torch.device,
                   valid_event: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Patient-level independent PRNG scores in canonical timestamp/slot order."""
    B, T, S = map(int, shape)
    if valid_event is None:
        valid_cpu = torch.ones((B, T, S), dtype=torch.bool)
    else:
        valid_cpu = valid_event.detach().cpu().bool()
    scores = torch.ones((B, T, S), dtype=torch.float32)
    for row, hid in enumerate(hadm_ids):
        flat_valid = valid_cpu[row].flatten()
        n = int(flat_valid.sum().item())
        if n == 0:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(seed, repeat, hid))
        scores[row].flatten()[flat_valid] = torch.rand(n, generator=generator, dtype=torch.float32)
    return scores.to(device, non_blocking=True)


def _apply_event_mcar(batch: Mapping[str, Any], rates: Sequence[float], *, seed: int, repeat: int,
                      device: torch.device, scores: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[float, Dict[str, Any]]]:
    batch_dev = dict(batch)
    if not all((not isinstance(v, torch.Tensor)) or v.device == device for v in batch_dev.values()):
        batch_dev = _batch_to_device(batch_dev, device)
    tokens = batch_dev["state_tokens"]
    values = batch_dev["state_values"]
    event_pad = batch_dev["state_pad_mask"].bool()
    seq_pad = batch_dev["seq_pad_mask"].bool()
    times = batch_dev["state_times"].float()
    hadm_ids = [str(x) for x in batch["hadm_id"]]
    valid_event = ~event_pad
    if scores is None:
        scores = _stable_scores(event_pad.shape, hadm_ids, int(seed), int(repeat), device, valid_event=valid_event)
    else:
        scores = scores.to(device, non_blocking=True).float()
        if tuple(scores.shape) != tuple(event_pad.shape):
            raise ValueError("Precomputed MCAR scores shape does not match state_event mask shape.")
    original_event_count = int(valid_event.sum().item())
    original_timestamp_count = int((~seq_pad).sum().item())
    C = len(rates)
    B, T, S = event_pad.shape
    expanded_tokens = tokens.unsqueeze(0).repeat(C, 1, 1, 1).reshape(C * B, T, S)
    expanded_values = values.unsqueeze(0).repeat(C, 1, 1, 1, 1).reshape(C * B, T, S, 1)
    expanded_event_pad = event_pad.unsqueeze(0).repeat(C, 1, 1, 1).reshape(C * B, T, S).clone()
    expanded_forced = torch.zeros_like(expanded_event_pad)
    expanded_seq_pad = seq_pad.unsqueeze(0).repeat(C, 1, 1).reshape(C * B, T)
    expanded_times = times.unsqueeze(0).repeat(C, 1, 1, 1).reshape(C * B, T, 1)
    stats: Dict[float, Dict[str, Any]] = {}
    for c, rate in enumerate(rates):
        dropped = valid_event & (scores < float(rate))
        retained = valid_event & (~dropped)
        forced = torch.zeros_like(retained)
        already_forced = torch.zeros((B,), dtype=torch.bool, device=device)
        for window_name in WINDOW_ORDER:
            cutoff = WINDOW_BY_KIND[window_name]
            window_ts = (~seq_pad) if cutoff is None else ((~seq_pad) & (times.squeeze(-1) <= float(cutoff)))
            window_valid = valid_event & window_ts.unsqueeze(-1)
            window_retained = retained & window_ts.unsqueeze(-1)
            empty = window_valid.flatten(1).any(dim=1) & (~window_retained.flatten(1).any(dim=1)) & (~already_forced)
            if bool(empty.any().item()):
                masked_scores = scores.masked_fill(~window_valid, -1.0)
                flat_idx = masked_scores.flatten(1).argmax(dim=1)
                rows = empty.nonzero(as_tuple=False).flatten()
                forced[rows, flat_idx[rows] // S, flat_idx[rows] % S] = True
                already_forced[rows] = True
        retained = retained | forced
        new_pad = ~retained
        sl = slice(c * B, (c + 1) * B)
        expanded_event_pad[sl] = new_pad
        expanded_forced[sl] = forced
        expanded_tokens[sl] = expanded_tokens[sl].masked_fill(new_pad, 0)
        expanded_values[sl] = expanded_values[sl].masked_fill(new_pad.unsqueeze(-1), 0.0)
        retained_ts = (~seq_pad) & retained.any(dim=-1)
        expanded_seq_pad[sl] = ~retained_ts
        window_stats: Dict[str, Dict[str, int]] = {}
        for window_name in WINDOW_ORDER:
            cutoff = WINDOW_BY_KIND[window_name]
            window_ts = (~seq_pad) if cutoff is None else ((~seq_pad) & (times.squeeze(-1) <= float(cutoff)))
            original_window = valid_event & window_ts.unsqueeze(-1)
            retained_window = retained & window_ts.unsqueeze(-1)
            forced_window = forced & window_ts.unsqueeze(-1)
            window_stats[window_name] = {
                "original_event_count": int(original_window.sum().item()),
                "retained_event_count": int(retained_window.sum().item()),
                "original_timestamp_count": int(window_ts.sum().item()),
                "retained_timestamp_count": int((window_ts & retained.any(dim=-1)).sum().item()),
                "forced_event_count": int(forced_window.sum().item()),
                "forced_patient_count": int(forced_window.flatten(1).any(dim=1).sum().item()),
            }
        stats[float(rate)] = {
            "original_event_count": original_event_count,
            "retained_event_count": int(retained.sum().item()),
            "original_timestamp_count": original_timestamp_count,
            "retained_timestamp_count": int(retained_ts.sum().item()),
            "forced_nonempty_count": int(forced.sum().item()),
            "forced_patient_count": int(forced.flatten(1).any(dim=1).sum().item()),
            "windows": window_stats,
        }
    return {
        "state_tokens": expanded_tokens,
        "state_values": expanded_values,
        "state_pad_mask": expanded_event_pad,
        "seq_pad_mask": expanded_seq_pad,
        "state_times": expanded_times,
        "forced_mask": expanded_forced,
        "hadm_id": [hid for _ in rates for hid in hadm_ids],
    }, stats


def _compact_latents_vectorized(latent: torch.Tensor, times: torch.Tensor, valid_ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latent.ndim != 3 or times.ndim != 2 or valid_ts.shape != times.shape:
        raise ValueError("Expected latent=[N,T,D], times=[N,T], valid_ts=[N,T].")
    N, _, D = latent.shape
    lengths = valid_ts.sum(dim=1).long()
    if (lengths < 1).any():
        raise RuntimeError("Robustness constructed an all-padding sequence; check cutoff cohort and forced-nonempty policy.")
    max_len = int(lengths.max().item())
    out_z = latent.new_zeros((N, max_len, D))
    out_t = times.new_zeros((N, max_len))
    out_pad = torch.ones((N, max_len), dtype=torch.bool, device=latent.device)
    rank = valid_ts.long().cumsum(dim=1) - 1
    rows = torch.arange(N, device=latent.device).view(N, 1).expand_as(valid_ts)
    out_z[rows[valid_ts], rank[valid_ts]] = latent[valid_ts]
    out_t[rows[valid_ts], rank[valid_ts]] = times[valid_ts]
    pos = torch.arange(max_len, device=latent.device).view(1, max_len)
    out_pad = pos >= lengths.view(N, 1)
    return out_z, out_t, out_pad


def _teacher_representations(teacher: JEPAModel, dropped: Mapping[str, torch.Tensor], rates: Sequence[float],
                             *, condition_chunk_size: int, device: torch.device,
                             precision: str = "bf16-mixed",
                             window_cohort_ids: Optional[Mapping[str, set[str]]] = None,
                             timings: Optional[Dict[str, float]] = None) -> Dict[float, Dict[str, Dict[str, torch.Tensor]]]:
    total = int(dropped["state_tokens"].size(0))
    C = len(rates)
    B = total // C
    out: Dict[float, Dict[str, Dict[str, torch.Tensor]]] = {float(rate): {} for rate in rates}
    chunk_size = max(1, int(condition_chunk_size)) * B
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        with torch.inference_mode():
            t0 = time.perf_counter()
            with _inference_precision_context(device, precision):
                seq_pad = dropped["seq_pad_mask"][start:end].bool()
                frontend = teacher.encode_teacher_sets(
                    dropped["state_tokens"][start:end], dropped["state_values"][start:end], dropped["state_pad_mask"][start:end].bool(), seq_pad
                )
                z_set_full = frontend["z_set"].detach()
            if timings is not None:
                timings["phase1_frontend_seconds"] = timings.get("phase1_frontend_seconds", 0.0) + (time.perf_counter() - t0)
            times_full = dropped["state_times"][start:end].float().squeeze(-1)
            valid_full = (~seq_pad) & (~dropped["state_pad_mask"][start:end].bool()).any(dim=-1)
        global_indices = torch.arange(start, end, device=device)
        rate_indices = torch.div(global_indices, B, rounding_mode="floor")
        row_indices = (global_indices % B).detach().cpu().tolist()
        for rate_pos, rate in enumerate(rates):
            select = rate_indices == int(rate_pos)
            if not bool(select.any().item()):
                continue
            selected_rows = [row_indices[i] for i in select.nonzero(as_tuple=False).flatten().detach().cpu().tolist()]
            for window_name, cutoff in WINDOW_BY_KIND.items():
                if window_cohort_ids is not None and window_name not in window_cohort_ids:
                    continue
                allowed = None if window_cohort_ids is None else window_cohort_ids.get(window_name)
                if allowed is not None:
                    keep_positions = [i for i, row in enumerate(selected_rows) if str(dropped["hadm_id"][row]) in allowed]
                    if not keep_positions:
                        continue
                    keep_t = torch.as_tensor(keep_positions, dtype=torch.long, device=device)
                    window_rows = [selected_rows[i] for i in keep_positions]
                    source_z = z_set_full[select].index_select(0, keep_t)
                    source_times = times_full[select].index_select(0, keep_t)
                    source_valid = valid_full[select].index_select(0, keep_t)
                else:
                    window_rows = selected_rows
                    source_z = z_set_full[select]
                    source_times = times_full[select]
                    source_valid = valid_full[select]
                current_valid = source_valid if cutoff is None else (source_valid & (source_times <= float(cutoff)))
                z, t, pad = _compact_latents_vectorized(source_z, source_times, current_valid)
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    with _inference_precision_context(device, precision):
                        z_context = teacher.contextualize_teacher_z_set(z, pad, t).detach()
                    if timings is not None:
                        timings["context_backbone_seconds"] = timings.get("context_backbone_seconds", 0.0) + (time.perf_counter() - t0)
                target = out[float(rate)].setdefault(window_name, {"z_set": [], "z_context": [], "time_min": [], "pad": [], "row": []})
                target["z_set"].append(z)
                target["z_context"].append(z_context)
                target["time_min"].append(t)
                target["pad"].append(pad)
                target["row"].extend(window_rows)
    packed: Dict[float, Dict[str, Dict[str, torch.Tensor]]] = {float(rate): {} for rate in rates}
    for rate, by_window in out.items():
        for window_name, rows in by_window.items():
            max_t = max(int(x.size(1)) for x in rows["time_min"])
            d = int(rows["z_set"][0].size(-1))
            n = len(rows["row"])
            z_set = rows["z_set"][0].new_zeros((n, max_t, d))
            z_context = rows["z_context"][0].new_zeros((n, max_t, d))
            time_min = rows["time_min"][0].new_zeros((n, max_t))
            pad = torch.ones((n, max_t), dtype=torch.bool, device=rows["z_set"][0].device)
            offset = 0
            for zs, zc, tm, pm in zip(rows["z_set"], rows["z_context"], rows["time_min"], rows["pad"]):
                count, length = int(tm.size(0)), int(tm.size(1))
                z_set[offset:offset + count, :length] = zs
                z_context[offset:offset + count, :length] = zc
                time_min[offset:offset + count, :length] = tm
                pad[offset:offset + count, :length] = pm
                offset += count
            packed[rate][window_name] = {"z_set": z_set, "z_context": z_context, "time_min": time_min, "pad": pad, "row": rows["row"]}
    return packed


def _teacher_representations_retry(teacher: JEPAModel, dropped: Mapping[str, torch.Tensor], rates: Sequence[float],
                                   *, condition_chunk_size: int, device: torch.device,
                                   precision: str = "bf16-mixed",
                                   window_cohort_ids: Optional[Mapping[str, set[str]]] = None,
                                   timings: Dict[str, float]) -> Tuple[Dict[float, Dict[str, Dict[str, torch.Tensor]]], int]:
    chunk = max(1, int(condition_chunk_size))
    retries = 0
    while True:
        try:
            return _teacher_representations(
                teacher, dropped, rates, condition_chunk_size=chunk, device=device,
                precision=precision, window_cohort_ids=window_cohort_ids, timings=timings,
            ), retries
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            retries += 1


def _probe_batch_gpu(rep: Mapping[str, torch.Tensor], source: str, hadm_ids: Sequence[str],
                     task: str, device: torch.device) -> Dict[str, Any]:
    row_order = [int(x) for x in rep["row"]]
    keep = list(range(len(row_order)))
    keep_t = torch.as_tensor(keep, dtype=torch.long, device=device)
    latent = rep[source].index_select(0, keep_t).float()
    time_min = rep["time_min"].index_select(0, keep_t).float()
    seq_pad_mask = rep["pad"].index_select(0, keep_t).bool()
    lengths = (~seq_pad_mask).sum(dim=1).long()
    if (lengths < 1).any():
        raise RuntimeError(f"All-padding robustness sequence reached probe task={task}.")
    delta = build_delta_time_min(time_min, seq_pad_mask, validate=True)
    ids = [str(hadm_ids[row_order[i]]) for i in keep]
    return {
        "latent_seq": latent, "time_min": time_min, "delta_time_min": delta,
        "seq_pad_mask": seq_pad_mask, "lengths": lengths,
        "hadm_id": ids,
    }


def _forward_probe(model: torch.nn.Module, batch: Mapping[str, Any], device: torch.device, precision: str) -> Tuple[list[str], torch.Tensor, Any]:
    batch_dev = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.inference_mode():
        with _inference_precision_context(device, precision):
            output = model(batch_dev)
    target = batch_dev["target"].detach().float().cpu() if "target" in batch_dev else torch.empty((0,), dtype=torch.float32)
    if isinstance(output, tuple):
        logits = tuple(x.detach().float().cpu() for x in output)
    else:
        logits = output.detach().float().cpu()
    return [str(x) for x in batch["hadm_id"]], target, logits


def _audit_task_window(batch: Mapping[str, Any], dropped: Mapping[str, torch.Tensor], rates: Sequence[float],
                       rate: float, task_ids: Sequence[str], window_name: str, device: torch.device) -> Dict[str, int]:
    task_set = set(map(str, task_ids))
    hadm_ids = [str(x) for x in batch["hadm_id"]]
    rows = [i for i, hid in enumerate(hadm_ids) if hid in task_set]
    if not rows:
        return {
            "original_event_count": 0, "retained_event_count": 0,
            "original_timestamp_count": 0, "retained_timestamp_count": 0,
            "forced_event_count": 0, "forced_patient_count": 0,
        }
    row_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    B = len(hadm_ids)
    rate_pos = list(map(float, rates)).index(float(rate))
    dropped_rows = row_t + rate_pos * B
    event_pad = batch["state_pad_mask"].to(device, non_blocking=True).bool().index_select(0, row_t)
    seq_pad = batch["seq_pad_mask"].to(device, non_blocking=True).bool().index_select(0, row_t)
    times = batch["state_times"].to(device, non_blocking=True).float().squeeze(-1).index_select(0, row_t)
    cutoff = WINDOW_BY_KIND[window_name]
    window_ts = (~seq_pad) if cutoff is None else ((~seq_pad) & (times <= float(cutoff)))
    original = (~event_pad) & window_ts.unsqueeze(-1)
    retained_pad = dropped["state_pad_mask"].index_select(0, dropped_rows).bool()
    retained_seq = dropped["seq_pad_mask"].index_select(0, dropped_rows).bool()
    retained_ts = (~retained_seq) if cutoff is None else ((~retained_seq) & (times <= float(cutoff)))
    retained = (~retained_pad) & retained_ts.unsqueeze(-1)
    forced = dropped["forced_mask"].index_select(0, dropped_rows).bool() & window_ts.unsqueeze(-1)
    return {
        "original_event_count": int(original.sum().item()),
        "retained_event_count": int(retained.sum().item()),
        "original_timestamp_count": int(window_ts.sum().item()),
        "retained_timestamp_count": int((retained_ts & retained.any(dim=-1)).sum().item()),
        "forced_event_count": int(forced.sum().item()),
        "forced_patient_count": int(forced.flatten(1).any(dim=1).sum().item()),
    }


def _condition_buffer_bytes(task: str, baseline: Mapping[str, Any]) -> int:
    logits = baseline["logits"]
    if task == "drg":
        return int(logits[0].numel() + logits[1].numel()) * 2
    return int(logits.numel()) * 2


def _init_condition_state(task: str, baseline: Mapping[str, Any]) -> Dict[str, Any]:
    n = len(baseline["ids"])
    if task == "drg":
        label_shape = tuple(baseline["logits"][0].shape[1:])
        card_shape = tuple(baseline["logits"][1].shape[1:])
        buffer: Any = (
            torch.empty((n, *label_shape), dtype=torch.float16),
            torch.empty((n, *card_shape), dtype=torch.float16),
        )
    else:
        buffer = torch.empty(tuple(baseline["logits"].shape), dtype=torch.float16)
    return {
        "ids": [str(x) for x in baseline["ids"]],
        "id_to_pos": {str(hid): i for i, hid in enumerate(baseline["ids"])},
        "logits_buffer": buffer,
        "filled": torch.zeros((n,), dtype=torch.bool),
        "original_event_count": 0, "retained_event_count": 0,
        "original_timestamp_count": 0, "retained_timestamp_count": 0,
        "forced_nonempty_count": 0, "forced_patient_count": 0,
        "prediction_buffer_bytes": _condition_buffer_bytes(task, baseline),
    }


def _assign_logits_to_state(task: str, state: Dict[str, Any], ids: Sequence[str], logits: Any) -> None:
    positions = torch.as_tensor([state["id_to_pos"][str(hid)] for hid in ids], dtype=torch.long)
    if state["filled"][positions].any():
        raise RuntimeError("Duplicate robustness prediction assignment.")
    state["filled"][positions] = True
    if task == "drg":
        state["logits_buffer"][0][positions] = logits[0].detach().cpu().to(torch.float16)
        state["logits_buffer"][1][positions] = logits[1].detach().cpu().to(torch.float16)
    else:
        state["logits_buffer"][positions] = logits.detach().cpu().to(torch.float16)


def _index_logits(task: str, logits: Any, indices: Sequence[int]) -> Any:
    idx = torch.as_tensor(indices, dtype=torch.long)
    if task == "drg":
        return logits[0][idx], logits[1][idx]
    return logits[idx]


def _primary_metric_fast(task: str, target: torch.Tensor, logits: Any) -> float:
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        y = target.float().numpy()
        p = torch.sigmoid(logits.float()).numpy()
        return float(compute_binary_metrics(y, p).get("auprc", float("nan")))
    if task in {"icd", "hcup"}:
        from sklearn.metrics import average_precision_score
        y = target.float().numpy()
        p = torch.sigmoid(logits.float()).numpy()
        if task == "icd":
            return float(average_precision_score(y.ravel(), p.ravel()))
        per_label = []
        for col in range(y.shape[1]):
            if len(np.unique(y[:, col])) > 1:
                per_label.append(float(average_precision_score(y[:, col], p[:, col])))
        return float(np.mean(per_label)) if per_label else float("nan")
    if task == "drg":
        y = target.float().numpy().astype(bool)
        pred = CompetitiveDRGProbe.decode(logits[0].float(), logits[1].float()).numpy()
        tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))
    raise ValueError(f"Unsupported task for primary bootstrap: {task}")


def _baseline_metric_result(metrics_json: Mapping[str, Any], task: str, source: str) -> Mapping[str, Any]:
    matches = [
        row for row in metrics_json.get("results", [])
        if str(row.get("task")) == str(task) and str(row.get("source_name", row.get("source"))) == str(source)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Baseline metrics must contain exactly one result for {task}/{source}; found {len(matches)}.")
    return matches[0]


def _load_prediction_npz(path: str, task: str) -> Tuple[list[str], torch.Tensor, Any]:
    arr = np.load(path, allow_pickle=True)
    ids = [str(x) for x in arr["hadm_id"].tolist()]
    target = torch.as_tensor(arr["target"]).float()
    if task == "drg":
        logits = (torch.as_tensor(arr["label_logits"]).float(), torch.as_tensor(arr["cardinality_logits"]).float())
    else:
        logits = torch.as_tensor(arr["logits"]).float()
    return ids, target, logits


def _targets_tensor_for_ids(targets: Mapping[str, Any], ids: Sequence[str]) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for hid in ids:
        key = str(hid)
        if key not in targets:
            raise RuntimeError(f"Missing expected target for admission {key}.")
        values.append(torch.as_tensor(targets[key]))
    if not values:
        return torch.empty((0,), dtype=torch.float32)
    shapes = [tuple(value.shape) for value in values]
    if len(set(shapes)) != 1:
        raise RuntimeError(f"Inconsistent expected target shapes: {sorted(set(shapes))}")
    return torch.stack(values, dim=0).float()


def _load_baselines(args: argparse.Namespace, tasks: Sequence[str], sources: Sequence[str],
                    probes: Mapping[Tuple[str, str], Dict[str, Any]],
                    task_cache_ids: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not args.baseline_eval_root:
        raise RuntimeError("--baseline_eval_root is required and must point to eval.py formal MIMIC suite output.")
    root = os.path.abspath(args.baseline_eval_root)
    metrics_path = os.path.join(root, "mimic_test_metrics_full.json")
    if not os.path.isfile(metrics_path):
        raise RuntimeError(f"Missing baseline metrics: {metrics_path}")
    with open(metrics_path, encoding="utf-8") as handle:
        metrics_json = json.load(handle)
    if str(metrics_json.get("split")) != "test":
        raise RuntimeError("Baseline eval suite must be split=test.")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task in tasks:
        for source in sources:
            result = _baseline_metric_result(metrics_json, task, source)
            pred_path = os.path.join(root, "predictions", f"{task}_{source}_test.npz")
            if not os.path.isfile(pred_path):
                raise RuntimeError(
                    f"Missing baseline predictions: {pred_path}. Re-run eval.py with --save_predictions."
                )
            probe_payload = probes[(task, source)]
            checkpoint = os.path.realpath(probe_payload["checkpoint"])
            if str(result.get("split")) != "test" or str(result.get("task")) != task or str(result.get("source_name", result.get("source"))) != source:
                raise RuntimeError(f"Baseline result task/source/split mismatch for {task}/{source}.")
            if result.get("phase1_checkpoint_sha256") != probe_payload["artifact"].get("phase1_checkpoint_sha256"):
                raise RuntimeError(f"Baseline Phase-1 provenance mismatch for {task}/{source}.")
            if result.get("cache_schema_version") != probe_payload["artifact"].get("cache_schema_version"):
                raise RuntimeError(f"Baseline cache schema mismatch for {task}/{source}.")
            if "cache_signature" in result and result["cache_signature"] != probe_payload["artifact"].get("cache_signature"):
                raise RuntimeError(f"Baseline cache signature mismatch for {task}/{source}.")
            if "probe_checkpoint_sha256" in result and result["probe_checkpoint_sha256"] != _sha256(checkpoint):
                raise RuntimeError(f"Baseline probe checkpoint SHA mismatch for {task}/{source}.")
            result_precision = str(result.get("precision", result.get("probe_precision", result.get("inference_precision", ""))))
            if not result_precision:
                raise RuntimeError(f"Baseline result lacks precision field for {task}/{source}.")
            requested_probe_precision = str(getattr(args, "probe_precision", "auto") or "auto")
            if requested_probe_precision != "auto" and requested_probe_precision != result_precision:
                raise RuntimeError(
                    f"Probe precision mismatch for {task}/{source}: baseline={result_precision} cli={requested_probe_precision}"
                )
            ids, target, logits = _load_prediction_npz(pred_path, task)
            if len(ids) != len(set(ids)) or int(result.get("n")) != len(ids):
                raise RuntimeError(f"Baseline prediction IDs invalid for {task}/{source}.")
            if task_cache_ids is not None:
                canonical_ids = [str(x) for x in task_cache_ids[task]]
                if len(ids) != len(canonical_ids) or set(ids) != set(canonical_ids):
                    raise RuntimeError(f"Baseline prediction IDs do not match canonical {task} test cache ID set.")
            expected_targets = probe_payload["targets"]
            expected = _targets_tensor_for_ids(expected_targets, ids)
            if expected.shape != target.shape or not torch.equal(expected, target):
                raise RuntimeError(f"Baseline prediction targets do not match labels/probe targets for {task}/{source}.")
            score, recomputed = _task_metric(task, target, logits)
            primary = str(result.get("primary_metric", _primary_metric_name(task)))
            expected_score = float(result.get("primary_value", result.get("score")))
            if not np.isfinite(score) or abs(float(score) - expected_score) > 5e-4:
                raise RuntimeError(f"Baseline primary metric mismatch for {task}/{source}: npz={score} json={expected_score}")
            for name, value in recomputed.items():
                if name in result.get("metrics", {}) and np.isfinite(float(value)):
                    if abs(float(value) - float(result["metrics"][name])) > 5e-4:
                        raise RuntimeError(f"Baseline metric mismatch for {task}/{source}/{name}.")
            out[(task, source)] = {
                "ids": ids, "target": target, "logits": logits, "metrics": dict(result.get("metrics", {})),
                "primary_metric": primary, "primary_value": expected_score, "score": expected_score,
                "prediction_path": pred_path, "eval_result": result, "precision": result_precision,
            }
    return out


def _resolve_probe_precision(args: argparse.Namespace, baselines: Mapping[Tuple[str, str], Mapping[str, Any]]) -> str:
    requested = str(getattr(args, "probe_precision", "auto") or "auto")
    precisions = sorted({str(payload["precision"]) for payload in baselines.values()})
    if requested == "auto":
        if len(precisions) != 1:
            raise RuntimeError(f"Selected baseline results use multiple probe precisions: {precisions}")
        return precisions[0]
    if any(precision != requested for precision in precisions):
        raise RuntimeError(f"Probe precision mismatch: baseline={precisions} cli={requested}")
    return requested


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _inference_precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or str(precision) in {"32", "32-true", "fp32"}:
        return contextlib.nullcontext()
    if str(precision) in {"bf16", "bf16-mixed"}:
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if str(precision) in {"fp16", "fp16-mixed", "16", "16-mixed"}:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported robustness precision {precision!r}.")


def _bootstrap_reps_by_task(args: argparse.Namespace, tasks: Sequence[str]) -> Dict[str, int]:
    defaults = {task: (0 if task == "icd" else int(args.bootstrap_reps)) for task in tasks}
    raw = str(getattr(args, "bootstrap_reps_by_task", "") or "").strip()
    if not raw:
        return defaults
    out = dict(defaults)
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError("--bootstrap_reps_by_task must use task=reps comma-separated entries.")
        task, value = [x.strip() for x in part.split("=", 1)]
        if task not in out:
            raise ValueError(f"Unknown task in --bootstrap_reps_by_task: {task}")
        reps = int(value)
        if reps < 0:
            raise ValueError("bootstrap reps must be non-negative.")
        out[task] = reps
    return out


def _bootstrap_draws(n: int, reps: int, seed: int) -> np.ndarray:
    if int(reps) < 0:
        raise ValueError("--bootstrap_reps must be non-negative.")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(reps), int(n)), endpoint=False, dtype=np.int32)


def _bootstrap_primary(task: str, target: torch.Tensor, logits: Any, draws: np.ndarray) -> np.ndarray:
    values = np.empty((draws.shape[0],), dtype=np.float64)
    for i, idx in enumerate(draws):
        values[i] = _primary_metric_fast(task, target[idx], _index_logits(task, logits, idx))
    return values


def _write_prediction_npz(path: str, task: str, ids: Sequence[str], target: torch.Tensor, logits: Any) -> None:
    payload: Dict[str, Any] = {"hadm_id": np.asarray(list(map(str, ids))), "target": target.detach().cpu().numpy()}
    if task == "drg":
        payload["label_logits"] = logits[0].detach().cpu().to(torch.float16).numpy()
        payload["cardinality_logits"] = logits[1].detach().cpu().to(torch.float16).numpy()
    else:
        payload["logits"] = logits.detach().cpu().to(torch.float16).numpy()
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _finalize(task: str, source: str, rate: float, repeat: int, state: Dict[str, Any],
              baseline: Mapping[str, Any], thresholds: Optional[Mapping[str, Any]],
              prediction_path: Optional[str], draws: np.ndarray, baseline_bootstrap: np.ndarray) -> Dict[str, Any]:
    ids = [str(x) for x in state["ids"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate predictions for {task}/{source}/rate={rate}/repeat={repeat}.")
    if not bool(state["filled"].all().item()):
        missing = [ids[i] for i, ok in enumerate(state["filled"].tolist()) if not ok]
        raise RuntimeError(f"Missing robustness predictions for {task}/{source}: {missing[:10]}")
    logits = state["logits_buffer"]
    if task == "drg":
        logits = (logits[0].float(), logits[1].float())
    else:
        logits = logits.float()
    baseline_ids = [str(x) for x in baseline["ids"]]
    if ids != baseline_ids:
        raise RuntimeError(f"Dropped prediction IDs do not match baseline order for {task}/{source}.")
    target = baseline["target"]
    score, metrics = _task_metric(task, target, logits)
    if task in {"icd", "hcup"} and thresholds is not None:
        metrics.update(multilabel_threshold_metrics(target, logits, thresholds))
    if prediction_path is not None:
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        _write_prediction_npz(prediction_path, task, ids, target, logits)
    dropped_bootstrap = _bootstrap_primary(task, target, logits, draws)
    delta_vector = dropped_bootstrap - baseline_bootstrap
    if delta_vector.size:
        ci_low = float(np.percentile(delta_vector, 2.5))
        ci_high = float(np.percentile(delta_vector, 97.5))
        ci_crosses_zero: Optional[bool] = bool(ci_low <= 0.0 <= ci_high)
    else:
        ci_low = ci_high = float("nan")
        ci_crosses_zero = None
    eval_baseline_value = float(baseline["primary_value"])
    absolute_delta = float(score - eval_baseline_value)
    oe = int(state["original_event_count"])
    re = int(state["retained_event_count"])
    ot = int(state["original_timestamp_count"])
    rt = int(state["retained_timestamp_count"])
    return {
        "task": task, "source": source, "split": "test", "missingness": "state_event_mcar",
        "drop_rate": float(rate), "repeat": int(repeat), "n": len(ids),
        "score": float(score), "primary_metric": _primary_metric_name(task), "primary_value": float(score),
        "eval_baseline_value": eval_baseline_value,
        "absolute_delta": absolute_delta,
        "relative_retention": float(score / eval_baseline_value) if eval_baseline_value != 0 else float("nan"),
        "bootstrap_delta": delta_vector.tolist(),
        "bootstrap_reps": int(draws.shape[0]),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_ci_crosses_zero": ci_crosses_zero,
        "metrics": metrics,
        "eval_baseline_metrics": dict(baseline.get("metrics", {})),
        "prediction_path": prediction_path,
        "original_event_count": oe, "retained_event_count": re,
        "original_timestamp_count": ot, "retained_timestamp_count": rt,
        "realized_event_retention": float(re / oe) if oe else 0.0,
        "realized_timestamp_retention": float(rt / ot) if ot else 0.0,
        "forced_nonempty_count": int(state["forced_nonempty_count"]),
        "forced_patient_count": int(state.get("forced_patient_count", 0)),
    }


def _metric_rows(results: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for result in results:
        for name, value in dict(result.get("metrics", {})).items():
            baseline_value = result.get("eval_baseline_metrics", {}).get(name)
            delta = None
            retention = None
            is_primary = name == result["primary_metric"]
            if baseline_value is not None and np.isfinite(float(value)) and np.isfinite(float(baseline_value)):
                delta = float(value) - float(baseline_value)
                retention = float(value) / float(baseline_value) if is_primary and float(baseline_value) != 0 else None
            rows.append({"task": result["task"], "source": result["source"], "drop_rate": result["drop_rate"],
                         "repeat": result["repeat"], "n": result["n"], "metric": name, "value": value,
                         "eval_baseline_value": baseline_value, "absolute_delta": delta,
                         "relative_retention": retention, "metric_direction": "higher_is_better" if is_primary else "secondary",
                         "is_primary": is_primary})
    return rows


def _retention_rows(results: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    fields = ["task", "source", "drop_rate", "repeat", "original_event_count", "retained_event_count",
              "original_timestamp_count", "retained_timestamp_count", "realized_event_retention",
              "realized_timestamp_retention", "forced_nonempty_count", "forced_patient_count"]
    return [{key: result.get(key) for key in fields} for result in results]


def _summary_rows(results: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault((str(result["task"]), str(result["source"]), float(result["drop_rate"])), []).append(result)
    rows: list[Dict[str, Any]] = []
    for (task, source, rate), reps in sorted(grouped.items()):
        reps = sorted(reps, key=lambda row: int(row["repeat"]))
        values = np.asarray([float(row["primary_value"]) for row in reps], dtype=np.float64)
        deltas = np.asarray([float(row["absolute_delta"]) for row in reps], dtype=np.float64)
        boot = np.asarray([row["bootstrap_delta"] for row in reps], dtype=np.float64)
        mean_boot = boot.mean(axis=0) if boot.size else np.asarray([], dtype=np.float64)
        ci_low = float(np.percentile(mean_boot, 2.5)) if mean_boot.size else float("nan")
        ci_high = float(np.percentile(mean_boot, 97.5)) if mean_boot.size else float("nan")
        rows.append({
            "task": task,
            "source": source,
            "drop_rate": rate,
            "n_repeats": len(reps),
            "primary_metric": reps[0]["primary_metric"],
            "eval_baseline_value": float(reps[0]["eval_baseline_value"]),
            "repeat_mean": float(values.mean()),
            "repeat_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mean_absolute_delta": float(deltas.mean()),
            "relative_retention": float(values.mean() / float(reps[0]["eval_baseline_value"])) if float(reps[0]["eval_baseline_value"]) != 0 else float("nan"),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_crosses_zero": bool(ci_low <= 0.0 <= ci_high) if np.isfinite(ci_low) and np.isfinite(ci_high) else None,
        })
    return rows


def _signature_payload(args: argparse.Namespace, tasks: Sequence[str], sources: Sequence[str],
                       drop_rates: Sequence[float], cache_signature: Mapping[str, Any],
                       phase1_sha: str, schema_sha: str, data_signature: str,
                       probes: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None) -> Dict[str, Any]:
    baseline_root = os.path.abspath(args.baseline_eval_root)
    baseline_predictions = {}
    for task in tasks:
        for source in sources:
            path = os.path.join(baseline_root, "predictions", f"{task}_{source}_test.npz")
            baseline_predictions[f"{task}/{source}"] = _sha256(path) if os.path.isfile(path) else None
    return {
        "robustness_version": ROBUSTNESS_VERSION,
        "script_sha256": _sha256(os.path.abspath(__file__)),
        "tasks": list(tasks),
        "sources": list(sources),
        "drop_rates": list(drop_rates),
        "seed": int(args.seed),
        "condition_chunk_size": int(args.condition_chunk_size),
        "encoder_precision": str(getattr(args, "encoder_precision", getattr(args, "precision", "bf16-mixed"))),
        "probe_precision": str(getattr(args, "probe_precision", "auto")),
        "save_predictions": bool(args.save_predictions),
        "phase1_checkpoint_sha256": phase1_sha,
        "schema_sha256": schema_sha,
        "dataset_signature": data_signature,
        "dataset_name": str(getattr(args, "dataset_name", "mimic_iv")),
        "cache_signature": cache_signature,
        "config_sha256": _sha256(os.path.abspath(args.config)) if os.path.isfile(args.config) else None,
        "probe_root": os.path.abspath(args.probe_root),
        "probe_sha256": {
            f"{task}/{source}": _sha256(probes[(task, source)]["checkpoint"])
            for task in tasks for source in sources
        } if probes is not None else {},
        "baseline_eval_root": os.path.abspath(args.baseline_eval_root),
        "baseline_metrics_sha256": _sha256(os.path.join(baseline_root, "mimic_test_metrics_full.json")),
        "baseline_prediction_sha256": baseline_predictions,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_reps_by_task": _bootstrap_reps_by_task(args, tasks),
    }


def _stable_json_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _repeat_paths(output_dir: str, repeat: int) -> Tuple[str, str]:
    root = os.path.join(output_dir, "repeats")
    return os.path.join(root, f"repeat_{int(repeat):03d}.json"), os.path.join(root, f"repeat_{int(repeat):03d}.complete")


def _load_repeat(output_dir: str, repeat: int, signature: str, *, save_predictions: bool = False) -> Optional[list[Dict[str, Any]]]:
    path, marker = _repeat_paths(output_dir, repeat)
    if not (os.path.isfile(path) and os.path.isfile(marker)):
        return None
    try:
        with open(marker, encoding="utf-8") as handle:
            if handle.read().strip() != signature:
                return None
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("signature") != signature or int(payload.get("repeat")) != int(repeat):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    if save_predictions:
        for result in results:
            pred = result.get("prediction_path")
            if not pred or not os.path.isfile(str(pred)):
                return None
    return results


def _save_repeat(output_dir: str, repeat: int, signature: str, results: Sequence[Mapping[str, Any]]) -> None:
    path, marker = _repeat_paths(output_dir, repeat)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_text(path, json.dumps(_json_safe({"repeat": int(repeat), "signature": signature, "results": list(results)}),
                                  indent=2, sort_keys=True, allow_nan=False))
    _atomic_text(marker, signature)


def _load_probes(args: argparse.Namespace, tasks: Sequence[str], sources: Sequence[str], device: torch.device,
                 cache_signature: Mapping[str, Any], phase1_sha: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task in tasks:
        for source in sources:
            checkpoint = _resolve_checkpoint(args.probe_root, task, source)
            model, artifact = load_exact_time_probe(checkpoint, device=device, expected_task=task, expected_source=source)
            if artifact.get("phase1_checkpoint_sha256") != phase1_sha or artifact.get("cache_signature") != cache_signature:
                raise RuntimeError(f"Probe/cache provenance mismatch for {task}/{source}: {checkpoint}")
            if str(artifact.get("transfer_mode", "")).lower().startswith("external"):
                raise RuntimeError(f"External-transfer probe artifact is not allowed for MIMIC robustness: {checkpoint}")
            thresholds = None
            if task in {"icd", "hcup"}:
                _validate_multilabel_threshold_artifact(artifact, checkpoint)
                thresholds = artifact["thresholds"]
            targets, _ = _targets_for_task(args.labels_csv, task, load_config_dict(args.config)["probe"], vocab_override=artifact.get("label_vocab"), allowed_splits=("test",))
            out[(task, source)] = {"model": model, "artifact": artifact, "checkpoint": checkpoint, "thresholds": thresholds, "targets": targets}
    return out


def _parity_preflight(teacher: JEPAModel, probes: Mapping[Tuple[str, str], Dict[str, Any]],
                      raw_ds: PatientSequenceDataset, cache_root: str, device: torch.device,
                      encoder_precision: str, probe_precision: str,
                      baselines: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
                      active_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    cache_by_window = {
        FULL_VISIBLE: load_dataset(cache_root, FULL_VISIBLE, split="test"),
        "retrospective_cutoff_1440": load_dataset(cache_root, RETROSPECTIVE_CUTOFF, split="test", cutoff_min=1440),
        "retrospective_cutoff_2880": load_dataset(cache_root, RETROSPECTIVE_CUTOFF, split="test", cutoff_min=2880),
    }
    cache_ds = cache_by_window[FULL_VISIBLE]
    full_ids = [str(x) for x in cache_ds.admission_ids()]
    active_set = set(map(str, active_ids)) if active_ids is not None else set(full_ids)
    ids = [hid for hid in full_ids if hid in active_set]
    length_map = dict(zip(full_ids, [int(x) for x in cache_ds.sequence_lengths()]))
    lengths = np.asarray([length_map[hid] for hid in ids], dtype=np.int64)
    if not ids:
        raise RuntimeError("Cannot run robustness parity preflight on an empty cache.")
    quantile_positions = np.unique(np.clip(np.quantile(np.arange(len(ids)), [0.1, 0.5, 0.9]).round().astype(int), 0, len(ids) - 1))
    length_order = np.argsort(lengths)
    chosen = [int(length_order[pos]) for pos in quantile_positions]
    raw_ids = raw_ds.admission_ids() if hasattr(raw_ds, "admission_ids") else _dataset_ids_from_files(raw_ds)
    raw_index = {hid: i for i, hid in enumerate(raw_ids)}
    max_z_set = max_z_context = max_cache_logit = max_baseline_logit = 0.0
    checked = 0
    for idx in chosen[:3]:
        hid = ids[idx]
        item = raw_ds[raw_index[hid]]
        batch = nested_padding_collate([item])
        batch_dev = _batch_to_device(batch, device)
        with torch.inference_mode():
            with _inference_precision_context(device, encoder_precision):
                seq_pad = batch_dev["seq_pad_mask"].bool()
                frontend = teacher.encode_teacher_sets(batch_dev["state_tokens"], batch_dev["state_values"], batch_dev["state_pad_mask"].bool(), seq_pad)
                z_set = frontend["z_set"].detach()
                times = batch_dev["state_times"].float().squeeze(-1)
                z_context = teacher.contextualize_teacher_z_set(z_set, seq_pad, times).detach()
        records = {}
        for window_name, ds in cache_by_window.items():
            stage_ids = [str(x) for x in ds.admission_ids()]
            if hid not in set(stage_ids):
                continue
            rec_idx = {current: pos for pos, current in enumerate(stage_ids)}[hid]
            rec = ds.sequence_record(rec_idx, "z_set")
            rec_context = ds.sequence_record(rec_idx, "z_context")
            n = int(rec["time_min"].numel())
            cutoff = WINDOW_BY_KIND[window_name]
            valid = (~batch_dev["seq_pad_mask"].bool())[0]
            if cutoff is not None:
                valid = valid & (times[0] <= float(cutoff))
            raw_times = times[0][valid].cpu()
            raw_z_set = z_set[0][valid].float().cpu()
            if cutoff is None:
                raw_z_context = z_context[0][valid].float().cpu()
            else:
                z_cut, t_cut, pad_cut = _compact_latents_vectorized(z_set[0:1], times[0:1], valid.view(1, -1))
                with torch.inference_mode():
                    with _inference_precision_context(device, encoder_precision):
                        raw_z_context_gpu = teacher.contextualize_teacher_z_set(z_cut, pad_cut, t_cut).detach()
                raw_z_context = raw_z_context_gpu[0, :int(raw_times.numel())].float().cpu()
            if n != int(raw_times.numel()) or not torch.allclose(raw_times, rec["time_min"].float(), atol=1e-4, rtol=0):
                raise RuntimeError(f"Robustness no-drop parity failed for {window_name} times: {hid}")
            max_z_set = max(max_z_set, float((raw_z_set - rec["z_set"].float()).abs().max().item()))
            max_z_context = max(max_z_context, float((raw_z_context - rec_context["z_context"].float()).abs().max().item()))
            records[window_name] = (rec, rec_context, raw_z_set, raw_z_context, raw_times)
        for (task, source), payload in probes.items():
            kind, cutoff = TASK_CACHE_KIND[task]
            window_name = FULL_VISIBLE if kind == FULL_VISIBLE else f"retrospective_cutoff_{int(cutoff)}"
            if window_name not in records:
                continue
            target = payload["targets"].get(hid)
            if target is None:
                continue
            rec, rec_context, raw_z_set, raw_z_context, raw_times = records[window_name]
            probe_batch = canonical_probe_collate([{"latent_seq": raw_z_set if source == "z_set" else raw_z_context, "time_min": raw_times, "target": target, "hadm_id": hid}])
            _, _, logits = _forward_probe(payload["model"], probe_batch, device, probe_precision)
            cache_batch = canonical_probe_collate([{"latent_seq": rec[source].float() if source in rec else rec_context[source].float(), "time_min": rec["time_min"].float(), "target": target, "hadm_id": hid}])
            _, _, cache_logits = _forward_probe(payload["model"], cache_batch, device, probe_precision)
            if isinstance(logits, tuple):
                diff = max(float((a - b).abs().max().item()) for a, b in zip(logits, cache_logits))
            else:
                diff = float((logits - cache_logits).abs().max().item())
            max_cache_logit = max(max_cache_logit, diff)
            if baselines is not None and (task, source) in baselines:
                baseline = baselines[(task, source)]
                if hid in set(baseline["ids"]):
                    pos = {current: i for i, current in enumerate(baseline["ids"])}[hid]
                    baseline_logits = _index_logits(task, baseline["logits"], [pos])
                    if isinstance(logits, tuple):
                        bdiff = max(float((a - b).abs().max().item()) for a, b in zip(logits, baseline_logits))
                    else:
                        bdiff = float((logits - baseline_logits).abs().max().item())
                    max_baseline_logit = max(max_baseline_logit, bdiff)
        checked += 1
    if max_z_set > 2e-2 or max_z_context > 7e-2:
        raise RuntimeError(
            "Robustness no-drop parity failed: "
            f"z_set={max_z_set} z_context={max_z_context} "
            f"cache_logit={max_cache_logit} baseline_logit={max_baseline_logit}"
        )
    return {
        "checked_admissions": checked,
        "z_set_max_abs": max_z_set,
        "z_context_max_abs": max_z_context,
        "cache_logit_max_abs": max_cache_logit,
        "baseline_logit_max_abs": max_baseline_logit,
        "cache_logit_checked_nonblocking": True,
        "baseline_logit_checked_nonblocking": True,
    }


def run_robustness(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config_dict(args.config)
    tasks = _selector(args.tasks, FORMAL_TASK_ORDER, FORMAL_TASK_ORDER)
    sources = _selector(args.sources, FORMAL_SOURCE_ORDER, FORMAL_SOURCE_ORDER)
    drop_rates = _rates(args.drop_rates)
    if not hasattr(args, "encoder_precision"):
        args.encoder_precision = getattr(args, "precision", "bf16-mixed")
    if not hasattr(args, "probe_precision"):
        args.probe_precision = "auto"
    if getattr(args, "split", "test") != "test":
        raise ValueError("Raw-event robustness is fixed to MIMIC test split.")
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be positive.")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    phase1_ckpt = os.path.abspath(args.phase1_ckpt)
    phase1_sha = _sha256(phase1_ckpt)
    schema_sha = _sha256(os.path.join(args.schema_dir, "schema_processed.csv"))
    teacher = JEPAModel.load_from_checkpoint(
        phase1_ckpt, map_location=device, schema_path=os.path.join(args.schema_dir, "schema_processed.csv"), weights_only=False,
    ).to(device).eval()
    teacher.requires_grad_(False)
    cache_signature = _load_cache_signature(args.cache_root)
    expected_cache_signature = build_phase1_cache_signature(
        data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=dict(cfg),
        teacher=teacher, phase1_ckpt_path=phase1_ckpt,
    )
    _assert_cache_signature(cache_signature, expected_cache_signature)
    data_signature = expected_cache_signature["dataset_signature"]
    task_ids = _task_cache_ids(args.cache_root, tasks)
    cache_ids = sorted(set().union(*(set(ids) for ids in task_ids.values())))
    raw_ds = _raw_dataset(args, cfg, cache_ids)
    raw_ids = _dataset_ids_from_files(raw_ds)
    if set(raw_ids) != set(cache_ids) or len(raw_ids) != len(set(raw_ids)):
        raise RuntimeError("Raw processed test cohort does not exactly match requested canonical task cache IDs.")
    raw_index = {hid: i for i, hid in enumerate(raw_ids)}
    raw_ds = _IndexedRawDataset(raw_ds, [raw_index[hid] for hid in cache_ids])
    probes = _load_probes(args, tasks, sources, device, cache_signature, phase1_sha)
    baselines = _load_baselines(args, tasks, sources, probes, task_ids)
    resolved_probe_precision = _resolve_probe_precision(args, baselines)
    for task in tasks:
        reference = baselines[(task, sources[0])]
        reference_target_by_id = {hid: reference["target"][i] for i, hid in enumerate(reference["ids"])}
        for source in sources[1:]:
            current = baselines[(task, source)]
            if set(current["ids"]) != set(reference["ids"]):
                raise RuntimeError(f"Baseline cohorts differ across sources for task={task}.")
            aligned_reference = torch.stack([torch.as_tensor(reference_target_by_id[hid]) for hid in current["ids"]]).float()
            if aligned_reference.shape != current["target"].shape or not torch.equal(aligned_reference, current["target"]):
                raise RuntimeError(f"Baseline targets differ across sources for task={task}.")
            if current["ids"] == reference["ids"]:
                current["target"] = reference["target"]
    peak_cpu_prediction_buffer_bytes = sum(
        _condition_buffer_bytes(task, baselines[(task, source)])
        for task in tasks for source in sources for _rate in drop_rates
    )
    signature_payload = _signature_payload(args, tasks, sources, drop_rates, cache_signature, phase1_sha, schema_sha, data_signature, probes)
    run_signature = _stable_json_sha(signature_payload)
    bootstrap_reps_by_task = _bootstrap_reps_by_task(args, tasks)
    if "icd" in bootstrap_reps_by_task and int(bootstrap_reps_by_task["icd"]) > 0:
        n_icd = len(baselines[("icd", sources[0])]["ids"]) if ("icd", sources[0]) in baselines else 0
        print(f"[robustness] ICD primary bootstrap enabled: reps={bootstrap_reps_by_task['icd']} admissions={n_icd}", flush=True)
    bootstrap_draws: Dict[str, np.ndarray] = {}
    bootstrap_draws_by_source: Dict[Tuple[str, str], np.ndarray] = {}
    baseline_bootstrap: Dict[Tuple[str, str], np.ndarray] = {}
    for task in tasks:
        first_source = sources[0]
        baseline_for_task = baselines[(task, first_source)]
        bootstrap_draws[task] = _bootstrap_draws(
            len(baseline_for_task["ids"]), int(bootstrap_reps_by_task[task]),
            _stable_seed(args.seed, "bootstrap", task, ",".join(baseline_for_task["ids"])),
        )
        for source in sources:
            baseline = baselines[(task, source)]
            source_pos = {hid: i for i, hid in enumerate(baseline["ids"])}
            source_draws = np.asarray(
                [source_pos[baseline_for_task["ids"][int(i)]] for i in bootstrap_draws[task].reshape(-1)],
                dtype=np.int32,
            ).reshape(bootstrap_draws[task].shape)
            bootstrap_draws_by_source[(task, source)] = source_draws
            baseline_bootstrap[(task, source)] = _bootstrap_primary(task, baseline["target"], baseline["logits"], source_draws)
    parity = _parity_preflight(
        teacher, probes, raw_ds, args.cache_root, device,
        str(args.encoder_precision), resolved_probe_precision, baselines, active_ids=cache_ids,
    )
    loader = _loader(args, cfg, raw_ds, device)
    started = time.perf_counter()
    timings: Dict[str, float] = {
        "loader_wait_seconds": 0.0,
        "missingness_seconds": 0.0,
        "phase1_frontend_seconds": 0.0,
        "context_backbone_seconds": 0.0,
        "probe_seconds": 0.0,
        "bootstrap_finalize_seconds": 0.0,
    }
    oom_retries = 0
    results = []
    for repeat in range(int(args.repeats)):
        if bool(args.resume):
            cached = _load_repeat(args.output_dir, repeat, run_signature, save_predictions=bool(args.save_predictions))
            if cached is not None:
                results.extend(cached)
                continue
        states: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
        for task in tasks:
            for source in sources:
                for rate in drop_rates:
                    states[(task, source, float(rate))] = _init_condition_state(task, baselines[(task, source)])
        iterator = iter(tqdm(loader, desc=f"robustness/raw_event_mcar/repeat{repeat}", ascii=True, mininterval=5.0))
        while True:
            t0 = time.perf_counter()
            try:
                batch_cpu = next(iterator)
            except StopIteration:
                break
            timings["loader_wait_seconds"] += time.perf_counter() - t0
            batch = _batch_to_device(dict(batch_cpu), device)
            hadm_ids = [str(x) for x in batch["hadm_id"]]
            event_pad = batch["state_pad_mask"].bool()
            scores = _stable_scores(event_pad.shape, hadm_ids, int(args.seed), repeat, device, valid_event=~event_pad)
            chunk_size = max(1, int(args.condition_chunk_size))
            window_cohort_ids = {
                window: set().union(*(set(task_ids[task]) for task in tasks
                                      if (FULL_VISIBLE if TASK_CACHE_KIND[task][0] == FULL_VISIBLE else f"retrospective_cutoff_{int(TASK_CACHE_KIND[task][1])}") == window))
                for window in WINDOW_ORDER
            }
            for rate_start in range(0, len(drop_rates), chunk_size):
                rate_chunk = drop_rates[rate_start:rate_start + chunk_size]
                t0 = time.perf_counter()
                dropped, _ = _apply_event_mcar(batch, rate_chunk, seed=int(args.seed), repeat=repeat, device=device,
                                               scores=scores)
                timings["missingness_seconds"] += time.perf_counter() - t0
                reps, retries = _teacher_representations_retry(
                    teacher, dropped, rate_chunk, condition_chunk_size=int(args.condition_chunk_size),
                    device=device, precision=str(args.encoder_precision), window_cohort_ids=window_cohort_ids,
                    timings=timings,
                )
                oom_retries += retries
                for rate in rate_chunk:
                    for task in tasks:
                        kind, cutoff = TASK_CACHE_KIND[task]
                        window = FULL_VISIBLE if kind == FULL_VISIBLE else f"retrospective_cutoff_{int(cutoff)}"
                        audit = _audit_task_window(batch, dropped, rate_chunk, float(rate), task_ids[task], window, device)
                        for source in sources:
                            state = states[(task, source, float(rate))]
                            for stat_key, stat_val in audit.items():
                                if stat_key == "forced_event_count":
                                    state["forced_nonempty_count"] += int(stat_val)
                                else:
                                    state[stat_key] += int(stat_val)
                            if window not in reps.get(float(rate), {}):
                                continue
                            payload = probes[(task, source)]
                            rep = reps[float(rate)][window]
                            probe_batch = _probe_batch_gpu(rep, source, hadm_ids, task, device)
                            t0 = time.perf_counter()
                            ids, target, logits = _forward_probe(payload["model"], probe_batch, device, resolved_probe_precision)
                            timings["probe_seconds"] += time.perf_counter() - t0
                            _assign_logits_to_state(task, state, ids, logits)
        repeat_results = []
        prediction_dir = os.path.join(args.output_dir, "predictions")
        for key, state in states.items():
            task, source, rate = key
            pred_path = os.path.join(prediction_dir, f"{task}_{source}_drop{rate:.6g}_repeat{repeat}.npz") if args.save_predictions else None
            t0 = time.perf_counter()
            repeat_results.append(_finalize(
                task, source, rate, repeat, state, baselines[(task, source)],
                probes[(task, source)]["thresholds"], pred_path,
                bootstrap_draws_by_source[(task, source)], baseline_bootstrap[(task, source)],
            ))
            timings["bootstrap_finalize_seconds"] += time.perf_counter() - t0
        _save_repeat(args.output_dir, repeat, run_signature, repeat_results)
        results.extend(repeat_results)
    metric_rows = _metric_rows(results)
    retention_rows = _retention_rows(results)
    summary_rows = _summary_rows(results)
    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "robustness_metrics.csv"), metric_rows,
               ["task", "source", "drop_rate", "repeat", "n", "metric", "value",
                "eval_baseline_value", "absolute_delta", "relative_retention", "metric_direction", "is_primary"])
    _write_csv(os.path.join(args.output_dir, "robustness_summary.csv"), summary_rows,
               ["task", "source", "drop_rate", "n_repeats", "primary_metric", "eval_baseline_value",
                "repeat_mean", "repeat_sd", "mean_absolute_delta", "relative_retention",
                "ci_low", "ci_high", "ci_crosses_zero"])
    _write_csv(os.path.join(args.output_dir, "missingness_audit.csv"), retention_rows,
               ["task", "source", "drop_rate", "repeat", "original_event_count", "retained_event_count",
                "original_timestamp_count", "retained_timestamp_count", "realized_event_retention",
                "realized_timestamp_retention", "forced_nonempty_count", "forced_patient_count"])
    report = {
        "robustness_version": ROBUSTNESS_VERSION,
        "script_path": os.path.abspath(__file__),
        "script_sha256": _sha256(os.path.abspath(__file__)),
        "missingness": "state_event_mcar",
        "split": "test",
        "tasks": list(tasks), "sources": list(sources), "drop_rates": list(drop_rates), "repeats": int(args.repeats),
        "args": vars(args),
        "phase1_ckpt": phase1_ckpt, "phase1_checkpoint_sha256": phase1_sha,
        "schema_sha256": schema_sha, "dataset_signature": data_signature,
        "dataset_name": str(getattr(args, "dataset_name", "mimic_iv")),
        "cache_signature": cache_signature,
        "expected_cache_signature": expected_cache_signature,
        "task_cache_ids": {task: len(ids) for task, ids in task_ids.items()},
        "no_drop_parity_preflight": parity,
        "condition_chunk_size": int(args.condition_chunk_size),
        "encoder_precision": str(args.encoder_precision),
        "probe_precision": resolved_probe_precision,
        "requested_probe_precision": str(args.probe_precision),
        "baseline_eval_root": None if args.baseline_eval_root is None else os.path.abspath(args.baseline_eval_root),
        "resume": bool(args.resume),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_reps_by_task": bootstrap_reps_by_task,
        "run_signature": run_signature,
        "signature_payload": signature_payload,
        "baseline_metrics_path": os.path.join(os.path.abspath(args.baseline_eval_root), "mimic_test_metrics_full.json"),
        "baseline_predictions": {
            f"{task}/{source}": baselines[(task, source)]["prediction_path"]
            for task in tasks for source in sources
        },
        "probe_checkpoints": {
            f"{task}/{source}": {
                "path": probes[(task, source)]["checkpoint"],
                "sha256": _sha256(probes[(task, source)]["checkpoint"]),
            }
            for task in tasks for source in sources
        },
        "runtime_seconds": time.perf_counter() - started,
        "patients_per_second": float((len(cache_ids) * int(args.repeats)) / max(time.perf_counter() - started, 1e-9)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cpu_prediction_buffer_bytes": int(peak_cpu_prediction_buffer_bytes),
        "stage_timings_seconds": timings,
        "oom_retry_count": int(oom_retries),
        "results": results,
        "summary": summary_rows,
    }
    _atomic_text(os.path.join(args.output_dir, "robustness_results.json"), json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    _atomic_text(os.path.join(args.output_dir, "manifest.json"), json.dumps(_json_safe({k: v for k, v in report.items() if k != "results"}), indent=2, sort_keys=True, allow_nan=False))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw-event MCAR robustness evaluation for canonical exact-time probes.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--schema_dir", required=True)
    parser.add_argument("--baseline_eval_root", required=True)
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--phase1_ckpt", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset_name", default="mimic_iv")
    parser.add_argument("--drop_rates", default="0.1,0.25,0.5,0.75")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--condition_chunk_size", type=int, default=2)
    parser.add_argument("--encoder_precision", default="bf16-mixed")
    parser.add_argument("--probe_precision", default="auto")
    parser.add_argument("--precision", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap_reps", type=int, default=200)
    parser.add_argument("--bootstrap_reps_by_task", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    return run_robustness(args)


if __name__ == "__main__":
    main()
