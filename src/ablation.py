"""Phase-1 masked numeric context ablation.

This CLI diagnoses which temporal context is used by the frozen Phase-1
masked numeric predictor.  Numeric decoding is delegated to the frozen
target-manifold readout artifact; this module does not fit decoders or
baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cache_bank import (
    READOUT_TRAINING_CACHE,
    _phase1_dataset_kwargs,
    build_readout_cache_signature,
    load_dataset,
    validate_readout_query_cache,
)
from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.model import JEPAModel
from src.readout import load_readout_artifact
from src.utils import load_config_dict


CONDITIONS: Tuple[str, ...] = (
    "query_only",
    "history_only",
    "current_only",
    "future_only",
    "history_current",
    "history_future",
    "current_future",
    "full_context",
)

CONDITION_FACTORS: Dict[str, Tuple[bool, bool, bool]] = {
    "query_only": (False, False, False),
    "history_only": (True, False, False),
    "current_only": (False, True, False),
    "future_only": (False, False, True),
    "history_current": (True, True, False),
    "history_future": (True, False, True),
    "current_future": (False, True, True),
    "full_context": (True, True, True),
}

FACTOR_NAMES = ("H", "C", "F")


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
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_csv(path: str, frame: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _slice_patient_batch(batch: Mapping[str, Any], patient_index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    skip_prefixes = ("packed_query_", "masked_targets_")
    for key, value in batch.items():
        if key.startswith(skip_prefixes):
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == len(batch["hadm_id"]):
            out[key] = value[patient_index:patient_index + 1].clone()
        elif key == "hadm_id":
            out[key] = [str(value[patient_index])]
        else:
            out[key] = value
    return out


def apply_hcf_ablation(batch: Mapping[str, Any], *, patient_index: int, t_idx: int,
                       slot_idx: int, condition: str,
                       canonical_masked_slots: Sequence[Tuple[int, int, int]] = ()) -> Dict[str, Any]:
    """Return one-patient batch with H/C/F visibility controlled exactly.

    H: timestamps < target timestamp.
    C: same timestamp, excluding the masked target slot.
    F: timestamps > target timestamp.

    The target timestamp remains a valid sequence position in all conditions.
    The target event itself remains present only as a masked query identity and
    is therefore invisible to the student path.
    """
    if condition not in CONDITION_FACTORS:
        raise ValueError(f"unknown ablation condition {condition!r}")
    keep_h, keep_c, keep_f = CONDITION_FACTORS[condition]
    out = _slice_patient_batch(batch, patient_index)
    state_pad = out["state_pad_mask"].bool()
    original_state_pad = state_pad.clone()
    original_seq_pad = out["seq_pad_mask"].bool().clone()
    t = int(t_idx)
    s = int(slot_idx)
    if t < 0 or t >= state_pad.size(1) or s < 0 or s >= state_pad.size(2):
        raise IndexError((t, s))
    if bool(original_state_pad[0, t, s]):
        raise RuntimeError("target slot is padded before ablation")

    timestamp_keep = torch.zeros_like(original_seq_pad)
    if keep_h and t > 0:
        timestamp_keep[:, :t] = ~original_seq_pad[:, :t]
    if keep_c:
        timestamp_keep[:, t] = ~original_seq_pad[:, t]
    if keep_f and t + 1 < timestamp_keep.size(1):
        timestamp_keep[:, t + 1:] = ~original_seq_pad[:, t + 1:]
    timestamp_keep[:, t] = True

    keep = torch.zeros_like(state_pad)
    if keep_h and t > 0:
        keep[:, :t, :] = ~original_state_pad[:, :t, :]
    if keep_c:
        keep[:, t, :] = ~original_state_pad[:, t, :]
    if keep_f and t + 1 < keep.size(1):
        keep[:, t + 1:, :] = ~original_state_pad[:, t + 1:, :]
    keep[:, t, s] = True
    other_canonical = {
        (int(mt), int(ms)) for mt, ms, _ in canonical_masked_slots
        if not (int(mt) == t and int(ms) == s)
    }
    for mt, ms in other_canonical:
        if 0 <= mt < keep.size(1) and 0 <= ms < keep.size(2):
            keep[:, mt, ms] = False

    out["state_pad_mask"] = original_state_pad | (~keep)
    out["state_pad_mask"][:, t, s] = False
    out["state_mask_labels"] = out["state_mask_labels"].clone()
    out["state_mask_labels"].zero_()
    out["state_mask_labels"][0, t, s, 0] = 1.0
    out["state_tokens"] = out["state_tokens"].masked_fill(out["state_pad_mask"], 0)
    out["state_tokens"][0, t, s] = batch["state_tokens"][patient_index, t, s]
    out["state_values"] = out["state_values"].masked_fill(out["state_pad_mask"].unsqueeze(-1), 0.0)
    out["state_values"][0, t, s, 0] = batch["state_values"][patient_index, t, s, 0]
    out["seq_pad_mask"] = original_seq_pad | (~timestamp_keep)
    out["seq_pad_mask"][0, t] = False

    if "inter_pad_mask" in out and "inter_t_step" in out:
        inter_t = out["inter_t_step"].long()
        inter_keep = torch.zeros_like(out["inter_pad_mask"].bool())
        if keep_h:
            inter_keep |= inter_t < t
        if keep_c:
            inter_keep |= inter_t == t
        if keep_f:
            inter_keep |= inter_t > t
        out["inter_pad_mask"] = out["inter_pad_mask"].bool() | (~inter_keep)
        for key in ("hist_inter_pad_mask",):
            if key in out:
                out[key] = out["inter_pad_mask"]

    if bool(out["seq_pad_mask"].all()):
        raise RuntimeError("ablation produced an all-padding sequence")
    if bool(out["state_pad_mask"][0, t, s]) or float(out["state_mask_labels"][0, t, s, 0]) <= 0.5:
        raise RuntimeError("ablation removed target query identity")
    for mt, ms in other_canonical:
        if 0 <= mt < out["state_pad_mask"].size(1) and 0 <= ms < out["state_pad_mask"].size(2):
            if not bool(out["state_pad_mask"][0, mt, ms]):
                raise RuntimeError("ablation leaked another canonical masked event")
    if bool(out["seq_pad_mask"][0, t]):
        raise RuntimeError("ablation removed target timestamp")
    return out






def _packed_record_list(batch: Mapping[str, Any], out: Mapping[str, torch.Tensor]) -> List[Tuple[Tuple[str, int, int, int], Dict[str, Any]]]:
    valid = ~out["packed_query_pad_mask"]
    z_pred = out["z_pred_packed"][valid].detach().float().cpu()
    z_target = out["target_z_event_packed"][valid].detach().float().cpu()
    value = out["packed_true_values"][valid].detach().float().cpu()
    tids = out["packed_timestamp_indices"][valid].detach().long().cpu()
    slots = out["packed_slot_indices"][valid].detach().long().cpu()
    packed_idx = out["packed_indices"].detach().long().cpu()
    hadm = [str(x) for x in batch["hadm_id"]]
    fids = batch["state_tokens"].detach().cpu()[packed_idx[:, 0], packed_idx[:, 1], packed_idx[:, 2]].long()
    bidx = out["packed_batch_indices"].detach().long().cpu()
    if not (z_pred.size(0) == z_target.size(0) == value.size(0) == tids.size(0) == slots.size(0) == fids.size(0) == bidx.size(0)):
        raise RuntimeError("packed output tensors are not aligned")
    records: List[Tuple[Tuple[str, int, int, int], Dict[str, Any]]] = []
    for i in range(int(z_pred.size(0))):
        key = (hadm[int(bidx[i])], int(tids[i]), int(slots[i]), int(fids[i]))
        records.append((key, {
            "z_pred": z_pred[i],
            "z_target": z_target[i],
            "target_value": float(value[i]),
            "feature_id": int(fids[i]),
        }))
    return records


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if y.shape[0] != pred.shape[0]:
        raise RuntimeError("metric length mismatch")
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite].astype(float)
    pred = pred[finite].astype(float)
    if y.size == 0:
        return {"count": 0, "mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    err = pred - y
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "count": int(y.size),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
    }




def _condition_subset(condition: str) -> Tuple[str, ...]:
    return tuple(name for name, keep in zip(FACTOR_NAMES, CONDITION_FACTORS[condition]) if keep)


def compute_shapley_from_errors(errors: Mapping[str, float]) -> Dict[str, float]:
    """Three-factor Shapley contribution to error reduction vs query_only."""
    n = 3
    by_subset = {_condition_subset(name): float(value) for name, value in errors.items()}
    out: Dict[str, float] = {}
    for factor in FACTOR_NAMES:
        phi = 0.0
        others = [x for x in FACTOR_NAMES if x != factor]
        for size in range(n):
            for subset in _subsets(others, size):
                s = tuple(sorted(subset, key=FACTOR_NAMES.index))
                sf = tuple(sorted((*subset, factor), key=FACTOR_NAMES.index))
                weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
                phi += weight * (by_subset[s] - by_subset[sf])
        out[factor] = float(phi)
    return out


def _subsets(items: Sequence[str], size: int) -> Iterable[Tuple[str, ...]]:
    if size == 0:
        yield ()
        return
    if size > len(items):
        return
    first, rest = items[0], items[1:]
    for tail in _subsets(rest, size - 1):
        yield (first, *tail)
    yield from _subsets(rest, size)


def clustered_bootstrap(values: Mapping[str, Sequence[float]], *, iterations: int = 1000,
                        seed: int = 42) -> Dict[str, Any]:
    groups = {str(k): np.asarray(v, dtype=float) for k, v in values.items() if len(v)}
    ids = list(groups)
    if not ids:
        return {"status": "not_evaluable", "paired_admissions": 0}
    pooled = np.concatenate([groups[k] for k in ids])
    rng = np.random.default_rng(int(seed))
    reps = []
    for _ in range(int(iterations)):
        draw = rng.choice(ids, size=len(ids), replace=True)
        vals = np.concatenate([groups[k] for k in draw])
        reps.append(float(np.mean(vals)))
    arr = np.asarray(reps, dtype=float)
    return {
        "status": "ok",
        "point": float(np.mean(pooled)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "win_probability": float(np.mean(arr < 0.0)),
        "valid_replicates": int(arr.size),
        "paired_admissions": int(len(ids)),
        "paired_queries": int(pooled.size),
    }


def _read_query_cache(cache_root: str, phase1_ckpt: str, schema_path: str, split: str) -> Tuple[List[Tuple[str, int, int, int]], Dict[Tuple[str, int, int, int], Dict[str, Any]], Dict[str, Any]]:
    ds = load_dataset(cache_root, READOUT_TRAINING_CACHE, split=split,
                      expected_phase1_ckpt=phase1_ckpt, expected_schema=schema_path)
    keys: List[Tuple[str, int, int, int]] = []
    rows: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    for i in tqdm(range(len(ds)), desc="ablation/readout_query_cache", unit="query", mininterval=5.0):
        item = ds[i]
        key = tuple(item["query_key"])
        if key in rows:
            raise RuntimeError(f"duplicate canonical query key: {key}")
        rows[key] = {
            "z_pred": item["z_pred"].float(),
            "z_target": item["z_target"].float(),
            "target_value": float(item["target_value"]),
            "feature_id": int(item["feature_id"]),
        }
        keys.append(key)
    meta_path = os.path.join(cache_root, READOUT_TRAINING_CACHE, split, "metadata.json")
    with open(meta_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    return keys, rows, metadata


def _canonical_masks_by_hadm(keys: Sequence[Tuple[str, int, int, int]]) -> Dict[str, List[Tuple[int, int, int]]]:
    out: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for hadm, t_idx, slot_idx, vid in keys:
        out[str(hadm)].append((int(t_idx), int(slot_idx), int(vid)))
    return dict(out)


def _keys_by_hadm(keys: Sequence[Tuple[str, int, int, int]]) -> Dict[str, List[Tuple[str, int, int, int]]]:
    out: Dict[str, List[Tuple[str, int, int, int]]] = defaultdict(list)
    for key in keys:
        out[str(key[0])].append(key)
    return dict(out)


def _strict_cache_metadata_check(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in (
        "readout_cache_schema_version", "readout_cache_contract_version",
        "phase1_checkpoint_sha256", "phase1_architecture_version",
        "schema_sha256", "dataset_content_signature",
        "readout_masking_signature", "dataset_policy_signature",
        "fraction", "split", "latent_dimension",
    ):
        if metadata.get(key) != expected.get(key):
            raise RuntimeError(f"readout cache metadata mismatch for {key}")


def _ablation_layer_counts(cfg: Mapping[str, Any], teacher: Any) -> Tuple[int, int, str]:
    recipe = cfg.get("phase1", {}).get("recipe", {}) if isinstance(cfg, Mapping) else {}
    context_layers = recipe.get("context_layers", getattr(getattr(teacher, "config", None), "context_layers", 4))
    set_layers = recipe.get("set_layers", getattr(getattr(teacher, "config", None), "set_layers", 4))
    try:
        return max(1, int(context_layers)), max(1, int(set_layers)), "config_or_model"
    except Exception:
        return 4, 4, "conservative_default"


def _dense_shape_and_cost(source_batch: Mapping[str, Any], *, n_context_layers: int,
                          n_set_layers: int) -> Tuple[int, int, int]:
    tokens = source_batch["state_tokens"]
    if int(tokens.size(0)) != 1:
        raise RuntimeError("ablation replay source DataLoader must yield exactly one admission")
    t = int(tokens.shape[1])
    s = int(tokens.shape[2])
    cost = int(n_context_layers) * t * t + int(n_set_layers) * t * s * s + t * s + t
    return t, s, max(1, int(cost))


def _repeat_admission_batch(source_batch: Mapping[str, Any], n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in source_batch.items():
        if key.startswith(("packed_query_", "masked_targets_")):
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) == 1:
            reps = [int(n)] + [1] * (value.ndim - 1)
            out[key] = value.repeat(*reps).clone()
        elif key == "hadm_id":
            out[key] = [str(value[0]) for _ in range(int(n))]
        else:
            out[key] = value
    return out


def _materialize_chunk(source_batch: Mapping[str, Any], chunk: Sequence[Mapping[str, Any]],
                       canonical_masked_slots: Sequence[Tuple[int, int, int]]) -> Dict[str, Any]:
    if not chunk:
        raise ValueError("cannot materialize an empty ablation chunk")
    micro = _repeat_admission_batch(source_batch, len(chunk))
    original_state_pad = source_batch["state_pad_mask"][0].bool()
    original_seq_pad = source_batch["seq_pad_mask"][0].bool()
    original_tokens = source_batch["state_tokens"][0]
    original_values = source_batch["state_values"][0]
    other_by_target: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for job in chunk:
        target = (int(job["t_idx"]), int(job["slot_idx"]))
        other_by_target.setdefault(target, [
            (int(mt), int(ms)) for mt, ms, _ in canonical_masked_slots
            if not (int(mt) == target[0] and int(ms) == target[1])
        ])
    for row, job in enumerate(chunk):
        condition = str(job["condition"])
        if condition not in CONDITION_FACTORS:
            raise ValueError(f"unknown ablation condition {condition!r}")
        keep_h, keep_c, keep_f = CONDITION_FACTORS[condition]
        t = int(job["t_idx"])
        s = int(job["slot_idx"])
        if t < 0 or t >= original_state_pad.size(0) or s < 0 or s >= original_state_pad.size(1):
            raise IndexError((t, s))
        if bool(original_state_pad[t, s]):
            raise RuntimeError("target slot is padded before ablation")
        timestamp_keep = torch.zeros_like(original_seq_pad)
        if keep_h and t > 0:
            timestamp_keep[:t] = ~original_seq_pad[:t]
        if keep_c:
            timestamp_keep[t] = ~original_seq_pad[t]
        if keep_f and t + 1 < timestamp_keep.size(0):
            timestamp_keep[t + 1:] = ~original_seq_pad[t + 1:]
        timestamp_keep[t] = True

        keep = torch.zeros_like(original_state_pad)
        if keep_h and t > 0:
            keep[:t, :] = ~original_state_pad[:t, :]
        if keep_c:
            keep[t, :] = ~original_state_pad[t, :]
        if keep_f and t + 1 < keep.size(0):
            keep[t + 1:, :] = ~original_state_pad[t + 1:, :]
        keep[t, s] = True
        for mt, ms in other_by_target[(t, s)]:
            if 0 <= mt < keep.size(0) and 0 <= ms < keep.size(1):
                keep[mt, ms] = False

        micro["state_pad_mask"][row] = original_state_pad | (~keep)
        micro["state_pad_mask"][row, t, s] = False
        micro["state_mask_labels"][row].zero_()
        micro["state_mask_labels"][row, t, s, 0] = 1.0
        micro["state_tokens"][row].masked_fill_(micro["state_pad_mask"][row], 0)
        micro["state_tokens"][row, t, s] = original_tokens[t, s]
        micro["state_values"][row].masked_fill_(micro["state_pad_mask"][row].unsqueeze(-1), 0.0)
        micro["state_values"][row, t, s, 0] = original_values[t, s, 0]
        micro["seq_pad_mask"][row] = original_seq_pad | (~timestamp_keep)
        micro["seq_pad_mask"][row, t] = False
        if "inter_pad_mask" in micro and "inter_t_step" in micro:
            inter_t = micro["inter_t_step"][row].long()
            inter_keep = torch.zeros_like(micro["inter_pad_mask"][row].bool())
            if keep_h:
                inter_keep |= inter_t < t
            if keep_c:
                inter_keep |= inter_t == t
            if keep_f:
                inter_keep |= inter_t > t
            micro["inter_pad_mask"][row] = micro["inter_pad_mask"][row].bool() | (~inter_keep)
            if "hist_inter_pad_mask" in micro:
                micro["hist_inter_pad_mask"][row] = micro["inter_pad_mask"][row]
        if bool(micro["seq_pad_mask"][row].all()):
            raise RuntimeError("ablation produced an all-padding sequence")
        if bool(micro["state_pad_mask"][row, t, s]) or float(micro["state_mask_labels"][row, t, s, 0]) <= 0.5:
            raise RuntimeError("ablation removed target query identity")
        if bool(micro["seq_pad_mask"][row, t]):
            raise RuntimeError("ablation removed target timestamp")
    return micro


def _execute_chunk(teacher: Any, source_batch: Mapping[str, Any], chunk: Sequence[Mapping[str, Any]],
                   canonical_masked_slots: Sequence[Tuple[int, int, int]], device: torch.device,
                   telemetry: Dict[str, Any], cost_per_job: int) -> List[Tuple[Tuple[Tuple[str, int, int, int], str], Dict[str, Any]]]:
    micro = _materialize_chunk(source_batch, chunk, canonical_masked_slots)
    micro_dev = _batch_to_device(micro, device)
    out = teacher._forward_phase1_numeric_jepa(micro_dev, output_mode="readout_cache")
    packed_rows = _packed_record_list(micro, out)
    if len(packed_rows) != len(chunk):
        raise RuntimeError("ablation microbatch produced an unexpected number of packed queries")
    telemetry["forward_count"] += 1
    telemetry["actual_microbatch_sizes"].append(len(chunk))
    telemetry["estimated_attention_costs"].append(int(cost_per_job) * len(chunk))
    out_rows = []
    for row_idx, (job, (local_key, selected)) in enumerate(zip(chunk, packed_rows)):
        key = job["query_key"]
        condition = str(job["condition"])
        expected_local_key = (str(micro["hadm_id"][row_idx]), int(key[1]), int(key[2]), int(key[3]))
        if tuple(local_key) != tuple(expected_local_key):
            raise RuntimeError(f"condition {condition} lost target query key {key}")
        out_rows.append(((key, condition), selected))
    return out_rows


def _run_admission_microbatches(teacher: Any, source_batch: Mapping[str, Any],
                                jobs: Sequence[Mapping[str, Any]], canonical_masked_slots: Sequence[Tuple[int, int, int]],
                                device: torch.device, forward_batch_size: int, max_attention_cost: int,
                                n_context_layers: int, n_set_layers: int,
                                safe_batch_size: Dict[Tuple[int, int], int]) -> Tuple[Dict[Tuple[Tuple[str, int, int, int], str], Dict[str, Any]], Dict[str, Any]]:
    results: Dict[Tuple[Tuple[str, int, int, int], str], Dict[str, Any]] = {}
    telemetry: Dict[str, Any] = {
        "forward_count": 0,
        "oom_retries": 0,
        "planned_microbatch_sizes": [],
        "actual_microbatch_sizes": [],
        "estimated_attention_costs": [],
        "t_values": [],
        "s_values": [],
        "cost_per_job_values": [],
    }
    if not jobs:
        return results, telemetry
    t, s, cost_per_job = _dense_shape_and_cost(
        source_batch, n_context_layers=n_context_layers, n_set_layers=n_set_layers
    )
    shape_key = (t, s)
    telemetry["t_values"].append(t)
    telemetry["s_values"].append(s)
    telemetry["cost_per_job_values"].append(cost_per_job)
    capacity_by_cost = max(1, int(max_attention_cost) // max(cost_per_job, 1))
    base_size = min(max(1, int(forward_batch_size)), capacity_by_cost, int(safe_batch_size.get(shape_key, 10**9)), len(jobs))
    start = 0
    while start < len(jobs):
        planned = min(base_size, len(jobs) - start)
        telemetry["planned_microbatch_sizes"].append(int(planned))
        size = int(planned)
        while True:
            chunk = jobs[start:start + size]
            try:
                for out_key, selected in _execute_chunk(
                    teacher, source_batch, chunk, canonical_masked_slots, device, telemetry, cost_per_job
                ):
                    results[out_key] = selected
                start += size
                break
            except torch.cuda.OutOfMemoryError:
                telemetry["oom_retries"] += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if size <= 1:
                    raise RuntimeError(f"ablation replay OOM for single job at dense shape T={t}, S={s}") from None
                size = max(1, size // 2)
                safe_batch_size[shape_key] = size
                base_size = min(base_size, size)
    return results, telemetry


def _decode_condition_records(readout: Any,
                              condition_records: Mapping[Tuple[Tuple[str, int, int, int], str], Mapping[str, Any]],
                              device: torch.device) -> Dict[Tuple[Tuple[str, int, int, int], str], Dict[str, Any]]:
    if not condition_records:
        return {}
    keys = list(condition_records.keys())
    z_pred_cpu = torch.stack([condition_records[key]["z_pred"].float() for key in keys], dim=0)
    z_target_cpu = torch.stack([condition_records[key]["z_target"].float() for key in keys], dim=0)
    feature_cpu = torch.tensor([int(condition_records[key]["feature_id"]) for key in keys], dtype=torch.long)
    with torch.inference_mode():
        decoded = readout(z_pred_cpu.to(device, non_blocking=True), feature_cpu.to(device, non_blocking=True))
    decoded_cpu = {name: value.detach().cpu().float() for name, value in decoded.items()}
    diff = z_pred_cpu - z_target_cpu
    latent_l1 = diff.abs().mean(dim=1)
    latent_l2 = diff.pow(2).sum(dim=1).sqrt()
    latent_cosine = torch.nn.functional.cosine_similarity(z_pred_cpu, z_target_cpu, dim=1)
    latent_norm_ratio = z_pred_cpu.norm(dim=1) / z_target_cpu.norm(dim=1).clamp_min(1e-12)
    out: Dict[Tuple[Tuple[str, int, int, int], str], Dict[str, Any]] = {}
    for i, key in enumerate(keys):
        row = condition_records[key]
        out[key] = {
            "z_pred": row["z_pred"],
            "z_target": row["z_target"],
            "target_value": float(row["target_value"]),
            "feature_id": int(row["feature_id"]),
            "prediction": float(decoded_cpu["numeric_mean"][i]),
            "numeric_sigma": float(decoded_cpu["numeric_sigma"][i]),
            "min_energy": float(decoded_cpu["min_energy"][i]),
            "posterior_entropy": float(decoded_cpu["posterior_entropy"][i]),
            "boundary_mass": float(decoded_cpu["boundary_mass"][i]),
            "latent_l1": float(latent_l1[i]),
            "latent_l2": float(latent_l2[i]),
            "latent_cosine": float(latent_cosine[i]),
            "latent_norm_ratio": float(latent_norm_ratio[i]),
        }
    return out


def run_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    cfg = load_config_dict(args.config)
    schema_path = os.path.join(args.schema_dir, "schema_processed.csv")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    teacher = JEPAModel.load_from_checkpoint(
        os.path.realpath(args.phase1_ckpt), map_location=device, weights_only=False,
        schema_path=schema_path,
    ).to(device).eval()
    teacher.requires_grad_(False)
    readout = load_readout_artifact(
        args.readout_artifact, map_location=device,
        expected_phase1_ckpt=args.phase1_ckpt,
        expected_schema=schema_path,
        expected_phase1_architecture=getattr(teacher, "phase1_architecture_version", None),
    ).to(device).eval()
    readout.requires_grad_(False)

    max_queries = None if args.max_queries is None or int(args.max_queries) <= 0 else int(args.max_queries)
    expected_signature = build_readout_cache_signature(
        data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg,
        phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split=args.split,
        fraction=float(args.fraction), max_seq_len_limit=args.max_seq_len_limit,
    )
    cache_metadata = validate_readout_query_cache(args.cache_dir, args.split, expected_signature)
    _strict_cache_metadata_check(cache_metadata, expected_signature)
    all_ordered_keys, canonical, cache_meta = _read_query_cache(args.cache_dir, args.phase1_ckpt, schema_path, args.split)
    target_keys = all_ordered_keys[:max_queries] if max_queries is not None else list(all_ordered_keys)
    wanted = set(target_keys)
    if not target_keys:
        raise RuntimeError("ablation query cohort is empty")

    ds_kwargs = _phase1_dataset_kwargs(cfg, max_seq_len_limit=args.max_seq_len_limit)
    ds = PatientSequenceDataset(
        data_dir=args.data_dir, schema_dir=args.schema_dir, split=args.split, train_pct=1.0,
        fraction=float(args.fraction), deterministic=True, **ds_kwargs,
    )
    pin_memory = device.type == "cuda"
    loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=pin_memory,
                        num_workers=int(args.num_workers), collate_fn=nested_padding_collate)

    out_tmp = os.path.abspath(args.output_dir) + f".tmp.{os.getpid()}"
    if os.path.isdir(out_tmp):
        shutil.rmtree(out_tmp)
    os.makedirs(out_tmp, exist_ok=True)

    condition_values: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
    condition_targets: List[float] = []
    condition_vids: List[int] = []
    paired_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    shapley_values: Dict[str, List[float]] = {f: [] for f in FACTOR_NAMES}
    shapley_efficiency_errors: List[float] = []
    seen: List[Tuple[str, int, int, int]] = []
    writer: Optional[pq.ParquetWriter] = None
    rows_buffer: List[Dict[str, Any]] = []
    canonical_masks = _canonical_masks_by_hadm(all_ordered_keys)
    keys_by_hadm = _keys_by_hadm(target_keys)
    n_context_layers, n_set_layers, layer_count_source = _ablation_layer_counts(cfg, teacher)
    safe_batch_size: Dict[Tuple[int, int], int] = {}
    forward_count = 0
    oom_retries = 0
    planned_microbatch_sizes: List[int] = []
    actual_microbatch_sizes: List[int] = []
    estimated_attention_costs: List[int] = []
    t_values: List[int] = []
    s_values: List[int] = []
    cost_per_job_values: List[int] = []
    admission_count = 0
    full_context_max_abs_z_pred_diff = 0.0
    full_context_max_abs_z_target_diff = 0.0
    printed_first_batching_plan = False
    print(
        f"[ablation] source_loader_batch_size=1 forward_batch_size={int(args.forward_batch_size)} "
        f"max_attention_cost={int(args.max_attention_cost)}",
        flush=True,
    )

    def flush_rows() -> None:
        nonlocal writer, rows_buffer
        if not rows_buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(rows_buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(os.path.join(out_tmp, "per_query_ablation.parquet"), table.schema)
        writer.write_table(table)
        rows_buffer = []

    pbar = tqdm(total=len(target_keys), desc="ablation/queries", unit="query", mininterval=5.0)
    with torch.inference_mode():
        for batch in loader:
            hadm_ids = [str(x) for x in batch["hadm_id"]]
            if len(hadm_ids) != 1:
                raise RuntimeError("ablation source DataLoader must use admission-level batch_size=1")
            admission_count += 1
            hadm = hadm_ids[0]
            batch_keys = []
            batch_keys.extend(keys_by_hadm.get(hadm, []))
            batch_keys = [key for key in batch_keys if key in wanted]
            jobs: List[Dict[str, Any]] = []
            for key in batch_keys:
                hadm, t_idx, slot_idx, feature_id = key
                b = 0
                if not bool((batch["state_tokens"][b, int(t_idx), int(slot_idx)] == int(feature_id)).item()):
                    raise RuntimeError(f"dataset/query cache identity mismatch for {key}")
                for condition in CONDITIONS:
                    jobs.append({
                        "query_key": key,
                        "condition": condition,
                        "t_idx": int(t_idx),
                        "slot_idx": int(slot_idx),
                    })
            if not jobs:
                continue
            condition_records, telemetry = _run_admission_microbatches(
                teacher, batch, jobs, canonical_masks.get(hadm, []), device,
                int(args.forward_batch_size), int(args.max_attention_cost),
                n_context_layers, n_set_layers, safe_batch_size,
            )
            if not printed_first_batching_plan:
                print(
                    "[ablation] first active admission batching: "
                    f"T={telemetry['t_values'][0] if telemetry['t_values'] else 'NA'} "
                    f"S={telemetry['s_values'][0] if telemetry['s_values'] else 'NA'} "
                    f"cost_per_job={telemetry['cost_per_job_values'][0] if telemetry['cost_per_job_values'] else 'NA'} "
                    f"planned_microbatch={telemetry['planned_microbatch_sizes'][0] if telemetry['planned_microbatch_sizes'] else 'NA'} "
                    f"actual_microbatch={telemetry['actual_microbatch_sizes'][0] if telemetry['actual_microbatch_sizes'] else 'NA'}",
                    flush=True,
                )
                printed_first_batching_plan = True
            decoded_records = _decode_condition_records(readout, condition_records, device)
            forward_count += int(telemetry["forward_count"])
            oom_retries += int(telemetry["oom_retries"])
            planned_microbatch_sizes.extend([int(x) for x in telemetry["planned_microbatch_sizes"]])
            actual_microbatch_sizes.extend([int(x) for x in telemetry["actual_microbatch_sizes"]])
            estimated_attention_costs.extend([int(x) for x in telemetry["estimated_attention_costs"]])
            t_values.extend([int(x) for x in telemetry["t_values"]])
            s_values.extend([int(x) for x in telemetry["s_values"]])
            cost_per_job_values.extend([int(x) for x in telemetry["cost_per_job_values"]])
            for key in batch_keys:
                if key in seen:
                    raise RuntimeError(f"query key repeated during ablation replay: {key}")
                hadm, t_idx, slot_idx, feature_id = key
                per_condition: Dict[str, Dict[str, Any]] = {}
                for condition in CONDITIONS:
                    selected = decoded_records[(key, condition)]
                    target_value = float(selected["target_value"])
                    if int(selected["feature_id"]) != int(feature_id):
                        raise RuntimeError(f"condition {condition} changed target VID for {key}")
                    if float(target_value) != float(canonical[key]["target_value"]):
                        raise RuntimeError(f"condition {condition} changed target value for {key}")
                    if condition == "full_context":
                        dz = float((selected["z_pred"] - canonical[key]["z_pred"]).abs().max().item())
                        dzt = float((selected["z_target"] - canonical[key]["z_target"]).abs().max().item())
                        full_context_max_abs_z_pred_diff = max(full_context_max_abs_z_pred_diff, dz)
                        full_context_max_abs_z_target_diff = max(full_context_max_abs_z_target_diff, dzt)
                        if dz > float(args.full_context_atol) or dzt > float(args.full_context_atol):
                            raise RuntimeError(f"full-context replay does not match canonical cache for {key}: z_pred_diff={dz}, z_target_diff={dzt}")
                    pred = float(selected["prediction"])
                    sigma = float(selected["numeric_sigma"])
                    min_energy = float(selected["min_energy"])
                    posterior_entropy = float(selected["posterior_entropy"])
                    boundary_mass = float(selected["boundary_mass"])
                    latent = {
                        "latent_l1": float(selected["latent_l1"]),
                        "latent_l2": float(selected["latent_l2"]),
                        "latent_cosine": float(selected["latent_cosine"]),
                        "latent_norm_ratio": float(selected["latent_norm_ratio"]),
                    }
                    err = (pred - target_value) ** 2
                    finite_values = [pred, sigma, min_energy, posterior_entropy, boundary_mass, err, target_value, latent["latent_l1"], latent["latent_l2"], latent["latent_cosine"], latent["latent_norm_ratio"]]
                    if not all(math.isfinite(float(x)) for x in finite_values):
                        raise RuntimeError(f"non-finite ablation output for {key}/{condition}")
                    per_condition[condition] = {"prediction": pred, "error": err}
                    rows_buffer.append({
                        "hadm_id": hadm, "t_idx": int(t_idx), "slot_idx": int(slot_idx),
                        "feature_id": int(feature_id), "query_key": str(key),
                        "condition": condition, "target_value": target_value,
                        "prediction": pred, "numeric_sigma": sigma,
                        "squared_error": err, "absolute_error": abs(pred - target_value),
                        "min_energy": min_energy,
                        "posterior_entropy": posterior_entropy,
                        "boundary_mass": boundary_mass,
                        **latent,
                    })
                    condition_values[condition].append(pred)
                y = float(canonical[key]["target_value"])
                condition_targets.append(y)
                condition_vids.append(int(feature_id))
                for condition in CONDITIONS:
                    paired_values[f"{condition}_minus_query_only"][hadm].append(per_condition[condition]["error"] - per_condition["query_only"]["error"])
                    paired_values[f"{condition}_minus_full_context"][hadm].append(per_condition[condition]["error"] - per_condition["full_context"]["error"])
                shapley = compute_shapley_from_errors({c: per_condition[c]["error"] for c in CONDITIONS})
                efficiency_error = abs(sum(shapley.values()) - (per_condition["query_only"]["error"] - per_condition["full_context"]["error"]))
                shapley_efficiency_errors.append(float(efficiency_error))
                if efficiency_error > 1.0e-8:
                    raise RuntimeError(f"Shapley efficiency check failed for {key}: {efficiency_error}")
                for factor, value in shapley.items():
                    shapley_values[factor].append(float(value))
                seen.append(key)
                pbar.update(1)
                if len(rows_buffer) >= int(args.parquet_row_group_size):
                    flush_rows()
                if max_queries is not None and len(seen) >= max_queries:
                    break
            if max_queries is not None and len(seen) >= max_queries:
                break
    pbar.close()
    flush_rows()
    if writer is not None:
        writer.close()
    if seen != target_keys:
        raise RuntimeError("ablation query order does not match canonical query cache")

    y_arr = np.asarray(condition_targets, dtype=float)
    vids = np.asarray(condition_vids, dtype=int)
    metrics_rows = []
    for condition in CONDITIONS:
        pred = np.asarray(condition_values[condition], dtype=float)
        pooled = _metrics(y_arr, pred)
        per_adm: Dict[str, List[float]] = defaultdict(list)
        condition_error_by_adm: Dict[str, List[float]] = defaultdict(list)
        per_vid = []
        per_vid_counts = []
        for key, yy, pp in zip(seen, y_arr, pred):
            se = (pp - yy) ** 2
            per_adm[str(key[0])].append(se)
            condition_error_by_adm[str(key[0])].append(se)
        admission_balanced_mse = float(np.mean([np.mean(v) for v in per_adm.values()])) if per_adm else float("nan")
        for vid in sorted(set(vids.tolist())):
            m = vids == vid
            per_vid.append(float(np.mean((pred[m] - y_arr[m]) ** 2)))
            per_vid_counts.append(int(m.sum()))
        ci = clustered_bootstrap(condition_error_by_adm, iterations=int(args.bootstrap_iters), seed=int(args.seed))
        adm_ci = clustered_bootstrap({k: [float(np.mean(v))] for k, v in per_adm.items()}, iterations=int(args.bootstrap_iters), seed=int(args.seed))
        metrics_rows.append({
            "condition": condition,
            **pooled,
            "admission_balanced_mse": admission_balanced_mse,
            "per_vid_macro_median_mse": float(np.median(per_vid)) if per_vid else float("nan"),
            "test_support_weighted_mse": float(np.average(np.asarray(per_vid), weights=np.asarray(per_vid_counts))) if per_vid else float("nan"),
            "pooled_mse_ci_low": ci.get("ci_low"), "pooled_mse_ci_high": ci.get("ci_high"),
            "admission_balanced_mse_ci_low": adm_ci.get("ci_low"), "admission_balanced_mse_ci_high": adm_ci.get("ci_high"),
        })
    _atomic_csv(os.path.join(out_tmp, "condition_metrics.csv"), pd.DataFrame(metrics_rows))

    delta_rows = []
    bootstrap_rows = []
    for name, by_adm in sorted(paired_values.items()):
        boot = clustered_bootstrap(by_adm, iterations=int(args.bootstrap_iters), seed=int(args.seed))
        bootstrap_rows.append({"comparison": name, **boot})
        delta_rows.append({"comparison": name, "mean_delta": boot.get("point"), "paired_admissions": boot.get("paired_admissions"), "paired_queries": boot.get("paired_queries")})
    _atomic_csv(os.path.join(out_tmp, "paired_deltas.csv"), pd.DataFrame(delta_rows))
    _atomic_csv(os.path.join(out_tmp, "bootstrap_intervals.csv"), pd.DataFrame(bootstrap_rows))

    shapley_rows = []
    for factor, values in shapley_values.items():
        arr = np.asarray(values, dtype=float)
        by_adm: Dict[str, List[float]] = defaultdict(list)
        by_vid: Dict[int, List[float]] = defaultdict(list)
        for key, value in zip(seen, arr):
            by_adm[str(key[0])].append(float(value))
            by_vid[int(key[3])].append(float(value))
        ci = clustered_bootstrap(by_adm, iterations=int(args.bootstrap_iters), seed=int(args.seed))
        vid_means = [float(np.mean(v)) for v in by_vid.values()]
        shapley_rows.append({
            "factor": factor,
            "mean_error_reduction": float(np.mean(arr)),
            "median_error_reduction": float(np.median(arr)),
            "admission_balanced_error_reduction": float(np.mean([np.mean(v) for v in by_adm.values()])),
            "per_vid_macro_median_error_reduction": float(np.median(vid_means)) if vid_means else float("nan"),
            "test_occurrence_weighted_error_reduction": float(np.average(vid_means, weights=[len(v) for v in by_vid.values()])) if vid_means else float("nan"),
            "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high"),
            "query_count": int(arr.size),
        })
    _atomic_csv(os.path.join(out_tmp, "shapley_contributions.csv"), pd.DataFrame(shapley_rows))

    parquet_path = os.path.join(out_tmp, "per_query_ablation.parquet")
    pq_file = pq.ParquetFile(parquet_path)
    expected_rows = len(seen) * len(CONDITIONS)
    if pq_file.metadata.num_rows != expected_rows:
        raise RuntimeError(f"parquet row count mismatch: {pq_file.metadata.num_rows} != {expected_rows}")

    def _summary(values: Sequence[int]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "median": 0.0, "p95": 0.0, "max": 0}
        arr = np.asarray(values, dtype=float)
        return {
            "min": int(np.min(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": int(np.max(arr)),
        }

    quality = {
        "status": "PASS",
        "source_loader_batch_size": 1,
        "source_loader_pin_memory": bool(pin_memory),
        "admission_count": int(admission_count),
        "query_count": int(len(seen)),
        "target_query_count": int(len(seen)),
        "condition_count": int(len(CONDITIONS)),
        "query_condition_replay_count": int(len(seen) * len(CONDITIONS)),
        "parquet_rows": int(pq_file.metadata.num_rows),
        "query_identity_hash": _stable_hash([list(k) for k in seen]),
        "conditions": {name: {"H": h, "C": c, "F": f} for name, (h, c, f) in CONDITION_FACTORS.items()},
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "forward_count": int(forward_count),
        "oom_retries": int(oom_retries),
        "queries_per_second": float(len(seen) / max(time.perf_counter() - started, 1e-12)),
        "query_condition_replays_per_second": float((len(seen) * len(CONDITIONS)) / max(time.perf_counter() - started, 1e-12)),
        "shape_count": int(len(set(zip(t_values, s_values)))),
        "t_summary": _summary(t_values),
        "s_summary": _summary(s_values),
        "cost_per_job_summary": _summary(cost_per_job_values),
        "planned_microbatch_size_summary": _summary(planned_microbatch_sizes),
        "actual_microbatch_size_summary": _summary(actual_microbatch_sizes),
        "safe_batch_size_by_shape": {f"{t}x{s}": int(v) for (t, s), v in sorted(safe_batch_size.items())},
        "max_estimated_attention_cost": int(max(estimated_attention_costs)) if estimated_attention_costs else 0,
        "ablation_cost_formula": "n_context_layers*T*T + n_set_layers*T*S*S + T*S + T",
        "ablation_cost_layers": {
            "n_context_layers": int(n_context_layers),
            "n_set_layers": int(n_set_layers),
            "source": layer_count_source,
        },
        "full_context_max_abs_z_pred_diff": float(full_context_max_abs_z_pred_diff),
        "full_context_max_abs_z_target_diff": float(full_context_max_abs_z_target_diff),
        "max_shapley_efficiency_error": float(max(shapley_efficiency_errors) if shapley_efficiency_errors else 0.0),
    }
    manifest = {
        "script": "src.ablation",
        "ablation_script_sha256": _sha256(__file__),
        "phase1_checkpoint": os.path.realpath(args.phase1_ckpt),
        "phase1_checkpoint_sha256": _sha256(args.phase1_ckpt),
        "schema_path": os.path.realpath(schema_path),
        "schema_sha256": _sha256(schema_path),
        "config_path": os.path.realpath(args.config),
        "config_sha256": _sha256(args.config),
        "data_dir": os.path.realpath(args.data_dir),
        "cache_dir": os.path.realpath(args.cache_dir),
        "readout_artifact": os.path.realpath(args.readout_artifact),
        "readout_artifact_sha256": _sha256(args.readout_artifact),
        "readout_cache_metadata": cache_meta,
        "conditions": quality["conditions"],
        "split": args.split,
        "max_queries": max_queries,
    }
    summary = {
        "status": "COMPLETE",
        "query_count": int(len(seen)),
        "condition_metrics": metrics_rows,
        "shapley": shapley_rows,
    }
    _atomic_json(os.path.join(out_tmp, "quality_control.json"), quality)
    _atomic_json(os.path.join(out_tmp, "run_manifest.json"), manifest)
    _atomic_json(os.path.join(out_tmp, "summary.json"), summary)

    final = os.path.abspath(args.output_dir)
    backup = f"{final}.old.{os.getpid()}"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(final):
        os.replace(final, backup)
    os.replace(out_tmp, final)
    if os.path.exists(backup):
        shutil.rmtree(backup)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 masked numeric H/C/F ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--schema_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--phase1_ckpt", required=True)
    parser.add_argument("--readout_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max_seq_len_limit", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Explicit smoke/debug cap. Formal runs should omit this.")
    parser.add_argument("--forward_batch_size", type=int, default=32,
                        help="Number of query-condition replays per frozen Phase-1 forward microbatch.")
    parser.add_argument("--max_attention_cost", type=int, default=67108864,
                        help="Maximum estimated sequence/set attention cost per ablation replay microbatch.")
    parser.add_argument("--full_context_atol", type=float, default=1.0e-4,
                        help="Maximum allowed absolute z_pred/z_target replay difference vs canonical cache.")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parquet_row_group_size", type=int, default=4096)
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
