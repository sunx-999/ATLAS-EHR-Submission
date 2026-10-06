import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import (
    benchmark_cache_root,
    benchmark_metadata_path,
    compute_binary_metrics,
    compute_multilabel_metrics,
    load_config_dict,
    load_labels_csv,
    load_schema_csv,
    set_seed,
)
from src.cache_bank import FULL_VISIBLE, RETROSPECTIVE_CUTOFF, load_dataset, _phase1_dataset_kwargs
from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.probes import TASK_CACHE_KIND, CompetitiveDRGProbe, _targets_for_task

SEQUENCE_BINARY_TASKS = {"mortality", "readmission", "los_ge_7d", "los_ge_3d"}
SEQUENCE_MULTILABEL_TASKS = {"icd", "hcup", "drg"}
SEQUENCE_TASKS = SEQUENCE_BINARY_TASKS | SEQUENCE_MULTILABEL_TASKS
SEQUENCE_TASK_ORDER = ["icd", "hcup", "drg", "mortality", "los_ge_7d", "los_ge_3d", "readmission"]
EXTERNAL_BINARY_TASKS = ("mortality", "los_ge_3d", "los_ge_7d")
CLASSIFICATION_BASELINES = {"lr", "xgb"}
TABULAR_SUMMARIES = ("last", "mean", "max", "presence", "count", "interval")
FEATURE_WINDOWS: Dict[str, Optional[float]] = {
    "window_1440": 1440.0,
    "window_2880": 2880.0,
    "full_visible": None,
}
FEATURE_BANK_VERSION = "benchmark_feature_bank_v2"
_XGBOOST_CAPABILITY_AUDIT_CACHE: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}


def _summaries_for_task() -> Tuple[str, ...]:
    return TABULAR_SUMMARIES


def _select_summaries(task: str, requested: Optional[Sequence[str]]) -> Tuple[str, ...]:
    base = list(_summaries_for_task())
    if not requested:
        return tuple(base)
    req = [str(x) for x in requested]
    selected = [s for s in base if s in req]
    if not selected:
        raise ValueError(f"No requested summaries available for task {task!r}: {req}")
    return tuple(selected)

# LOS threshold per task name
_LOS_THRESHOLD: Dict[str, float] = {
    "los_ge_7d": 7.0,
    "los_ge_3d": 3.0,
}

def _task_base_dir(output_dir: str, task: str, baseline: str) -> str:
    return os.path.join(str(output_dir), "benchmark", str(task), str(baseline))


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_bank_split_dir(cache_dir: str, split: str) -> str:
    return os.path.join(benchmark_cache_root(cache_dir), FEATURE_BANK_VERSION, str(split))


def _split_file_signature(data_dir: str, split: str) -> str:
    split_dir = os.path.join(str(data_dir), str(split))
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(split_dir, "*.pt"))):
        st = os.stat(p)
        h.update(json.dumps({
            "name": os.path.basename(p),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _feature_contract_signature(meta: Mapping[str, Any]) -> str:
    payload = {
        "version": FEATURE_BANK_VERSION,
        "dense_to_vid": list(meta["dense_to_vid"]),
        "dense_model_type": list(meta["dense_model_type"]),
        "schema_sha256": str(meta["schema_sha256"]),
        "train_data_signature": str(meta["train_data_signature"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _benchmark_dataset_policy(cfg_dict: Mapping[str, Any]) -> Dict[str, Any]:
    ds_kwargs = _phase1_dataset_kwargs(dict(cfg_dict), max_seq_len_limit=None)
    ds_kwargs["deterministic"] = True
    ds_kwargs["return_legacy_nested"] = False
    ds_kwargs["max_seq_len_limit"] = None
    ds_kwargs["mask_target_ratio"] = 0.0
    ds_kwargs["mask_rarity_strength"] = 0.0
    ds_kwargs["mask_min_prob"] = 0.0
    ds_kwargs["mask_max_prob"] = 0.0
    ds_kwargs["static_mask_prob"] = 0.0
    return ds_kwargs


def _write_run_manifest(output_dir: str, payload: Dict[str, Any]) -> None:
    p = os.path.join(str(output_dir), "benchmark", "run_manifest.json")
    _save_json(p, payload)


def _require_config_section(config: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
    cur: Any = config
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"Missing required config section: {'.'.join(path)}")
        cur = cur[p]
    if not isinstance(cur, dict):
        raise TypeError(f"Config section is not a dict: {'.'.join(path)}")
    return cur


def _ensure_no_inverse_transform_usage(schema_dir: str) -> None:
    schema_path = os.path.join(schema_dir, "schema_processed.csv")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema_processed.csv not found in {schema_dir}")
    schema = load_schema_csv(schema_path)
    # Hard guard: benchmark must never reconstruct raw-value space from mean/std.
    if "mean" in schema.columns or "std" in schema.columns:
        # Allowed to read schema for feature typing only; forbidden to use mean/std numerically.
        # This function is a runtime guard hook to make the policy explicit.
        return


def _load_labels_df(labels_csv: str) -> pd.DataFrame:
    df = load_labels_csv(labels_csv)
    if "hadm_id" not in df.columns or "split" not in df.columns:
        raise RuntimeError("labels.csv must contain hadm_id and split columns.")
    df["hadm_id_str"] = df["hadm_id"].astype(str)
    _assert_patient_level_split(df)
    return df


def _assert_patient_level_split(labels_df: pd.DataFrame, *, require_patient_id: bool = False) -> None:
    patient_col = None
    for candidate in ("subject_id", "uniquepid"):
        if candidate in labels_df.columns:
            patient_col = candidate
            break
    if patient_col is None:
        if require_patient_id:
            raise RuntimeError("Formal benchmark labels must contain subject_id or uniquepid for patient-level split validation.")
        return
    split_sets = labels_df.groupby(labels_df[patient_col].astype(str))["split"].apply(lambda s: set(s.astype(str)))
    leaked = {patient: sorted(splits) for patient, splits in split_sets.items() if len(splits) > 1}
    if leaked:
        examples = dict(list(leaked.items())[:10])
        raise RuntimeError(f"Benchmark patient-level split leakage detected using {patient_col}: {examples}")


def _resolve_path_existing(path: str, *, base_dir: Optional[str] = None) -> str:
    candidates = [str(path)]
    if base_dir and not os.path.isabs(str(path)):
        candidates.append(os.path.join(str(base_dir), str(path)))
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(str(path))


def _validate_hcup_mapping_artifact(path: str, *, config_dir: Optional[str] = None) -> Dict[str, Any]:
    resolved = _resolve_path_existing(path, base_dir=config_dir)
    with open(resolved, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("HCUP mapping artifact must be a JSON list.")
    ordered: List[Tuple[int, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or "index" not in row or "hcup_label" not in row:
            raise ValueError("HCUP mapping artifact rows must contain index and hcup_label.")
        ordered.append((int(row["index"]), str(row["hcup_label"])))
    ordered = sorted(ordered)
    indices = [idx for idx, _ in ordered]
    if indices != list(range(26)):
        raise ValueError("HCUP mapping artifact must define exactly 26 mapping-ordered labels with indices 0..25.")
    labels = [label for _, label in ordered]
    if len(set(labels)) != len(labels):
        raise ValueError("HCUP mapping artifact contains duplicate labels.")
    return {
        "path": resolved,
        "sha256": _sha256_file(resolved),
        "num_labels": 26,
        "codes": labels,
    }


def _binary_label_map(labels_df: pd.DataFrame, split: str, task: str, smoke_mode: bool = False) -> Dict[str, float]:
    _BINARY_LABEL_COLS_LOCAL = {
        "mortality":  "label_mortality",
        "readmission": "label_readmission_30d",
        "los_ge_7d":  "label_los_days",
        "los_ge_3d":  "label_los_days",
    }
    target_col = _BINARY_LABEL_COLS_LOCAL[task]
    if bool(smoke_mode):
        part = labels_df.copy()
    else:
        part = labels_df[labels_df["split"].astype(str) == str(split)].copy()
    out: Dict[str, float] = {}
    los_thr = _LOS_THRESHOLD.get(task, None)
    for _, row in part.iterrows():
        hid = str(row["hadm_id_str"])
        raw = row.get(target_col, np.nan)
        if pd.isna(raw):
            continue
        if los_thr is not None:
            out[hid] = float(1.0 if float(raw) >= float(los_thr) else 0.0)
        else:
            out[hid] = float(raw)
    return out


@dataclass
class MultiLabelTaskSpec:
    task: str
    label_col: str
    min_code_count: int
    max_codes: Optional[int]
    include_other_bucket: bool
    codes: List[str]
    code_to_idx: Dict[str, int]
    vocab_hash: str


def _build_multilabel_task_spec(config: Dict[str, Any], labels_csv: str, task: str) -> MultiLabelTaskSpec:
    pp = _require_config_section(config, ["probe", "tasks", task])
    hcup_mapping = pp.get("mapping_artifact") if task == "hcup" else None
    if task == "hcup" and not hcup_mapping:
        raise ValueError("HCUP benchmark requires probe.tasks.hcup.mapping_artifact; implicit/wide HCUP parsing is forbidden.")
    if task == "hcup":
        audit = _validate_hcup_mapping_artifact(str(hcup_mapping), config_dir=config.get("_config_dir"))
        hcup_mapping = audit["path"]
        print(f"[benchmark] HCUP mapping artifact labels=26 sha256={audit['sha256']} path={audit['path']}", flush=True)
    _, vocab = _targets_for_task(
        labels_csv,
        task,
        config.get("probe", {}),
        hcup_mapping_json=(None if hcup_mapping in (None, "") else str(hcup_mapping)),
        allowed_splits=("train",),
    )
    if vocab is None:
        raise RuntimeError(f"Task {task} did not produce a canonical vocabulary artifact.")
    codes = [str(x) for x in vocab.get("codes", [])]
    if task == "hcup" and len(codes) != 26:
        raise ValueError(f"HCUP benchmark expected exactly 26 mapping-ordered labels, observed {len(codes)}.")
    payload = json.dumps(codes, ensure_ascii=True, separators=(",", ":"))
    vocab_hash = hashlib.md5(payload.encode("ascii")).hexdigest()
    return MultiLabelTaskSpec(
        task=task,
        label_col=str(vocab.get("label_col", pp.get("label_col", f"{task}_codes"))),
        min_code_count=int(pp.get("min_code_count", 1)),
        max_codes=len(codes),
        include_other_bucket=bool(vocab.get("include_other_bucket", False)),
        codes=codes,
        code_to_idx={str(k): int(v) for k, v in dict(vocab.get("code_to_idx", {})).items()},
        vocab_hash=vocab_hash,
    )


def _scan_active_state_vids(data_dir: str, labels_df: pd.DataFrame, split: str = "train", smoke_mode: bool = False) -> set:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Expected split directory not found: {split_dir}")
    if bool(smoke_mode):
        hadm_set = set(labels_df["hadm_id"].astype(str).tolist())
    else:
        hadm_set = set(labels_df[labels_df["split"].astype(str) == str(split)]["hadm_id"].astype(str).tolist())
    active = set()
    for p in sorted(glob.glob(os.path.join(split_dir, "*.pt"))):
        hid = os.path.basename(p).replace(".pt", "")
        if hid not in hadm_set:
            continue
        shard = _torch_load_cpu(p)
        s_tokens = shard.get("state_tokens")
        if s_tokens is None:
            continue
        active.update([int(v) for v in s_tokens.tolist()])
    return active


def _build_state_only_metadata(args: argparse.Namespace, labels_df: pd.DataFrame) -> Dict[str, Any]:
    meta_path = benchmark_metadata_path(args.cache_dir)
    os.makedirs(benchmark_cache_root(args.cache_dir), exist_ok=True)
    schema_path = os.path.join(args.schema_dir, "schema_processed.csv")
    schema_sha = _sha256_file(schema_path)
    train_sig = _split_file_signature(args.data_dir, "train")
    current_cache_key = {
        "data_dir": os.path.abspath(str(args.data_dir)),
        "schema_dir": os.path.abspath(str(args.schema_dir)),
        "labels_csv": os.path.abspath(str(args.labels_csv)),
        "smoke_mode": bool(getattr(args, "smoke", False)),
        "schema_sha256": schema_sha,
        "train_data_signature": train_sig,
    }
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            cached = json.load(f)
        req = {
            "state_vid_to_dense",
            "state_vocab_size",
            "state_features_count",
            "dense_to_vid",
            "dense_model_type",
            "schema_sha256",
            "train_data_signature",
            "feature_contract_signature",
            "dataset_policy_signature",
        }
        if req.issubset(set(cached.keys())) and cached.get("schema_sha256") == schema_sha:
            return cached

    schema = load_schema_csv(schema_path)
    if "feature_category" not in schema.columns or "vid" not in schema.columns:
        raise RuntimeError("schema_processed.csv must contain feature_category and vid")

    schema_state = schema[schema["feature_category"].astype(str) == "state"].copy()
    schema_state_vids = set(schema_state["vid"].astype(int).tolist())
    schema_state["vid_int"] = schema_state["vid"].astype(int)
    model_type_map = {int(r["vid_int"]): str(r.get("model_type", "numeric")) for _, r in schema_state.iterrows()}

    active_state = _scan_active_state_vids(args.data_dir, labels_df, split="train", smoke_mode=bool(getattr(args, "smoke", False)))
    final_vids = sorted(list(active_state.intersection(schema_state_vids)))
    if not final_vids:
        raise RuntimeError("No active state VIDs found in train split.")

    state_vid_to_dense = {int(vid): i + 1 for i, vid in enumerate(final_vids)}
    dense_to_vid = [0] + final_vids
    dense_model_type = ["pad"] + [str(model_type_map.get(int(vid), "numeric")) for vid in final_vids]

    meta = {
        "state_vid_to_dense": state_vid_to_dense,
        "state_vocab_size": len(final_vids) + 1,
        "state_features_count": len(final_vids),
        "dense_to_vid": dense_to_vid,
        "dense_model_type": dense_model_type,
        "schema_sha256": schema_sha,
        "train_data_signature": train_sig,
        "cache_key": current_cache_key,
        "feature_contract_source": "current_train_split",
        "feature_contract_source_data_dir": os.path.abspath(str(args.data_dir)),
        "dataset_policy": dict(getattr(args, "_benchmark_dataset_policy", {})),
        "dataset_policy_signature": _json_sha256(dict(getattr(args, "_benchmark_dataset_policy", {}))),
    }
    meta["feature_contract_signature"] = _feature_contract_signature(meta)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _load_feature_contract(cache_dir: str) -> Dict[str, Any]:
    path = benchmark_metadata_path(str(cache_dir))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Benchmark feature contract metadata not found: {path}")
    with open(path) as f:
        meta = json.load(f)
    required = {"state_vid_to_dense", "state_features_count", "dense_to_vid", "dense_model_type", "feature_contract_signature"}
    missing = sorted(required - set(meta))
    if missing:
        raise RuntimeError(f"Invalid benchmark feature contract {path}; missing {missing}")
    return meta


def _install_feature_contract(cache_dir: str, meta: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    out = dict(meta)
    out["feature_contract_source"] = str(source)
    out["external_reused_feature_contract"] = True
    path = benchmark_metadata_path(str(cache_dir))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        if existing.get("feature_contract_signature") != out.get("feature_contract_signature"):
            raise RuntimeError(f"Existing feature contract mismatch in {cache_dir}; use a new --cache_dir.")
        return existing
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def _assert_eicu_admission_level_processed(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(str(data_dir), "prep_summary.json")
    if not os.path.isfile(path):
        raise RuntimeError("eICU benchmark requires prep_summary.json from admission-level eICU prep.")
    with open(path) as f:
        summary = json.load(f)
    if (
        summary.get("sample_unit") != "patienthealthsystemstayid"
        or summary.get("time_origin") != "hospital_admission"
        or summary.get("validation_protocol") != "hospital_admission_level_external_validation"
    ):
        raise RuntimeError("Refusing legacy eICU ICU-stay processed data; expected patienthealthsystemstayid hospital-admission-level data.")
    return summary


def _assert_external_cache_contract(cache_root: str) -> Dict[str, Any]:
    metadata_path = os.path.join(str(cache_root), "cache_bank_metadata.json")
    if not os.path.isfile(metadata_path):
        raise RuntimeError("External canonical cache metadata is missing; rebuild hospital-admission cache.")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if (
        metadata.get("sample_unit") != "patienthealthsystemstayid"
        or metadata.get("validation_protocol") != "hospital_admission_level_external_validation"
    ):
        raise RuntimeError("Refusing legacy external ICU-stay cache; expected hospital-admission-level cache.")
    return metadata


def _canonicalize_external_labels_for_benchmark(args: argparse.Namespace) -> str:
    raw = load_labels_csv(str(args.labels_csv))
    if {"label_mortality", "label_los_days"}.issubset(raw.columns):
        return str(args.labels_csv)
    if not {"hospital_mortality", "hospital_los"}.issubset(raw.columns):
        raise RuntimeError(
            "eICU benchmark labels must be canonical (label_mortality/label_los_days) "
            "or admission-level raw labels with hospital_mortality/hospital_los."
        )
    from src.external_validation import build_canonical_external_labels

    out_dir = os.path.join(str(args.output_dir), "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    canonical_path = os.path.join(out_dir, "canonical_external_labels.csv")
    audit_path = os.path.join(out_dir, "canonical_external_labels_audit.json")
    build_canonical_external_labels(str(args.labels_csv), canonical_path, audit_path)
    canonical = load_labels_csv(canonical_path)
    required = {"subject_id", "hadm_id", "split", "label_mortality", "label_los_days"}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise RuntimeError(f"Canonical external labels are missing required columns: {missing}")
    args._canonical_external_labels_csv = canonical_path
    return canonical_path


def _summarize_events_torch(
    tokens: torch.Tensor,
    values: torch.Tensor,
    times: torch.Tensor,
    vid_to_col: torch.Tensor,
    num_features: int,
    cutoff_min: Optional[float],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    tokens = tokens.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
    values = values.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
    times = times.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
    mapping = vid_to_col.to(device=device, non_blocking=True)
    in_range = (tokens >= 0) & (tokens < mapping.numel())
    cols = torch.full_like(tokens, -1)
    cols[in_range] = mapping[tokens[in_range]]
    valid = in_range & (cols >= 0)
    if cutoff_min is not None:
        valid &= times <= float(cutoff_min)
    cols = cols[valid]
    vals = values[valid]
    ts = times[valid]
    d = int(num_features)
    zero = torch.zeros((d,), dtype=torch.float32, device=device)
    if cols.numel() == 0:
        return {s: zero.cpu().numpy().astype(np.float32, copy=False) for s in TABULAR_SUMMARIES}

    count = torch.zeros((d,), dtype=torch.float32, device=device)
    count.scatter_add_(0, cols, torch.ones_like(vals))
    total = torch.zeros((d,), dtype=torch.float32, device=device)
    total.scatter_add_(0, cols, vals)
    mean = torch.divide(total, count.clamp_min(1.0))
    presence = (count > 0).to(torch.float32)
    maxv = torch.full((d,), -torch.inf, dtype=torch.float32, device=device)
    maxv.scatter_reduce_(0, cols, vals, reduce="amax", include_self=True)
    maxv = torch.where(torch.isfinite(maxv), maxv, zero)

    order = torch.arange(cols.numel(), dtype=torch.long, device=device)
    first_ord = torch.full((d,), cols.numel(), dtype=torch.long, device=device)
    last_ord = torch.full((d,), -1, dtype=torch.long, device=device)
    first_ord.scatter_reduce_(0, cols, order, reduce="amin", include_self=True)
    last_ord.scatter_reduce_(0, cols, order, reduce="amax", include_self=True)
    seen = last_ord >= 0
    last = torch.zeros((d,), dtype=torch.float32, device=device)
    first_t = torch.zeros((d,), dtype=torch.float32, device=device)
    last_t = torch.zeros((d,), dtype=torch.float32, device=device)
    if bool(seen.any()):
        last[seen] = vals[last_ord[seen]]
        first_t[seen] = ts[first_ord[seen]]
        last_t[seen] = ts[last_ord[seen]]
    interval = torch.zeros((d,), dtype=torch.float32, device=device)
    has_interval = count > 1
    interval[has_interval] = (last_t[has_interval] - first_t[has_interval]) / (count[has_interval] - 1.0)
    out = {
        "last": last,
        "mean": mean,
        "max": maxv,
        "presence": presence,
        "count": count,
        "interval": interval,
    }
    return {k: v.detach().cpu().numpy().astype(np.float32, copy=False) for k, v in out.items()}


def _summarize_dense_batch_torch(
    state_tokens: torch.Tensor,
    state_values: torch.Tensor,
    state_times: torch.Tensor,
    state_pad_mask: torch.Tensor,
    seq_pad_mask: torch.Tensor,
    vid_to_col: torch.Tensor,
    num_features: int,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    tokens = state_tokens.to(device=device, dtype=torch.long, non_blocking=True)
    values = state_values[..., 0].to(device=device, dtype=torch.float32, non_blocking=True)
    times = state_times[..., 0].to(device=device, dtype=torch.float32, non_blocking=True)
    event_pad = state_pad_mask.to(device=device, non_blocking=True)
    seq_pad = seq_pad_mask.to(device=device, non_blocking=True)
    mapping = vid_to_col.to(device=device, non_blocking=True)
    bsz, t_len, s_len = tokens.shape
    d = int(num_features)
    flat_b = torch.arange(bsz, device=device).view(bsz, 1, 1).expand_as(tokens).reshape(-1)
    flat_t = torch.arange(t_len, device=device).view(1, t_len, 1).expand_as(tokens).reshape(-1)
    flat_s = torch.arange(s_len, device=device).view(1, 1, s_len).expand_as(tokens).reshape(-1)
    flat_tokens = tokens.reshape(-1)
    flat_values = values.reshape(-1)
    flat_times = times.unsqueeze(-1).expand_as(tokens).reshape(-1)
    flat_event_valid = (~event_pad).reshape(-1) & (flat_tokens > 0) & torch.isfinite(flat_times)
    in_range = (flat_tokens >= 0) & (flat_tokens < mapping.numel())
    cols = torch.full_like(flat_tokens, -1)
    cols[in_range] = mapping[flat_tokens[in_range]]
    mapped_valid_base = flat_event_valid & in_range & (cols >= 0)
    raw_out: Dict[str, np.ndarray] = {}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for window, cutoff in FEATURE_WINDOWS.items():
        raw_valid = flat_event_valid
        if cutoff is not None:
            raw_valid = raw_valid & (flat_times <= float(cutoff))
        raw_event = torch.zeros((bsz,), dtype=torch.float32, device=device)
        raw_event.scatter_add_(0, flat_b[raw_valid], torch.ones_like(flat_values[raw_valid]))
        ts_valid = (~seq_pad) & torch.isfinite(times)
        if cutoff is not None:
            ts_valid = ts_valid & (times <= float(cutoff))
        raw_ts = ts_valid.to(torch.float32).sum(dim=1)
        raw_out[f"{window}__raw_state_event_count"] = raw_event.detach().cpu().numpy().astype(np.int64)
        raw_out[f"{window}__raw_timestamp_count"] = raw_ts.detach().cpu().numpy().astype(np.int64)

        valid = mapped_valid_base
        if cutoff is not None:
            valid = valid & (flat_times <= float(cutoff))
        global_col = flat_b[valid] * d + cols[valid]
        vals = flat_values[valid]
        ts = flat_times[valid]
        n_total = bsz * d
        zero = torch.zeros((n_total,), dtype=torch.float32, device=device)
        if vals.numel() == 0:
            out[window] = {s: zero.view(bsz, d).detach().cpu().numpy().astype(np.float32) for s in TABULAR_SUMMARIES}
            continue
        count = torch.zeros((n_total,), dtype=torch.float32, device=device)
        count.scatter_add_(0, global_col, torch.ones_like(vals))
        total = torch.zeros((n_total,), dtype=torch.float32, device=device)
        total.scatter_add_(0, global_col, vals)
        mean = total / count.clamp_min(1.0)
        presence = (count > 0).to(torch.float32)
        maxv = torch.full((n_total,), -torch.inf, dtype=torch.float32, device=device)
        maxv.scatter_reduce_(0, global_col, vals, reduce="amax", include_self=True)
        maxv = torch.where(torch.isfinite(maxv), maxv, zero)

        order_score = ts * float(s_len + 1) + flat_s[valid].to(torch.float32)
        first_score = torch.full((n_total,), torch.inf, dtype=torch.float32, device=device)
        last_score = torch.full((n_total,), -torch.inf, dtype=torch.float32, device=device)
        first_score.scatter_reduce_(0, global_col, order_score, reduce="amin", include_self=True)
        last_score.scatter_reduce_(0, global_col, order_score, reduce="amax", include_self=True)
        seen = count > 0
        last = torch.zeros((n_total,), dtype=torch.float32, device=device)
        first_t = torch.zeros((n_total,), dtype=torch.float32, device=device)
        last_t = torch.zeros((n_total,), dtype=torch.float32, device=device)
        if bool(seen.any()):
            first_idx = torch.full((n_total,), vals.numel(), dtype=torch.long, device=device)
            last_idx = torch.full((n_total,), -1, dtype=torch.long, device=device)
            ar = torch.arange(vals.numel(), dtype=torch.long, device=device)
            first_idx.scatter_reduce_(0, global_col, ar, reduce="amin", include_self=True)
            last_idx.scatter_reduce_(0, global_col, ar, reduce="amax", include_self=True)
            last[seen] = vals[last_idx[seen]]
            first_t[seen] = ts[first_idx[seen]]
            last_t[seen] = ts[last_idx[seen]]
        interval = torch.zeros((n_total,), dtype=torch.float32, device=device)
        ok = count > 1
        interval[ok] = (last_t[ok] - first_t[ok]) / (count[ok] - 1.0)
        out[window] = {
            "last": last.view(bsz, d).detach().cpu().numpy().astype(np.float32),
            "mean": mean.view(bsz, d).detach().cpu().numpy().astype(np.float32),
            "max": maxv.view(bsz, d).detach().cpu().numpy().astype(np.float32),
            "presence": presence.view(bsz, d).detach().cpu().numpy().astype(np.float32),
            "count": count.view(bsz, d).detach().cpu().numpy().astype(np.float32),
            "interval": interval.view(bsz, d).detach().cpu().numpy().astype(np.float32),
        }
    return out, raw_out


def _summarize_events_reference(
    tokens: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    state_vid_to_dense: Mapping[int, int],
    num_features: int,
    cutoff_min: Optional[float],
) -> Dict[str, np.ndarray]:
    tokens = np.asarray(tokens).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    times = np.asarray(times, dtype=np.float32).reshape(-1)
    last = np.zeros((num_features,), dtype=np.float32)
    total = np.zeros((num_features,), dtype=np.float32)
    maxv = np.full((num_features,), -np.inf, dtype=np.float32)
    count = np.zeros((num_features,), dtype=np.float32)
    first_t = np.zeros((num_features,), dtype=np.float32)
    last_t = np.zeros((num_features,), dtype=np.float32)
    for vid, val, t in zip(tokens.tolist(), values.tolist(), times.tolist()):
        col = int(state_vid_to_dense.get(int(vid), 0)) - 1
        if col < 0:
            continue
        if cutoff_min is not None and float(t) > float(cutoff_min):
            continue
        if count[col] == 0:
            first_t[col] = float(t)
        last_t[col] = float(t)
        last[col] = float(val)
        total[col] += float(val)
        count[col] += 1.0
        maxv[col] = max(float(maxv[col]), float(val))
    presence = (count > 0).astype(np.float32)
    mean = np.divide(total, np.maximum(count, 1.0)).astype(np.float32)
    maxv = np.where(np.isfinite(maxv), maxv, 0.0).astype(np.float32)
    interval = np.zeros((num_features,), dtype=np.float32)
    ok = count > 1
    interval[ok] = (last_t[ok] - first_t[ok]) / (count[ok] - 1.0)
    return {"last": last, "mean": mean, "max": maxv, "presence": presence, "count": count, "interval": interval}




def _build_vid_to_col_tensor(state_vid_to_dense: Mapping[int, int]) -> torch.Tensor:
    max_vid = max([0] + [int(v) for v in state_vid_to_dense.keys()])
    out = torch.full((max_vid + 1,), -1, dtype=torch.long)
    for vid, dense in state_vid_to_dense.items():
        if int(vid) >= 0:
            out[int(vid)] = int(dense) - 1
    return out


def _sha256_id_list(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for hid in ids:
        h.update(str(hid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _is_ordered_subset(candidate: Sequence[str], canonical: Sequence[str]) -> bool:
    pos = -1
    canonical_pos = {str(hid): i for i, hid in enumerate(canonical)}
    for hid in map(str, candidate):
        i = canonical_pos.get(hid)
        if i is None or i <= pos:
            return False
        pos = i
    return True


def _build_smoke_ids_from_canonical(
    args: argparse.Namespace,
    tasks: Sequence[str],
    smoke_n: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
    root = _canonical_cache_root(args)
    if root is None:
        return {}, {}
    by_task_split: Dict[str, Dict[str, List[str]]] = {}
    by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    seen_by_split: Dict[str, set] = {split: set() for split in by_split}
    n = max(1, int(smoke_n))
    for task in tasks:
        by_task_split[str(task)] = {}
        for split in ("train", "val", "test"):
            try:
                ids = _canonical_cache_ids(root, split, str(task))[:n]
            except RuntimeError as exc:
                print(
                    "[benchmark smoke] canonical cache IDs unavailable; "
                    f"falling back to labels-defined smoke cohort. root={os.path.abspath(root)} "
                    f"task={task} split={split} error={exc}",
                    flush=True,
                )
                return {}, {}
            by_task_split[str(task)][split] = list(ids)
            for hid in ids:
                s = str(hid)
                if s not in seen_by_split[split]:
                    by_split[split].append(s)
                    seen_by_split[split].add(s)
    return by_task_split, by_split


def _build_feature_bank_split(args: argparse.Namespace, labels_df: pd.DataFrame, split: str, meta: Dict[str, Any]) -> None:
    state_vid_to_dense = {int(k): int(v) for k, v in meta["state_vid_to_dense"].items()}
    num_features = int(meta["state_features_count"])
    out_dir = _feature_bank_split_dir(args.cache_dir, split)
    tmp_dir = f"{out_dir}.tmp_{os.getpid()}"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    bench_cfg = getattr(args, "_benchmark_cfg", {}) or {}
    shard_size = int(((bench_cfg.get("runtime", {}) or {}).get("shard_size", 256)))
    batch_size = int(((bench_cfg.get("runtime", {}) or {}).get("feature_batch_size", min(64, shard_size))))
    max_logical_gb = float(((bench_cfg.get("runtime", {}) or {}).get("max_feature_bank_logical_gb", 512.0)))
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    vid_to_col = _build_vid_to_col_tensor(state_vid_to_dense)
    smoke_ids_by_split = getattr(args, "_benchmark_smoke_ids_by_split", {}) or {}
    if bool(getattr(args, "smoke", False)) and split in smoke_ids_by_split:
        target_ids = [str(x) for x in smoke_ids_by_split[str(split)]]
    else:
        label_ids = set(labels_df.loc[labels_df["split"].astype(str) == str(split), "hadm_id"].astype(str).tolist())
        split_files = sorted(glob.glob(os.path.join(str(args.data_dir), str(split), "*.pt")))
        target_ids = [
            os.path.basename(fp).replace(".pt", "")
            for fp in split_files
            if os.path.basename(fp).replace(".pt", "") in label_ids
        ]
        if not target_ids:
            raise RuntimeError(f"Benchmark found no label-matched processed admissions for split={split}.")
    if bool(getattr(args, "smoke", False)) and split not in smoke_ids_by_split:
        smoke_n = int(((bench_cfg.get("runtime", {}) or {}).get("smoke_max_patients_per_split", 32)))
        target_ids = target_ids[:max(1, smoke_n)]
    ds_kwargs = dict(getattr(args, "_benchmark_dataset_policy", {}))
    ds_kwargs.update({
        "data_dir": args.data_dir,
        "schema_dir": args.schema_dir,
        "split": split,
        "target_ids": target_ids,
        "target_task": "mortality",
    })
    ds = PatientSequenceDataset(**ds_kwargs)
    loader = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        collate_fn=nested_padding_collate,
    )
    logical_bytes = int(len(ds)) * len(FEATURE_WINDOWS) * len(TABULAR_SUMMARIES) * int(num_features) * 4
    logical_gb = logical_bytes / float(1024 ** 3)
    if logical_gb > max_logical_gb:
        raise RuntimeError(f"Projected benchmark feature bank logical size {logical_gb:.2f} GiB exceeds budget {max_logical_gb:.2f} GiB.")
    print(
        f"[benchmark_feature_bank] split={split} rows={len(ds)} D={num_features} "
        f"logical_size_gib={logical_gb:.3f} storage=npy_memmap_shards",
        flush=True,
    )

    buffers: Dict[str, Dict[str, List[np.ndarray]]] = {w: {s: [] for s in TABULAR_SUMMARIES} for w in FEATURE_WINDOWS}
    raw_event_buffers: Dict[str, List[int]] = {w: [] for w in FEATURE_WINDOWS}
    raw_timestamp_buffers: Dict[str, List[int]] = {w: [] for w in FEATURE_WINDOWS}
    buf_hadm: List[str] = []
    shards_meta: List[Dict[str, Any]] = []
    rows_total = 0
    shard_idx = 0
    t0 = time.time()
    peak_gpu = 0

    def _flush() -> None:
        nonlocal rows_total, shard_idx, peak_gpu
        if not buf_hadm:
            return
        payload: Dict[str, Any] = {
            "hadm_id": np.asarray(buf_hadm, dtype=str),
            "n": np.asarray(len(buf_hadm), dtype=np.int64),
        }
        for w in FEATURE_WINDOWS:
            for s in TABULAR_SUMMARIES:
                payload[f"{w}__{s}"] = np.concatenate(buffers[w][s], axis=0).astype(np.float32, copy=False)
            payload[f"{w}__raw_state_event_count"] = np.concatenate([np.asarray(x, dtype=np.int64).reshape(-1) for x in raw_event_buffers[w]], axis=0)
            payload[f"{w}__raw_timestamp_count"] = np.concatenate([np.asarray(x, dtype=np.int64).reshape(-1) for x in raw_timestamp_buffers[w]], axis=0)
        shard_name = f"features_{shard_idx:05d}"
        shard_dir = os.path.join(tmp_dir, shard_name)
        os.makedirs(shard_dir, exist_ok=True)
        for key, value in payload.items():
            np.save(os.path.join(shard_dir, f"{key}.npy"), value, allow_pickle=False)
        shards_meta.append({"dir": shard_name, "n": int(len(buf_hadm)), "format": "npy_memmap"})
        rows_total += int(len(buf_hadm))
        shard_idx += 1
        buf_hadm.clear()
        for w in FEATURE_WINDOWS:
            for s in TABULAR_SUMMARIES:
                buffers[w][s].clear()
            raw_event_buffers[w].clear()
            raw_timestamp_buffers[w].clear()
        if torch.cuda.is_available():
            peak_gpu = max(peak_gpu, int(torch.cuda.max_memory_allocated()))

    for batch in tqdm(loader, desc=f"benchmark_feature_bank/{split}", ascii=True):
        hadm_ids = [str(x) for x in batch["hadm_id"]]
        batch_feats, batch_raw = _summarize_dense_batch_torch(
            batch["state_tokens"],
            batch["state_values"],
            batch["state_times"],
            batch["state_pad_mask"],
            batch["seq_pad_mask"],
            vid_to_col,
            num_features,
            device,
        )
        buf_hadm.extend(hadm_ids)
        for window in FEATURE_WINDOWS:
            raw_event_buffers[window].append(np.asarray(batch_raw[f"{window}__raw_state_event_count"], dtype=np.int64))
            raw_timestamp_buffers[window].append(np.asarray(batch_raw[f"{window}__raw_timestamp_count"], dtype=np.int64))
            for summary in TABULAR_SUMMARIES:
                buffers[window][summary].append(np.asarray(batch_feats[window][summary], dtype=np.float32))
        if len(buf_hadm) >= shard_size:
            _flush()
    _flush()
    if rows_total <= 0:
        raise RuntimeError(f"No benchmark feature rows built for split={split}")
    index = {
        "version": FEATURE_BANK_VERSION,
        "split": str(split),
        "num_features": int(num_features),
        "windows": list(FEATURE_WINDOWS.keys()),
        "summaries": list(TABULAR_SUMMARIES),
        "feature_contract_signature": str(meta["feature_contract_signature"]),
        "schema_sha256": str(meta["schema_sha256"]),
        "labels_sha256": _sha256_file(str(args.labels_csv)),
        "split_data_signature": _split_file_signature(args.data_dir, split),
        "dataset_policy_signature": str(meta.get("dataset_policy_signature")),
        "summary_definitions": {
            "last": "last value by timestamp-major order after dataset set-size policy",
            "mean": "mean of mapped event values",
            "max": "max of mapped event values",
            "presence": "1 if mapped VID appears in window",
            "count": "mapped event count",
            "interval": "(last timestamp - first timestamp) / (count - 1)",
        },
        "projected_logical_size_gib": float(logical_gb),
        "storage_format": "npy_memmap_shards",
        "build_device": str(device),
        "build_sec": float(time.time() - t0),
        "peak_gpu_memory_bytes": int(peak_gpu),
        "shards": shards_meta,
        "num_rows": int(rows_total),
    }
    with open(os.path.join(tmp_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(tmp_dir, ".complete"), "w") as f:
        f.write("complete\n")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)


def _ensure_state_only_cache(args: argparse.Namespace, labels_df: pd.DataFrame, splits: Optional[Sequence[str]] = None) -> None:
    meta = _build_state_only_metadata(args, labels_df)
    for split in (list(splits) if splits is not None else ["train", "val", "test"]):
        split_dir = _feature_bank_split_dir(args.cache_dir, split)
        idx_path = os.path.join(split_dir, "index.json")
        complete_path = os.path.join(split_dir, ".complete")
        if not os.path.exists(idx_path) or not os.path.exists(complete_path):
            _build_feature_bank_split(args, labels_df, split, meta)
            continue
        with open(idx_path, "r") as f:
            idx_blob = json.load(f)
        if idx_blob.get("version") != FEATURE_BANK_VERSION:
            raise RuntimeError(f"Benchmark feature bank split '{split}' version mismatch; use a new --cache_dir.")
        if idx_blob.get("feature_contract_signature") != meta.get("feature_contract_signature"):
            raise RuntimeError(f"Benchmark feature bank split '{split}' feature contract mismatch; use a new --cache_dir.")
        if idx_blob.get("dataset_policy_signature") != meta.get("dataset_policy_signature"):
            raise RuntimeError(f"Benchmark feature bank split '{split}' dataset policy mismatch; use a new --cache_dir.")
        if idx_blob.get("split_data_signature") != _split_file_signature(args.data_dir, split):
            raise RuntimeError(f"Benchmark feature bank split '{split}' data signature mismatch; use a new --cache_dir.")
        if idx_blob.get("labels_sha256") != _sha256_file(str(args.labels_csv)):
            raise RuntimeError(f"Benchmark feature bank split '{split}' labels signature mismatch; use a new --cache_dir.")


class SequenceClassificationDataset:
    def __init__(self, cache_dir: str, split: str, task: str, labels_df: pd.DataFrame, config: Dict[str, Any], multilabel_spec: Optional[MultiLabelTaskSpec] = None, cutoff_min: Optional[float] = None):
        self.split = str(split)
        self.task = str(task)
        self.config = config
        self._cutoff_min = cutoff_min
        self.smoke_mode = bool(((config or {}).get("runtime", {}) or {}).get("smoke", False))
        self.multilabel_spec = multilabel_spec
        self._shared_root = _feature_bank_split_dir(cache_dir, split)
        self._payloads: Dict[int, Dict[str, Any]] = {}
        self._last_payload_sid: Optional[int] = None

        with open(os.path.join(self._shared_root, "index.json"), "r") as f:
            idx_blob = json.load(f)
        self._num_features = int(idx_blob["num_features"])
        self._shards = list(idx_blob["shards"])
        self._row_offsets = np.zeros((len(self._shards) + 1,), dtype=np.int64)
        for i, sh in enumerate(self._shards):
            self._row_offsets[i + 1] = self._row_offsets[i] + int(sh.get("n", 0))
        self._row_offsets_list = [int(x) for x in self._row_offsets.tolist()]
        self._n_rows = int(self._row_offsets[-1])

        if self.task in SEQUENCE_BINARY_TASKS:
            self._label_map = _binary_label_map(labels_df, split=self.split, task=self.task, smoke_mode=self.smoke_mode)
            self._is_multilabel = False
        elif self.task in SEQUENCE_MULTILABEL_TASKS:
            if self.multilabel_spec is None:
                raise ValueError("multilabel_spec is required for icd/drg")
            probe_cfg = dict((config or {}).get("probe", {}))
            task_cfg = dict(probe_cfg.get("tasks", {}).get(self.task, {}))
            hcup_mapping = task_cfg.get("mapping_artifact") if self.task == "hcup" else None
            if self.task == "hcup" and not hcup_mapping:
                raise ValueError("HCUP benchmark requires probe.tasks.hcup.mapping_artifact; implicit/wide HCUP parsing is forbidden.")
            targets, vocab = _targets_for_task(
                config["runtime"]["labels_csv"],
                self.task,
                probe_cfg,
                vocab_override={
                    "label_col": self.multilabel_spec.label_col,
                    "codes": self.multilabel_spec.codes,
                    "code_to_idx": self.multilabel_spec.code_to_idx,
                    "include_other_bucket": self.multilabel_spec.include_other_bucket,
                },
                hcup_mapping_json=(None if hcup_mapping in (None, "") else str(hcup_mapping)),
                allowed_splits=(None if self.smoke_mode else (self.split,)),
            )
            label_map = {str(hid): np.asarray(target, dtype=np.float32) for hid, target in targets.items()}
            self._label_map = label_map
            self._is_multilabel = True
        else:
            raise ValueError(f"Unsupported task: {task}")

        self._all_n_rows = int(self._n_rows)
        self._valid_global_rows, self._admission_ids = self._compute_valid_rows()
        smoke_ids = (
            ((config or {}).get("runtime", {}) or {})
            .get("smoke_ids_by_task_split", {})
            .get(self.task, {})
            .get(self.split)
        )
        if smoke_ids is not None:
            row_by_id = {str(hid): int(row) for hid, row in zip(self._admission_ids, self._valid_global_rows)}
            missing = [str(hid) for hid in smoke_ids if str(hid) not in row_by_id]
            if missing:
                raise RuntimeError(
                    f"Smoke benchmark IDs missing from feature bank for task={self.task} split={self.split}: {missing[:10]}"
                )
            self._admission_ids = [str(hid) for hid in smoke_ids]
            self._valid_global_rows = [row_by_id[str(hid)] for hid in self._admission_ids]
        self._n_rows = int(len(self._valid_global_rows))

    def __len__(self) -> int:
        return int(self._n_rows)

    def _resolve_global_row(self, idx: int) -> Tuple[int, int]:
        i = int(self._valid_global_rows[int(idx)])
        sidx = int(bisect.bisect_right(self._row_offsets_list, i) - 1)
        ridx = int(i - int(self._row_offsets[sidx]))
        return sidx, ridx

    def _window_tag(self) -> str:
        if self._cutoff_min is None:
            return "full_visible"
        return f"window_{int(round(float(self._cutoff_min)))}"

    def _load_shard_payload(self, shard_idx: int) -> Dict[str, Any]:
        sid = int(shard_idx)
        if sid not in self._payloads:
            shard = self._shards[sid]
            if "dir" in shard:
                shard_dir = os.path.join(self._shared_root, str(shard["dir"]))
                payload = {
                    os.path.splitext(name)[0]: np.load(os.path.join(shard_dir, name), mmap_mode="r", allow_pickle=False)
                    for name in os.listdir(shard_dir)
                    if name.endswith(".npy")
                }
            else:
                p = os.path.join(self._shared_root, str(shard["file"]))
                if p.endswith(".npz"):
                    with np.load(p, allow_pickle=False) as z:
                        payload = {k: z[k] for k in z.files}
                else:
                    payload = _torch_load_cpu(p)
            if self._last_payload_sid is not None and self._last_payload_sid in self._payloads:
                del self._payloads[self._last_payload_sid]
            self._payloads[sid] = payload
            self._last_payload_sid = sid
        return self._payloads[sid]

    def _compute_valid_rows(self) -> Tuple[List[int], List[str]]:
        rows: List[int] = []
        ids: List[str] = []
        pos = 0
        for sidx, sh in enumerate(self._shards):
            payload = self._load_shard_payload(sidx)
            hids = [str(x) for x in payload["hadm_id"].tolist()]
            n_rows = int(sh.get("n", len(hids)))
            raw_count_key = f"{self._window_tag()}__raw_state_event_count"
            raw_count = payload.get(raw_count_key)
            for ridx in range(n_rows):
                hid = str(hids[ridx])
                if hid not in self._label_map and not self.smoke_mode:
                    pos += 1
                    continue
                if raw_count is not None and int(np.asarray(raw_count[ridx]).item()) <= 0:
                    pos += 1
                    continue
                rows.append(int(pos))
                ids.append(hid)
                pos += 1
        return rows, ids

    def admission_ids(self) -> List[str]:
        return list(self._admission_ids)

    def _summary_xm_label(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        sidx, ridx = self._resolve_global_row(idx)
        payload = self._load_shard_payload(sidx)
        hadm = str(payload["hadm_id"][ridx].item() if hasattr(payload["hadm_id"][ridx], "item") else payload["hadm_id"][ridx])
        if str(hadm) in self._label_map:
            label = self._label_map[str(hadm)]
        elif self.smoke_mode:
            if self.task in SEQUENCE_BINARY_TASKS:
                label = 0.0
            else:
                label = np.zeros((len(self.multilabel_spec.codes),), dtype=np.float32)
        else:
            raise KeyError(str(hadm))

        w = self._window_tag()
        return (
            np.asarray(payload[f"{w}__last"][ridx], dtype=np.float32).reshape(1, -1),
            np.asarray(payload[f"{w}__mean"][ridx], dtype=np.float32).reshape(1, -1),
            np.asarray(payload[f"{w}__max"][ridx], dtype=np.float32).reshape(1, -1),
            np.asarray(payload[f"{w}__presence"][ridx], dtype=np.float32),
            np.asarray(payload[f"{w}__count"][ridx], dtype=np.float32),
            np.asarray(payload[f"{w}__interval"][ridx], dtype=np.float32),
            label,
        )

    def feature_matrix(self, summary: str) -> np.ndarray:
        if summary not in TABULAR_SUMMARIES:
            raise KeyError(summary)
        key = f"{self._window_tag()}__{summary}"
        parts: List[np.ndarray] = []
        for global_row in self._valid_global_rows:
            sidx = int(bisect.bisect_right(self._row_offsets_list, global_row) - 1)
            ridx = int(global_row - int(self._row_offsets[sidx]))
            payload = self._load_shard_payload(sidx)
            parts.append(np.asarray(payload[key][ridx], dtype=np.float32))
        if not parts:
            return np.zeros((0, int(self._num_features)), dtype=np.float32)
        return np.stack(parts, axis=0).astype(np.float32, copy=False)

    def labels_numpy(self) -> np.ndarray:
        vals = []
        for hid in self._admission_ids:
            if hid in self._label_map:
                vals.append(self._label_map[hid])
            elif self.smoke_mode:
                vals.append(0.0 if self.task in SEQUENCE_BINARY_TASKS else np.zeros((len(self.multilabel_spec.codes),), dtype=np.float32))
            else:
                raise KeyError(hid)
        if vals and isinstance(vals[0], np.ndarray):
            return np.stack([np.asarray(v, dtype=np.float32) for v in vals], axis=0)
        return np.asarray(vals, dtype=np.float32)




def build_tabular_feature_summary(
    dataset: SequenceClassificationDataset,
    num_features: int,
    cache_dir: str,
    split: str,
    task: str,
    state_vid_to_dense: Dict[int, int],
    cutoff_min: Optional[float],
    summary: str,
    *,
    max_matrix_gib: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    t0 = time.time()
    if summary not in TABULAR_SUMMARIES:
        raise KeyError(summary)
    projected_gib = (int(len(dataset)) * int(num_features) * 4) / float(1024 ** 3)
    if max_matrix_gib is not None and projected_gib > float(max_matrix_gib):
        raise RuntimeError(
            f"Projected benchmark training matrix {split}/{task}/{summary} is {projected_gib:.2f} GiB, "
            f"exceeding budget {float(max_matrix_gib):.2f} GiB."
        )
    y = dataset.labels_numpy().astype(np.float32, copy=False)
    x = dataset.feature_matrix(summary)
    if x.shape != (len(dataset), int(num_features)):
        raise RuntimeError(f"Feature bank matrix shape mismatch for {split}/{task}/{summary}: {x.shape}")
    return x, y, {"cache_sec": float(time.time() - t0), "feature_build_sec": 0.0, "projected_matrix_gib": float(projected_gib)}


def _warm_benchmark_tabular_cache(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    labels_df: pd.DataFrame,
    tasks: Sequence[str],
    cutoff_map: Dict[str, List[Optional[float]]],
    state_vid_to_dense: Dict[int, int],
    num_features: int,
) -> None:
    runtime_cfg = {
        "runtime": {
            "labels_csv": str(args.labels_csv),
            "smoke": bool(args.smoke),
            "smoke_ids_by_task_split": getattr(args, "_benchmark_smoke_ids_by_task_split", {}) or {},
        },
        "probe": config.get("probe", {}),
    }
    for task in tasks:
        ml_spec = _build_multilabel_task_spec(config, str(args.labels_csv), task) if task in SEQUENCE_MULTILABEL_TASKS else None
        for cutoff in cutoff_map[task]:
            tr_ds = SequenceClassificationDataset(args.cache_dir, "train", task, labels_df, runtime_cfg, ml_spec, cutoff_min=cutoff)
            va_ds = SequenceClassificationDataset(args.cache_dir, "val", task, labels_df, runtime_cfg, ml_spec, cutoff_min=cutoff)
            for summary in _select_summaries(task, getattr(args, "summaries", None)):
                build_tabular_feature_summary(tr_ds, num_features, args.cache_dir, "train", task, state_vid_to_dense, cutoff, summary)
                build_tabular_feature_summary(va_ds, num_features, args.cache_dir, "val", task, state_vid_to_dense, cutoff, summary)


def _f1_micro_macro_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = (np.asarray(y_prob, dtype=np.float32) >= float(threshold)).astype(np.float32)
    tp_micro = float((yt * yp).sum())
    fp_micro = float(((1.0 - yt) * yp).sum())
    fn_micro = float((yt * (1.0 - yp)).sum())
    p_micro = tp_micro / max(tp_micro + fp_micro, 1e-8)
    r_micro = tp_micro / max(tp_micro + fn_micro, 1e-8)
    f1_micro = (2.0 * p_micro * r_micro) / max(p_micro + r_micro, 1e-8)
    tp = (yt * yp).sum(axis=0)
    fp = ((1.0 - yt) * yp).sum(axis=0)
    fn = (yt * (1.0 - yp)).sum(axis=0)
    p = tp / np.maximum(tp + fp, 1e-8)
    r = tp / np.maximum(tp + fn, 1e-8)
    f1 = (2.0 * p * r) / np.maximum(p + r, 1e-8)
    valid = (yt.sum(axis=0) > 0)
    f1_macro = float(np.nanmean(f1[valid])) if np.any(valid) else float("nan")
    return {"f1_micro@0.5": float(f1_micro), "f1_macro@0.5": float(f1_macro)}


def _augment_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    base = compute_multilabel_metrics(y_true=y_true, y_prob=y_prob, ks=[5, 10], threshold=0.5)
    f1 = _f1_micro_macro_at_threshold(y_true, y_prob, threshold=0.5)
    pred = (y_prob >= 0.5).astype(np.float32)
    out = {
        "micro_auprc": float(base.get("micro_auprc", np.nan)),
        "macro_auprc": float(base.get("macro_auprc", np.nan)),
        "micro_auroc": float(base.get("micro_auroc", np.nan)),
        "macro_auroc": float(base.get("macro_auroc", np.nan)),
        "precision_at_5": float(base.get("precision_at_5", np.nan)),
        "recall_at_5": float(base.get("recall_at_5", np.nan)),
        "precision_at_10": float(base.get("precision_at_10", np.nan)),
        "recall_at_10": float(base.get("recall_at_10", np.nan)),
        "f1_micro@0.5": float(f1["f1_micro@0.5"]),
        "f1_macro@0.5": float(f1["f1_macro@0.5"]),
        "predicted_label_count_mean": float(pred.sum(axis=1).mean()),
        "predicted_label_count_std": float(pred.sum(axis=1).std()),
        "true_label_count_mean": float(y_true.sum(axis=1).mean()),
        "true_label_count_std": float(y_true.sum(axis=1).std()),
        "positive_rate_pred": float(pred.mean()),
        "positive_rate_true": float(y_true.mean()),
    }
    return out


def _val_score_for_multilabel(task_name: Optional[str], metrics: Dict[str, float]) -> float:
    t = str(task_name or "").lower()
    if t == "drg":
        return float(metrics.get("f1_micro@0.5", 0.0))
    if t == "hcup":
        return float(metrics.get("macro_auprc", 0.0))
    return float(metrics.get("micro_auprc", 0.0))


def _standardize_train_val_test(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mean = X_tr.astype(np.float32, copy=False).mean(axis=0, keepdims=True)
    std = X_tr.astype(np.float32, copy=False).std(axis=0, keepdims=True)
    std = np.where(std > 1.0e-6, std, 1.0).astype(np.float32)
    return (
        ((X_tr.astype(np.float32, copy=False) - mean) / std).astype(np.float32, copy=False),
        ((X_va.astype(np.float32, copy=False) - mean) / std).astype(np.float32, copy=False),
        ((X_te.astype(np.float32, copy=False) - mean) / std).astype(np.float32, copy=False),
        {"mean": mean.squeeze(0).astype(np.float32), "std": std.squeeze(0).astype(np.float32)},
    )


def _linear_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _linear_candidate_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    wd = cfg.get("weight_decay_candidates", cfg.get("l2_candidates", [float(cfg.get("weight_decay", 0.0))]))
    cw = cfg.get("class_weight_candidates", ["none", "balanced"])
    out = []
    for w in wd:
        for c in cw:
            out.append({"weight_decay": float(w), "class_weight": str(c)})
    return out


def _compute_full_objective_gradient_norm(
    *,
    model_type: str,
    models: Sequence[nn.Module],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    card_train: Optional[torch.Tensor],
    pos_weight: Optional[torch.Tensor],
    batch_size: int,
) -> float:
    if not models:
        raise ValueError("models must be non-empty")
    n = int(x_train.shape[0])
    if n <= 0:
        raise ValueError("x_train must contain at least one row")
    was_training = [bool(m.training) for m in models]
    for m in models:
        m.train()
        m.zero_grad(set_to_none=True)
    bs = max(1, int(batch_size))
    with torch.enable_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            xb = x_train[s:e]
            weight = float(e - s) / float(n)
            if str(model_type) == "ordinary":
                logits = models[0](xb)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    y_train[s:e],
                    pos_weight=pos_weight,
                )
            elif str(model_type) == "drg":
                if len(models) != 2 or card_train is None:
                    raise ValueError("DRG full-gradient requires label and cardinality models plus card_train")
                label_logits = models[0](xb)
                card_logits = models[1](xb)
                loss = F.binary_cross_entropy_with_logits(
                    label_logits,
                    y_train[s:e],
                    pos_weight=pos_weight,
                )
                loss = loss + F.cross_entropy(card_logits, card_train[s:e])
            else:
                raise ValueError(f"Unsupported LR model_type: {model_type}")
            (loss * weight).backward()
    grad_norm_sq = 0.0
    for m in models:
        for p in m.parameters():
            if p.grad is not None:
                grad_norm_sq += float(p.grad.detach().pow(2).sum().item())
    for m, training in zip(models, was_training):
        m.zero_grad(set_to_none=True)
        m.train(training)
    return float(grad_norm_sq ** 0.5)


def _assert_selected_lr_checkpoint_converged(
    *,
    task_name: str,
    stop_reason: str,
    epochs_run: int,
    best_epoch: int,
    selected_objective_relative_change: float,
    selected_full_objective_gradient_norm: float,
    cfg: Mapping[str, Any],
) -> None:
    if not bool(cfg.get("fail_on_unconverged_max_epochs", False)):
        return
    if str(stop_reason) != "max_epochs":
        return
    final_rel = float(selected_objective_relative_change)
    final_grad = float(selected_full_objective_gradient_norm)
    rel_threshold = float(cfg.get("max_epoch_relative_change_fail_threshold", 1.0e-3))
    if "max_epoch_gradient_norm_fail_threshold" not in cfg:
        raise RuntimeError(f"{task_name} requires max_epoch_gradient_norm_fail_threshold when fail_on_unconverged_max_epochs=true.")
    grad_threshold = float(cfg["max_epoch_gradient_norm_fail_threshold"])
    relative_bad = final_rel > rel_threshold
    gradient_bad = final_grad > grad_threshold
    if not (relative_bad or gradient_bad):
        return
    raise RuntimeError(
        f"{task_name} selected checkpoint reached max_epochs without convergence: "
        f"epochs_run={int(epochs_run)} "
        f"best_epoch={int(best_epoch)} "
        f"stop_reason={stop_reason} "
        f"selected_checkpoint_objective_relative_change={final_rel:.6g} "
        f"relative_threshold={rel_threshold:.6g} "
        f"selected_checkpoint_full_objective_gradient_norm={final_grad:.6g} "
        f"gradient_threshold={grad_threshold:.6g}"
    )


def _cfg_candidates(cfg: Dict[str, Any], key: str) -> List[Any]:
    cand_key = f"{key}_candidates"
    if cand_key in cfg:
        vals = cfg[cand_key]
        return list(vals if isinstance(vals, (list, tuple)) else [vals])
    return [cfg[key]]


def _xgb_candidate_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n_estimators in _cfg_candidates(cfg, "n_estimators"):
        for max_depth in _cfg_candidates(cfg, "max_depth"):
            for learning_rate in _cfg_candidates(cfg, "learning_rate"):
                for subsample in _cfg_candidates(cfg, "subsample"):
                    for colsample in _cfg_candidates(cfg, "colsample_bytree"):
                        out.append({
                            "n_estimators": int(n_estimators),
                            "max_depth": int(max_depth),
                            "learning_rate": float(learning_rate),
                            "subsample": float(subsample),
                            "colsample_bytree": float(colsample),
                        })
    return out


def _xgb_common_params(cfg: Dict[str, Any], cand: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": int(cand["n_estimators"]),
        "max_depth": int(cand["max_depth"]),
        "subsample": float(cand["subsample"]),
        "colsample_bytree": float(cand["colsample_bytree"]),
        "learning_rate": float(cand["learning_rate"]),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "early_stopping_rounds": int(cfg.get("early_stopping_rounds", 20)),
        "random_state": int(cfg.get("seed", 42)),
        "verbosity": 0,
    }


def _xgb_prob(model: Any, x: np.ndarray) -> np.ndarray:
    pp = model.predict_proba(x)
    if isinstance(pp, list):
        return np.stack([np.asarray(p)[:, 1] for p in pp], axis=1).astype(np.float32)
    arr = np.asarray(pp)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[:, :, 1].astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1].astype(np.float32)
    if arr.ndim == 2:
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    raise RuntimeError(f"Unexpected XGBoost predict_proba shape: {arr.shape}")


def _torch_logistic_metrics(y_true: np.ndarray, prob: np.ndarray, is_binary: bool) -> Dict[str, float]:
    if is_binary:
        return compute_binary_metrics(y_true.astype(np.float32, copy=False), prob.astype(np.float32, copy=False))
    return _augment_multilabel_metrics(y_true.astype(np.float32, copy=False), prob.astype(np.float32, copy=False))


def _require_xgboost_cuda() -> Tuple[Any, Dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal XGBoost benchmark requires CUDA; CPU fallback is not allowed.")
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError(f"xgboost not available: {e}")
    info: Dict[str, Any] = {"version": str(getattr(xgb, "__version__", "unknown"))}
    try:
        info["build_info"] = xgb.build_info()
    except Exception as e:
        info["build_info_error"] = str(e)
    info["cuda_device"] = torch.cuda.get_device_name(0)
    build_text = json.dumps(info.get("build_info", {}), default=str).lower()
    if "cuda" not in build_text:
        raise RuntimeError("Installed xgboost build does not report CUDA support; formal benchmark fails closed.")
    return XGBClassifier, info


def _validate_xgboost_gpu_multioutput(XGBClassifier: Any, cfg: Dict[str, Any], num_labels: int) -> List[Dict[str, Any]]:
    strategy = str(cfg.get("multi_strategy", "one_output_per_tree"))
    if strategy == "multi_output_tree":
        raise RuntimeError("XGBoost multi_output_tree is exploratory and is not allowed in formal benchmark.")
    cache_key = (strategy, int(num_labels))
    if cache_key in _XGBOOST_CAPABILITY_AUDIT_CACHE:
        return list(_XGBOOST_CAPABILITY_AUDIT_CACHE[cache_key])
    requested = {int(num_labels)}
    if int(num_labels) >= 26:
        requested.add(26)
    if int(num_labels) >= 512:
        requested.add(512)
    if int(num_labels) >= 2048:
        requested.add(2048)
    audits: List[Dict[str, Any]] = []
    for n_labels in sorted(requested):
        n_labels = int(max(2, n_labels))
        n_rows = max(32, 2 * n_labels)
        rng = np.random.default_rng(123)
        x = rng.normal(size=(n_rows, 3)).astype(np.float32)
        y = np.zeros((n_rows, n_labels), dtype=np.int32)
        for j in range(n_labels):
            y[j, j] = 1
            y[n_labels + j if n_labels + j < n_rows else j, j] = 1
        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            multi_strategy=strategy,
            n_estimators=2,
            max_depth=2,
            learning_rate=0.2,
            n_jobs=1,
            verbosity=0,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.fit(x, y)
        fit_sec = float(time.time() - t0)
        pred = model.predict_proba(x)
        if isinstance(pred, list):
            shape = (len(pred[0]), len(pred))
        else:
            arr = np.asarray(pred)
            shape = arr.shape[:2] if arr.ndim == 3 else arr.shape
        if tuple(shape) != (n_rows, n_labels):
            raise RuntimeError(f"XGBoost GPU multi-output capability test returned invalid shape {shape}, expected {(n_rows, n_labels)}")
        audits.append({
            "num_labels": int(n_labels),
            "n_rows": int(n_rows),
            "fit_sec": fit_sec,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        })
    _XGBOOST_CAPABILITY_AUDIT_CACHE[cache_key] = list(audits)
    return audits


def _train_torch_logistic_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    cfg: Dict[str, Any],
    is_binary: bool,
    task_name: Optional[str],
) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    tfit0 = time.time()
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    X_tr_s, X_va_s, X_te_s, scaler = _standardize_train_val_test(X_tr, X_va, X_te)
    device = _linear_device()
    xtr = torch.from_numpy(X_tr_s).to(device)
    xva = torch.from_numpy(X_va_s).to(device)
    xte = torch.from_numpy(X_te_s).to(device)
    yva_np = y_va.astype(np.float32, copy=False)
    yte_np = y_te.astype(np.float32, copy=False)
    ytr = torch.from_numpy(y_tr.astype(np.float32, copy=False)).to(device)
    if is_binary:
        ytr = ytr.view(-1, 1)
    n, d = xtr.shape
    out_dim = 1 if is_binary else int(ytr.shape[1])
    bs = int(max(32, int(cfg.get("shared_batch_size", 1024))))
    epochs = int(max(1, int(cfg.get("shared_epochs", 25))))
    patience = int(max(1, int(cfg.get("shared_patience", 3))))
    lr = float(cfg.get("shared_lr", cfg.get("lr", 1.0e-3)))
    rel_tol = float(cfg.get("objective_relative_tolerance", 0.0))
    grad_tol = float(cfg.get("gradient_norm_tolerance", -1.0))
    best: Dict[str, Any] = {"score": -np.inf}
    candidate_audit: List[Dict[str, Any]] = []
    for cand in _linear_candidate_grid(cfg):
        model = nn.Linear(int(d), int(out_dim), bias=bool(cfg.get("intercept", True))).to(device)
        if model.bias is not None:
            nn.init.zeros_(model.bias)
        opt_params = [{"params": [model.weight], "weight_decay": float(cand["weight_decay"])}]
        if model.bias is not None:
            opt_params.append({"params": [model.bias], "weight_decay": 0.0})
        opt = torch.optim.AdamW(opt_params, lr=lr)
        if str(cand["class_weight"]) == "balanced":
            pos = ytr.sum(dim=0)
            neg = ytr.shape[0] - pos
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 100.0)
        else:
            pos_weight = torch.ones((out_dim,), dtype=torch.float32, device=device)
        curve: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_score = -np.inf
        best_epoch = -1
        bad = 0
        stop_reason = "max_epochs"
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(int(n), device=device)
            train_loss_sum = 0.0
            train_count = 0
            for s in range(0, int(n), bs):
                idx = perm[s:s + bs]
                logits = model(xtr.index_select(0, idx))
                target = ytr.index_select(0, idx)
                loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm_sq += float(p.grad.detach().pow(2).sum().item())
                opt.step()
                train_loss_sum += float(loss.detach().item()) * int(idx.numel())
                train_count += int(idx.numel())
            model.eval()
            with torch.no_grad():
                va_prob = torch.sigmoid(model(xva)).detach().cpu().numpy().astype(np.float32)
            if is_binary:
                va_prob_eval = va_prob.reshape(-1)
            else:
                va_prob_eval = va_prob
            va_metrics = _torch_logistic_metrics(yva_np, va_prob_eval, is_binary)
            score = float(va_metrics.get("auprc", -np.inf) if is_binary else _val_score_for_multilabel(task_name, va_metrics))
            train_loss_epoch = float(train_loss_sum / max(train_count, 1))
            rel_change = 0.0 if not curve else abs(float(curve[-1]["train_loss"]) - train_loss_epoch) / max(abs(float(curve[-1]["train_loss"])), 1e-8)
            curve.append({
                "epoch": int(epoch),
                "train_loss": train_loss_epoch,
                "val_primary": float(score),
                "grad_norm": float(grad_norm_sq ** 0.5),
                "objective_relative_change": float(rel_change),
            })
            min_delta = float(cfg.get("min_delta", 0.0))
            if score > best_score + min_delta:
                best_score = float(score)
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    stop_reason = "validation_patience"
                    break
            if epoch > 0 and rel_tol > 0.0 and grad_tol >= 0.0 and rel_change <= rel_tol:
                full_grad_epoch = _compute_full_objective_gradient_norm(
                    model_type="ordinary",
                    models=[model],
                    x_train=xtr,
                    y_train=ytr,
                    card_train=None,
                    pos_weight=pos_weight,
                    batch_size=bs,
                )
                if full_grad_epoch <= grad_tol:
                    stop_reason = "objective_converged"
                    break
        final_training_full_grad = _compute_full_objective_gradient_norm(
            model_type="ordinary",
            models=[model],
            x_train=xtr,
            y_train=ytr,
            card_train=None,
            pos_weight=pos_weight,
            batch_size=bs,
        )
        if curve:
            curve[-1]["final_training_state_full_objective_gradient_norm"] = float(final_training_full_grad)
        if best_state is None or best_epoch < 0 or best_epoch >= len(curve):
            raise RuntimeError(f"ordinary LR/{task_name or 'unknown'} did not produce a valid validation-selected checkpoint.")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        selected_full_grad = _compute_full_objective_gradient_norm(
            model_type="ordinary",
            models=[model],
            x_train=xtr,
            y_train=ytr,
            card_train=None,
            pos_weight=pos_weight,
            batch_size=bs,
        )
        selected_rel_change = float(curve[int(best_epoch)]["objective_relative_change"])
        selected_converged = bool(
            rel_tol > 0.0
            and grad_tol >= 0.0
            and selected_rel_change <= rel_tol
            and selected_full_grad <= grad_tol
        )
        model.eval()
        with torch.no_grad():
            va_prob_final = torch.sigmoid(model(xva)).detach().cpu().numpy().astype(np.float32)
        va_prob_eval_final = va_prob_final.reshape(-1) if is_binary else va_prob_final
        va_metrics_final = _torch_logistic_metrics(yva_np, va_prob_eval_final, is_binary)
        selected_score = float(va_metrics_final.get("auprc", -np.inf) if is_binary else _val_score_for_multilabel(task_name, va_metrics_final))
        if not np.isclose(selected_score, float(best_score), rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError(
                f"ordinary LR/{task_name or 'unknown'} validation score mismatch after restoring selected checkpoint: "
                f"selected_score={selected_score:.8g} recorded_best_score={float(best_score):.8g}"
            )
        candidate_audit.append({
            "weight_decay": float(cand["weight_decay"]),
            "class_weight": str(cand["class_weight"]),
            "best_epoch": int(best_epoch),
            "best_val_primary": float(best_score),
            "epochs_run": int(len(curve)),
            "final_epoch": int(curve[-1]["epoch"]) if curve else None,
            "converged": bool(selected_converged),
            "stop_reason": str(stop_reason),
            "final_training_state_objective_relative_change": float(curve[-1]["objective_relative_change"]) if curve else None,
            "final_training_state_full_objective_gradient_norm": float(final_training_full_grad),
            "selected_checkpoint_objective_relative_change": float(selected_rel_change),
            "selected_checkpoint_full_objective_gradient_norm": float(selected_full_grad),
            "last_minibatch_gradient_norm": float(curve[-1]["grad_norm"]) if curve else None,
            "relative_change_threshold": float(cfg.get("max_epoch_relative_change_fail_threshold", 1.0e-3)),
            "gradient_norm_threshold": float(cfg["max_epoch_gradient_norm_fail_threshold"]) if "max_epoch_gradient_norm_fail_threshold" in cfg else None,
        })
        _assert_selected_lr_checkpoint_converged(
            task_name=f"ordinary LR/{task_name or 'unknown'}",
            stop_reason=stop_reason,
            epochs_run=len(curve),
            best_epoch=best_epoch,
            selected_objective_relative_change=selected_rel_change,
            selected_full_objective_gradient_norm=selected_full_grad,
            cfg=cfg,
        )
        if best_score > float(best.get("score", -np.inf)):
            best = {
                "score": float(best_score),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "val_prob": va_prob_eval_final.astype(np.float32, copy=False),
                "best_epoch": int(best_epoch),
                "loss_curve": curve,
                "regularization": cand,
            }
    selected = nn.Linear(int(d), int(out_dim), bias=bool(cfg.get("intercept", True))).to(device)
    selected.load_state_dict({k: v.to(device) for k, v in best["state_dict"].items()})
    selected.eval()
    t_test_pred = time.time()
    with torch.no_grad():
        te_prob = torch.sigmoid(selected(xte)).detach().cpu().numpy().astype(np.float32)
    test_inference_sec = float(time.time() - t_test_pred)
    te_prob_eval = te_prob.reshape(-1) if is_binary else te_prob
    te_metrics = _torch_logistic_metrics(yte_np, te_prob_eval, is_binary)
    best["metrics"] = te_metrics
    best["prob"] = te_prob_eval.astype(np.float32, copy=False)
    best["test_inference_sec"] = test_inference_sec
    artifact = {
        "kind": "torch_cuda_logistic_regression" if device.type == "cuda" else "torch_cpu_logistic_regression",
        "model_family": "logistic_regression_linear_classifier",
        "input_dim": int(d),
        "num_labels": int(out_dim),
        "state_dict": best["state_dict"],
        "val_probability": best["val_prob"],
        "test_probability": best["prob"],
        "scaler": scaler,
        "convergence_audit": {
            "candidate_audit": candidate_audit,
            "loss_curve": best["loss_curve"],
            "best_epoch": int(best["best_epoch"]),
            "selected_regularization": best["regularization"],
            "device": str(device),
            "test_inference_sec": float(best["test_inference_sec"]),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        },
        "cfg": {
            "shared_lr": lr,
            "shared_batch_size": int(max(32, int(cfg.get("shared_batch_size", 1024)))),
            "shared_epochs": epochs,
            "shared_patience": patience,
        },
    }
    return dict(best["metrics"]), float(best["score"]), float(time.time() - tfit0), artifact


def _predict_binary_from_artifact(artifact: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    kind = str(artifact.get("kind", ""))
    if kind == "binary_prior_constant":
        return np.full((X.shape[0],), float(artifact.get("prior", 0.0)), dtype=np.float32)
    if "xgb" in kind:
        model = artifact.get("model")
        if model is None:
            raise RuntimeError("XGBoost artifact is missing model.")
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float32)
    if "logistic_regression" in kind:
        scaler = artifact.get("scaler")
        if not isinstance(scaler, Mapping) or "mean" not in scaler or "std" not in scaler:
            raise RuntimeError("Torch logistic artifact is missing train-only scaler.")
        x = ((X.astype(np.float32, copy=False) - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)).astype(np.float32)
        state = artifact.get("state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError("Torch logistic artifact is missing state_dict.")
        weight = state.get("weight")
        bias = state.get("bias")
        if weight is None:
            raise RuntimeError("Torch logistic artifact state_dict lacks weight.")
        device = _linear_device()
        model = nn.Linear(int(x.shape[1]), 1, bias=(bias is not None)).to(device)
        model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else torch.as_tensor(v, device=device) for k, v in state.items()})
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(x).to(device))).detach().cpu().numpy().reshape(-1)
        return prob.astype(np.float32, copy=False)
    raise RuntimeError(f"Unsupported benchmark artifact kind for binary prediction: {kind}")


def _predict_torch_logistic_from_artifact(artifact: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    scaler = artifact.get("scaler")
    if not isinstance(scaler, Mapping) or "mean" not in scaler or "std" not in scaler:
        raise RuntimeError("Torch logistic artifact is missing train-only scaler.")
    x = ((X.astype(np.float32, copy=False) - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)).astype(np.float32)
    state = artifact.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Torch logistic artifact is missing state_dict.")
    weight = state.get("weight")
    bias = state.get("bias")
    if weight is None:
        raise RuntimeError("Torch logistic artifact state_dict lacks weight.")
    out_dim = int(weight.shape[0]) if hasattr(weight, "shape") and len(weight.shape) == 2 else 1
    device = _linear_device()
    model = nn.Linear(int(x.shape[1]), int(out_dim), bias=(bias is not None)).to(device)
    model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else torch.as_tensor(v, device=device) for k, v in state.items()})
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x).to(device))).detach().cpu().numpy()
    return prob.astype(np.float32, copy=False)


def _predict_drg_lr_from_artifact(artifact: Mapping[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = artifact.get("scaler")
    if not isinstance(scaler, Mapping) or "mean" not in scaler or "std" not in scaler:
        raise RuntimeError("DRG LR artifact is missing train-only scaler.")
    x = ((X.astype(np.float32, copy=False) - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)).astype(np.float32)
    label_state = artifact.get("label_state_dict")
    card_state = artifact.get("cardinality_state_dict")
    if not isinstance(label_state, Mapping) or not isinstance(card_state, Mapping):
        raise RuntimeError("DRG LR artifact is missing label/cardinality state_dict.")
    label_weight = label_state.get("weight")
    card_weight = card_state.get("weight")
    if label_weight is None or card_weight is None:
        raise RuntimeError("DRG LR artifact state_dict lacks weight.")
    device = _linear_device()
    label_head = nn.Linear(int(x.shape[1]), int(label_weight.shape[0]), bias=("bias" in label_state)).to(device)
    card_head = nn.Linear(int(x.shape[1]), int(card_weight.shape[0]), bias=("bias" in card_state)).to(device)
    label_head.load_state_dict({k: v.to(device) if torch.is_tensor(v) else torch.as_tensor(v, device=device) for k, v in label_state.items()})
    card_head.load_state_dict({k: v.to(device) if torch.is_tensor(v) else torch.as_tensor(v, device=device) for k, v in card_state.items()})
    label_head.eval()
    card_head.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        label_scores = torch.sigmoid(label_head(xt)).detach().cpu().numpy().astype(np.float32)
        card_prob = torch.softmax(card_head(xt), dim=-1).detach().cpu().numpy().astype(np.float32)
    pred, card_pred = _drg_decode(label_scores, card_prob)
    return label_scores, pred, card_pred


def _predict_drg_xgb_from_artifact(artifact: Mapping[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    card_head = artifact.get("cardinality_head")
    rel_head = artifact.get("label_relevance_head")
    if not isinstance(card_head, Mapping) or not isinstance(rel_head, Mapping):
        raise RuntimeError("DRG XGBoost artifact is missing cardinality or relevance head.")
    rel_model = rel_head.get("model")
    if rel_model is None:
        raise RuntimeError("DRG XGBoost relevance head is missing model.")
    card_prob = _predict_multiclass_xgb_candidate(card_head, X, 3)
    label_scores = _xgb_prob(rel_model, X)
    pred, card_pred = _drg_decode(label_scores, card_prob)
    return label_scores.astype(np.float32, copy=False), pred, card_pred


def _eval_existing_benchmark_artifact(
    *,
    task: str,
    baseline: str,
    artifact: Mapping[str, Any],
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    kind = str(artifact.get("kind", ""))
    if task in SEQUENCE_BINARY_TASKS:
        prob = _predict_binary_from_artifact(artifact, X_te).reshape(-1)
        metrics = compute_binary_metrics(y_te.astype(np.float32, copy=False), prob.astype(np.float32, copy=False))
        return metrics, {"probability": prob.astype(np.float32, copy=False)}
    if task in ("icd", "hcup"):
        if baseline == "lr" or "logistic_regression" in kind:
            prob = _predict_torch_logistic_from_artifact(artifact, X_te)
        elif "xgb" in kind:
            model = artifact.get("model")
            if model is None:
                raise RuntimeError(f"{task} XGBoost artifact is missing model.")
            prob = _xgb_prob(model, X_te)
        else:
            raise RuntimeError(f"Unsupported {task} benchmark artifact kind for eval-only: {kind}")
        metrics = _augment_multilabel_metrics(y_te.astype(np.float32, copy=False), prob.astype(np.float32, copy=False))
        return metrics, {"probability": prob.astype(np.float32, copy=False)}
    if task == "drg":
        card_te = _drg_cardinality_target(y_te)
        if baseline == "lr" or "drg_logistic_regression" in kind:
            label_scores, pred, card_pred = _predict_drg_lr_from_artifact(artifact, X_te)
        elif "xgb" in kind:
            label_scores, pred, card_pred = _predict_drg_xgb_from_artifact(artifact, X_te)
        else:
            raise RuntimeError(f"Unsupported DRG benchmark artifact kind for eval-only: {kind}")
        metrics = _drg_set_metrics(y_te, pred, card_te, card_pred, label_scores)
        return metrics, {
            "label_scores": label_scores.astype(np.float32, copy=False),
            "prediction": pred.astype(np.float32, copy=False),
            "predicted_cardinality": card_pred.astype(np.int64, copy=False),
        }
    raise RuntimeError(f"Unsupported task for eval-only benchmark: {task}")


def _load_existing_benchmark_artifact_for_eval(
    *,
    ckpt_path: str,
    task: str,
    baseline: str,
    cutoff_tag: str,
    summary: str,
    feature_contract_signature: str,
) -> Mapping[str, Any]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"eval-existing benchmark requires existing model.ckpt: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    run_sig = ckpt.get("run_signature")
    if not isinstance(run_sig, Mapping):
        raise RuntimeError(f"Existing benchmark ckpt lacks run_signature: {ckpt_path}")
    if str(run_sig.get("task")) != str(task) or str(run_sig.get("baseline")) != str(baseline):
        raise RuntimeError(f"Existing benchmark ckpt task/baseline mismatch: {ckpt_path}")
    if str(run_sig.get("summary")) != str(summary) or str(run_sig.get("window")) != str(cutoff_tag):
        raise RuntimeError(f"Existing benchmark ckpt summary/window mismatch: {ckpt_path}")
    if str(run_sig.get("feature_contract_signature")) != str(feature_contract_signature):
        raise RuntimeError(f"Existing benchmark ckpt feature contract mismatch: {ckpt_path}")
    artifact = ckpt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"Existing benchmark ckpt lacks artifact: {ckpt_path}")
    return artifact


def _train_shared_multilabel_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    cfg: Dict[str, Any],
    task_name: Optional[str],
    ) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    XGBClassifier, xgb_info = _require_xgboost_cuda()
    capability_audit = _validate_xgboost_gpu_multioutput(XGBClassifier, cfg, y_tr.shape[1])
    tfit0 = time.time()
    best: Dict[str, Any] = {"score": -np.inf}
    candidate_audit: List[Dict[str, Any]] = []
    for cand in _xgb_candidate_grid(cfg):
        m = XGBClassifier(
            objective="binary:logistic",
            multi_strategy=str(cfg.get("multi_strategy", "one_output_per_tree")),
            eval_metric="logloss",
            **_xgb_common_params(cfg, cand),
        )
        t_fit = time.time()
        m.fit(X_tr, y_tr.astype(np.int32, copy=False), eval_set=[(X_va, y_va.astype(np.int32, copy=False))], verbose=False)
        fit_sec = float(time.time() - t_fit)
        t_val = time.time()
        va_prob = _xgb_prob(m, X_va)
        val_inference_sec = float(time.time() - t_val)
        va_metrics = _augment_multilabel_metrics(y_va.astype(np.float32, copy=False), va_prob)
        score = _val_score_for_multilabel(task_name, va_metrics)
        candidate_audit.append({
            **cand,
            "best_iteration": int(getattr(m, "best_iteration", -1) if getattr(m, "best_iteration", None) is not None else -1),
            "fit_sec": float(fit_sec),
            "val_inference_sec": float(val_inference_sec),
            "val_primary": float(score),
        })
        if score > float(best.get("score", -np.inf)):
            best = {
                "score": float(score),
                "model": m,
                "val_probability": va_prob,
                "selected_params": dict(cand),
                "fit_sec": float(fit_sec),
                "val_inference_sec": float(val_inference_sec),
            }
    t_test = time.time()
    te_prob = _xgb_prob(best["model"], X_te)
    test_inference_sec = float(time.time() - t_test)
    te_metrics = _augment_multilabel_metrics(y_te.astype(np.float32, copy=False), te_prob)
    artifact = {
        "kind": "shared_multilabel_xgb_one_output_per_tree",
        "model": best["model"],
        "val_probability": best["val_probability"],
        "test_probability": te_prob,
        "xgboost_info": xgb_info,
        "xgboost_multioutput_capability_audit": capability_audit,
        "candidate_audit": candidate_audit,
        "runtime_audit": {
            "fit_sec": float(best["fit_sec"]),
            "val_inference_sec": float(best["val_inference_sec"]),
            "test_inference_sec": test_inference_sec,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        },
        "cfg": {**best["selected_params"], "n_jobs": int(cfg.get("n_jobs", -1)), "multi_strategy": str(cfg.get("multi_strategy", "one_output_per_tree"))},
    }
    return te_metrics, float(best["score"]), float(time.time() - tfit0), artifact


def train_lr(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, cfg: Dict[str, Any], is_binary: bool, task_name: Optional[str] = None) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    if is_binary and len(np.unique(y_tr)) < 2:
        X_tr_s, X_va_s, X_te_s, scaler = _standardize_train_val_test(X_tr, X_va, X_te)
        p0 = float(np.mean(y_tr)) if y_tr.size > 0 else 0.0
        va_prob = np.full((X_va_s.shape[0],), p0, dtype=np.float32)
        te_prob = np.full((X_te_s.shape[0],), p0, dtype=np.float32)
        artifact = {
            "kind": "binary_prior_constant",
            "reason": "single_class_train",
            "prior": float(p0),
            "scaler": scaler,
            "val_probability": va_prob,
            "test_probability": te_prob,
            "convergence_audit": {"single_class_train": True},
        }
        return compute_binary_metrics(y_te, te_prob), float(compute_binary_metrics(y_va, va_prob).get("auprc", 0.0)), 0.0, artifact
    return _train_torch_logistic_classifier(X_tr, y_tr, X_va, y_va, X_te, y_te, cfg, is_binary, task_name)


def train_xgb(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, cfg: Dict[str, Any], is_binary: bool, task_name: Optional[str] = None) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    tfit0 = time.time()

    if is_binary:
        if len(np.unique(y_tr)) < 2:
            p0 = float(np.mean(y_tr)) if y_tr.size > 0 else 0.0
            va_prob = np.full((X_va.shape[0],), p0, dtype=np.float32)
            te_prob = np.full((X_te.shape[0],), p0, dtype=np.float32)
            artifact = {"kind": "binary_prior_constant", "prior": float(p0)}
        else:
            XGBClassifier, xgb_info = _require_xgboost_cuda()
            pos = float(np.sum(y_tr))
            neg = float(y_tr.shape[0] - pos)
            spw = float(np.clip(neg / max(pos, 1e-6), 1.0, 100.0)) if (pos > 0.0 and neg > 0.0) else 1.0
            best: Dict[str, Any] = {"score": -np.inf}
            candidate_audit: List[Dict[str, Any]] = []
            for cand in _xgb_candidate_grid(cfg):
                m = XGBClassifier(
                    objective="binary:logistic",
                    scale_pos_weight=spw,
                    eval_metric="aucpr",
                    **_xgb_common_params(cfg, cand),
                )
                t_fit_xgb = time.time()
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                fit_xgb_sec = float(time.time() - t_fit_xgb)
                t_val_xgb = time.time()
                va_prob = _xgb_prob(m, X_va).reshape(-1)
                val_pred_sec = float(time.time() - t_val_xgb)
                score = float(compute_binary_metrics(y_va, va_prob).get("auprc", 0.0))
                candidate_audit.append({
                    **cand,
                    "best_iteration": int(getattr(m, "best_iteration", -1) if getattr(m, "best_iteration", None) is not None else -1),
                    "fit_sec": float(fit_xgb_sec),
                    "val_inference_sec": float(val_pred_sec),
                    "val_primary": float(score),
                })
                if score > float(best.get("score", -np.inf)):
                    best = {
                        "score": float(score),
                        "model": m,
                        "val_probability": va_prob.astype(np.float32, copy=False),
                        "selected_params": dict(cand),
                        "fit_sec": float(fit_xgb_sec),
                        "val_inference_sec": float(val_pred_sec),
                    }
            t_test_xgb = time.time()
            te_prob = _xgb_prob(best["model"], X_te).reshape(-1).astype(np.float32, copy=False)
            test_pred_sec = float(time.time() - t_test_xgb)
            artifact = {
                "kind": "binary_xgb_classifier",
                "model": best["model"],
                "xgboost_info": xgb_info,
                "candidate_audit": candidate_audit,
                "runtime_audit": {
                    "fit_sec": float(best["fit_sec"]),
                    "val_inference_sec": float(best["val_inference_sec"]),
                    "test_inference_sec": test_pred_sec,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                },
                "cfg": {**best["selected_params"], "n_jobs": int(cfg.get("n_jobs", -1))},
            }
            va_prob = best["val_probability"]
            score = float(best["score"])
        if len(np.unique(y_tr)) < 2:
            score = float(compute_binary_metrics(y_va, va_prob).get("auprc", 0.0))
        metrics = compute_binary_metrics(y_te, te_prob)
        artifact["val_probability"] = va_prob
        artifact["test_probability"] = te_prob
    else:
        return _train_shared_multilabel_xgb(
            X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te, cfg=cfg, task_name=task_name,
        )
    return metrics, score, float(time.time() - tfit0), artifact


def _drg_cardinality_target(y: np.ndarray) -> np.ndarray:
    counts = np.asarray(y, dtype=np.float32).sum(axis=1)
    if np.any(counts > 2.0):
        raise RuntimeError("DRG competitive benchmark supports cardinality 0/1/2; targets with >2 labels are invalid.")
    return counts.astype(np.int64)


def _drg_decode(label_scores: np.ndarray, card_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    label_scores = np.asarray(label_scores, dtype=np.float32)
    card_prob = np.asarray(card_prob, dtype=np.float32)
    eps = np.float32(1e-6)
    label_logits = torch.from_numpy(np.log(np.clip(label_scores, eps, 1.0 - eps) / np.clip(1.0 - label_scores, eps, 1.0))).float()
    card_logits = torch.from_numpy(np.log(np.clip(card_prob, eps, 1.0))).float()
    pred = CompetitiveDRGProbe.decode(label_logits, card_logits).cpu().numpy().astype(np.float32)
    card_pred = card_prob.argmax(axis=1).astype(np.int64)
    return pred, card_pred


def _drg_set_metrics(y_true: np.ndarray, y_pred: np.ndarray, card_true: np.ndarray, card_pred: np.ndarray, label_scores: np.ndarray) -> Dict[str, float]:
    yt = y_true.astype(np.float32)
    yp = y_pred.astype(np.float32)
    label_scores = np.asarray(label_scores, dtype=np.float32)
    tp = float((yt * yp).sum())
    fp = float(((1.0 - yt) * yp).sum())
    fn = float((yt * (1.0 - yp)).sum())
    p = tp / max(tp + fp, 1e-8)
    r = tp / max(tp + fn, 1e-8)
    micro_f1 = float(0.0 if (p + r) <= 0 else 2.0 * p * r / (p + r))

    per = []
    for j in range(yt.shape[1]):
        tpj = float((yt[:, j] * yp[:, j]).sum())
        fpj = float(((1.0 - yt[:, j]) * yp[:, j]).sum())
        fnj = float((yt[:, j] * (1.0 - yp[:, j])).sum())
        pj = tpj / max(tpj + fpj, 1e-8)
        rj = tpj / max(tpj + fnj, 1e-8)
        per.append(0.0 if (pj + rj) <= 0 else 2.0 * pj * rj / (pj + rj))
    macro_f1 = float(np.mean(per)) if per else 0.0

    exact = float(np.mean(np.all(yt == yp, axis=1)))
    card_acc = float(np.mean(card_true == card_pred))
    true_k = yt.sum(axis=1)
    eligible = true_k > 0
    eligible_count = int(eligible.sum())
    if eligible_count:
        ranks = np.argsort(-label_scores, axis=1)
        eligible_rows = np.where(eligible)[0]
        top1_hit = float(np.mean([bool(yt[i, ranks[i, 0]]) for i in eligible_rows]))
        top2_hit = float(np.mean([bool(yt[i, ranks[i, : min(2, yt.shape[1])]].any()) for i in eligible_rows]))
        top5_recall = float(np.mean([
            float(yt[i, ranks[i, : min(5, yt.shape[1])]].sum() / max(true_k[i], 1.0))
            for i in eligible_rows
        ]))
    else:
        top1_hit = top2_hit = top5_recall = float("nan")
    aux = _augment_multilabel_metrics(y_true, label_scores)

    out = {
        "exact_set_match": exact,
        "cardinality_accuracy": card_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "top1_hit": top1_hit,
        "top2_hit": top2_hit,
        "top5_recall": top5_recall,
        "topk_eligible_patient_count": eligible_count,
        "predicted_label_count_mean": float(yp.sum(axis=1).mean()),
        "predicted_label_count_std": float(yp.sum(axis=1).std()),
        "true_label_count_mean": float(yt.sum(axis=1).mean()),
        "true_label_count_std": float(yt.sum(axis=1).std()),
    }
    for k, v in aux.items():
        out[f"relevance_{k}"] = float(v)
    return out




def _fit_multiclass_xgb_candidate(
    XGBClassifier: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Dict[str, Any],
    cand: Mapping[str, Any],
    num_classes: int,
) -> Dict[str, Any]:
    if y_tr.size == 0 or len(np.unique(y_tr)) < 2:
        cls = int(y_tr[0]) if y_tr.size > 0 else 0
        cls = max(0, min(int(num_classes - 1), cls))
        p_va = np.zeros((X_va.shape[0], num_classes), dtype=np.float32)
        p_va[:, cls] = 1.0
        return {"kind": "constant_class", "constant_class": int(cls), "model": None, "val_probability": p_va, "fit_sec": 0.0}
    x_fit = X_tr
    y_fit = y_tr.astype(np.int64, copy=False)
    sample_weight = None
    present = set(int(v) for v in np.unique(y_fit))
    missing = [c for c in range(int(num_classes)) if c not in present]
    if missing:
        if X_tr.shape[0] == 0:
            raise RuntimeError("Cannot fit XGBoost cardinality head with empty training matrix.")
        dummy_x = np.repeat(X_tr[:1], len(missing), axis=0)
        dummy_y = np.asarray(missing, dtype=np.int64)
        x_fit = np.concatenate([X_tr, dummy_x], axis=0)
        y_fit = np.concatenate([y_fit, dummy_y], axis=0)
        sample_weight = np.concatenate([
            np.ones(X_tr.shape[0], dtype=np.float32),
            np.zeros(len(missing), dtype=np.float32),
        ])
    m = XGBClassifier(
        objective="multi:softprob",
        num_class=int(num_classes),
        eval_metric="mlogloss",
        **_xgb_common_params(cfg, cand),
    )
    t0 = time.time()
    fit_kwargs = {"eval_set": [(X_va, y_va)], "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    m.fit(x_fit, y_fit, **fit_kwargs)
    fit_sec = float(time.time() - t0)
    val_probability = m.predict_proba(X_va).astype(np.float32)
    if val_probability.ndim != 2 or val_probability.shape[1] != int(num_classes):
        raise RuntimeError(
            f"XGBoost cardinality head returned probability shape {tuple(val_probability.shape)}, "
            f"expected (*, {int(num_classes)})."
        )
    return {
        "kind": "xgb_multiclass",
        "model": m,
        "val_probability": val_probability,
        "fit_sec": fit_sec,
        "best_iteration": int(getattr(m, "best_iteration", -1) if getattr(m, "best_iteration", None) is not None else -1),
        "zero_weight_missing_classes": missing,
    }


def _predict_multiclass_xgb_candidate(head: Mapping[str, Any], X: np.ndarray, num_classes: int) -> np.ndarray:
    if head.get("kind") == "constant_class":
        cls = int(head.get("constant_class", 0))
        p = np.zeros((X.shape[0], int(num_classes)), dtype=np.float32)
        p[:, cls] = 1.0
        return p
    model = head.get("model")
    if model is None:
        raise RuntimeError("XGBoost cardinality head is missing model.")
    return model.predict_proba(X).astype(np.float32)


def _fit_multilabel_xgb_candidate(
    XGBClassifier: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Dict[str, Any],
    cand: Mapping[str, Any],
) -> Dict[str, Any]:
    m = XGBClassifier(
        objective="binary:logistic",
        multi_strategy=str(cfg.get("multi_strategy", "one_output_per_tree")),
        eval_metric="logloss",
        **_xgb_common_params(cfg, cand),
    )
    t0 = time.time()
    m.fit(
        X_tr,
        y_tr.astype(np.int32, copy=False),
        eval_set=[(X_va, y_va.astype(np.int32, copy=False))],
        verbose=False,
    )
    fit_sec = float(time.time() - t0)
    return {
        "kind": "xgb_relevance_one_output_per_tree",
        "model": m,
        "val_probability": _xgb_prob(m, X_va),
        "fit_sec": fit_sec,
        "best_iteration": int(getattr(m, "best_iteration", -1) if getattr(m, "best_iteration", None) is not None else -1),
    }


def _train_torch_drg_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    tfit0 = time.time()
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    X_tr_s, X_va_s, X_te_s, scaler = _standardize_train_val_test(X_tr, X_va, X_te)
    device = _linear_device()
    xtr = torch.from_numpy(X_tr_s).to(device)
    xva = torch.from_numpy(X_va_s).to(device)
    xte = torch.from_numpy(X_te_s).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32, copy=False)).to(device)
    card_tr = torch.from_numpy(_drg_cardinality_target(y_tr)).to(device)
    card_va = _drg_cardinality_target(y_va)
    card_te = _drg_cardinality_target(y_te)
    n, d = xtr.shape
    l = int(ytr.shape[1])
    bs = int(max(32, int(cfg.get("shared_batch_size", 1024))))
    epochs = int(max(1, int(cfg.get("shared_epochs", 25))))
    patience = int(max(1, int(cfg.get("shared_patience", 3))))
    lr = float(cfg.get("shared_lr", cfg.get("lr", 1.0e-3)))
    rel_tol = float(cfg.get("objective_relative_tolerance", 0.0))
    grad_tol = float(cfg.get("gradient_norm_tolerance", -1.0))
    best: Dict[str, Any] = {"score": -np.inf}
    candidate_audit: List[Dict[str, Any]] = []
    for cand in _linear_candidate_grid(cfg):
        label_head = nn.Linear(int(d), int(l), bias=bool(cfg.get("intercept", True))).to(device)
        card_head = nn.Linear(int(d), 3, bias=bool(cfg.get("intercept", True))).to(device)
        if label_head.bias is not None:
            nn.init.zeros_(label_head.bias)
        if card_head.bias is not None:
            nn.init.zeros_(card_head.bias)
        opt_params = [
            {"params": [label_head.weight, card_head.weight], "weight_decay": float(cand["weight_decay"])},
        ]
        bias_params = []
        if label_head.bias is not None:
            bias_params.append(label_head.bias)
        if card_head.bias is not None:
            bias_params.append(card_head.bias)
        if bias_params:
            opt_params.append({"params": bias_params, "weight_decay": 0.0})
        opt = torch.optim.AdamW(opt_params, lr=lr)
        if str(cand["class_weight"]) == "balanced":
            pos = ytr.sum(dim=0)
            neg = ytr.shape[0] - pos
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 100.0)
        else:
            pos_weight = torch.ones((l,), dtype=torch.float32, device=device)
        curve: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, Any]] = None
        best_score = -np.inf
        best_epoch = -1
        bad = 0
        stop_reason = "max_epochs"
        for epoch in range(epochs):
            perm = torch.randperm(int(n), device=device)
            loss_sum = 0.0
            count = 0
            label_head.train()
            card_head.train()
            for s in range(0, int(n), bs):
                idx = perm[s:s + bs]
                xb = xtr.index_select(0, idx)
                label_logits = label_head(xb)
                card_logits = card_head(xb)
                loss = F.binary_cross_entropy_with_logits(label_logits, ytr.index_select(0, idx), pos_weight=pos_weight)
                loss = loss + F.cross_entropy(card_logits, card_tr.index_select(0, idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm_sq = 0.0
                for p in list(label_head.parameters()) + list(card_head.parameters()):
                    if p.grad is not None:
                        grad_norm_sq += float(p.grad.detach().pow(2).sum().item())
                opt.step()
                loss_sum += float(loss.detach().item()) * int(idx.numel())
                count += int(idx.numel())
            label_head.eval()
            card_head.eval()
            with torch.no_grad():
                va_label = torch.sigmoid(label_head(xva)).detach().cpu().numpy().astype(np.float32)
                va_card = torch.softmax(card_head(xva), dim=-1).detach().cpu().numpy().astype(np.float32)
            va_pred, va_card_pred = _drg_decode(va_label, va_card)
            va_metrics = _drg_set_metrics(y_va, va_pred, card_va, va_card_pred, va_label)
            score = float(va_metrics["micro_f1"])
            train_loss_epoch = float(loss_sum / max(count, 1))
            rel_change = 0.0 if not curve else abs(float(curve[-1]["train_loss"]) - train_loss_epoch) / max(abs(float(curve[-1]["train_loss"])), 1e-8)
            curve.append({
                "epoch": int(epoch),
                "train_loss": train_loss_epoch,
                "val_primary": float(score),
                "grad_norm": float(grad_norm_sq ** 0.5),
                "objective_relative_change": float(rel_change),
            })
            min_delta = float(cfg.get("min_delta", 0.0))
            if score > best_score + min_delta:
                best_score = float(score)
                best_epoch = int(epoch)
                best_state = {
                    "label": {k: v.detach().cpu().clone() for k, v in label_head.state_dict().items()},
                    "card": {k: v.detach().cpu().clone() for k, v in card_head.state_dict().items()},
                }
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    stop_reason = "validation_patience"
                    break
            if epoch > 0 and rel_tol > 0.0 and grad_tol >= 0.0 and rel_change <= rel_tol:
                full_grad_epoch = _compute_full_objective_gradient_norm(
                    model_type="drg",
                    models=[label_head, card_head],
                    x_train=xtr,
                    y_train=ytr,
                    card_train=card_tr,
                    pos_weight=pos_weight,
                    batch_size=bs,
                )
                if full_grad_epoch <= grad_tol:
                    stop_reason = "objective_converged"
                    break
        final_training_full_grad = _compute_full_objective_gradient_norm(
            model_type="drg",
            models=[label_head, card_head],
            x_train=xtr,
            y_train=ytr,
            card_train=card_tr,
            pos_weight=pos_weight,
            batch_size=bs,
        )
        if curve:
            curve[-1]["final_training_state_full_objective_gradient_norm"] = float(final_training_full_grad)
        if best_state is None or best_epoch < 0 or best_epoch >= len(curve):
            raise RuntimeError("DRG LR/drg did not produce a valid validation-selected checkpoint.")
        label_head.load_state_dict({k: v.to(device) for k, v in best_state["label"].items()})
        card_head.load_state_dict({k: v.to(device) for k, v in best_state["card"].items()})
        selected_full_grad = _compute_full_objective_gradient_norm(
            model_type="drg",
            models=[label_head, card_head],
            x_train=xtr,
            y_train=ytr,
            card_train=card_tr,
            pos_weight=pos_weight,
            batch_size=bs,
        )
        selected_rel_change = float(curve[int(best_epoch)]["objective_relative_change"])
        selected_converged = bool(
            rel_tol > 0.0
            and grad_tol >= 0.0
            and selected_rel_change <= rel_tol
            and selected_full_grad <= grad_tol
        )
        label_head.eval()
        card_head.eval()
        with torch.no_grad():
            va_label_final = torch.sigmoid(label_head(xva)).detach().cpu().numpy().astype(np.float32)
            va_card_final = torch.softmax(card_head(xva), dim=-1).detach().cpu().numpy().astype(np.float32)
        va_pred_final, va_card_pred_final = _drg_decode(va_label_final, va_card_final)
        selected_metrics = _drg_set_metrics(y_va, va_pred_final, card_va, va_card_pred_final, va_label_final)
        selected_score = float(selected_metrics["micro_f1"])
        if not np.isclose(selected_score, float(best_score), rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError(
                f"DRG LR/drg validation micro-F1 mismatch after restoring selected checkpoint: "
                f"selected_score={selected_score:.8g} recorded_best_score={float(best_score):.8g}"
            )
        candidate_audit.append({
            "weight_decay": float(cand["weight_decay"]),
            "class_weight": str(cand["class_weight"]),
            "best_epoch": int(best_epoch),
            "best_val_primary": float(best_score),
            "epochs_run": int(len(curve)),
            "final_epoch": int(curve[-1]["epoch"]) if curve else None,
            "converged": bool(selected_converged),
            "stop_reason": str(stop_reason),
            "final_training_state_objective_relative_change": float(curve[-1]["objective_relative_change"]) if curve else None,
            "final_training_state_full_objective_gradient_norm": float(final_training_full_grad),
            "selected_checkpoint_objective_relative_change": float(selected_rel_change),
            "selected_checkpoint_full_objective_gradient_norm": float(selected_full_grad),
            "last_minibatch_gradient_norm": float(curve[-1]["grad_norm"]) if curve else None,
            "relative_change_threshold": float(cfg.get("max_epoch_relative_change_fail_threshold", 1.0e-3)),
            "gradient_norm_threshold": float(cfg["max_epoch_gradient_norm_fail_threshold"]) if "max_epoch_gradient_norm_fail_threshold" in cfg else None,
        })
        _assert_selected_lr_checkpoint_converged(
            task_name="DRG LR/drg",
            stop_reason=stop_reason,
            epochs_run=len(curve),
            best_epoch=best_epoch,
            selected_objective_relative_change=selected_rel_change,
            selected_full_objective_gradient_norm=selected_full_grad,
            cfg=cfg,
        )
        if best_score > float(best.get("score", -np.inf)):
            best = {
                "score": float(best_score),
                "label_state_dict": {k: v.detach().cpu() for k, v in label_head.state_dict().items()},
                "card_state_dict": {k: v.detach().cpu() for k, v in card_head.state_dict().items()},
                "val_label_scores": va_label_final,
                "val_prediction": va_pred_final,
                "val_cardinality": va_card_pred_final,
                "best_epoch": int(best_epoch),
                "loss_curve": curve,
                "regularization": cand,
            }
    selected_label = nn.Linear(int(d), int(l), bias=bool(cfg.get("intercept", True))).to(device)
    selected_card = nn.Linear(int(d), 3, bias=bool(cfg.get("intercept", True))).to(device)
    selected_label.load_state_dict({k: v.to(device) for k, v in best["label_state_dict"].items()})
    selected_card.load_state_dict({k: v.to(device) for k, v in best["card_state_dict"].items()})
    selected_label.eval()
    selected_card.eval()
    t_test_pred = time.time()
    with torch.no_grad():
        te_label = torch.sigmoid(selected_label(xte)).detach().cpu().numpy().astype(np.float32)
        te_card = torch.softmax(selected_card(xte), dim=-1).detach().cpu().numpy().astype(np.float32)
    test_inference_sec = float(time.time() - t_test_pred)
    te_pred, te_card_pred = _drg_decode(te_label, te_card)
    te_metrics = _drg_set_metrics(y_te, te_pred, card_te, te_card_pred, te_label)
    best["metrics"] = te_metrics
    best["test_label_scores"] = te_label
    best["test_prediction"] = te_pred
    best["test_cardinality"] = te_card_pred
    best["test_inference_sec"] = test_inference_sec
    artifact = {
        "kind": "torch_cuda_drg_logistic_regression" if device.type == "cuda" else "torch_cpu_drg_logistic_regression",
        "model_family": "logistic_regression_linear_classifier",
        "decode_contract": "unordered_cardinality_topk",
        "input_dim": int(d),
        "num_labels": int(l),
        "label_state_dict": best["label_state_dict"],
        "cardinality_state_dict": best["card_state_dict"],
        "val_label_scores": best["val_label_scores"],
        "val_prediction": best["val_prediction"],
        "val_cardinality": best["val_cardinality"],
        "test_label_scores": best["test_label_scores"],
        "test_prediction": best["test_prediction"],
        "test_cardinality": best["test_cardinality"],
        "scaler": scaler,
        "convergence_audit": {
            "candidate_audit": candidate_audit,
            "loss_curve": best["loss_curve"],
            "best_epoch": int(best["best_epoch"]),
            "selected_regularization": best["regularization"],
            "device": str(device),
            "test_inference_sec": float(best["test_inference_sec"]),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        },
    }
    return dict(best["metrics"]), float(best["score"]), float(time.time() - tfit0), artifact


def _run_drg_competitive(
    baseline: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    lr_cfg: Dict[str, Any],
    xgb_cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    tfit0 = time.time()
    card_tr = _drg_cardinality_target(y_tr)
    card_va = _drg_cardinality_target(y_va)
    card_te = _drg_cardinality_target(y_te)

    if baseline == "lr":
        return _train_torch_drg_classifier(X_tr, y_tr, X_va, y_va, X_te, y_te, lr_cfg)
    else:
        XGBClassifier, xgb_info = _require_xgboost_cuda()
        capability_audit = _validate_xgboost_gpu_multioutput(XGBClassifier, xgb_cfg, y_tr.shape[1])
        best: Dict[str, Any] = {"score": -np.inf}
        joint_audit: List[Dict[str, Any]] = []
        candidate_grid = _xgb_candidate_grid(xgb_cfg)
        cardinality_heads: List[Dict[str, Any]] = []
        relevance_heads: List[Dict[str, Any]] = []
        for card_cand in candidate_grid:
            card_head = _fit_multiclass_xgb_candidate(XGBClassifier, X_tr, card_tr, X_va, card_va, xgb_cfg, card_cand, 3)
            card_head["candidate"] = dict(card_cand)
            cardinality_heads.append(card_head)
        for rel_cand in candidate_grid:
            rel_head = _fit_multilabel_xgb_candidate(XGBClassifier, X_tr, y_tr, X_va, y_va, xgb_cfg, rel_cand)
            rel_head["candidate"] = dict(rel_cand)
            relevance_heads.append(rel_head)
        for card_head in cardinality_heads:
            for rel_head in relevance_heads:
                label_va_prob = np.asarray(rel_head["val_probability"], dtype=np.float32)
                card_va_prob = np.asarray(card_head["val_probability"], dtype=np.float32)
                y_pred_va, card_pred_va = _drg_decode(label_va_prob, card_va_prob)
                val_m = _drg_set_metrics(y_va, y_pred_va, card_va, card_pred_va, label_va_prob)
                score = float(val_m["micro_f1"])
                joint_row = {
                    "cardinality_candidate": dict(card_head.get("candidate", {})),
                    "relevance_candidate": dict(rel_head.get("candidate", {})),
                    "cardinality_fit_sec": float(card_head.get("fit_sec", 0.0)),
                    "relevance_fit_sec": float(rel_head.get("fit_sec", 0.0)),
                    "validation_decoded_micro_f1": score,
                    "validation_metrics": {k: float(v) for k, v in val_m.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))},
                }
                joint_audit.append(joint_row)
                if score > float(best.get("score", -np.inf)):
                    best = {
                        "score": score,
                        "cardinality_head": card_head,
                        "relevance_head": rel_head,
                        "val_label_scores": label_va_prob,
                        "val_cardinality_probability": card_va_prob,
                        "val_prediction": y_pred_va,
                        "val_cardinality": card_pred_va,
                        "joint_candidate": joint_row,
                    }
        if float(best.get("score", -np.inf)) == -np.inf:
            raise RuntimeError("DRG XGBoost did not produce a valid validation candidate.")
        card_te_prob = _predict_multiclass_xgb_candidate(best["cardinality_head"], X_te, 3)
        label_te_prob = _xgb_prob(best["relevance_head"]["model"], X_te)
        y_pred_te, card_pred_te = _drg_decode(label_te_prob, card_te_prob)
        test_m = _drg_set_metrics(y_te, y_pred_te, card_te, card_pred_te, label_te_prob)
        artifact = {
            "kind": "drg_competitive_xgb",
            "xgboost_info": xgb_info,
            "xgboost_multioutput_capability_audit": capability_audit,
            "cardinality_head": best["cardinality_head"],
            "label_relevance_head": best["relevance_head"],
            "decode_contract": "unordered_cardinality_topk",
            "joint_candidate_audit": joint_audit,
            "selected_joint_candidate": best["joint_candidate"],
            "val_label_scores": best["val_label_scores"],
            "val_prediction": best["val_prediction"],
            "val_cardinality": best["val_cardinality"],
            "test_label_scores": label_te_prob,
            "test_prediction": y_pred_te,
            "test_cardinality": card_pred_te,
        }
        return test_m, float(best["score"]), float(time.time() - tfit0), artifact




def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _benchmark_code_signature() -> Dict[str, Any]:
    path = os.path.abspath(__file__)
    return {"path": path, "sha256": _sha256_file(path)}


def _library_runtime_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        out["cuda_device_count"] = int(torch.cuda.device_count())
        out["cuda_device_0"] = torch.cuda.get_device_name(0)
    try:
        import xgboost as xgb
        out["xgboost"] = str(getattr(xgb, "__version__", "unknown"))
        try:
            out["xgboost_build_info"] = xgb.build_info()
        except Exception as e:
            out["xgboost_build_info_error"] = str(e)
    except Exception as e:
        out["xgboost_import_error"] = str(e)
    return out


def _base_run_signature(args: argparse.Namespace, config: Mapping[str, Any], feature_meta: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "protocol_version": "benchmark_formal_internal_v1",
        "data_dir": os.path.abspath(str(args.data_dir)),
        "schema_dir": os.path.abspath(str(args.schema_dir)),
        "labels_csv": os.path.abspath(str(args.labels_csv)),
        "schema_sha256": _sha256_file(os.path.join(str(args.schema_dir), "schema_processed.csv")),
        "labels_sha256": _sha256_file(str(args.labels_csv)),
        "config_sha256": _sha256_file(str(args.config)),
        "config_signature": _json_sha256(config),
        "feature_contract_signature": str(feature_meta.get("feature_contract_signature")),
        "feature_contract": {
            "version": FEATURE_BANK_VERSION,
            "state_features_count": int(feature_meta.get("state_features_count", 0)),
            "schema_sha256": str(feature_meta.get("schema_sha256")),
            "train_data_signature": str(feature_meta.get("train_data_signature")),
        },
        "code": _benchmark_code_signature(),
        "runtime_libraries": _library_runtime_info(),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
    }


def _variant_signature(base_signature: Mapping[str, Any], *, task: str, baseline: str, cutoff: Optional[float], summary: str, primary_metric: str) -> Dict[str, Any]:
    payload = dict(base_signature)
    payload.update({
        "task": str(task),
        "baseline": str(baseline),
        "cutoff_min": None if cutoff is None else float(cutoff),
        "window": "full_visible" if cutoff is None else f"window_{int(round(cutoff))}",
        "summary": str(summary),
        "primary_metric": str(primary_metric),
    })
    payload["signature_sha256"] = _json_sha256(payload)
    return payload


def _save_ckpt(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


_PREDICTION_ARTIFACT_KEYS = {
    "val_probability",
    "test_probability",
    "val_label_scores",
    "val_prediction",
    "val_cardinality",
    "test_label_scores",
    "test_prediction",
    "test_cardinality",
}


def _strip_prediction_arrays_for_ckpt(artifact: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if artifact is None:
        return None
    out: Dict[str, Any] = {}
    for key, value in dict(artifact).items():
        if key in _PREDICTION_ARTIFACT_KEYS:
            continue
        out[key] = value
    return out


def _task_cache_kind_and_cutoff(task: str) -> Tuple[str, Optional[float]]:
    if task not in TASK_CACHE_KIND:
        raise KeyError(f"Task {task!r} is not in src.probes.TASK_CACHE_KIND")
    kind, cutoff = TASK_CACHE_KIND[task]
    return str(kind), (None if cutoff is None else float(cutoff))


def _task_cutoff_variants(task: str) -> List[Optional[float]]:
    _, cutoff = _task_cache_kind_and_cutoff(task)
    return [cutoff]


def _task_primary_metric_name(config: Dict[str, Any], task: str) -> str:
    task_cfg = _require_config_section(config, ["probe", "tasks", task])
    monitor = str(task_cfg.get("monitor", "val/auprc"))
    name = monitor.removeprefix("val/").removeprefix("test/")
    if task in SEQUENCE_BINARY_TASKS:
        return "auprc"
    if task == "hcup":
        return "macro_auprc"
    if task == "icd":
        return "micro_auprc"
    if task == "drg":
        return "micro_f1"
    return name


def _task_val_metric_name(config: Dict[str, Any], task: str) -> str:
    return "val/" + _task_primary_metric_name(config, task)




def _canonical_cache_root(args: argparse.Namespace) -> Optional[str]:
    root = getattr(args, "phase1_cache_root", None) or getattr(args, "canonical_cache_root", None)
    return None if root in (None, "") else str(root)


def _canonical_cache_ids(cache_root: str, split: str, task: str) -> List[str]:
    kind, cutoff = _task_cache_kind_and_cutoff(task)
    if kind == FULL_VISIBLE:
        ds = load_dataset(cache_root, FULL_VISIBLE, split=split)
    elif kind == RETROSPECTIVE_CUTOFF:
        ds = load_dataset(cache_root, RETROSPECTIVE_CUTOFF, split=split, cutoff_min=int(cutoff))
    else:
        raise RuntimeError(f"Unsupported canonical cache kind for benchmark audit: {kind}")
    return [str(x) for x in ds.admission_ids()]


def _assert_no_test_leakage(train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    tr, va, te = set(map(str, train_ids)), set(map(str, val_ids)), set(map(str, test_ids))
    if tr & va or tr & te or va & te:
        raise RuntimeError("Benchmark split leakage detected across train/val/test admission IDs.")


def _assert_canonical_cohort_parity(args: argparse.Namespace, task: str, split_to_ids: Dict[str, Sequence[str]]) -> Dict[str, Any]:
    root = _canonical_cache_root(args)
    if root is None:
        if not bool(getattr(args, "smoke", False)):
            raise RuntimeError("--phase1_cache_root is required for formal benchmark/canonical cache cohort parity audit.")
        return {"checked": False, "parity_mode": "unchecked", "reason": "--phase1_cache_root not provided"}
    smoke_mode = bool(getattr(args, "smoke", False))
    report: Dict[str, Any] = {
        "checked": True,
        "cache_root": os.path.abspath(root),
        "parity_mode": "ordered_subset" if smoke_mode else "exact_set",
        "splits": {},
    }
    for split, ids in split_to_ids.items():
        ours_ordered = [str(x) for x in ids]
        try:
            canonical_ordered = _canonical_cache_ids(root, split, task)
        except RuntimeError as exc:
            if smoke_mode:
                report["checked"] = False
                report["parity_mode"] = "unchecked"
                report["reason"] = f"canonical cache unavailable in smoke mode: {exc}"
                report["splits"][split] = {
                    "benchmark_n": int(len(ours_ordered)),
                    "parity_mode": "unchecked",
                }
                continue
            raise
        if smoke_mode:
            if len(set(ours_ordered)) != len(ours_ordered) or not _is_ordered_subset(ours_ordered, canonical_ordered):
                canonical_set = set(canonical_ordered)
                extra = [x for x in ours_ordered if x not in canonical_set][:10]
                raise RuntimeError(
                    f"Benchmark/canonical cache ordered-subset parity failed for task={task} split={split}: "
                    f"smoke_n={len(ours_ordered)} canonical_n={len(canonical_ordered)} extra_or_unordered={extra}"
                )
            report["splits"][split] = {
                "canonical_n": int(len(canonical_ordered)),
                "smoke_n": int(len(ours_ordered)),
                "smoke_id_sha256": _sha256_id_list(ours_ordered),
                "parity_mode": "ordered_subset",
            }
            continue
        ours = set(ours_ordered)
        canonical = set(canonical_ordered)
        if ours != canonical:
            missing = sorted(canonical - ours)[:10]
            extra = sorted(ours - canonical)[:10]
            raise RuntimeError(
                f"Benchmark/canonical cache ID parity failed for task={task} split={split}: "
                f"benchmark_n={len(ours)} canonical_n={len(canonical)} missing_from_benchmark={missing} extra_in_benchmark={extra}"
            )
        report["splits"][split] = {"n": int(len(ours)), "parity_mode": "exact_set"}
    return report


def _variant_dir(task_dir: str, variant_key: str) -> str:
    return os.path.join(task_dir, variant_key)


def _variant_ready(task_dir: str, variant_key: str, expected_signature: Optional[Mapping[str, Any]] = None) -> bool:
    d = _variant_dir(task_dir, variant_key)
    complete = os.path.join(d, ".complete")
    sig_path = os.path.join(d, "run_signature.json")
    required = [
        complete,
        sig_path,
        os.path.join(d, "model.ckpt"),
        os.path.join(d, "metrics.json"),
        os.path.join(d, "summary.json"),
        os.path.join(d, "val_predictions.npz"),
        os.path.join(d, "test_predictions.npz"),
    ]
    if not all(os.path.exists(p) for p in required):
        return False
    if expected_signature is not None:
        found = _load_json(sig_path)
        if str(found.get("signature_sha256")) != str(expected_signature.get("signature_sha256")):
            raise RuntimeError(
                f"Benchmark variant signature mismatch for {d}; remove the stale variant directory or use a fresh output_dir."
            )
    return True


def _load_variant_rep(
    *,
    task_dir: str,
    variant_key: str,
    task: str,
    cutoff_min: Optional[float],
    summary: str,
    config: Dict[str, Any],
    labels_df: pd.DataFrame,
) -> Dict[str, Any]:
    d = _variant_dir(task_dir, variant_key)
    summary_path = os.path.join(d, "summary.json")
    if os.path.exists(summary_path):
        rep = _load_json(summary_path)
        if isinstance(rep, dict):
            return rep

    metrics = _load_json(os.path.join(d, "metrics.json"))
    runtime_path = os.path.join(d, "runtime.json")
    meta_path = os.path.join(d, "meta.json")
    runtime = _load_json(runtime_path) if os.path.exists(runtime_path) else {}
    meta = _load_json(meta_path) if os.path.exists(meta_path) else {}

    train_n = int((labels_df["split"].astype(str) == "train").sum())
    val_n = int((labels_df["split"].astype(str) == "val").sum())
    num_labels = 1
    if task in SEQUENCE_MULTILABEL_TASKS:
        pp = _require_config_section(config, ["probe", "tasks", task])
        num_labels = int(pp.get("max_codes", 26 if task == "hcup" else 1))

    rep = {
        "variant_key": str(variant_key),
        "cutoff_min": (None if cutoff_min is None else float(cutoff_min)),
        "source": str(summary),
        "val_metric_name": str(meta.get("val_metric_name", _task_val_metric_name(config, task))),
        "val_metric": float(
            meta.get(
                "val_metric",
                metrics.get(
                    str(meta.get("val_metric_name", _task_val_metric_name(config, task))),
                    metrics.get("val/auprc", metrics.get("val/micro_f1", metrics.get("val/micro_auprc", np.nan))),
                ),
            )
        ),
        "aux_monitor": float(metrics.get("val/exact_set_match")) if "val/exact_set_match" in metrics else None,
        "metrics": metrics,
        "runtime": runtime,
        "input_dim": int(meta.get("input_dim", 0)),
        "n_train": int(train_n),
        "n_val": int(val_n),
        "n_test": int(meta.get("n_test", 0)),
        "num_labels": int(meta.get("num_labels", num_labels)),
        "best_ckpt_path": str(meta.get("best_ckpt_path", os.path.join(d, "model.ckpt"))),
        "val_predictions_path": str(meta.get("val_predictions_path", os.path.join(d, "val_predictions.npz"))),
        "test_predictions_path": str(meta.get("test_predictions_path", os.path.join(d, "test_predictions.npz"))),
        "ids_path": str(meta.get("ids_path", os.path.join(d, "split_ids.npz"))),
        "run_signature_sha256": str(meta.get("run_signature_sha256", "")),
    }
    return rep


def _build_task_summary_from_variants(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    task: str,
    baseline: str,
    labels_df: pd.DataFrame,
    task_dir: str,
    cutoff_variants: Sequence[Optional[float]],
    summaries_for_task: Sequence[str],
) -> Dict[str, Any]:
    all_variant_results: Dict[str, Any] = {}
    best_val = -np.inf
    best_variant_key = None
    val_metric_name = _task_val_metric_name(config, task)

    for cutoff in cutoff_variants:
        cutoff_tag = "full_visible" if cutoff is None else f"window_{int(round(cutoff))}"
        for summary in summaries_for_task:
            variant_key = f"{cutoff_tag}_{summary}"
            try:
                rep = _load_variant_rep(
                    task_dir=task_dir,
                    variant_key=variant_key,
                    task=task,
                    cutoff_min=cutoff,
                    summary=summary,
                    config=config,
                    labels_df=labels_df,
                )
            except FileNotFoundError:
                raise RuntimeError(f"Missing benchmark variant artifact for {task}/{baseline}/{variant_key}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid benchmark variant JSON for {task}/{baseline}/{variant_key}")
            except KeyError:
                raise RuntimeError(f"Invalid benchmark variant payload for {task}/{baseline}/{variant_key}")
            all_variant_results[variant_key] = rep
            val_score = float(rep.get("val_metric", np.nan))
            if float(val_score) > best_val:
                best_val = float(val_score)
                best_variant_key = variant_key

    vocab_audit = None
    if task in SEQUENCE_MULTILABEL_TASKS:
        ml_spec = _build_multilabel_task_spec(config, args.labels_csv, task)
        vocab_audit = {
            "label_col": ml_spec.label_col,
            "min_code_count": int(ml_spec.min_code_count),
            "max_codes": ml_spec.max_codes,
            "include_other_bucket": bool(ml_spec.include_other_bucket),
            "num_labels": int(len(ml_spec.codes)),
            "vocab_hash": ml_spec.vocab_hash,
        }

    verdict = "FAIR_COMPARABLE"
    if task == "icd":
        verdict = "FAIR_COMPARABLE_STANDARD_MULTILABEL_BASELINE"
    elif task == "hcup":
        verdict = "FAIR_COMPARABLE_HCUP_FIXED_26_BASELINE"
    elif task == "drg":
        verdict = "FAIR_COMPARABLE_COMPETITIVE_SET_BASELINE"

    out = {
        "task": task,
        "baseline": baseline,
        "state_only": True,
        "uses_interventions": False,
        "uses_atlas_ehr_latents": False,
        "split_source": "labels_csv.split",
        "cutoff_variants": [("full_visible" if c is None else f"window_{int(round(c))}") for c in cutoff_variants],
        "vocab_audit": vocab_audit,
        "input_representation": {
            "value_space": "standardized",
            "missing_strategy": "zero_imputation",
            "uses_presence": False,
            "inverse_transform": False,
        },
        "variant_results": all_variant_results,
        "best_variant_key": str(best_variant_key) if best_variant_key else None,
        "best_variant_ckpt_path": (
            str(all_variant_results[str(best_variant_key)]["best_ckpt_path"])
            if (best_variant_key is not None and str(best_variant_key) in all_variant_results)
            else None
        ),
        "best_val_metric_name": val_metric_name,
        "best_val_metric": float(best_val),
        "comparability_verdict": verdict,
        "runtime": {
            "total_sec": float(0.0),
        },
    }
    _save_json(os.path.join(task_dir, "summary.json"), out)
    return out


def _run_baseline_for_task(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    task: str,
    baseline: str,
    labels_df: pd.DataFrame,
    num_features: int,
    state_vid_to_dense: Dict[int, int],
    lr_cfg: Dict[str, Any],
    xgb_cfg: Dict[str, Any],
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    is_binary = task in SEQUENCE_BINARY_TASKS
    is_multilabel = task in SEQUENCE_MULTILABEL_TASKS

    cutoff_variants = _task_cutoff_variants(task)

    ml_spec = _build_multilabel_task_spec(config, args.labels_csv, task) if is_multilabel else None

    # --- Determine val metric name ---
    val_metric_name = _task_val_metric_name(config, task)

    task_dir = _task_base_dir(args.output_dir, task, baseline)
    os.makedirs(task_dir, exist_ok=True)

    def _write_summary_artifacts(summary_name: str, rep: Dict[str, Any]) -> None:
        summary_dir = os.path.join(task_dir, summary_name)
        os.makedirs(summary_dir, exist_ok=True)
        _save_json(os.path.join(summary_dir, "summary.json"), rep)
        metrics_payload = dict(rep.get("metrics", {}))
        _save_json(os.path.join(summary_dir, "metrics.json"), metrics_payload)
        _save_json(os.path.join(summary_dir, "runtime.json"), rep.get("runtime", {}))
        if isinstance(rep.get("run_signature"), dict):
            _save_json(os.path.join(summary_dir, "run_signature.json"), rep["run_signature"])
        _save_json(os.path.join(summary_dir, "meta.json"), {
            "task": task,
            "baseline": baseline,
            "summary": summary_name,
            "val_metric_name": rep.get("val_metric_name"),
            "val_metric": rep.get("val_metric"),
            "input_dim": rep.get("input_dim"),
            "num_labels": rep.get("num_labels"),
            "n_train": rep.get("n_train"),
            "n_val": rep.get("n_val"),
            "n_test": rep.get("n_test"),
            "best_ckpt_path": rep.get("best_ckpt_path"),
            "val_predictions_path": rep.get("val_predictions_path"),
            "test_predictions_path": rep.get("test_predictions_path"),
            "ids_path": rep.get("ids_path"),
            "run_signature_sha256": rep.get("run_signature_sha256"),
        })
        with open(os.path.join(summary_dir, ".complete"), "w") as f:
            f.write("complete\n")

    t0_total = time.time()
    all_variant_results: Dict[str, Any] = {}
    best_val = -np.inf
    best_variant_key = None

    summaries_for_task = _select_summaries(task, args.summaries)
    for cutoff in cutoff_variants:
        runtime_cfg = {
            "runtime": {
                "labels_csv": args.labels_csv,
                "smoke": bool(args.smoke),
                "smoke_ids_by_task_split": getattr(args, "_benchmark_smoke_ids_by_task_split", {}) or {},
            },
            "probe": config.get("probe", {}),
        }

        eval_only = bool(getattr(args, "eval_existing_models_only", False))
        tr_ds = None if eval_only else SequenceClassificationDataset(
            args.cache_dir, "train", task, labels_df, runtime_cfg, ml_spec, cutoff_min=cutoff)
        va_ds = None if eval_only else SequenceClassificationDataset(
            args.cache_dir, "val", task, labels_df, runtime_cfg, ml_spec, cutoff_min=cutoff)
        te_ds = SequenceClassificationDataset(
            args.cache_dir, "test", task, labels_df, runtime_cfg, ml_spec, cutoff_min=cutoff)
        if not eval_only:
            _assert_no_test_leakage(tr_ds.admission_ids(), va_ds.admission_ids(), te_ds.admission_ids())
        cohort_audit = _assert_canonical_cohort_parity(
            args,
            task,
            {"test": te_ds.admission_ids()} if eval_only else {"train": tr_ds.admission_ids(), "val": va_ds.admission_ids(), "test": te_ds.admission_ids()},
        )
        tr_ids = np.asarray([] if eval_only else tr_ds.admission_ids(), dtype=str)
        va_ids = np.asarray([] if eval_only else va_ds.admission_ids(), dtype=str)
        te_ids = np.asarray(te_ds.admission_ids(), dtype=str)

        cutoff_tag = "full_visible" if cutoff is None else f"window_{int(round(cutoff))}"
        runtime_budget = ((getattr(args, "_benchmark_cfg", {}) or {}).get("runtime", {}) or {})
        max_matrix_gib = float(runtime_budget.get("max_training_matrix_gib", 64.0))

        for summary in summaries_for_task:
            variant_key = f"{cutoff_tag}_{summary}"
            tfit = time.time()
            ckpt_path = os.path.join(task_dir, variant_key, "model.ckpt")
            expected_signature = _variant_signature(
                getattr(args, "_benchmark_base_signature", {}),
                task=task,
                baseline=baseline,
                cutoff=cutoff,
                summary=summary,
                primary_metric=_task_primary_metric_name(config, task),
            )
            reused = False if eval_only else _variant_ready(task_dir, variant_key, expected_signature)

            if reused:
                rep = _load_variant_rep(
                    task_dir=task_dir,
                    variant_key=variant_key,
                    task=task,
                    cutoff_min=cutoff,
                    summary=summary,
                    config=config,
                    labels_df=labels_df,
                )
                test_metrics = dict(rep.get("test_metrics") or {k.removeprefix("test/").removeprefix("val/"): v for k, v in rep.get("metrics", {}).items()})
                val_score = float(rep.get("val_metric", np.nan))
                fit_sec = float(rep.get("runtime", {}).get("fit_sec", 0.0))
                eval_sec = float(rep.get("runtime", {}).get("eval_sec", 0.0))
                aux_monitor = float(rep.get("aux_monitor")) if rep.get("aux_monitor") is not None else np.nan
                ckpt_payload = None
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix({"task": task, "baseline": baseline, "variant": variant_key, "mode": "reuse"}, refresh=False)
            elif bool(getattr(args, "eval_existing_models_only", False)):
                prior_summary_path = os.path.join(task_dir, variant_key, "summary.json")
                if not os.path.isfile(prior_summary_path):
                    raise FileNotFoundError(f"eval-existing benchmark requires prior variant summary: {prior_summary_path}")
                prior_summary = _load_json(prior_summary_path)
                artifact = _load_existing_benchmark_artifact_for_eval(
                    ckpt_path=ckpt_path,
                    task=task,
                    baseline=baseline,
                    cutoff_tag=cutoff_tag,
                    summary=summary,
                    feature_contract_signature=str(getattr(args, "_benchmark_base_signature", {}).get("feature_contract_signature")),
                )
                X_te, te_y, te_t = build_tabular_feature_summary(
                    te_ds, num_features, args.cache_dir, "test", task, state_vid_to_dense, cutoff, summary,
                    max_matrix_gib=max_matrix_gib,
                )
                cache_sec = float(te_t["cache_sec"])
                feature_sec = float(te_t["feature_build_sec"])
                t_eval = time.time()
                test_metrics, prediction_payload = _eval_existing_benchmark_artifact(
                    task=task,
                    baseline=baseline,
                    artifact=artifact,
                    X_te=X_te,
                    y_te=te_y,
                )
                eval_sec = float(time.time() - t_eval)
                val_score = float(prior_summary.get("val_metric", np.nan))
                if not np.isfinite(val_score):
                    raise RuntimeError(f"eval-existing benchmark prior summary lacks finite val_metric: {prior_summary_path}")
                fit_sec = 0.0
                aux_monitor = float(test_metrics.get("exact_set_match", np.nan)) if task == "drg" else np.nan
                ckpt_payload = {
                    "kind": str(artifact.get("kind", "eval_existing_artifact")),
                    "eval_existing_models_only": True,
                    **prediction_payload,
                }
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix({"task": task, "baseline": baseline, "variant": variant_key, "mode": "eval_existing"}, refresh=False)
                del X_te
            else:
                X_tr, tr_y, tr_t = build_tabular_feature_summary(
                    tr_ds, num_features, args.cache_dir, "train", task, state_vid_to_dense, cutoff, summary,
                    max_matrix_gib=max_matrix_gib,
                )
                X_va, va_y, va_t = build_tabular_feature_summary(
                    va_ds, num_features, args.cache_dir, "val", task, state_vid_to_dense, cutoff, summary,
                    max_matrix_gib=max_matrix_gib,
                )
                X_te, te_y, te_t = build_tabular_feature_summary(
                    te_ds, num_features, args.cache_dir, "test", task, state_vid_to_dense, cutoff, summary,
                    max_matrix_gib=max_matrix_gib,
                )
                cache_sec = tr_t["cache_sec"] + va_t["cache_sec"] + te_t["cache_sec"]
                feature_sec = tr_t["feature_build_sec"] + va_t["feature_build_sec"] + te_t["feature_build_sec"]
                estimated_ram_gib = float(X_tr.nbytes + X_va.nbytes + X_te.nbytes) / float(1024 ** 3)
                print(
                    f"[benchmark_train_matrix] task={task} baseline={baseline} variant={variant_key} "
                    f"train/val/test_shapes={X_tr.shape}/{X_va.shape}/{X_te.shape} estimated_ram_gib={estimated_ram_gib:.3f}",
                    flush=True,
                )
                if task == "drg":
                    test_metrics, val_score, fit_sec, ckpt_payload = _run_drg_competitive(
                        baseline=baseline,
                        X_tr=X_tr, y_tr=tr_y,
                        X_va=X_va, y_va=va_y,
                        X_te=X_te, y_te=te_y,
                        lr_cfg=lr_cfg, xgb_cfg=xgb_cfg,
                    )
                    eval_sec = 0.0
                    aux_monitor = float(test_metrics.get("exact_set_match", np.nan))
                elif baseline == "lr":
                    test_metrics, val_score, fit_sec, ckpt_payload = train_lr(
                        X_tr=X_tr, y_tr=tr_y, X_va=X_va, y_va=va_y, X_te=X_te, y_te=te_y,
                        cfg=lr_cfg, is_binary=is_binary, task_name=task,
                    )
                    eval_sec = 0.0
                    aux_monitor = np.nan
                else:  # xgb, non-drg
                    test_metrics, val_score, fit_sec, ckpt_payload = train_xgb(
                        X_tr=X_tr, y_tr=tr_y, X_va=X_va, y_va=va_y, X_te=X_te, y_te=te_y,
                        cfg=xgb_cfg, is_binary=is_binary, task_name=task,
                    )
                    eval_sec = 0.0
                    aux_monitor = np.nan
                _save_ckpt(ckpt_path, {
                    "task": str(task),
                    "baseline": str(baseline),
                    "variant_key": str(variant_key),
                    "cutoff_min": (None if cutoff is None else float(cutoff)),
                    "summary_source": str(summary),
                    "run_signature": expected_signature,
                    "artifact": _strip_prediction_arrays_for_ckpt(ckpt_payload),
                })
                pred_dir = os.path.join(task_dir, variant_key)
                os.makedirs(pred_dir, exist_ok=True)
                ids_path = os.path.join(pred_dir, "split_ids.npz")
                np.savez_compressed(ids_path, train_hadm_id=tr_ids, val_hadm_id=va_ids, test_hadm_id=te_ids)
                val_pred_path = os.path.join(pred_dir, "val_predictions.npz")
                pred_path = os.path.join(pred_dir, "test_predictions.npz")
                if task == "drg":
                    np.savez_compressed(
                        val_pred_path,
                        hadm_id=va_ids,
                        target=va_y.astype(np.float32, copy=False),
                        label_scores=np.asarray(ckpt_payload.get("val_label_scores"), dtype=np.float32),
                        prediction=np.asarray(ckpt_payload.get("val_prediction"), dtype=np.float32),
                        predicted_cardinality=np.asarray(ckpt_payload.get("val_cardinality"), dtype=np.int64),
                    )
                    np.savez_compressed(
                        pred_path,
                        hadm_id=te_ids,
                        target=te_y.astype(np.float32, copy=False),
                        label_scores=np.asarray(ckpt_payload.get("test_label_scores"), dtype=np.float32),
                        prediction=np.asarray(ckpt_payload.get("test_prediction"), dtype=np.float32),
                        predicted_cardinality=np.asarray(ckpt_payload.get("test_cardinality"), dtype=np.int64),
                    )
                else:
                    np.savez_compressed(
                        val_pred_path,
                        hadm_id=va_ids,
                        target=va_y.astype(np.float32, copy=False),
                        probability=np.asarray(ckpt_payload.get("val_probability"), dtype=np.float32),
                    )
                    np.savez_compressed(
                        pred_path,
                        hadm_id=te_ids,
                        target=te_y.astype(np.float32, copy=False),
                        probability=np.asarray(ckpt_payload.get("test_probability"), dtype=np.float32),
                    )
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix({"task": task, "baseline": baseline, "variant": variant_key, "mode": "train"}, refresh=False)
                del X_tr, X_va, X_te
            runtime = {
                "cache_sec": float(cache_sec) if not reused else 0.0,
                "feature_build_sec": float(feature_sec) if not reused else 0.0,
                "fit_sec": float(fit_sec),
                "eval_sec": float(eval_sec),
                "total_sec": float(time.time() - tfit),
            }
            rep = {
                "variant_key": variant_key,
                "cutoff_min": (None if cutoff is None else float(cutoff)),
                "source": str(summary),
                "val_metric_name": val_metric_name,
                "val_metric": float(val_score),
                "aux_monitor": (None if not np.isfinite(aux_monitor) else float(aux_monitor)),
                "metrics": {f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))},
                "test_metrics": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))},
                "runtime": runtime,
                "input_dim": int(num_features),
                "n_train": int(prior_summary.get("n_train", 0)) if bool(getattr(args, "eval_existing_models_only", False)) else int(len(tr_ds)),
                "n_val": int(prior_summary.get("n_val", 0)) if bool(getattr(args, "eval_existing_models_only", False)) else int(len(va_ds)),
                "n_test": int(len(te_ds)),
                "ids_path": os.path.join(task_dir, variant_key, "split_ids.npz"),
                "ordered_test_hadm_id_sha256": hashlib.sha256("\n".join(te_ids.tolist()).encode("utf-8")).hexdigest(),
                "val_predictions_path": os.path.join(task_dir, variant_key, "val_predictions.npz"),
                "test_predictions_path": os.path.join(task_dir, variant_key, "test_predictions.npz"),
                "cohort_audit": cohort_audit,
                "num_labels": (1 if ml_spec is None else int(len(ml_spec.codes))),
                "best_ckpt_path": ckpt_path,
                "run_signature": expected_signature,
                "run_signature_sha256": expected_signature["signature_sha256"],
                "eval_existing_models_only": bool(getattr(args, "eval_existing_models_only", False)),
            }
            all_variant_results[variant_key] = rep
            _write_summary_artifacts(variant_key, rep)

            if float(val_score) > best_val:
                best_val = float(val_score)
                best_variant_key = variant_key

    vocab_audit = None
    if ml_spec is not None:
        vocab_audit = {
            "label_col": ml_spec.label_col,
            "min_code_count": int(ml_spec.min_code_count),
            "max_codes": ml_spec.max_codes,
            "include_other_bucket": bool(ml_spec.include_other_bucket),
            "num_labels": int(len(ml_spec.codes)),
            "vocab_hash": ml_spec.vocab_hash,
        }

    verdict = "FAIR_COMPARABLE"
    if task == "icd":
        verdict = "FAIR_COMPARABLE_STANDARD_MULTILABEL_BASELINE"
    elif task == "hcup":
        verdict = "FAIR_COMPARABLE_HCUP_FIXED_26_BASELINE"
    elif task == "drg":
        verdict = "FAIR_COMPARABLE_COMPETITIVE_SET_BASELINE"

    out = {
        "task": task,
        "baseline": baseline,
        "state_only": True,
        "uses_interventions": False,
        "uses_atlas_ehr_latents": False,
        "split_source": "labels_csv.split",
        "cutoff_variants": [("full_visible" if c is None else f"window_{int(round(c))}") for c in cutoff_variants],
        "vocab_audit": vocab_audit,
        "input_representation": {
            "value_space": "standardized",
            "missing_strategy": "zero_imputation",
            "uses_presence": False,
            "inverse_transform": False,
        },
        "variant_results": all_variant_results,
        "best_variant_key": str(best_variant_key) if best_variant_key else None,
        "best_variant_ckpt_path": (
            str(all_variant_results[str(best_variant_key)]["best_ckpt_path"])
            if (best_variant_key is not None and str(best_variant_key) in all_variant_results)
            else None
        ),
        "best_val_metric_name": val_metric_name,
        "best_val_metric": float(best_val),
        "comparability_verdict": verdict,
        "runtime": {
            "total_sec": float(time.time() - t0_total),
        },
    }
    _save_json(os.path.join(task_dir, "summary.json"), out)
    return out


def _task_all_variants_ready(
    task_dir: str,
    cutoff_variants: Sequence[Optional[float]],
    summaries_for_task: Sequence[str],
    *,
    base_signature: Optional[Mapping[str, Any]] = None,
    task: Optional[str] = None,
    baseline: Optional[str] = None,
    primary_metric: Optional[str] = None,
) -> bool:
    for cutoff in cutoff_variants:
        cutoff_tag = "full_visible" if cutoff is None else f"window_{int(round(cutoff))}"
        for summary in summaries_for_task:
            expected = None
            if base_signature is not None and task is not None and baseline is not None and primary_metric is not None:
                expected = _variant_signature(
                    base_signature,
                    task=task,
                    baseline=baseline,
                    cutoff=cutoff,
                    summary=summary,
                    primary_metric=primary_metric,
                )
            if not _variant_ready(task_dir, f"{cutoff_tag}_{summary}", expected):
                return False
    return True


def _write_formal_benchmark_outputs(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
    baselines: Sequence[str],
    summaries_by_task: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, str]:
    rows: List[Dict[str, Any]] = []
    paper_rows: List[Dict[str, Any]] = []
    full: Dict[str, Any] = {
        "protocol": "formal_internal_benchmark",
        "selection_rule": "best validation primary metric selects the paper summary; test is evaluated once after selection.",
        "post_hoc_fields": ["post_hoc_descriptive_test_oracle_max_primary"],
        "tasks": {},
    }
    seen = set()
    for task in tasks:
        full["tasks"][task] = {}
        primary = _task_primary_metric_name(load_config_dict(args.config), task)
        for baseline in baselines:
            summary_path = os.path.join(args.output_dir, "benchmark", task, baseline, "summary.json")
            if not os.path.exists(summary_path):
                raise RuntimeError(f"Missing benchmark task summary: {summary_path}")
            rep = _load_json(summary_path)
            variants = dict(rep.get("variant_results") or {})
            if not variants:
                raise RuntimeError(f"Benchmark task summary has no variants: {summary_path}")
            full["tasks"][task][baseline] = rep
            best_key = str(rep.get("best_variant_key"))
            test_primary_values: List[float] = []
            for variant_key, v in variants.items():
                key = (task, baseline, variant_key)
                if key in seen:
                    raise RuntimeError(f"Duplicate benchmark variant row: {key}")
                seen.add(key)
                test_metrics = dict(v.get("test_metrics") or {})
                if primary not in test_metrics or not np.isfinite(float(test_metrics[primary])):
                    raise RuntimeError(f"Missing/NaN test primary metric {primary} for {task}/{baseline}/{variant_key}")
                if not np.isfinite(float(v.get("val_metric"))):
                    raise RuntimeError(f"Missing/NaN validation primary metric for {task}/{baseline}/{variant_key}")
                test_primary = float(test_metrics[primary])
                test_primary_values.append(test_primary)
                row = {
                    "task": task,
                    "baseline": baseline,
                    "variant_key": variant_key,
                    "window": "full_visible" if v.get("cutoff_min") is None else f"window_{int(round(float(v['cutoff_min'])))}",
                    "summary": str(v.get("source")),
                    "primary_metric": primary,
                    "val_primary": float(v.get("val_metric")),
                    "test_primary": test_primary,
                    "is_best_val_selected": bool(variant_key == best_key),
                    "n_train": int(v.get("n_train", 0)),
                    "n_val": int(v.get("n_val", 0)),
                    "n_test": int(v.get("n_test", 0)),
                    "model_path": str(v.get("best_ckpt_path")),
                    "val_predictions_path": str(v.get("val_predictions_path")),
                    "test_predictions_path": str(v.get("test_predictions_path")),
                    "ids_path": str(v.get("ids_path")),
                    "run_signature_sha256": str(v.get("run_signature_sha256")),
                }
                eval_existing = bool(getattr(args, "eval_existing_models_only", False))
                for path_key in ("model_path", "val_predictions_path", "test_predictions_path", "ids_path"):
                    if eval_existing and path_key in ("val_predictions_path", "test_predictions_path", "ids_path"):
                        continue
                    if not os.path.exists(row[path_key]):
                        raise RuntimeError(f"Benchmark variant is incomplete; missing {path_key}: {row[path_key]}")
                rows.append(row)
            best = variants.get(best_key)
            if not isinstance(best, dict):
                raise RuntimeError(f"best_variant_key {best_key!r} not found for {task}/{baseline}")
            best_test = float(dict(best.get("test_metrics") or {})[primary])
            paper_rows.append({
                "task": task,
                "baseline": baseline,
                "selected_by": "validation_primary_metric",
                "best_val_selected_variant": best_key,
                "primary_metric": primary,
                "best_val_primary": float(best.get("val_metric")),
                "best_val_selected_test_primary": best_test,
                "post_hoc_descriptive_test_oracle_max_primary": float(max(test_primary_values)),
                "n_test": int(best.get("n_test", 0)),
                "test_predictions_path": str(best.get("test_predictions_path")),
            })
    if summaries_by_task is not None:
        expected_variants = sum(len(tuple(summaries_by_task[str(task)])) * len(baselines) for task in tasks)
        all_summary_mode = all(tuple(summaries_by_task[str(task)]) == TABULAR_SUMMARIES for task in tasks)
        if set(tasks) == set(SEQUENCE_TASK_ORDER) and set(baselines) == CLASSIFICATION_BASELINES and all_summary_mode and len(rows) != 84:
            raise RuntimeError(f"Formal all-summary benchmark expected 84 variants, observed {len(rows)}.")
        if len(rows) != expected_variants:
            raise RuntimeError(f"Benchmark expected {expected_variants} requested variants, observed {len(rows)}.")
    compact_path = os.path.join(args.output_dir, "benchmark_metrics_compact.csv")
    paper_path = os.path.join(args.output_dir, "benchmark_paper_ready.csv")
    full_path = os.path.join(args.output_dir, "benchmark_full_results.json")
    pd.DataFrame(rows).to_csv(compact_path, index=False)
    pd.DataFrame(paper_rows).to_csv(paper_path, index=False)
    _save_json(full_path, full)
    return {
        "compact_csv": compact_path,
        "paper_ready_csv": paper_path,
        "full_json": full_path,
    }


def _delete_eval_only_prediction_npz(
    *,
    output_dir: str,
    tasks: Sequence[str],
    baselines: Sequence[str],
    summaries_by_task: Mapping[str, Sequence[str]],
) -> List[str]:
    removed: List[str] = []
    for task in tasks:
        cutoff = _task_cutoff_variants(str(task))[0]
        cutoff_tag = "full_visible" if cutoff is None else f"window_{int(round(float(cutoff)))}"
        for baseline in baselines:
            for summary in summaries_by_task[str(task)]:
                variant_dir = os.path.join(str(output_dir), "benchmark", str(task), str(baseline), f"{cutoff_tag}_{summary}")
                for name in ("val_predictions.npz", "test_predictions.npz"):
                    path = os.path.join(variant_dir, name)
                    if os.path.exists(path):
                        os.remove(path)
                        removed.append(path)
    return removed


def _external_output_root(output_dir: str, mode: str) -> str:
    return os.path.join(str(output_dir), "external_benchmark", str(mode))


def _external_tasks_from_args(task: str) -> List[str]:
    if str(task) == "all":
        return list(EXTERNAL_BINARY_TASKS)
    if str(task) not in EXTERNAL_BINARY_TASKS:
        raise ValueError(f"eICU external benchmark supports only {list(EXTERNAL_BINARY_TASKS)}, got {task!r}")
    return [str(task)]


def _run_external_reuse_mimic(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    labels_df: pd.DataFrame,
    tasks: Sequence[str],
    baselines: Sequence[str],
    num_features: int,
) -> Dict[str, Any]:
    if not args.mimic_benchmark_output_dir:
        raise ValueError("--mimic-benchmark-output-dir is required for --external-mode reuse_mimic")
    if not args.mimic_feature_cache_dir:
        raise ValueError("--mimic-feature-cache-dir is required for --external-mode reuse_mimic")
    mimic_feature_meta = _load_feature_contract(args.mimic_feature_cache_dir)
    mimic_feature_sig = str(mimic_feature_meta.get("feature_contract_signature"))
    mode_root = _external_output_root(args.output_dir, "reuse_mimic")
    os.makedirs(mode_root, exist_ok=True)
    runtime_cfg = {"runtime": {"labels_csv": args.labels_csv, "smoke": bool(args.smoke)}, "probe": config.get("probe", {})}
    results: Dict[str, Any] = {"mode": "reuse_mimic", "tasks": {}}
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        cutoff = _task_cutoff_variants(task)[0]
        cutoff_tag = "full_visible" if cutoff is None else f"window_{int(round(cutoff))}"
        te_ds = SequenceClassificationDataset(args.cache_dir, "test", task, labels_df, runtime_cfg, None, cutoff_min=cutoff)
        cohort_audit = _assert_canonical_cohort_parity(args, task, {"test": te_ds.admission_ids()})
        y_te = te_ds.labels_numpy()
        te_ids = np.asarray(te_ds.admission_ids(), dtype=str)
        results["tasks"][task] = {}
        for baseline in baselines:
            mimic_task_dir = os.path.join(str(args.mimic_benchmark_output_dir), "benchmark", task, baseline)
            mimic_summary_path = os.path.join(mimic_task_dir, "summary.json")
            if not os.path.isfile(mimic_summary_path):
                raise FileNotFoundError(f"Missing MIMIC benchmark summary for reuse: {mimic_summary_path}")
            mimic_summary = _load_json(mimic_summary_path)
            mimic_best_key = str(mimic_summary.get("best_variant_key"))
            if not mimic_best_key or mimic_best_key == "None":
                raise RuntimeError(f"MIMIC benchmark summary lacks best_variant_key for reuse: {mimic_summary_path}")
            task_dir = os.path.join(mode_root, "benchmark", task, baseline)
            os.makedirs(task_dir, exist_ok=True)
            variant_results: Dict[str, Any] = {}
            for summary in TABULAR_SUMMARIES:
                variant_key = f"{cutoff_tag}_{summary}"
                ckpt_path = os.path.join(mimic_task_dir, variant_key, "model.ckpt")
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError(f"Missing MIMIC benchmark model for reuse: {ckpt_path}")
                with open(ckpt_path, "rb") as f:
                    ckpt = pickle.load(f)
                artifact = ckpt.get("artifact")
                if not isinstance(artifact, Mapping):
                    raise RuntimeError(f"Invalid MIMIC benchmark model artifact: {ckpt_path}")
                run_sig = ckpt.get("run_signature")
                if not isinstance(run_sig, Mapping):
                    raise RuntimeError(f"MIMIC benchmark model lacks run_signature: {ckpt_path}")
                if str(run_sig.get("feature_contract_signature")) != mimic_feature_sig:
                    raise RuntimeError(f"MIMIC model feature-contract SHA does not match --mimic-feature-cache-dir: {ckpt_path}")
                if str(run_sig.get("task")) != str(task) or str(run_sig.get("baseline")) != str(baseline):
                    raise RuntimeError(f"MIMIC model task/baseline mismatch for reuse: {ckpt_path}")
                if str(run_sig.get("summary")) != str(summary) or str(run_sig.get("window")) != str(cutoff_tag):
                    raise RuntimeError(f"MIMIC model summary/window mismatch for reuse: {ckpt_path}")
                if int(artifact.get("input_dim", num_features)) != int(num_features):
                    raise RuntimeError(f"MIMIC model input_dim does not match eICU feature contract for reuse: {ckpt_path}")
                X_te = te_ds.feature_matrix(summary)
                prob = _predict_binary_from_artifact(artifact, X_te)
                metrics = compute_binary_metrics(y_te.astype(np.float32, copy=False), prob)
                pred_dir = os.path.join(task_dir, variant_key)
                os.makedirs(pred_dir, exist_ok=True)
                pred_path = os.path.join(pred_dir, "test_predictions.npz")
                np.savez_compressed(
                    pred_path,
                    hadm_id=te_ids,
                    target=y_te.astype(np.float32, copy=False),
                    probability=prob.astype(np.float32, copy=False),
                )
                rep = {
                    "task": task,
                    "baseline": baseline,
                    "variant_key": variant_key,
                    "cutoff_min": None if cutoff is None else float(cutoff),
                    "source": summary,
                    "external_mode": "reuse_mimic",
                    "selection_source": "mimic_validation",
                    "is_mimic_val_selected": bool(variant_key == mimic_best_key),
                    "val_metric_name": mimic_summary.get("best_val_metric_name"),
                    "val_metric": None,
                    "metrics": {f"test/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))},
                    "test_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))},
                    "n_train": 0,
                    "n_val": 0,
                    "n_test": int(y_te.shape[0]),
                    "input_dim": int(num_features),
                    "num_labels": 1,
                    "cohort_audit": cohort_audit,
                    "mimic_model_path": ckpt_path,
                    "test_predictions_path": pred_path,
                    "ordered_test_hadm_id_sha256": hashlib.sha256("\n".join(te_ids.tolist()).encode("utf-8")).hexdigest(),
                }
                _save_json(os.path.join(pred_dir, "summary.json"), rep)
                _save_json(os.path.join(pred_dir, "metrics.json"), rep["metrics"])
                with open(os.path.join(pred_dir, ".complete"), "w") as f:
                    f.write("complete\n")
                variant_results[variant_key] = rep
                rows.append({
                    "external_mode": "reuse_mimic",
                    "task": task,
                    "baseline": baseline,
                    "variant_key": variant_key,
                    "summary": summary,
                    "primary_metric": _task_primary_metric_name(config, task),
                    "test_primary": float(metrics[_task_primary_metric_name(config, task)]),
                    "is_best_summary": bool(variant_key == mimic_best_key),
                    "n_test": int(y_te.shape[0]),
                    "test_predictions_path": pred_path,
                })
            if mimic_best_key not in variant_results:
                raise RuntimeError(f"MIMIC best summary {mimic_best_key!r} was not evaluated in eICU reuse for {task}/{baseline}.")
            task_summary = {
                "task": task,
                "baseline": baseline,
                "external_mode": "reuse_mimic",
                "best_variant_key": mimic_best_key,
                "best_summary_selection": "inherited_from_mimic_validation",
                "variant_results": variant_results,
            }
            _save_json(os.path.join(task_dir, "summary.json"), task_summary)
            results["tasks"][task][baseline] = task_summary
    compact = os.path.join(mode_root, "external_benchmark_metrics_compact.csv")
    pd.DataFrame(rows).to_csv(compact, index=False)
    full_path = os.path.join(mode_root, "external_benchmark_metrics_full.json")
    _save_json(full_path, results)
    results["compact_csv"] = compact
    results["full_json"] = full_path
    return results


def _run_external_benchmark(args: argparse.Namespace, config: Dict[str, Any], labels_df: pd.DataFrame) -> Dict[str, Any]:
    if args.external_mode not in {"reuse_mimic", "retrain_eicu"}:
        raise ValueError(f"Unsupported external mode: {args.external_mode}")
    _assert_eicu_admission_level_processed(args.data_dir)
    _assert_external_cache_contract(args.phase1_cache_root)
    tasks = _external_tasks_from_args(args.task)
    baselines = [str(x) for x in args.baselines]
    if not args.mimic_feature_cache_dir:
        raise ValueError("--mimic-feature-cache-dir is required for eICU benchmark so MIMIC VID contract is reused.")
    mimic_meta = _load_feature_contract(args.mimic_feature_cache_dir)
    meta = _install_feature_contract(args.cache_dir, mimic_meta, source=os.path.abspath(str(args.mimic_feature_cache_dir)))
    args._benchmark_cfg = _require_config_section(config, ["benchmark"])
    args._benchmark_base_signature = _base_run_signature(args, config, meta)
    state_vid_to_dense = {int(k): int(v) for k, v in meta["state_vid_to_dense"].items()}
    _ensure_state_only_cache(
        args,
        labels_df,
        splits=(["test"] if bool(getattr(args, "eval_existing_models_only", False)) else None),
    )
    if args.external_mode == "reuse_mimic":
        return _run_external_reuse_mimic(
            args=args,
            config=config,
            labels_df=labels_df,
            tasks=tasks,
            baselines=baselines,
            num_features=int(meta["state_features_count"]),
        )

    mode_root = _external_output_root(args.output_dir, "retrain_eicu")
    ext_args = argparse.Namespace(**vars(args))
    ext_args.output_dir = mode_root
    lr_cfg = _require_config_section(config, ["benchmark", "linear"])
    xgb_cfg = _require_config_section(config, ["benchmark", "xgboost"])
    aggregate = {"mode": "retrain_eicu", "tasks": {}}
    summaries_for_request = {task: _select_summaries(task, ext_args.summaries) for task in tasks}
    progress = tqdm(
        total=sum(len(summaries_for_request[task]) * len(baselines) for task in tasks),
        desc="eicu_benchmark_retrain",
        ascii=True,
    )
    for task in tasks:
        aggregate["tasks"][task] = {}
        for baseline in baselines:
            rep = _run_baseline_for_task(
                args=ext_args,
                config=config,
                task=task,
                baseline=baseline,
                labels_df=labels_df,
                num_features=int(meta["state_features_count"]),
                state_vid_to_dense=state_vid_to_dense,
                lr_cfg=lr_cfg,
                xgb_cfg=xgb_cfg,
                progress=progress,
            )
            aggregate["tasks"][task][baseline] = rep
    progress.close()
    aggregate["formal_outputs"] = _write_formal_benchmark_outputs(args=ext_args, tasks=tasks, baselines=baselines, summaries_by_task=summaries_for_request)
    _save_json(os.path.join(mode_root, "external_benchmark_metrics_full.json"), aggregate)
    return aggregate


def _resolve_paths(args: argparse.Namespace) -> None:
    if args.data_dir and args.labels_csv and args.schema_dir and args.cache_dir and args.output_dir:
        return
    # smoke-friendly defaults
    root = os.getcwd()
    data = os.path.join(root, "sample_data")
    if not args.data_dir:
        args.data_dir = data
    if not args.labels_csv:
        args.labels_csv = os.path.join(data, "labels.csv")
    if not args.schema_dir:
        args.schema_dir = data
    if not args.cache_dir:
        args.cache_dir = os.path.join(root, "outputs", "cache")
    if not args.output_dir:
        args.output_dir = os.path.join(root, "outputs")


def _apply_benchmark_smoke_overrides(config: Dict[str, Any]) -> None:
    bench = config.setdefault("benchmark", {})
    runtime = bench.setdefault("runtime", {})
    runtime["smoke_max_patients_per_split"] = int(min(int(runtime.get("smoke_max_patients_per_split", 32)), 32))
    runtime["shard_size"] = int(min(int(runtime.get("shard_size", 256)), 32))
    runtime["feature_batch_size"] = int(min(int(runtime.get("feature_batch_size", 32)), 16))
    linear = bench.setdefault("linear", {})
    linear["shared_epochs"] = int(min(int(linear.get("shared_epochs", 200)), 3))
    linear["shared_patience"] = int(min(int(linear.get("shared_patience", 20)), 2))
    linear["weight_decay_candidates"] = list(linear.get("weight_decay_candidates", [0.0]))[:1]
    linear["class_weight_candidates"] = list(linear.get("class_weight_candidates", ["none"]))[:1]
    linear["fail_on_unconverged_max_epochs"] = False
    xgb = bench.setdefault("xgboost", {})
    xgb["n_estimators"] = int(min(int(xgb.get("n_estimators", 200)), 5))
    xgb["n_estimators_candidates"] = [int(xgb["n_estimators"])]
    xgb["max_depth_candidates"] = list(xgb.get("max_depth_candidates", [2]))[:1]
    xgb["learning_rate_candidates"] = list(xgb.get("learning_rate_candidates", [0.1]))[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description="State-only benchmark runner (LR/XGB only)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--schema_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--phase1_cache_root", type=str, default=None,
                        help="Canonical Phase-1 probe cache root used only for admission-ID cohort parity audit.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "mortality", "los_ge_7d", "los_ge_3d", "readmission", "icd", "hcup", "drg"])
    parser.add_argument("--baselines", nargs="+", default=["lr", "xgb"])
    parser.add_argument("--summaries", nargs="+", default=None,
                        choices=["last", "mean", "max", "presence", "count", "interval"])
    parser.add_argument("--benchmark-workers", type=int, default=1,
                        help="Parallel task/baseline workers for benchmark training. Use 1 for serial execution.")
    parser.add_argument("--eval_existing_models_only", action="store_true",
                        help="Load existing benchmark model.ckpt files and recompute test metrics without fitting any model.")
    parser.add_argument("--external-mode", type=str, default="none", choices=["none", "reuse_mimic", "retrain_eicu"],
                        help="Run eICU admission-level external benchmark mode instead of internal MIMIC benchmark.")
    parser.add_argument("--mimic-benchmark-output-dir", type=str, default=None,
                        help="MIMIC formal benchmark output_dir containing benchmark/<task>/<baseline> artifacts for reuse_mimic.")
    parser.add_argument("--mimic-feature-cache-dir", type=str, default=None,
                        help="MIMIC benchmark cache_dir containing the frozen feature contract for eICU.")
    parser.add_argument("--hcup_mapping_artifact", type=str, default=None,
                        help="Override probe.tasks.hcup.mapping_artifact with an explicit 26-label HCUP mapping JSON path.")

    args = parser.parse_args()
    _resolve_paths(args)
    os.makedirs(os.path.join(args.output_dir, "benchmark"), exist_ok=True)
    _write_run_manifest(
        args.output_dir,
        {
            "started_at_unix": float(time.time()),
            "config": os.path.abspath(str(args.config)),
            "task": str(args.task),
            "baselines": [str(x) for x in args.baselines],
            "summaries": None if args.summaries is None else [str(x) for x in args.summaries],
            "smoke": bool(args.smoke),
            "cache_dir": os.path.abspath(str(args.cache_dir)),
            "phase1_cache_root": None if args.phase1_cache_root is None else os.path.abspath(str(args.phase1_cache_root)),
            "labels_csv": os.path.abspath(str(args.labels_csv)),
            "external_mode": str(args.external_mode),
            "mimic_benchmark_output_dir": None if args.mimic_benchmark_output_dir is None else os.path.abspath(str(args.mimic_benchmark_output_dir)),
            "mimic_feature_cache_dir": None if args.mimic_feature_cache_dir is None else os.path.abspath(str(args.mimic_feature_cache_dir)),
            "hcup_mapping_artifact": None if args.hcup_mapping_artifact is None else os.path.abspath(str(args.hcup_mapping_artifact)),
        },
    )
    set_seed(int(args.seed))

    config = load_config_dict(args.config)
    config["_config_dir"] = os.path.dirname(os.path.abspath(str(args.config)))
    if args.hcup_mapping_artifact not in (None, ""):
        _require_config_section(config, ["probe", "tasks", "hcup"])["mapping_artifact"] = os.path.abspath(str(args.hcup_mapping_artifact))
    if args.smoke:
        _apply_benchmark_smoke_overrides(config)

    illegal = [b for b in args.baselines if b not in CLASSIFICATION_BASELINES]
    if illegal:
        raise ValueError(
            f"Unsupported baseline(s): {illegal}. Allowed baselines are exactly {sorted(CLASSIFICATION_BASELINES)}. "
            "GRU-D has been removed from benchmark."
        )

    _ensure_no_inverse_transform_usage(args.schema_dir)
    args._benchmark_dataset_policy = _benchmark_dataset_policy(config)
    if args.external_mode != "none":
        args.labels_csv = _canonicalize_external_labels_for_benchmark(args)
    labels_df = _load_labels_df(args.labels_csv)
    if not bool(args.smoke):
        _assert_patient_level_split(labels_df, require_patient_id=True)
    if args.external_mode != "none":
        result = _run_external_benchmark(args, config, labels_df)
        print(json.dumps({"external_benchmark": result}, indent=2, default=str))
        return

    if _require_config_section(config, ["probe", "tasks", "drg"]).get("max_codes", None) is None:
        raise ValueError("probe.tasks.drg.max_codes must be explicitly set (non-null) for benchmark/probes alignment.")
    if _require_config_section(config, ["probe", "tasks", "icd"]).get("max_codes", None) is None:
        raise ValueError("probe.tasks.icd.max_codes must be explicitly set (non-null) for benchmark/probes alignment.")
    # HCUP uses a strict mapping artifact; no implicit label order is allowed.

    args._benchmark_cfg = _require_config_section(config, ["benchmark"])
    tasks = list(SEQUENCE_TASK_ORDER) if args.task == "all" else [str(args.task)]
    tasks = [t for t in tasks if t in SEQUENCE_TASKS]
    if bool(args.smoke):
        smoke_n = int(((args._benchmark_cfg.get("runtime", {}) or {}).get("smoke_max_patients_per_split", 32)))
        by_task_split, by_split = _build_smoke_ids_from_canonical(args, tasks, smoke_n)
        args._benchmark_smoke_ids_by_task_split = by_task_split
        args._benchmark_smoke_ids_by_split = by_split
    else:
        args._benchmark_smoke_ids_by_task_split = {}
        args._benchmark_smoke_ids_by_split = {}

    meta = _build_state_only_metadata(args, labels_df)
    args._benchmark_base_signature = _base_run_signature(args, config, meta)
    state_vid_to_dense = {int(k): int(v) for k, v in meta["state_vid_to_dense"].items()}
    _ensure_state_only_cache(
        args,
        labels_df,
        splits=(["test"] if bool(getattr(args, "eval_existing_models_only", False)) else None),
    )
    if args.cache_only:
        print("[CACHE-ONLY] benchmark feature bank ready.")
        return

    lr_cfg = _require_config_section(config, ["benchmark", "linear"])
    xgb_cfg = _require_config_section(config, ["benchmark", "xgboost"])

    summaries_for_request: Dict[str, Tuple[str, ...]] = {task: _select_summaries(task, args.summaries) for task in tasks}
    cutoff_map: Dict[str, List[Optional[float]]] = {task: _task_cutoff_variants(task) for task in tasks}
    benchmark_workers = max(1, int(args.benchmark_workers))
    if torch.cuda.is_available() and benchmark_workers > 1:
        print("[benchmark] CUDA available; forcing serial baseline execution in this process. Use separate CUDA_VISIBLE_DEVICES processes for multi-GPU.")
        benchmark_workers = 1
    if "xgb" in [str(b) for b in args.baselines] and benchmark_workers > 1:
        print("[benchmark] xgboost selected; forcing serial execution. Multi-GPU XGBoost must use separate processes with CUDA_VISIBLE_DEVICES.")
        benchmark_workers = 1
    parallel_mode = benchmark_workers > 1
    if parallel_mode:
        _warm_benchmark_tabular_cache(
            args=args,
            config=config,
            labels_df=labels_df,
            tasks=tasks,
            cutoff_map=cutoff_map,
            state_vid_to_dense=state_vid_to_dense,
            num_features=int(meta["state_features_count"]),
        )
        xgb_cfg = dict(xgb_cfg)
        cpu_count = int(os.cpu_count() or 1)
        default_parallel_n_jobs = max(1, cpu_count // benchmark_workers)
        requested_n_jobs = int(xgb_cfg.get("n_jobs", default_parallel_n_jobs))
        xgb_cfg["n_jobs"] = max(1, min(requested_n_jobs if requested_n_jobs > 0 else default_parallel_n_jobs, default_parallel_n_jobs))
        print(f"[benchmark] parallel workers={benchmark_workers}, xgb_n_jobs={xgb_cfg['n_jobs']}")

    total_steps = 0
    for task in tasks:
        cutoffs = cutoff_map[task]
        n_summaries = len(summaries_for_request[task])
        total_steps += len(args.baselines) * len(cutoffs) * n_summaries

    aggregate = {"tasks": {}}
    progress = tqdm(total=total_steps, desc="benchmark", ascii=True)
    task_plan: Dict[Tuple[str, str], str] = {}
    for task in tasks:
        for baseline in args.baselines:
            task_dir = _task_base_dir(args.output_dir, task, baseline)
            if bool(getattr(args, "eval_existing_models_only", False)):
                task_plan[(task, baseline)] = "run"
            elif _task_all_variants_ready(
                task_dir,
                cutoff_map[task],
                summaries_for_request[task],
                base_signature=args._benchmark_base_signature,
                task=task,
                baseline=baseline,
                primary_metric=_task_primary_metric_name(config, task),
            ):
                task_plan[(task, baseline)] = "reuse"
            else:
                task_plan[(task, baseline)] = "run"

    if all(mode == "reuse" for mode in task_plan.values()):
        for task in tasks:
            aggregate["tasks"][task] = {}
            for baseline in args.baselines:
                task_dir = _task_base_dir(args.output_dir, task, baseline)
                rep = _build_task_summary_from_variants(
                    args=args,
                    config=config,
                    task=task,
                    baseline=baseline,
                    labels_df=labels_df,
                    task_dir=task_dir,
                    cutoff_variants=cutoff_map[task],
                    summaries_for_task=summaries_for_request[task],
                )
                out = os.path.join(args.output_dir, "benchmark", task, baseline, "summary.json")
                _save_json(out, rep)
                aggregate["tasks"][task][baseline] = out
        os.makedirs(args.output_dir, exist_ok=True)
        agg_path = os.path.join(args.output_dir, "benchmark_classification_results.json")
        aggregate["formal_outputs"] = _write_formal_benchmark_outputs(args=args, tasks=tasks, baselines=args.baselines, summaries_by_task=summaries_for_request)
        if bool(getattr(args, "eval_existing_models_only", False)):
            aggregate["eval_existing_deleted_prediction_npz"] = _delete_eval_only_prediction_npz(
                output_dir=args.output_dir,
                tasks=tasks,
                baselines=args.baselines,
                summaries_by_task=summaries_for_request,
            )
        _save_json(agg_path, aggregate)
        print(f"[DONE] Reused existing benchmark outputs and refreshed summary: {agg_path}")
        return

    if not parallel_mode:
        for task in tasks:
            aggregate["tasks"][task] = {}
            for baseline in args.baselines:
                if task_plan[(task, baseline)] == "reuse":
                    rep = _build_task_summary_from_variants(
                        args=args,
                        config=config,
                        task=task,
                        baseline=baseline,
                        labels_df=labels_df,
                        task_dir=_task_base_dir(args.output_dir, task, baseline),
                        cutoff_variants=cutoff_map[task],
                        summaries_for_task=summaries_for_request[task],
                    )
                else:
                    _run_baseline_for_task(
                        args=args,
                        config=config,
                        task=task,
                        baseline=baseline,
                        labels_df=labels_df,
                        num_features=int(meta["state_features_count"]),
                        state_vid_to_dense=state_vid_to_dense,
                        lr_cfg=lr_cfg,
                        xgb_cfg=xgb_cfg,
                        progress=progress,
                    )
                    rep = _build_task_summary_from_variants(
                        args=args,
                        config=config,
                        task=task,
                        baseline=baseline,
                        labels_df=labels_df,
                        task_dir=_task_base_dir(args.output_dir, task, baseline),
                        cutoff_variants=cutoff_map[task],
                        summaries_for_task=summaries_for_request[task],
                    )
                out = os.path.join(args.output_dir, "benchmark", task, baseline, "summary.json")
                _save_json(out, rep)
                aggregate["tasks"][task][baseline] = out
    else:
        pending_jobs: Dict[Any, Tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=benchmark_workers) as pool:
            for task in tasks:
                aggregate["tasks"][task] = {}
                for baseline in args.baselines:
                    if task_plan[(task, baseline)] == "reuse":
                        rep = _build_task_summary_from_variants(
                            args=args,
                            config=config,
                            task=task,
                            baseline=baseline,
                            labels_df=labels_df,
                            task_dir=_task_base_dir(args.output_dir, task, baseline),
                            cutoff_variants=cutoff_map[task],
                            summaries_for_task=summaries_for_request[task],
                        )
                        out = os.path.join(args.output_dir, "benchmark", task, baseline, "summary.json")
                        _save_json(out, rep)
                        aggregate["tasks"][task][baseline] = out
                        progress.update(1)
                    else:
                        fut = pool.submit(
                            _run_baseline_for_task,
                            args=args,
                            config=config,
                            task=task,
                            baseline=baseline,
                            labels_df=labels_df,
                            num_features=int(meta["state_features_count"]),
                            state_vid_to_dense=state_vid_to_dense,
                            lr_cfg=lr_cfg,
                            xgb_cfg=xgb_cfg,
                            progress=None,
                        )
                        pending_jobs[fut] = (task, baseline)
            for fut in as_completed(pending_jobs):
                task, baseline = pending_jobs[fut]
                _ = fut.result()
                out = os.path.join(args.output_dir, "benchmark", task, baseline, "summary.json")
                rep = _build_task_summary_from_variants(
                    args=args,
                    config=config,
                    task=task,
                    baseline=baseline,
                    labels_df=labels_df,
                    task_dir=_task_base_dir(args.output_dir, task, baseline),
                    cutoff_variants=cutoff_map[task],
                    summaries_for_task=summaries_for_request[task],
                )
                _save_json(out, rep)
                aggregate["tasks"][task][baseline] = out
                progress.update(1)

    progress.close()
    os.makedirs(args.output_dir, exist_ok=True)
    agg_path = os.path.join(args.output_dir, "benchmark_classification_results.json")
    aggregate["formal_outputs"] = _write_formal_benchmark_outputs(args=args, tasks=tasks, baselines=args.baselines, summaries_by_task=summaries_for_request)
    if bool(getattr(args, "eval_existing_models_only", False)):
        aggregate["eval_existing_deleted_prediction_npz"] = _delete_eval_only_prediction_npz(
            output_dir=args.output_dir,
            tasks=tasks,
            baselines=args.baselines,
            summaries_by_task=summaries_for_request,
        )
    _save_json(agg_path, aggregate)
    print(f"[DONE] Wrote benchmark report: {agg_path}")


if __name__ == "__main__":
    main()
