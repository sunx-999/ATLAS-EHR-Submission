#!/usr/bin/env python3
"""Canonical eICU external validation for the Phase-1 exact-time probes.

The external path deliberately consumes the same cache, probe and metric
contracts as MIMIC.  It has no eICU-specific encoder or task model.
"""

from __future__ import annotations

# Keep the CLI observable even when a first-time import of a scientific
# dependency is building a user-level cache before ``main()`` is reached.
if __name__ == "__main__":
    print("[external] starting CLI imports", flush=True)

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# These project modules are intentionally imported lazily.  In particular,
# importing ``src.eval`` may initialize Matplotlib's font cache, which can be
# slow on a fresh machine and otherwise makes the CLI appear hung before its
# first progress message.
FULL_VISIBLE = None
TASK_CACHE_KIND = None
build_phase1_probe_cache = None
load_dataset = None
_sha256 = None
evaluate_exact_time_probe = None
load_exact_time_probe = None
AttentionCostBatchSampler = None
run_probe_phase = None
JEPAModel = None
load_config_dict = None
compute_binary_metrics = None


def _ensure_runtime_imports() -> None:
    global FULL_VISIBLE, TASK_CACHE_KIND, build_phase1_probe_cache, load_dataset
    global _sha256, evaluate_exact_time_probe, load_exact_time_probe
    global AttentionCostBatchSampler, run_probe_phase, JEPAModel, load_config_dict, compute_binary_metrics
    if FULL_VISIBLE is not None:
        return
    from src.cache_bank import FULL_VISIBLE as _full_visible, build_phase1_probe_cache as _build_cache, load_dataset as _load_dataset
    from src.eval import _sha256 as _hash, evaluate_exact_time_probe as _evaluate, load_exact_time_probe as _load_probe
    from src.probes import AttentionCostBatchSampler as _sampler, TASK_CACHE_KIND as _task_cache_kind, run_probe_phase as _run_probe
    from src.model import JEPAModel as _jepa_model
    from src.utils import load_config_dict as _load_config
    from src.utils import compute_binary_metrics as _compute_binary_metrics
    FULL_VISIBLE = _full_visible
    TASK_CACHE_KIND = _task_cache_kind
    build_phase1_probe_cache = _build_cache
    load_dataset = _load_dataset
    _sha256 = _hash
    evaluate_exact_time_probe = _evaluate
    load_exact_time_probe = _load_probe
    AttentionCostBatchSampler = _sampler
    run_probe_phase = _run_probe
    JEPAModel = _jepa_model
    load_config_dict = _load_config
    compute_binary_metrics = _compute_binary_metrics


EXTERNAL_TASKS = ("mortality", "los_ge_3d", "los_ge_7d")
EXTERNAL_SOURCES = ("z_set", "z_context")
EXTERNAL_MODES = ("reuse_pretrained", "retrain_probe")


class _IndexedExternalDataset:
    """A zero-copy view over one canonical cache stage for one endpoint."""

    def __init__(self, base: Any, indices: Sequence[int]):
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def sequence_record(self, index: int, source_name: str) -> Mapping[str, Any]:
        return self.base.sequence_record(self.indices[index], source_name)

    def admission_ids(self) -> list[str]:
        ids = self.base.admission_ids()
        return [str(ids[i]) for i in self.indices]

    def sequence_lengths(self) -> list[int]:
        lengths = self.base.sequence_lengths()
        return [int(lengths[i]) for i in self.indices]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _task_target(labels: pd.DataFrame, task: str) -> Tuple[np.ndarray, str]:
    """Return only the hospital-level endpoint used by the MIMIC contract."""
    if task == "mortality":
        column = "label_mortality"
        if column not in labels:
            raise RuntimeError("canonical external labels require label_mortality from hospital_mortality.")
        mortality = pd.to_numeric(labels[column], errors="coerce").to_numpy(dtype=np.float32)
        return mortality, "hospital_mortality"
    if task in {"los_ge_3d", "los_ge_7d"}:
        column = "label_los_days"
        if column not in labels:
            raise RuntimeError("canonical external labels require label_los_days from hospital_los.")
        threshold = 3 if task == "los_ge_3d" else 7
        los = pd.to_numeric(labels[column], errors="coerce").to_numpy(dtype=np.float64)
        out = np.full(los.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(los)
        out[valid] = (los[valid] >= threshold).astype(np.float32)
        return out, "hospital_los"
    raise ValueError(f"Unsupported external task: {task}")


def _formal_task_target(labels: pd.DataFrame, task: str) -> Tuple[np.ndarray, str]:
    return _task_target(labels, task)


def _formal_assert_patient_split_leakage(labels: pd.DataFrame) -> None:
    if "split" not in labels:
        raise RuntimeError("External labels require a split column.")
    for column in ("subject_id", "hadm_id"):
        if column in labels and labels.groupby(column)["split"].nunique().gt(1).any():
            raise RuntimeError("patient split leakage detected")
    if labels["hadm_id"].astype(str).duplicated().any():
        raise RuntimeError("duplicate hadm_id in external labels")


def _expected_ids(labels: pd.DataFrame, split: str) -> list[str]:
    return labels.loc[labels["split"].astype(str) == str(split), "hadm_id"].astype(str).tolist()


def _prediction_alignment_and_merge(*, ids: Sequence[str], y: np.ndarray, labels_df: pd.DataFrame,
                                    task: str, split: str, logits: Optional[np.ndarray] = None,
                                    probabilities: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    ids = [str(x) for x in ids]
    expected = _expected_ids(labels_df, split)
    duplicate = sorted({x for x in ids if ids.count(x) > 1})
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(set(ids) - set(expected))
    alignment = {"aligned": not duplicate and not missing and not unexpected and len(ids) == len(expected),
                 "duplicate_prediction_ids": duplicate, "missing_prediction_ids": missing,
                 "unexpected_prediction_ids": unexpected, "expected_count": len(expected),
                 "prediction_count": len(ids)}
    if len(ids) != len(y):
        raise RuntimeError("Prediction ID and target counts differ.")
    frame = pd.DataFrame({"hadm_id": ids, "target_from_probe": np.asarray(y, dtype=np.float32)})
    if logits is not None:
        frame["raw_logit"] = np.asarray(logits, dtype=np.float32)
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=np.float32)
        if logits is not None and not np.allclose(p, 1.0 / (1.0 + np.exp(-np.asarray(logits))), atol=1e-2):
            raise RuntimeError("Probability does not match sigmoid(raw_logit).")
        frame["probability"] = p
    labels = labels_df.loc[labels_df["split"].astype(str) == str(split)].copy()
    labels["hadm_id"] = labels["hadm_id"].astype(str)
    target, _ = _task_target(labels, task)
    labels["target_from_labels"] = target
    if duplicate:
        return alignment, frame
    return alignment, frame.merge(labels[["hadm_id", "target_from_labels"]], on="hadm_id", how="left", validate="one_to_one")


def _validate_prediction_alignment(**kwargs: Any) -> Tuple[Dict[str, Any], pd.DataFrame]:
    alignment, merged = _prediction_alignment_and_merge(**kwargs)
    if not alignment["aligned"]:
        raise RuntimeError(f"Strict prediction/target alignment failed: {alignment}")
    if not np.allclose(merged["target_from_probe"], merged["target_from_labels"]):
        raise RuntimeError("Probe targets do not match canonical hospital labels.")
    return alignment, merged


def build_canonical_external_labels(labels_csv: str, output_csv: str, audit_path: Optional[str] = None) -> Dict[str, Any]:
    labels = pd.read_csv(labels_csv)
    required = {"subject_id", "hadm_id", "split", "hospital_mortality", "hospital_los"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise RuntimeError(f"External labels lack formal hospital endpoints: {missing}")
    _formal_assert_patient_split_leakage(labels)
    mortality = pd.to_numeric(labels["hospital_mortality"], errors="coerce")
    los = pd.to_numeric(labels["hospital_los"], errors="coerce")
    invalid_mortality = mortality.notna() & ~mortality.isin([0, 1])
    invalid_los = los.notna() & (~np.isfinite(los) | (los < 0))
    if invalid_mortality.any():
        details = {
            "invalid_mortality_count": int(invalid_mortality.sum()),
            "invalid_mortality_examples": labels.loc[invalid_mortality, ["hadm_id", "hospital_mortality"]].head(10).to_dict("records"),
            "invalid_los_count": int(invalid_los.sum()),
            "invalid_los_examples": labels.loc[invalid_los, ["hadm_id", "hospital_los"]].head(10).to_dict("records"),
        }
        raise RuntimeError(f"Invalid hospital mortality or hospital LOS endpoint: {details}")
    # Negative/non-finite hospital LOS is an invalid endpoint, not a negative
    # class.  Exclude it as missing while preserving the latent cache.
    los = los.mask(invalid_los)
    frame = pd.DataFrame({
        "subject_id": labels["subject_id"].astype(str), "hadm_id": labels["hadm_id"].astype(str),
        "split": labels["split"].astype(str), "label_mortality": mortality,
        "label_los_days": los, "source_hospital_mortality": labels["hospital_mortality"],
        "source_hospital_los": labels["hospital_los"],
    })
    if "patienthealthsystemstayid" in labels.columns:
        frame["patienthealthsystemstayid"] = labels["patienthealthsystemstayid"]
    frame["label_mortality"] = frame["label_mortality"].astype("Int64")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    frame.to_csv(output_csv, index=False)
    summary: Dict[str, Any] = {"endpoint_protocol": "hospital_admission_level_external_validation",
                               "invalid_hospital_los_excluded_count": int(invalid_los.sum()),
                               "sample_unit": "patienthealthsystemstayid", "endpoint_unit": "hospital_admission", "splits": {}}
    for split in ("train", "val", "test"):
        part = frame.loc[frame["split"] == split]
        complete_mortality = part["label_mortality"].notna()
        complete_los = part["label_los_days"].notna()
        valid = complete_mortality & complete_los
        summary["splits"][split] = {
            "total_stays": int(len(part)), "complete_endpoint_stays": int(valid.sum()),
            "missing_mortality": int((~complete_mortality).sum()), "missing_los": int((~complete_los).sum()),
            "mortality_prevalence": float(part.loc[complete_mortality, "label_mortality"].mean()) if complete_mortality.any() else None,
            "los3_prevalence": float((part.loc[complete_los, "label_los_days"] >= 3).mean()) if complete_los.any() else None,
            "los7_prevalence": float((part.loc[complete_los, "label_los_days"] >= 7).mean()) if complete_los.any() else None,
        }
    if audit_path:
        _atomic_text(audit_path, json.dumps(_json_safe(summary), indent=2, allow_nan=False))
    return summary


def _load_time_origin_metadata(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "prep_summary.json")
    if not os.path.isfile(path):
        raise RuntimeError("eICU prep_summary.json is required; time origin cannot be inferred.")
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("time_origin") != "hospital_admission" or summary.get("sample_unit") != "patienthealthsystemstayid":
        raise RuntimeError(
            "External processed data must be hospital-admission-relative and "
            "hospital-admission-level (patienthealthsystemstayid)."
        )
    return summary


def _audit_processed_stays(data_dir: str, schema_dir: Optional[str]) -> Dict[str, Any]:
    schema_vids = None
    if schema_dir:
        schema_path = os.path.join(schema_dir, "schema_processed.csv")
        if not os.path.isfile(schema_path):
            raise FileNotFoundError(schema_path)
        schema = pd.read_csv(schema_path)
        column = "vid" if "vid" in schema.columns else schema.columns[0]
        schema_vids = set(pd.to_numeric(schema[column], errors="raise").astype(int)) | {0}
    count = 0
    split_counts: Dict[str, int] = {}
    ids_by_split: Dict[str, list[str]] = {}
    subjects: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        directory = Path(data_dir) / split
        if not directory.is_dir():
            raise RuntimeError(f"Processed split directory is missing: {directory}")
        paths = sorted(directory.glob("*.pt"))
        if not paths:
            raise RuntimeError(f"Processed split directory is empty: {directory}")
        split_counts[split] = len(paths)
        ids_by_split[split] = []
        for path in tqdm(paths, desc=f"preflight {split}", unit="stay", leave=False):
            item = torch.load(path, map_location="cpu", weights_only=False)
            if item.get("time_origin") != "hospital_admission" or item.get("sample_unit") != "patienthealthsystemstayid":
                raise RuntimeError(f"External processed sample is not hospital-admission-level: {path}")
            hadm_id = str(item.get("hadm_id", path.stem))
            if str(item.get("patienthealthsystemstayid", "")) != hadm_id:
                raise RuntimeError(f"Processed sample hadm_id/patienthealthsystemstayid mismatch: {path}")
            members = [str(value) for value in item.get("patientunitstayids", [])]
            if not members or len(members) != len(set(members)):
                raise RuntimeError(f"Processed sample has invalid ICU-stay membership: {path}")
            if hadm_id in ids_by_split[split] or any(hadm_id in prior for prior in ids_by_split.values()):
                raise RuntimeError(f"Duplicate or cross-split processed hadm_id: {hadm_id}")
            ids_by_split[split].append(hadm_id)
            subject_id = str(item.get("subject_id", ""))
            if subject_id:
                if subject_id in subjects and subjects[subject_id] != split:
                    raise RuntimeError(f"Processed subject crosses splits: {subject_id}")
                subjects[subject_id] = split
            tokens = torch.as_tensor(item.get("state_tokens", []), dtype=torch.long).reshape(-1)
            values = torch.as_tensor(item.get("state_values", []), dtype=torch.float32).reshape(-1)
            times = torch.as_tensor(item.get("state_times", []), dtype=torch.float32).reshape(-1)
            if tokens.numel() != values.numel() or values.numel() != times.numel() or not len(times):
                raise RuntimeError(f"Invalid external processed stay: {path}")
            if not torch.isfinite(values).all() or not torch.isfinite(times).all() or not torch.all(times[1:] >= times[:-1]):
                raise RuntimeError(f"Non-finite or non-monotonic external stay: {path}")
            if bool((times < -1e-6).any()):
                raise RuntimeError(f"Negative hospital-relative time: {path}")
            hospital_los = item.get("labels", {}).get("hospital_los")
            if hospital_los is not None and bool((times > float(hospital_los) * 1440.0 + 5.0).any()):
                raise RuntimeError(f"Event exceeds hospital LOS tolerance: {path}")
            if schema_vids is not None and set(tokens.tolist()).difference(schema_vids):
                raise RuntimeError(f"External processed stay contains an out-of-schema VID: {path}")
            count += 1
    return {"processed_admission_count": count, "split_counts": split_counts, "ids_by_split": ids_by_split}


def run_external_validation_preflight(args: Any, cfg_dict: Mapping[str, Any]) -> Dict[str, Any]:
    labels_csv = str(getattr(args, "canonical_labels_csv", None) or getattr(args, "labels_csv"))
    source_labels_csv = labels_csv
    labels = pd.read_csv(labels_csv)
    if not {"label_mortality", "label_los_days"}.issubset(labels.columns):
        canonical_path = os.path.join(str(args.output_dir), "canonical_external_labels.csv")
        build_canonical_external_labels(source_labels_csv, canonical_path)
        labels_csv, labels = canonical_path, pd.read_csv(canonical_path)
    required = {"subject_id", "hadm_id", "split", "label_mortality", "label_los_days"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise RuntimeError(f"External preflight labels missing {missing}")
    mortality_values = pd.to_numeric(labels["label_mortality"], errors="coerce")
    los_values = pd.to_numeric(labels["label_los_days"], errors="coerce")
    if (mortality_values.notna() & ~mortality_values.isin([0, 1])).any():
        raise RuntimeError("External canonical mortality endpoint must be 0 or 1.")
    if (los_values.notna() & (~np.isfinite(los_values.to_numpy(dtype=np.float64)) | (los_values < 0))).any():
        raise RuntimeError("External canonical LOS endpoint must be finite and non-negative.")
    _formal_assert_patient_split_leakage(labels)
    if set(labels["split"].astype(str)) != {"train", "val", "test"}:
        raise RuntimeError("External labels must contain train, val and test splits.")
    time_summary = _load_time_origin_metadata(str(args.data_dir)) if getattr(args, "data_dir", None) else {}
    processed_summary = _audit_processed_stays(str(args.data_dir), getattr(args, "schema_dir", None)) if getattr(args, "data_dir", None) else {}
    for split in ("train", "val"):
        part = labels.loc[labels["split"].astype(str) == split]
        for task in EXTERNAL_TASKS:
            y, _ = _task_target(part, task)
            valid_y = np.asarray(y)[np.isfinite(np.asarray(y, dtype=np.float64))]
            if len(valid_y) == 0 or len(np.unique(valid_y)) < 2:
                raise RuntimeError(f"External {split} lacks both classes for {task}.")
    canonical_output = os.path.join(str(args.output_dir), "canonical_external_labels.csv")
    if {"label_mortality", "label_los_days"}.issubset(labels.columns):
        audit_frame = labels.copy()
        audit = {"endpoint_protocol": "hospital_admission_level_external_validation",
                 "sample_unit": "patienthealthsystemstayid", "endpoint_unit": "hospital_admission", "splits": {}}
        for split in ("train", "val", "test"):
            part = audit_frame.loc[audit_frame["split"].astype(str) == split]
            complete_mortality = part["label_mortality"].notna()
            complete_los = part["label_los_days"].notna()
            valid = complete_mortality & complete_los
            audit["splits"][split] = {"total_stays": int(len(part)), "complete_endpoint_stays": int(valid.sum()),
                                       "missing_mortality": int((~complete_mortality).sum()), "missing_los": int((~complete_los).sum())}
        audit_frame.to_csv(canonical_output, index=False)
    else:
        audit = build_canonical_external_labels(source_labels_csv, canonical_output)
    processed_ids_by_split: Dict[str, set[str]] = {split: set() for split in ("train", "val", "test")}
    if processed_summary:
        processed_ids_by_split = {
            split: set(map(str, processed_summary.get("ids_by_split", {}).get(split, [])))
            for split in processed_ids_by_split
        }
    audit_rows = []
    split_audit: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        part = labels.loc[labels["split"].astype(str) == split]
        label_ids = set(part["hadm_id"].astype(str))
        processed_ids = processed_ids_by_split[split]
        if getattr(args, "data_dir", None) and processed_ids != label_ids.intersection(processed_ids):
            # Processed stays without labels are always a hard contract violation.
            unknown = sorted(processed_ids.difference(label_ids))
            if unknown:
                raise RuntimeError(f"Processed {split} stays lack labels: {unknown[:20]}")
        split_audit[split] = {
            "label_count": len(label_ids), "processed_count": len(processed_ids),
            "matched_count": len(label_ids & processed_ids) if getattr(args, "data_dir", None) else 0,
            "label_only_count": len(label_ids - processed_ids) if getattr(args, "data_dir", None) else 0,
            "processed_without_label_count": len(processed_ids - label_ids) if getattr(args, "data_dir", None) else 0,
            "missing_mortality_count": int(pd.to_numeric(part["label_mortality"], errors="coerce").isna().sum()),
            "missing_los_count": int(pd.to_numeric(part["label_los_days"], errors="coerce").isna().sum()),
        }
        for row in part.itertuples(index=False):
            mortality_ok = pd.notna(getattr(row, "label_mortality"))
            los_ok = pd.notna(getattr(row, "label_los_days"))
            audit_rows.append({"hadm_id": str(row.hadm_id), "split": split,
                               "endpoint_complete": bool(mortality_ok and los_ok),
                               "exclusion_reason": "" if mortality_ok and los_ok else
                               "missing_hospital_mortality" if not mortality_ok else "missing_hospital_los"})
    audit_path = os.path.join(str(args.output_dir), "external_cohort_audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    if processed_summary and processed_summary.get("processed_admission_count", 0) == 0:
        raise RuntimeError("No hospital-admission processed samples found")
    return {"verdict": "PASS", "time_origin": "hospital_admission", "validation_protocol": "hospital_admission_level_external_validation",
            "sample_unit": "patienthealthsystemstayid", "endpoint_unit": "hospital_admission", "split_audit": split_audit,
            "time_summary": time_summary,
            "processed_summary": processed_summary,
            "label_summary": audit}


def _parse_selection(value: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    requested = list(allowed) if str(value).lower() == "all" else [x.strip() for x in str(value).split(",") if x.strip()]
    if not requested or set(requested).difference(allowed) or len(set(requested)) != len(requested):
        raise ValueError(f"Invalid selection {value!r}; allowed={list(allowed)}")
    return tuple(x for x in allowed if x in requested)


def _stage_key(task: str) -> Tuple[str, Optional[int]]:
    _ensure_runtime_imports()
    kind, cutoff = TASK_CACHE_KIND[task]
    return kind, None if kind == FULL_VISIBLE else int(cutoff)


def _probe_training_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    training = dict((cfg.get("probe", {}) or {}).get("training", {}) or {})
    training.setdefault("max_pairwise_elements_per_batch", 262144)
    training.setdefault("max_patients_per_batch", 64)
    training.setdefault("prefetch_factor", 2)
    return training


def _probe_paths(root: str, tasks: Sequence[str], sources: Sequence[str]) -> list[Tuple[str, str, str]]:
    paths = [(task, source, os.path.join(root, task, source, "checkpoints", "best.pt"))
             for task in tasks for source in sources]
    missing = [path for _, _, path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing external probe best.pt files: {missing}")
    return paths


def _assert_external_cache_contract(cache_root: str) -> None:
    metadata_path = os.path.join(cache_root, "cache_bank_metadata.json")
    if not os.path.isfile(metadata_path):
        raise RuntimeError("External cache metadata is missing; rebuild hospital-admission cache.")
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("sample_unit") != "patienthealthsystemstayid" or metadata.get("validation_protocol") != "hospital_admission_level_external_validation":
        raise RuntimeError("Refusing legacy ICU-stay cache; expected hospital-admission-level cache.")


def _external_evaluate(*, probe_root: str, phase1_ckpt: str, cache_root: str, labels_csv: str,
                       cfg: Mapping[str, Any], output_dir: str, mode: str,
                       tasks: Sequence[str], sources: Sequence[str], device: str, num_workers: int,
                       save_predictions: bool = False) -> Dict[str, Any]:
    _ensure_runtime_imports()
    print(f"[external {mode}] validating cache contract", flush=True)
    _assert_external_cache_contract(cache_root)
    print(f"[external {mode}] checking {len(tasks) * len(sources)} probe artifacts", flush=True)
    paths = _probe_paths(probe_root, tasks, sources)
    with open(os.path.join(cache_root, "cache_manifest.json"), encoding="utf-8") as handle:
        cache_signature = json.load(handle).get("cache_signature")
    if not isinstance(cache_signature, dict) or cache_signature.get("phase1_checkpoint_sha256") != _sha256(os.path.realpath(phase1_ckpt)):
        raise RuntimeError("External cache and CLI Phase-1 checkpoint provenance mismatch.")
    artifacts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    models: Dict[Tuple[str, str], Any] = {}
    artifact_progress = tqdm(paths, desc=f"external {mode}: artifacts", unit="probe", dynamic_ncols=True)
    for task, source, path in artifact_progress:
        artifact_progress.set_postfix_str(f"{task}/{source}")
        model, artifact = load_exact_time_probe(path, device=torch.device("cpu"), expected_task=task, expected_source=source)
        model.requires_grad_(False)
        signature = artifact.get("cache_signature")
        required_external = ("phase1_checkpoint_sha256", "phase1_architecture_version", "latent_dimension",
                             "cache_schema_version", "cache_contract_version")
        if not isinstance(signature, dict):
            raise RuntimeError(f"External probe/cache compatibility failed for {task}/{source}: missing probe cache_signature.")
        mismatched = [key for key in required_external if signature.get(key) != cache_signature.get(key)]
        if mismatched:
            details = {
                key: {"probe": signature.get(key), "external_cache": cache_signature.get(key)}
                for key in mismatched
            }
            raise RuntimeError(f"External probe/cache compatibility failed for {task}/{source}: {details}")
        artifacts[(task, source)] = artifact
        models[(task, source)] = model
    stages: Dict[Tuple[str, Optional[int]], Any] = {}
    lengths: Dict[Tuple[str, Optional[int]], Sequence[int]] = {}
    full_key = (FULL_VISIBLE, None)
    print(f"[external {mode}] opening test cache views", flush=True)
    stages[full_key] = load_dataset(cache_root, FULL_VISIBLE, split="test", cutoff_min=None)
    lengths[full_key] = stages[full_key].sequence_lengths()
    for task in tqdm(tasks, desc=f"external {mode}: cache stages", unit="task", dynamic_ncols=True):
        key = _stage_key(task)
        if key not in stages:
            stages[key] = load_dataset(cache_root, key[0], split="test", cutoff_min=key[1])
            lengths[key] = stages[key].sequence_lengths()
    targets: Dict[str, Mapping[str, Any]] = {}
    task_bases: Dict[str, _IndexedExternalDataset] = {}
    task_lengths: Dict[str, Sequence[int]] = {}
    task_counts: Dict[str, Dict[str, int]] = {}
    label_frame = pd.read_csv(labels_csv)
    for task in tasks:
        test_labels = label_frame.loc[label_frame["split"].astype(str) == "test"].copy()
        values, _ = _task_target(test_labels, task)
        targets[task] = {str(row.hadm_id): float(value) for row, value in zip(test_labels.itertuples(), values)
                         if np.isfinite(float(value))}
        key = _stage_key(task)
        ids = [str(x) for x in stages[key].admission_ids()]
        selected = [i for i, hadm_id in enumerate(ids) if hadm_id in targets[task]]
        if not selected:
            raise RuntimeError(f"No endpoint-complete external test admissions for {task}.")
        task_bases[task] = _IndexedExternalDataset(stages[key], selected)
        task_lengths[task] = task_bases[task].sequence_lengths()
        task_counts[task] = {
            "full_label_test_count": int(len(test_labels)), "cache_test_count": int(len(ids)),
            "evaluated_test_count": int(len(selected)),
            "endpoint_missing_excluded_count": int(len(test_labels) - len(targets[task])),
            "cutoff_empty_excluded_count": int(
                len(set(map(str, stages[full_key].admission_ids())) - set(ids))
            ) if key[0] != FULL_VISIBLE else 0,
        }
    results: list[Dict[str, Any]] = []
    source_cohorts: Dict[str, Tuple[list[str], np.ndarray]] = {}
    started = time.perf_counter()
    run_progress = tqdm(total=len(tasks) * len(sources), desc=f"external {mode}", unit="probe", dynamic_ncols=True)
    train_cfg = _probe_training_cfg(cfg)
    for task in tasks:
        key = _stage_key(task)
        task_base = task_bases[task]
        task_lengths_for_run = task_lengths[task]
        base_ids = task_base.admission_ids()
        sampler = AttentionCostBatchSampler(
            task_lengths_for_run, int(train_cfg["max_pairwise_elements_per_batch"]),
            int(train_cfg["max_patients_per_batch"]), False,
        )
        prediction_order = [base_ids[index] for batch in sampler for index in batch]
        missing = sorted(set(base_ids).difference(targets[task]))
        if missing:
            raise RuntimeError(f"External test cache IDs lack canonical targets for {task}: {missing[:20]}")
        for source in sources:
            print(f"[external {mode}] evaluating {task}/{source}", flush=True)
            path = os.path.join(probe_root, task, source, "checkpoints", "best.pt")
            model = models.pop((task, source))
            artifact = artifacts[(task, source)]
            prediction_path = os.path.join(output_dir, "predictions", f"{mode}_{task}_{source}_test.npz") if save_predictions else None
            if prediction_path:
                os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
            report = evaluate_exact_time_probe(
                probe_checkpoint=path, cache_root=cache_root, labels_csv=labels_csv,
                cfg_dict=cfg, split="test", task=task, cutoff_min=key[1], device=device,
                num_workers=num_workers, targets_override=targets[task], phase1_ckpt=phase1_ckpt,
                external_transfer=True, prediction_path=prediction_path, model_override=model,
                artifact_override=artifact, base_override=task_base, lengths_override=task_lengths_for_run,
                expected_ids=prediction_order,
            )
            y = np.asarray([targets[task][x] for x in base_ids], dtype=np.float32)
            cohort_key = task
            if cohort_key in source_cohorts:
                prior_ids, prior_y = source_cohorts[cohort_key]
                if base_ids != prior_ids or not np.array_equal(y, prior_y):
                    raise RuntimeError(f"External source cohort mismatch for {task}.")
            else:
                source_cohorts[cohort_key] = (list(base_ids), y.copy())
            result = {"mode": mode, "task": task, "source": source,
                      "endpoint": "hospital_mortality" if task == "mortality" else "hospital_los",
                      "window": "48h" if key[1] == 2880 else "24h", "n": report["n"],
                      "positive_count": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None,
                      "metrics": report["metrics"], "runtime_seconds": report["runtime_seconds"],
                      "cache_kind": key[0], "checkpoint": path,
                      "ordered_test_ids": list(base_ids),
                      "canonical_labels_csv": os.path.realpath(labels_csv),
                      "cache_root": os.path.realpath(cache_root), **task_counts[task]}
            results.append(result)
            run_progress.set_postfix_str(f"{task}/{source} n={report['n']}")
            run_progress.update(1)
            del model
    run_progress.close()
    compact = []
    for result in results:
        metrics = result["metrics"]
        compact.append({"mode": result["mode"], "task": result["task"], "source": result["source"],
                        "endpoint": result["endpoint"], "window": result["window"], "n": result["n"],
                        "positive_count": result["positive_count"], "prevalence": result["prevalence"],
                        "auprc": metrics.get("auprc"), "auroc": metrics.get("auroc"),
                        "brier": metrics.get("brier"), "ece": metrics.get("ece"),
                        "runtime_seconds": result["runtime_seconds"]})
    os.makedirs(output_dir, exist_ok=True)
    fields = ["mode", "task", "source", "endpoint", "window", "n", "positive_count", "prevalence", "auprc", "auroc", "brier", "ece", "runtime_seconds"]
    from io import StringIO
    buf = StringIO(); writer = csv.DictWriter(buf, fieldnames=fields); writer.writeheader(); writer.writerows(_json_safe(compact))
    _atomic_text(os.path.join(output_dir, "external_validation_metrics_compact.csv"), buf.getvalue())
    md = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in compact:
        md.append("| " + " | ".join("" if row[field] is None else (f"{row[field]:.4f}" if isinstance(row[field], float) else str(row[field])) for field in fields) + " |")
    _atomic_text(os.path.join(output_dir, "external_validation_metrics_compact.md"), "\n".join(md) + "\n")
    full = {"mode": mode, "tasks": list(tasks), "sources": list(sources), "total_runs": len(results),
            "validation_protocol": "hospital_admission_level_external_validation",
            "sample_unit": "patienthealthsystemstayid", "endpoint_unit": "hospital_admission",
            "total_runtime_seconds": time.perf_counter() - started, "results": _json_safe(results)}
    _atomic_text(os.path.join(output_dir, "external_validation_metrics_full.json"), json.dumps(full, indent=2, allow_nan=False))
    return full


def evaluate_external_probe_suite(*, probe_root: str, phase1_ckpt: str, cache_root: str,
                                  labels_csv: str, cfg_dict: Mapping[str, Any], output_dir: str,
                                  tasks: Sequence[str], sources: Sequence[str], mode: str,
                                  device: str, num_workers: int,
                                  save_predictions: bool = False) -> Dict[str, Any]:
    """Public inference-only external suite delegating to the canonical evaluator."""
    requested_tasks = tuple(tasks)
    requested_sources = tuple(sources)
    if not requested_tasks or set(requested_tasks) - set(EXTERNAL_TASKS):
        raise ValueError(f"External tasks must be drawn from {EXTERNAL_TASKS}")
    if not requested_sources or set(requested_sources) - set(EXTERNAL_SOURCES):
        raise ValueError(f"External sources must be drawn from {EXTERNAL_SOURCES}")
    if mode not in EXTERNAL_MODES:
        raise ValueError(f"Unknown external evaluation mode: {mode}")
    return _external_evaluate(probe_root=probe_root, phase1_ckpt=phase1_ckpt,
                              cache_root=cache_root, labels_csv=labels_csv,
                              cfg=cfg_dict, output_dir=output_dir, mode=mode,
                              tasks=requested_tasks, sources=requested_sources,
                              device=device, num_workers=num_workers,
                              save_predictions=save_predictions)


def retrain_external_probe_suite(*, phase1_ckpt: str, cache_root: str,
                                 labels_csv: str, cfg_dict: Mapping[str, Any],
                                 output_dir: str, tasks: Sequence[str], source: str,
                                 architecture_variant: str, device: str,
                                 num_workers: int, smoke: bool, resume: bool) -> Dict[str, Any]:
    """Train external probes through the canonical retrain implementation and evaluate test."""
    requested_tasks = tuple(tasks)
    if source not in EXTERNAL_SOURCES or set(requested_tasks) - set(EXTERNAL_TASKS):
        raise ValueError("Invalid external task/source selection")
    cfg = copy.deepcopy(dict(cfg_dict))
    cfg.setdefault("probe", {}).setdefault("encoder", {})["architecture_variant"] = architecture_variant
    cfg["probe"]["source_name"] = source
    args = SimpleNamespace(
        phase1_ckpt=phase1_ckpt, cache_root=cache_root, canonical_labels_csv=labels_csv,
        config="", output_dir=output_dir, device=device, num_workers=num_workers,
        smoke_mode=smoke, resume=None, save_predictions=True,
    )
    report = _retrain(args, cfg, requested_tasks, (source,))
    evaluated = evaluate_external_probe_suite(
        probe_root=output_dir, phase1_ckpt=phase1_ckpt, cache_root=cache_root,
        labels_csv=labels_csv, cfg_dict=cfg, output_dir=os.path.join(output_dir, "test_evaluation"),
        tasks=requested_tasks, sources=(source,), mode="retrain_probe", device=device,
        num_workers=num_workers, save_predictions=True)
    return {"retrain": report, "evaluation": evaluated}


def _retrain(args: Any, cfg: Mapping[str, Any], tasks: Sequence[str], sources: Sequence[str]) -> Dict[str, Any]:
    _ensure_runtime_imports()
    print("[external retrain] validating cache contract", flush=True)
    _assert_external_cache_contract(args.cache_root)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[external retrain] loading canonical labels: {args.canonical_labels_csv}", flush=True)
    canonical = pd.read_csv(args.canonical_labels_csv)
    parity = {
        "architecture_version": cfg["probe"].get("architecture_version"),
        "encoder": cfg["probe"].get("encoder"),
        "training": cfg["probe"].get("training"),
        "tasks": {task: cfg["probe"].get("tasks", {}).get(task, {}) for task in tasks},
        "sources": list(sources),
    }
    _atomic_text(os.path.join(args.output_dir, "external_parity_audit.json"), json.dumps(_json_safe(parity), indent=2, allow_nan=False))
    print("[external retrain] parity=" + json.dumps(_json_safe(parity), sort_keys=True), flush=True)
    reports = []
    train_progress = tqdm(total=len(tasks) * len(sources), desc="external retrain", unit="probe", dynamic_ncols=True)
    cache_views: Dict[Tuple[str, float, str], Any] = {}
    cohort_audit: Dict[str, Any] = {}
    for task in tasks:
        print(f"[external retrain] preparing {task} train/val cohort", flush=True)
        if task == "mortality":
            valid = pd.to_numeric(canonical["label_mortality"], errors="coerce").notna()
        else:
            valid = pd.to_numeric(canonical["label_los_days"], errors="coerce").notna()
        task_frame = canonical.loc[valid & canonical["split"].astype(str).isin(["train", "val"])].copy()
        cache_kind, cutoff_min = TASK_CACHE_KIND[task]
        available_ids: Dict[str, set[str]] = {}
        for split in ("train", "val"):
            view_key = (cache_kind, float(cutoff_min), split)
            if view_key not in cache_views:
                cache_views[view_key] = load_dataset(
                    args.cache_root, cache_kind, split=split, cutoff_min=cutoff_min
                )
            available_ids[split] = {str(value) for value in cache_views[view_key].admission_ids()}
        task_frame["hadm_id"] = task_frame["hadm_id"].astype(str)
        missing_by_split = {
            split: sorted(
                set(task_frame.loc[task_frame["split"].astype(str) == split, "hadm_id"])
                - available_ids[split]
            )
            for split in ("train", "val")
        }
        split_values = task_frame["split"].astype(str)
        keep = pd.Series(False, index=task_frame.index)
        for split in ("train", "val"):
            keep |= split_values.eq(split) & task_frame["hadm_id"].isin(available_ids[split])
        task_frame = task_frame.loc[keep].copy()
        cohort_audit[task] = {
            "cache_kind": cache_kind,
            "cutoff_min": float(cutoff_min),
            "endpoint_valid_count": int(valid.sum()),
            "cache_aligned_count": int(len(task_frame)),
            "cutoff_empty_excluded_count": int(sum(len(values) for values in missing_by_split.values())),
            "excluded_ids_by_split": {split: values[:100] for split, values in missing_by_split.items()},
        }
        if task_frame.empty:
            raise RuntimeError(
                f"No endpoint-valid external cohort remains in the {cache_kind} train/val cache for {task}."
            )
        task_labels_path = os.path.join(args.output_dir, f"_retrain_{task}_labels.csv")
        task_frame.to_csv(task_labels_path, index=False)
        selected_ids = {
            split: task_frame.loc[task_frame["split"].astype(str) == split, "hadm_id"].astype(str).tolist()
            for split in ("train", "val")
        }
        for source in sources:
            print(f"[external retrain] starting {task}/{source}", flush=True)
            local_cfg = copy.deepcopy(cfg)
            local_cfg["probe"]["source_name"] = source
            run_args = SimpleNamespace(
                prediction_task=task, task_type=task, phase1_ckpt=args.phase1_ckpt,
                config=args.config, labels_csv=task_labels_path, hcup_mapping_json=None,
                device=args.device, num_workers=args.num_workers, seed=42,
                smoke_mode=bool(getattr(args, "smoke_mode", False)), smoke_train_patients=64,
                smoke_val_patients=32, smoke_test_patients=32, max_steps=-1, resume=None,
                save_predictions=False, probe_protocol_hash=None, probe_selected_ids=selected_ids,
                autotune_probe_runtime=bool(getattr(args, "autotune_probe_runtime", False)),
            )
            run_dir = os.path.join(args.output_dir, task, source)
            reports.append(run_probe_phase(phase="probe_train", args=run_args, output_dir=run_dir,
                                           cache_root=args.cache_root, cfg_dict=local_cfg, evaluate_test=False))
            print(f"[external retrain] completed {task}/{source}", flush=True)
            train_progress.set_postfix_str(f"{task}/{source}")
            train_progress.update(1)
    train_progress.close()
    _atomic_text(
        os.path.join(args.output_dir, "retrain_cohort_audit.json"),
        json.dumps(_json_safe(cohort_audit), indent=2, allow_nan=False),
    )
    _atomic_text(os.path.join(args.output_dir, "retrain_manifest.json"), json.dumps(_json_safe(reports), indent=2, allow_nan=False))
    return {"reports": reports}


def _combine_mode_results(reuse_dir: str, retrain_dir: str, output_dir: str) -> None:
    frames = [pd.read_csv(os.path.join(reuse_dir, "external_validation_metrics_compact.csv")),
              pd.read_csv(os.path.join(retrain_dir, "external_validation_metrics_compact.csv"))]
    combined = pd.concat(frames, ignore_index=True)
    expected = [(mode, task, source) for mode in EXTERNAL_MODES for task in EXTERNAL_TASKS for source in EXTERNAL_SOURCES]
    actual = list(combined.loc[:, ["mode", "task", "source"]].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError("External mode results do not form the fixed 12-row order.")
    with open(os.path.join(reuse_dir, "external_validation_metrics_full.json"), encoding="utf-8") as handle:
        reuse_full = json.load(handle)
    with open(os.path.join(retrain_dir, "external_validation_metrics_full.json"), encoding="utf-8") as handle:
        retrain_full = json.load(handle)
    reuse_by = {(r["task"], r["source"]): r for r in reuse_full.get("results", [])}
    retrain_by = {(r["task"], r["source"]): r for r in retrain_full.get("results", [])}
    for key in reuse_by:
        left, right = reuse_by[key], retrain_by.get(key)
        if right is None or left.get("ordered_test_ids") != right.get("ordered_test_ids"):
            raise RuntimeError(f"External mode cohort ID mismatch for {key}.")
        for field in ("n", "positive_count", "prevalence", "cache_root", "canonical_labels_csv"):
            if left.get(field) != right.get(field):
                raise RuntimeError(f"External mode cohort field mismatch for {key}: {field}")
    os.makedirs(output_dir, exist_ok=True)
    fields = list(combined.columns)
    from io import StringIO
    buf = StringIO(); combined.to_csv(buf, index=False); _atomic_text(os.path.join(output_dir, "external_validation_metrics_compact.csv"), buf.getvalue())
    md = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in combined.to_dict(orient="records"):
        md.append("| " + " | ".join("" if pd.isna(row[field]) else str(row[field]) for field in fields) + " |")
    _atomic_text(os.path.join(output_dir, "external_validation_metrics_compact.md"), "\n".join(md) + "\n")
    full_results = list(reuse_full.get("results", [])) + list(retrain_full.get("results", []))
    _atomic_text(os.path.join(output_dir, "external_validation_metrics_full.json"), json.dumps(_json_safe({"total_runs": len(full_results), "results": full_results}), indent=2, allow_nan=False))
    _atomic_text(os.path.join(output_dir, "external_validation_audit.json"), json.dumps({"combined_modes": list(EXTERNAL_MODES)}, indent=2))


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase1_ckpt", required=True)
    parser.add_argument("--data_dir", required=False)
    parser.add_argument("--schema_dir", required=False)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--labels_csv", required=False)
    parser.add_argument("--canonical_labels_csv", required=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical eICU external validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight"); _common_parser(preflight)
    cache = sub.add_parser("cache_build"); _common_parser(cache); cache.add_argument("--rebuild_cache", action="store_true")
    reuse = sub.add_parser("reuse_pretrained"); _common_parser(reuse); reuse.add_argument("--mimic_probe_root", required=True)
    retrain = sub.add_parser("retrain_probe"); _common_parser(retrain); retrain.add_argument("--smoke_mode", action="store_true"); retrain.add_argument("--autotune_probe_runtime", action="store_true")
    evaluate = sub.add_parser("evaluate_retrained"); _common_parser(evaluate); evaluate.add_argument("--probe_root", required=True); evaluate.add_argument("--reuse_results_dir", default=None); evaluate.add_argument("--combined_output_dir", default=None)
    args = parser.parse_args()
    print(f"[external] command={args.command} device={args.device} num_workers={args.num_workers}", flush=True)
    _ensure_runtime_imports()
    print("[external] loading config", flush=True)
    cfg = load_config_dict(args.config)
    tasks = _parse_selection(args.tasks, EXTERNAL_TASKS)
    sources = _parse_selection(args.sources, EXTERNAL_SOURCES)
    if args.command == "preflight":
        print("[external] preflight: validating labels, processed stays and split alignment", flush=True)
        result = run_external_validation_preflight(args, cfg)
        os.makedirs(args.output_dir, exist_ok=True)
        _atomic_text(os.path.join(args.output_dir, "external_preflight.json"), json.dumps(_json_safe(result), indent=2, allow_nan=False))
        print(json.dumps(_json_safe(result), indent=2)); return
    if args.command == "cache_build":
        if not args.data_dir or not args.schema_dir or not args.labels_csv:
            raise ValueError("cache_build requires --data_dir, --schema_dir and raw --labels_csv.")
        print("[external] cache_build: running preflight once", flush=True)
        preflight_report = run_external_validation_preflight(args, cfg)
        _atomic_text(
            os.path.join(args.output_dir, "external_preflight.json"),
            json.dumps(_json_safe(preflight_report), indent=2, allow_nan=False),
        )
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        schema_path = os.path.join(args.schema_dir, "schema_processed.csv")
        teacher = JEPAModel.load_from_checkpoint(args.phase1_ckpt, map_location=device, weights_only=False, schema_path=schema_path).to(device).eval()
        teacher.requires_grad_(False)
        print("[external] cache_build: extracting full_visible/cutoff_24h/cutoff_48h", flush=True)
        report = build_phase1_probe_cache(cache_root=args.cache_root, data_dir=args.data_dir, schema_dir=args.schema_dir,
                                          cfg_dict=cfg, teacher=teacher, phase1_ckpt_path=args.phase1_ckpt,
                                          device=device, labels_csv=args.labels_csv, rebuild=args.rebuild_cache,
                                          num_workers=args.num_workers)
        print(json.dumps(_json_safe(report), indent=2)); return
    labels = args.canonical_labels_csv
    if not labels:
        raise ValueError("External evaluation requires --canonical_labels_csv.")
    args.canonical_labels_csv = labels
    if not os.path.isfile(labels):
        raise FileNotFoundError(labels)
    if args.command == "retrain_probe":
        result = _retrain(args, cfg, tasks, sources)
        print(json.dumps(_json_safe(result), indent=2)); return
    root = args.mimic_probe_root if args.command == "reuse_pretrained" else args.probe_root
    result = _external_evaluate(probe_root=root, phase1_ckpt=args.phase1_ckpt, cache_root=args.cache_root,
                                labels_csv=labels, cfg=cfg, output_dir=args.output_dir,
                                mode="reuse_pretrained" if args.command == "reuse_pretrained" else "retrain_probe",
                                tasks=tasks, sources=sources, device=args.device,
                                num_workers=args.num_workers, save_predictions=args.save_predictions)
    if args.command == "evaluate_retrained" and args.reuse_results_dir and args.combined_output_dir:
        _combine_mode_results(args.reuse_results_dir, args.output_dir, args.combined_output_dir)
    _atomic_text(os.path.join(args.output_dir, "external_validation_audit.json"), json.dumps(_json_safe({"endpoint_protocol": "hospital_admission_level_external_validation", "sample_unit": "patienthealthsystemstayid", "endpoint_unit": "hospital_admission", "tasks": list(tasks), "sources": list(sources)}), indent=2, allow_nan=False))
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
