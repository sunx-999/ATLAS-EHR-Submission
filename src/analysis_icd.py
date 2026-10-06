"""ICD probe-attribution workflow implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.cache_bank import FULL_VISIBLE, _phase1_dataset_kwargs, load_dataset
from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.eval import load_exact_time_probe
from src.model import JEPAModel
from src.probes import build_delta_time_min, canonical_probe_collate
from src.utils import get_autocast_dtype, load_config_dict

def _icd_schema_maps(schema_dir: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    schema = pd.read_csv(os.path.join(schema_dir, "schema_processed.csv"))
    label_col = "normalized_concept_label" if "normalized_concept_label" in schema.columns else None
    if label_col is None:
        for candidate in ("label", "concept_label", "feature_name"):
            if candidate in schema.columns:
                label_col = candidate
                break
    labels: Dict[int, str] = {}
    types: Dict[int, str] = {}
    if "vid" not in schema.columns:
        return labels, types
    for _, row in schema.iterrows():
        try:
            vid = int(row["vid"])
        except Exception:
            continue
        labels[vid] = str(row.get(label_col, vid)) if label_col is not None else str(vid)
        types[vid] = str(row.get("model_type", "")).lower()
    return labels, types


def _icd_sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _icd_label_code(artifact: Mapping[str, Any], label_index: int) -> Optional[str]:
    vocab = artifact.get("label_vocab")
    if isinstance(vocab, Mapping):
        codes = vocab.get("codes")
        if isinstance(codes, Sequence) and 0 <= int(label_index) < len(codes):
            return str(codes[int(label_index)])
    return None


@dataclass(frozen=True)
class _ICDLabelIdentity:
    label_index: int
    label_code: Optional[str]
    vocabulary_source: str


@dataclass
class _ICDAttributionRuntime:
    device: torch.device
    probe: torch.nn.Module
    probe_artifact: Mapping[str, Any]
    source: str
    cache_signature: Mapping[str, Any]
    cache_dataset: Any
    cache_index: Dict[str, int]
    label_identity: _ICDLabelIdentity
    phase1_sha256: Optional[str]
    probe_sha256: Optional[str]
    schema_sha256: Optional[str]
    cache_schema_version: Optional[str]
    cfg_dict: Optional[Mapping[str, Any]] = None
    teacher: Optional[JEPAModel] = None
    raw_dataset: Optional[Any] = None
    raw_index: Optional[Dict[str, int]] = None
    label_map: Optional[Dict[int, str]] = None
    type_map: Optional[Dict[int, str]] = None
    phase1_ckpt: Optional[str] = None
    schema_dir: Optional[str] = None
    data_dir: Optional[str] = None


def _icd_resolve_label_identity(artifact: Mapping[str, Any], label_index: int,
                                label_code: Optional[str] = None) -> _ICDLabelIdentity:
    idx = int(label_index)
    artifact_code = _icd_label_code(artifact, idx)
    if label_code is not None and artifact_code is not None and str(label_code) != str(artifact_code):
        raise RuntimeError(
            f"ICD label identity mismatch: label_index={idx} maps to {artifact_code!r}, "
            f"but label_code={label_code!r} was requested."
        )
    resolved = str(label_code) if label_code is not None else artifact_code
    source = "artifact_label_vocab.codes" if artifact_code is not None else "user_label_code"
    return _ICDLabelIdentity(label_index=idx, label_code=resolved, vocabulary_source=source)


def _icd_cache_signature(cache_root: str) -> Mapping[str, Any]:
    manifest_path = os.path.join(os.path.abspath(cache_root), "cache_manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            return json.load(handle)["cache_signature"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("ICD attribution requires a signed canonical probe cache.") from exc


def _icd_validate_cache_probe_signature(cache_signature: Mapping[str, Any],
                                        artifact: Mapping[str, Any]) -> None:
    for key in ("phase1_checkpoint_sha256", "schema_sha256", "cache_schema_version"):
        if cache_signature.get(key) != artifact.get(key):
            raise RuntimeError(f"ICD attribution cache/probe provenance mismatch for {key}.")


def _icd_index_by_hadm(dataset: Any) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for idx in range(len(dataset)):
        hadm_id = str(dataset[idx]["hadm_id"])
        if hadm_id in index:
            raise RuntimeError(f"Duplicate admission in ICD attribution dataset: {hadm_id}")
        index[hadm_id] = int(idx)
    return index


def _icd_full_visible_runtime(*, cache_root: str, probe_checkpoint: str, label_index: int,
                              label_code: Optional[str] = None, device: str = "cuda",
                              phase1_ckpt: Optional[str] = None,
                              data_dir: Optional[str] = None,
                              schema_dir: Optional[str] = None,
                              config: str = "configs/params.yaml",
                              raw_target_ids: Optional[Sequence[str]] = None) -> _ICDAttributionRuntime:
    resolved = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    probe, artifact = load_exact_time_probe(probe_checkpoint, device=resolved, expected_task="icd")
    source = artifact["source_name"]
    if source not in {"z_set", "z_context"}:
        raise RuntimeError("ICD attribution only accepts canonical z_set/z_context sources.")
    label_identity = _icd_resolve_label_identity(artifact, int(label_index), label_code)
    cache_signature = _icd_cache_signature(cache_root)
    _icd_validate_cache_probe_signature(cache_signature, artifact)
    phase1_sha = cache_signature.get("phase1_checkpoint_sha256")
    if phase1_ckpt:
        phase1_sha = _icd_sha256_file(str(phase1_ckpt))
        if phase1_sha != cache_signature.get("phase1_checkpoint_sha256") or phase1_sha != artifact.get("phase1_checkpoint_sha256"):
            raise RuntimeError("ICD attribution Phase1 checkpoint SHA256 mismatch.")
    schema_sha = cache_signature.get("schema_sha256")
    if schema_dir:
        schema_path = os.path.join(str(schema_dir), "schema_processed.csv")
        schema_sha = _icd_sha256_file(schema_path)
        if schema_sha != cache_signature.get("schema_sha256") or schema_sha != artifact.get("schema_sha256"):
            raise RuntimeError("ICD attribution schema SHA256 mismatch.")
    cache_dataset = load_dataset(cache_root, FULL_VISIBLE, split="test")
    cache_index = _icd_index_by_hadm(cache_dataset)

    cfg_dict = None
    teacher = None
    raw_dataset = None
    raw_index = None
    label_map = None
    type_map = None
    if phase1_ckpt or data_dir or schema_dir:
        missing = [name for name, value in {
            "phase1_ckpt": phase1_ckpt, "data_dir": data_dir, "schema_dir": schema_dir,
        }.items() if not value]
        if missing:
            raise ValueError("Feature/VID ICD attribution requires: " + ", ".join(missing))
        cfg_dict = load_config_dict(config)
        ds_kwargs = _phase1_dataset_kwargs(dict(cfg_dict))
        # Match FULL_VISIBLE probe cache extraction: long admissions must not
        # be silently cropped by the Phase-1 training sequence cap.
        ds_kwargs.pop("max_seq_len_limit", None)
        ds_kwargs.update({
            "mask_target_ratio": 0.0,
            "mask_min_prob": 0.0,
            "mask_max_prob": 0.0,
            "static_mask_prob": 0.0,
            "static_numeric_vids": [],
        })
        raw_dataset = PatientSequenceDataset(
            data_dir=str(data_dir),
            schema_dir=str(schema_dir),
            split="test",
            target_ids=[str(x) for x in raw_target_ids] if raw_target_ids is not None else None,
            fraction=1.0,
            deterministic=True,
            return_legacy_nested=False,
            **ds_kwargs,
        )
        raw_index = _icd_index_by_hadm(raw_dataset)
        label_map, type_map = _icd_schema_maps(str(schema_dir))
        teacher = JEPAModel.load_from_checkpoint(
            str(phase1_ckpt),
            map_location=resolved,
            schema_path=os.path.join(str(schema_dir), "schema_processed.csv"),
            weights_only=False,
        ).to(resolved).eval()
        teacher.requires_grad_(False)

    return _ICDAttributionRuntime(
        device=resolved,
        probe=probe,
        probe_artifact=artifact,
        source=source,
        cache_signature=cache_signature,
        cache_dataset=cache_dataset,
        cache_index=cache_index,
        label_identity=label_identity,
        phase1_sha256=phase1_sha,
        probe_sha256=_icd_sha256_file(probe_checkpoint) if os.path.isfile(probe_checkpoint) else None,
        schema_sha256=schema_sha,
        cache_schema_version=cache_signature.get("cache_schema_version"),
        cfg_dict=cfg_dict,
        teacher=teacher,
        raw_dataset=raw_dataset,
        raw_index=raw_index,
        label_map=label_map,
        type_map=type_map,
        phase1_ckpt=phase1_ckpt,
        schema_dir=schema_dir,
        data_dir=data_dir,
    )


def _icd_attach_raw_runtime(runtime: _ICDAttributionRuntime, *, phase1_ckpt: str,
                            data_dir: str, schema_dir: str, config: str,
                            raw_target_ids: Sequence[str]) -> _ICDAttributionRuntime:
    cfg_dict = load_config_dict(config)
    ds_kwargs = _phase1_dataset_kwargs(dict(cfg_dict))
    # Match FULL_VISIBLE probe cache extraction: no Phase-1 training sequence
    # cap during raw replay, otherwise raw/cache parity fails for long stays.
    ds_kwargs.pop("max_seq_len_limit", None)
    ds_kwargs.update({
        "mask_target_ratio": 0.0,
        "mask_min_prob": 0.0,
        "mask_max_prob": 0.0,
        "static_mask_prob": 0.0,
        "static_numeric_vids": [],
    })
    runtime.raw_dataset = PatientSequenceDataset(
        data_dir=str(data_dir),
        schema_dir=str(schema_dir),
        split="test",
        target_ids=[str(x) for x in raw_target_ids],
        fraction=1.0,
        deterministic=True,
        return_legacy_nested=False,
        **ds_kwargs,
    )
    runtime.raw_index = _icd_index_by_hadm(runtime.raw_dataset)
    runtime.label_map, runtime.type_map = _icd_schema_maps(str(schema_dir))
    runtime.cfg_dict = cfg_dict
    runtime.teacher = JEPAModel.load_from_checkpoint(
        str(phase1_ckpt),
        map_location=runtime.device,
        schema_path=os.path.join(str(schema_dir), "schema_processed.csv"),
        weights_only=False,
    ).to(runtime.device).eval()
    runtime.teacher.requires_grad_(False)
    runtime.phase1_ckpt = phase1_ckpt
    runtime.data_dir = data_dir
    runtime.schema_dir = schema_dir
    return runtime


def _icd_parse_label_list(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(x) for x in value}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (list, tuple, set)):
            return {str(x) for x in parsed}
    except Exception:
        pass
    for sep in (";", ",", "|"):
        if sep in text:
            return {x.strip() for x in text.split(sep) if x.strip()}
    return {text}


def _icd_true_target(labels_csv: Optional[str], hadm_id: str, label_code: Optional[str]) -> Optional[int]:
    if not labels_csv or not label_code or not os.path.isfile(labels_csv):
        return None
    frame = pd.read_csv(labels_csv)
    if "hadm_id" not in frame.columns:
        return None
    row = frame.loc[frame["hadm_id"].astype(str) == str(hadm_id)]
    if row.empty:
        return None
    if str(label_code) in row.columns:
        value = pd.to_numeric(row.iloc[0][str(label_code)], errors="coerce")
        return int(float(value) > 0) if pd.notna(value) else 0
    for column in ("icd_codes", "icd", "icd_labels"):
        if column in row.columns:
            return int(str(label_code) in _icd_parse_label_list(row.iloc[0][column]))
    return None


def _icd_label_target_lookup(labels_csv: str, label_code: str, split: Optional[str] = None) -> Dict[str, int]:
    frame = pd.read_csv(labels_csv)
    if "hadm_id" not in frame.columns:
        raise RuntimeError("ICD labels_csv must contain hadm_id.")
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == str(split)]
    out: Dict[str, int] = {}
    for _, row in frame.iterrows():
        target = None
        if str(label_code) in frame.columns:
            value = pd.to_numeric(row[str(label_code)], errors="coerce")
            target = int(float(value) > 0) if pd.notna(value) else 0
        else:
            for column in ("icd_codes", "icd", "icd_labels"):
                if column in frame.columns:
                    target = int(str(label_code) in _icd_parse_label_list(row[column]))
                    break
        if target is None:
            raise RuntimeError(
                "ICD labels_csv must contain an ICD list column or a one-hot label_code column."
            )
        out[str(row["hadm_id"])] = int(target)
    return out




def _icd_observation_rows(item: Mapping[str, Any], label_map: Mapping[int, str],
                          type_map: Mapping[int, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = item["state_tokens_flat"].long()
    values = item["state_values_flat"].float()
    times = item["state_times"].float().view(-1)
    counts = item["state_counts"].long()
    offsets = item["state_offsets"].long()
    for t_idx in range(int(counts.numel())):
        lo, hi = int(offsets[t_idx]), int(offsets[t_idx + 1])
        for flat_idx in range(lo, hi):
            vid = int(tokens[flat_idx].item())
            slot_idx = int(flat_idx - lo + 1)
            rows.append({
                "flat_index": int(flat_idx),
                "t_idx": int(t_idx),
                "slot_idx": slot_idx,
                "time_min": float(times[t_idx].item()),
                "vid": vid,
                "feature_label": str(label_map.get(vid, vid)),
                "feature_type": str(type_map.get(vid, "")),
                "value": float(values[flat_idx].item()),
            })
    return rows


# Historical BMP/CMP definitions from the removed icd_label_attribution.py.
# The explicit VID whitelists were empty in its newest version.
BMP_KEYWORDS = [
    "sodium", "potassium", "chloride", "bicarbonate", "total co2",
    "carbon dioxide", "urea nitrogen", "bun", "creatinine", "glucose", "calcium",
]
BMP_VID_WHITELIST: set[int] = set()
CMP_KEYWORDS = BMP_KEYWORDS + [
    "albumin", "bilirubin", "alkaline phosphatase", "aspartate aminotransferase",
    "alanine aminotransferase", "ast", "alt", "total protein",
]
CMP_VID_WHITELIST: set[int] = set()


def _icd_panel_normalize_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s)


def _panel_match(label: str, keywords: List[str]) -> bool:
    label_norm = _icd_panel_normalize_text(label)
    return any(
        (re.search(rf"\b{re.escape(k)}\b", label_norm) is not None)
        if len(k) <= 4 or " " not in k else k in label_norm
        for k in keywords
    )


def is_bmp_feature(label: str, vid: Optional[int] = None) -> bool:
    if vid is not None and int(vid) in BMP_VID_WHITELIST:
        return True
    return _panel_match(label, BMP_KEYWORDS)


def is_cmp_feature(label: str, vid: Optional[int] = None) -> bool:
    if vid is not None and int(vid) in CMP_VID_WHITELIST:
        return True
    return _panel_match(label, CMP_KEYWORDS)


def _icd_panel_rows(rows: Sequence[Mapping[str, Any]], panel: str) -> List[Mapping[str, Any]]:
    matcher = is_bmp_feature if panel == "BMP" else is_cmp_feature
    return [row for row in rows if str(row.get("feature_type", "")) == "numeric"
            and matcher(str(row.get("feature_label", "")), int(row["vid"]))]


def _icd_panel_membership_audit(schema_dir: str) -> pd.DataFrame:
    schema = pd.read_csv(os.path.join(schema_dir, "schema_processed.csv"))
    if "vid" not in schema.columns:
        raise RuntimeError("schema_processed.csv must contain vid for panel membership audit.")
    name_col = next((c for c in ("canonical_name", "normalized_concept_label", "label", "concept_label", "feature_name") if c in schema.columns), None)
    source_col = next((c for c in ("source_table", "table_name", "source_table_name") if c in schema.columns), None)
    rows: List[Dict[str, Any]] = []
    for _, record in schema.iterrows():
        try:
            vid = int(record["vid"])
        except (TypeError, ValueError):
            continue
        feature_type = str(record.get("model_type", "")).lower()
        if feature_type != "numeric":
            continue
        name = str(record.get(name_col, vid)) if name_col else str(vid)
        for panel, matcher in (("BMP", is_bmp_feature), ("CMP", is_cmp_feature)):
            if matcher(name, vid):
                rows.append({
                    "panel": panel, "vid": vid, "canonical_feature_name": name,
                    "feature_type": feature_type,
                    "source": str(record.get(source_col, "")) if source_col else "",
                })
    return pd.DataFrame(rows, columns=["panel", "vid", "canonical_feature_name", "feature_type", "source"]).sort_values(
        ["panel", "vid"], kind="stable"
    )


def _icd_apply_perturbation(batch: Dict[str, Any], jobs: Sequence[Mapping[str, Any]]) -> None:
    tokens = batch["state_tokens"]
    values = batch["state_values"]
    pad = batch["state_pad_mask"]
    for b, job in enumerate(jobs):
        method = str(job["method"])
        vid = int(job["vid"])
        if str(job["level"]) == "observation":
            positions = [(int(job["t_idx"]), int(job["slot_idx"]))]
        else:
            hit = ((tokens[b] == vid) & (~pad[b])).nonzero(as_tuple=False)
            positions = [(int(t.item()), int(s.item())) for t, s in hit]
        if method == "drop":
            for t_idx, slot_idx in positions:
                pad[b, t_idx, slot_idx] = True
                tokens[b, t_idx, slot_idx] = 0
                values[b, t_idx, slot_idx, 0] = 0.0
        elif method == "replace_with_train_feature_median":
            for t_idx, slot_idx in positions:
                values[b, t_idx, slot_idx, 0] = float(job.get("replacement_value", 0.0))
        elif method == "within_admission_same_vid_shuffle":
            feature_values = values[b][(tokens[b] == vid) & (~pad[b]), 0]
            if int(feature_values.numel()) < 2:
                continue
            replacement = torch.roll(feature_values, shifts=1)
            k = 0
            for t_idx, slot_idx in positions:
                values[b, t_idx, slot_idx, 0] = replacement[k % int(replacement.numel())]
                k += 1
        else:
            raise ValueError(f"Unsupported ICD perturbation method: {method}")


def _icd_compact_flat_item(item: Mapping[str, Any], keep: torch.Tensor,
                           retain_empty_timestamps: bool = False) -> Dict[str, Any]:
    tokens = item["state_tokens_flat"].long()
    values = item["state_values_flat"].float()
    masks = item["state_masks_flat"].float()
    counts = item["state_counts"].long()
    offsets = item["state_offsets"].long()
    times = item["state_times"].float().view(-1)
    if keep.numel() != tokens.numel():
        raise RuntimeError("ICD perturbation keep mask has wrong length.")
    new_tokens: List[torch.Tensor] = []
    new_values: List[torch.Tensor] = []
    new_masks: List[torch.Tensor] = []
    new_counts: List[int] = []
    new_times: List[float] = []
    for t_idx in range(int(counts.numel())):
        lo, hi = int(offsets[t_idx]), int(offsets[t_idx + 1])
        local_keep = keep[lo:hi]
        if int(local_keep.sum().item()) == 0 and not retain_empty_timestamps:
            continue
        new_tokens.append(tokens[lo:hi][local_keep])
        new_values.append(values[lo:hi][local_keep])
        new_masks.append(masks[lo:hi][local_keep])
        new_counts.append(int(local_keep.sum().item()))
        new_times.append(float(times[t_idx].item()))
    if not new_counts:
        raise RuntimeError("ICD perturbation removed every timestamp; refusing empty admission.")
    time_tensor = torch.tensor(new_times, dtype=torch.float32).view(-1, 1)
    if time_tensor.numel() > 1 and not bool(torch.all(time_tensor[1:, 0] > time_tensor[:-1, 0]).item()):
        raise RuntimeError("ICD perturbation produced non-increasing timestamps.")
    out = dict(item)
    out["state_tokens_flat"] = torch.cat(new_tokens).long()
    out["state_values_flat"] = torch.cat(new_values).float()
    out["state_masks_flat"] = torch.cat(new_masks).float()
    out["state_counts"] = torch.tensor(new_counts, dtype=torch.long)
    out["state_offsets"] = torch.cat([torch.zeros(1, dtype=torch.long), out["state_counts"].cumsum(dim=0)])
    out["state_times"] = time_tensor
    delta = torch.zeros((len(new_counts),), dtype=torch.float32)
    if len(new_counts) > 1:
        delta[1:] = time_tensor[1:, 0] - time_tensor[:-1, 0]
    out["delta_times"] = delta.view(-1, 1)
    return out


def _icd_materialize_panel_item(item: Mapping[str, Any], panel_rows: Sequence[Mapping[str, Any]],
                                method: str, train_medians: Mapping[int, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    affected = [int(row["flat_index"]) for row in panel_rows]
    if not affected:
        return dict(item), {"status": "ineligible_no_panel_members"}
    if method == "panel_value_neutralization":
        missing = sorted({int(row["vid"]) for row in panel_rows if int(row["vid"]) not in train_medians})
        if missing:
            return dict(item), {"status": "ineligible_missing_train_median", "missing_median_vids": missing}
        out = dict(item)
        out["state_tokens_flat"] = item["state_tokens_flat"].long().clone()
        out["state_values_flat"] = item["state_values_flat"].float().clone()
        out["state_masks_flat"] = item["state_masks_flat"].float().clone()
        for row in panel_rows:
            out["state_values_flat"][int(row["flat_index"])] = float(train_medians[int(row["vid"])])
        return out, {"status": "evaluated"}
    if method == "panel_removal":
        keep = torch.ones((item["state_tokens_flat"].numel(),), dtype=torch.bool)
        keep[affected] = False
        return _icd_compact_flat_item(item, keep, retain_empty_timestamps=True), {"status": "evaluated"}
    raise ValueError(f"Unsupported ICD panel perturbation method: {method}")


def _icd_panel_timestamp_audit(hadm_id: str, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    audit_rows: List[Dict[str, Any]] = []
    for panel in ("BMP", "CMP"):
        members = _icd_panel_rows(rows, panel)
        by_timestamp: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for row in members:
            by_timestamp[int(row["t_idx"])].append(row)
        for t_idx, group in sorted(by_timestamp.items()):
            audit_rows.append({
                "hadm_id": str(hadm_id), "timestamp_index": int(t_idx),
                "time_min": float(group[0]["time_min"]), "panel": panel,
                "member_count": int(len(group)),
                "member_vids": json.dumps([int(row["vid"]) for row in group]),
                "member_names": json.dumps([str(row["feature_label"]) for row in group]),
                "interpretation": "BMP-like/CMP-like co-measurement group; exact timestamp equality does not prove a common clinical order.",
            })
    return pd.DataFrame(audit_rows, columns=[
        "hadm_id", "timestamp_index", "time_min", "panel", "member_count",
        "member_vids", "member_names", "interpretation",
    ])


def _icd_panel_attribution(*, teacher: JEPAModel, probe: torch.nn.Module, source: str,
                           item: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                           label_index: int, baseline_logit: float,
                           batch_size: int, device: torch.device,
                           train_medians: Mapping[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    audit = _icd_panel_timestamp_audit(str(item["hadm_id"]), rows)
    jobs: List[Tuple[str, str, List[Mapping[str, Any]], Dict[str, Any]]] = []
    result_rows: List[Dict[str, Any]] = []
    for panel in ("BMP", "CMP"):
        members = _icd_panel_rows(rows, panel)
        unique_vids = sorted({int(row["vid"]) for row in members})
        affected_timestamps = len({int(row["t_idx"]) for row in members})
        for method in ("panel_value_neutralization", "panel_removal"):
            _, meta = _icd_materialize_panel_item(item, members, method, train_medians)
            row = {
                "hadm_id": str(item["hadm_id"]), "panel": panel, "perturbation_method": method,
                "baseline_logit": float(baseline_logit), "perturbed_logit": np.nan, "logit_drop": np.nan,
                "number_affected_observations": int(len(members)),
                "number_affected_timestamps": int(affected_timestamps),
                "number_unique_member_vids": int(len(unique_vids)), "member_vids": json.dumps(unique_vids),
                "status": str(meta["status"]),
            }
            result_rows.append(row)
            if meta["status"] == "evaluated":
                jobs.append((panel, method, members, row))
    for start in range(0, len(jobs), max(1, int(batch_size))):
        chunk = jobs[start:start + max(1, int(batch_size))]
        materialized = [_icd_materialize_panel_item(item, members, method, train_medians)[0]
                        for _, method, members, _ in chunk]
        # Reuse the canonical raw replay/re-encoding path with batched items.
        logits = _icd_recompute_probe_logits_for_items(
            teacher=teacher, probe=probe, source=source, items=materialized,
            label_index=label_index, device=device,
        )
        for (_, _, _, row), logit in zip(chunk, logits.tolist()):
            row["perturbed_logit"] = float(logit)
            row["logit_drop"] = float(float(baseline_logit) - float(logit))
    return pd.DataFrame(result_rows), audit


def _icd_materialize_job_item(item: Mapping[str, Any], job: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if str(job.get("level")) == "none":
        return dict(item), {"status": "evaluated"}
    method = str(job["method"])
    vid = int(job["vid"])
    tokens = item["state_tokens_flat"].long()
    values = item["state_values_flat"].float()
    keep = torch.ones((tokens.numel(),), dtype=torch.bool)
    meta = {"status": "evaluated", "replacement_value": None, "donor_t_idx": None, "donor_slot_idx": None, "value_delta": None}
    target_flat = int(job["flat_index"])
    if method == "drop":
        if str(job["level"]) == "observation":
            keep[target_flat] = False
        elif str(job["level"]) == "timestamp":
            counts = item["state_counts"].long()
            offsets = item["state_offsets"].long()
            t_idx = int(job["t_idx"])
            keep[int(offsets[t_idx]):int(offsets[t_idx + 1])] = False
        else:
            keep &= tokens != vid
        return _icd_compact_flat_item(item, keep), meta
    out = dict(item)
    out["state_tokens_flat"] = tokens.clone()
    out["state_values_flat"] = values.clone()
    out["state_masks_flat"] = item["state_masks_flat"].float().clone()
    if method == "replace_with_train_feature_median":
        replacement = job.get("replacement_value")
        if replacement is None or not np.isfinite(float(replacement)):
            meta["status"] = "ineligible_missing_train_median"
            return out, meta
        affected = torch.tensor([target_flat], dtype=torch.long) if str(job["level"]) == "observation" else (tokens == vid).nonzero(as_tuple=True)[0]
        for idx in affected.tolist():
            old = float(out["state_values_flat"][idx].item())
            out["state_values_flat"][idx] = float(replacement)
            meta["replacement_value"] = float(replacement)
            meta["value_delta"] = float(float(replacement) - old)
        if all(float(values[idx].item()) == float(replacement) for idx in affected.tolist()):
            meta["status"] = "no_op_equal_value"
        return out, meta
    if method == "within_admission_same_vid_shuffle":
        donors = (tokens == vid).nonzero(as_tuple=True)[0].tolist()
        if str(job["level"]) == "observation":
            donors = [idx for idx in donors if int(idx) != target_flat]
            if not donors:
                meta["status"] = "ineligible_singleton"
                return out, meta
            donor = donors[int(job.get("donor_rank", 0)) % len(donors)]
            replacement = float(values[donor].item())
            out["state_values_flat"][target_flat] = replacement
            meta["replacement_value"] = replacement
            meta["value_delta"] = float(replacement - float(values[target_flat].item()))
            meta["donor_t_idx"] = int(job.get("donor_t_idx", -1))
            meta["donor_slot_idx"] = int(job.get("donor_slot_idx", -1))
            if replacement == float(values[target_flat].item()):
                meta["status"] = "no_op_equal_value"
            return out, meta
        if len(donors) < 2:
            meta["status"] = "ineligible_singleton"
            return out, meta
        donor_values = torch.roll(values[donors], shifts=1)
        out["state_values_flat"][donors] = donor_values
        if bool(torch.equal(values[donors], donor_values)):
            meta["status"] = "no_op_equal_value"
        return out, meta
    raise ValueError(f"Unsupported ICD perturbation method: {method}")


def _icd_perturbation_meta(item: Mapping[str, Any], job: Mapping[str, Any]) -> Dict[str, Any]:
    if str(job.get("level")) == "none":
        return {"status": "evaluated", "replacement_value": None, "donor_t_idx": None, "donor_slot_idx": None, "value_delta": None}
    method = str(job["method"])
    vid = int(job["vid"])
    tokens = item["state_tokens_flat"].long()
    values = item["state_values_flat"].float()
    meta = {"status": "evaluated", "replacement_value": None, "donor_t_idx": None, "donor_slot_idx": None, "value_delta": None}
    target_flat = int(job["flat_index"])
    if method == "drop":
        return meta
    if method == "replace_with_train_feature_median":
        replacement = job.get("replacement_value")
        if replacement is None or not np.isfinite(float(replacement)):
            meta["status"] = "ineligible_missing_train_median"
            return meta
        affected = [target_flat] if str(job["level"]) == "observation" else (tokens == vid).nonzero(as_tuple=True)[0].tolist()
        deltas = []
        for idx in affected:
            deltas.append(float(float(replacement) - float(values[int(idx)].item())))
        meta["replacement_value"] = float(replacement)
        meta["value_delta"] = float(deltas[-1]) if deltas else None
        if affected and all(float(values[int(idx)].item()) == float(replacement) for idx in affected):
            meta["status"] = "no_op_equal_value"
        return meta
    if method == "within_admission_same_vid_shuffle":
        donors = (tokens == vid).nonzero(as_tuple=True)[0].tolist()
        if str(job["level"]) == "observation":
            donors = [idx for idx in donors if int(idx) != target_flat]
            if not donors:
                meta["status"] = "ineligible_singleton"
                return meta
            donor = donors[int(job.get("donor_rank", 0)) % len(donors)]
            replacement = float(values[donor].item())
            meta["replacement_value"] = replacement
            meta["value_delta"] = float(replacement - float(values[target_flat].item()))
            meta["donor_t_idx"] = int(job.get("donor_t_idx", -1))
            meta["donor_slot_idx"] = int(job.get("donor_slot_idx", -1))
            if replacement == float(values[target_flat].item()):
                meta["status"] = "no_op_equal_value"
            return meta
        if len(donors) < 2:
            meta["status"] = "ineligible_singleton"
            return meta
        donor_values = torch.roll(values[donors], shifts=1)
        if bool(torch.equal(values[donors], donor_values)):
            meta["status"] = "no_op_equal_value"
        return meta
    raise ValueError(f"Unsupported ICD perturbation method: {method}")


def _icd_recompute_probe_logits_dense_value_jobs(*, teacher: JEPAModel, probe: torch.nn.Module,
                                                 source: str, item: Mapping[str, Any],
                                                 jobs: Sequence[Mapping[str, Any]],
                                                 label_index: int, device: torch.device) -> torch.Tensor:
    if not jobs:
        return torch.empty((0,), dtype=torch.float32)
    batch = nested_padding_collate([item])
    bsz = len(jobs)
    state_tokens = batch["state_tokens"].repeat(bsz, 1, 1)
    state_values = batch["state_values"].repeat(bsz, 1, 1, 1)
    state_pad_mask = batch["state_pad_mask"].repeat(bsz, 1, 1)
    seq_pad_mask = batch["seq_pad_mask"].repeat(bsz, 1)
    state_times = batch["state_times"].repeat(bsz, 1, 1)
    for b, job in enumerate(jobs):
        method = str(job["method"])
        vid = int(job["vid"])
        if method == "replace_with_train_feature_median":
            replacement = float(job["replacement_value"])
            if str(job["level"]) == "observation":
                state_values[b, int(job["t_idx"]), int(job["slot_idx"]), 0] = replacement
            else:
                hit = (state_tokens[b] == vid) & (~state_pad_mask[b])
                values_b = state_values[b, ..., 0]
                values_b[hit] = replacement
        elif method == "within_admission_same_vid_shuffle":
            hit = (state_tokens[b] == vid) & (~state_pad_mask[b])
            if str(job["level"]) == "observation":
                state_values[b, int(job["t_idx"]), int(job["slot_idx"]), 0] = float(job["replacement_value"])
            else:
                values_b = state_values[b, ..., 0]
                feature_values = values_b[hit]
                values_b[hit] = torch.roll(feature_values, shifts=1)
        else:
            raise ValueError(f"Dense ICD perturbation only supports value-preserving-shape jobs, got {method!r}.")
    batch_dev = {
        "state_tokens": state_tokens.to(device, non_blocking=True),
        "state_values": state_values.to(device, non_blocking=True),
        "state_pad_mask": state_pad_mask.to(device, non_blocking=True),
        "seq_pad_mask": seq_pad_mask.to(device, non_blocking=True),
        "state_times": state_times.to(device, non_blocking=True),
    }
    with torch.inference_mode():
        seq_pad = batch_dev["seq_pad_mask"].bool()
        autocast_dtype = get_autocast_dtype(device.type)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
            frontend = teacher.encode_teacher_sets(
                batch_dev["state_tokens"],
                batch_dev["state_values"],
                batch_dev["state_pad_mask"].bool(),
                seq_pad,
            )
            z_set = frontend["z_set"].detach()
            times = batch_dev["state_times"].float().squeeze(-1)
            z_context = teacher.contextualize_teacher_z_set(z_set, seq_pad, times).detach()
        latent = z_set if source == "z_set" else z_context
        direct_batch = {
            "latent_seq": latent.float(),
            "time_min": times.float(),
            "delta_time_min": build_delta_time_min(times.float(), seq_pad),
            "seq_pad_mask": seq_pad,
            "lengths": (~seq_pad).sum(dim=1).long(),
            "hadm_id": [str(item["hadm_id"])] * bsz,
        }
        logits = probe(direct_batch)
    return logits[:, int(label_index)].detach().cpu()


def _icd_recompute_probe_logits_for_items(*, teacher: JEPAModel, probe: torch.nn.Module,
                                          source: str, items: Sequence[Mapping[str, Any]],
                                          label_index: int, device: torch.device) -> torch.Tensor:
    batch = nested_padding_collate(list(items))
    batch_dev = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
    with torch.inference_mode():
        seq_pad = batch_dev["seq_pad_mask"].bool()
        autocast_dtype = get_autocast_dtype(device.type)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
            frontend = teacher.encode_teacher_sets(
                batch_dev["state_tokens"], batch_dev["state_values"],
                batch_dev["state_pad_mask"].bool(), seq_pad,
            )
            z_set = frontend["z_set"].detach()
            times = batch_dev["state_times"].float().squeeze(-1)
            z_context = teacher.contextualize_teacher_z_set(z_set, seq_pad, times).detach()
        latent = z_set if source == "z_set" else z_context
        logits = probe({
            "latent_seq": latent.float(), "time_min": times.float(),
            "delta_time_min": build_delta_time_min(times.float(), seq_pad),
            "seq_pad_mask": seq_pad, "lengths": (~seq_pad).sum(dim=1).long(),
            "hadm_id": [str(item["hadm_id"]) for item in items],
        })
    return logits[:, int(label_index)].detach().cpu()


def _icd_recompute_probe_logits(*, teacher: JEPAModel, probe: torch.nn.Module, source: str,
                                item: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]],
                                label_index: int, device: torch.device,
                                return_latents: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    materialized = [_icd_materialize_job_item(item, job)[0] for job in jobs]
    batch = nested_padding_collate(materialized)
    batch_dev = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
    with torch.inference_mode():
        seq_pad = batch_dev["seq_pad_mask"].bool()
        autocast_dtype = get_autocast_dtype(device.type)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
            frontend = teacher.encode_teacher_sets(
                batch_dev["state_tokens"], batch_dev["state_values"],
                batch_dev["state_pad_mask"].bool(), seq_pad,
            )
            z_set = frontend["z_set"].detach()
            times = batch_dev["state_times"].float().squeeze(-1)
            z_context = teacher.contextualize_teacher_z_set(z_set, seq_pad, times).detach()
        latent = z_set if source == "z_set" else z_context
        direct_batch = {
            "latent_seq": latent.float(), "time_min": times.float(),
            "delta_time_min": build_delta_time_min(times.float(), seq_pad),
            "seq_pad_mask": seq_pad, "lengths": (~seq_pad).sum(dim=1).long(),
            "hadm_id": [str(item["hadm_id"])] * len(jobs),
        }
        logits = probe(direct_batch)
    selected = logits[:, int(label_index)].detach().cpu()
    if return_latents:
        return selected, z_set.detach().cpu(), z_context.detach().cpu(), times.detach().cpu()
    return selected


def _icd_validate_raw_cache_parity(*, teacher: JEPAModel, probe: torch.nn.Module,
                                   item: Mapping[str, Any], cache_record: Mapping[str, Any],
                                   source: str, label_index: int, cached_logit: float,
                                   device: torch.device, z_atol: float, logit_atol: float,
                                   require_latent: bool = True,
                                   require_logit: bool = True) -> Dict[str, Any]:
    logits, z_set, z_context, times = _icd_recompute_probe_logits(
        teacher=teacher, probe=probe, source=source, item=item,
        jobs=[{"level": "none", "method": "drop", "vid": -1, "t_idx": 0, "slot_idx": 0, "flat_index": 0}],
        label_index=label_index, device=device, return_latents=True,
    )
    cached_times = cache_record["time_min"].float().cpu()
    if str(cache_record["hadm_id"]) != str(item["hadm_id"]):
        raise RuntimeError("Raw/cache parity failed: admission ID mismatch.")
    if times.shape[1] != cached_times.numel() or not torch.allclose(times[0], cached_times, atol=1.0e-4, rtol=0.0):
        raise RuntimeError("Raw/cache parity failed: timestamp count or values mismatch.")
    z_set_diff = float((z_set[0, :cached_times.numel()].float() - cache_record["z_set"].float()).abs().max().item())
    z_context_diff = float((z_context[0, :cached_times.numel()].float() - cache_record["z_context"].float()).abs().max().item())
    logit_diff = abs(float(logits[0].item()) - float(cached_logit))
    latent_pass = z_set_diff <= float(z_atol) and z_context_diff <= float(z_atol)
    logit_pass = logit_diff <= float(logit_atol)
    if (bool(require_logit) and not logit_pass) or (bool(require_latent) and not latent_pass):
        raise RuntimeError(
            "Raw/cache parity failed: "
            f"z_set_max_abs={z_set_diff:.6g}, z_context_max_abs={z_context_diff:.6g}, logit_abs={logit_diff:.6g}"
        )
    return {
        "z_set_max_abs": z_set_diff,
        "z_context_max_abs": z_context_diff,
        "logit_abs": logit_diff,
        "latent_pass": bool(latent_pass),
        "logit_pass": bool(logit_pass),
        "latent_required": bool(require_latent),
        "logit_required": bool(require_logit),
    }


def _icd_train_feature_medians(*, data_dir: str, schema_dir: str,
                               vids: Sequence[int]) -> Dict[int, float]:
    wanted = {int(v) for v in vids}
    if not wanted:
        return {}
    numeric = _icd_numeric_vids(schema_dir)
    wanted &= numeric
    if not wanted:
        return {}
    with tempfile.TemporaryDirectory(prefix="icd_train_median_") as tmpdir:
        buffers: Dict[int, List[float]] = defaultdict(list)
        counters: Dict[int, int] = defaultdict(int)
        for path in tqdm(_icd_split_files(data_dir, "train"), desc="icd_train_feature_medians", ascii=True, mininterval=10.0):
            raw = torch.load(path, map_location="cpu", weights_only=False)
            tokens = raw["state_tokens"].long().numpy()
            vals = raw["state_values"].float().numpy()
            for vid, val in zip(tokens, vals):
                vid_int = int(vid)
                value = float(val)
                if vid_int not in wanted or not np.isfinite(value):
                    continue
                buffers[vid_int].append(value)
                if len(buffers[vid_int]) >= 8192:
                    _icd_flush_median_value_buffers(buffers, tmpdir, counters)
        _icd_flush_median_value_buffers(buffers, tmpdir, counters)
        medians: Dict[int, float] = {}
        root = os.path.join(tmpdir, "median_value_shards")
        for vid in sorted(wanted):
            directory = os.path.join(root, f"vid={vid}")
            if not os.path.isdir(directory):
                continue
            arrays = [
                np.load(os.path.join(directory, name), mmap_mode="r")
                for name in sorted(os.listdir(directory))
                if name.endswith(".npy")
            ]
            if arrays:
                medians[vid] = float(np.median(np.concatenate([np.asarray(arr, dtype=np.float32) for arr in arrays])))
        return medians


def _icd_feature_attribution(*, teacher: JEPAModel, probe: torch.nn.Module, source: str,
                             item: Mapping[str, Any], label_index: int,
                             rows: Sequence[Mapping[str, Any]],
                             methods: Sequence[str], batch_size: int,
                             device: torch.device,
                             train_medians: Mapping[int, float],
                             seed: int, show_progress: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    base_job = [{"level": "none", "method": "drop", "vid": -1, "t_idx": 0, "slot_idx": 0}]
    # Bypass perturbation for this one baseline by using an observation that is
    # guaranteed absent. The method machinery then leaves the batch unchanged.
    base_job[0]["method"] = "drop"
    recomputed_baseline = float(_icd_recompute_probe_logits(
        teacher=teacher, probe=probe, source=source, item=item, jobs=base_job,
        label_index=label_index, device=device
    )[0].item())
    jobs: List[Dict[str, Any]] = []
    rows_by_vid: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_vid[int(row["vid"])].append(row)
    rng = np.random.default_rng(int(seed))
    donor_rank_by_flat: Dict[int, int] = {}
    for vid, members in rows_by_vid.items():
        for row in members:
            donor_rank_by_flat[int(row["flat_index"])] = int(rng.integers(0, max(1, len(members) - 1)))
    for row in rows:
        for method in methods:
            if method != "drop" and str(row.get("feature_type", "")) != "numeric":
                continue
            vid = int(row["vid"])
            donors = [r for r in rows_by_vid[vid] if int(r["flat_index"]) != int(row["flat_index"])]
            donor = donors[donor_rank_by_flat[int(row["flat_index"])] % len(donors)] if donors else None
            jobs.append({
                **dict(row), "level": "observation", "method": method,
                "replacement_value": train_medians.get(vid),
                "donor_rank": donor_rank_by_flat[int(row["flat_index"])],
                "donor_t_idx": None if donor is None else int(donor["t_idx"]),
                "donor_slot_idx": None if donor is None else int(donor["slot_idx"]),
            })
    seen_vid: set[int] = set()
    for row in rows:
        vid = int(row["vid"])
        if vid in seen_vid:
            continue
        seen_vid.add(vid)
        for method in methods:
            if method != "drop" and str(row.get("feature_type", "")) != "numeric":
                continue
            jobs.append({**dict(row), "level": "vid", "method": method,
                         "replacement_value": train_medians.get(vid), "donor_rank": 0})

    metas = [_icd_perturbation_meta(item, job) for job in jobs]
    logits_by_index: Dict[int, float] = {}
    indexed_drop = [(idx, job) for idx, (job, meta) in enumerate(zip(jobs, metas)) if meta["status"] == "evaluated" and str(job["method"]) == "drop"]
    indexed_value = [
        (idx, {**dict(job), "replacement_value": meta["replacement_value"]})
        for idx, (job, meta) in enumerate(zip(jobs, metas))
        if meta["status"] == "evaluated" and str(job["method"]) != "drop"
    ]
    work_units = [("drop", indexed_drop), ("value", indexed_value)]
    for kind, indexed in work_units:
        if not indexed:
            continue
        iterator = range(0, len(indexed), max(1, int(batch_size)))
        for start in tqdm(iterator, desc=f"icd_feature_attribution/{kind}", ascii=True, disable=not bool(show_progress)):
            chunk_indexed = indexed[start:start + max(1, int(batch_size))]
            chunk_jobs = [job for _, job in chunk_indexed]
            if kind == "drop":
                logits = _icd_recompute_probe_logits(
                    teacher=teacher, probe=probe, source=source, item=item, jobs=chunk_jobs,
                    label_index=label_index, device=device,
                )
            else:
                logits = _icd_recompute_probe_logits_dense_value_jobs(
                    teacher=teacher, probe=probe, source=source, item=item, jobs=chunk_jobs,
                    label_index=label_index, device=device,
                )
            for (idx, _), logit in zip(chunk_indexed, logits.tolist()):
                logits_by_index[int(idx)] = float(logit)
    out_rows: List[Dict[str, Any]] = []
    for idx, (job, meta) in enumerate(zip(jobs, metas)):
        logit = logits_by_index.get(int(idx), np.nan)
        out_rows.append({
            **{k: v for k, v in job.items() if k not in {"flat_index"}},
            **meta,
            "perturbed_logit": float(logit),
            "logit_drop": float(recomputed_baseline - float(logit)) if np.isfinite(float(logit)) else np.nan,
        })
    df = pd.DataFrame(out_rows)
    vid_df = df[df["level"] == "vid"].copy()
    return df[df["level"] == "observation"].copy(), vid_df, recomputed_baseline


def _icd_raw_timestamp_attribution(*, teacher: JEPAModel, probe: torch.nn.Module, source: str,
                                   item: Mapping[str, Any], label_index: int,
                                   baseline_logit: float, batch_size: int,
                                   device: torch.device, show_progress: bool = True) -> pd.DataFrame:
    times = item["state_times"].float().view(-1)
    jobs = [
        {"level": "timestamp", "method": "drop", "vid": -1, "flat_index": 0, "t_idx": int(t), "slot_idx": 0}
        for t in range(int(times.numel()))
    ]
    rows: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(jobs), max(1, int(batch_size))), desc="icd_raw_timestamp_attribution", ascii=True, disable=not bool(show_progress)):
        chunk = jobs[start:start + max(1, int(batch_size))]
        logits = _icd_recompute_probe_logits(
            teacher=teacher, probe=probe, source=source, item=item, jobs=chunk,
            label_index=label_index, device=device,
        )
        for job, logit in zip(chunk, logits.tolist()):
            rows.append({
                "timestamp_index": int(job["t_idx"]),
                "time_min": float(times[int(job["t_idx"])].item()),
                "perturbed_logit": float(logit),
                "logit_drop": float(baseline_logit - float(logit)),
                "interpretation": "raw_timestamp_deletion_recomputed_phase1",
            })
    return pd.DataFrame(rows)


def _icd_cached_latent_token_occlusion(*, model: torch.nn.Module, latent: torch.Tensor,
                                       time_min: torch.Tensor, hadm_id: str,
                                       label_index: int, baseline_logit: float,
                                       device: torch.device, batch_size: int) -> List[Dict[str, Any]]:
    t = int(latent.size(0))
    if t <= 1:
        return [{
            "timestamp_index": 0,
            "time_min": float(time_min[0]) if t else None,
            "perturbed_logit": None,
            "logit_drop": None,
            "status": "ineligible_single_token_sequence",
            "interpretation": "cached_latent_token_deletion_not_raw_timestamp_attribution",
        }]
    rows: List[Dict[str, Any]] = []
    latent_cpu = latent.detach().cpu().float()
    time_cpu = time_min.detach().cpu().float()
    for start in range(0, t, max(1, int(batch_size))):
        items = []
        indices = list(range(start, min(start + max(1, int(batch_size)), t)))
        for index in indices:
            keep = torch.ones((t,), dtype=torch.bool)
            keep[index] = False
            items.append({
                "latent_seq": latent_cpu[keep].contiguous(),
                "time_min": time_cpu[keep].contiguous(),
                "target": 0,
                "hadm_id": str(hadm_id),
            })
        batch = canonical_probe_collate(items)
        batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(batch)[:, int(label_index)].detach().cpu()
        for index, logit in zip(indices, logits.tolist()):
            rows.append({
                "timestamp_index": int(index),
                "time_min": float(time_cpu[index].item()),
                "perturbed_logit": float(logit),
                "logit_drop": float(baseline_logit - float(logit)),
                "status": "evaluated",
                "interpretation": "cached_latent_token_deletion_not_raw_timestamp_attribution",
            })
    return rows


def _write_icd_outputs(output_dir: str, result: Mapping[str, Any],
                       observation_df: Optional[pd.DataFrame],
                       vid_df: Optional[pd.DataFrame],
                       raw_timestamp_df: Optional[pd.DataFrame],
                       panel_df: Optional[pd.DataFrame] = None,
                       panel_timestamp_df: Optional[pd.DataFrame] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cached_path = os.path.join(output_dir, "icd_cached_token_occlusion.csv")
    tmp = f"{cached_path}.tmp"
    pd.DataFrame(result["cached_latent_token_occlusion"]).to_csv(tmp, index=False)
    os.replace(tmp, cached_path)
    if raw_timestamp_df is not None:
        path = os.path.join(output_dir, "icd_raw_timestamp_attribution.csv")
        tmp = f"{path}.tmp"
        raw_timestamp_df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    if observation_df is not None:
        path = os.path.join(output_dir, "icd_observation_attribution.csv")
        tmp = f"{path}.tmp"
        observation_df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    if vid_df is not None:
        path = os.path.join(output_dir, "icd_vid_attribution.csv")
        tmp = f"{path}.tmp"
        vid_df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    if panel_df is not None:
        _icd_write_csv_atomic(panel_df, os.path.join(output_dir, "icd_panel_attribution.csv"))
    if panel_timestamp_df is not None:
        _icd_write_csv_atomic(panel_timestamp_df, os.path.join(output_dir, "icd_panel_timestamp_audit.csv"))
    path = os.path.join(output_dir, "icd_attribution_summary.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_icd_json_safe(result), handle, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _icd_write_csv_atomic(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _icd_positive_label_hadm_ids(labels_csv: str, label_code: str, split: str = "test") -> set[str]:
    return {hadm_id for hadm_id, target in _icd_label_target_lookup(labels_csv, label_code, split=split).items() if int(target) == 1}




def _icd_read_optional_csv(path: str, *, hadm_id: str, label_index: int, label_code: Optional[str]) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path)
    for column, value in (("label_code", label_code), ("label_index", int(label_index)), ("hadm_id", str(hadm_id))):
        if column in frame.columns:
            frame[column] = value
        else:
            frame.insert(0, column, value)
    return frame


def _icd_append_csv(frame: pd.DataFrame, path: str) -> None:
    if frame.empty:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)


def _icd_add_cohort_columns(frame: pd.DataFrame, *, hadm_id: str, label_index: int,
                            label_code: Optional[str], baseline_logit: Any,
                            baseline_probability: Any) -> pd.DataFrame:
    out = frame.copy()
    for column, value in (
        ("hadm_id", str(hadm_id)),
        ("label_index", int(label_index)),
        ("label_code", label_code),
        ("baseline_logit", baseline_logit),
        ("baseline_probability", baseline_probability),
    ):
        out[column] = value
    if "level" not in out.columns:
        out["level"] = ""
    if "method" not in out.columns:
        out["method"] = ""
    if "status" not in out.columns:
        out["status"] = "evaluated"
    return out


def _icd_cohort_vid_summary(vid_rows: pd.DataFrame, cohort_size: int) -> pd.DataFrame:
    if vid_rows.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("method", "vid", "feature_label", "feature_type") if col in vid_rows.columns]
    if not group_cols or "logit_drop" not in vid_rows.columns:
        return pd.DataFrame()
    work = vid_rows.copy()
    work["status"] = work.get("status", "evaluated").astype(str)
    work["logit_drop"] = pd.to_numeric(work["logit_drop"], errors="coerce")
    work["evaluated"] = work["status"].eq("evaluated") & np.isfinite(work["logit_drop"])
    # Prevent high-frequency admissions from receiving extra weight if a VID
    # unexpectedly has multiple rows for one method/admission.
    admission_level = (
        work.groupby(["hadm_id"] + group_cols, dropna=False)
        .agg(
            present=("hadm_id", "size"),
            evaluated=("evaluated", "any"),
            logit_drop=("logit_drop", lambda x: float(np.nanmean(x)) if np.isfinite(pd.to_numeric(x, errors="coerce")).any() else np.nan),
        )
        .reset_index()
    )
    evaluated = admission_level[admission_level["evaluated"]].copy()
    def _q(p: float):
        return lambda x: float(np.nanquantile(pd.to_numeric(x, errors="coerce"), p))
    summary = (
        admission_level.groupby(group_cols, dropna=False)
        .agg(
            n_present_admissions=("hadm_id", "nunique"),
            n_evaluated_admissions=("evaluated", "sum"),
        )
        .reset_index()
    )
    effect = (
        evaluated.groupby(group_cols, dropna=False)
        .agg(
            n_evaluated_rows=("hadm_id", "size"),
            mean_logit_drop=("logit_drop", "mean"),
            median_logit_drop=("logit_drop", "median"),
            q25_logit_drop=("logit_drop", _q(0.25)),
            q75_logit_drop=("logit_drop", _q(0.75)),
            positive_drop_fraction=("logit_drop", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            negative_drop_fraction=("logit_drop", lambda x: float((pd.to_numeric(x, errors="coerce") < 0).mean())),
        )
        .reset_index()
    )
    out = summary.merge(effect, on=group_cols, how="left")
    out.insert(0, "cohort_size", int(cohort_size))
    out["presence_rate"] = out["n_present_admissions"] / max(1, int(cohort_size))
    out["evaluation_rate_among_present"] = out["n_evaluated_admissions"] / out["n_present_admissions"].replace(0, np.nan)
    ordered = [
        "method", "vid", "feature_label", "feature_type", "cohort_size",
        "n_present_admissions", "presence_rate", "n_evaluated_admissions",
        "evaluation_rate_among_present", "n_evaluated_rows", "mean_logit_drop",
        "median_logit_drop", "q25_logit_drop", "q75_logit_drop",
        "positive_drop_fraction", "negative_drop_fraction",
    ]
    return out[[c for c in ordered if c in out.columns]].sort_values(
        ["mean_logit_drop", "n_present_admissions"], ascending=[False, False], na_position="last"
    )


def _icd_vid_summary_admission_rows(vid_rows: pd.DataFrame) -> pd.DataFrame:
    if vid_rows.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("method", "vid", "feature_label", "feature_type") if col in vid_rows.columns]
    if not group_cols or "logit_drop" not in vid_rows.columns or "hadm_id" not in vid_rows.columns:
        return pd.DataFrame()
    work = vid_rows.copy()
    work["status"] = work.get("status", "evaluated").astype(str)
    work["logit_drop"] = pd.to_numeric(work["logit_drop"], errors="coerce")
    work["evaluated"] = work["status"].eq("evaluated") & np.isfinite(work["logit_drop"])
    out = (
        work.groupby(["hadm_id"] + group_cols, dropna=False)
        .agg(
            status=("status", lambda x: "evaluated" if (x.astype(str) == "evaluated").any() else str(x.iloc[0])),
            logit_drop=("logit_drop", lambda x: float(np.nanmean(x)) if np.isfinite(pd.to_numeric(x, errors="coerce")).any() else np.nan),
        )
        .reset_index()
    )
    return out[["hadm_id"] + group_cols + ["status", "logit_drop"]]


def _icd_cohort_panel_summary(panel_rows: pd.DataFrame) -> pd.DataFrame:
    if panel_rows.empty:
        return pd.DataFrame()
    work = panel_rows.copy()
    work["logit_drop"] = pd.to_numeric(work["logit_drop"], errors="coerce")
    work = work.loc[work["status"].astype(str).eq("evaluated") & np.isfinite(work["logit_drop"])]
    if work.empty:
        return pd.DataFrame(columns=["panel", "perturbation_method", "n_eligible_admissions"])
    # A panel-method is evaluated once per admission; the groupby is a guard
    # against accidental duplicate rows, never a timestamp-level analysis.
    admission = work.groupby(["hadm_id", "panel", "perturbation_method"], as_index=False, dropna=False).agg(
        logit_drop=("logit_drop", "mean")
    )
    def _quantile(q: float):
        return lambda values: float(np.quantile(pd.to_numeric(values, errors="coerce"), q))
    return admission.groupby(["panel", "perturbation_method"], as_index=False, dropna=False).agg(
        n_eligible_admissions=("hadm_id", "nunique"),
        mean_logit_drop=("logit_drop", "mean"), median_logit_drop=("logit_drop", "median"),
        q025_logit_drop=("logit_drop", _quantile(0.025)), q25_logit_drop=("logit_drop", _quantile(0.25)),
        q75_logit_drop=("logit_drop", _quantile(0.75)), q975_logit_drop=("logit_drop", _quantile(0.975)),
        positive_drop_fraction=("logit_drop", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
    )


def _icd_numeric_vid_union(runtime: _ICDAttributionRuntime, selected_ids: Sequence[str]) -> set[int]:
    if runtime.raw_dataset is None or runtime.raw_index is None:
        return set()
    type_map = runtime.type_map or {}
    vids: set[int] = set()
    for hadm_id in selected_ids:
        item = runtime.raw_dataset[runtime.raw_index[str(hadm_id)]]
        for row in _icd_observation_rows(item, runtime.label_map or {}, type_map):
            if str(row.get("feature_type")) == "numeric":
                vids.add(int(row["vid"]))
    return vids


def _icd_preflight_raw_cache_lengths(runtime: _ICDAttributionRuntime,
                                     selected_ids: Sequence[str]) -> Dict[str, Any]:
    if runtime.raw_dataset is None or runtime.raw_index is None:
        return {"checked_admissions": 0, "mismatch_count": 0, "mismatches": []}
    mismatches: List[Dict[str, Any]] = []
    for hadm_id in selected_ids:
        hid = str(hadm_id)
        if hid not in runtime.raw_index:
            mismatches.append({"hadm_id": hid, "error": "missing_raw_record"})
            continue
        if hid not in runtime.cache_index:
            mismatches.append({"hadm_id": hid, "error": "missing_cache_record"})
            continue
        raw_item = runtime.raw_dataset[runtime.raw_index[hid]]
        cache_record = runtime.cache_dataset[runtime.cache_index[hid]]
        raw_len = int(len(raw_item["state_counts"]))
        cache_time = cache_record["time_min"]
        cache_len = int(cache_time.numel() if isinstance(cache_time, torch.Tensor) else len(cache_time))
        if raw_len != cache_len:
            mismatches.append({
                "hadm_id": hid,
                "error": "sequence_length_mismatch",
                "raw_sequence_length": raw_len,
                "cache_sequence_length": cache_len,
            })
    if mismatches:
        preview = "; ".join(
            f"{row['hadm_id']}:{row.get('error')} raw={row.get('raw_sequence_length')} cache={row.get('cache_sequence_length')}"
            for row in mismatches[:5]
        )
        raise RuntimeError(
            f"ICD raw/cache length preflight failed for {len(mismatches)} selected admissions: {preview}"
        )
    return {"checked_admissions": int(len(selected_ids)), "mismatch_count": 0, "mismatches": []}


def _icd_run_label_cohort_attribution(*, cache_root: str, probe_checkpoint: str,
                                      label_index: int, labels_csv: str, output_dir: str,
                                      label_code: Optional[str] = None,
                                      cohort_max_admissions: int = 0,
                                      device: str = "cuda", ablation_batch_size: int = 32,
                                      phase1_ckpt: Optional[str] = None,
                                      data_dir: Optional[str] = None,
                                      schema_dir: Optional[str] = None,
                                      config: str = "configs/params.yaml",
                                      perturbation_methods: str = "drop,replace_with_train_feature_median,within_admission_same_vid_shuffle",
                                      panel_co_measurement_analysis: bool = False,
                                      seed: int = 42, parity_z_atol: float = 5.0e-3,
                                      parity_logit_atol: float = 5.0e-3) -> Dict[str, Any]:
    if not labels_csv:
        raise ValueError("cohort-level ICD attribution requires --labels_csv.")
    if not output_dir:
        raise ValueError("cohort-level ICD attribution requires --output_dir.")
    runtime = _icd_full_visible_runtime(
        cache_root=cache_root,
        probe_checkpoint=probe_checkpoint,
        label_index=int(label_index),
        label_code=label_code,
        device=device,
    )
    resolved_label_code = runtime.label_identity.label_code
    if not resolved_label_code:
        raise ValueError("cohort-level ICD attribution requires --label_code when the probe artifact has no label vocab code.")

    target_lookup = _icd_label_target_lookup(labels_csv, str(resolved_label_code), split="test")
    positive_ids = {hadm_id for hadm_id, target in target_lookup.items() if int(target) == 1}
    cache_ids = list(runtime.cache_index.keys())
    positive_in_cache = sorted(set(cache_ids) & positive_ids)
    missing_from_cache = sorted(positive_ids - set(cache_ids))
    candidate_ids = sorted(positive_in_cache)
    selected_ids = list(candidate_ids)
    if int(cohort_max_admissions) > 0:
        rng = np.random.default_rng(int(seed))
        k = min(int(cohort_max_admissions), len(candidate_ids))
        selected_ids = sorted(rng.choice(np.asarray(candidate_ids, dtype=object), size=k, replace=False).astype(str).tolist())
    if not selected_ids:
        raise RuntimeError(f"No positive test admissions found for ICD label {resolved_label_code!r}.")

    if panel_co_measurement_analysis and not (phase1_ckpt and data_dir and schema_dir):
        raise ValueError("--panel_co_measurement_analysis requires --phase1_ckpt, --data_dir, and --schema_dir.")
    if phase1_ckpt or data_dir or schema_dir:
        missing = [name for name, value in {
            "phase1_ckpt": phase1_ckpt, "data_dir": data_dir, "schema_dir": schema_dir,
        }.items() if not value]
        if missing:
            raise ValueError("Feature/VID ICD attribution requires: " + ", ".join(missing))
        _icd_attach_raw_runtime(
            runtime,
            phase1_ckpt=str(phase1_ckpt),
            data_dir=str(data_dir),
            schema_dir=str(schema_dir),
            config=config,
            raw_target_ids=selected_ids,
        )
    raw_cache_length_preflight = _icd_preflight_raw_cache_lengths(runtime, selected_ids)
    numeric_vids = _icd_numeric_vid_union(runtime, selected_ids)
    train_files_scanned = 0
    train_medians: Mapping[int, float] = {}
    if numeric_vids:
        train_files_scanned = len(_icd_split_files(str(data_dir), "train")) if data_dir else 0
        train_medians = _icd_train_feature_medians(
            data_dir=str(data_dir),
            schema_dir=str(schema_dir),
            vids=sorted(numeric_vids),
        )
    missing_median_vids = sorted(int(v) for v in numeric_vids if int(v) not in train_medians)

    staging = f"{os.path.abspath(output_dir)}.tmp_icd_cohort"
    if os.path.exists(staging):
        raise RuntimeError(f"Stale ICD cohort staging directory exists: {staging}")
    os.makedirs(staging, exist_ok=False)
    admission_root = os.path.join(staging, "admissions")
    summary_rows: List[Dict[str, Any]] = []
    vid_summary_input_path = os.path.join(staging, "_icd_vid_summary_admission_rows.csv")
    panel_summary_input_path = os.path.join(staging, "_icd_panel_summary_admission_rows.csv")
    failed: List[Dict[str, str]] = []
    if panel_co_measurement_analysis:
        _icd_write_csv_atomic(_icd_panel_membership_audit(str(schema_dir)), os.path.join(staging, "icd_panel_membership_audit.csv"))

    for hadm_id in tqdm(selected_ids, desc="icd_cohort_admissions", ascii=True):
        subdir = os.path.join(admission_root, str(hadm_id))
        try:
            result = _icd_attribute_admission_with_runtime(
                runtime=runtime,
                hadm_id=str(hadm_id),
                ablation_batch_size=int(ablation_batch_size),
                output_dir=subdir,
                labels_csv=labels_csv,
                target_lookup=target_lookup,
                perturbation_methods=perturbation_methods,
                seed=int(seed),
                parity_z_atol=float(parity_z_atol),
                parity_logit_atol=float(parity_logit_atol),
                train_medians=train_medians,
                panel_co_measurement_analysis=bool(panel_co_measurement_analysis),
                require_latent_parity=False,
                show_progress=False,
            )
        except Exception as exc:
            failed.append({"hadm_id": str(hadm_id), "error": str(exc)})
            break
        if result.get("true_target") != 1:
            failed.append({"hadm_id": str(hadm_id), "error": "selected admission was not label-positive on labels_csv recheck"})
            break
        summary_rows.append({
            "hadm_id": str(hadm_id),
            "label_index": int(label_index),
            "label_code": resolved_label_code,
            "true_target": result.get("true_target"),
            "logit": result.get("logit"),
            "probability": result.get("probability"),
            "sequence_length": result.get("sequence_length"),
            "feature_level_status": result.get("feature_level_status"),
        })
        for source_name, cohort_name in (
            ("icd_cached_token_occlusion.csv", "icd_cohort_cached_token_occlusion.csv"),
            ("icd_raw_timestamp_attribution.csv", "icd_cohort_raw_timestamp_attribution.csv"),
            ("icd_observation_attribution.csv", "icd_cohort_observation_attribution.csv"),
            ("icd_vid_attribution.csv", "icd_cohort_vid_attribution.csv"),
            ("icd_panel_attribution.csv", "icd_cohort_panel_attribution.csv"),
            ("icd_panel_timestamp_audit.csv", "icd_panel_timestamp_audit.csv"),
        ):
            frame = _icd_read_optional_csv(
                os.path.join(subdir, source_name),
                hadm_id=str(hadm_id),
                label_index=int(label_index),
                label_code=resolved_label_code,
            )
            if frame is not None:
                frame = _icd_add_cohort_columns(
                    frame,
                    hadm_id=str(hadm_id),
                    label_index=int(label_index),
                    label_code=resolved_label_code,
                    baseline_logit=result.get("logit"),
                    baseline_probability=result.get("probability"),
                )
                _icd_append_csv(frame, os.path.join(staging, cohort_name))
                if source_name == "icd_vid_attribution.csv":
                    _icd_append_csv(_icd_vid_summary_admission_rows(frame), vid_summary_input_path)
                if source_name == "icd_panel_attribution.csv":
                    _icd_append_csv(frame, panel_summary_input_path)
        shutil.rmtree(subdir, ignore_errors=True)
    if failed:
        failure_path = os.path.join(staging, "failure_summary.json")
        with open(failure_path, "w", encoding="utf-8") as handle:
            json.dump({"failed_admissions": failed}, handle, indent=2, allow_nan=False)
        raise RuntimeError(f"ICD cohort attribution failed: {failed[0]['hadm_id']} {failed[0]['error']}")

    summary_df = pd.DataFrame(summary_rows)
    _icd_write_csv_atomic(summary_df, os.path.join(staging, "icd_cohort_admissions.csv"))
    output_files = {"icd_cohort_admissions.csv": os.path.join(output_dir, "icd_cohort_admissions.csv")}
    for name in (
        "icd_cohort_cached_token_occlusion.csv",
        "icd_cohort_raw_timestamp_attribution.csv",
        "icd_cohort_observation_attribution.csv",
        "icd_cohort_vid_attribution.csv",
        "icd_cohort_panel_attribution.csv",
        "icd_panel_membership_audit.csv",
        "icd_panel_timestamp_audit.csv",
    ):
        if os.path.exists(os.path.join(staging, name)):
            output_files[name] = os.path.join(output_dir, name)
    if os.path.exists(vid_summary_input_path):
        vid_summary = _icd_cohort_vid_summary(pd.read_csv(vid_summary_input_path), cohort_size=len(selected_ids))
        path = os.path.join(staging, "icd_cohort_vid_summary.csv")
        _icd_write_csv_atomic(vid_summary, path)
        output_files["icd_cohort_vid_summary.csv"] = os.path.join(output_dir, "icd_cohort_vid_summary.csv")
        os.remove(vid_summary_input_path)
    if os.path.exists(panel_summary_input_path):
        panel_summary = _icd_cohort_panel_summary(pd.read_csv(panel_summary_input_path))
        path = os.path.join(staging, "icd_cohort_panel_summary.csv")
        _icd_write_csv_atomic(panel_summary, path)
        output_files["icd_cohort_panel_summary.csv"] = os.path.join(output_dir, "icd_cohort_panel_summary.csv")
        os.remove(panel_summary_input_path)

    result = {
        "analysis_scope": "icd_label_positive_cohort_descriptive_attribution",
        "cohort_level_inference": False,
        "descriptive_cohort_summary": True,
        "causal_interpretation": False,
        "scientific_interpretation": (
            "Descriptive aggregation of per-admission perturbation sensitivity among test admissions positive "
            "for the selected ICD label. This analysis does not retrain the probe, construct value-destroyed "
            "counterfactual datasets, estimate causal effects, or establish that workflow shortcuts are excluded."
        ),
        "label_index": int(label_index),
        "label_code": resolved_label_code,
        "label_vocabulary_source": runtime.label_identity.vocabulary_source,
        "phase1_checkpoint_path": phase1_ckpt,
        "phase1_checkpoint_sha256": runtime.phase1_sha256,
        "probe_checkpoint_path": probe_checkpoint,
        "probe_checkpoint_sha256": runtime.probe_sha256,
        "schema_path": os.path.join(str(schema_dir), "schema_processed.csv") if schema_dir else None,
        "schema_sha256": runtime.schema_sha256,
        "cache_root": cache_root,
        "cache_schema_version": runtime.cache_schema_version,
        "data_dir": data_dir,
        "labels_csv": labels_csv,
        "positive_admissions_in_labels_csv": int(len(positive_ids)),
        "positive_admissions_found_in_cache": int(len(positive_in_cache)),
        "positive_admissions_missing_from_cache": int(len(missing_from_cache)),
        "eligible_candidate_count": int(len(candidate_ids)),
        "n_analyzed_admissions": int(len(selected_ids)),
        "cohort_max_admissions": int(cohort_max_admissions),
        "sampling_policy": "all_sorted" if int(cohort_max_admissions) <= 0 else "stable_seeded_without_replacement_then_sorted",
        "seed": int(seed),
        "selected_admission_ids": selected_ids,
        "missing_cache_admission_ids": missing_from_cache,
        "perturbation_methods": [m.strip() for m in str(perturbation_methods).split(",") if m.strip()],
        "panel_co_measurement_analysis": {
            "enabled": bool(panel_co_measurement_analysis),
            "historical_definition": "BMP/CMP keywords and matching logic copied from removed icd_label_attribution.py; historical explicit VID whitelists were empty.",
            "value_neutralization_interpretation": "baseline logit minus panel-value-neutralized logit measures dependence on patient-specific quantitative chemistry values conditional on the observed pattern of what was measured, when, and how often.",
            "removal_interpretation": "Panel removal additionally perturbs measurement presence and contextual/set-composition information.",
            "timestamp_audit_interpretation": "Timestamp groups are BMP-like/CMP-like co-measurement groups; exact timestamp equality does not prove a common clinical order.",
            "causal_interpretation": False,
            "shortcut_learning_conclusion": "Neither perturbation is causal evidence or fully excludes shortcut learning.",
        },
        "parity_z_atol": float(parity_z_atol),
        "parity_logit_atol": float(parity_logit_atol),
        "successful_admissions": int(len(summary_rows)),
        "failed_admissions": [],
        "raw_cache_length_preflight": raw_cache_length_preflight,
        "train_median_provenance": {
            "requested_numeric_vid_count": int(len(numeric_vids)),
            "resolved_median_count": int(len(train_medians)),
            "missing_median_vids": missing_median_vids,
            "train_files_scanned": int(train_files_scanned),
            "median_source": "processed_train_split",
            "median_policy": "per-VID median over finite numeric values for selected cohort VID union",
        },
        "vid_summary_definitions": {
            "cohort_size": "selected admissions analyzed in this run",
            "n_present_admissions": "admissions where this VID appears in raw input",
            "presence_rate": "n_present_admissions / cohort_size",
            "n_evaluated_admissions": "admissions with finite evaluated attribution for this method and VID",
            "evaluation_rate_among_present": "n_evaluated_admissions / n_present_admissions",
            "positive_drop_fraction": "fraction of evaluated admission-level logit_drop values > 0",
        },
        "output_files": output_files,
    }
    path = os.path.join(staging, "icd_cohort_attribution_summary.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_icd_json_safe(result), handle, indent=2, allow_nan=False)
    os.replace(tmp, path)
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(staging):
        if name == "admissions":
            shutil.rmtree(os.path.join(staging, name), ignore_errors=True)
            continue
        os.replace(os.path.join(staging, name), os.path.join(output_dir, name))
    os.rmdir(staging)
    return result


def _icd_attribute_admission_with_runtime(*, runtime: _ICDAttributionRuntime, hadm_id: str,
                                          ablation_batch_size: int = 32,
                                          output_dir: Optional[str] = None,
                                          labels_csv: Optional[str] = None,
                                          target_lookup: Optional[Mapping[str, int]] = None,
                                          perturbation_methods: str = "drop,replace_with_train_feature_median,within_admission_same_vid_shuffle",
                                          seed: int = 42,
                                          parity_z_atol: float = 5.0e-3,
                                          parity_logit_atol: float = 5.0e-3,
                                          train_medians: Optional[Mapping[int, float]] = None,
                                          panel_co_measurement_analysis: bool = False,
                                          require_latent_parity: bool = True,
                                          show_progress: bool = True) -> Dict[str, Any]:
    if str(hadm_id) not in runtime.cache_index:
        raise KeyError(f"Admission not found in full_visible/test: {hadm_id}")
    record = runtime.cache_dataset[runtime.cache_index[str(hadm_id)]]
    latent = record[runtime.source].float()
    time_min = record["time_min"].float()
    batch = canonical_probe_collate([{"latent_seq": latent, "time_min": time_min, "target": 0, "hadm_id": str(hadm_id)}])
    batch = {key: value.to(runtime.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    with torch.inference_mode():
        logits = runtime.probe(batch)
    label_index = int(runtime.label_identity.label_index)
    if label_index < 0 or label_index >= logits.size(1):
        raise IndexError("label_index is outside the artifact vocabulary.")
    baseline_logit = logits[0, label_index]
    t = int(latent.size(0))
    cached_latent_token_occlusion = _icd_cached_latent_token_occlusion(
        model=runtime.probe, latent=latent, time_min=time_min, hadm_id=str(hadm_id),
        label_index=int(label_index), baseline_logit=float(baseline_logit.detach().cpu()),
        device=runtime.device, batch_size=int(ablation_batch_size),
    )
    raw_replay_requested = runtime.teacher is not None
    if raw_replay_requested and output_dir is None:
        raise ValueError("Feature/VID ICD attribution requires output_dir.")
    label_code = runtime.label_identity.label_code
    result: Dict[str, Any] = {"hadm_id": str(hadm_id), "source_name": runtime.source, "label_index": int(label_index),
            "analysis_scope": "single_admission_illustrative_analysis",
            "cohort_level_inference": False,
            "causal_interpretation": False,
            "scientific_interpretation": (
                "Qualitative illustration of prediction sensitivity for one retrospectively selected admission; "
                "not evidence that workflow shortcuts are excluded at the cohort level."
            ),
            "label_code": label_code,
            "label_vocabulary_source": runtime.label_identity.vocabulary_source,
            "true_target": int(target_lookup[str(hadm_id)]) if target_lookup is not None and str(hadm_id) in target_lookup else _icd_true_target(labels_csv, str(hadm_id), label_code),
            "logit": float(baseline_logit), "probability": float(torch.sigmoid(baseline_logit)),
            "sequence_length": t,
            "cached_latent_token_occlusion": cached_latent_token_occlusion,
            "phase1_checkpoint_sha256": runtime.phase1_sha256,
            "probe_checkpoint_sha256": runtime.probe_sha256,
            "schema_sha256": runtime.schema_sha256,
            "cache_schema_version": runtime.cache_schema_version,
            "seed": int(seed),
            "feature_level_status": "not_requested"}
    observation_df: Optional[pd.DataFrame] = None
    vid_df: Optional[pd.DataFrame] = None
    raw_timestamp_df: Optional[pd.DataFrame] = None
    panel_df: Optional[pd.DataFrame] = None
    panel_timestamp_df: Optional[pd.DataFrame] = None
    if raw_replay_requested:
        if runtime.raw_dataset is None or runtime.raw_index is None or str(hadm_id) not in runtime.raw_index:
            raise KeyError(f"Raw processed admission not found for ICD attribution: {hadm_id}")
        item = runtime.raw_dataset[runtime.raw_index[str(hadm_id)]]
        if len(item["state_counts"]) != t:
            raise RuntimeError(f"Raw/cache sequence length mismatch for {hadm_id}: raw={len(item['state_counts'])} cache={t}")
        label_map = runtime.label_map or {}
        type_map = runtime.type_map or {}
        obs_rows = _icd_observation_rows(item, label_map, type_map)
        parity = _icd_validate_raw_cache_parity(
            teacher=runtime.teacher, probe=runtime.probe, item=item, cache_record=record, source=runtime.source,
            label_index=int(label_index), cached_logit=float(baseline_logit.detach().cpu()),
            device=runtime.device, z_atol=float(parity_z_atol), logit_atol=float(parity_logit_atol),
            require_latent=bool(require_latent_parity),
            require_logit=bool(require_latent_parity),
        )
        methods = [m.strip() for m in str(perturbation_methods).split(",") if m.strip()]
        allowed = {"drop", "replace_with_train_feature_median", "within_admission_same_vid_shuffle"}
        bad = sorted(set(methods) - allowed)
        if bad:
            raise ValueError(f"Unsupported ICD perturbation_methods: {bad}")
        numeric_vids = sorted({int(row["vid"]) for row in obs_rows if str(row.get("feature_type")) == "numeric"})
        if train_medians is None:
            train_medians = _icd_train_feature_medians(
                data_dir=str(runtime.data_dir), schema_dir=str(runtime.schema_dir),
                vids=numeric_vids
            )
        raw_timestamp_df = _icd_raw_timestamp_attribution(
            teacher=runtime.teacher, probe=runtime.probe, source=runtime.source, item=item, label_index=int(label_index),
            baseline_logit=float(baseline_logit.detach().cpu()),
            batch_size=int(ablation_batch_size), device=runtime.device, show_progress=bool(show_progress),
        )
        observation_df, vid_df, recomputed = _icd_feature_attribution(
            teacher=runtime.teacher, probe=runtime.probe, source=runtime.source, item=item, label_index=int(label_index),
            rows=obs_rows, methods=methods,
            batch_size=int(ablation_batch_size), device=runtime.device, train_medians=train_medians,
            seed=int(seed), show_progress=bool(show_progress),
        )
        if panel_co_measurement_analysis:
            panel_df, panel_timestamp_df = _icd_panel_attribution(
                teacher=runtime.teacher, probe=runtime.probe, source=runtime.source, item=item,
                rows=obs_rows, label_index=int(label_index), baseline_logit=float(baseline_logit),
                batch_size=int(ablation_batch_size), device=runtime.device, train_medians=train_medians,
            )
        result.update({
            "feature_level_status": "complete",
            "raw_cache_parity": parity,
            "raw_recomputed_logit": float(recomputed),
            "raw_recomputed_logit_delta_vs_cache": float(recomputed - float(baseline_logit.detach().cpu())),
            "observation_attribution_rows": int(len(observation_df)),
            "vid_attribution_rows": int(len(vid_df)),
            "perturbation_methods": methods,
        })
        if panel_df is not None:
            result["panel_co_measurement_analysis"] = {
                "enabled": True,
                "panel_attribution_rows": int(len(panel_df)),
                "timestamp_audit_rows": int(len(panel_timestamp_df)),
            }
    if output_dir:
        _write_icd_outputs(str(output_dir), result, observation_df, vid_df, raw_timestamp_df, panel_df, panel_timestamp_df)
    return result


def _icd_attribute_admission(*, cache_root: str, probe_checkpoint: str, hadm_id: str,
                        label_index: int, device: str = "cuda", ablation_batch_size: int = 32,
                        phase1_ckpt: Optional[str] = None, data_dir: Optional[str] = None,
                        schema_dir: Optional[str] = None, config: str = "configs/params.yaml",
                        output_dir: Optional[str] = None, labels_csv: Optional[str] = None,
                        perturbation_methods: str = "drop,replace_with_train_feature_median,within_admission_same_vid_shuffle",
                        seed: int = 42, parity_z_atol: float = 5.0e-3,
                        parity_logit_atol: float = 5.0e-3,
                        label_code: Optional[str] = None) -> Dict[str, Any]:
    runtime = _icd_full_visible_runtime(
        cache_root=cache_root,
        probe_checkpoint=probe_checkpoint,
        label_index=int(label_index),
        label_code=label_code,
        device=device,
        phase1_ckpt=phase1_ckpt,
        data_dir=data_dir,
        schema_dir=schema_dir,
        config=config,
        raw_target_ids=[str(hadm_id)] if (phase1_ckpt or data_dir or schema_dir) else None,
    )
    return _icd_attribute_admission_with_runtime(
        runtime=runtime,
        hadm_id=str(hadm_id),
        ablation_batch_size=int(ablation_batch_size),
        output_dir=output_dir,
        labels_csv=labels_csv,
        perturbation_methods=perturbation_methods,
        seed=int(seed),
        parity_z_atol=float(parity_z_atol),
        parity_logit_atol=float(parity_logit_atol),
    )


def _icd_numeric_vids(schema_dir: str) -> set[int]:
    _, type_map = _icd_schema_maps(schema_dir)
    return {int(vid) for vid, typ in type_map.items() if typ == "numeric"}


def _icd_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _icd_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_icd_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _icd_split_files(data_dir: str, split: str, max_patients: int = 0) -> List[str]:
    split_dir = os.path.join(data_dir, split)
    files = sorted(
        os.path.join(split_dir, name)
        for name in os.listdir(split_dir)
        if name.endswith(".pt") and os.path.isfile(os.path.join(split_dir, name))
    )
    return files[:int(max_patients)] if int(max_patients) > 0 else files


def _icd_flush_median_value_buffers(buffers: Dict[int, List[float]], root: str, counters: Dict[int, int]) -> None:
    for vid, vals in list(buffers.items()):
        if not vals:
            continue
        shard_dir = os.path.join(root, "median_value_shards", f"vid={int(vid)}")
        os.makedirs(shard_dir, exist_ok=True)
        part = int(counters[int(vid)])
        counters[int(vid)] += 1
        path = os.path.join(shard_dir, f"part-{part:06d}.npy")
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as handle:
            np.save(handle, np.asarray(vals, dtype=np.float32))
        os.replace(tmp, path)
        buffers[vid] = []


def _icd_main() -> None:
    parser = argparse.ArgumentParser(description="Canonical exact-time ICD attribution.")
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--probe_checkpoint")
    parser.add_argument("--hadm_id")
    parser.add_argument("--label_index", type=int)
    parser.add_argument("--label_code")
    parser.add_argument("--cohort_level", action="store_true")
    parser.add_argument("--cohort_max_admissions", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation_batch_size", type=int, default=32)
    parser.add_argument("--phase1_ckpt")
    parser.add_argument("--data_dir")
    parser.add_argument("--schema_dir")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--output_dir")
    parser.add_argument("--labels_csv")
    parser.add_argument("--perturbation_methods", default="drop,replace_with_train_feature_median,within_admission_same_vid_shuffle")
    parser.add_argument("--panel_co_measurement_analysis", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parity_z_atol", type=float, default=5.0e-3)
    parser.add_argument("--parity_logit_atol", type=float, default=5.0e-3)
    args = parser.parse_args()
    if args.cohort_level:
        if not args.probe_checkpoint or args.label_index is None:
            raise SystemExit("cohort-level ICD attribution requires --probe_checkpoint and --label_index")
        cohort_args = vars(args).copy()
        cohort_args.pop("cohort_level", None)
        cohort_args.pop("hadm_id", None)
        print(json.dumps(_icd_run_label_cohort_attribution(**cohort_args), indent=2))
        return
    if not args.probe_checkpoint or not args.hadm_id or args.label_index is None:
        raise SystemExit("single-admission attribution requires --probe_checkpoint, --hadm_id and --label_index")
    if args.panel_co_measurement_analysis:
        raise SystemExit("--panel_co_measurement_analysis is available only with --cohort_level")
    single_args = vars(args).copy()
    for key in ("cohort_level", "cohort_max_admissions", "panel_co_measurement_analysis"):
        single_args.pop(key, None)
    print(json.dumps(_icd_attribute_admission(**single_args), indent=2))
