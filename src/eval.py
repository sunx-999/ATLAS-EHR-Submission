#!/usr/bin/env python3
"""Evaluation for the single exact-time downstream probe implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache_bank import CACHE_SCHEMA_VERSION, FULL_VISIBLE, load_dataset
from src.probes import (
    PROBE_ARCHITECTURE_VERSION, AttentionCostBatchSampler, CanonicalProbeDataset,
    TASK_CACHE_KIND, canonical_probe_collate, _epoch, _make_task_model, _targets_for_task, _task_metric,
    _vocab_hash, multilabel_threshold_metrics,
)
from src.utils import load_config_dict


FORMAL_TASK_ORDER = (
    "mortality", "los_ge_3d", "los_ge_7d", "readmission", "icd", "hcup", "drg",
)
FORMAL_SOURCE_ORDER = ("z_set", "z_context")


def _canonical_probe_cfg(cfg_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Return probe config with runtime defaults for legacy Phase-1-only YAMLs."""
    cfg = dict(cfg_dict.get("probe", {}) or {})
    training = dict(cfg.get("training", {}) or {})
    training.setdefault("max_pairwise_elements_per_batch", 262144)
    training.setdefault("max_patients_per_batch", 64)
    training.setdefault("prefetch_factor", 2)
    cfg["training"] = training
    cfg.setdefault("tasks", {})
    return cfg


def _canonical_probe_training_cfg(cfg_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(_canonical_probe_cfg(cfg_dict)["training"])


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exact_time_probe(path: str, *, device: torch.device,
                          expected_task: Optional[str] = None,
                          expected_source: Optional[str] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {"architecture_version", "task", "source_name", "encoder", "latent_dim", "state_dict",
                "cache_schema_version", "phase1_checkpoint_sha256", "schema_sha256"}
    if not isinstance(artifact, dict) or required.difference(artifact):
        raise RuntimeError("Invalid legacy or incomplete exact-time probe artifact.")
    if artifact["architecture_version"] != PROBE_ARCHITECTURE_VERSION:
        raise RuntimeError("Probe architecture version mismatch; legacy checkpoint is rejected.")
    if expected_task is not None and artifact["task"] != expected_task:
        raise RuntimeError("Probe task does not match the requested evaluation task.")
    if expected_source is not None and artifact["source_name"] != expected_source:
        raise RuntimeError("Probe latent source does not match the requested source.")
    vocab = artifact.get("label_vocab")
    n_labels = None if vocab is None else len(vocab.get("codes", []))
    model = _make_task_model(str(artifact["task"]), int(artifact["latent_dim"]),
                             {"encoder": artifact["encoder"]}, n_labels, str(artifact["source_name"]))
    model.load_state_dict(artifact["state_dict"], strict=True)
    model = model.float().to(device).eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Loaded evaluation probe is not fully frozen.")
    return model, artifact


def _write_prediction_npz(path: str, task: str, ids: list[str], target: torch.Tensor, logits: Any) -> None:
    payload: Dict[str, Any] = {"hadm_id": np.asarray(ids), "target": target.detach().cpu().numpy()}
    if task == "drg":
        payload["target"] = payload["target"].astype(np.uint8, copy=False)
        payload["label_logits"] = logits[0].detach().cpu().to(torch.float16).numpy()
        payload["cardinality_logits"] = logits[1].detach().cpu().to(torch.float16).numpy()
    else:
        payload["target"] = (payload["target"].astype(np.uint8, copy=False)
                             if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission", "icd", "hcup"}
                             else payload["target"])
        payload["logits"] = logits.detach().cpu().to(torch.float16).numpy()
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def evaluate_exact_time_probe(*, probe_checkpoint: str, cache_root: str, labels_csv: str,
                              cfg_dict: Mapping[str, Any], split: str, task: Optional[str] = None,
                              cutoff_min: Optional[int] = None, device: str = "cuda",
                              num_workers: int = 0,
                              targets_override: Optional[Mapping[str, Any]] = None,
                              phase1_ckpt: Optional[str] = None,
                              external_transfer: bool = False,
                              prediction_path: Optional[str] = None,
                              model_override: Optional[torch.nn.Module] = None,
                              artifact_override: Optional[Mapping[str, Any]] = None,
                              base_override: Optional[Any] = None,
                              lengths_override: Optional[Sequence[int]] = None,
                              precision: str = "32-true",
                              expected_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    requested = task
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if model_override is None:
        model, artifact = load_exact_time_probe(probe_checkpoint, device=resolved_device, expected_task=requested)
    else:
        model = model_override.float().to(resolved_device).eval()
        model.requires_grad_(False)
        artifact = dict(artifact_override or {})
        if not artifact:
            raise ValueError("artifact_override is required with model_override.")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Evaluation probe must be eval-mode and fully frozen.")
    if not phase1_ckpt:
        raise ValueError("Evaluation requires the exact --phase1_ckpt used for cache and probe training.")
    phase1_ckpt = os.path.realpath(phase1_ckpt)
    if not os.path.isfile(phase1_ckpt):
        raise FileNotFoundError(f"--phase1_ckpt not found: {phase1_ckpt}")
    cli_checkpoint_sha256 = _sha256(phase1_ckpt)
    task, source = str(artifact["task"]), str(artifact["source_name"])
    if source not in {"z_set", "z_context"}:
        raise RuntimeError("Probe artifact carries a non-canonical latent source.")
    manifest_path = os.path.join(cache_root, "cache_manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            cache_signature = json.load(handle)["cache_signature"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("Evaluation requires a signed canonical cache manifest.") from exc
    if cache_signature.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError("Cache schema version mismatch; rebuild the cache.")
    for key in (("phase1_checkpoint_sha256",) if external_transfer else ("phase1_checkpoint_sha256", "schema_sha256")):
        if artifact.get(key) != cache_signature.get(key):
            raise RuntimeError(f"Probe/cache {key} mismatch; zero-shot evaluation is refused.")
    artifact_signature = artifact.get("cache_signature")
    if not isinstance(artifact_signature, dict):
        raise RuntimeError("Probe artifact lacks its immutable training cache signature.")
    if not external_transfer:
        if artifact_signature != cache_signature:
            raise RuntimeError("Internal evaluation requires an exact training-cache signature match.")
    else:
        required_external_keys = ("phase1_checkpoint_sha256", "phase1_architecture_version", "latent_dimension",
                                  "cache_schema_version", "cache_contract_version")
        mismatched = [key for key in required_external_keys if artifact_signature.get(key) != cache_signature.get(key)]
        if mismatched:
            details = {
                key: {"probe": artifact_signature.get(key), "external_cache": cache_signature.get(key)}
                for key in mismatched
            }
            raise RuntimeError(f"External transfer provenance mismatch: {details}; only dataset/cohort/provenance-only fields may differ.")
    if cli_checkpoint_sha256 != cache_signature.get("phase1_checkpoint_sha256"):
        raise RuntimeError("CLI --phase1_ckpt SHA256 does not match cache/probe provenance.")
    if task not in TASK_CACHE_KIND:
        raise ValueError(f"Unsupported formal probe task {task!r}.")
    kind, expected_cutoff = TASK_CACHE_KIND[task]
    if kind == FULL_VISIBLE:
        if cutoff_min is not None:
            raise ValueError(f"{task} uses full_visible and does not accept --cutoff_min.")
    elif cutoff_min is None:
        cutoff_min = int(expected_cutoff)
    elif int(cutoff_min) != int(expected_cutoff):
        raise ValueError(f"{task} requires cutoff_min={expected_cutoff}, got {cutoff_min}.")
    base = (base_override if base_override is not None else load_dataset(
        cache_root, kind, split=split,
        cutoff_min=None if kind == FULL_VISIBLE else int(cutoff_min)))
    artifact_vocab = artifact.get("label_vocab")
    if task in {"icd", "hcup"}:
        try:
            _validate_multilabel_threshold_artifact(artifact, probe_checkpoint)
            thresholds = artifact["thresholds"]
        except RuntimeError:
            if os.path.basename(probe_checkpoint) != "last.pt":
                raise
            thresholds = None
            print(f"[eval] WARNING: {probe_checkpoint} has no validation thresholds; ranking metrics only.", flush=True)
    probe_cfg = _canonical_probe_cfg(cfg_dict)
    targets, _ = (_targets_for_task(labels_csv, task, probe_cfg, vocab_override=artifact_vocab,
                                    allowed_splits=(split,))
                  if targets_override is None else (dict(targets_override), None))
    dataset = CanonicalProbeDataset(base, source, targets)
    lengths = list(lengths_override) if lengths_override is not None else base.sequence_lengths()
    train_cfg = _canonical_probe_training_cfg(cfg_dict)
    loader_kwargs: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": resolved_device.type == "cuda",
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    loader = DataLoader(dataset, batch_sampler=AttentionCostBatchSampler(
        lengths, int(train_cfg["max_pairwise_elements_per_batch"]),
        int(train_cfg["max_patients_per_batch"]), False), collate_fn=canonical_probe_collate,
        **loader_kwargs)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    with torch.inference_mode():
        _, ids, target, logits = _epoch(model, loader, task, resolved_device, None, {}, precision=precision)
    if len(set(ids)) != len(ids):
        raise RuntimeError("Evaluation prediction IDs contain duplicates.")
    if expected_ids is not None:
        expected = [str(x) for x in expected_ids]
        if len(set(expected)) != len(expected) or list(ids) != expected:
            raise RuntimeError("Evaluation prediction IDs are not identical to the canonical cache order.")
    score, metrics = _task_metric(task, target, logits)
    if task in {"icd", "hcup"} and thresholds is not None:
        metrics.update(multilabel_threshold_metrics(target, logits, artifact["thresholds"]))
    if prediction_path is not None:
        _write_prediction_npz(prediction_path, task, ids, target, logits)
    report = {"probe_checkpoint": os.path.abspath(probe_checkpoint), "probe_checkpoint_sha256": _sha256(probe_checkpoint),
              "task": task, "source_name": source, "split": split, "score": score, "metrics": metrics,
              "cache_schema_version": CACHE_SCHEMA_VERSION, "phase1_checkpoint_sha256": cli_checkpoint_sha256,
              "phase1_checkpoint_path": phase1_ckpt, "n": len(ids), "cache_kind": kind,
              "cutoff_min": cutoff_min, "runtime_seconds": time.perf_counter() - started,
              "precision": precision,
              "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else 0,
              "probe_architecture": {"version": artifact["architecture_version"], "encoder": artifact["encoder"]}}
    return report


def _suite_selector(value: str, allowed: Sequence[str], order: Sequence[str]) -> Tuple[str, ...]:
    requested = list(order) if str(value).strip().lower() == "all" else [x.strip() for x in str(value).split(",") if x.strip()]
    unknown = sorted(set(requested).difference(allowed))
    if unknown or not requested:
        raise ValueError(f"Invalid suite selection: {unknown or value!r}.")
    if len(set(requested)) != len(requested):
        raise ValueError("Suite task/source selections must be unique.")
    return tuple(x for x in order if x in requested)


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


def _suite_window(task: str) -> str:
    kind, cutoff = TASK_CACHE_KIND[task]
    if kind == FULL_VISIBLE:
        return "full"
    return "24h" if int(cutoff) == 1440 else "48h"


def _validate_multilabel_threshold_artifact(artifact: Mapping[str, Any], checkpoint: str) -> None:
    """Require thresholds fitted on validation before any formal test forward."""
    vocab = artifact.get("label_vocab")
    thresholds = artifact.get("thresholds")
    expected_hash = artifact.get("vocab_hash")
    if not isinstance(vocab, Mapping) or not isinstance(thresholds, Mapping):
        raise RuntimeError(
            f"{checkpoint} has no validation threshold artifact. "
            "Re-run the ICD/HCUP probe training to completion; test evaluation never fits thresholds."
        )
    if thresholds.get("vocab_hash") != expected_hash or expected_hash != _vocab_hash(vocab):
        raise RuntimeError(
            f"{checkpoint} has a threshold/vocabulary mismatch. "
            "Re-run the ICD/HCUP probe training with the checkpoint's training vocabulary."
        )
    values = np.asarray(thresholds.get("per_label", []), dtype=np.float64)
    codes = vocab.get("codes")
    if not isinstance(codes, Sequence) or values.shape != (len(codes),) or not np.isfinite(values).all() or not np.all((values > 0) & (values < 1)):
        raise RuntimeError(f"{checkpoint} has an invalid validation threshold artifact.")


def _resolve_suite_checkpoint(probe_root: str, task: str, source: str) -> str:
    checkpoint_dir = os.path.join(probe_root, task, source, "checkpoints")
    best = os.path.join(checkpoint_dir, "best.pt")
    if os.path.isfile(best):
        return best
    last = os.path.join(checkpoint_dir, "last.pt")
    if os.path.isfile(last):
        return last
    raise FileNotFoundError(f"Neither best.pt nor last.pt exists for {task}/{source}: {checkpoint_dir}")


def evaluate_mimic_probe_suite(
    *, probe_root: str, phase1_ckpt: str, cache_root: str,
    labels_csv: str, cfg_dict: Mapping[str, Any], output_dir: str,
    tasks: Sequence[str] = FORMAL_TASK_ORDER,
    sources: Sequence[str] = FORMAL_SOURCE_ORDER,
    split: str = "test", device: str = "cuda", num_workers: int = 0,
    save_predictions: bool = False, precision: str = "32-true",
    allow_non_test_split: bool = False,
) -> Dict[str, Any]:
    """Run the fixed, inference-only MIMIC test suite in one process."""
    if split != "test" and not allow_non_test_split:
        raise ValueError("Formal MIMIC suite evaluation is test-only.")
    if len(set(tasks)) != len(tuple(tasks)) or len(set(sources)) != len(tuple(sources)):
        raise ValueError("Suite task/source selections must be unique.")
    requested_tasks, requested_sources = set(tasks), set(sources)
    invalid_tasks = requested_tasks.difference(FORMAL_TASK_ORDER)
    invalid_sources = requested_sources.difference(FORMAL_SOURCE_ORDER)
    if invalid_tasks or invalid_sources:
        raise ValueError(f"Unknown suite task/source: tasks={sorted(invalid_tasks)}, sources={sorted(invalid_sources)}")
    task_order = tuple(t for t in FORMAL_TASK_ORDER if t in requested_tasks)
    source_order = tuple(s for s in FORMAL_SOURCE_ORDER if s in requested_sources)
    if not task_order or not source_order:
        raise ValueError("Suite requires at least one formal task and source.")
    phase1_path = os.path.realpath(phase1_ckpt)
    if not os.path.isfile(phase1_path):
        raise FileNotFoundError(f"--phase1_ckpt not found: {phase1_path}")
    cli_phase1_sha = _sha256(phase1_path)
    manifest_path = os.path.join(cache_root, "cache_manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            cache_signature = json.load(handle)["cache_signature"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("Suite requires a canonical cache manifest.") from exc
    if cache_signature.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError("Cache schema version mismatch; rebuild the cache.")
    if cache_signature.get("phase1_checkpoint_sha256") != cli_phase1_sha:
        raise RuntimeError("CLI Phase-1 checkpoint does not match canonical cache provenance.")
    requested: list[Tuple[str, str, str]] = []
    for task in task_order:
        for source in source_order:
            checkpoint = _resolve_suite_checkpoint(probe_root, task, source)
            requested.append((task, source, checkpoint))
    print("task | source | checkpoint | cache window | status", flush=True)
    artifacts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    models: Dict[Tuple[str, str], torch.nn.Module] = {}
    for task, source, checkpoint in tqdm(requested, desc="MIMIC test evaluation", unit="probe", dynamic_ncols=True):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Required suite checkpoint is missing: {checkpoint}")
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(artifact, dict) or artifact.get("task") != task or artifact.get("source_name") != source:
            raise RuntimeError(f"Artifact task/source mismatch for {checkpoint}")
        if artifact.get("architecture_version") != PROBE_ARCHITECTURE_VERSION:
            raise RuntimeError(f"Artifact architecture mismatch for {checkpoint}")
        if artifact.get("phase1_checkpoint_sha256") != cli_phase1_sha:
            raise RuntimeError(f"Artifact Phase-1 provenance mismatch for {checkpoint}")
        if artifact.get("cache_signature") != cache_signature:
            raise RuntimeError(f"Artifact/cache signature mismatch for {checkpoint}")
        if task in {"icd", "hcup"}:
            try:
                _validate_multilabel_threshold_artifact(artifact, checkpoint)
            except RuntimeError:
                if os.path.basename(checkpoint) != "last.pt":
                    raise
                print(f"[eval] WARNING: using {checkpoint} without validation thresholds; ranking metrics only.", flush=True)
        # Construct and freeze each artifact before any test forward so a bad
        # request cannot leave a partial formal result directory.
        model, _ = load_exact_time_probe(checkpoint, device=torch.device("cpu"), expected_task=task, expected_source=source)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(f"Artifact is not frozen: {checkpoint}")
        artifacts[(task, source)] = artifact
        models[(task, source)] = model
        print(f"{task} | {source} | {checkpoint} | {_suite_window(task)} | ready", flush=True)

    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    stage_cache: Dict[Tuple[str, Optional[int]], Any] = {}
    stage_lengths: Dict[Tuple[str, Optional[int]], Sequence[int]] = {}
    for task in task_order:
        kind, cutoff = TASK_CACHE_KIND[task]
        key = (kind, None if kind == FULL_VISIBLE else int(cutoff))
        if key not in stage_cache:
            stage_cache[key] = load_dataset(cache_root, kind, split=split, cutoff_min=key[1])
            stage_lengths[key] = stage_cache[key].sequence_lengths()
    targets_by_task: Dict[str, Mapping[str, Any]] = {}
    label_counts: Dict[str, Dict[str, int]] = {}
    for task in task_order:
        artifact_vocab = artifacts[(task, source_order[0])].get("label_vocab")
        targets, _ = _targets_for_task(labels_csv, task, _canonical_probe_cfg(cfg_dict), vocab_override=artifact_vocab,
                                       allowed_splits=(split,))
        targets_by_task[task] = targets
        kind, cutoff = TASK_CACHE_KIND[task]
        key = (kind, None if kind == FULL_VISIBLE else int(cutoff))
        base_ids = [str(x) for x in stage_cache[key].admission_ids()]
        if len(base_ids) != len(set(base_ids)):
            raise RuntimeError(f"Duplicate test admission IDs in {kind}.")
        missing = sorted(set(base_ids).difference(targets))
        if missing:
            raise RuntimeError(f"Test cache admissions lack targets for {task}: {missing[:20]}")
        label_counts[task] = {
            "label_test_count": len(targets), "cache_test_count": len(base_ids),
            "evaluated_test_count": len(base_ids),
            "label_only_excluded_count": len(set(targets).difference(base_ids)),
        }
    # Vocabularies are part of the artifact contract and must agree between sources.
    for task in task_order:
        if task not in {"icd", "hcup", "drg"}:
            continue
        first = artifacts[(task, source_order[0])].get("label_vocab")
        for source in source_order[1:]:
            if artifacts[(task, source)].get("label_vocab") != first:
                raise RuntimeError(f"{task} source vocabularies do not match.")

    results: list[Dict[str, Any]] = []
    suite_started = time.perf_counter()
    total = len(requested)
    for index, (task, source, checkpoint) in enumerate(requested, 1):
        kind, cutoff = TASK_CACHE_KIND[task]
        key = (kind, None if kind == FULL_VISIBLE else int(cutoff))
        artifact = artifacts[(task, source)]
        model = models.pop((task, source))
        base_ids = [str(x) for x in stage_cache[key].admission_ids()]
        train_cfg = _canonical_probe_training_cfg(cfg_dict)
        sampler = AttentionCostBatchSampler(
            stage_lengths[key], int(train_cfg["max_pairwise_elements_per_batch"]),
            int(train_cfg["max_patients_per_batch"]), False)
        prediction_order = [base_ids[index] for batch in sampler for index in batch]
        prediction_path = (os.path.join(output_dir, "predictions", f"{task}_{source}_test.npz")
                           if save_predictions else None)
        if prediction_path:
            os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        run = evaluate_exact_time_probe(
            probe_checkpoint=checkpoint, cache_root=cache_root, labels_csv=labels_csv,
            cfg_dict=cfg_dict, split=split, task=task, cutoff_min=cutoff,
            device=str(resolved_device), num_workers=num_workers,
            targets_override=targets_by_task[task], phase1_ckpt=phase1_path,
            prediction_path=prediction_path, model_override=model,
            artifact_override=artifact, base_override=stage_cache[key],
            lengths_override=stage_lengths[key], precision=precision,
            expected_ids=prediction_order,
        )
        run.update(label_counts[task])
        run["checkpoint"] = os.path.abspath(checkpoint)
        run["window"] = _suite_window(task)
        run["primary_metric"] = "micro_auprc" if task == "icd" else "macro_auprc" if task == "hcup" else "micro_f1" if task == "drg" else "auprc"
        run["primary_value"] = run["score"]
        results.append(run)
        metrics = run["metrics"]
        print(f"[{index}/{total}] {task}/{source} n={run['n']} AUPRC={metrics.get('auprc', metrics.get('micro_auprc', metrics.get('macro_auprc', metrics.get('micro_f1', float('nan'))))):.4f} AUROC={metrics.get('auroc', metrics.get('micro_auroc', float('nan'))):.4f} sec={run['runtime_seconds']:.1f}", flush=True)
        del model, run

    compact_specs = {
        "binary": ("auprc", ("auroc", "brier", "ece")),
        "icd": ("micro_auprc", ("macro_auprc", "micro_auroc", "macro_auroc")),
        "hcup": ("macro_auprc", ("micro_auprc", "macro_auroc", "micro_auroc")),
        "drg": ("micro_f1", ("macro_f1", "exact_set_match", "cardinality_accuracy")),
    }
    compact: list[Dict[str, Any]] = []
    for run in results:
        primary, secondary = compact_specs["binary" if run["task"] in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"} else run["task"]]
        row = {"task": run["task"], "source": run["source_name"], "window": run["window"], "n": run["n"],
               "primary_metric": primary, "primary_value": run["metrics"].get(primary)}
        for idx, name in enumerate(secondary, 1):
            row[f"secondary_{idx}"] = name
            row[f"secondary_{idx}_value"] = run["metrics"].get(name)
        row["runtime_seconds"] = run["runtime_seconds"]
        compact.append(row)
    os.makedirs(output_dir, exist_ok=True)
    fields = ["task", "source", "window", "n", "primary_metric", "primary_value", "secondary_1", "secondary_1_value", "secondary_2", "secondary_2_value", "secondary_3", "secondary_3_value", "runtime_seconds"]
    csv_lines = []
    from io import StringIO
    csv_buffer = StringIO(); writer = csv.DictWriter(csv_buffer, fieldnames=fields); writer.writeheader(); writer.writerows(_json_safe(compact)); csv_lines.append(csv_buffer.getvalue())
    _atomic_text(os.path.join(output_dir, "mimic_test_metrics_compact.csv"), "".join(csv_lines))
    md = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in compact:
        md.append("| " + " | ".join("" if row.get(field) is None else (f"{row[field]:.4f}" if isinstance(row[field], float) else str(row[field])) for field in fields) + " |")
    _atomic_text(os.path.join(output_dir, "mimic_test_metrics_compact.md"), "\n".join(md) + "\n")
    full = {"split": split, "tasks": list(task_order), "sources": list(source_order), "total_runs": len(results),
            "total_runtime_seconds": time.perf_counter() - suite_started, "results": _json_safe(results)}
    _atomic_text(os.path.join(output_dir, "mimic_test_metrics_full.json"), json.dumps(full, indent=2, allow_nan=False))
    print("\n".join(md), flush=True)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a canonical exact-time downstream probe.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--probe_checkpoint")
    checkpoint_group.add_argument("--probe_root")
    parser.add_argument("--phase1_ckpt", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", default=None, choices=["mortality", "los_ge_3d", "los_ge_7d", "readmission", "icd", "hcup", "drg"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--cutoff_min", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--allow_non_test_split", action="store_true")
    parser.add_argument("--precision", default="32-true", choices=["32-true", "bf16-mixed", "fp16-mixed"])
    args = parser.parse_args()
    cfg = load_config_dict(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.probe_root:
        if args.cutoff_min is not None:
            raise ValueError("Suite mode does not accept --cutoff_min.")
        if args.task is not None:
            raise ValueError("Suite mode uses --tasks, not --task.")
        task_order = _suite_selector(args.tasks, FORMAL_TASK_ORDER, FORMAL_TASK_ORDER)
        source_order = _suite_selector(args.sources, FORMAL_SOURCE_ORDER, FORMAL_SOURCE_ORDER)
        evaluate_mimic_probe_suite(
            probe_root=args.probe_root, phase1_ckpt=args.phase1_ckpt, cache_root=args.cache_root,
            labels_csv=args.labels_csv, cfg_dict=cfg, output_dir=args.output_dir,
            tasks=task_order, sources=source_order, split=args.split, device=args.device,
            num_workers=args.num_workers, save_predictions=args.save_predictions,
            precision=args.precision, allow_non_test_split=args.allow_non_test_split,
        )
        return
    prediction_path = os.path.join(args.output_dir, f"predictions_{args.split}.npz") if args.save_predictions else None
    report = evaluate_exact_time_probe(probe_checkpoint=args.probe_checkpoint, cache_root=args.cache_root,
        labels_csv=args.labels_csv, cfg_dict=cfg, split=args.split, task=args.task, cutoff_min=args.cutoff_min,
        device=args.device, num_workers=args.num_workers, phase1_ckpt=args.phase1_ckpt,
        prediction_path=prediction_path, precision=args.precision)
    _atomic_text(os.path.join(args.output_dir, "evaluation.json"), json.dumps(_json_safe(report), indent=2, allow_nan=False))
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
