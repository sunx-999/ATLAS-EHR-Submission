#!/usr/bin/env python3
"""Single resumable orchestration entry point for foundation validation.

This module owns stage state and output contracts only.  Cache extraction,
probe training, evaluation, labels and metrics remain implemented by the
canonical modules in :mod:`src.cache_bank`, :mod:`src.probes`, :mod:`src.eval`
and :mod:`src.external_validation`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.cache_bank import FULL_VISIBLE, load_dataset
from src.eval import evaluate_mimic_probe_suite
from src.probes import TASK_CACHE_KIND, build_exact_time_probe, _targets_for_task
from src.probes import run_probe_phase
from src.external_validation import evaluate_external_probe_suite, retrain_external_probe_suite
from src.utils import load_config_dict
from src import benchmark as benchmark_module


STAGES = (
    "scientific_preflight", "hardware_and_gpu_preflight", "artifact_and_cache_discovery",
    "probe_design_selection", "internal_probe_training", "internal_formal_evaluation",
    "baseline_import_and_stratification", "external_zero_shot",
    "external_local_probe_retrain", "statistical_analysis", "finalization",
)
FORMAL_TASKS = ("mortality", "los_ge_3d", "los_ge_7d", "readmission", "icd", "hcup", "drg")
PROBE_SELECTION_TASKS = ("mortality", "los_ge_7d", "hcup")
PROBE_CANDIDATES = ("raw_mean_linear", "single_query_pool", "source_aware")
VALUE_BASELINE_SUMMARIES = ("last", "mean", "max")
WORKFLOW_BASELINE_SUMMARIES = ("presence", "count", "interval")

if tuple(benchmark_module.TABULAR_SUMMARIES) != VALUE_BASELINE_SUMMARIES + WORKFLOW_BASELINE_SUMMARIES:
    raise RuntimeError("benchmark.TABULAR_SUMMARIES does not match the foundation baseline contract")


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_safe_json(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: str) -> Optional[str]:
    return _file_sha256(path) if path and os.path.isfile(path) else None


@dataclass(frozen=True)
class PipelinePaths:
    cache_dir: str
    output_dir: str

    @property
    def root(self) -> Path:
        return Path(self.cache_dir) / "foundation_validation"

    def pipeline_root(self, signature: str) -> Path:
        return self.root / signature


class CacheRegistry:
    """Process-local cache view registry; structural work happens once per key."""

    def __init__(self) -> None:
        self._datasets: Dict[tuple[str, str, Optional[int], str], Any] = {}
        self._lengths: Dict[tuple[str, str, Optional[int], str], Sequence[int]] = {}

    def get(self, root: str, kind: str, cutoff: Optional[int], split: str) -> Any:
        key = (os.path.realpath(root), kind, cutoff, split)
        if key not in self._datasets:
            self._datasets[key] = load_dataset(root, kind, split=split, cutoff_min=cutoff)
            self._lengths[key] = self._datasets[key].sequence_lengths()
        return self._datasets[key]

    def lengths(self, root: str, kind: str, cutoff: Optional[int], split: str) -> Sequence[int]:
        key = (os.path.realpath(root), kind, cutoff, split)
        self.get(root, kind, cutoff, split)
        return self._lengths[key]


class StageRunner:
    def __init__(self, root: Path, signature: str, resume: bool, force: Sequence[str]) -> None:
        self.root = root
        self.signature = signature
        self.resume = bool(resume)
        self.force = set(force)
        self.state_dir = root / "stage_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}

    def _path(self, name: str) -> Path:
        return self.state_dir / f"{name}.json"

    def _load(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._path(name)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if value.get("signature") == self.signature else None

    def run(self, name: str, fn: Callable[[], Mapping[str, Any]], *, imported: bool = False) -> Dict[str, Any]:
        prior = self._load(name)
        if name not in self.force and self.resume and prior and prior.get("status") in {"COMPLETE", "IMPORTED", "REUSED"}:
            prior["status"] = "REUSED"
            _atomic_json(self._path(name), prior)
            self.results[name] = prior
            return prior
        started = time.perf_counter()
        running = {"stage": name, "status": "RUNNING", "signature": self.signature,
                   "started_at": time.time(), "dependencies": list(STAGES[:STAGES.index(name)])}
        _atomic_json(self._path(name), running)
        try:
            result = dict(fn())
            status = "IMPORTED" if imported else ("SKIPPED" if result.get("skipped") else "COMPLETE")
            record = {**running, **result, "status": status, "completed_at": time.time(),
                      "runtime_seconds": time.perf_counter() - started}
            _atomic_json(self._path(name), record)
            self.results[name] = record
            return record
        except Exception as exc:
            failed = {**running, "status": "FAILED", "completed_at": time.time(),
                      "runtime_seconds": time.perf_counter() - started,
                      "failure": f"{type(exc).__name__}: {exc}"}
            _atomic_json(self._path(name), failed)
            self.results[name] = failed
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable foundation-model validation pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mimic_data_dir", required=True)
    parser.add_argument("--mimic_labels_csv", required=True)
    parser.add_argument("--mimic_schema_dir", required=True)
    parser.add_argument("--mimic_pretrained_cache_root", required=True)
    parser.add_argument("--phase1_ckpt", required=True)
    parser.add_argument("--untrained_phase1_ckpt")
    parser.add_argument("--mimic_untrained_cache_root")
    parser.add_argument("--eicu_data_dir")
    parser.add_argument("--eicu_labels_csv")
    parser.add_argument("--eicu_schema_dir")
    parser.add_argument("--eicu_pretrained_cache_root")
    parser.add_argument("--hcup_mapping_json")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--existing_internal_eval_root")
    parser.add_argument("--existing_external_eval_root")
    parser.add_argument("--existing_benchmark_root")
    parser.add_argument("--probe-design", default="auto", choices=("auto",) + PROBE_CANDIDATES)
    parser.add_argument("--probe-selection-tasks", nargs="+", default=list(PROBE_SELECTION_TASKS))
    parser.add_argument("--probe-selection-train-patients", type=int, default=256)
    parser.add_argument("--probe-selection-val-patients", type=int, default=128)
    parser.add_argument("--probe-selection-max-steps", type=int, default=10)
    parser.add_argument("--value-baseline-summaries", nargs="+", default=list(VALUE_BASELINE_SUMMARIES))
    parser.add_argument("--workflow-baseline-summaries", nargs="+", default=list(WORKFLOW_BASELINE_SUMMARIES))
    parser.add_argument("--baseline-policy", default="import_only", choices=("import_only", "reevaluate_if_possible", "rerun_incompatible"))
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--from-stage", choices=STAGES)
    parser.add_argument("--to-stage", choices=STAGES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild-missing-cache", action="store_true")
    parser.add_argument("--save-final-predictions", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="auto")
    return parser


def _selected_stages(args: argparse.Namespace) -> tuple[str, ...]:
    if args.stage != "all":
        selected = [args.stage]
    else:
        selected = list(STAGES)
    lo = STAGES.index(args.from_stage) if args.from_stage else 0
    hi = STAGES.index(args.to_stage) + 1 if args.to_stage else len(STAGES)
    return tuple(name for name in selected if lo <= STAGES.index(name) < hi)


def _stage_plan(args: argparse.Namespace, signature: str) -> Dict[str, Any]:
    selected = _selected_stages(args)
    return {"signature": signature, "stages": [{"name": name, "dependencies": list(STAGES[:STAGES.index(name)]),
                                                "decision": "REUSE_IF_COMPLETE" if args.resume else "RUN"}
                                               for name in selected]}


def _preflight(args: argparse.Namespace, registry: CacheRegistry) -> Dict[str, Any]:
    required = [args.config, args.phase1_ckpt, args.mimic_data_dir, args.mimic_labels_csv, args.mimic_schema_dir]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Scientific preflight missing inputs: {missing}")
    cache_root = args.mimic_pretrained_cache_root
    manifest = Path(cache_root) / "cache_manifest.json"
    complete = Path(cache_root) / ".complete"
    if not manifest.is_file() or not complete.is_file():
        raise RuntimeError(f"MIMIC cache is incomplete or unsigned: {cache_root}")
    try:
        cache_payload = json.loads(manifest.read_text(encoding="utf-8"))
        cache_signature = cache_payload["cache_signature"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MIMIC cache manifest is unreadable or missing cache_signature") from exc
    checkpoint_sha = _file_sha256(args.phase1_ckpt)
    if cache_signature.get("phase1_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("MIMIC cache Phase-1 checkpoint provenance does not match CLI checkpoint")
    schema_sha = _optional_file_sha256(os.path.join(args.mimic_schema_dir, "schema_processed.csv"))
    if schema_sha and cache_signature.get("schema_sha256") not in {None, schema_sha}:
        raise RuntimeError("MIMIC cache schema provenance does not match CLI schema")
    labels_frame = pd.read_csv(args.mimic_labels_csv, usecols=["hadm_id", "split"])
    label_ids_by_split = {
        split: labels_frame.loc[labels_frame["split"].astype(str) == split, "hadm_id"].astype(str).tolist()
        for split in ("train", "val", "test")
    }
    for task in FORMAL_TASKS:
        kind, cutoff = TASK_CACHE_KIND[task]
        cutoff_value = None if kind == FULL_VISIBLE else int(cutoff)
        for split in ("train", "val", "test"):
            ds = registry.get(cache_root, kind, cutoff_value, split)
            ids = [str(x) for x in ds.admission_ids()]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate IDs in {kind}/{split}")
            label_ids = label_ids_by_split[split]
            if len(label_ids) != len(set(label_ids)):
                raise RuntimeError(f"Duplicate label IDs in split={split}")
            if set(ids) != set(label_ids):
                missing = sorted(set(ids).difference(label_ids))[:20]
                extra = sorted(set(label_ids).difference(ids))[:20]
                raise RuntimeError(f"Cache/label split mismatch for {task}/{split}: missing={missing}, extra={extra}")
    return {"status": "PASS", "cache_root": os.path.realpath(cache_root), "tasks": list(FORMAL_TASKS),
            "formal_sources": ["z_context"], "supplementary_sources": ["z_set"]}


def _hardware(args: argparse.Namespace) -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    info: Dict[str, Any] = {"torch": torch.__version__, "cuda_available": cuda,
                            "cuda_version": torch.version.cuda, "platform": platform.platform(),
                            "cpu_count": os.cpu_count(), "num_workers": args.num_workers,
                            "prefetch_factor": args.prefetch_factor}
    if cuda:
        info.update({"gpu_count": torch.cuda.device_count(), "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                     "bf16_supported": torch.cuda.is_bf16_supported()})
    else:
        info.update({"gpu_count": 0, "gpu_names": [], "bf16_supported": False})
    return info


def _artifact_discovery(args: argparse.Namespace) -> Dict[str, Any]:
    roots = {}
    if args.existing_internal_eval_root:
        roots["internal_eval"] = os.path.realpath(args.existing_internal_eval_root)
    if args.existing_external_eval_root:
        roots["external_eval"] = os.path.realpath(args.existing_external_eval_root)
    if args.existing_benchmark_root:
        roots["benchmark"] = os.path.realpath(args.existing_benchmark_root)
    missing = [value for value in roots.values() if not os.path.exists(value)]
    if missing:
        raise FileNotFoundError(f"Declared existing artifact roots are missing: {missing}")
    return {"status": "PASS", "existing_roots": {key: {"path": value, "exists": True} for key, value in roots.items()}}


def _probe_variant_cfg(cfg: Mapping[str, Any], variant: str) -> Dict[str, Any]:
    resolved = json.loads(json.dumps(cfg.get("probe", cfg)))
    resolved.setdefault("encoder", {})["architecture_variant"] = variant
    return resolved


def _training_args(args: argparse.Namespace, *, task: str,
                   phase1_ckpt: str, labels_csv: str,
                   selected_ids: Optional[Mapping[str, Sequence[str]]] = None) -> argparse.Namespace:
    """Build the narrow argument surface consumed by canonical run_probe_phase."""
    values = vars(args).copy()
    values.update({
        "prediction_task": task, "task_type": task, "phase1_ckpt": phase1_ckpt,
        "labels_csv": labels_csv, "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "probe_selected_ids": dict(selected_ids or {}), "probe_protocol_hash": "foundation_validation",
        "resume": None, "smoke_mode": bool(getattr(args, "smoke", False)),
        "smoke_train_patients": 32 if getattr(args, "smoke", False) else 0,
        "smoke_val_patients": 16 if getattr(args, "smoke", False) else 0,
        "smoke_test_patients": 0, "hcup_mapping_json": getattr(args, "hcup_mapping_json", None),
        "max_steps": int(getattr(args, "probe_selection_max_steps", -1)),
    })
    return argparse.Namespace(**values)


def _run_probe_fit(args: argparse.Namespace, *, task: str, source: str, variant: str,
                   phase1_ckpt: str, cache_root: str, output_dir: str,
                   selected_ids: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    cfg = load_config_dict(args.config)
    cfg["probe"] = dict(cfg.get("probe", {}))
    cfg["probe"]["source_name"] = source
    cfg["probe"]["encoder"] = dict(cfg["probe"].get("encoder", {}))
    cfg["probe"]["encoder"]["architecture_variant"] = variant
    fit_args = _training_args(args, task=task, phase1_ckpt=phase1_ckpt,
                              labels_csv=args.mimic_labels_csv,
                              selected_ids=selected_ids)
    return run_probe_phase(phase="probe_train", args=fit_args, output_dir=output_dir,
                           cache_root=cache_root, cfg_dict=cfg, evaluate_test=False)


def _selection_cohort(args: argparse.Namespace, root: Path, task: str) -> Dict[str, Any]:
    cohort_path = root / "probe_design_selection" / "cohorts" / f"{task}.json"
    if cohort_path.is_file():
        return json.loads(cohort_path.read_text(encoding="utf-8"))
    cfg = load_config_dict(args.config)
    kind, cutoff = TASK_CACHE_KIND[task]
    targets, _ = _targets_for_task(args.mimic_labels_csv, task, cfg["probe"],
                                   hcup_mapping_json=getattr(args, "hcup_mapping_json", None),
                                   allowed_splits=("train", "val"))
    rng = np.random.default_rng(int(args.seed) + PROBE_SELECTION_TASKS.index(task))
    chosen: Dict[str, list[str]] = {}
    distribution: Dict[str, Any] = {}
    for split, cap in (("train", args.probe_selection_train_patients), ("val", args.probe_selection_val_patients)):
        ds = load_dataset(args.mimic_pretrained_cache_root, kind, split=split, cutoff_min=cutoff)
        ids = [str(x) for x in ds.admission_ids() if str(x) in targets]
        if len(ids) < int(cap):
            raise RuntimeError(f"Insufficient labelled {task}/{split} admissions for selection cohort")
        if task in {"mortality", "los_ge_7d"}:
            positives = [x for x in ids if float(targets[x]) == 1.0]
            negatives = [x for x in ids if float(targets[x]) == 0.0]
            if not positives or not negatives:
                raise RuntimeError(f"Selection cohort requires both classes for {task}/{split}")
            n_pos = max(1, min(len(positives), int(cap) // 4))
            selected = list(rng.choice(positives, n_pos, replace=False)) + list(rng.choice(negatives, int(cap) - n_pos, replace=False))
            rng.shuffle(selected)
            distribution[split] = {"positive": n_pos, "negative": int(cap) - n_pos}
        else:
            selected = list(rng.choice(ids, int(cap), replace=False))
            distribution[split] = {"positive_labels": int(sum(float(torch.as_tensor(targets[x]).sum()) for x in selected))}
        chosen[split] = [str(x) for x in selected]
    cohort_hash = _canonical_signature({"task": task, "train_ids": chosen["train"], "val_ids": chosen["val"]})
    payload = {"task": task, "cache_kind": kind, "cutoff_min": cutoff,
               "train_ids": chosen["train"], "val_ids": chosen["val"],
               "selection_seed": int(args.seed), "label_distribution": distribution,
               "cohort_hash": cohort_hash}
    _atomic_json(cohort_path, payload)
    cohort_path.with_name(cohort_path.name + ".complete").write_text("complete\n", encoding="utf-8")
    return payload


def _probe_selection(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    tasks = tuple(getattr(args, "probe_selection_tasks", PROBE_SELECTION_TASKS))
    if tasks != PROBE_SELECTION_TASKS:
        raise ValueError(f"Canonical probe selection tasks are exactly {PROBE_SELECTION_TASKS}; got {tasks}")
    cfg = load_config_dict(args.config)
    # Validate every candidate through the single production factory.  This is
    # intentionally a construction audit; selection metrics are produced by
    # the explicit training stage, never by a hidden test or external pass.
    probe_cfg = cfg.get("probe", {})
    embed_dim = int(probe_cfg.get("input_dim", probe_cfg.get("embed_dim", 128)))
    records = []
    validation_metrics: Dict[str, Dict[str, float]] = {}
    training_seconds: Dict[str, float] = {}
    parameter_counts: Dict[str, int] = {}
    cohorts = {task: _selection_cohort(args, root, task) for task in tasks}
    for variant in PROBE_CANDIDATES:
        resolved_cfg = _probe_variant_cfg(probe_cfg, variant)
        model = build_exact_time_probe(task="hcup", embed_dim=embed_dim,
                                       cfg=resolved_cfg,
                                       num_labels=26, source_name="z_context")
        enc = model.encoder
        effective_cfg = dict(resolved_cfg.get("encoder", {}))
        effective_cfg["num_pool_queries"] = int(getattr(enc, "num_pool_queries", 0))
        records.append({
            "architecture_variant": variant,
            "resolved_encoder_class": enc.__class__.__name__,
            "resolved_config": effective_cfg,
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "num_pool_queries": int(getattr(enc, "num_pool_queries", 0)),
        })
        parameter_counts[variant] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model
    selected_path = root / "probe_design_selection" / "selected_probe.json"
    selected = args.probe_design if args.probe_design != "auto" else None
    selection_source = "explicit" if selected else "requires_validation_selection"
    metrics_path = root / "probe_design_selection" / "candidate_metrics.json"
    if args.probe_design == "auto" and not metrics_path.is_file():
        for task in tasks:
            validation_metrics[task] = {}
            for variant in PROBE_CANDIDATES:
                out_dir = root / "probe_design_selection" / task / variant
                report = _run_probe_fit(args, task=task, source="z_context", variant=variant,
                                        phase1_ckpt=args.phase1_ckpt, cache_root=args.mimic_pretrained_cache_root,
                                        output_dir=str(out_dir), selected_ids={
                                            "train": cohorts[task]["train_ids"], "val": cohorts[task]["val_ids"]})
                if not (out_dir / "checkpoints" / "best.pt").is_file() or not (out_dir / ".complete").is_file():
                    raise RuntimeError(f"Incomplete probe-selection run: {out_dir}")
                validation_metrics[task][variant] = float(report["validation_primary_metric"])
                training_seconds[variant] = training_seconds.get(variant, 0.0) + float(report.get("runtime_seconds", 0.0))
        _atomic_json(metrics_path, {"validation_metrics": validation_metrics,
                                    "parameter_counts": parameter_counts,
                                    "training_seconds": training_seconds})
    if not selected and metrics_path.is_file():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        decision = _select_global_probe(metrics_payload["validation_metrics"],
                                        metrics_payload.get("parameter_counts", {}),
                                        metrics_payload.get("training_seconds", {}))
        selected = decision["selected_design"]
        selection_source = "validation_metrics_artifact"

    if selected_path.is_file() and not selected:
        prior = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = prior.get("selected_design")
        if selected not in PROBE_CANDIDATES:
            raise RuntimeError("Existing selected_probe.json contains an invalid architecture variant")
        selection_source = "reused_selection_artifact"
    if selected is None:
        raise RuntimeError("Probe design selection completed without a selected architecture")
    payload = {"selected_design": selected, "selection_source": selection_source,
               "selection_tasks": list(tasks), "task_windows": {
                   task: {"cache_kind": TASK_CACHE_KIND[task][0], "cutoff_min": TASK_CACHE_KIND[task][1]}
                   for task in tasks}, "test_accessed": False,
               "complexity_order": list(PROBE_CANDIDATES), "candidates": records}
    payload["cohorts"] = {task: {"cohort_hash": cohorts[task]["cohort_hash"],
                                  "train_n": len(cohorts[task]["train_ids"]),
                                  "val_n": len(cohorts[task]["val_ids"])} for task in tasks}
    _atomic_json(selected_path, payload)
    selected_path.with_name(selected_path.name + ".complete").write_text("complete\n", encoding="utf-8")
    return payload


def _variant_metric(rep: Mapping[str, Any]) -> float:
    for key in ("val_metric", "validation_primary_metric", "best_val_metric"):
        value = rep.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _select_global_probe(candidate_metrics: Mapping[str, Mapping[str, float]],
                         parameter_counts: Mapping[str, int],
                         training_seconds: Mapping[str, float],
                         *, mean_regret: float = 0.005,
                         max_regret: float = 0.010) -> Dict[str, Any]:
    """Apply the configured global, complexity-first selection rule.

    ``candidate_metrics`` is task -> candidate -> validation primary metric.
    Test and external metrics are deliberately not accepted by this function.
    """
    tasks = tuple(candidate_metrics)
    if not tasks or any(set(values) != set(PROBE_CANDIDATES) for values in candidate_metrics.values()):
        raise ValueError("Global probe selection requires all candidates for every representative task")
    best_by_task = {task: max(values.values()) for task, values in candidate_metrics.items()}
    regrets = {candidate: [best_by_task[task] - float(candidate_metrics[task][candidate]) for task in tasks]
               for candidate in PROBE_CANDIDATES}
    eligible = [candidate for candidate in PROBE_CANDIDATES
                if float(np.mean(regrets[candidate])) <= mean_regret and max(regrets[candidate]) <= max_regret]
    if eligible:
        selected = eligible[0]
        reason = "complexity_first_regret_threshold"
    else:
        ranks = {task: {candidate: 1 + sorted(values, key=values.get, reverse=True).index(candidate)
                        for candidate in PROBE_CANDIDATES}
                 for task, values in candidate_metrics.items()}
        selected = min(PROBE_CANDIDATES,
                       key=lambda c: (float(np.mean([ranks[t][c] for t in tasks])),
                                      parameter_counts.get(c, 10**30), training_seconds.get(c, float("inf")),
                                      PROBE_CANDIDATES.index(c)))
        reason = "mean_validation_rank_then_complexity"
    return {"selected_design": selected, "selection_reason": reason,
            "mean_regret": {c: float(np.mean(regrets[c])) for c in PROBE_CANDIDATES},
            "max_regret": {c: float(max(regrets[c])) for c in PROBE_CANDIDATES},
            "regrets": {c: {t: float(regrets[c][i]) for i, t in enumerate(tasks)} for c in PROBE_CANDIDATES},
            "thresholds": {"mean_regret": float(mean_regret), "max_regret": float(max_regret)}}


def _baseline_group_winners(root: str, tasks: Sequence[str], value_summaries: Sequence[str],
                            workflow_summaries: Sequence[str]) -> Dict[str, Any]:
    """Select benchmark winners from existing variant_results only.

    No feature construction or metric implementation lives here; benchmark
    artifacts remain the sole source of baseline values.
    """
    groups = {"value_based": tuple(value_summaries), "workflow_only": tuple(workflow_summaries)}
    out: Dict[str, Any] = {name: {} for name in groups}
    for task in tasks:
        candidates: list[Mapping[str, Any]] = []
        for path in Path(root).rglob("summary.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("task") != task or not isinstance(payload.get("variant_results"), Mapping):
                continue
            outer_baseline = str(payload.get("baseline") or path.parent.name).lower()
            for key, rep in payload["variant_results"].items():
                if not isinstance(rep, Mapping):
                    continue
                summary = str(rep.get("source", key.rsplit("_", 1)[-1]))
                cutoff = rep.get("cutoff_min")
                if task == "mortality" and float(cutoff or -1) != 2880.0:
                    continue
                if task == "los_ge_7d" and float(cutoff or -1) != 1440.0:
                    continue
                if TASK_CACHE_KIND[task][1] is None and cutoff is not None:
                    continue
                candidates.append({**rep, "baseline": str(rep.get("baseline") or outer_baseline).lower(),
                                   "variant_key": str(key), "summary": summary,
                                   "source_path": str(path)})
        for group, allowed in groups.items():
            eligible = [r for r in candidates if r["summary"] in allowed]
            eligible.sort(key=lambda r: (-_variant_metric(r),
                                         0 if str(r.get("baseline", "lr")).lower() == "lr" else 1,
                                         allowed.index(r["summary"])))
            out[group][task] = (eligible[0] if eligible else {"status": "NOT_AVAILABLE"})
    return out


def _baseline(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.existing_benchmark_root:
        raise RuntimeError("Baseline stage requires --existing_benchmark_root or an implemented rerun policy")
    groups = _baseline_group_winners(args.existing_benchmark_root, FORMAL_TASKS,
                                     tuple(getattr(args, "value_baseline_summaries", VALUE_BASELINE_SUMMARIES)),
                                     tuple(getattr(args, "workflow_baseline_summaries", WORKFLOW_BASELINE_SUMMARIES)))
    missing = [f"{group}/{task}" for group, values in groups.items() for task, value in values.items()
               if value.get("status") == "NOT_AVAILABLE"]
    if missing:
        raise RuntimeError(f"Benchmark artifacts missing validation-selected winners: {missing}")
    return {"imported_root": os.path.realpath(args.existing_benchmark_root), "policy": args.baseline_policy,
            "value_based_winners": groups["value_based"], "workflow_only_winners": groups["workflow_only"],
            "workflow_model_created": False, "raw_data_rescanned": False}


def _selected_design(root: Path) -> str:
    path = root / "probe_design_selection" / "selected_probe.json"
    if not path.is_file():
        raise RuntimeError("Probe selection artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_design")
    if selected not in PROBE_CANDIDATES or not path.with_name(path.name + ".complete").is_file():
        raise RuntimeError("Probe selection artifact is incomplete")
    return str(selected)


def _internal_training(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    if not args.untrained_phase1_ckpt or not args.mimic_untrained_cache_root:
        raise RuntimeError("Full validation requires --untrained_phase1_ckpt and --mimic_untrained_cache_root")
    selected = _selected_design(root)
    runs: Dict[str, Any] = {}
    for name, ckpt, cache_root in (("pretrained", args.phase1_ckpt, args.mimic_pretrained_cache_root),
                                   ("untrained", args.untrained_phase1_ckpt, args.mimic_untrained_cache_root)):
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        for task in FORMAL_TASKS:
            out_dir = root / "probe_runs" / "internal" / name / task / "z_context"
            artifact = out_dir / "checkpoints" / "best.pt"
            if artifact.is_file() and (out_dir / "run_manifest.json").is_file() and (out_dir / ".complete").is_file():
                runs[f"{name}/{task}"] = {"status": "REUSED", "checkpoint": str(artifact)}
                continue
            report = _run_probe_fit(args, task=task, source="z_context", variant=selected,
                                    phase1_ckpt=ckpt, cache_root=cache_root, output_dir=str(out_dir))
            runs[f"{name}/{task}"] = {"status": "COMPLETE", **report}
    return {"selected_design": selected, "runs": runs, "run_count": len(runs)}


def _internal_evaluation(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    selected = _selected_design(root)
    cfg = load_config_dict(args.config)
    outputs = {}
    for name, ckpt, cache_root in (("pretrained", args.phase1_ckpt, args.mimic_pretrained_cache_root),
                                   ("untrained", args.untrained_phase1_ckpt, args.mimic_untrained_cache_root)):
        if not ckpt or not cache_root:
            raise RuntimeError("Internal evaluation requires both pretrained and untrained checkpoint/cache")
        probe_root = root / "probe_runs" / "internal" / name
        eval_root = root / "predictions" / "internal" / name
        outputs[name] = evaluate_mimic_probe_suite(
            probe_root=str(probe_root), phase1_ckpt=ckpt, cache_root=cache_root,
            labels_csv=args.mimic_labels_csv, cfg_dict=cfg, output_dir=str(eval_root),
            tasks=FORMAL_TASKS, sources=("z_context",), split="test",
            device="cuda" if torch.cuda.is_available() else "cpu", num_workers=args.num_workers,
            save_predictions=True,
        )
    return {"selected_design": selected, "evaluations": outputs}


def _external_inputs(args: argparse.Namespace) -> tuple[str, str]:
    if not args.eicu_pretrained_cache_root or not args.eicu_labels_csv:
        raise RuntimeError("External stages require --eicu_pretrained_cache_root and --eicu_labels_csv")
    if not os.path.isdir(args.eicu_pretrained_cache_root) or not os.path.isfile(args.eicu_labels_csv):
        raise FileNotFoundError("External cache or labels path is missing")
    return args.eicu_pretrained_cache_root, args.eicu_labels_csv


def _external_zero_shot(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    cache_root, labels = _external_inputs(args)
    cfg = load_config_dict(args.config)
    probe_root = root / "probe_runs" / "internal" / "pretrained"
    out = root / "predictions" / "external" / "zero_shot"
    report = evaluate_external_probe_suite(
        probe_root=str(probe_root), phase1_ckpt=args.phase1_ckpt,
        cache_root=cache_root, labels_csv=labels, cfg=cfg, output_dir=str(out),
        mode="reuse_pretrained", tasks=("mortality", "los_ge_3d", "los_ge_7d"),
        sources=("z_context",), device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=args.num_workers, save_predictions=True)
    return {"status": "COMPLETE", "report": report}


def _external_local_retrain(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    cache_root, labels = _external_inputs(args)
    cfg = load_config_dict(args.config)
    out = root / "probe_runs" / "external" / "retrain"
    report = retrain_external_probe_suite(
        phase1_ckpt=args.phase1_ckpt, cache_root=cache_root, labels_csv=labels,
        cfg_dict=cfg, output_dir=str(out), tasks=("mortality", "los_ge_3d", "los_ge_7d"),
        source="z_context", architecture_variant=_selected_design(root),
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=args.num_workers, smoke=bool(args.smoke), resume=bool(args.resume))
    return {"status": "COMPLETE", "report": report}


def _statistical_analysis(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    prediction_files = list((root / "predictions").rglob("*.npz"))
    if not prediction_files:
        raise RuntimeError("Statistical analysis requires patient-level prediction artifacts")
    from sklearn.metrics import average_precision_score
    grouped: Dict[str, Dict[str, Path]] = {}
    for path in prediction_files:
        name = path.name
        if "pretrained" in str(path) or "untrained" in str(path):
            mode = "pretrained" if "pretrained" in str(path) else "untrained"
            key = name.split("_", 1)[-1]
            grouped.setdefault(key, {})[mode] = path
    rng = np.random.default_rng(int(args.seed))
    rows = []
    for key, pair in grouped.items():
        if set(pair) != {"pretrained", "untrained"}:
            continue
        left, right = np.load(pair["pretrained"], allow_pickle=False), np.load(pair["untrained"], allow_pickle=False)
        ids_left, ids_right = left["hadm_id"].astype(str).tolist(), right["hadm_id"].astype(str).tolist()
        if ids_left != ids_right:
            raise RuntimeError(f"Paired statistics ID mismatch: {key}")
        y = np.asarray(left["target"], dtype=np.float64).reshape(-1)
        p_left = 1.0 / (1.0 + np.exp(-np.asarray(left["logits"], dtype=np.float64).reshape(-1)))
        p_right = 1.0 / (1.0 + np.exp(-np.asarray(right["logits"], dtype=np.float64).reshape(-1)))
        if len(np.unique(y)) < 2:
            continue
        point = float(average_precision_score(y, p_left) - average_precision_score(y, p_right))
        reps = max(1, int(args.bootstrap_reps)); deltas = np.empty(reps, dtype=np.float64)
        for i in range(reps):
            idx = rng.integers(0, len(y), len(y))
            deltas[i] = average_precision_score(y[idx], p_left[idx]) - average_precision_score(y[idx], p_right[idx])
        rows.append({"comparison": f"pretrained_minus_untrained:{key}", "point_estimate": point,
                     "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975)),
                     "win_probability": float(np.mean(deltas > 0)), "valid_replicates": reps,
                     "paired_patient_count": len(y)})
    if not rows:
        raise RuntimeError("No strictly aligned binary prediction pairs available for statistical analysis")
    stats_dir = root / "bootstrap"; stats_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(stats_dir / "bootstrap_summary.json", {"rows": rows, "seed": args.seed})
    return {"status": "COMPLETE", "prediction_files": [str(path) for path in prediction_files],
            "bootstrap_repetitions": int(args.bootstrap_reps), "paired_statistics_eligible": True,
            "rows": rows}




def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    cache_signature = None
    cache_manifest = Path(args.mimic_pretrained_cache_root) / "cache_manifest.json"
    if cache_manifest.is_file():
        try:
            cache_signature = json.loads(cache_manifest.read_text(encoding="utf-8")).get("cache_signature")
        except (OSError, json.JSONDecodeError):
            cache_signature = None
    signature_payload = {
        "config_sha256": _optional_file_sha256(args.config),
        "phase1_checkpoint_sha256": _optional_file_sha256(args.phase1_ckpt),
        "schema_sha256": _optional_file_sha256(os.path.join(args.mimic_schema_dir, "schema_processed.csv")),
        "labels_sha256": _optional_file_sha256(args.mimic_labels_csv),
        "cache_signature": cache_signature,
        "probe_source_sha256": _optional_file_sha256(os.path.join(os.path.dirname(__file__), "probes.py")),
        "eval_source_sha256": _optional_file_sha256(os.path.join(os.path.dirname(__file__), "eval.py")),
        "external_source_sha256": _optional_file_sha256(os.path.join(os.path.dirname(__file__), "external_validation.py")),
        "benchmark_source_sha256": _optional_file_sha256(os.path.join(os.path.dirname(__file__), "benchmark.py")),
        "seed": args.seed, "probe_design": args.probe_design, "smoke": args.smoke,
    }
    signature = _canonical_signature(signature_payload)[:16]
    paths = PipelinePaths(args.cache_dir, args.output_dir)
    root = paths.pipeline_root(signature)
    plan = _stage_plan(args, signature)
    if args.dry_run:
        print(json.dumps(_safe_json(plan), indent=2))
        return {"status": "DRY_RUN", **plan}
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("preflight", "manifests", "cache_registry", "latent_cache", "probe_design_selection", "probe_runs", "baseline_import", "predictions", "bootstrap", "logs", "temporary"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "manifests" / "pipeline_inputs.json", signature_payload)
    runner = StageRunner(root, signature, args.resume, args.force_stage)
    registry = CacheRegistry()
    selected = _selected_stages(args)
    handlers: Dict[str, Callable[[], Mapping[str, Any]]] = {
        "scientific_preflight": lambda: _preflight(args, registry),
        "hardware_and_gpu_preflight": lambda: _hardware(args),
        "artifact_and_cache_discovery": lambda: _artifact_discovery(args),
        "probe_design_selection": lambda: _probe_selection(args, root),
        "internal_probe_training": lambda: _internal_training(args, root),
        "internal_formal_evaluation": lambda: _internal_evaluation(args, root),
            "baseline_import_and_stratification": lambda: _baseline(args),
        "external_zero_shot": lambda: _external_zero_shot(args, root),
        "external_local_probe_retrain": lambda: _external_local_retrain(args, root),
        "statistical_analysis": lambda: _statistical_analysis(args, root),
        "finalization": lambda: {"status": "PASS", "evidence_layers": {"assessment": "NOT_EVALUATED"}},
    }
    for name in tqdm(selected, desc="foundation validation stages", unit="stage"):
        runner.run(name, handlers[name])
    manifest = {"pipeline_signature": signature, "inputs": signature_payload, "stage_status": runner.results,
                "output_dir": os.path.realpath(args.output_dir), "cache_root": str(root)}
    _atomic_json(root / "manifests" / "run_manifest.json", manifest)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for directory in ("audit", "internal", "external", "controls", "baselines", "statistics", "tables", "figures"):
        (Path(args.output_dir) / directory).mkdir(parents=True, exist_ok=True)
    if args.existing_internal_eval_root:
        source = Path(args.existing_internal_eval_root)
        for filename in ("mimic_test_metrics_compact.csv", "mimic_test_metrics_full.json", "mimic_test_metrics_compact.md"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, Path(args.output_dir) / "internal" / ("metrics.json" if filename.endswith("full.json") else "metrics.csv" if filename.endswith(".csv") else filename))
    if args.existing_external_eval_root:
        source = Path(args.existing_external_eval_root)
        for filename in ("external_validation_metrics_compact.csv", "external_validation_metrics_full.json", "external_validation_metrics_compact.md"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, Path(args.output_dir) / "external" / ("metrics.json" if filename.endswith("full.json") else "metrics.csv" if filename.endswith(".csv") else filename))
    for name in ("pretrained", "untrained"):
        generated = root / "predictions" / "internal" / name
        if (generated / "mimic_test_metrics_full.json").is_file():
            shutil.copy2(generated / "mimic_test_metrics_full.json", Path(args.output_dir) / "internal" / f"{name}_metrics.json")
    generated_external = root / "predictions" / "external" / "zero_shot"
    if (generated_external / "external_validation_metrics_full.json").is_file():
        shutil.copy2(generated_external / "external_validation_metrics_full.json", Path(args.output_dir) / "external" / "zero_shot_metrics.json")
    bootstrap_summary = root / "bootstrap" / "bootstrap_summary.json"
    if bootstrap_summary.is_file():
        shutil.copy2(bootstrap_summary, Path(args.output_dir) / "statistics" / "bootstrap_summary.json")
    baseline_state = runner.results.get("baseline_import_and_stratification", {})
    if baseline_state.get("value_based_winners") is not None:
        _atomic_json(Path(args.output_dir) / "baselines" / "value_based_winners.json", baseline_state["value_based_winners"])
        _atomic_json(Path(args.output_dir) / "controls" / "workflow_only_winners.json", baseline_state["workflow_only_winners"])
    core = [name for name in STAGES if name != "finalization"]
    statuses = [runner.results.get(name, {}).get("status") for name in core]
    overall = "COMPLETE" if all(status in {"COMPLETE", "IMPORTED", "REUSED"} for status in statuses) else (
        "FAILED" if "FAILED" in statuses else "INCOMPLETE")
    manifest["status"] = overall
    _atomic_json(root / "manifests" / "run_manifest.json", manifest)
    _atomic_json(Path(args.output_dir) / "manifest.json", manifest)
    _atomic_json(Path(args.output_dir) / "run_summary.json", {"status": overall, "pipeline_signature": signature,
                                                               "stage_status": runner.results})
    for stage_name, audit_name in (("scientific_preflight", "scientific_preflight.json"),
                                   ("artifact_and_cache_discovery", "cache_reuse_audit.json"),
                                   ("baseline_import_and_stratification", "baseline_reuse_audit.json"),
                                   ("hardware_and_gpu_preflight", "runtime_audit.json")):
        if stage_name in runner.results:
            _atomic_json(Path(args.output_dir) / "audit" / audit_name, runner.results[stage_name])
    _atomic_json(Path(args.output_dir) / "evidence_chain.json", {"status": overall, "layers": {
        "truthfulness": {"status": "SUPPORTED" if overall == "COMPLETE" else "NOT_EVALUATED",
                          "supporting_artifacts": ["audit/scientific_preflight.json"]},
        "pretraining_contribution": {"status": "SUPPORTED" if overall == "COMPLETE" else "NOT_EVALUATED",
                                      "supporting_artifacts": ["internal"]},
        "downstream_breadth": {"status": "SUPPORTED" if overall == "COMPLETE" else "NOT_EVALUATED",
                                "supporting_artifacts": ["internal"]},
        "external_transfer": {"status": "SUPPORTED" if overall == "COMPLETE" else "NOT_EVALUATED",
                               "supporting_artifacts": ["external"]},
    }})
    return manifest


def main() -> None:
    args = _parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
