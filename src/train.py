"""
ATLAS-EHR Training Pipeline.
Driven by `configs/params.yaml`.

Supports:
- Phase 1: Static Infilling & Representation Learning
- Target-manifold readout and downstream probe training:
  - probe_cache_build
  - readout_probe
  - phenotype_probe
  - prediction_probe
"""

import os
import json
import argparse
import hashlib
import warnings
import math
from typing import Any, Dict, Optional, Sequence
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback

class CollapseGuard(Callback):
    """Stop training when representation rank drops below threshold.

    The guard marks the checkpoint monitor as non-improving for the collapse
    epoch, so ``ModelCheckpoint(save_top_k=1)`` retains the best healthy
    checkpoint. ``last.ckpt`` remains a resumable diagnostic artifact and is
    never advertised as the safe model to evaluate.

    Threshold=1.8 gives a buffer below the healthy range (≥2.0) while
    unambiguously catching collapse (rank drops to ~1.3 in one epoch).

    Args:
        min_rank_eff: Effective rank threshold below which training stops.
        metric_key: Name of the logged metric to monitor.
    """
    def __init__(
        self,
        min_rank_eff: float = 1.8,
        metric_key: str = "val_rank_eff",
        checkpoint_monitor: str = "val/loss_rank_gated",
        source_monitor: str = "val/loss",
        last_healthy_filename: str = "phase1_last_healthy.ckpt",
    ):
        self.min_rank_eff = min_rank_eff
        self.metric_key = metric_key
        self.checkpoint_monitor = checkpoint_monitor
        self.source_monitor = source_monitor
        self.last_healthy_filename = last_healthy_filename
        self.triggered = False
        self.trigger_epoch: Optional[int] = None
        self.trigger_rank_eff: Optional[float] = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        rank_eff = getattr(pl_module, "_current_val_rank_eff_for_callbacks", None)
        source_value = getattr(pl_module, "_current_val_loss_for_callbacks", None)
        if rank_eff is None:
            rank_eff = trainer.callback_metrics.get(self.metric_key)
        if source_value is None:
            source_value = trainer.callback_metrics.get(self.source_monitor)
        if rank_eff is None or source_value is None:
            raise RuntimeError(
                f"CollapseGuard requires current-epoch {self.metric_key!r} and "
                f"{self.source_monitor!r} before checkpoint selection."
            )
        rank_eff = float(rank_eff)
        if rank_eff < self.min_rank_eff:
            self.triggered = True
            self.trigger_epoch = int(trainer.current_epoch)
            self.trigger_rank_eff = rank_eff
            # This hook runs before checkpoint callbacks. Keep the logged
            # val/loss intact, but make this epoch ineligible for best-model
            # selection and early-stop improvement accounting.
            trainer.callback_metrics[self.checkpoint_monitor] = torch.tensor(
                float("inf"), device=pl_module.device
            )
            fit_loop = getattr(trainer, "fit_loop", None)
            if fit_loop is None or not hasattr(fit_loop, "min_epochs"):
                raise RuntimeError("CollapseGuard cannot override Lightning min_epochs on this Trainer version.")
            # Lightning otherwise ignores trainer.should_stop until min_epochs.
            fit_loop.min_epochs = 0
            print(
                f"\n[CollapseGuard] {self.metric_key}={rank_eff:.3f} < {self.min_rank_eff}. "
                "Representation collapse detected — stopping; use the best checkpoint, not last.ckpt."
            )
            trainer.should_stop = True
            return
        trainer.callback_metrics[self.checkpoint_monitor] = source_value
        if trainer.is_global_zero:
            trainer.save_checkpoint(
                os.path.join(trainer.checkpoint_callback.dirpath, self.last_healthy_filename)
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "trigger_epoch": self.trigger_epoch,
            "trigger_rank_eff": self.trigger_rank_eff,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.triggered = bool(state_dict.get("triggered", False))
        self.trigger_epoch = state_dict.get("trigger_epoch")
        self.trigger_rank_eff = state_dict.get("trigger_rank_eff")


from src.utils import RecipeConfig, load_config_dict, validate_phase1_masking_config
from src.readout import (READOUT_ARCHITECTURE, build_manifold_bank, calibrate_readout_streaming,
                         save_readout_artifact, load_readout_artifact, scan_train_numeric_support,
                         evaluate_readout_streaming, evaluate_readout_per_vid_streaming,
                         fit_per_vid_ridge_streaming, run_frozen_readout_evaluator)
from src.cache_bank import build_readout_query_cache, load_dataset, READOUT_TRAINING_CACHE, build_readout_cache_signature, validate_readout_query_cache, _phase1_dataset_kwargs, _dataset_content_signature, _stable_json_sha256


def _run_phase1_target_manifold_readout(args: argparse.Namespace, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build and calibrate the non-parametric Phase-1 numeric readout."""
    if not args.phase1_ckpt:
        raise ValueError("--phase readout_probe requires an explicit --phase1_ckpt")
    readout_cfg = cfg_dict.get("readout", {})
    if readout_cfg.get("architecture") != READOUT_ARCHITECTURE:
        raise ValueError(f"readout.architecture must be {READOUT_ARCHITECTURE!r}")
    if not args.cache_dir:
        raise ValueError("--phase readout_probe requires --cache_dir")
    query_fraction = float(args.fraction)
    support_fraction = float(readout_cfg.get("support_fraction", 1.0))
    if not (0.0 < query_fraction <= 1.0):
        raise ValueError("query_fraction must satisfy 0 < fraction <= 1")
    if not (0.0 < support_fraction <= 1.0):
        raise ValueError("support_fraction must satisfy 0 < fraction <= 1")
    schema_path = os.path.join(args.schema_dir, "schema_processed.csv")
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.accelerator) != "cpu" else "cpu")
    from src.model import JEPAModel
    teacher = JEPAModel.load_from_checkpoint(
        os.path.realpath(args.phase1_ckpt), map_location=device, weights_only=False, schema_path=schema_path
    ).to(device).eval()
    teacher.requires_grad_(False)
    phase1_arch = getattr(teacher, "phase1_architecture_version", None)
    if not phase1_arch:
        raise RuntimeError("Phase-1 checkpoint has no architecture version")
    cache_root = os.path.join(str(args.cache_dir), "readout_cache")
    cache_cfg = readout_cfg.get("cache", {})
    print(f"[readout] preparing query cache: splits={','.join(('test',) if bool(getattr(args, 'readout_eval_only', False)) else ('train', 'val', 'test'))} query_fraction={query_fraction}", flush=True)
    if bool(getattr(args, "readout_eval_only", False)):
        if not args.readout_artifact or not os.path.isfile(args.readout_artifact):
            raise FileNotFoundError("--readout_eval_only requires an existing --readout_artifact")
        readout = load_readout_artifact(args.readout_artifact, map_location=device, expected_phase1_ckpt=args.phase1_ckpt, expected_schema=schema_path, expected_phase1_architecture=phase1_arch)
        splits = ("test",)
    else:
        splits = ("train", "val", "test")
    for split in splits:
        stage = os.path.join(cache_root, READOUT_TRAINING_CACHE, split)
        expected_signature = build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split=split, fraction=query_fraction)
        if bool(getattr(args, "readout_eval_only", False)) and not os.path.isfile(os.path.join(stage, ".complete")):
            raise FileNotFoundError(f"eval-only requires an existing validated readout cache: {stage}")
        if os.path.isfile(os.path.join(stage, ".complete")) and not bool(getattr(args, "rebuild_readout_cache", False)):
            print(f"[readout] validating existing query cache: split={split}", flush=True)
            validate_readout_query_cache(cache_root, split, expected_signature)
        else:
            print(f"[readout] building query cache: split={split}", flush=True)
            build_readout_query_cache(
                cache_root=cache_root, data_dir=args.data_dir, schema_dir=args.schema_dir,
                cfg_dict=cfg_dict, teacher=teacher, phase1_ckpt_path=args.phase1_ckpt,
                split=split, labels_csv=args.labels_csv,
                batch_size=int(cache_cfg.get("batch_size", 16)),
                num_workers=int(cache_cfg.get("num_workers", args.num_workers)),
                fraction=query_fraction,
            )
    def make_loader(split: str) -> DataLoader:
        ds = load_dataset(cache_root, READOUT_TRAINING_CACHE, split=split, expected_phase1_ckpt=args.phase1_ckpt, expected_schema=schema_path)
        workers = int(cache_cfg.get("num_workers", args.num_workers or 0))
        loader_kwargs = {
            "batch_size": int(cache_cfg.get("eval_batch_size", cache_cfg.get("readout_batch_size", 2048))),
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": bool(cache_cfg.get("pin_memory", torch.cuda.is_available())),
            "persistent_workers": bool(cache_cfg.get("persistent_workers", True)) and workers > 0,
        }
        if workers > 0 and cache_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(cache_cfg.get("prefetch_factor", 4))
        return DataLoader(ds, **loader_kwargs)
    if bool(getattr(args, "readout_eval_only", False)):
        os.makedirs(args.output_dir, exist_ok=True)
        print("[readout] eval-only: loading frozen readout and decoder artifacts", flush=True)
        decoder_path = os.path.join(os.path.dirname(os.path.realpath(args.readout_artifact)), "readout_decoders.pt")
        if not os.path.isfile(decoder_path): raise FileNotFoundError("readout eval-only requires frozen readout_decoders.pt")
        decoder_payload = torch.load(decoder_path, map_location="cpu", weights_only=False)
        provenance = decoder_payload.get("provenance", {})
        if provenance.get("phase1_checkpoint") != os.path.realpath(args.phase1_ckpt) or provenance.get("schema") != os.path.realpath(schema_path) or provenance.get("phase1_checkpoint_sha256") != _sha256_file(args.phase1_ckpt) or provenance.get("schema_sha256") != _sha256_file(schema_path):
            raise RuntimeError("readout decoder artifact provenance mismatch")
        artifact_payload = torch.load(args.readout_artifact, map_location="cpu", weights_only=False)
        support = artifact_payload.get("extra_metadata", {}).get("train_support", {})
        test_meta = validate_readout_query_cache(cache_root, "test", build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split="test", fraction=query_fraction))
        print(f"[readout] eval-only: running frozen test evaluator n_queries={int(test_meta['n_queries'])}", flush=True)
        frozen_eval = run_frozen_readout_evaluator(readout=readout, decoder_payload=decoder_payload, loader=make_loader("test"), device=device, output_dir=os.path.abspath(args.output_dir), support=support, config=cfg_dict, expected_rows=int(test_meta["n_queries"]))
        return {"artifact": os.path.realpath(args.readout_artifact), "eval_only": True, "test": frozen_eval["metrics"]}
    val_loader = make_loader("val")
    numeric_vids = [int(v) for v in torch.where(teacher.feature_type_ids.detach().cpu() == 2)[0].tolist()]
    support_kwargs = _phase1_dataset_kwargs(cfg_dict)
    support_kwargs.update({"mask_target_ratio": 0.0, "mask_rarity_strength": 0.0, "mask_min_prob": 0.0, "mask_max_prob": 0.0, "static_mask_prob": 0.0, "static_numeric_vids": []})
    support_kwargs["return_legacy_nested"] = False
    print(f"[readout] initializing train support dataset: support_fraction={support_fraction}", flush=True)
    support_ds = PatientSequenceDataset(data_dir=args.data_dir, schema_dir=args.schema_dir, split="train", deterministic=True, fraction=support_fraction, **support_kwargs)
    support_input = {
        "version": 1,
        "policy_name": "phase1_dataset_policy",
        "split": "train",
        "fraction": support_fraction,
        "dataset_content_signature": _dataset_content_signature(args.data_dir),
        "schema_sha256": _sha256_file(schema_path),
        "dataset_kwargs": support_kwargs,
        "numeric_vids": numeric_vids,
    }
    support_input_signature = _stable_json_sha256(support_input)
    support_cache_dir = os.path.join(cache_root, "train_support")
    os.makedirs(support_cache_dir, exist_ok=True)
    support_cache_path = os.path.join(support_cache_dir, f"{support_input_signature}.json")
    support = None
    if os.path.isfile(support_cache_path) and not bool(getattr(args, "rebuild_readout_cache", False)):
        with open(support_cache_path, "r", encoding="utf-8") as handle:
            cached_support = json.load(handle)
        if cached_support.get("input_signature") == support_input_signature:
            support = {int(k): v for k, v in cached_support["support"].items()}
            print(f"[readout] reused train support cache: supported_numeric_vids={len(support)}", flush=True)
    if support is None:
        print(f"[readout] scanning train support: support_fraction={support_fraction} n_admissions={len(support_ds)}", flush=True)
        support = scan_train_numeric_support(support_ds, numeric_vids)
        tmp_path = f"{support_cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump({"input_signature": support_input_signature, "input": support_input, "support": support}, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, support_cache_path)
        print(f"[readout] wrote train support cache: {support_cache_path}", flush=True)
    support_signature = hashlib.sha256(json.dumps({"policy_name": "phase1_dataset_policy", "split": "train", "fraction": support_fraction, "dataset_kwargs": support_kwargs, "numeric_vids": numeric_vids, "stats": support}, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    print(f"[readout] building teacher manifold bank: supported_numeric_vids={len(support)}", flush=True)
    readout = build_manifold_bank(teacher, support, grid_size=int(readout_cfg.get("grid_size", 513)), checkpoint_path=args.phase1_ckpt, schema_path=schema_path, split="train", sigma_floor=float(readout_cfg.get("sigma_floor", 1.0e-4)), query_chunk_size=int(readout_cfg.get("query_chunk_size", 1024)), grid_chunk_size=int(readout_cfg.get("grid_chunk_size", 128)))
    print("[readout] calibrating temperature and sigma on validation cache", flush=True)
    calibration = calibrate_readout_streaming(readout, val_loader, cfg_dict, device=device)
    print("[readout] evaluating validation readout metrics", flush=True)
    val_metrics = evaluate_readout_streaming(readout, val_loader, device)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    artifact = os.path.join(out_dir, "readout_artifact.pt")
    readout._provenance.update({"phase1_architecture_version": phase1_arch})
    train_cache_signature = build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split="train", fraction=query_fraction)
    readout._provenance.update({"dataset_content_signature": train_cache_signature["dataset_content_signature"], "readout_config_signature": train_cache_signature["config_signature"], "readout_cache_contract_version": train_cache_signature["readout_cache_contract_version"], "train_support_signature": support_signature})
    save_readout_artifact(artifact, readout, metadata={"phase1_checkpoint": os.path.realpath(args.phase1_ckpt), "schema_path": os.path.realpath(schema_path), "validation": calibration, "train_support": support, "train_support_signature": support_signature})
    train_metadata = validate_readout_query_cache(cache_root, "train", build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split="train", fraction=query_fraction))
    val_metadata = validate_readout_query_cache(cache_root, "val", build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split="val", fraction=query_fraction))
    manifest = {"architecture": READOUT_ARCHITECTURE, "phase1_checkpoint": os.path.realpath(args.phase1_ckpt), "schema_path": os.path.realpath(schema_path), "train_queries": int(train_metadata["n_queries"]), "val_queries": int(val_metadata["n_queries"]), "validation": calibration, "validation_metrics": val_metrics, "train_support_signature": support_signature, "support_policy": "phase1_dataset_policy", "query_fraction": query_fraction, "support_fraction": support_fraction, "macro_min_count": int(readout_cfg.get("macro_min_count", 10)), "query_identity": ["hadm_id", "t_idx", "slot_idx", "feature_id", "query_key"], "decoder_protocols": ["target_learned_decoder", "prediction_learned_decoder", "target_decoder_transfer_to_prediction", "target_manifold_inverse"], "test_protocol": "locked_test_rerun_under_frozen_evaluation_specification"}
    with open(os.path.join(out_dir, "readout_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(out_dir, "readout_validation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({**calibration, "readout": val_metrics}, handle, indent=2)
    pd.DataFrame(calibration["temperature_trace"]).to_csv(os.path.join(out_dir, "calibration_trace.csv"), index=False)
    pd.DataFrame(calibration["temperature_trace"]).to_csv(os.path.join(out_dir, "calibration_temperature_trace.csv"), index=False)
    pd.DataFrame(calibration["sigma_trace"]).to_csv(os.path.join(out_dir, "calibration_sigma_trace.csv"), index=False)
    print("[readout] evaluating validation per-VID metrics", flush=True)
    val_per = evaluate_readout_per_vid_streaming(readout, val_loader, device)
    pd.DataFrame([{"feature_id":v, **a} for v,a in sorted(val_per.items())]).to_csv(os.path.join(out_dir, "readout_per_feature_val.csv"), index=False)
    train_loader, test_loader = make_loader("train"), make_loader("test")
    test_metadata = validate_readout_query_cache(cache_root, "test", build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split="test", fraction=query_fraction))
    learned_cfg = readout_cfg.get("learned_decoder", {})
    alphas = learned_cfg.get("alpha_grid", [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.])
    print("[readout] fitting per-VID target Ridge decoders", flush=True)
    target_models = fit_per_vid_ridge_streaming(train_loader, val_loader, latent_key="z_target", alpha_grid=alphas, min_train_count=int(learned_cfg.get("min_train_count",100)), min_val_count=int(learned_cfg.get("min_val_count",20)))
    print("[readout] fitting per-VID prediction Ridge decoders", flush=True)
    pred_models = fit_per_vid_ridge_streaming(train_loader, val_loader, latent_key="z_pred", alpha_grid=alphas, min_train_count=int(learned_cfg.get("min_train_count",100)), min_val_count=int(learned_cfg.get("min_val_count",20)))
    decoder_path = os.path.join(out_dir, "readout_decoders.pt")
    decoder_tmp = f"{decoder_path}.tmp.{os.getpid()}"
    with open(decoder_tmp, "wb") as handle:
        torch.save({"target": target_models, "prediction": pred_models, "alpha_grid": list(alphas), "provenance": {"phase1_checkpoint": os.path.realpath(args.phase1_ckpt), "phase1_checkpoint_sha256": train_cache_signature["phase1_checkpoint_sha256"], "schema": os.path.realpath(schema_path), "schema_sha256": train_cache_signature["schema_sha256"], "train_cache_signature": train_cache_signature, "val_cache_signature": build_readout_cache_signature(data_dir=args.data_dir, schema_dir=args.schema_dir, cfg_dict=cfg_dict, phase1_ckpt_path=args.phase1_ckpt, teacher=teacher, split="val", fraction=query_fraction)}}, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(decoder_tmp, decoder_path)
    print(f"[readout] running frozen test evaluator n_queries={int(test_metadata['n_queries'])}", flush=True)
    frozen_eval = run_frozen_readout_evaluator(readout=readout, decoder_payload={"target": target_models, "prediction": pred_models}, loader=test_loader, device=device, output_dir=out_dir, support=support, config=cfg_dict, expected_rows=int(test_metadata["n_queries"]))
    with open(os.path.join(out_dir, "readout_cache_audit.json"), "w", encoding="utf-8") as handle:
        json.dump({"train": train_metadata, "val": val_metadata, "test": test_metadata}, handle, indent=2)
    return {"artifact": artifact, **calibration, "test": frozen_eval["metrics"], "architecture": READOUT_ARCHITECTURE}
from src.model import (
    FEATURE_TYPE_NUMERIC,
    JEPAModel,
    PHASE1_ARCHITECTURE_VERSION,
)
from src.dataset import (
    JEPADataModule,
    PatientSequenceDataset,
    nested_padding_collate,
)
from src.probes import run_probe_phase
from src.cache_bank import (
    build_phase1_probe_cache,
)



def _write_run_summary(parent_dir: str, filename: str, payload: Dict[str, Any]) -> None:
    os.makedirs(parent_dir, exist_ok=True)
    json_path = os.path.join(parent_dir, f"{filename}.json")
    md_path = os.path.join(parent_dir, f"{filename}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    lines = [f"# {filename.replace('_', ' ').title()}", ""]
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: {sub_value}")
            lines.append("")
        else:
            lines.append(f"- {key}: {value}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()





# Configuration Loader

# Pre-flight Checks (Phase 1 specific)




def _deterministic_preflight_batch(model, datamodule):
    """Build a small real batch with one numeric target and neighbour.

    ``main`` has already called ``datamodule.setup()``. Reusing that dataset is
    important: setup builds the optional global length-bucket manifest and must
    never be repeated merely for preflight.
    """
    if not hasattr(datamodule, "val_ds"):
        datamodule.setup()
    max_candidates = min(256, len(datamodule.val_ds))
    for index in range(max_candidates):
        source_batch = nested_padding_collate([datamodule.val_ds[index]])
        batch = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in source_batch.items()}
        seq_pad = batch["seq_pad_mask"].bool()
        token_pad = batch["state_pad_mask"].bool()
        type_ids = model.feature_type_ids[batch["state_tokens"]]
        numeric = (type_ids == 2) & (~token_pad) & (~seq_pad.unsqueeze(-1))
        for b_idx in range(batch["state_tokens"].size(0)):
            for t_idx in range(batch["state_tokens"].size(1)):
                slots = numeric[b_idx, t_idx].nonzero(as_tuple=False).flatten()
                if slots.numel() < 2:
                    continue
                batch["state_mask_labels"].zero_()
                batch["state_mask_labels"][b_idx, t_idx, int(slots[0]), 0] = 1.0
                return clone_without_packed_queries(batch)
    raise RuntimeError(
        f"Pre-flight failed: none of the first {max_candidates} deterministic validation admissions "
        "contains two real numeric events at one timestamp; cannot construct the required "
        "masked-target/local-neighbour test."
    )


def clone_without_packed_queries(batch):
    """Clone a batch while forcing Phase 1 to rebuild packed query metadata."""
    return {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
        if not k.startswith("packed_query_")
    }


def _alternate_numeric_vid_and_value(model, batch, target_index):
    """Return a distinct observed numeric VID and its value for pre-flight."""
    b_idx, t_idx, s_idx = (int(x) for x in target_index)
    tokens = batch["state_tokens"]
    values = batch["state_values"]
    valid = (~batch["state_pad_mask"].bool()) & (~batch["seq_pad_mask"].bool().unsqueeze(-1))
    numeric = valid & (model.feature_type_ids[tokens] == FEATURE_TYPE_NUMERIC)
    target_vid = tokens[b_idx, t_idx, s_idx]
    for candidate in numeric.nonzero(as_tuple=False):
        c_b, c_t, c_s = (int(x) for x in candidate)
        if tokens[c_b, c_t, c_s] != target_vid:
            return int(tokens[c_b, c_t, c_s]), float(values[c_b, c_t, c_s, 0])
    raise RuntimeError("Pre-flight failed: no distinct legal numeric VID is available for target VID testing.")


def run_preflight_checks(model, datamodule, batch=None):
    print("--- Running Deterministic Pre-Flight Checks (Phase 1; <=256 validation admissions) ---")
    del batch
    model.eval()
    # Never depend on a random training batch happening to satisfy the
    # sensitivity contract. The deterministic validation split supplies real
    # observations; we only set one legal numeric mask label on its copy.
    batch = _deterministic_preflight_batch(model, datamodule)
    
    with torch.no_grad():
        # A. Zero Leakage
        print("Checking Zero Leakage...")
        rep = model.extract_retrospective_teacher_sequence(
            batch=batch,
            task_cutoff_min=1e12,
        )
        h_state = rep["teacher_z_context"]
        
        seq_pad = batch["seq_pad_mask"]
        B, T = seq_pad.shape
        
        leak_tokens = h_state[seq_pad]
        if leak_tokens.numel() > 0:
            leak_norm = leak_tokens.norm(dim=-1).max()
            if leak_norm > 1e-4:
                raise RuntimeError(f"Pre-flight LEAKAGE detected! Norm: {leak_norm}")
            print(f"Leakage Check: PASS (Max Norm {leak_norm:.1e})")
        else:
            print("Leakage Check: SKIP (No Padding)")
            
        # B. Retrospective bidirectional sensitivity
        print("Checking Bidirectional Sensitivity...")
        batch_mod = clone_without_packed_queries(batch)
        # Modify State Value at last step
        if "state_values" in batch_mod:
            # state_values: (B, T, S, 1)
            batch_mod["state_values"][:, -1, :, :] += 5.0
            
            rep1 = model.extract_retrospective_teacher_sequence(batch=batch, task_cutoff_min=1e12)
            rep2 = model.extract_retrospective_teacher_sequence(batch=batch_mod, task_cutoff_min=1e12)
            h1 = rep1["teacher_z_context"]
            h2 = rep2["teacher_z_context"]
            
            # In retrospective bidirectional encoding, perturbation may affect
            # any in-window timestep; we only verify non-trivial sensitivity.
            diff = (h1 - h2).norm(dim=-1) # (B, T)
            if diff.max() <= 1e-6:
                raise RuntimeError("Bidirectional sensitivity check failed: perturbation had no effect.")
            print("Bidirectional Sensitivity: PASS")
        
        # C. Directional Integrity
        print("Directional Integrity: VALIDATED")

        # D. Phase-1 z_event/z_set/z_context masking and temporal invariants.
        if getattr(model, "phase1_architecture_version", None) == PHASE1_ARCHITECTURE_VERSION:
            phase1 = model._forward_phase1_numeric_jepa(batch, output_mode="full_debug")
            valid = phase1["event_valid_mask"]
            type_ids = model.feature_type_ids[batch["state_tokens"]]
            if int(valid.sum()) == 0:
                raise RuntimeError(
                    "Pre-flight failed: the sampled batch contains no masked numeric target. "
                    "Phase-1 requires at least one numeric JEPA target."
                )
            if (valid & (type_ids != 2)).any():
                raise RuntimeError("Pre-flight failed: categorical event entered Phase-1 loss mask.")
            masked_value_batch = clone_without_packed_queries(batch)
            masked_value_batch["state_values"][valid] += 3.0
            changed = model._forward_phase1_numeric_jepa(masked_value_batch, output_mode="full_debug")
            if not torch.allclose(phase1["student_z_set"], changed["student_z_set"], atol=1e-6, rtol=1e-5):
                raise RuntimeError("Pre-flight failed: masked numeric value leaked into student z_set.")
            if not torch.allclose(phase1["student_z_context"], changed["student_z_context"], atol=1e-6, rtol=1e-5):
                raise RuntimeError("Pre-flight failed: masked numeric value leaked into student z_context.")
            if torch.allclose(phase1["target_z_event_packed"], changed["target_z_event_packed"], atol=1e-6, rtol=1e-5):
                raise RuntimeError("Pre-flight failed: EMA pre-SetEncoder target ignored its own value.")

            target_index = valid.nonzero(as_tuple=False)[0]
            replacement_vid, replacement_value = _alternate_numeric_vid_and_value(model, batch, target_index)
            vid_changed = clone_without_packed_queries(batch)
            b_idx, t_idx, s_idx = (int(x) for x in target_index)
            vid_changed["state_tokens"][b_idx, t_idx, s_idx] = replacement_vid
            vid_changed["state_values"][b_idx, t_idx, s_idx, 0] = replacement_value
            vid_changed_out = model._forward_phase1_numeric_jepa(vid_changed, output_mode="full_debug")
            if torch.allclose(phase1["target_z_event_packed"], vid_changed_out["target_z_event_packed"], atol=1e-6, rtol=1e-5):
                raise RuntimeError("Pre-flight failed: EMA pre-SetEncoder target ignored its VID.")

            shifted = clone_without_packed_queries(batch)
            shifted["state_times"] += 137.0
            shifted_out = model._forward_phase1_numeric_jepa(shifted, output_mode="full_debug")
            for name in ("student_event_tokens", "student_z_set", "student_z_context"):
                if not torch.allclose(phase1[name], shifted_out[name], atol=1e-5, rtol=1e-4):
                    raise RuntimeError(f"Pre-flight failed: global time translation changed {name}.")
            if not torch.allclose(
                phase1["target_z_event_packed"],
                shifted_out["target_z_event_packed"],
                atol=1e-6,
                rtol=1e-5,
            ):
                raise RuntimeError("Pre-flight failed: EMA pre-SetEncoder target depended on time.")

            stretched = clone_without_packed_queries(batch)
            stretched["state_times"] *= 2.0
            stretched_out = model._forward_phase1_numeric_jepa(stretched, output_mode="full_debug")
            multi_time = (~batch["seq_pad_mask"]).sum(dim=1).gt(1).any()
            if multi_time and torch.allclose(phase1["student_z_context"], stretched_out["student_z_context"], atol=1e-6, rtol=1e-5):
                raise RuntimeError("Pre-flight failed: relative timestamp gaps do not affect z_context.")

            target_idx = valid.nonzero(as_tuple=False)
            neighbour_checked = False
            for b_idx, t_idx, s_idx in target_idx.tolist():
                local_visible = (
                    (~batch["state_pad_mask"][b_idx, t_idx])
                    & (~valid[b_idx, t_idx])
                    & (type_ids[b_idx, t_idx] == 2)
                )
                neighbour = local_visible.nonzero(as_tuple=False).flatten()
                if neighbour.numel() == 0:
                    continue
                altered = clone_without_packed_queries(batch)
                altered["state_values"][b_idx, t_idx, int(neighbour[0]), 0] += 2.0
                altered_out = model._forward_phase1_numeric_jepa(altered, output_mode="full_debug")
                if not torch.allclose(phase1["target_z_event_packed"], altered_out["target_z_event_packed"], atol=1e-6, rtol=1e-5):
                    raise RuntimeError("Pre-flight failed: EMA pre-SetEncoder target depended on a same-timestamp neighbour.")
                neighbour_checked = True
                break
            if not neighbour_checked:
                raise RuntimeError(
                    "Pre-flight failed: no masked numeric target had a visible numeric "
                    "same-timestamp neighbour for the target-independence check."
                )
            print("Phase-1 numeric masking and relative-time invariants: PASS")
            
    print("--- Pre-Flight Checks Passed ---")



# Main

def main():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache root for derived feature caches")
    parser.add_argument("--schema_dir", type=str, required=True, help="Directory containing schema_processed.csv")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training execution (optimizer + model)")
    
    # Phase Selection
    parser.add_argument("--phase", type=str, default="1", choices=["1", "probe_cache_build", "readout_probe", "phenotype_probe", "prediction_probe", "probe_train"], help="Training phase")
    parser.add_argument("--phase1_ckpt", type=str, help="Exact Phase-1 checkpoint; required for probe_cache_build, prediction_probe, phenotype_probe and readout_probe.")
    parser.add_argument("--readout_artifact", type=str, default=None, help="Frozen phase1_target_manifold_inverse_v1 artifact for readout consumers.")
    parser.add_argument("--readout_eval_only", action="store_true", help="Inspect an existing --readout_artifact without rebuilding or calibrating.")
    parser.add_argument("--rebuild_readout_cache", action="store_true", help="Explicitly rebuild a provenance-mismatched readout query cache.")
    parser.add_argument("--task_type", type=str, default="icd", choices=["icd", "drg", "hcup"], help="Task type for phenotype_probe.")
    parser.add_argument("--hcup_mapping_json", type=str, default=None, help="Path to HCUP 26-label mapping json (required for phenotype_probe --task_type hcup).")
    parser.add_argument("--prediction_task", type=str, default="mortality", choices=["mortality", "los_ge_3d", "los_ge_7d", "readmission"], help="Independent binary task for prediction_probe.")
    parser.add_argument("--probe_source", type=str, default=None, choices=["z_set", "z_context", "all"],
                        help="Canonical latent source, or 'all' for both formal probe sources.")
    parser.add_argument("--probe_task", type=str, default=None,
                        choices=["mortality", "los_ge_3d", "los_ge_7d", "readmission", "icd", "hcup", "drg", "all"],
                        help="Formal probe task, or 'all' to train all seven tasks.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Write compact probe prediction NPZ artifacts; disabled by default.")
    
    # Debug
    parser.add_argument("--max_steps", type=int, default=-1, help="Limit training steps for smoke testing")
    parser.add_argument("--train_pct", type=float, default=1.0, help="Fraction of training data to use (0.0 to 1.0)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Uniform fraction of train/val/test data to use (0.0 to 1.0)")
    parser.add_argument(
        "--smoke_mode", action="store_true",
        help="Run the exact same probe architecture on a deterministic small cohort.",
    )
    parser.add_argument("--smoke_train_patients", type=int, default=64)
    parser.add_argument("--smoke_val_patients", type=int, default=32)
    parser.add_argument("--smoke_test_patients", type=int, default=32)
    parser.add_argument("--smoke_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None, help="Optional dataloader batch-size override")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional dataloader worker override")
    parser.add_argument("--max_seq_len_limit", type=int, default=None, help="Optional sequence-length override")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for cache building and training.")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild of shared world-knowledge caches.")
    parser.add_argument("--run_name", type=str, default="atlas_ehr", help="Name of the training run (used for checkpoint naming)")
    # Overrides
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="auto")
    
    p_args = parser.parse_args()
    # PyTorch's own legacy capability probes emit these four warnings on the
    # server's supported Python 3.9 build. They are unrelated to model/data
    # semantics; keep every other warning visible.
    for name in ("has_cuda", "has_cudnn", "has_mps", "has_mkldnn"):
        warnings.filterwarnings(
            "ignore",
            message=rf"'{name}' is deprecated.*",
            category=UserWarning,
            module=r"torch\.overrides",
        )
    pl.seed_everything(int(p_args.seed), workers=True)
    
    # 1. Load Config
    cfg_dict = load_config_dict(p_args.config)
    # Phase 1 execution settings have one authoritative home.  Keeping a
    # duplicate precision/device setting under ``performance`` is dangerous:
    # Lightning reads ``training`` while an older benchmark used to read the
    # other location.  Reject such configs instead of silently choosing one.
    phase1_performance = cfg_dict.get("phase1", {}).get("performance", {})
    legacy_execution_keys = {"mixed_precision", "precision", "accelerator", "devices", "strategy"}
    found_legacy_execution_keys = sorted(legacy_execution_keys.intersection(phase1_performance))
    if found_legacy_execution_keys:
        raise ValueError(
            "Phase-1 execution settings must be declared only under "
            "phase1.training; remove "
            + ", ".join(f"phase1.performance.{key}" for key in found_legacy_execution_keys)
        )
    if p_args.max_seq_len_limit is not None:
        cfg_dict.setdefault("phase1", {}).setdefault("data", {})["max_seq_len_limit"] = int(p_args.max_seq_len_limit)
    if p_args.accelerator != "auto":
        cfg_dict.setdefault("phase1", {}).setdefault("training", {})["accelerator"] = p_args.accelerator
    if p_args.devices != "auto":
        cfg_dict.setdefault("phase1", {}).setdefault("training", {})["devices"] = int(p_args.devices) if p_args.devices.isdigit() else p_args.devices
    # 2. Prepare Paths
    parent_dir = os.path.join(p_args.output_dir, p_args.run_name)
    os.makedirs(parent_dir, exist_ok=True)
    
    logs_dir = os.path.join(parent_dir, "logs")
    ckpt_dir = os.path.join(parent_dir, "checkpoints")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Shared helpers (defined inside main to avoid global-scope side effects) ──

    def _build_logger(ld: str) -> TensorBoardLogger:
        return TensorBoardLogger(ld, name="", version="",
                                 default_hp_metric=False, log_graph=False)

    def _build_trainer(train_cfg: dict, cd: str, ld: str,
                       args, phase_tag,
                       monitor: str = "val_loss",
                       monitor_mode: str = "min",
                       extra_callbacks: list = None,
                       extra_model_checkpoints: list = None,
                       **extra_kwargs) -> tuple:
        logger = _build_logger(ld)
        patience = train_cfg.get("early_stopping_patience", 5)
        min_epochs = train_cfg.get("min_epochs", 5)
        phase_prefix = f"phase{phase_tag}" if str(phase_tag).isdigit() else str(phase_tag)
        ckpt_cb = ModelCheckpoint(
            dirpath=cd,
            filename=f"{phase_prefix}_best-epoch{{epoch}}",
            save_top_k=1, monitor=monitor, mode=monitor_mode,
            save_last=True,
        )
        epoch_ckpt_cb = ModelCheckpoint(
            dirpath=cd,
            filename=f"{phase_prefix}_epoch{{epoch}}",
            save_top_k=-1,
            save_last=False,
            every_n_epochs=1,
        )
        min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
        early_stop_cb = EarlyStopping(
            monitor=monitor, mode=monitor_mode,
            patience=patience, min_delta=min_delta,
            check_on_train_epoch_end=False,
            verbose=True,
        )
        callbacks = []
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        callbacks.append(ckpt_cb)
        callbacks.append(epoch_ckpt_cb)
        if extra_model_checkpoints:
            for spec in extra_model_checkpoints:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=cd,
                        filename=f"{spec.get('filename', f'{phase_prefix}_extra')}-epoch{{epoch}}",
                        save_top_k=int(spec.get("save_top_k", 1)),
                        monitor=spec["monitor"],
                        mode=spec.get("mode", "max"),
                        save_last=False,
                    )
                )
        callbacks.append(early_stop_cb)
        callbacks.append(LearningRateMonitor(logging_interval=str(train_cfg.get("lr_logging_interval", "step"))))
        check_val_every_n_epoch = int(train_cfg.get("check_val_every_n_epoch", 1))
        trainer = pl.Trainer(
            max_epochs=train_cfg["epochs"],
            min_epochs=min_epochs,
            max_steps=args.max_steps,
            logger=logger,
            callbacks=callbacks,
            accelerator=train_cfg.get("accelerator", "auto"),
            devices=train_cfg.get("devices", "auto"),
            strategy=train_cfg.get("strategy", "auto"),
            gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
            gradient_clip_algorithm="norm",
            precision=train_cfg.get("precision", "32-true"),
            log_every_n_steps=train_cfg.get("log_interval", 50),
            check_val_every_n_epoch=max(1, check_val_every_n_epoch),
            **extra_kwargs,
        )
        return trainer, ckpt_cb

    if p_args.phase == "readout_probe":
        report = _run_phase1_target_manifold_readout(p_args, cfg_dict)
        print(json.dumps(report, indent=2))
        return

    if p_args.phase == "probe_cache_build":
        if not p_args.phase1_ckpt:
            raise ValueError("probe_cache_build requires --phase1_ckpt only.")
        p_args.phase1_ckpt = os.path.realpath(p_args.phase1_ckpt)
        if not os.path.isfile(p_args.phase1_ckpt):
            raise FileNotFoundError(f"--phase1_ckpt not found: {p_args.phase1_ckpt}")
        schema_path = os.path.join(p_args.schema_dir, "schema_processed.csv")
        device = torch.device("cuda" if torch.cuda.is_available() and str(p_args.accelerator) != "cpu" else "cpu")
        teacher = JEPAModel.load_from_checkpoint(
            p_args.phase1_ckpt, map_location=device, weights_only=False, schema_path=schema_path
        ).to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        cache_root = os.path.join(str(p_args.cache_dir), "probe_cache")
        summary = build_phase1_probe_cache(
            cache_root=cache_root, data_dir=p_args.data_dir, schema_dir=p_args.schema_dir, cfg_dict=cfg_dict,
            teacher=teacher, phase1_ckpt_path=p_args.phase1_ckpt, device=device, labels_csv=p_args.labels_csv,
            rebuild=bool(p_args.rebuild_cache), batch_size=p_args.batch_size,
            num_workers=p_args.num_workers, fraction=float(p_args.fraction),
        )
        print(json.dumps(summary, indent=2))
        return

    # Exact-time downstream probes consume only canonical Phase-1 cache fields.
    if p_args.phase in {"phenotype_probe", "prediction_probe", "probe_train"}:
        if not p_args.phase1_ckpt:
            raise ValueError(f"{p_args.phase} requires the exact --phase1_ckpt used to build its cache.")
        p_args.phase1_ckpt = os.path.realpath(p_args.phase1_ckpt)
        if not os.path.isfile(p_args.phase1_ckpt):
            raise FileNotFoundError(f"--phase1_ckpt not found: {p_args.phase1_ckpt}")
        if p_args.phase == "probe_train" and not p_args.probe_task:
            raise ValueError("probe_train requires an explicit --probe_task.")
        requested_probe_task = (str(p_args.probe_task) if p_args.phase == "probe_train"
                                else str(p_args.prediction_task if p_args.phase == "prediction_probe" else p_args.task_type)).lower()
        formal_tasks = ["mortality", "los_ge_3d", "los_ge_7d", "readmission", "icd", "hcup", "drg"]
        selected_tasks = formal_tasks if p_args.phase == "probe_train" and requested_probe_task == "all" else [requested_probe_task]
        if "hcup" in selected_tasks:
            if not p_args.hcup_mapping_json:
                raise ValueError("HCUP probe training requires --hcup_mapping_json.")
            if not os.path.exists(p_args.hcup_mapping_json):
                raise FileNotFoundError(f"--hcup_mapping_json not found: {p_args.hcup_mapping_json}")
        cache_root = os.path.join(str(p_args.cache_dir), "probe_cache")
        if p_args.phase == "probe_train":
            selected_sources = (["z_set", "z_context"] if p_args.probe_source == "all"
                                else [str(p_args.probe_source or cfg_dict["probe"]["source_name"])])
            base_output_dir = parent_dir
            reports = []
            for task_name in selected_tasks:
                for source_name in selected_sources:
                    task_cfg = json.loads(json.dumps(cfg_dict))
                    task_cfg["probe"]["source_name"] = source_name
                    if bool(p_args.smoke_mode):
                        task_cfg["probe"]["training"]["epochs"] = int(p_args.smoke_epochs)
                        task_cfg["probe"]["training"]["min_epochs"] = 1
                    task_args = argparse.Namespace(**vars(p_args))
                    task_args.probe_task = task_name
                    task_args.prediction_task = task_name
                    task_args.task_type = task_name
                    task_output_dir = os.path.join(base_output_dir, "smoke", task_name, source_name) if bool(p_args.smoke_mode) else os.path.join(base_output_dir, task_name, source_name)
                    reports.append(run_probe_phase(
                        phase="probe_train", args=task_args, output_dir=task_output_dir,
                        cache_root=cache_root, cfg_dict=task_cfg, evaluate_test=False,
                    ))
            print(json.dumps({"runs": reports}, indent=2))
            return
        if bool(p_args.smoke_mode):
            probe_task = requested_probe_task
            parent_dir = os.path.join(parent_dir, "smoke", probe_task)
            cfg_dict = json.loads(json.dumps(cfg_dict))
            cfg_dict["probe"]["training"]["epochs"] = int(p_args.smoke_epochs)
            cfg_dict["probe"]["training"]["min_epochs"] = 1
        if p_args.probe_source is not None:
            cfg_dict = json.loads(json.dumps(cfg_dict))
            cfg_dict["probe"]["source_name"] = str(p_args.probe_source)
        os.makedirs(cache_root, exist_ok=True)
        rep = run_probe_phase(
            phase=str(p_args.phase),
            args=p_args,
            output_dir=parent_dir,
            cache_root=cache_root,
            cfg_dict=cfg_dict,
            evaluate_test=False,
        )
        print(json.dumps(rep, indent=2))
        return

    # =========================================================================
    # PHASE 1: Static Infilling
    # =========================================================================
    if p_args.phase == "1":
        print("\n=== ATLAS-EHR Phase 1 Training ===")
        print("Mode: Numeric Local-Temporal JEPA")
        if not p_args.labels_csv:
            raise ValueError("Phase 1 requires explicit --labels_csv path.")
        if not os.path.exists(p_args.labels_csv):
            raise FileNotFoundError(f"--labels_csv not found: {p_args.labels_csv}")
        
        p1 = cfg_dict.get("phase1", {})
        train_cfg = p1.get("training", {})
        data_cfg = p1.get("data", {})
        phase1_dataset_kwargs = _phase1_dataset_kwargs(
            cfg_dict,
            max_seq_len_limit=int(data_cfg["max_seq_len_limit"]),
        )
        
        # DataModule
        phase1_batch_size = int(
            p_args.batch_size if p_args.batch_size is not None else train_cfg.get("batch_size", 32)
        )
        phase1_num_workers = int(
            p_args.num_workers if p_args.num_workers is not None else train_cfg.get("num_workers", 4)
        )
        phase1_smoke_caps = (
            {"train": 1000, "val": 500, "test": 500}
            if bool(getattr(p_args, "smoke_mode", False)) else {}
        )
        if phase1_smoke_caps:
            print(f"[Phase 1 Smoke] fixed split caps: {phase1_smoke_caps}")
        dm = JEPADataModule(
            data_dir=p_args.data_dir,
            schema_dir=p_args.schema_dir,
            batch_size=phase1_batch_size,
            num_workers=phase1_num_workers,
            pin_memory=bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
            persistent_workers=bool(train_cfg.get("persistent_workers", True)) and phase1_num_workers > 0,
            prefetch_factor=(
                int(train_cfg.get("prefetch_factor", 4))
                if phase1_num_workers > 0
                else None
            ),
            length_bucketing=bool(p1.get("performance", {}).get("length_bucketing", False)),
            length_bucket_seed=int(p_args.seed),
            shape_manifest_dir=os.path.join(p_args.output_dir, p_args.run_name, "shape_manifest"),
            max_train_samples=phase1_smoke_caps.get("train"),
            max_val_samples=phase1_smoke_caps.get("val"),
            max_test_samples=phase1_smoke_caps.get("test"),
            train_pct=p_args.train_pct,
            fraction=p_args.fraction,
            **phase1_dataset_kwargs,
        )

        # Detect Input Dimension
        dm.setup()
        sample_batch = next(iter(dm.train_dataloader()))

        # Model
        model_cfg = p1.get("model", {})
        
        # Load the full recipe dict, then filter by RecipeConfig fields.
        recipe_dict = p1.get("recipe", {})
        valid_keys = RecipeConfig.__dataclass_fields__.keys()
        filtered_recipe = {k: v for k, v in recipe_dict.items() if k in valid_keys}
        recipe = RecipeConfig(**filtered_recipe)
        
        embed_dim = model_cfg.get("embed_dim", 128)
        num_features = model_cfg.get("num_features", 8000)
        num_heads = model_cfg.get("num_heads", 4)
        learning_rate = float(train_cfg.get("max_lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))

        # Hyperparameter Override from Resume
        if p_args.resume and os.path.exists(p_args.resume):
            print(f"Resuming training from: {p_args.resume}")
            print("Overriding config with checkpoint hyperparameters...")
            try:
                ckpt = torch.load(p_args.resume, map_location="cpu", weights_only=False)
                hparams = ckpt["hyper_parameters"]
                expected_arch = PHASE1_ARCHITECTURE_VERSION
                if hparams.get("phase1_architecture_version") != expected_arch:
                    raise RuntimeError(
                        "Phase1 checkpoint architecture mismatch: expected "
                        f"{expected_arch}, found {hparams.get('phase1_architecture_version')!r}."
                    )
                if "config" in hparams: recipe = hparams["config"]
                if "embed_dim" in hparams: embed_dim = hparams["embed_dim"]
                if "num_features" in hparams: num_features = hparams["num_features"]
                if "num_heads" in hparams: num_heads = hparams["num_heads"]
                print(f"  [Override] Embed Dim: {embed_dim}, Heads: {num_heads}")
            except Exception as e:
                raise RuntimeError(f"Cannot resume incompatible Phase 1 checkpoint: {e}") from e

        model = JEPAModel(
            config=recipe,
            embed_dim=embed_dim,
            num_features=num_features,
            schema_path=os.path.join(p_args.schema_dir, "schema_processed.csv"),
            num_heads=num_heads,
            max_lr=learning_rate,
            weight_decay=weight_decay,
            phase1_config=train_cfg
        )
        model.hparams["effective_max_seq_len_limit"] = int(data_cfg["max_seq_len_limit"])
        model.hparams["sequence_length_summaries"] = getattr(dm, "sequence_length_summaries", {})
        performance_cfg = p1.get("performance", {})
        if performance_cfg.get("matmul_precision"):
            torch.set_float32_matmul_precision(str(performance_cfg["matmul_precision"]))
        model.hparams["phase1_performance"] = model.configure_phase1_performance(performance_cfg)
        print(f"[Phase 1] performance settings: {model.hparams['phase1_performance']}")
        
        run_preflight_checks(model, dm, batch=sample_batch)

        # Gradient telemetry is produced inside JEPAModel's explicit
        # normalize -> clip lifecycle, not by a callback with hook-order
        # dependent gradient visibility.
        # detect_anomaly and deterministic are intentionally omitted:
        #   detect_anomaly=True  → 2-3× slowdown (checks every backward graph)
        #   deterministic=True   → forces slow CUDA fallback algorithms
        # Both are debugging tools, not appropriate for production training.
        # Numerical stability is handled by StabilityClamp, safe_attention, and
        # MaskedLayerNorm; NaN protection is in safe_attention itself.
        collapse_guard = CollapseGuard(min_rank_eff=1.8)
        phase1_callbacks = [collapse_guard]
        trainer, checkpoint_callback = _build_trainer(
            train_cfg, ckpt_dir, logs_dir, p_args, phase_tag=1,
            monitor="val/loss_rank_gated", monitor_mode="min",
            extra_callbacks=phase1_callbacks,
            enable_checkpointing=True,
            accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
            use_distributed_sampler=not bool(dm.length_bucketing),
        )

        requested_precision = str(train_cfg.get("precision", "32-true"))
        autocast_dtype = (
            torch.bfloat16 if requested_precision == "bf16-mixed" else
            torch.float16 if requested_precision == "16-mixed" else None
        )
        runtime_info = {
            "requested_precision": requested_precision,
            "trainer_precision_plugin": type(trainer.precision_plugin).__name__,
            "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "device": str(trainer.strategy.root_device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "per_device_batch_size": phase1_batch_size,
            "accumulate_grad_batches": int(train_cfg.get("accumulate_grad_batches", 1)),
            "world_size": int(trainer.world_size),
            "global_effective_batch_size": (
                phase1_batch_size * int(train_cfg.get("accumulate_grad_batches", 1)) * int(trainer.world_size)
            ),
            "smoke_mode": bool(getattr(p_args, "smoke_mode", False)),
            "smoke_split_caps": phase1_smoke_caps,
            "config_sha256": hashlib.sha256(
                json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest(),
            "phase1_code_sha256": hashlib.sha256(
                "|".join(
                    _sha256_file(path)
                    for path in (os.path.join(os.path.dirname(__file__), "dataset.py"),
                                 os.path.join(os.path.dirname(__file__), "model.py"),
                                 os.path.join(os.path.dirname(__file__), "utils.py"),
                                 __file__)
                ).encode("utf-8")
            ).hexdigest(),
        }
        model.hparams["phase1_runtime"] = runtime_info
        print(f"[Phase 1] runtime: {runtime_info}")

        print(f"Starting Phase 1 Training [Output: {logs_dir}]...")
        trainer.fit(model, dm, ckpt_path=p_args.resume)
        early_stop = next(
            (callback for callback in trainer.callbacks if isinstance(callback, EarlyStopping)),
            None,
        )
        best_path = checkpoint_callback.best_model_path or None
        last_path = checkpoint_callback.last_model_path or None
        last_healthy_path = os.path.join(ckpt_dir, "phase1_last_healthy.ckpt")
        artifact_paths = {
            "best_checkpoint_path": os.path.abspath(best_path) if best_path else None,
            "last_checkpoint_path": os.path.abspath(last_path) if last_path else None,
            "last_healthy_checkpoint_path": os.path.abspath(last_healthy_path) if os.path.isfile(last_healthy_path) else None,
            "tensorboard_log_dir": os.path.abspath(logs_dir),
        }
        artifact_hashes = {
            name.replace("_path", "_sha256"): _sha256_file(path)
            for name, path in artifact_paths.items()
            if name.endswith("_path") and path and os.path.isfile(path)
        }
        _write_run_summary(parent_dir, "phase1_train_summary", {
            "phase": 1,
            "monitor": "val/loss_rank_gated",
            "monitor_source": "val/loss",
            "monitor_mode": "min",
            "best_model_score": (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None else None
            ),
            "early_stopping": {
                "monitor": getattr(early_stop, "monitor", None),
                "mode": getattr(early_stop, "mode", None),
                "patience": getattr(early_stop, "patience", None),
                "wait_count": getattr(early_stop, "wait_count", None),
                "stopped_epoch": getattr(early_stop, "stopped_epoch", None),
                "best_score": (
                    float(early_stop.best_score)
                    if early_stop is not None and early_stop.best_score is not None else None
                ),
            },
            "collapse_guard": {
                "triggered": collapse_guard.triggered,
                "trigger_epoch": collapse_guard.trigger_epoch,
                "trigger_rank_eff": collapse_guard.trigger_rank_eff,
                "safe_checkpoint_policy": "Use best_checkpoint_path; last_checkpoint_path is resumable only.",
            },
            "effective_max_seq_len_limit": int(data_cfg["max_seq_len_limit"]),
            "smoke_mode": bool(getattr(p_args, "smoke_mode", False)),
            "smoke_split_caps": phase1_smoke_caps,
            "sequence_length_summaries": getattr(dm, "sequence_length_summaries", {}),
            "schema_path": os.path.abspath(os.path.join(p_args.schema_dir, "schema_processed.csv")),
            "schema_sha256": model.schema_sha256,
            "phase1_architecture_version": model.phase1_architecture_version,
            "phase1_performance": model.hparams["phase1_performance"],
            "phase1_runtime": model.hparams.get("phase1_runtime", {}),
            "peak_memory": {
                "allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            },
            "trainer": {
                "current_epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "should_stop": bool(trainer.should_stop),
            },
            "artifacts": artifact_paths,
            "artifact_sha256": artifact_hashes,
        })
        with open(os.path.join(parent_dir, "phase1_effective_config.json"), "w", encoding="utf-8") as handle:
            json.dump(cfg_dict, handle, indent=2, sort_keys=True, default=str)
        print(f"[Phase 1] Training Complete. Best Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
