"""Imputation workflow implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.cache_bank import _dataset_content_signature, _phase1_dataset_kwargs, _sha256_file, _stable_json_sha256
from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.model import FEATURE_TYPE_NUMERIC, JEPAModel
from src.readout import load_readout_artifact
from src.utils import load_config_dict

# ===== imputation implementation =====
matplotlib.use("Agg")

def _imp__phase1_forward(
    teacher: JEPAModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Full Phase 1 infilling forward using the target (EMA) frontend.
    Returns z_pred (B, T, S, D).
    """
    st = batch["state_tokens"].to(device)
    sv = batch["state_values"].to(device)
    sp = batch["state_pad_mask"].to(device)
    sq = batch["seq_pad_mask"].to(device)
    ml = batch["state_mask_labels"].to(device)
    tm = batch["state_times"].to(device)
    del st, sv, sp, sq, ml, tm
    model_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    return teacher.predict_masked_numeric_latents(model_batch)["z_pred_tokens"]


def _imp__has_vid(args: Tuple[str, int]) -> bool:
    file_path, vid = args
    raw = torch.load(file_path, map_location="cpu", weights_only=True)
    flat = raw.get("state_tokens")
    return flat is not None and bool((flat == vid).any().item())

def _imp__find_occurrences(
    dataset: PatientSequenceDataset,
    vid: int,
    max_samples: Optional[int],
    cache_path: Optional[str] = None,
    num_workers: int = 4,
) -> List[Tuple[int, int, float, float, str, int]]:
    """
    Returns list of (dataset_idx, t_mask, true_value, abs_time_min, hadm_id, n_total_T).

    Two-pass: parallel raw .pt filter (pass 1) then full dataset[i] only for
    patients that contain vid (pass 2). Results cached to disk if cache_path given.
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("vid") == vid and cached.get("data_dir") == dataset.split_dir:
            results = [tuple(row) for row in cached["occurrences"]]
            print(f"[imputation] loaded {len(results)} occurrences from cache {cache_path}")
            return results  # type: ignore[return-value]
        print("[imputation] cache mismatch, re-scanning")

    args_list = [(dataset.files[i], vid) for i in range(len(dataset))]
    if num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            hits = list(pool.map(_imp__has_vid, args_list,
                                 chunksize=max(1, len(args_list) // (num_workers * 4))))
    else:
        hits = [_imp__has_vid(a) for a in args_list]
    candidate_indices = [i for i, hit in enumerate(hits) if hit]
    print(f"[imputation] pass-1: {len(candidate_indices)}/{len(dataset)} patients contain vid={vid}")

    results: List[Tuple[int, int, float, float, str, int]] = []
    seen_patients: set = set()

    for i in candidate_indices:
        item = dataset[i]
        T = len(item["state"])
        found = False
        for t in range(T):
            tokens_t = item["state"][t]["tokens"]
            hit_idx = (tokens_t == vid).nonzero(as_tuple=True)[0]
            if hit_idx.numel() == 0:
                continue
            slot = int(hit_idx[0].item())
            results.append((
                i, t,
                float(item["state"][t]["values"][slot].item()),
                float(item["state_times"][t, 0].item()),
                str(item["hadm_id"]),
                T,
            ))
            found = True
        if found:
            seen_patients.add(i)
            if max_samples is not None and len(seen_patients) >= max_samples:
                break

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"vid": vid, "data_dir": dataset.split_dir,
                       "occurrences": [list(r) for r in results]}, f)
        print(f"[imputation] saved cache → {cache_path}")

    return results

def _imp__build_masked_item(
    item: Dict[str, Any],
    t_mask: int,
    vid: int,
) -> Tuple[Optional[Dict[str, Any]], int, int, int]:
    """
    Remove vid from t_mask's context and append it as a masked query slot.
    Returns (modified_item, n_visible, item_query_slot, batch_query_slot)
    or (None, 0, -1, -1).
    """
    T = len(item["state"])
    new_state: List[Dict[str, Any]] = []
    n_visible = 0
    item_query_slot = -1
    batch_query_slot = -1

    for t in range(T):
        tokens_t = item["state"][t]["tokens"].clone()
        values_t = item["state"][t]["values"].clone()
        if t == t_mask:
            hit = (tokens_t == vid).nonzero(as_tuple=True)[0]
            if hit.numel() == 0:
                return None, 0, -1, -1
            keep = torch.ones(len(tokens_t), dtype=torch.bool)
            keep[int(hit[0].item())] = False
            ctx_tok = tokens_t[keep]
            ctx_val = values_t[keep]
            n_visible = int(keep.sum().item())
            item_query_slot = n_visible
            batch_query_slot = n_visible + 1
            q_tok = torch.cat([ctx_tok, tokens_t.new_tensor([vid])])
            q_val = torch.cat([ctx_val, values_t.new_zeros(1)])
            q_mask = torch.zeros(len(q_tok))
            q_mask[-1] = 1.0
            new_state.append({"tokens": q_tok, "values": q_val, "mask": q_mask})
        else:
            new_state.append({
                "tokens": tokens_t, "values": values_t,
                "mask": torch.zeros(len(tokens_t)),
            })

    return ({"state": new_state, "interv": item["interv"],
             "state_times": item["state_times"], "delta_times": item["delta_times"],
             "study_start": item["study_start"], "labels": item["labels"],
             "hadm_id": item["hadm_id"]},
            n_visible, item_query_slot, batch_query_slot)

def _imp__build_query_item(
    item: Dict[str, Any],
    t_query: int,
    vid: int,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Build an item that queries vid at t_query regardless of whether vid is
    present there. If vid exists at t_query, remove it from context first.
    Returns (modified_item, item_query_slot, batch_query_slot).
    """
    T = len(item["state"])
    new_state: List[Dict[str, Any]] = []
    item_query_slot = -1
    batch_query_slot = -1

    for t in range(T):
        tokens_t = item["state"][t]["tokens"].clone()
        values_t = item["state"][t]["values"].clone()
        if t == t_query:
            hit = (tokens_t == vid).nonzero(as_tuple=True)[0]
            if hit.numel() > 0:
                keep = torch.ones(len(tokens_t), dtype=torch.bool)
                keep[int(hit[0].item())] = False
                tokens_t = tokens_t[keep]
                values_t = values_t[keep]
            n_visible = len(tokens_t)
            item_query_slot = n_visible
            batch_query_slot = n_visible + 1
            q_tok = torch.cat([tokens_t, tokens_t.new_tensor([vid])])
            q_val = torch.cat([values_t, values_t.new_zeros(1)])
            q_mask = torch.zeros(len(q_tok))
            q_mask[-1] = 1.0
            new_state.append({"tokens": q_tok, "values": q_val, "mask": q_mask})
        else:
            new_state.append({
                "tokens": tokens_t, "values": values_t,
                "mask": torch.zeros(len(tokens_t)),
            })

    return ({"state": new_state, "interv": item["interv"],
             "state_times": item["state_times"], "delta_times": item["delta_times"],
             "study_start": item["study_start"], "labels": item["labels"],
             "hadm_id": item["hadm_id"]},
            item_query_slot, batch_query_slot)

class _imp__OccurrenceDataset(Dataset):
    def __init__(self, base: PatientSequenceDataset,
                 occurrences: List[Tuple[int, int, float, float, str, int]],
                 vid: int) -> None:
        self.base = base
        self.occurrences = occurrences
        self.vid = vid

    def __len__(self) -> int:
        return len(self.occurrences)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        ds_idx, t_mask, true_val, abs_t, hadm_id, n_total = self.occurrences[i]
        item = self.base[ds_idx]
        mod_item, n_vis, item_query_slot, batch_query_slot = _imp__build_masked_item(item, t_mask, self.vid)
        if mod_item is None:
            return None
        assert int(mod_item["state"][t_mask]["tokens"][item_query_slot].item()) == int(self.vid)
        assert float(mod_item["state"][t_mask]["mask"][item_query_slot].item()) == 1.0
        return {"item": mod_item,
                "meta": {"hadm_id": hadm_id, "t_mask": t_mask,
                         "abs_time_min": float(abs_t), "true_value": float(true_val),
                         "n_visible_same_t": n_vis, "n_total_timesteps": n_total,
                         "item_query_slot": item_query_slot,
                         "batch_query_slot": batch_query_slot}}

def _imp__occurrence_collate(samples):
    valid = [s for s in samples if s is not None]
    if not valid:
        return None
    return nested_padding_collate([s["item"] for s in valid]), [s["meta"] for s in valid]

class _imp__AllVidsOccurrenceDataset(Dataset):
    """Single dataset over all vids' occurrences — enables one inference pass for all-vids mode.

    Occurrences are kept in dataset_idx order (as produced by _build_occurrence_index),
    so consecutive items tend to come from the same patient. This gives good OS
    page-cache locality without preloading all patients into RAM.
    """
    def __init__(self, base: PatientSequenceDataset, occurrence_index: "pd.DataFrame") -> None:
        self.base = base
        # Sort by dataset_idx to maximise PT file cache locality.
        self.records = occurrence_index[
            ["vid", "dataset_idx", "t_mask", "true_value", "abs_time_min", "hadm_id", "n_total_T"]
        ].sort_values("dataset_idx").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        row = self.records.iloc[i]
        vid = int(row["vid"])
        item = self.base[int(row["dataset_idx"])]
        mod_item, n_vis, item_query_slot, batch_query_slot = _imp__build_masked_item(item, int(row["t_mask"]), vid)
        if mod_item is None:
            return None
        return {"item": mod_item,
                "meta": {"vid": vid, "hadm_id": str(row["hadm_id"]),
                         "t_mask": int(row["t_mask"]),
                         "abs_time_min": float(row["abs_time_min"]),
                         "true_value": float(row["true_value"]),
                         "n_visible_same_t": n_vis,
                         "n_total_timesteps": int(row["n_total_T"]),
                         "item_query_slot": item_query_slot,
                         "batch_query_slot": batch_query_slot}}

def _imp__all_vids_collate(samples):
    valid = [s for s in samples if s is not None]
    if not valid:
        return None
    return nested_padding_collate([s["item"] for s in valid]), [s["meta"] for s in valid]

def _imp__dense_predict(
    teacher: JEPAModel,
    probe: Any,
    dataset: PatientSequenceDataset,
    ds_idx: int,
    vid: int,
    vid_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict vid at every timestep for one patient.

    Returns (abs_times, mus, sigmas, obs_abs_times, obs_values).
    abs_times/mus/sigmas cover all T timesteps.
    obs_abs_times/obs_values are only the timesteps where vid was actually observed.
    """
    item = dataset[ds_idx]
    T = len(item["state"])

    items_batch = []
    batch_query_slots = []
    for t in range(T):
        mod_item, item_query_slot, batch_query_slot = _imp__build_query_item(item, t, vid)
        assert int(mod_item["state"][t]["tokens"][item_query_slot].item()) == int(vid)
        assert float(mod_item["state"][t]["mask"][item_query_slot].item()) == 1.0
        items_batch.append(mod_item)
        batch_query_slots.append(batch_query_slot)

    batch = nested_padding_collate(items_batch)
    z_pred = _imp__phase1_forward(teacher, batch, device)  # (T, T_seq, S, D)

    abs_times = item["state_times"][:, 0].numpy()  # (T,)
    mus = np.zeros(T)
    sigmas = np.zeros(T)

    for t, batch_query_slot in enumerate(batch_query_slots):
        if t >= z_pred.shape[1] or batch_query_slot >= z_pred.shape[2]:
            continue
        assert int(batch["state_tokens"][t, t, batch_query_slot].item()) == int(vid)
        assert float(batch["state_mask_labels"][t, t, batch_query_slot, 0].item()) == 1.0
        assert bool(batch["state_pad_mask"][t, t, batch_query_slot].item()) is False
        z_q = z_pred[t, t, batch_query_slot].unsqueeze(0).to(device)
        try:
            out = probe(z_q, vid_tensor)
        except Exception as e:
            raise RuntimeError("Phase-1 target-manifold readout call failed in dense prediction.") from e
        mus[t] = float(out["numeric_mean"][0].item())
        sigmas[t] = float(out["numeric_sigma"][0].item())

    # True observations
    obs_t_list, obs_v_list = [], []
    for t in range(T):
        tokens_t = item["state"][t]["tokens"]
        hit = (tokens_t == vid).nonzero(as_tuple=True)[0]
        if hit.numel() > 0:
            slot = int(hit[0].item())
            obs_t_list.append(float(item["state_times"][t, 0].item()))
            obs_v_list.append(float(item["state"][t]["values"][slot].item()))

    return (abs_times, mus, sigmas,
            np.array(obs_t_list), np.array(obs_v_list))


def _imp__plot_imputation(
    abs_times: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    obs_abs_times: np.ndarray,
    obs_values: np.ndarray,
    feature_label: str,
    hadm_id: str,
    output_path: str,
) -> None:
    """
    Plot dense exploratory trace for one patient.

    Blue line     — model prediction (mu) at every timestep
    Gray band     — mu ± sigma (predictive uncertainty)
    Black dots    — true observed values
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Dense exploratory plots intentionally omit carry-forward baselines to focus
    # on model predictions versus observed values.

    # Uncertainty band (light gray)
    ax.fill_between(abs_times, mus - sigmas, mus + sigmas,
                    color="lightgray", alpha=0.6, label="Pred ±σ")

    # Model prediction (blue)
    ax.plot(abs_times, mus, color="steelblue", linewidth=1.5, label="Model prediction (μ)")

    # True observations (black dots)
    if obs_abs_times.size > 0:
        ax.scatter(obs_abs_times, obs_values, color="black", s=30, zorder=5,
                   label="True observations")

    ax.set_xlabel("Absolute time (min)")
    ax.set_ylabel(f"{feature_label}")
    ax.set_title(f"Dense exploratory query trace — {feature_label}  |  hadm_id={hadm_id}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[imputation] plot → {output_path}")

def _imp_select_representative_patient(
    df_vid: pd.DataFrame,
    *,
    min_plot_observations: int = 5,
) -> Dict[str, Any]:
    x = df_vid.copy()
    x["sq_err"] = (x["pred_mu"] - x["true_value"]) ** 2
    grp = (
        x.groupby("hadm_id", as_index=False)
        .agg(
            n_observations=("sq_err", "count"),
            patient_mse=("sq_err", "mean"),
        )
    )
    cand = grp[grp["n_observations"] >= int(min_plot_observations)].copy()
    used_min = int(min_plot_observations)
    if cand.empty:
        used_min = 3
        cand = grp[grp["n_observations"] >= 3].copy()
    if cand.empty:
        cand = grp.copy()
    med = float(cand["patient_mse"].median()) if not cand.empty else float("nan")
    cand["dist_to_median"] = np.abs(cand["patient_mse"] - med)
    cand = cand.sort_values(
        ["dist_to_median", "n_observations", "hadm_id"],
        ascending=[True, False, True],
    )
    sel = cand.iloc[0]
    return {
        "selected_hadm_id": str(sel["hadm_id"]),
        "n_observations": int(sel["n_observations"]),
        "patient_mse": float(sel["patient_mse"]),
        "median_candidate_mse": med,
        "candidate_count": int(len(cand)),
        "selection_rule": f"closest_to_median_mse_with_min_obs_{used_min}_tie_break_nobs_then_hadm",
    }

def _imp_compute_observed_only_locf(values: np.ndarray) -> np.ndarray:
    locf = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) <= 1:
        return locf
    locf[1:] = values[:-1]
    return locf

def _imp__compute_locf_per_hadm(df: pd.DataFrame, value_col: str = "true_value") -> np.ndarray:
    """Compute LOCF baseline using only previous observed target values per hadm."""
    locf = np.full((len(df),), np.nan, dtype=np.float64)
    group_cols = ["hadm_id"]
    if "vid" in df.columns:
        group_cols = ["vid", "hadm_id"]
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        order = np.array(list(idx), dtype=np.int64)
        d = df.iloc[order].sort_values("abs_time_min")
        vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(dtype=np.float64)
        l = _imp_compute_observed_only_locf(vals)
        locf[d.index.to_numpy(dtype=np.int64)] = l
    return locf

def _imp__merge_locf_with_fallback(locf: np.ndarray, fallback: Union[float, np.ndarray]) -> np.ndarray:
    locf = np.asarray(locf, dtype=np.float64)
    out = locf.copy()
    if np.isscalar(fallback):
        out[~np.isfinite(out)] = float(fallback)
        return out
    fb = np.asarray(fallback, dtype=np.float64)
    if fb.shape != out.shape:
        raise ValueError("Fallback array shape does not match LOCF shape.")
    out[~np.isfinite(out)] = fb[~np.isfinite(out)]
    return out

def _imp__regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sigma: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        out = {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
        }
        if sigma is not None:
            out.update({
                "nll_gaussian": float("nan"),
                "mean_sigma": float("nan"),
                "coverage_68": float("nan"),
                "coverage_95": float("nan"),
            })
        return out
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0.0 else float("nan")
    if yt.size > 1:
        yt_std = float(np.std(yt))
        yp_std = float(np.std(yp))
        corr = float(np.corrcoef(yt, yp)[0, 1]) if yt_std > 0.0 and yp_std > 0.0 else float("nan")
    else:
        corr = float("nan")
    out = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": corr,
    }
    if sigma is not None:
        sig = np.asarray(sigma, dtype=np.float64)[mask]
        sig = np.clip(sig, 1.0e-6, None)
        out.update({
            "nll_gaussian": float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + err ** 2 / (2 * sig ** 2))),
            "mean_sigma": float(np.mean(sig)),
            "coverage_68": float(np.mean(np.abs(err) <= sig)),
            "coverage_95": float(np.mean(np.abs(err) <= 2.0 * sig)),
        })
    return out

def _imp__validate_standardized_imputation_df(df: pd.DataFrame) -> None:
    required = [
        "metric_space",
        "true_value",
        "pred_mu",
        "pred_sigma",
        "true_value_std",
        "pred_mu_std",
        "pred_sigma_std",
        "true_value_raw",
        "pred_mu_raw",
        "pred_sigma_raw",
        "strict_locf_pred",
        "strict_locf_pred_raw",
        "locf_pred",
        "item_mean_pred",
        "last_value_carry_forward_pred",
        "variable_mean_pred",
        "locf_pred_raw",
        "item_mean_pred_raw",
        "last_value_carry_forward_pred_raw",
        "variable_mean_pred_raw",
        "residual_model",
        "residual_strict_locf",
        "residual_locf",
        "residual_variable_mean",
        "residual_model_raw",
        "residual_strict_locf_raw",
        "residual_locf_raw",
        "residual_variable_mean_raw",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Imputation outputs are missing required standardized/raw columns: "
            + ", ".join(missing)
        )
    if not df["metric_space"].astype(str).str.lower().eq("standardized").all():
        raise RuntimeError("Imputation outputs must be standardized-space only for eval integration.")
    if not np.allclose(
        pd.to_numeric(df["true_value"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["true_value_std"], errors="coerce").to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("Unsuffixed true_value must equal true_value_std for paper-facing outputs.")
    if not np.allclose(
        pd.to_numeric(df["pred_mu"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["pred_mu_std"], errors="coerce").to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("Unsuffixed pred_mu must equal pred_mu_std for paper-facing outputs.")
    if not np.allclose(
        pd.to_numeric(df["pred_sigma"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["pred_sigma_std"], errors="coerce").to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("Unsuffixed pred_sigma must equal pred_sigma_std for paper-facing outputs.")
    if not np.allclose(
        pd.to_numeric(df["locf_pred"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["last_value_carry_forward_pred"], errors="coerce").to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("locf_pred must equal last_value_carry_forward_pred for paper-facing outputs.")
    if not np.allclose(pd.to_numeric(df["variable_mean_pred"], errors="coerce").to_numpy(dtype=np.float64), 0.0, equal_nan=True):
        raise RuntimeError("Standardized variable_mean_pred must be 0.0 for all rows.")

def _imp__safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")

def _imp__metric_from_report(
    per_feature_df: Optional[pd.DataFrame],
    vid: int,
    col: str,
) -> float:
    if per_feature_df is None or col not in per_feature_df.columns:
        return float("nan")
    cur = per_feature_df[per_feature_df["feature_id"].astype(int) == int(vid)]
    if cur.empty:
        return float("nan")
    return _imp__safe_float(cur.iloc[0][col])


def _imp__json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _imp__json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_imp__json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _imp__write_json_atomic(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_imp__json_safe(dict(payload)), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def _imp__atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _imp__atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _imp__stable_signature(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_imp__json_safe(dict(payload)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _imp__as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _imp__schema_maps(schema_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    schema_df = pd.read_csv(schema_path)
    if "vid" not in schema_df.columns:
        raise RuntimeError(f"schema file missing vid column: {schema_path}")
    label_col = "normalized_concept_label" if "normalized_concept_label" in schema_df.columns else None
    type_col = "feature_type" if "feature_type" in schema_df.columns else "model_type" if "model_type" in schema_df.columns else None
    labels: Dict[int, str] = {}
    types: Dict[int, str] = {}
    for _, row in schema_df.iterrows():
        if pd.isna(row.get("vid")):
            continue
        vid = int(row["vid"])
        labels[vid] = str(row[label_col]) if label_col and pd.notna(row.get(label_col)) else f"VID {vid}"
        types[vid] = str(row[type_col]) if type_col and pd.notna(row.get(type_col)) else ""
    return labels, types


def _imp__load_support_summary(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"train_support_summary.csv not found: {path}")
    df = pd.read_csv(path)
    if "feature_id" not in df.columns:
        if "vid" in df.columns:
            df = df.rename(columns={"vid": "feature_id"})
        else:
            raise RuntimeError("train_support_summary.csv must include feature_id")
    if "count" not in df.columns:
        raise RuntimeError("train_support_summary.csv must include count")
    df["feature_id"] = pd.to_numeric(df["feature_id"], errors="raise").astype(int)
    return df


def _imp__extract_readout_count(metrics: Mapping[str, Any]) -> Optional[int]:
    for key in ("numeric_mean", "target_manifold_inverse"):
        value = metrics.get(key)
        if isinstance(value, Mapping) and value.get("count") is not None:
            return int(value["count"])
    return None


def _imp__load_and_validate_readout_predictions(
    readout_dir: str,
    *,
    expected_metrics: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = os.path.join(readout_dir, "readout_predictions_test.parquet")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"readout_predictions_test.parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {
        "hadm_id",
        "t_idx",
        "slot_idx",
        "feature_id",
        "query_key",
        "target_value",
        "numeric_mean",
        "locf_available",
        "strict_locf_prediction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"readout prediction parquet missing columns: {missing}")
    df = df.copy()
    df["hadm_id"] = df["hadm_id"].astype(str)
    for col in ("t_idx", "slot_idx", "feature_id"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    duplicate = df.duplicated("query_key", keep=False)
    if bool(duplicate.any()):
        bad = str(df.loc[duplicate, "query_key"].iloc[0])
        raise RuntimeError(f"duplicate query key in readout predictions: {bad}")
    expected_key = df.apply(
        lambda r: str((str(r["hadm_id"]), int(r["t_idx"]), int(r["slot_idx"]), int(r["feature_id"]))),
        axis=1,
    )
    if not expected_key.astype(str).equals(df["query_key"].astype(str)):
        raise RuntimeError("readout prediction parquet query_key does not match identity columns")
    for col in ("target_value", "numeric_mean"):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"readout prediction parquet contains non-finite {col}")
    count = _imp__extract_readout_count(expected_metrics)
    if count is not None and int(count) != int(len(df)):
        raise RuntimeError(f"readout row count mismatch: parquet={len(df)} metrics={count}")
    audit = {"row_count": int(len(df)), "unique_query_keys": int(df["query_key"].nunique())}
    metric_mse = None
    if isinstance(expected_metrics.get("numeric_mean"), Mapping):
        metric_mse = expected_metrics["numeric_mean"].get("mse")
    if metric_mse is not None:
        mse = float(np.mean((df["numeric_mean"].to_numpy(dtype=np.float64) - df["target_value"].to_numpy(dtype=np.float64)) ** 2))
        if not np.isclose(mse, float(metric_mse), rtol=1e-5, atol=1e-8):
            raise RuntimeError(f"readout parquet/metrics parity failed for numeric_mean mse: parquet={mse} metrics={metric_mse}")
        audit["numeric_mean_mse"] = mse
        audit["metrics_numeric_mean_mse"] = float(metric_mse)
    return df, audit


def _imp__clustered_bootstrap_delta(
    df: pd.DataFrame,
    *,
    better_pred_col: str,
    baseline_pred_col: str,
    target_col: str = "target_value",
    admission_col: str = "hadm_id",
    iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    required = {better_pred_col, baseline_pred_col, target_col, admission_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"bootstrap input missing columns: {sorted(missing)}")
    work = df[list(required)].copy()
    for col in (better_pred_col, baseline_pred_col, target_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[np.isfinite(work[better_pred_col]) & np.isfinite(work[baseline_pred_col]) & np.isfinite(work[target_col])]
    if work.empty:
        return {
            "status": "not_evaluable",
            "delta_mse": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "paired_admission_count": 0,
            "paired_query_count": 0,
            "iterations": int(iterations),
            "seed": int(seed),
        }
    y = work[target_col].to_numpy(dtype=np.float64)
    atlas_sq = (work[better_pred_col].to_numpy(dtype=np.float64) - y) ** 2
    base_sq = (work[baseline_pred_col].to_numpy(dtype=np.float64) - y) ** 2
    work = work.assign(_delta=base_sq - atlas_sq, _n=1)
    grouped = work.groupby(admission_col, sort=True).agg(delta_sum=("_delta", "sum"), n=("_n", "sum")).reset_index()
    ids = grouped[admission_col].astype(str).tolist()
    delta_sum = grouped["delta_sum"].to_numpy(dtype=np.float64)
    counts = grouped["n"].to_numpy(dtype=np.float64)
    point = float(delta_sum.sum() / counts.sum()) if counts.sum() > 0 else float("nan")
    if iterations <= 0 or len(ids) == 0:
        return {
            "status": "ok",
            "delta_mse": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "paired_admission_count": int(len(ids)),
            "paired_query_count": int(counts.sum()),
            "iterations": int(iterations),
            "seed": int(seed),
        }
    rng = np.random.default_rng(int(seed))
    estimates = np.empty((int(iterations),), dtype=np.float64)
    for i in range(int(iterations)):
        draw = rng.integers(0, len(ids), size=len(ids))
        den = float(counts[draw].sum())
        estimates[i] = float(delta_sum[draw].sum() / den) if den > 0 else np.nan
    estimates = estimates[np.isfinite(estimates)]
    return {
        "status": "ok",
        "delta_mse": point,
        "ci_low": float(np.quantile(estimates, 0.025)) if estimates.size else float("nan"),
        "ci_high": float(np.quantile(estimates, 0.975)) if estimates.size else float("nan"),
        "paired_admission_count": int(len(ids)),
        "paired_query_count": int(counts.sum()),
        "iterations": int(iterations),
        "valid_replicates": int(estimates.size),
        "seed": int(seed),
    }


def _imp__admission_signflip_p_value(
    df: pd.DataFrame,
    *,
    better_pred_col: str,
    baseline_pred_col: str,
    target_col: str = "target_value",
    admission_col: str = "hadm_id",
    iterations: int = 1000,
    seed: int = 42,
) -> float:
    work = df[[admission_col, better_pred_col, baseline_pred_col, target_col]].copy()
    for col in (better_pred_col, baseline_pred_col, target_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[np.isfinite(work[better_pred_col]) & np.isfinite(work[baseline_pred_col]) & np.isfinite(work[target_col])]
    if work.empty:
        return float("nan")
    y = work[target_col].to_numpy(dtype=np.float64)
    delta = (work[baseline_pred_col].to_numpy(dtype=np.float64) - y) ** 2 - (work[better_pred_col].to_numpy(dtype=np.float64) - y) ** 2
    grouped = work.assign(_delta=delta, _n=1).groupby(admission_col, sort=True).agg(delta_sum=("_delta", "sum"), n=("_n", "sum"))
    delta_sum = grouped["delta_sum"].to_numpy(dtype=np.float64)
    counts = grouped["n"].to_numpy(dtype=np.float64)
    den = float(counts.sum())
    if den <= 0:
        return float("nan")
    observed = abs(float(delta_sum.sum() / den))
    if iterations <= 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(iterations)):
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=delta_sum.shape[0], replace=True)
        stat = abs(float((signs * delta_sum).sum() / den))
        exceed += int(stat >= observed - 1.0e-15)
    return float((exceed + 1) / (int(iterations) + 1))


def _imp__bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = float(len(order))
    prev = 1.0
    for rank, original in enumerate(order[::-1], start=1):
        true_rank = len(order) - rank + 1
        val = min(prev, p[original] * m / float(true_rank))
        q[original] = val
        prev = val
    return q


def _imp__metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = _imp__regression_metrics(y_true, y_pred)
    return {"mse": m["mse"], "mae": m["mae"], "r2": m["r2"], "rmse": m["rmse"]}


def _imp__paired_locf_by_vid(
    predictions: pd.DataFrame,
    support_df: pd.DataFrame,
    labels: Mapping[int, str],
    types: Mapping[int, str],
    *,
    min_queries: int,
    min_admissions: int,
    bootstrap_iters: int,
    permutation_iters: int,
    seed: int,
) -> pd.DataFrame:
    work = predictions[_imp__as_bool_series(predictions["locf_available"])].copy()
    work["strict_locf_prediction"] = pd.to_numeric(work["strict_locf_prediction"], errors="coerce")
    work = work[np.isfinite(work["strict_locf_prediction"])].copy()
    support = support_df.set_index("feature_id").to_dict(orient="index")
    rows: List[Dict[str, Any]] = []
    for vid, group in work.groupby("feature_id", sort=True):
        v = int(vid)
        y = group["target_value"].to_numpy(dtype=np.float64)
        atlas = group["numeric_mean"].to_numpy(dtype=np.float64)
        locf = group["strict_locf_prediction"].to_numpy(dtype=np.float64)
        n_adm = int(group["hadm_id"].nunique())
        atlas_m = _imp__metric_row(y, atlas)
        locf_m = _imp__metric_row(y, locf)
        boot = _imp__clustered_bootstrap_delta(
            group,
            better_pred_col="numeric_mean",
            baseline_pred_col="strict_locf_prediction",
            iterations=bootstrap_iters,
            seed=seed + v,
        )
        p_value = _imp__admission_signflip_p_value(
            group,
            better_pred_col="numeric_mean",
            baseline_pred_col="strict_locf_prediction",
            iterations=permutation_iters,
            seed=seed + 50000 + v,
        )
        support_count = float(support.get(v, {}).get("count", np.nan))
        rows.append({
            "vid": v,
            "feature_id": v,
            "feature_label": labels.get(v, f"VID {v}"),
            "feature_type": types.get(v, ""),
            "paired_query_count": int(len(group)),
            "unique_admission_count": n_adm,
            "train_support_count": support_count,
            "atlas_mse": atlas_m["mse"],
            "atlas_mae": atlas_m["mae"],
            "atlas_r2": atlas_m["r2"],
            "strict_locf_mse": locf_m["mse"],
            "strict_locf_mae": locf_m["mae"],
            "strict_locf_r2": locf_m["r2"],
            "delta_mse": float(locf_m["mse"] - atlas_m["mse"]) if np.isfinite(locf_m["mse"]) and np.isfinite(atlas_m["mse"]) else float("nan"),
            "delta_mse_ci_low": boot["ci_low"],
            "delta_mse_ci_high": boot["ci_high"],
            "p_value": p_value,
            "bootstrap_iterations": int(bootstrap_iters),
            "permutation_iterations": int(permutation_iters),
            "bootstrap_seed": int(seed + v),
            "signflip_seed": int(seed + 50000 + v),
            "eligibility_status": "ok" if len(group) >= int(min_queries) and n_adm >= int(min_admissions) else "insufficient",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = np.nan
    eligible_mask = out["eligibility_status"].eq("ok").to_numpy()
    out.loc[eligible_mask, "q_value"] = _imp__bh_fdr(out.loc[eligible_mask, "p_value"].to_numpy(dtype=np.float64))
    verdicts: List[str] = []
    for _, row in out.iterrows():
        if row["eligibility_status"] != "ok":
            verdicts.append("insufficient")
        elif np.isfinite(float(row["q_value"])) and float(row["q_value"]) <= 0.05 and np.isfinite(float(row["delta_mse_ci_low"])) and float(row["delta_mse_ci_low"]) > 0.0:
            verdicts.append("atlas_better")
        elif np.isfinite(float(row["q_value"])) and float(row["q_value"]) <= 0.05 and np.isfinite(float(row["delta_mse_ci_high"])) and float(row["delta_mse_ci_high"]) < 0.0:
            verdicts.append("locf_better")
        else:
            verdicts.append("inconclusive")
    out["significant_fdr_0.05"] = np.isfinite(out["q_value"].to_numpy(dtype=np.float64)) & (out["q_value"].to_numpy(dtype=np.float64) <= 0.05)
    out["verdict"] = verdicts
    return out


@dataclass
class _ImpTrajectoryIndex:
    time_by_hadm_t: Dict[Tuple[str, int], float]
    obs_by_hadm_vid: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    lag_stats: Dict[int, Dict[str, Dict[str, Dict[str, float]]]]
    parity_audit: Dict[str, Any]
    raw_event_count: int


def _imp__events_from_dataset_item(item: Mapping[str, Any]) -> Tuple[str, List[Tuple[int, int, int, float, float]]]:
    hadm_id = str(item.get("hadm_id"))
    times = torch.as_tensor(item["state_times"]).reshape(-1).detach().cpu().to(torch.float64).numpy()
    events: List[Tuple[int, int, int, float, float]] = []
    if item.get("state") is not None and len(item.get("state", [])) > 0:
        for t_idx, timestamp in enumerate(item["state"]):
            tokens = torch.as_tensor(timestamp["tokens"]).reshape(-1).detach().cpu()
            values = torch.as_tensor(timestamp["values"]).reshape(-1).detach().cpu()
            for slot_idx in range(int(tokens.numel())):
                vid = int(tokens[slot_idx].item())
                value = float(values[slot_idx].item())
                if vid > 0 and np.isfinite(value):
                    events.append((int(t_idx), int(slot_idx) + 1, vid, float(times[t_idx]), value))
        return hadm_id, events
    tokens = torch.as_tensor(item["state_tokens_flat"]).reshape(-1).detach().cpu()
    values = torch.as_tensor(item["state_values_flat"]).reshape(-1).detach().cpu()
    offsets = torch.as_tensor(item["state_offsets"]).reshape(-1).detach().cpu().to(torch.long).numpy()
    for t_idx in range(len(offsets) - 1):
        start, end = int(offsets[t_idx]), int(offsets[t_idx + 1])
        for slot_idx, pos in enumerate(range(start, end)):
            vid = int(tokens[pos].item())
            value = float(values[pos].item())
            if vid > 0 and np.isfinite(value):
                events.append((int(t_idx), int(slot_idx) + 1, vid, float(times[t_idx]), value))
    return hadm_id, events


def _imp__dataset_kwargs_for_formal_trajectories(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        raise ValueError("Formal imputation from --readout_dir requires --config to reproduce Phase-1 dataset policy.")
    cfg = load_config_dict(config_path)
    kwargs = _phase1_dataset_kwargs(cfg)
    kwargs.update({
        "mask_target_ratio": 0.0,
        "mask_rarity_strength": 0.0,
        "mask_min_prob": 0.0,
        "mask_max_prob": 0.0,
        "static_mask_prob": 0.0,
        "static_numeric_vids": [],
        "deterministic": True,
        "return_legacy_nested": True,
        "fraction": 1.0,
    })
    return kwargs


def _imp__readout_cache_contract_audit(
    *,
    readout_dir: str,
    data_dir: str,
    schema_dir: str,
    config_path: str,
    split: str,
) -> Dict[str, Any]:
    path = os.path.join(readout_dir, "readout_cache_audit.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"readout_cache_audit.json not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    split_audit = audit.get(split)
    if not isinstance(split_audit, Mapping):
        raise RuntimeError(f"readout_cache_audit.json missing split={split}")
    cfg = load_config_dict(config_path)
    ds_kwargs = _phase1_dataset_kwargs(cfg)
    masking = {k: ds_kwargs.get(k) for k in ("mask_target_ratio", "mask_rarity_strength", "mask_min_prob", "mask_max_prob", "static_mask_prob", "static_numeric_vids")}
    policy = {k: ds_kwargs.get(k) for k in ("max_seq_len_limit", "max_set_size_limit", "value_limit", "phase", "target_task")}
    expected = {
        "schema_sha256": _sha256_file(os.path.join(schema_dir, "schema_processed.csv")),
        "dataset_content_signature": _dataset_content_signature(data_dir),
        "config_signature": _stable_json_sha256(cfg),
        "readout_masking_signature": _stable_json_sha256(masking),
        "dataset_policy_signature": _stable_json_sha256(policy),
    }
    blocking_keys = {
        "schema_sha256",
        "dataset_content_signature",
        "readout_masking_signature",
        "dataset_policy_signature",
    }
    mismatches = {}
    for key in sorted(blocking_keys):
        value = expected[key]
        found = split_audit.get(key)
        if found is None or str(found) != str(value):
            mismatches[key] = {"found": found, "expected": value}
    if mismatches:
        raise RuntimeError("readout cache audit provenance mismatch: " + json.dumps(mismatches, sort_keys=True))
    found_config_signature = split_audit.get("config_signature")
    return {
        "path": path,
        "sha256": _sha256_file(path),
        "split": split,
        "checked_fields": sorted(blocking_keys),
        "nonblocking_config_signature": {
            "found": found_config_signature,
            "expected": expected["config_signature"],
            "matches": found_config_signature is not None and str(found_config_signature) == str(expected["config_signature"]),
        },
        "dataset_policy": policy,
        "masking": masking,
    }


def _imp__build_trajectory_index(
    predictions: pd.DataFrame,
    *,
    data_dir: str,
    schema_dir: str,
    split: str,
    config_path: Optional[str],
    value_tolerance: float,
) -> _ImpTrajectoryIndex:
    needed_hadm = set(predictions["hadm_id"].astype(str).unique().tolist())
    needed_vid_by_hadm: Dict[str, set] = defaultdict(set)
    query_groups: Dict[str, pd.DataFrame] = {}
    for hadm, group in predictions.groupby("hadm_id", sort=False):
        h = str(hadm)
        query_groups[h] = group
        needed_vid_by_hadm[h].update(int(v) for v in group["feature_id"].unique().tolist())
    ds_kwargs = _imp__dataset_kwargs_for_formal_trajectories(config_path)
    dataset = PatientSequenceDataset(
        data_dir=data_dir,
        schema_dir=schema_dir,
        split=split,
        target_ids=sorted(needed_hadm),
        **ds_kwargs,
    )
    time_by_hadm_t: Dict[Tuple[str, int], float] = {}
    obs_lists: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = defaultdict(list)
    lag_stats: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: {
            "lt6h": {},
            "6to24h": {},
            "gt24h": {},
        }
    )
    failures: List[Dict[str, Any]] = []
    checked = 0
    raw_event_count = 0
    seen_hadm: set = set()
    for i in tqdm(range(len(dataset)), desc="imputation/formal-trajectory-index", ascii=True, mininterval=5.0):
        hadm_id, events = _imp__events_from_dataset_item(dataset[i])
        seen_hadm.add(hadm_id)
        raw_event_count += len(events)
        by_t_slot = {(t, s): (v, value, tm) for t, s, v, tm, value in events}
        for t, _, _, tm, _ in events:
            time_by_hadm_t[(hadm_id, int(t))] = float(tm)
        for row in query_groups.get(hadm_id, pd.DataFrame()).itertuples(index=False):
            key = str(getattr(row, "query_key"))
            t_idx = int(getattr(row, "t_idx"))
            slot_idx = int(getattr(row, "slot_idx"))
            expected_vid = int(getattr(row, "feature_id"))
            expected_value = float(getattr(row, "target_value"))
            actual = by_t_slot.get((t_idx, slot_idx))
            checked += 1
            if actual is None:
                failures.append({"query_key": key, "reason": "missing_t_idx_slot"})
                continue
            actual_vid, actual_value, _ = actual
            if int(actual_vid) != expected_vid or abs(float(actual_value) - expected_value) > float(value_tolerance):
                failures.append({
                    "query_key": key,
                    "reason": "identity_or_value_mismatch",
                    "expected_vid": expected_vid,
                    "actual_vid": int(actual_vid),
                    "expected_value": expected_value,
                    "actual_value": float(actual_value),
                })
        wanted = needed_vid_by_hadm.get(hadm_id, set())
        events_by_vid: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        for t_idx, slot_idx, vid, tm, value in events:
            if int(vid) in wanted:
                obs_lists[(hadm_id, int(vid))].append((int(t_idx), float(tm), float(value)))
            events_by_vid[int(vid)].append((int(t_idx), float(tm), float(value)))
        readout_vids = set().union(*needed_vid_by_hadm.values()) if needed_vid_by_hadm else set()
        for vid, vid_events in events_by_vid.items():
            if int(vid) not in readout_vids:
                continue
            if len(vid_events) < 2:
                continue
            arr = sorted(vid_events, key=lambda x: (x[1], x[0]))
            times = np.asarray([x[1] for x in arr], dtype=np.float64)
            pairs_by_bin: Dict[str, Tuple[List[float], List[float]]] = {
                "lt6h": ([], []),
                "6to24h": ([], []),
                "gt24h": ([], []),
            }
            for j in range(len(arr) - 1):
                dt = float(times[j + 1] - times[j])
                if dt <= 0:
                    continue
                bin_name = "lt6h" if dt <= 6.0 * 60.0 else "6to24h" if dt <= 24.0 * 60.0 else "gt24h"
                pairs_by_bin[bin_name][0].append(float(arr[j][2]))
                pairs_by_bin[bin_name][1].append(float(arr[j + 1][2]))
            for bin_name, (prev_values, next_values) in pairs_by_bin.items():
                if len(prev_values) < 2:
                    continue
                x = np.asarray(prev_values, dtype=np.float64)
                y = np.asarray(next_values, dtype=np.float64)
                x = x - float(x.mean())
                y = y - float(y.mean())
                lag_stats[int(vid)][bin_name][hadm_id] = {
                    "n": float(x.size),
                    "sum_x2": float(np.dot(x, x)),
                    "sum_y2": float(np.dot(y, y)),
                    "sum_xy": float(np.dot(x, y)),
                }
    missing_hadm = sorted(needed_hadm - seen_hadm)
    if missing_hadm:
        failures.append({"reason": "missing_hadm", "count": len(missing_hadm), "examples": missing_hadm[:10]})
    if failures:
        audit = {
            "checked_query_count": int(checked),
            "failure_count": int(len(failures)),
            "failures": failures[:20],
            "value_tolerance": float(value_tolerance),
            "dataset_policy": ds_kwargs,
        }
        raise RuntimeError("readout query/raw trajectory parity failed: " + json.dumps(_imp__json_safe(audit), sort_keys=True))
    obs_index: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, rows in obs_lists.items():
        ordered = sorted(rows, key=lambda x: (x[0], x[1]))
        obs_index[key] = (
            np.asarray([x[0] for x in ordered], dtype=np.int64),
            np.asarray([x[1] for x in ordered], dtype=np.float64),
            np.asarray([x[2] for x in ordered], dtype=np.float64),
        )
    return _ImpTrajectoryIndex(
        time_by_hadm_t=time_by_hadm_t,
        obs_by_hadm_vid=obs_index,
        lag_stats={int(vid): {str(bin_name): dict(by_hadm) for bin_name, by_hadm in by_bin.items()} for vid, by_bin in lag_stats.items()},
        parity_audit={
            "checked_query_count": int(checked),
            "failure_count": 0,
            "value_tolerance": float(value_tolerance),
            "raw_event_count": int(raw_event_count),
            "dataset_policy": ds_kwargs,
            "needed_hadm_count": int(len(needed_hadm)),
            "seen_hadm_count": int(len(seen_hadm)),
        },
        raw_event_count=int(raw_event_count),
    )




def _imp__lag1_persistence(
    trajectory: _ImpTrajectoryIndex,
    labels: Mapping[int, str],
    *,
    min_pairs: int,
    min_admissions: int,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    def _corr_from_stat_rows(stat_rows: Sequence[Mapping[str, float]]) -> Tuple[float, int]:
        n = int(sum(float(s.get("n", 0.0)) for s in stat_rows))
        sum_x2 = float(sum(float(s.get("sum_x2", 0.0)) for s in stat_rows))
        sum_y2 = float(sum(float(s.get("sum_y2", 0.0)) for s in stat_rows))
        sum_xy = float(sum(float(s.get("sum_xy", 0.0)) for s in stat_rows))
        denom = math.sqrt(sum_x2 * sum_y2) if sum_x2 > 0.0 and sum_y2 > 0.0 else 0.0
        if n < 2 or denom <= 0.0:
            return float("nan"), n
        return float(sum_xy / denom), n

    def _quantile_sorted(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return float("nan")
        ordered = np.sort(values.astype(np.float64, copy=False))
        pos = float(q) * float(ordered.size - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        weight = pos - float(lo)
        return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)

    rows: List[Dict[str, Any]] = []
    bins = ("lt6h", "6to24h", "gt24h")
    for vid, by_bin in sorted(trajectory.lag_stats.items()):
        row: Dict[str, Any] = {"vid": int(vid), "feature_id": int(vid), "feature_label": labels.get(int(vid), f"VID {int(vid)}")}
        for name in bins:
            by_hadm = {h: s for h, s in by_bin.get(name, {}).items() if int(float(s.get("n", 0.0))) >= 2}
            n_adm = len(by_hadm)
            corr, n_pairs = _corr_from_stat_rows(list(by_hadm.values()))
            if n_pairs >= int(min_pairs) and n_adm >= int(min_admissions):
                status = "ok"
            else:
                corr = float("nan")
                status = "insufficient"
            boot = np.empty((0,), dtype=np.float64)
            if status == "ok" and int(bootstrap_iters) > 0:
                hadms = sorted(by_hadm)
                rng = np.random.default_rng(int(seed) + int(vid) * 17 + len(name))
                vals: List[float] = []
                for _ in range(int(bootstrap_iters)):
                    draw = rng.choice(hadms, size=len(hadms), replace=True)
                    c, _ = _corr_from_stat_rows([by_hadm[h] for h in draw])
                    if np.isfinite(c):
                        vals.append(c)
                boot = np.asarray(vals, dtype=np.float64)
            row[f"lag1_persistence_{name}"] = corr
            row[f"lag1_ci_low_{name}"] = _quantile_sorted(boot, 0.025)
            row[f"lag1_ci_high_{name}"] = _quantile_sorted(boot, 0.975)
            row[f"lag1_pairs_{name}"] = n_pairs
            row[f"lag1_admissions_{name}"] = int(n_adm)
            row[f"lag1_status_{name}"] = status
        row["primary_persistence"] = row["lag1_persistence_lt6h"]
        row["primary_persistence_status"] = row["lag1_status_lt6h"]
        rows.append(row)
    return pd.DataFrame(rows)


def _imp__attach_interpolation(predictions: pd.DataFrame, trajectory: _ImpTrajectoryIndex) -> pd.DataFrame:
    cols = list(predictions.columns) + ["linear_interpolation_prediction", "interpolation_available"]
    preds: List[float] = []
    avail: List[bool] = []
    for _, row in predictions.iterrows():
        key = (str(row["hadm_id"]), int(row["feature_id"]))
        obs = trajectory.obs_by_hadm_vid.get(key)
        if obs is None:
            preds.append(float("nan"))
            avail.append(False)
            continue
        t_idx = int(row["t_idx"])
        t_indices, times, values = obs
        pos = int(np.searchsorted(t_indices, t_idx, side="left"))
        before_pos = pos - 1
        after_pos = int(np.searchsorted(t_indices, t_idx, side="right"))
        if before_pos < 0 or after_pos >= len(t_indices):
            preds.append(float("nan"))
            avail.append(False)
            continue
        t0 = float(times[before_pos])
        t1 = float(times[after_pos])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            preds.append(float("nan"))
            avail.append(False)
            continue
        tq = trajectory.time_by_hadm_t.get((key[0], t_idx))
        if tq is None:
            preds.append(float("nan"))
            avail.append(False)
            continue
        v0 = float(values[before_pos])
        v1 = float(values[after_pos])
        alpha = min(1.0, max(0.0, (tq - t0) / (t1 - t0)))
        preds.append(float(v0 + alpha * (v1 - v0)))
        avail.append(True)
    out = predictions.copy()
    out["linear_interpolation_prediction"] = preds
    out["interpolation_available"] = avail
    return out[cols]


def _imp__interpolation_sensitivity(
    predictions_with_interp: pd.DataFrame,
    labels: Mapping[int, str],
    types: Mapping[int, str],
    *,
    min_queries: int,
    min_admissions: int,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    work = predictions_with_interp[predictions_with_interp["interpolation_available"].astype(bool)].copy()
    rows: List[Dict[str, Any]] = []
    for vid, group in work.groupby("feature_id", sort=True):
        v = int(vid)
        y = group["target_value"].to_numpy(dtype=np.float64)
        atlas = group["numeric_mean"].to_numpy(dtype=np.float64)
        interp = group["linear_interpolation_prediction"].to_numpy(dtype=np.float64)
        n_adm = int(group["hadm_id"].nunique())
        atlas_m = _imp__metric_row(y, atlas)
        interp_m = _imp__metric_row(y, interp)
        boot = _imp__clustered_bootstrap_delta(
            group,
            better_pred_col="numeric_mean",
            baseline_pred_col="linear_interpolation_prediction",
            iterations=bootstrap_iters,
            seed=seed + 100000 + v,
        )
        if len(group) < int(min_queries) or n_adm < int(min_admissions):
            verdict = "insufficient"
        elif np.isfinite(boot["ci_low"]) and float(boot["ci_low"]) > 0.0:
            verdict = "atlas_better"
        elif np.isfinite(boot["ci_high"]) and float(boot["ci_high"]) < 0.0:
            verdict = "interpolation_better"
        else:
            verdict = "inconclusive"
        rows.append({
            "vid": v,
            "feature_id": v,
            "feature_label": labels.get(v, f"VID {v}"),
            "feature_type": types.get(v, ""),
            "paired_query_count": int(len(group)),
            "unique_admission_count": n_adm,
            "atlas_mse": atlas_m["mse"],
            "atlas_mae": atlas_m["mae"],
            "atlas_r2": atlas_m["r2"],
            "interpolation_mse": interp_m["mse"],
            "interpolation_mae": interp_m["mae"],
            "interpolation_r2": interp_m["r2"],
            "delta_mse": float(interp_m["mse"] - atlas_m["mse"]) if np.isfinite(interp_m["mse"]) and np.isfinite(atlas_m["mse"]) else float("nan"),
            "delta_mse_ci_low": boot["ci_low"],
            "delta_mse_ci_high": boot["ci_high"],
            "bootstrap_iterations": int(bootstrap_iters),
            "bootstrap_seed": int(seed + 100000 + v),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def _imp__quadrants(per_vid: pd.DataFrame, persistence: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if per_vid.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = per_vid.merge(
        persistence[["feature_id", "primary_persistence", "primary_persistence_status", "lag1_pairs_lt6h", "lag1_admissions_lt6h"]] if not persistence.empty else pd.DataFrame(columns=["feature_id", "primary_persistence", "primary_persistence_status", "lag1_pairs_lt6h", "lag1_admissions_lt6h"]),
        on="feature_id",
        how="left",
    )
    valid_p = merged[np.isfinite(pd.to_numeric(merged["primary_persistence"], errors="coerce"))]
    threshold = 0.5
    def label(row: pd.Series) -> str:
        p = row.get("primary_persistence")
        if p is None or not np.isfinite(float(p)):
            return "insufficient_persistence"
        if row.get("verdict") == "insufficient":
            return "insufficient"
        if row.get("verdict") == "inconclusive":
            return "inconclusive"
        if row.get("verdict") not in {"atlas_better", "locf_better"}:
            return "inconclusive"
        p_label = "high_persistence" if float(p) >= threshold else "low_persistence"
        win_label = "atlas_wins" if row.get("verdict") == "atlas_better" else "locf_wins"
        return f"{p_label}__{win_label}"
    merged["persistence_threshold"] = threshold
    merged["quadrant"] = merged.apply(label, axis=1)
    summary = (
        merged.groupby("quadrant", dropna=False)
        .agg(
            vid_count=("feature_id", "count"),
            total_paired_queries=("paired_query_count", "sum"),
            median_delta_mse=("delta_mse", "median"),
            mean_delta_mse=("delta_mse", "mean"),
        )
        .reset_index()
    )
    return merged, summary


def _imp__plot_quadrants(quadrants: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if quadrants.empty or "primary_persistence" not in quadrants.columns:
        ax.text(0.5, 0.5, "No evaluable imputation quadrant data", ha="center", va="center")
    else:
        q = quadrants[
            np.isfinite(pd.to_numeric(quadrants["primary_persistence"], errors="coerce"))
            & np.isfinite(pd.to_numeric(quadrants["delta_mse"], errors="coerce"))
            & quadrants["verdict"].isin(["atlas_better", "locf_better", "inconclusive"])
            & quadrants["eligibility_status"].eq("ok")
        ].copy()
        if q.empty:
            ax.text(0.5, 0.5, "No evaluable imputation quadrant data", ha="center", va="center")
        else:
            sc = None
            for marker, verdicts in (("o", ["atlas_better", "locf_better"]), ("x", ["inconclusive"])):
                part = q[q["verdict"].isin(verdicts)]
                if part.empty:
                    continue
                sizes = np.clip(np.sqrt(part["paired_query_count"].to_numpy(dtype=np.float64)), 10, 120)
                colors = pd.to_numeric(part["train_support_count"], errors="coerce").to_numpy(dtype=np.float64)
                sc = ax.scatter(part["primary_persistence"], part["delta_mse"], s=sizes, c=colors, cmap="viridis", alpha=0.75, marker=marker, label="/".join(verdicts))
            ax.axhline(0.0, color="black", linewidth=0.8)
            if np.isfinite(q["persistence_threshold"].iloc[0]):
                ax.axvline(float(q["persistence_threshold"].iloc[0]), color="gray", linewidth=0.8, linestyle="--")
            if sc is not None:
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("Train support count")
            ax.legend(fontsize=8)
    ax.set_xlabel("Lag-1 persistence (<6h)")
    ax.set_ylabel("ΔMSE = strict LOCF MSE − ATLAS MSE")
    ax.set_title("ATLAS vs strict LOCF by temporal persistence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _imp__formal_signature(
    *,
    readout_dir: str,
    data_dir: str,
    schema_dir: str,
    split: str,
    config_path: Optional[str],
    bootstrap_iters: int,
    permutation_iters: int,
    seed: int,
    min_paired_queries: int,
    min_paired_admissions: int,
    min_persistence_pairs: int,
    min_persistence_admissions: int,
    query_value_tolerance: float,
) -> Dict[str, Any]:
    schema_path = os.path.join(schema_dir, "schema_processed.csv")
    files = {
        "readout_predictions_test_parquet": os.path.join(readout_dir, "readout_predictions_test.parquet"),
        "readout_test_metrics_json": os.path.join(readout_dir, "readout_test_metrics.json"),
        "train_support_summary_csv": os.path.join(readout_dir, "train_support_summary.csv"),
        "readout_cache_audit_json": os.path.join(readout_dir, "readout_cache_audit.json"),
        "schema_processed_csv": schema_path,
    }
    optional_files = {
        "readout_manifest_json": os.path.join(readout_dir, "readout_manifest.json"),
        "readout_artifact_pt": os.path.join(readout_dir, "readout_artifact.pt"),
    }
    if config_path:
        optional_files["config_yaml"] = config_path
    split_dir = os.path.join(data_dir, split)
    split_manifest = []
    if os.path.isdir(split_dir):
        for name in sorted(os.listdir(split_dir)):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(split_dir, name)
            st = os.stat(path)
            split_manifest.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    payload = {
        "protocol": "imputation_from_frozen_readout_outputs_v2",
        "readout_dir_realpath": os.path.realpath(readout_dir),
        "data_dir_realpath": os.path.realpath(data_dir),
        "schema_dir_realpath": os.path.realpath(schema_dir),
        "split": str(split),
        "bootstrap_iters": int(bootstrap_iters),
        "permutation_iters": int(permutation_iters),
        "seed": int(seed),
        "min_paired_queries": int(min_paired_queries),
        "min_paired_admissions": int(min_paired_admissions),
        "min_persistence_pairs": int(min_persistence_pairs),
        "min_persistence_admissions": int(min_persistence_admissions),
        "query_value_tolerance": float(query_value_tolerance),
        "persistence_bins": {"lt6h": [0, 360], "6to24h": [360, 1440], "gt24h": [1440, None]},
        "primary_persistence": "within_admission_centered_lag1_lt6h",
        "trajectory_split_manifest_sha256": hashlib.sha256("\n".join(split_manifest).encode("utf-8")).hexdigest(),
        "trajectory_split_file_count": len(split_manifest),
        "files": {name: {"path": os.path.realpath(path), "sha256": _sha256_file(path)} for name, path in files.items() if os.path.isfile(path)},
    }
    for name, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"required imputation input missing: {path}")
    for name, path in optional_files.items():
        if os.path.isfile(path):
            payload["files"][name] = {"path": os.path.realpath(path), "sha256": _sha256_file(path)}
    payload["signature_sha256"] = _imp__stable_signature(payload)
    return payload


def _imp__nested_values_for_key(value: Any, key: str) -> List[Any]:
    found: List[Any] = []
    if isinstance(value, Mapping):
        for k, v in value.items():
            if str(k) == key:
                found.append(v)
            found.extend(_imp__nested_values_for_key(v, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_imp__nested_values_for_key(item, key))
    return found


def _imp__readout_manifest_audit(readout_dir: str, schema_path: str) -> Dict[str, Any]:
    path = os.path.join(readout_dir, "readout_manifest.json")
    if not os.path.isfile(path):
        return {"manifest_present": False, "schema_sha256_checked": False}
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    schema_sha = _sha256_file(schema_path)
    manifest_schema_shas = [str(v) for v in _imp__nested_values_for_key(manifest, "schema_sha256") if v is not None]
    if manifest_schema_shas and schema_sha not in manifest_schema_shas:
        raise RuntimeError("readout manifest schema_sha256 does not match current schema")
    return {
        "manifest_present": True,
        "path": path,
        "sha256": _sha256_file(path),
        "schema_sha256_checked": bool(manifest_schema_shas),
        "schema_sha256": schema_sha,
        "phase1_checkpoint_sha256_values": sorted(set(str(v) for v in _imp__nested_values_for_key(manifest, "phase1_checkpoint_sha256") if v is not None)),
        "readout_contract_values": sorted(set(str(v) for v in _imp__nested_values_for_key(manifest, "readout_cache_contract_version") if v is not None)),
    }


def _imp_run_readout_output_imputation(
    *,
    readout_dir: str,
    data_dir: str,
    schema_dir: str,
    output_dir: str,
    split: str = "test",
    config_path: Optional[str] = None,
    bootstrap_iters: int = 1000,
    permutation_iters: int = 10000,
    seed: int = 42,
    min_paired_queries: int = 100,
    min_paired_admissions: int = 20,
    min_persistence_pairs: int = 20,
    min_persistence_admissions: int = 5,
    query_value_tolerance: float = 1.0e-5,
    resume: bool = False,
) -> Dict[str, Any]:
    if not config_path:
        raise ValueError("Formal imputation from readout outputs requires config_path/--config.")
    signature = _imp__formal_signature(
        readout_dir=readout_dir,
        data_dir=data_dir,
        schema_dir=schema_dir,
        split=split,
        config_path=config_path,
        bootstrap_iters=bootstrap_iters,
        permutation_iters=permutation_iters,
        seed=seed,
        min_paired_queries=min_paired_queries,
        min_paired_admissions=min_paired_admissions,
        min_persistence_pairs=min_persistence_pairs,
        min_persistence_admissions=min_persistence_admissions,
        query_value_tolerance=query_value_tolerance,
    )
    complete = os.path.join(output_dir, ".complete")
    summary_path = os.path.join(output_dir, "imputation_summary.json")
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if not (os.path.isfile(complete) and os.path.isfile(summary_path)):
            raise RuntimeError("Existing imputation output directory is non-empty without a valid .complete; use a fresh output_dir.")
        with open(summary_path, "r", encoding="utf-8") as handle:
            old_for_signature = json.load(handle)
        if old_for_signature.get("input_signature", {}).get("signature_sha256") != signature["signature_sha256"]:
            raise RuntimeError("Existing imputation output signature mismatch; use a fresh output_dir or remove stale outputs.")
        if not resume:
            raise RuntimeError("Existing imputation output already complete; pass --resume to reuse it or choose a fresh output_dir.")
    if resume and os.path.isfile(complete) and os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            old = json.load(handle)
        if old.get("input_signature", {}).get("signature_sha256") == signature["signature_sha256"]:
            return {"mode": "resume", "output_dir": output_dir, **old}
        raise RuntimeError("Existing imputation output signature mismatch; use a fresh output_dir or remove stale outputs.")

    parent_dir = os.path.dirname(os.path.abspath(output_dir)) or "."
    os.makedirs(parent_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".tmp_imputation_", dir=parent_dir)
    try:
        schema_path = os.path.join(schema_dir, "schema_processed.csv")
        labels, types = _imp__schema_maps(schema_path)
        manifest_audit = _imp__readout_manifest_audit(readout_dir, schema_path)
        cache_contract_audit = _imp__readout_cache_contract_audit(
            readout_dir=readout_dir,
            data_dir=data_dir,
            schema_dir=schema_dir,
            config_path=config_path,
            split=split,
        )
        support_df = _imp__load_support_summary(os.path.join(readout_dir, "train_support_summary.csv"))
        with open(os.path.join(readout_dir, "readout_test_metrics.json"), "r", encoding="utf-8") as handle:
            readout_metrics = json.load(handle)
        predictions, parquet_audit = _imp__load_and_validate_readout_predictions(readout_dir, expected_metrics=readout_metrics)
        per_vid = _imp__paired_locf_by_vid(
            predictions,
            support_df,
            labels,
            types,
            min_queries=min_paired_queries,
            min_admissions=min_paired_admissions,
            bootstrap_iters=bootstrap_iters,
            permutation_iters=permutation_iters,
            seed=seed,
        )
        trajectory = _imp__build_trajectory_index(
            predictions,
            data_dir=data_dir,
            schema_dir=schema_dir,
            split=split,
            config_path=config_path,
            value_tolerance=query_value_tolerance,
        )
        persistence = _imp__lag1_persistence(
            trajectory,
            labels,
            min_pairs=min_persistence_pairs,
            min_admissions=min_persistence_admissions,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        )
        predictions_interp = _imp__attach_interpolation(predictions, trajectory)
        interp = _imp__interpolation_sensitivity(
            predictions_interp,
            labels,
            types,
            min_queries=min_paired_queries,
            min_admissions=min_paired_admissions,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        )
        quadrants, quadrant_summary = _imp__quadrants(per_vid, persistence)
        _imp__atomic_write_csv(per_vid, os.path.join(staging, "imputation_per_vid.csv"))
        _imp__atomic_write_csv(persistence, os.path.join(staging, "imputation_persistence.csv"))
        _imp__atomic_write_csv(quadrants, os.path.join(staging, "imputation_quadrants.csv"))
        _imp__atomic_write_csv(quadrant_summary, os.path.join(staging, "imputation_quadrant_summary.csv"))
        _imp__atomic_write_csv(interp, os.path.join(staging, "imputation_interpolation_sensitivity.csv"))
        _imp__write_json_atomic(os.path.join(staging, "imputation_query_parity_audit.json"), trajectory.parity_audit)
        _imp__plot_quadrants(quadrants, os.path.join(staging, "imputation_quadrants.png"))
        summary = {
            "analysis_scope": "formal_readout_output_imputation_vs_locf",
            "phase1_forward_executed": False,
            "readout_forward_executed": False,
            "formal_inputs": {
                "readout_predictions_test": os.path.join(readout_dir, "readout_predictions_test.parquet"),
                "readout_test_metrics": os.path.join(readout_dir, "readout_test_metrics.json"),
                "train_support_summary": os.path.join(readout_dir, "train_support_summary.csv"),
                "schema": schema_path,
                "raw_test_trajectories": os.path.join(data_dir, split),
                "config": config_path,
            },
            "input_signature": signature,
            "readout_parquet_audit": parquet_audit,
            "readout_manifest_audit": manifest_audit,
            "readout_cache_contract_audit": cache_contract_audit,
            "n_prediction_rows": int(len(predictions)),
            "n_unique_query_keys": int(predictions["query_key"].nunique()),
            "n_vids_total": int(predictions["feature_id"].nunique()),
            "n_vids_paired_locf": int(len(per_vid)),
            "n_locf_available_queries": int(_imp__as_bool_series(predictions["locf_available"]).sum()),
            "n_vids_interpolation": int(len(interp)),
            "n_raw_events_loaded": int(trajectory.raw_event_count),
            "query_parity_audit": trajectory.parity_audit,
            "parity_failure_count": int(trajectory.parity_audit.get("failure_count", 0)),
            "min_paired_queries": int(min_paired_queries),
            "min_paired_admissions": int(min_paired_admissions),
            "min_persistence_pairs": int(min_persistence_pairs),
            "min_persistence_admissions": int(min_persistence_admissions),
            "bootstrap_iterations": int(bootstrap_iters),
            "permutation_iterations": int(permutation_iters),
            "seed": int(seed),
            "verdict_counts": per_vid["verdict"].value_counts(dropna=False).to_dict() if "verdict" in per_vid.columns else {},
            "fdr_significant_count": int(per_vid["significant_fdr_0.05"].sum()) if "significant_fdr_0.05" in per_vid.columns else 0,
            "locf_paired_query_count": int(per_vid["paired_query_count"].sum()) if "paired_query_count" in per_vid.columns else 0,
            "locf_paired_admission_count": int(predictions[_imp__as_bool_series(predictions["locf_available"])]["hadm_id"].nunique()),
            "interpolation_paired_query_count": int(interp["paired_query_count"].sum()) if "paired_query_count" in interp.columns else 0,
            "interpolation_paired_admission_count": int(predictions_interp[predictions_interp["interpolation_available"].astype(bool)]["hadm_id"].nunique()) if "interpolation_available" in predictions_interp.columns else 0,
            "outputs": {
                "imputation_per_vid": os.path.join(output_dir, "imputation_per_vid.csv"),
                "imputation_persistence": os.path.join(output_dir, "imputation_persistence.csv"),
                "imputation_quadrants": os.path.join(output_dir, "imputation_quadrants.csv"),
                "imputation_quadrant_summary": os.path.join(output_dir, "imputation_quadrant_summary.csv"),
                "imputation_interpolation_sensitivity": os.path.join(output_dir, "imputation_interpolation_sensitivity.csv"),
                "imputation_query_parity_audit": os.path.join(output_dir, "imputation_query_parity_audit.json"),
                "imputation_quadrants_png": os.path.join(output_dir, "imputation_quadrants.png"),
            },
        }
        _imp__write_json_atomic(os.path.join(staging, "imputation_summary.json"), summary)
        _imp__atomic_write_text(os.path.join(staging, ".complete"), "imputation_from_frozen_readout_outputs_v2\n")
        if os.path.isdir(output_dir):
            os.rmdir(output_dir)
        os.replace(staging, output_dir)
        staging = ""
        return summary
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)

def _imp__run_single_vid_inference(
    *,
    teacher: JEPAModel,
    probe: Any,
    dataset: PatientSequenceDataset,
    schema_df: pd.DataFrame,
    vid: int,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    device: torch.device,
    max_samples: Optional[int],
    occurrence_cache: Optional[str],
    precomputed_occurrences: Optional[List[Tuple[int, int, float, float, str, int]]] = None,
    readout_artifact: str,
    write_outputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, str]:
    row = schema_df[schema_df["vid"].astype(int) == int(vid)]
    if row.empty:
        raise ValueError(f"VID {vid} not found in schema_processed.csv.")
    std_i = float(row["std"].iloc[0])
    mean_i = float(row["mean"].iloc[0])
    feat_label = str(row["normalized_concept_label"].iloc[0])
    if not np.isfinite(std_i) or std_i <= 0.0:
        raise ValueError(f"Invalid std for vid={vid}: {std_i}")
    if not np.isfinite(mean_i):
        raise ValueError(f"Invalid mean for vid={vid}: {mean_i}")

    ft_table = teacher.feature_type_ids.cpu()
    vid_clamped = min(int(vid), ft_table.numel() - 1)
    vid_ftype = int(ft_table[vid_clamped].item())
    if vid_ftype != int(FEATURE_TYPE_NUMERIC):
        raise ValueError(f"VID {vid} is not numeric (feature_type_id={vid_ftype}).")

    if precomputed_occurrences is not None:
        occurrences = [tuple(r) for r in precomputed_occurrences]
    else:
        occ_cache_vid = None
        if occurrence_cache is not None and occurrence_cache != "":
            root, ext = os.path.splitext(str(occurrence_cache))
            occ_cache_vid = f"{root}_vid{int(vid)}{ext or '.json'}"
        occurrences = _imp__find_occurrences(
            dataset, int(vid), max_samples,
            cache_path=occ_cache_vid, num_workers=num_workers,
        )
    if not occurrences:
        return pd.DataFrame(), {"n_samples": 0, "n_occurrences": 0, "vid": int(vid)}, pd.DataFrame(), feat_label

    occ_ds = _imp__OccurrenceDataset(dataset, occurrences, int(vid))
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory and device.type == "cuda",
        "collate_fn": _imp__occurrence_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(occ_ds, **loader_kwargs)

    records: List[Dict[str, Any]] = []
    vid_tensor = torch.tensor([int(vid)], dtype=torch.long, device=device)

    pbar = tqdm(loader, desc=f"imputing vid={int(vid)}", unit="batch",
                total=len(loader), dynamic_ncols=True)
    for result in pbar:
        if result is None:
            continue
        batch, meta_batch = result
        with torch.no_grad():
            z_pred = _imp__phase1_forward(teacher, batch, device)
        for b, meta in enumerate(meta_batch):
            t_m = meta["t_mask"]
            q_s = meta["batch_query_slot"]
            if t_m >= z_pred.shape[1]:
                continue
            assert 0 <= int(q_s) < int(z_pred.shape[2])
            assert int(batch["state_tokens"][b, t_m, q_s].item()) == int(vid)
            assert float(batch["state_mask_labels"][b, t_m, q_s, 0].item()) == 1.0
            assert bool(batch["state_pad_mask"][b, t_m, q_s].item()) is False
            z_q = z_pred[b, t_m, q_s].unsqueeze(0)
            with torch.no_grad():
                out = probe(z_q, vid_tensor)
            mu = float(out["numeric_mean"][0].item())
            sigma = float(out["numeric_sigma"][0].item())
            true_std = float(meta["true_value"])
            true_raw = float(true_std * std_i + mean_i)
            pred_raw = float(mu * std_i + mean_i)
            sigma_raw = float(sigma * std_i)
            records.append({
                "hadm_id": meta["hadm_id"],
                "t_mask": t_m,
                "abs_time_min": meta["abs_time_min"],
                "time_hours": float(meta["abs_time_min"]) / 60.0,
                "vid": int(vid),
                "feature_name": feat_label,
                "metric_space": "standardized",
                "true_value_std": true_std,
                "pred_mu_std": mu,
                "pred_sigma_std": sigma,
                "true_value": true_std,
                "pred_mu": mu,
                "pred_sigma": sigma,
                "true_value_raw": true_raw,
                "pred_mu_raw": pred_raw,
                "pred_sigma_raw": sigma_raw,
                "n_visible_same_t": meta["n_visible_same_t"],
                "n_total_timesteps": meta["n_total_timesteps"],
            })
        pbar.set_postfix({"records": len(records)})

    if not records:
        return pd.DataFrame(), {"n_samples": 0, "n_occurrences": 0, "vid": int(vid)}, pd.DataFrame(), feat_label
    df = pd.DataFrame(records)
    df = df.sort_values(["hadm_id", "abs_time_min"]).reset_index(drop=True)
    df["item_mean_pred"] = 0.0
    df["variable_mean_pred"] = 0.0
    df["item_mean_pred_raw"] = float(mean_i)
    df["variable_mean_pred_raw"] = float(mean_i)
    df["strict_locf_pred"] = _imp__compute_locf_per_hadm(df, value_col="true_value")
    df["strict_locf_pred_raw"] = _imp__compute_locf_per_hadm(df, value_col="true_value_raw")
    df["locf_pred"] = _imp__merge_locf_with_fallback(df["strict_locf_pred"].to_numpy(dtype=np.float64), 0.0)
    df["last_value_carry_forward_pred"] = df["locf_pred"]
    df["locf_pred_raw"] = _imp__merge_locf_with_fallback(df["strict_locf_pred_raw"].to_numpy(dtype=np.float64), float(mean_i))
    df["last_value_carry_forward_pred_raw"] = df["locf_pred_raw"]
    df["residual_model"] = df["pred_mu"] - df["true_value"]
    df["residual_strict_locf"] = df["strict_locf_pred"] - df["true_value"]
    df["residual_locf"] = df["locf_pred"] - df["true_value"]
    df["residual_variable_mean"] = df["variable_mean_pred"] - df["true_value"]
    df["residual_model_raw"] = df["pred_mu_raw"] - df["true_value_raw"]
    df["residual_strict_locf_raw"] = df["strict_locf_pred_raw"] - df["true_value_raw"]
    df["residual_locf_raw"] = df["locf_pred_raw"] - df["true_value_raw"]
    df["residual_variable_mean_raw"] = df["variable_mean_pred_raw"] - df["true_value_raw"]
    df["metric_space"] = "standardized"

    std_metrics = _imp__regression_metrics(
        df["true_value"].to_numpy(dtype=np.float64),
        df["pred_mu"].to_numpy(dtype=np.float64),
        sigma=df["pred_sigma"].to_numpy(dtype=np.float64),
    )
    raw_metrics = _imp__regression_metrics(
        df["true_value_raw"].to_numpy(dtype=np.float64),
        df["pred_mu_raw"].to_numpy(dtype=np.float64),
        sigma=df["pred_sigma_raw"].to_numpy(dtype=np.float64),
    )
    strict_locf_valid = np.isfinite(df["strict_locf_pred"].to_numpy(dtype=np.float64))
    locf_valid = strict_locf_valid
    locf_raw_valid = np.isfinite(df["strict_locf_pred_raw"].to_numpy(dtype=np.float64))
    if np.any(locf_valid):
        locf_metrics = _imp__regression_metrics(
            df.loc[locf_valid, "true_value"].to_numpy(dtype=np.float64),
            df.loc[locf_valid, "strict_locf_pred"].to_numpy(dtype=np.float64),
        )
        atlas_on_locf_valid = _imp__regression_metrics(
            df.loc[locf_valid, "true_value"].to_numpy(dtype=np.float64),
            df.loc[locf_valid, "pred_mu"].to_numpy(dtype=np.float64),
        )
    else:
        locf_metrics = _imp__regression_metrics(np.array([]), np.array([]))
        atlas_on_locf_valid = _imp__regression_metrics(np.array([]), np.array([]))
    if np.any(locf_raw_valid):
        locf_raw_metrics = _imp__regression_metrics(
            df.loc[locf_raw_valid, "true_value_raw"].to_numpy(dtype=np.float64),
            df.loc[locf_raw_valid, "strict_locf_pred_raw"].to_numpy(dtype=np.float64),
        )
        atlas_on_locf_valid_raw = _imp__regression_metrics(
            df.loc[locf_raw_valid, "true_value_raw"].to_numpy(dtype=np.float64),
            df.loc[locf_raw_valid, "pred_mu_raw"].to_numpy(dtype=np.float64),
        )
    else:
        locf_raw_metrics = _imp__regression_metrics(np.array([]), np.array([]))
        atlas_on_locf_valid_raw = _imp__regression_metrics(np.array([]), np.array([]))
    mean_metrics = _imp__regression_metrics(
        df["true_value"].to_numpy(dtype=np.float64),
        df["variable_mean_pred"].to_numpy(dtype=np.float64),
    )
    mean_raw_metrics = _imp__regression_metrics(
        df["true_value_raw"].to_numpy(dtype=np.float64),
        df["variable_mean_pred_raw"].to_numpy(dtype=np.float64),
    )
    merged_locf_metrics = _imp__regression_metrics(
        df["true_value"].to_numpy(dtype=np.float64),
        df["locf_pred"].to_numpy(dtype=np.float64),
    )
    merged_locf_raw_metrics = _imp__regression_metrics(
        df["true_value_raw"].to_numpy(dtype=np.float64),
        df["locf_pred_raw"].to_numpy(dtype=np.float64),
    )
    valid_locf_frac = float(np.mean(locf_valid)) if len(df) else float("nan")
    valid_locf_count = int(np.sum(locf_valid))

    summary = {
        "n_samples": int(df["hadm_id"].nunique()),
        "n_occurrences": int(len(df)),
        "vid": int(vid),
        "is_numeric": True,
        "target_mean": float(mean_i),
        "target_std": float(std_i),
        "protocol": "masked_observed_reconstruction",
        "context_protocol": "full_context_bidirectional",
        "metric_space": "standardized",
        "paper_metric_space": "standardized",
        "raw_unit_metrics_available": True,
        "target_value_removed_from_same_timestamp_context": True,
        "same_timestamp_non_target_variables_visible": True,
        "future_context_visible": True,
        "causal_protocol": False,
        "official_baselines": ["last-value-carry-forward"],
        "readout_mse_macro": std_metrics["mse"],
        "readout_rmse_macro": std_metrics["rmse"],
        "readout_mae_macro": std_metrics["mae"],
        "readout_r2_macro": std_metrics["r2"],
        "numeric_mse_macro": std_metrics["mse"],
        "numeric_rmse_macro": std_metrics["rmse"],
        "numeric_mae_macro": std_metrics["mae"],
        "numeric_r2_macro": std_metrics["r2"],
        "feature_readout_mse_macro": std_metrics["mse"],
        "feature_mean_baseline_mse_macro": mean_metrics["mse"],
        "feature_mean_baseline_r2_macro": mean_metrics["r2"],
        "last_value_baseline_mse_macro": merged_locf_metrics["mse"],
        "last_value_baseline_r2_macro": merged_locf_metrics["r2"],
        "mse": std_metrics["mse"],
        "rmse": std_metrics["rmse"],
        "mae": std_metrics["mae"],
        "r2": std_metrics["r2"],
        "pearson_r": std_metrics["pearson_r"],
        "nll_gaussian": std_metrics.get("nll_gaussian", float("nan")),
        "mean_sigma": std_metrics.get("mean_sigma", float("nan")),
        "coverage_68": std_metrics.get("coverage_68", float("nan")),
        "coverage_95": std_metrics.get("coverage_95", float("nan")),
        "baseline_protocol": "strict_locf_then_schema_mean_fallback",
        "last_value_carry_forward_mse": merged_locf_metrics["mse"],
        "last_value_carry_forward_rmse": merged_locf_metrics["rmse"],
        "last_value_carry_forward_mae": merged_locf_metrics["mae"],
        "last_value_carry_forward_r2": merged_locf_metrics["r2"],
        "last_value_carry_forward_valid_fraction": valid_locf_frac,
        "last_value_carry_forward_n_valid": valid_locf_count,
        "strict_last_value_carry_forward_mse": locf_metrics["mse"],
        "strict_last_value_carry_forward_rmse": locf_metrics["rmse"],
        "strict_last_value_carry_forward_mae": locf_metrics["mae"],
        "strict_last_value_carry_forward_r2": locf_metrics["r2"],
        "variable_mean_mse": mean_metrics["mse"],
        "variable_mean_rmse": mean_metrics["rmse"],
        "variable_mean_mae": mean_metrics["mae"],
        "variable_mean_r2": mean_metrics["r2"],
        "delta_mse_vs_last_value_carry_forward": float(merged_locf_metrics["mse"] - std_metrics["mse"]) if np.isfinite(merged_locf_metrics["mse"]) and np.isfinite(std_metrics["mse"]) else float("nan"),
        "delta_mse_vs_variable_mean": float(mean_metrics["mse"] - std_metrics["mse"]) if np.isfinite(mean_metrics["mse"]) else float("nan"),
        "delta_r2_vs_last_value_carry_forward": float(std_metrics["r2"] - merged_locf_metrics["r2"]) if np.isfinite(std_metrics["r2"]) and np.isfinite(merged_locf_metrics["r2"]) else float("nan"),
        "delta_r2_vs_variable_mean": float(std_metrics["r2"] - mean_metrics["r2"]) if np.isfinite(std_metrics["r2"]) and np.isfinite(mean_metrics["r2"]) else float("nan"),
        "percent_mse_reduction_vs_last_value_carry_forward": float((merged_locf_metrics["mse"] - std_metrics["mse"]) / merged_locf_metrics["mse"]) if np.isfinite(merged_locf_metrics["mse"]) and merged_locf_metrics["mse"] != 0.0 and np.isfinite(std_metrics["mse"]) else float("nan"),
        "percent_mse_reduction_vs_variable_mean": float((mean_metrics["mse"] - std_metrics["mse"]) / mean_metrics["mse"]) if np.isfinite(mean_metrics["mse"]) and mean_metrics["mse"] != 0.0 and np.isfinite(std_metrics["mse"]) else float("nan"),
        "atlas_mse_on_locf_valid_subset": atlas_on_locf_valid["mse"],
        "atlas_r2_on_locf_valid_subset": atlas_on_locf_valid["r2"],
        "delta_mse_vs_last_value_carry_forward_on_valid_subset": float(locf_metrics["mse"] - atlas_on_locf_valid["mse"]) if np.isfinite(locf_metrics["mse"]) and np.isfinite(atlas_on_locf_valid["mse"]) else float("nan"),
        "delta_r2_vs_last_value_carry_forward_on_valid_subset": float(atlas_on_locf_valid["r2"] - locf_metrics["r2"]) if np.isfinite(atlas_on_locf_valid["r2"]) and np.isfinite(locf_metrics["r2"]) else float("nan"),
        "actual_readout_call_mode": "phase1_target_manifold_inverse_v1",
        "readout_artifact": str(readout_artifact),
        "raw_unit_metrics": {
            "metric_space": "raw",
            "mse": raw_metrics["mse"],
            "rmse": raw_metrics["rmse"],
            "mae": raw_metrics["mae"],
            "r2": raw_metrics["r2"],
            "pearson_r": raw_metrics["pearson_r"],
            "nll_gaussian": raw_metrics.get("nll_gaussian", float("nan")),
            "mean_sigma": raw_metrics.get("mean_sigma", float("nan")),
            "coverage_68": raw_metrics.get("coverage_68", float("nan")),
            "coverage_95": raw_metrics.get("coverage_95", float("nan")),
            "strict_last_value_carry_forward_mse": locf_raw_metrics["mse"],
            "strict_last_value_carry_forward_rmse": locf_raw_metrics["rmse"],
            "strict_last_value_carry_forward_mae": locf_raw_metrics["mae"],
            "strict_last_value_carry_forward_r2": locf_raw_metrics["r2"],
            "strict_last_value_carry_forward_valid_fraction": float(np.mean(strict_locf_valid)) if len(df) else float("nan"),
            "strict_last_value_carry_forward_n_valid": int(np.sum(strict_locf_valid)),
            "last_value_carry_forward_mse": merged_locf_raw_metrics["mse"],
            "last_value_carry_forward_rmse": merged_locf_raw_metrics["rmse"],
            "last_value_carry_forward_mae": merged_locf_raw_metrics["mae"],
            "last_value_carry_forward_r2": merged_locf_raw_metrics["r2"],
            "last_value_carry_forward_valid_fraction": float(np.mean(strict_locf_valid)) if len(df) else float("nan"),
            "last_value_carry_forward_n_valid": int(np.sum(strict_locf_valid)),
            "variable_mean_mse": mean_raw_metrics["mse"],
            "variable_mean_rmse": mean_raw_metrics["rmse"],
            "variable_mean_mae": mean_raw_metrics["mae"],
            "variable_mean_r2": mean_raw_metrics["r2"],
        },
    }

    _imp__validate_standardized_imputation_df(df)
    d_plot_out = df[[
        "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
        "true_value", "pred_mu", "pred_sigma", "locf_pred", "item_mean_pred",
        "last_value_carry_forward_pred", "variable_mean_pred",
        "residual_model", "residual_locf", "residual_variable_mean",
        "true_value_raw", "pred_mu_raw", "pred_sigma_raw", "locf_pred_raw",
        "item_mean_pred_raw", "last_value_carry_forward_pred_raw", "variable_mean_pred_raw",
        "residual_model_raw", "residual_locf_raw", "residual_variable_mean_raw",
    ]].copy()
    if write_outputs:
        os.makedirs(output_dir, exist_ok=True)
        standardized_plot = d_plot_out[[
            "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
            "true_value", "pred_mu", "pred_sigma", "locf_pred", "item_mean_pred",
            "last_value_carry_forward_pred", "variable_mean_pred",
            "residual_model", "residual_locf", "residual_variable_mean",
        ]].copy()
        standardized_plot["metric_space"] = "standardized"
        raw_plot = d_plot_out[[
            "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
            "true_value_raw", "pred_mu_raw", "pred_sigma_raw", "locf_pred_raw",
            "item_mean_pred_raw", "last_value_carry_forward_pred_raw", "variable_mean_pred_raw",
            "residual_model_raw", "residual_locf_raw", "residual_variable_mean_raw",
        ]].copy()
        raw_plot["metric_space"] = "raw"
        standardized_plot.to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(vid)}_standardized.csv"), index=False)
        raw_plot.to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(vid)}_raw.csv"), index=False)
        standardized_plot.to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(vid)}.csv"), index=False)
        df.to_parquet(os.path.join(output_dir, "results.parquet"), index=False)
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        per_pt = (
            df.assign(sq_err=(df["pred_mu"] - df["true_value"]) ** 2, abs_err=np.abs(df["pred_mu"] - df["true_value"]))
            .groupby("hadm_id")
            .agg(mse=("sq_err", "mean"), mae=("abs_err", "mean"), n_obs=("sq_err", "count"))
            .reset_index()
        )
        per_pt["metric_space"] = "standardized"
        per_pt.to_csv(os.path.join(output_dir, "per_patient_mse.csv"), index=False)
        per_pt_raw = (
            df.assign(sq_err=(df["pred_mu_raw"] - df["true_value_raw"]) ** 2, abs_err=np.abs(df["pred_mu_raw"] - df["true_value_raw"]))
            .groupby("hadm_id")
            .agg(mse=("sq_err", "mean"), mae=("abs_err", "mean"), n_obs=("sq_err", "count"))
            .reset_index()
        )
        per_pt_raw["metric_space"] = "raw"
        per_pt_raw.to_csv(os.path.join(output_dir, "per_patient_mse_raw.csv"), index=False)
        sel_meta = _imp_select_representative_patient(df, min_plot_observations=5)
        with open(os.path.join(output_dir, "plot_selection.json"), "w") as f:
            json.dump({
                "vid": int(vid),
                "feature_name": str(feat_label),
                **sel_meta,
            }, f, indent=2)
    return df, summary, d_plot_out, feat_label

def _imp_plot_observed_only_reconstruction(
    *,
    points_df: pd.DataFrame,
    feature_label: str,
    hadm_id: str,
    output_path: str,
    show_uncertainty: bool = True,
    prefer_raw: bool = True,
) -> None:
    """Predictions are shown only at timestamps where the target feature was observed and masked.
    This is a retrospective reconstruction diagnostic, not continuous-time prospective imputation."""
    d = points_df.sort_values("abs_time_min").copy()
    use_raw = bool(prefer_raw) and {"true_value_raw", "pred_mu_raw", "pred_sigma_raw"}.issubset(d.columns)
    if use_raw:
        y_true = d["true_value_raw"].to_numpy(dtype=np.float64)
        y_mu = d["pred_mu_raw"].to_numpy(dtype=np.float64)
        y_sigma = d["pred_sigma_raw"].to_numpy(dtype=np.float64)
        locf = d["locf_pred_raw"].to_numpy(dtype=np.float64) if "locf_pred_raw" in d.columns else np.full(len(d), np.nan, dtype=np.float64)
        y_label = f"{feature_label} (raw units)"
        title = f"Masked-observed reconstruction — raw-unit diagnostic plot — {feature_label} | hadm_id={hadm_id}"
    else:
        y_true = d["true_value"].to_numpy(dtype=np.float64)
        y_mu = d["pred_mu"].to_numpy(dtype=np.float64)
        y_sigma = d["pred_sigma"].to_numpy(dtype=np.float64)
        locf = d["locf_pred"].to_numpy(dtype=np.float64)
        y_label = feature_label
        title = f"Masked-observed reconstruction — {feature_label} | hadm_id={hadm_id}"
    x = d["time_hours"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    if show_uncertainty:
        ax.fill_between(
            x,
            y_mu - y_sigma,
            y_mu + y_sigma,
            color="lightsteelblue",
            alpha=0.25,
            label="Diagnostic uncertainty (μ±σ)",
            zorder=1,
        )
    ax.vlines(x, y_mu, y_true, colors="steelblue", linewidth=1.0, alpha=0.9, zorder=2)
    ax.scatter(x, y_true, color="black", s=28, label="True observed value", zorder=4)
    ax.scatter(x, y_mu, color="steelblue", s=26, label="Readout reconstruction", zorder=5)
    valid_locf = np.isfinite(locf)
    if np.any(valid_locf):
        ax.scatter(x[valid_locf], locf[valid_locf], color="green", s=24, label="LOCF→mean baseline", zorder=4)
        ax.vlines(x[valid_locf], locf[valid_locf], y_true[valid_locf], colors="green", linewidth=0.9, linestyles="--", alpha=0.8, zorder=2)

    ax.set_xlabel("Hours from admission")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[imputation] plot → {output_path}")

def _imp__sample_points(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= int(max_points):
        return df.copy()
    rs = np.random.RandomState(int(seed))
    idx = rs.choice(np.arange(len(df)), size=int(max_points), replace=False)
    idx = np.sort(idx)
    return df.iloc[idx].copy()

def _imp__plot_overall_summary_panel(
    ax: Any,
    summary_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    if summary_payload is None:
        ax.text(0.02, 0.6, "imputation summary not provided", transform=ax.transAxes, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return {}

    js = dict(summary_payload)
    m_last = _imp__safe_float(js.get("last_value_baseline_mse_macro", js.get("last_value_baseline_mse", float("nan"))))
    atlas_keys = [
        "readout_mse_macro",
        "numeric_mse_macro",
        "feature_readout_mse_macro",
        "val_numeric_mse_macro",
        "mse_macro",
        "val_mse_macro",
        "best_monitored_metric",
    ]
    m_atlas = float("nan")
    atlas_mse_source = None
    for k in atlas_keys:
        v = _imp__safe_float(js.get(k, float("nan")))
        if np.isfinite(v):
            m_atlas = float(v)
            atlas_mse_source = f"direct:{k}"
            break
    if not np.isfinite(m_atlas):
        gain_lv = _imp__safe_float(js.get("readout_minus_last_value_gain_macro", float("nan")))
        if np.isfinite(m_last) and np.isfinite(gain_lv):
            m_atlas = float(m_last - gain_lv)
            atlas_mse_source = "derived_from_last_value_gain"
    labels = ["Merged baseline", "ATLAS-EHR"]
    vals = [m_last, m_atlas]
    ys = np.arange(len(labels), dtype=np.float64)
    colors = ["#E69F00", "#0072B2"]
    draw_vals = [v if np.isfinite(v) else 0.0 for v in vals]
    ax.barh(ys, draw_vals, color=colors, height=0.48, alpha=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Macro MSE (lower is better)", fontsize=8)
    for y, v in zip(ys, vals):
        if np.isfinite(v):
            ax.text(float(v), float(y), f"  {v:.3f}", va="center", ha="left", fontsize=7)
    if not np.isfinite(m_atlas):
        ax.text(0.02, 0.18, "ATLAS-EHR MSE unavailable", transform=ax.transAxes, fontsize=7, color="#0072B2")

    cal = js.get("calibration_diagnostics", {})
    if not isinstance(cal, dict):
        cal = {}

    def _pick_with_source(cands: List[Tuple[str, Any]]) -> Tuple[float, Optional[str]]:
        for src, val in cands:
            v = _imp__safe_float(val)
            if np.isfinite(v):
                return float(v), str(src)
        return float("nan"), None

    cov68, cov68_src = _pick_with_source([
        ("coverage_68", js.get("coverage_68", float("nan"))),
        ("val/coverage_68", js.get("val/coverage_68", float("nan"))),
        ("calibration_diagnostics.val/coverage_68", cal.get("val/coverage_68", float("nan"))),
    ])
    cov95, cov95_src = _pick_with_source([
        ("coverage_95", js.get("coverage_95", float("nan"))),
        ("val/coverage_95", js.get("val/coverage_95", float("nan"))),
        ("calibration_diagnostics.val/coverage_95", cal.get("val/coverage_95", float("nan"))),
    ])
    sigma_mean, sigma_src = _pick_with_source([
        ("numeric_sigma_mean", js.get("numeric_sigma_mean", float("nan"))),
        ("mean_sigma", js.get("mean_sigma", float("nan"))),
        ("val/numeric_sigma_mean", js.get("val/numeric_sigma_mean", float("nan"))),
        ("calibration_diagnostics.val/numeric_sigma_mean", cal.get("val/numeric_sigma_mean", float("nan"))),
    ])

    ann: List[str] = []
    n_val_q = js.get("n_val_queries", None)
    if n_val_q is not None:
        ann.append(f"n_val_queries: {n_val_q}")
    n_feat = js.get("number_of_numeric_feature_ids", js.get("n_numeric_feature_ids", None))
    if n_feat is not None:
        ann.append(f"n_numeric_feature_ids: {n_feat}")
    r2_med = _imp__safe_float(js.get("per_feature_r2_median", float("nan")))
    if np.isfinite(r2_med):
        ann.append(f"per_feature_r2_median: {r2_med:.3f}")
    if np.isfinite(cov68) and np.isfinite(cov95):
        ann.append(f"coverage_68 / 95: {cov68:.3f} / {cov95:.3f}")
    elif np.isfinite(cov68):
        ann.append(f"coverage_68: {cov68:.3f}")
    elif np.isfinite(cov95):
        ann.append(f"coverage_95: {cov95:.3f}")
    if np.isfinite(sigma_mean):
        ann.append(f"numeric_sigma_mean: {sigma_mean:.3f}")
    if len(ann) > 0:
        ax.text(1.01, 0.02, "\n".join(ann), transform=ax.transAxes, va="bottom", ha="left", fontsize=7)
    meta = dict(js)
    meta["atlas_mse_lookup_priority"] = atlas_keys
    meta["atlas_mse_source"] = atlas_mse_source
    if not np.isfinite(m_atlas):
        meta["atlas_mse_unavailable"] = True
        meta["available_summary_keys"] = sorted([str(k) for k in js.keys()])
    meta["panel_A_final_values"] = {
        "last_value_baseline_mse_macro": float(m_last) if np.isfinite(m_last) else None,
        "atlas_mse_macro": float(m_atlas) if np.isfinite(m_atlas) else None,
        "coverage_68": float(cov68) if np.isfinite(cov68) else None,
        "coverage_95": float(cov95) if np.isfinite(cov95) else None,
        "numeric_sigma_mean": float(sigma_mean) if np.isfinite(sigma_mean) else None,
    }
    meta["coverage_68_source"] = cov68_src
    meta["coverage_95_source"] = cov95_src
    meta["numeric_sigma_mean_source"] = sigma_src
    return meta

def _imp__plot_feature_scatter_panel(
    ax: Any,
    *,
    df_full: pd.DataFrame,
    vid: int,
    feature_label: str,
    per_feature_df: Optional[pd.DataFrame],
    max_points_per_feature: int,
    seed: int,
    show_scatter_uncertainty: bool = False,
) -> Dict[str, Any]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.12, linewidth=0.5)
    d = df_full.copy()
    d = d[np.isfinite(d["true_value"]) & np.isfinite(d["pred_mu"])].copy()
    if d.empty:
        ax.text(0.5, 0.5, f"No data for VID {vid}", transform=ax.transAxes, ha="center", va="center")
        return {"vid": int(vid), "n": 0}

    d_sub = _imp__sample_points(d, max_points_per_feature, seed)
    x = d_sub["true_value"].to_numpy(dtype=np.float64)
    y = d_sub["pred_mu"].to_numpy(dtype=np.float64)
    s = d_sub["pred_sigma"].to_numpy(dtype=np.float64)
    locf = d_sub["locf_pred"].to_numpy(dtype=np.float64)
    item_mean = float(d["item_mean_pred"].iloc[0])

    vals = [d["true_value"].to_numpy(dtype=np.float64), d["pred_mu"].to_numpy(dtype=np.float64)]
    if np.isfinite(item_mean):
        vals.append(np.array([item_mean], dtype=np.float64))
    vlocf = d["locf_pred"].to_numpy(dtype=np.float64)
    vals.append(vlocf[np.isfinite(vlocf)])
    allv = np.concatenate([v for v in vals if v.size > 0])
    lo = float(np.percentile(allv, 1))
    hi = float(np.percentile(allv, 99))
    pad = 0.06 * max(1e-6, hi - lo)
    lo, hi = lo - pad, hi + pad

    ax.plot([lo, hi], [lo, hi], color="#111111", linewidth=1.0, alpha=0.9, zorder=1)
    if bool(show_scatter_uncertainty):
        ax.errorbar(x, y, yerr=s, fmt="none", ecolor="#0072B2", alpha=0.20, elinewidth=0.6, zorder=2, capsize=0)
    ax.scatter(x, y, s=14, color="#0072B2", alpha=0.85, zorder=3, label="ATLAS-EHR")
    mask_locf = np.isfinite(locf)
    if np.any(mask_locf):
        ax.scatter(x[mask_locf], locf[mask_locf], s=18, facecolors="none", edgecolors="#E69F00", linewidths=0.9, zorder=3, label="Last value")
    if np.isfinite(item_mean):
        ax.axhline(item_mean, color="#7A7A7A", linestyle="--", linewidth=0.9, zorder=1, label="Item mean")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True observed value", fontsize=8)
    ax.set_ylabel("Imputed value", fontsize=8)
    ax.set_title(str(feature_label), fontsize=8)

    mse_model = float(np.mean((d["pred_mu"].to_numpy(dtype=np.float64) - d["true_value"].to_numpy(dtype=np.float64)) ** 2))
    locf_full = d["locf_pred"].to_numpy(dtype=np.float64)
    valid_locf = np.isfinite(locf_full)
    mse_locf = float(np.mean((locf_full[valid_locf] - d.loc[valid_locf, "true_value"].to_numpy(dtype=np.float64)) ** 2)) if np.any(valid_locf) else float("nan")
    r2 = _imp__metric_from_report(per_feature_df, int(vid), "r2_value")
    if not np.isfinite(r2):
        t = d["true_value"].to_numpy(dtype=np.float64)
        p = d["pred_mu"].to_numpy(dtype=np.float64)
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = float(1.0 - np.sum((p - t) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    rep_mse = _imp__metric_from_report(per_feature_df, int(vid), "readout_mse")
    rep_locf = _imp__metric_from_report(per_feature_df, int(vid), "last_value_baseline_mse")
    if np.isfinite(rep_mse):
        mse_model = rep_mse
    if np.isfinite(rep_locf):
        mse_locf = rep_locf
    dmse = float(mse_locf - mse_model) if np.isfinite(mse_locf) else float("nan")
    txt = (
        f"n={len(d)}\n"
        f"MSE atlas={mse_model:.3f}\n"
        f"MSE last={mse_locf:.3f}\n"
        f"ΔMSE={dmse:.3f}\n"
        f"R²={r2:.3f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    return {
        "vid": int(vid),
        "feature_name": str(feature_label),
        "n": int(len(d)),
        "mse_atlas": float(mse_model),
        "mse_last_value": float(mse_locf),
        "delta_mse": float(dmse),
        "r2": float(r2),
    }

def _imp_make_multi_feature_reconstruction_figure(
    *,
    summary_payload: Optional[Dict[str, Any]],
    feature_payloads: List[Dict[str, Any]],
    per_feature_df: Optional[pd.DataFrame],
    max_points_per_feature: int,
    output_dir: str,
    show_scatter_uncertainty: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=150)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7, width=0.6, length=3)
        for side in ["left", "bottom"]:
            ax.spines[side].set_linewidth(0.6)

    summary_blob = {"panel_A": {}, "features": []}
    summary_blob["panel_A"] = _imp__plot_overall_summary_panel(axes[0, 0], summary_payload)
    letters = ["A", "B", "C", "D"]
    axes[0, 0].text(-0.18, 1.05, letters[0], transform=axes[0, 0].transAxes, fontsize=10, fontweight="bold")
    for i in range(3):
        r = 0 if i == 0 else 1
        c = 1 if i == 0 else (i - 1)
        ax = axes[r, c]
        if i >= len(feature_payloads):
            ax.axis("off")
            continue
        fp = feature_payloads[i]
        m = _imp__plot_feature_scatter_panel(
            ax,
            df_full=fp["df"],
            vid=int(fp["vid"]),
            feature_label=str(fp["feature_name"]),
            per_feature_df=per_feature_df,
            max_points_per_feature=max_points_per_feature,
            seed=1000 + i,
            show_scatter_uncertainty=bool(show_scatter_uncertainty),
        )
        summary_blob["features"].append(m)
        ax.text(-0.18, 1.05, letters[i + 1], transform=ax.transAxes, fontsize=10, fontweight="bold")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.98])

    out_png = os.path.join(output_dir, "imputation_reconstruction_main.png")
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    return out_png, summary_blob

def _imp_run_imputation(
    *,
    phase1_ckpt: str,
    readout_artifact: str,
    data_dir: str,
    schema_dir: str,
    vid: int,
    output_dir: str,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    device: torch.device,
    occurrence_cache: Optional[str] = None,
    split: str = "test",
    plot_hadm_id: Optional[str] = None,
    plot_dense_exploratory: bool = False,
    min_plot_observations: int = 5,
    per_feature_report: Optional[str] = None,
    plot_vids: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    make_multi_feature_figure_flag: bool = False,
    max_points_per_feature: int = 500,
) -> Dict[str, Any]:
    """Main imputation loop. Returns summary metrics dict."""
    schema_path = os.path.join(schema_dir, "schema_processed.csv")
    schema_df = pd.read_csv(schema_path)

    teacher = JEPAModel.load_from_checkpoint(
        phase1_ckpt, map_location="cpu", weights_only=False,
        schema_path=schema_path,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    probe = load_readout_artifact(readout_artifact, map_location=device, expected_phase1_ckpt=phase1_ckpt, expected_schema=schema_path, expected_phase1_architecture=getattr(teacher, "phase1_architecture_version", None)).to(device).eval()
    probe.requires_grad_(False)

    dataset = PatientSequenceDataset(
        data_dir=data_dir, schema_dir=schema_dir, split=str(split),
        phase=1, mask_target_ratio=0.0, mask_min_prob=0.0,
        mask_max_prob=0.0, static_mask_prob=0.0, static_numeric_vids=[],
    )
    print(f"[imputation] dataset={len(dataset)}")
    os.makedirs(output_dir, exist_ok=True)

    run_vids: List[int] = [int(vid)]
    if bool(make_multi_feature_figure_flag) and plot_vids is not None and len(plot_vids) > 0:
        run_vids = [int(v) for v in plot_vids]
        if int(vid) not in run_vids:
            run_vids = [int(vid)] + run_vids
    # stable unique
    seen = set()
    run_vids = [v for v in run_vids if not (v in seen or seen.add(v))]

    name_map_multi: Dict[int, str] = {}
    if feature_names is not None and plot_vids is not None:
        for i, v in enumerate(plot_vids):
            if i < len(feature_names):
                name_map_multi[int(v)] = str(feature_names[i])

    per_feature_df: Optional[pd.DataFrame] = None
    if per_feature_report is not None and os.path.isfile(str(per_feature_report)):
        per_feature_df = pd.read_csv(str(per_feature_report))

    all_vid_outputs: Dict[int, Dict[str, Any]] = {}
    for cur_vid in run_vids:
        vid_out_dir = os.path.join(output_dir, "per_vid", f"vid_{int(cur_vid)}")
        df_vid, summary_vid, d_plot_vid, feat_label = _imp__run_single_vid_inference(
            teacher=teacher,
            probe=probe,
            dataset=dataset,
            schema_df=schema_df,
            vid=int(cur_vid),
            output_dir=vid_out_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            device=device,
            max_samples=max_samples,
            occurrence_cache=occurrence_cache,
            readout_artifact=str(readout_artifact),
        )
        if not d_plot_vid.empty:
            d_plot_vid["feature_name"] = str(name_map_multi.get(int(cur_vid), feat_label))
        all_vid_outputs[int(cur_vid)] = {
            "df": df_vid,
            "summary": summary_vid,
            "plot_points": d_plot_vid,
            "feature_name": str(name_map_multi.get(int(cur_vid), feat_label)),
            "out_dir": vid_out_dir,
        }

    if int(vid) not in all_vid_outputs or all_vid_outputs[int(vid)]["df"].empty:
        print("[imputation] No records produced for primary VID.")
        return {"n_samples": 0, "n_occurrences": 0}

    # Backward-compatible root outputs for primary VID.
    primary = all_vid_outputs[int(vid)]
    primary_df: pd.DataFrame = primary["df"]
    primary_summary: Dict[str, Any] = dict(primary["summary"])
    primary_plot_df: pd.DataFrame = primary["plot_points"]
    primary_feature_name = str(primary["feature_name"])
    primary_df.to_parquet(os.path.join(output_dir, "results.parquet"), index=False)
    primary_plot_df[[
        "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
        "true_value", "pred_mu", "pred_sigma", "locf_pred", "item_mean_pred",
        "last_value_carry_forward_pred", "variable_mean_pred",
        "residual_model", "residual_locf", "residual_variable_mean",
    ]].to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(vid)}_standardized.csv"), index=False)
    primary_plot_df[[
        "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
        "true_value_raw", "pred_mu_raw", "pred_sigma_raw", "locf_pred_raw",
        "item_mean_pred_raw", "last_value_carry_forward_pred_raw", "variable_mean_pred_raw",
        "residual_model_raw", "residual_locf_raw", "residual_variable_mean_raw",
    ]].to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(vid)}_raw.csv"), index=False)
    primary_plot_df[[
        "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
        "true_value", "pred_mu", "pred_sigma", "locf_pred", "item_mean_pred",
        "last_value_carry_forward_pred", "variable_mean_pred",
        "residual_model", "residual_locf", "residual_variable_mean",
    ]].to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(vid)}.csv"), index=False)
    pd.read_csv(os.path.join(primary["out_dir"], "per_patient_mse.csv")).to_csv(
        os.path.join(output_dir, "per_patient_mse.csv"), index=False
    )
    if os.path.isfile(os.path.join(primary["out_dir"], "per_patient_mse_raw.csv")):
        pd.read_csv(os.path.join(primary["out_dir"], "per_patient_mse_raw.csv")).to_csv(
            os.path.join(output_dir, "per_patient_mse_raw.csv"), index=False
        )

    # Representative-patient default plot remains available for single-VID mode.
    sel_meta = _imp_select_representative_patient(primary_df, min_plot_observations=int(min_plot_observations))
    plot_hadm = str(plot_hadm_id) if plot_hadm_id is not None else str(sel_meta["selected_hadm_id"])
    d_single = primary_plot_df[primary_plot_df["hadm_id"].astype(str) == plot_hadm].copy()
    if d_single.empty:
        d_single = primary_plot_df.copy().sort_values("abs_time_min")
        plot_hadm = str(d_single["hadm_id"].iloc[0])
    _imp_plot_observed_only_reconstruction(
        points_df=d_single,
        feature_label=primary_feature_name,
        hadm_id=plot_hadm,
        output_path=os.path.join(output_dir, "masked_observed_reconstruction_plot.png"),
        show_uncertainty=True,
        prefer_raw=True,
    )

    # Optional dense exploratory figure (single primary VID only).
    if bool(plot_dense_exploratory):
        occurrences_primary = _imp__find_occurrences(
            dataset, int(vid), max_samples,
            cache_path=None, num_workers=num_workers,
        )
        ds_idx = next((o[0] for o in occurrences_primary if str(o[4]) == plot_hadm), occurrences_primary[0][0])
        vid_tensor = torch.tensor([int(vid)], dtype=torch.long, device=device)
        abs_times, mus, sigmas, obs_abs_t, obs_v = _imp__dense_predict(
            teacher, probe, dataset, ds_idx, int(vid),
            vid_tensor, device,
        )
        row = schema_df[schema_df["vid"].astype(int) == int(vid)].iloc[0]
        mean_i, std_i = float(row["mean"]), float(row["std"])
        pred_mu_std_std = float(np.std(mus)) if mus.size > 0 else float("nan")
        primary_summary["pred_mu_std_std"] = pred_mu_std_std
        primary_summary["pred_mu_std_min"] = float(np.min(mus)) if mus.size > 0 else float("nan")
        primary_summary["pred_mu_std_max"] = float(np.max(mus)) if mus.size > 0 else float("nan")
        mus = (mus * std_i) + mean_i
        primary_summary["pred_mu_raw_std"] = float(np.std(mus)) if mus.size > 0 else float("nan")
        if np.isfinite(pred_mu_std_std) and pred_mu_std_std < 1.0e-6:
            primary_summary["prediction_constant_warning"] = "Predictions are nearly constant; z_q may still be collapsed."
        _imp__plot_imputation(
            abs_times, mus, sigmas * std_i, obs_abs_t, (obs_v * std_i) + mean_i,
            primary_feature_name, plot_hadm,
            os.path.join(output_dir, "dense_exploratory_plot.png"),
        )

    # Build manuscript multi-feature outputs.
    combined_points = []
    for v in run_vids:
        p = all_vid_outputs[int(v)]["plot_points"]
        if not p.empty:
            combined_points.append(p)
            p_std = p[[
                "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
                "true_value", "pred_mu", "pred_sigma", "locf_pred", "item_mean_pred",
                "last_value_carry_forward_pred", "variable_mean_pred",
                "residual_model", "residual_locf", "residual_variable_mean",
            ]].copy()
            p_std["metric_space"] = "standardized"
            p_raw = p[[
                "hadm_id", "vid", "feature_name", "abs_time_min", "time_hours", "metric_space",
                "true_value_raw", "pred_mu_raw", "pred_sigma_raw", "locf_pred_raw",
                "item_mean_pred_raw", "last_value_carry_forward_pred_raw", "variable_mean_pred_raw",
                "residual_model_raw", "residual_locf_raw", "residual_variable_mean_raw",
            ]].copy()
            p_raw["metric_space"] = "raw"
            p_std.to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(v)}_standardized.csv"), index=False)
            p_raw.to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(v)}_raw.csv"), index=False)
            p_std.to_csv(os.path.join(output_dir, f"plotted_points_vid_{int(v)}.csv"), index=False)
    if combined_points:
        df_combined = pd.concat(combined_points, ignore_index=True)
        df_combined.to_csv(os.path.join(output_dir, "multi_feature_plotted_points.csv"), index=False)

    payloads = []
    requested_for_main = [int(v) for v in (plot_vids or [vid])]
    for v in requested_for_main[:3]:
        if int(v) in all_vid_outputs and not all_vid_outputs[int(v)]["df"].empty:
            payloads.append({
                "vid": int(v),
                "df": all_vid_outputs[int(v)]["plot_points"],
                "feature_name": all_vid_outputs[int(v)]["feature_name"],
            })
    fig_main_path, multi_blob = _imp_make_multi_feature_reconstruction_figure(
        summary_payload=primary_summary,
        feature_payloads=payloads,
        per_feature_df=per_feature_df,
        max_points_per_feature=int(max_points_per_feature),
        output_dir=output_dir,
    )

    multi_summary = {
        "plot_protocol": "Predictions are shown only at timestamps where the target feature was observed and masked. This is a retrospective reconstruction diagnostic, not continuous-time prospective imputation.",
        "main_figure": fig_main_path,
        "vids_evaluated": run_vids,
        "per_vid_output_dirs": {str(v): all_vid_outputs[int(v)]["out_dir"] for v in run_vids},
        "panel_metrics": multi_blob,
    }
    with open(os.path.join(output_dir, "multi_feature_summary.json"), "w") as f:
        json.dump(multi_summary, f, indent=2)

    primary_summary.update({
        "slot_coordinate_system": "item_slot_plus_one_to_batch_slot_due_to_nested_padding_collate",
        "batch_slot_offset": 1,
        "readout_call_mode": "phase1_target_manifold_inverse_v1",
        "split": str(split),
        "dense_exploratory_enabled": bool(plot_dense_exploratory),
        "plot_protocol": multi_summary["plot_protocol"],
    })
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(primary_summary, f, indent=2)
    return primary_summary

def _imp_compute_autocorrelation_from_occurrence_index(
    occurrence_index_path: str,
    output_path: str,
    schema_path: Optional[str] = None,
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Compute per-variable temporal autocorrelation metrics from occurrence_index.parquet.

    For each variable (vid), groups observations by (vid, hadm_id), sorts by
    abs_time_min, and computes lag-1 statistics across all admissions.

    Metrics (all in standardized space, scale-invariant for correlations):
      lag1_pearson      – mean Pearson r between consecutive measurements (primary)
      lag1_spearman     – mean Spearman r (rank-based, robust to outliers)
      mean_abs_change   – mean |x_t − x_{t−1}| (variability proxy)
      n_pairs           – total lag-1 consecutive pairs
      n_admissions      – admissions contributing ≥2 measurements
      n_occurrences     – total observations for this variable
    """
    from scipy.stats import pearsonr, spearmanr

    if not os.path.isfile(occurrence_index_path):
        raise FileNotFoundError(f"occurrence_index.parquet not found: {occurrence_index_path}")

    print(f"[autocorrelation] loading {occurrence_index_path}")
    occ = pd.read_parquet(occurrence_index_path)
    required = {"vid", "hadm_id", "abs_time_min", "true_value"}
    if not required.issubset(occ.columns):
        raise RuntimeError(f"occurrence_index.parquet missing columns: {required - set(occ.columns)}")

    occ["vid"] = pd.to_numeric(occ["vid"], errors="coerce").astype("Int64")
    occ["true_value"] = pd.to_numeric(occ["true_value"], errors="coerce")
    occ["abs_time_min"] = pd.to_numeric(occ["abs_time_min"], errors="coerce")
    occ = occ.dropna(subset=["vid", "true_value", "abs_time_min"])

    label_map: Dict[int, str] = {}
    if schema_path and os.path.isfile(schema_path):
        schema_df = pd.read_csv(schema_path)
        label_map = {int(r["vid"]): str(r["normalized_concept_label"])
                     for _, r in schema_df.iterrows()
                     if pd.notna(r.get("vid")) and pd.notna(r.get("normalized_concept_label"))}

    rows_out: List[Dict[str, Any]] = []
    vids = sorted(occ["vid"].dropna().unique().tolist())
    print(f"[autocorrelation] computing for {len(vids)} vids ...")

    for vid in tqdm(vids, desc="autocorrelation/vids", ascii=True):
        vid_df = occ[occ["vid"] == vid].copy()
        n_occ = len(vid_df)

        # Per-admission lag-1 statistics
        pearson_vals: List[float] = []
        spearman_vals: List[float] = []
        abs_changes: List[float] = []
        n_pairs_total = 0
        n_adm_with_pairs = 0

        for hadm_id, grp in vid_df.groupby("hadm_id", sort=False):
            grp = grp.sort_values("abs_time_min")
            vals = grp["true_value"].to_numpy(dtype=np.float64)
            valid = np.isfinite(vals)
            vals = vals[valid]
            if len(vals) < 2:
                continue
            n_adm_with_pairs += 1
            x, y = vals[:-1], vals[1:]
            n_pairs_total += len(x)
            abs_changes.extend(np.abs(y - x).tolist())

            # Pearson lag-1
            if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                try:
                    r, _ = pearsonr(x, y)
                    if np.isfinite(r):
                        pearson_vals.append(float(r))
                except Exception:
                    pass

            # Spearman lag-1
            if len(x) >= 3 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
                try:
                    rs, _ = spearmanr(x, y)
                    if np.isfinite(rs):
                        spearman_vals.append(float(rs))
                except Exception:
                    pass

        if n_pairs_total < min_pairs:
            continue

        rows_out.append({
            "vid": int(vid),
            "feature_name": label_map.get(int(vid), f"VID {vid}"),
            "n_occurrences": int(n_occ),
            "n_admissions": int(n_adm_with_pairs),
            "n_pairs": int(n_pairs_total),
            "lag1_pearson": float(np.mean(pearson_vals)) if pearson_vals else float("nan"),
            "lag1_pearson_std": float(np.std(pearson_vals)) if len(pearson_vals) > 1 else float("nan"),
            "lag1_spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
            "mean_abs_change": float(np.mean(abs_changes)) if abs_changes else float("nan"),
        })

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(output_path, index=False)
    print(f"[autocorrelation] saved {len(df_out)} vars → {output_path}")
    return df_out



def _imp_main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature imputation via Phase 1 JEPA + target-manifold readout")
    parser.add_argument("--phase1_ckpt", required=False)
    parser.add_argument("--readout_artifact", required=False)
    parser.add_argument("--readout_dir", required=False,
                        help="Formal imputation input directory containing frozen readout outputs.")
    parser.add_argument("--config", required=False,
                        help="Optional config used to recreate the formal Phase-1 dataset policy for query/raw parity.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--schema_dir", required=True)
    parser.add_argument("--vid", required=False, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_patients", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--occurrence_cache", type=str, default=None)
    parser.add_argument("--plot_hadm_id", type=str, default=None,
                        help="hadm_id of patient to plot; default: representative patient by median patient-MSE rule")
    parser.add_argument("--plot_dense_exploratory", action="store_true", default=False,
                        help="Optional impute-mode single-patient dense all-timestep query plot (not manuscript default).")
    parser.add_argument("--min_plot_observations", type=int, default=5)
    parser.add_argument("--per_feature_report", type=str, default=None)
    parser.add_argument("--plot_vids", type=int, nargs="*", default=None)
    parser.add_argument("--feature_names", type=str, nargs="*", default=None)
    parser.add_argument("--make_multi_feature_figure", action="store_true", default=False,
                        help="Optional impute-mode manuscript multi-feature reconstruction figure from --plot_vids.")
    parser.add_argument("--max_points_per_feature", type=int, default=500)
    parser.add_argument("--all_vids", action="store_true", default=False,
                        help="Deprecated formal path; use --readout_dir. Kept only to fail closed.")
    parser.add_argument("--autocorrelation_only", action="store_true", default=False,
                        help="Skip inference; compute autocorrelation.csv from existing occurrence_index.parquet.")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--permutation_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_paired_queries", type=int, default=100)
    parser.add_argument("--min_paired_admissions", type=int, default=20)
    parser.add_argument("--min_persistence_pairs", type=int, default=20)
    parser.add_argument("--min_persistence_admissions", type=int, default=5)
    parser.add_argument("--query_value_tolerance", type=float, default=1.0e-5)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="impute", choices=["impute"],
                        help="Imputation analysis mode. Only 'impute' is supported.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.readout_dir:
        summary = _imp_run_readout_output_imputation(
            readout_dir=args.readout_dir,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            output_dir=args.output_dir,
            split=args.split,
            config_path=args.config,
            bootstrap_iters=int(args.bootstrap_iters),
            permutation_iters=int(args.permutation_iters),
            seed=int(args.seed),
            min_paired_queries=int(args.min_paired_queries),
            min_paired_admissions=int(args.min_paired_admissions),
            min_persistence_pairs=int(args.min_persistence_pairs),
            min_persistence_admissions=int(args.min_persistence_admissions),
            query_value_tolerance=float(args.query_value_tolerance),
            resume=bool(args.resume),
        )
    elif bool(getattr(args, "autocorrelation_only", False)):
        occ_path = os.path.join(args.output_dir, "occurrence_index.parquet")
        out_path = os.path.join(args.output_dir, "autocorrelation.csv")
        schema_path = os.path.join(args.schema_dir, "schema_processed.csv")
        summary = {"autocorrelation_path": out_path}
        _imp_compute_autocorrelation_from_occurrence_index(
            occurrence_index_path=occ_path,
            output_path=out_path,
            schema_path=schema_path,
        )
    elif bool(args.all_vids):
        raise RuntimeError("--all_vids re-runs Phase-1/readout inference and is no longer a formal imputation mode; pass --readout_dir instead.")
    else:
        if args.phase1_ckpt is None or args.readout_artifact is None:
            raise ValueError("--phase1_ckpt and --readout_artifact are required when --mode impute")
        if args.vid is None and not (bool(args.make_multi_feature_figure) and args.plot_vids is not None and len(args.plot_vids) > 0):
            raise ValueError("--vid is required when --mode impute unless --make_multi_feature_figure with --plot_vids is provided")
        primary_vid = int(args.vid) if args.vid is not None else int(args.plot_vids[0])
        summary = _imp_run_imputation(
            phase1_ckpt=args.phase1_ckpt,
            readout_artifact=args.readout_artifact,
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            vid=primary_vid,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            device=device,
            occurrence_cache=args.occurrence_cache,
            split=args.split,
            plot_hadm_id=args.plot_hadm_id,
            plot_dense_exploratory=bool(args.plot_dense_exploratory),
            min_plot_observations=int(args.min_plot_observations),
            per_feature_report=args.per_feature_report,
            plot_vids=args.plot_vids,
            feature_names=args.feature_names,
            make_multi_feature_figure_flag=bool(args.make_multi_feature_figure),
            max_points_per_feature=int(args.max_points_per_feature),
        )
    print(json.dumps(summary, indent=2))
