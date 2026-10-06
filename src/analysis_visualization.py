"""Visualization analysis workflow implementation."""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.cache_bank import FULL_VISIBLE, _phase1_dataset_kwargs, load_dataset
from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.model import JEPAModel
from src.utils import load_config_dict, load_labels_csv, set_seed

# ===== visualize implementation =====
"""
ATLAS-EHR Visualization: unified section-based visualization module.

Two sections dispatched from a single CLI entry point:

  latent_visualization        — canonical z_set / z_context HDBSCAN + t-SNE
  temporal_embeddings         — JEPAModel time embedding UMAP (model weights + synthetic grid)

Data sources:
  - Section 1 → FULL_VISIBLE cache + labels.csv
  - Section 2 → synthetic integer grid [1..max_time_int] passed through model time embedding modules

Dimensionality reduction: UMAP + t-SNE.
No matplotlib.
"""

matplotlib.use("Agg")

try:
    from sklearn.cluster import HDBSCAN  # type: ignore
except Exception:
    try:
        from hdbscan import HDBSCAN  # type: ignore
    except Exception as exc:
        HDBSCAN = None  # type: ignore
        _HDBSCAN_IMPORT_ERROR = exc
    else:
        _HDBSCAN_IMPORT_ERROR = None
else:
    _HDBSCAN_IMPORT_ERROR = None

_viz_VISUALIZATION_VERSION = "visualization_v5_canonical_probe_cache"

warnings.filterwarnings(
    "ignore",
    message=r".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)

_viz__HCUP_26_LABELS: Dict[int, str] = {
    0: "Acute and unspecified renal failure",
    1: "Acute cerebrovascular disease",
    2: "Acute myocardial infarction",
    3: "Cardiac dysrhythmias",
    4: "Chronic kidney disease",
    5: "Chronic obstructive pulmonary disease",
    6: "Complications of surgical/medical care",
    7: "Conduction disorders",
    8: "Congestive heart failure; nonhypertensive",
    9: "Coronary atherosclerosis and related",
    10: "Diabetes mellitus with complications",
    11: "Diabetes mellitus without complication",
    12: "Disorders of lipid metabolism",
    13: "Essential hypertension",
    14: "Fluid and electrolyte disorders",
    15: "Gastrointestinal hemorrhage",
    16: "Hypertension with complications",
    17: "Other liver diseases",
    18: "Other lower respiratory disease",
    19: "Other upper respiratory disease",
    20: "Pleurisy; pneumothorax; pulmonary collapse",
    21: "Pneumonia",
    22: "Respiratory failure; insufficiency; arrest",
    23: "Septicemia (except in labor)",
    24: "Shock",
    25: "Heart failure",
}

_viz_LATENT_DISPLAY_NAMES: Dict[str, str] = {
    "z_set": "z_set",
    "z_context": "z_context",
    "z_event": "z_event",
}

def _viz__display_source_label(source_name: Any) -> str:
    canonical = _viz_LATENT_DISPLAY_NAMES.get(str(source_name), str(source_name))
    if canonical in {"z_set", "z_context", "z_event"}:
        return f"ATLAS-EHR ({canonical})"
    return str(canonical)

_viz__LATENT_SOURCES: List[str] = [
    "z_set",
    "z_context",
]

def _viz_ensure_dir(path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path

def _viz__ensure_dir(path: str) -> str:
    return _viz_ensure_dir(path)

def _viz_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _viz_to_py(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _viz_to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_viz_to_py(v) for v in value]
    return value

def _viz_save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_viz_to_py(payload), fh, indent=2, ensure_ascii=False)

def _viz_save_dataframe(base_path: str, df: pd.DataFrame) -> None:
    df.to_csv(f"{base_path}.csv", index=False)
    try:
        df.to_parquet(f"{base_path}.parquet", index=False)
    except Exception:
        pass

def _viz_save_plotly_html(fig: go.Figure, path: str) -> None:
    # Self-contained HTML is required on compute nodes and offline review
    # environments where the Plotly CDN is unavailable.
    fig.write_html(path, include_plotlyjs=True, full_html=True)

def _viz_resolve_device(value: str) -> str:
    text = str(value).strip().lower()
    if text != "auto":
        return str(value)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _viz_fit_umap_2d(
    x: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    metric: str = "euclidean",
) -> np.ndarray:
    """Fit and transform x → 2D using UMAP.  x: (N, D) float32.  Returns (N, 2) float32."""
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn is required for visualization: pip install umap-learn"
        ) from exc
    model = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
        transform_seed=int(seed),
        low_memory=True,
        n_jobs=1,
    )
    return np.asarray(model.fit_transform(x), dtype=np.float32)

def _viz_subsample_points(
    arrays: List[np.ndarray],
    max_points: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Sub-sample all arrays to max_points rows (same random index set).

    Returns: (subsampled_arrays, indices)
    """
    n = int(arrays[0].shape[0])
    if max_points <= 0 or n <= max_points:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    return [arr[idx] for arr in arrays], idx

def _viz__unique_rows_with_inverse(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique rows and inverse index to reconstruct original order."""
    if x.ndim != 2 or x.shape[0] == 0:
        return x, np.arange(int(x.shape[0]), dtype=np.int64)
    x_c = np.ascontiguousarray(x)
    uniq, inv = np.unique(x_c, axis=0, return_inverse=True)
    return uniq.astype(np.float32, copy=False), inv.astype(np.int64, copy=False)

def _viz_load_full_visible_bank(
    cache_root: str,
    split: str,
    max_patients: int,
    seed: int,
    config: str,
    schema_dir: Optional[str] = None,
    teacher: Optional[JEPAModel] = None,
    data_dir: Optional[str] = None,
    include_z_event: bool = False,
) -> Dict[str, Any]:
    """Load latents + metadata from FULL_VISIBLE cache.

    Returns dict:
      "latents"    → Dict[str, np.ndarray]  — key per latent source, shape (N, D)
      "meta"       → List[Dict]              — per-point metadata
      "n_patients" → int
      "n_points"   → int
    """
    ds = load_dataset(cache_root, FULL_VISIBLE, split=split)
    source_names = list(_viz__LATENT_SOURCES)
    raw_ds = None
    raw_by_id: Dict[str, int] = {}
    if include_z_event or data_dir:
        if include_z_event and teacher is None:
            raise ValueError("--include_z_event requires --phase1_ckpt and --data_dir.")
        cfg = load_config_dict(str(config))
        ds_kwargs = _phase1_dataset_kwargs(cfg)
        ds_kwargs.pop("max_seq_len_limit", None)
        ds_kwargs.update({
            "mask_target_ratio": 0.0,
            "mask_min_prob": 0.0,
            "mask_max_prob": 0.0,
            "static_mask_prob": 0.0,
            "static_numeric_vids": [],
        })
        raw_ds = PatientSequenceDataset(
            data_dir=str(data_dir), schema_dir=str(schema_dir or ""), split=split,
            fraction=1.0, deterministic=True, return_legacy_nested=False,
            **ds_kwargs,
        )
        raw_by_id = {os.path.basename(path).rsplit(".", 1)[0]: i for i, path in enumerate(raw_ds.files)}
        if include_z_event:
            source_names.append("z_event")
    n_total = len(ds)

    # Determine which patients to load.
    hadm_ids_order: List[int] = list(range(n_total))
    if max_patients > 0 and n_total > max_patients:
        rng = np.random.default_rng(int(seed))
        hadm_ids_order = sorted(
            rng.choice(n_total, size=int(max_patients), replace=False).tolist()
        )

    latent_bufs: Dict[str, List[np.ndarray]] = {name: [] for name in source_names}
    state_code_bufs: List[np.ndarray] = []
    state_val_bufs: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    for idx in tqdm(hadm_ids_order, desc=f"load_full_visible/{split}", ascii=True, mininterval=10.0):
        item = ds[idx]
        hid = str(item["hadm_id"])
        # Canonical packed probe cache fields are (T,D) latents and (T,) time_min.
        abs_time = item["time_min"].float().reshape(-1)
        t_seq = int(abs_time.size(0))

        # Collect per-timestep rows (vectorized)
        abs_time_np = abs_time.numpy().astype(np.float32, copy=False)
        dt_np = np.diff(abs_time_np, prepend=abs_time_np[:1])
        dt_np[0] = 0.0
        terminal_flags = np.zeros(t_seq, dtype=bool)
        if t_seq > 0:
            terminal_flags[-1] = True
        for t in range(t_seq):
            meta_rows.append({
                "hadm_id": hid,
                "t_index": int(t),
                "abs_time": float(abs_time_np[t]),
                "t_post": float(abs_time_np[t]),
                "delta_time": float(max(0.0, dt_np[t])),
                "is_terminal": bool(terminal_flags[t]),
                "raw_set": "",
            })
        for name in _viz__LATENT_SOURCES:
            if name not in item:
                raise RuntimeError(f"Canonical probe cache item is missing {name!r}.")
            vec = item[name].float().reshape(t_seq, -1)
            latent_bufs[name].append(vec.numpy().astype(np.float32, copy=False))
        if raw_ds is not None:
            raw_idx = raw_by_id.get(hid)
            if raw_idx is None:
                raise RuntimeError(f"Raw processed sample for canonical admission {hid} was not found.")
            raw_item = raw_ds[raw_idx]
            state_tokens_flat = raw_item["state_tokens_flat"].detach().cpu().long().reshape(-1)
            state_values_flat = raw_item["state_values_flat"].detach().cpu().float().reshape(-1)
            state_counts = raw_item["state_counts"].detach().cpu().long().reshape(-1)
            state_offsets = raw_item["state_offsets"].detach().cpu().long().reshape(-1)
            if int(state_counts.numel()) != t_seq:
                raise RuntimeError(
                    f"Raw/cache sequence length mismatch for admission {hid}: "
                    f"raw={int(state_counts.numel())} cache={t_seq}."
                )
            if state_offsets.numel() != t_seq + 1:
                raise RuntimeError(
                    f"Raw state offset/count mismatch for admission {hid}: "
                    f"offsets={int(state_offsets.numel())} counts={t_seq}."
                )
            if int(state_offsets[-1]) != int(state_tokens_flat.numel()) or state_tokens_flat.numel() != state_values_flat.numel():
                raise RuntimeError(
                    f"Raw flat state bounds mismatch for admission {hid}: "
                    f"offset_end={int(state_offsets[-1])} tokens={int(state_tokens_flat.numel())} "
                    f"values={int(state_values_flat.numel())}."
                )
            codes = []
            vals = []
            for t in range(t_seq):
                lo = int(state_offsets[t])
                hi = int(state_offsets[t + 1])
                codes.append(state_tokens_flat[lo:hi].numpy().astype(np.int32, copy=False))
                vals.append(state_values_flat[lo:hi].numpy().astype(np.float32, copy=False))
            # meta_rows is timestamp-major, so feature rows must be flattened
            # in exactly the same order (one entry per timestamp), not kept as
            # one nested array per admission.
            state_code_bufs.extend(codes)
            state_val_bufs.extend(vals)
        if include_z_event:
            raw_idx = raw_by_id.get(hid)
            if raw_idx is None or raw_ds is None or teacher is None:
                raise RuntimeError(f"Raw Phase-1 sample for canonical admission {hid} was not found.")
            raw_item = raw_ds[raw_idx]
            raw_batch = nested_padding_collate([raw_item])
            device = next(teacher.parameters()).device
            raw_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw_batch.items()
            }
            with torch.inference_mode():
                token_latents = teacher._encode_event_tokens(
                    raw_batch["state_tokens"], raw_batch["state_values"]
                )
            valid = ~raw_batch["state_pad_mask"]
            pooled = (token_latents * valid.unsqueeze(-1)).sum(dim=2) / valid.sum(dim=2).clamp_min(1).unsqueeze(-1)
            z_event = pooled[0, :t_seq].detach().cpu().numpy().astype(np.float32, copy=False)
            if z_event.shape[0] != t_seq:
                raise RuntimeError(f"Raw/cache sequence length mismatch for admission {hid}.")
            latent_bufs["z_event"].append(z_event)

    # Stack; if a latent source was absent from all items it stays empty.
    latent_arrs: Dict[str, Optional[np.ndarray]] = {}
    for name in source_names:
        pieces = latent_bufs[name]
        if not pieces:
            latent_arrs[name] = None
            continue
        stacked = np.concatenate(pieces, axis=0).astype(np.float32)
        if stacked.shape[0] != len(meta_rows):
            # If some items lacked the field, we cannot align; skip.
            latent_arrs[name] = None
        else:
            latent_arrs[name] = stacked

    if not meta_rows:
        raise RuntimeError(
            f"No data loaded from FULL_VISIBLE/{split} cache. "
            "Run `python -m src.cache_bank --mode full_visible` first."
        )
    if raw_ds is not None and (len(state_code_bufs) != len(meta_rows) or len(state_val_bufs) != len(meta_rows)):
        raise RuntimeError(
            "Raw feature/cache alignment failed: "
            f"feature_rows=({len(state_code_bufs)}, {len(state_val_bufs)}) "
            f"meta_rows={len(meta_rows)}."
        )
    return {
        "latents": latent_arrs,
        "meta": meta_rows,
        "state_code": None,
        "state_val": None,
        "state_code_rows": state_code_bufs,
        "state_val_rows": state_val_bufs,
        "canonical_probe_cache": True,
        "n_patients": int(len(hadm_ids_order)),
        "n_points": int(len(meta_rows)),
    }

def _viz_join_labels(
    meta: List[Dict[str, Any]],
    labels_csv: Optional[str],
) -> pd.DataFrame:
    """Build a per-point DataFrame; join outcome labels if labels_csv is provided."""
    df = pd.DataFrame(meta)
    if not labels_csv:
        df["mortality"] = np.nan
        df["readmission"] = np.nan
        df["los_days"] = np.nan
        return df
    ldf = load_labels_csv(labels_csv)
    required = {"hadm_id"}
    missing = required - set(ldf.columns)
    if missing:
        raise RuntimeError(f"labels.csv missing columns: {sorted(missing)}")
    ldf["hadm_id"] = ldf["hadm_id"].astype(str)
    df["hadm_id"] = df["hadm_id"].astype(str)
    for col, alias in [
        ("label_mortality", "mortality"),
        ("label_readmission_30d", "readmission"),
        ("label_los_days", "los_days"),
    ]:
        if col in ldf.columns:
            ldf = ldf.rename(columns={col: alias})
    keep_cols = ["hadm_id"] + [c for c in ["mortality", "readmission", "los_days", "HCUP"] if c in ldf.columns]
    merged = df.merge(ldf[keep_cols].drop_duplicates("hadm_id"), on="hadm_id", how="left")
    for col in ["mortality", "readmission", "los_days"]:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged

def _viz__parse_hcup_indices(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    txt = str(raw).strip()
    if not txt or txt.lower() in {"nan", "none", "null"}:
        return []
    try:
        vals = json.loads(txt)
    except Exception:
        return []
    if not isinstance(vals, list):
        return []
    out: List[int] = []
    seen = set()
    for v in vals:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out

def _viz__load_hcup_label_names() -> Dict[int, str]:
    return dict(_viz__HCUP_26_LABELS)

def _viz_load_teacher(
    phase1_ckpt: str,
    schema_dir: str,
    device: torch.device,
) -> JEPAModel:
    teacher = JEPAModel.load_from_checkpoint(
        phase1_ckpt,
        map_location=device,
        schema_path=os.path.join(schema_dir, "schema_processed.csv"),
        weights_only=False,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

def _viz_run_latent_visualization(
    args: argparse.Namespace,
    output_dir: str,
    bank: Dict[str, Any],
    df_meta: pd.DataFrame,
) -> Dict[str, Any]:
    # Keep this analysis mode self-contained and flat.  Source/mode prefixes
    # make the files unambiguous without requiring nested figures/tables
    # directories that are difficult to inspect on shared filesystems.
    figs_dir = _viz_ensure_dir(output_dir)
    tables_dir = _viz_ensure_dir(output_dir)
    if HDBSCAN is None:
        raise RuntimeError(
            f"HDBSCAN is required for latent_visualization: {_HDBSCAN_IMPORT_ERROR}"
        )
    if "mortality" not in df_meta.columns:
        raise RuntimeError("labels_csv with mortality is required for latent_visualization.")
    if not df_meta["mortality"].notna().any():
        raise RuntimeError("labels_csv with non-null mortality labels is required for latent_visualization.")
    z_set = bank["latents"].get("z_set")
    z_context = bank["latents"].get("z_context")
    if z_set is None or z_context is None:
        raise RuntimeError("Canonical FULL_VISIBLE probe cache must contain z_set and z_context.")

    point_df = df_meta.copy()
    point_df["row_idx"] = np.arange(len(point_df), dtype=np.int64)
    point_df = point_df[point_df["mortality"].isin([0, 1])].copy()
    point_df["mortality"] = point_df["mortality"].astype(int)
    if point_df.empty:
        raise RuntimeError("No mortality-labeled points found for latent_visualization.")

    ridx_all = point_df["row_idx"].to_numpy(dtype=np.int64)
    z_set_all = z_set[ridx_all]
    z_context_all = z_context[ridx_all]
    state_code_rows_all = bank.get("state_code_rows")
    state_val_rows_all = bank.get("state_val_rows")
    n_all = int(z_set_all.shape[0])

    # All downstream statistics and clustering use the complete labeled
    # cohort. max_points controls only the rendered/display subset.
    rng = np.random.default_rng(int(args.seed))
    max_pts = int(args.max_points) if int(args.max_points) > 0 else n_all
    if n_all > max_pts:
        display_idx = np.sort(rng.choice(n_all, size=max_pts, replace=False))
    else:
        display_idx = np.arange(n_all, dtype=np.int64)

    analysis_df = point_df.reset_index(drop=True)
    z_set_analysis = z_set_all
    z_context_analysis = z_context_all
    state_code_rows_analysis = ([state_code_rows_all[int(i)] for i in ridx_all]
                                if state_code_rows_all else None)
    state_val_rows_analysis = ([state_val_rows_all[int(i)] for i in ridx_all]
                               if state_val_rows_all else None)
    sel = analysis_df.iloc[display_idx].copy().reset_index(drop=True)
    z_set_sel = z_set_analysis[display_idx]
    z_context_sel = z_context_analysis[display_idx]
    selected_latents: Dict[str, np.ndarray] = {
        "z_set": z_set_analysis,
        "z_context": z_context_analysis,
    }
    state_code_rows_sel = state_code_rows_analysis
    state_val_rows_sel = state_val_rows_analysis
    n = int(z_set_sel.shape[0])
    if n < 10:
        raise RuntimeError(f"Too few display points for latent_visualization: n={n}")

    # This is the distance to the last observed timestamp, not a measured
    # death time. Do not label it as a clinical death-time variable.
    death_rel = np.full((n,), np.nan, dtype=np.float32)
    if "abs_time" in sel.columns:
        for pid, g in sel.groupby("hadm_id"):
            idx = g.index.to_numpy(dtype=np.int64)
            mort = int(g["mortality"].iloc[0]) if "mortality" in g.columns else 0
            if mort != 1:
                continue
            t_end = float(g["abs_time"].max())
            death_rel[idx] = (t_end - g["abs_time"].to_numpy(dtype=np.float32)).astype(np.float32)
    sel["last_observation_rel_min"] = death_rel
    hcup_name_map = _viz__load_hcup_label_names()
    cluster_summary_rows: List[Dict[str, Any]] = []
    time_sensitivity_rows: List[Dict[str, Any]] = []
    fingerprint_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []

    def _save_static_panel_2x2(
        panel_rows: Dict[str, Dict[str, np.ndarray]],
        dead_mask: np.ndarray,
        last_observation_rel_min: np.ndarray,
    ) -> None:
        row_order = ["z_set", "z_context"] + (["z_event"] if "z_event" in panel_rows else [])
        row_title = {k: _viz__display_source_label(k) for k in row_order}
        fig, axes = plt.subplots(len(row_order), 2, figsize=(8.0, 4.0 * len(row_order)), dpi=300, squeeze=False)
        fig.patch.set_facecolor("white")
        palette = matplotlib.colormaps.get_cmap("tab20").resampled(20)
        for r, src in enumerate(row_order):
            if src not in panel_rows:
                continue
            emb = panel_rows[src]["emb"]
            labs = panel_rows[src]["cluster"]
            # Cluster view
            axc = axes[r, 0]
            uniq = sorted(np.unique(labs).tolist())
            for i, cid in enumerate(uniq):
                m = labs == cid
                clr = "#bdbdbd" if int(cid) == -1 else palette(int(i % 20))
                axc.scatter(emb[m, 0], emb[m, 1], s=4, c=[clr], alpha=0.85, linewidths=0)
            axc.set_title(f"{row_title[src]} — Cluster", fontsize=8)
            # Mortality view
            axm = axes[r, 1]
            alive = ~dead_mask
            if np.any(alive):
                axm.scatter(emb[alive, 0], emb[alive, 1], s=4, c="#c7c7c7", alpha=0.55, linewidths=0)
            if np.any(dead_mask):
                sc = axm.scatter(
                    emb[dead_mask, 0], emb[dead_mask, 1], s=5,
                    c=last_observation_rel_min[dead_mask], cmap="Reds_r", alpha=0.9, linewidths=0,
                )
                if r == 1:
                    cbar = fig.colorbar(sc, ax=axm, fraction=0.046, pad=0.02)
                    cbar.ax.tick_params(labelsize=6)
                    cbar.set_label("last_observation_rel_min", fontsize=6)
            axm.set_title(f"{row_title[src]} — Mortality", fontsize=8)
            for c in [0, 1]:
                ax = axes[r, c]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(labelsize=6, width=0.4, length=2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.5)
                ax.spines["bottom"].set_linewidth(0.5)
                if r < 2:
                    ax.set_xticklabels([])
                if c == 1:
                    ax.set_yticklabels([])
        fig.tight_layout()
        out_base = os.path.join(figs_dir, "latent_visualization_2x2")
        fig.savefig(out_base + ".png", format="png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _cluster_fingerprints(cluster_labels: np.ndarray, src_name: str) -> Dict[str, Any]:
        uniq = sorted([int(x) for x in np.unique(cluster_labels).tolist() if int(x) != -1])
        out: Dict[str, Any] = {}
        source_latents = selected_latents[src_name]
        global_mean = np.mean(source_latents, axis=0, dtype=np.float64)
        global_std = np.std(source_latents, axis=0, dtype=np.float64)
        global_std = np.maximum(global_std, 1e-8)
        schema_name_map: Dict[int, str] = {}
        schema_type_map: Dict[int, str] = {}
        schema_path = os.path.join(str(args.schema_dir), "schema_processed.csv")
        if os.path.exists(schema_path):
            sdf = pd.read_csv(schema_path)
            for _, row in sdf.iterrows():
                try:
                    vid = int(row.get("vid"))
                except Exception:
                    continue
                schema_name_map[vid] = str(row.get("normalized_concept_label", f"vid_{vid}"))
                schema_type_map[vid] = str(row.get("model_type", "")).strip().lower()
        for cid in uniq:
            m = cluster_labels == cid
            if not np.any(m):
                continue
            cluster_mean = np.mean(source_latents[m], axis=0, dtype=np.float64)
            shift = (cluster_mean - global_mean) / global_std
            top_dims = np.argsort(np.abs(shift))[::-1][:10]
            admission_labels = analysis_df.loc[m, ["hadm_id", "mortality"]].drop_duplicates("hadm_id")
            admission_mortality = float(admission_labels["mortality"].mean())
            record = {
                "size": int(np.sum(m)),
                "timestamp_count": int(np.sum(m)),
                "unique_admission_count": int(analysis_df.loc[m, "hadm_id"].nunique()),
                "mortality_prevalence": admission_mortality,
                "mean_abs_latent": float(np.mean(np.abs(source_latents[m]))),
                "mean_time_min": float(np.mean(analysis_df.loc[m, "abs_time"].to_numpy(dtype=np.float32))),
                # Packed probe caches do not retain raw VID names. This is a
                # directly reproducible latent fingerprint: dimensions with
                # the largest standardized shift from the full cohort.
                "feature_fingerprint": {
                    "type": "latent_dimension_shift",
                    "source": src_name,
                    "raw_vid_names_available": False,
                    "interpretation": "cluster_characterization_not_feature_attribution",
                    "top_dimensions": [
                        {
                            "dimension": int(dim),
                            "cluster_mean": float(cluster_mean[dim]),
                            "global_mean": float(global_mean[dim]),
                            "standardized_shift": float(shift[dim]),
                        }
                        for dim in top_dims
                    ],
                },
            }
            if state_code_rows_sel is not None and state_val_rows_sel is not None:
                feature_records: List[Dict[str, Any]] = []
                for point_idx in np.where(m)[0].tolist():
                    for codes, vals in zip(state_code_rows_sel[point_idx], state_val_rows_sel[point_idx]):
                        valid = (codes > 0) & np.isfinite(vals)
                        for vid, value in zip(codes[valid].tolist(), vals[valid].tolist()):
                            feature_records.append({
                                "hadm_id": str(analysis_df.iloc[point_idx]["hadm_id"]),
                                "vid": int(vid),
                                "value": float(value),
                            })
                top_features: List[Dict[str, Any]] = []
                if feature_records:
                    feature_df = pd.DataFrame(feature_records)
                    cluster_unique_admission_count = int(analysis_df.loc[m, "hadm_id"].nunique())
                    support_threshold = max(20, int(math.ceil(0.01 * cluster_unique_admission_count)))
                    for vid, group in feature_df.groupby("vid", sort=False):
                        feature_type = schema_type_map.get(int(vid), "")
                        event_count = int(len(group))
                        unique_admission_count = int(group["hadm_id"].nunique())
                        prevalence = float(event_count / max(1, len(feature_records)))
                        admission_prevalence = float(unique_admission_count / max(1, cluster_unique_admission_count))
                        mean_value = float(group["value"].mean())
                        numeric = feature_type == "numeric"
                        support_eligible = unique_admission_count >= support_threshold
                        exclusion_reasons: List[str] = []
                        if not numeric:
                            exclusion_reasons.append("non_numeric_feature")
                        if not support_eligible:
                            exclusion_reasons.append("below_cluster_support_threshold")
                        row = {
                            "source": src_name,
                            "cluster_id": int(cid),
                            "vid": int(vid),
                            "feature_name": schema_name_map.get(int(vid), f"vid_{int(vid)}"),
                            "feature_type": feature_type,
                            "mean_value": mean_value,
                            "mean_standardized_value": mean_value,
                            "event_count_within_cluster": event_count,
                            "unique_admission_count_within_cluster": unique_admission_count,
                            "cluster_unique_admission_count": cluster_unique_admission_count,
                            "admission_fraction_within_cluster": admission_prevalence,
                            "event_fraction_within_cluster": prevalence,
                            "support_threshold_unique_admissions": support_threshold,
                            "support_eligible": bool(support_eligible),
                            "eligible": bool(numeric and support_eligible),
                            "exclusion_reason": ";".join(exclusion_reasons),
                            "abs_mean_standardized_value": abs(mean_value),
                            "eligible_rank": None,
                        }
                        fingerprint_rows.append(row)
                        if numeric and support_eligible:
                            top_features.append({
                                "vid": int(vid),
                                "name": schema_name_map.get(int(vid), f"vid_{int(vid)}"),
                                "type": feature_type,
                                "mean_value": mean_value,
                                "event_prevalence": prevalence,
                                "admission_prevalence": admission_prevalence,
                                "significance": abs(mean_value),
                                "event_count_within_cluster": event_count,
                                "unique_admission_count_within_cluster": unique_admission_count,
                                "cluster_unique_admission_count": cluster_unique_admission_count,
                                "support_threshold_unique_admissions": support_threshold,
                            })
                    top_features.sort(key=lambda item: (-float(item["significance"]), int(item["vid"])))
                    eligible_rank = {
                        int(item["vid"]): rank
                        for rank, item in enumerate(top_features, start=1)
                    }
                    for row in fingerprint_rows:
                        if row["source"] == src_name and row["cluster_id"] == int(cid):
                            row["eligible_rank"] = eligible_rank.get(int(row["vid"]))
                record["feature_fingerprint"] = {
                    "type": "observed_vid",
                    "source": src_name,
                    "raw_vid_names_available": True,
                    "interpretation": "cluster_characterization_not_feature_attribution",
                    "support_rule": "unique_admission_count_within_cluster >= max(20, ceil(0.01 * cluster_unique_admission_count))",
                    "ranking_scope": "all observed VIDs within each cluster before support filtering",
                    "top_features": top_features[:12],
                }
            cluster_summary_rows.append({
                "source": src_name,
                "cluster_id": int(cid),
                "timestamp_count": int(np.sum(m)),
                "unique_admission_count": int(analysis_df.loc[m, "hadm_id"].nunique()),
                "mortality_prevalence": admission_mortality,
                "mean_time_min": float(np.mean(analysis_df.loc[m, "abs_time"].to_numpy(dtype=np.float32))),
                "top_vids": ",".join(
                    str(item["vid"])
                    for item in record["feature_fingerprint"].get("top_features", [])[:10]
                ),
            })
            out[str(cid)] = record
        _viz_save_json(os.path.join(tables_dir, f"{src_name}_cluster_fingerprints.json"), out)
        return out

    def _run_topo_clustering_oldstyle(z: np.ndarray) -> np.ndarray:
        n_pts = int(z.shape[0])
        if n_pts < 5:
            return np.zeros((n_pts,), dtype=np.int64)
        rng_local = np.random.default_rng(int(args.seed))
        skel_n = min(n_pts, 8000)
        if skel_n < n_pts:
            skel_idx = np.sort(rng_local.choice(n_pts, size=skel_n, replace=False))
            z_skel = z[skel_idx]
        else:
            skel_idx = np.arange(n_pts, dtype=np.int64)
            z_skel = z

        base_min_cluster_size = max(20, int(np.sqrt(max(1, skel_n))))
        base_min_samples = 10
        labels = HDBSCAN(
            min_cluster_size=base_min_cluster_size,
            min_samples=base_min_samples,
            metric="euclidean",
        ).fit_predict(z_skel).astype(np.int64)

        # Recursive split for giant clusters (old-style behavior)
        next_label = int(np.max(labels)) + 1 if np.any(labels >= 0) else 0
        max_refine_rounds = 3
        split_min_size = max(200, int(0.05 * skel_n))
        for _ in range(max_refine_rounds):
            did_split = False
            uniq = [int(c) for c in np.unique(labels).tolist() if int(c) >= 0]
            for cid in uniq:
                c_idx = np.where(labels == cid)[0]
                c_n = int(c_idx.size)
                if c_n < split_min_size:
                    continue
                sub_min_cluster_size = max(20, int(np.sqrt(max(1, c_n)) * 0.8))
                sub = HDBSCAN(
                    min_cluster_size=sub_min_cluster_size,
                    min_samples=5,
                    metric="euclidean",
                ).fit_predict(z_skel[c_idx]).astype(np.int64)
                sub_uniq = [int(x) for x in np.unique(sub).tolist() if int(x) >= 0]
                if len(sub_uniq) <= 1:
                    continue
                did_split = True
                for sc in sub_uniq:
                    m = sub == sc
                    labels[c_idx[m]] = next_label
                    next_label += 1
                # sub-noise keeps original parent label for stability
            if not did_split:
                break

        # Project from skeleton to full data via KNN
        valid = labels >= 0
        if np.any(valid):
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(z_skel[valid], labels[valid])
            if skel_n < n_pts:
                all_labels = np.full((n_pts,), -1, dtype=np.int64)
                all_labels[skel_idx] = labels
                non_skeleton = np.ones((n_pts,), dtype=bool)
                non_skeleton[skel_idx] = False
                all_labels[non_skeleton] = knn.predict(z[non_skeleton]).astype(np.int64)
            else:
                # Preserve HDBSCAN noise labels when every point is in the
                # skeleton; noise must not be silently converted to a cluster.
                all_labels = labels.copy()
        else:
            all_labels = np.full((n_pts,), -1, dtype=np.int64)

        # Remap labels by descending cluster size for deterministic colors
        uniq_full = [int(c) for c in np.unique(all_labels).tolist() if int(c) >= 0]
        counts = [(cid, int(np.sum(all_labels == cid))) for cid in uniq_full]
        counts.sort(key=lambda x: x[1], reverse=True)
        remap = {cid: i for i, (cid, _) in enumerate(counts)}
        out = np.asarray([
            -1 if int(c) == -1 else remap[int(c)]
            for c in all_labels.tolist()
        ], dtype=np.int64)
        return out

    def _run_one_source(source_name: str, z: np.ndarray) -> Dict[str, Any]:
        selected_latents[source_name] = z
        cluster_labels_full = _run_topo_clustering_oldstyle(z)
        assignment_rows.extend(
            {
                "source": source_name,
                "hadm_id": str(row["hadm_id"]),
                "t_index": int(row["t_index"]),
                "abs_time": float(row["abs_time"]),
                "cluster_id": int(cid),
            }
            for row, cid in zip(analysis_df.to_dict("records"), cluster_labels_full.tolist())
        )
        _cluster_fingerprints(cluster_labels_full, source_name)
        z_display = z[display_idx]
        cluster_labels = cluster_labels_full[display_idx]
        palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
        cluster_colors = [
            ("#bdbdbd" if int(c) == -1 else palette[int(c) % len(palette)])
            for c in cluster_labels
        ]

        z_unique, inv = _viz__unique_rows_with_inverse(z_display)
        if z_unique.shape[0] < 2:
            raise RuntimeError(f"Too few unique points for projection in {source_name}: {z_unique.shape[0]}")

        tsne_perp = float(max(5, min(30, n - 1)))
        tsne_perp_u = float(max(1, min(tsne_perp, z_unique.shape[0] - 1)))
        tsne_xy_u = TSNE(
            n_components=2,
            perplexity=tsne_perp_u,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=int(args.seed),
        ).fit_transform(z_unique).astype(np.float32)
        tsne_xy = tsne_xy_u[inv]
        umap_neighbors = max(2, min(int(args.umap_n_neighbors), int(z_unique.shape[0] - 1)))
        umap_xy_u = _viz_fit_umap_2d(
            z_unique, umap_neighbors, float(args.umap_min_dist), int(args.seed)
        )
        umap_xy = umap_xy_u[inv]
        projections = [
            ("tsne_2d", "t-SNE", tsne_xy),
            ("umap_2d", "UMAP", umap_xy),
        ]

        hadm_hcup: Dict[str, List[int]] = {}
        if "HCUP" in sel.columns:
            for _, row in sel[["hadm_id", "HCUP"]].drop_duplicates("hadm_id").iterrows():
                hadm_hcup[str(row["hadm_id"])] = _viz__parse_hcup_indices(row["HCUP"])
        all_hcup = list(range(26))

        for out_name, proj_name, emb in projections:
            fig = go.Figure()
            dead_mask = sel["mortality"].to_numpy(dtype=np.int64) == 1
            alive_mask = ~dead_mask
            hover_raw = sel["raw_set"].fillna("").astype(str).tolist() if "raw_set" in sel.columns else [""] * n
            # Mortality view: non-death gray, death cases colored by time since
            # their last observed timestamp (not a measured death time).
            if np.any(alive_mask):
                fig.add_trace(
                    go.Scattergl(
                        x=emb[alive_mask, 0], y=emb[alive_mask, 1], mode="markers",
                        marker=dict(size=4, color="lightgray", opacity=0.7),
                        name="No Death",
                        text=[hover_raw[i] for i in np.where(alive_mask)[0]],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
            if np.any(dead_mask):
                fig.add_trace(
                    go.Scattergl(
                        x=emb[dead_mask, 0], y=emb[dead_mask, 1], mode="markers",
                        marker=dict(
                            size=5,
                            color=sel.loc[dead_mask, "last_observation_rel_min"].to_numpy(dtype=np.float32),
                            colorscale="Reds_r",
                            showscale=True,
                            colorbar=dict(title="last_observation_rel_min"),
                            opacity=0.9,
                        ),
                        name="Death (colored by last_observation_rel_min)",
                        text=[hover_raw[i] for i in np.where(dead_mask)[0]],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
            n_mort = len(fig.data)

            # Cluster view
            fig.add_trace(
                go.Scattergl(
                    x=emb[:, 0], y=emb[:, 1], mode="markers", visible=False,
                    marker=dict(size=4, color=cluster_colors),
                    name="Cluster Topology",
                    text=hover_raw,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            idx_cluster = len(fig.data) - 1

            # HCUP view (default mortality-like base + per-label toggles via buttons)
            hcup_base_v1 = ["lightgray"] * n
            hcup_base_v2 = ["lightgray"] * n
            for i, hid in enumerate(sel["hadm_id"].astype(str).tolist()):
                hs = hadm_hcup.get(hid, [])
                if hs:
                    hcup_base_v1[i] = "steelblue"
                    hcup_base_v2[i] = "steelblue"
            fig.add_trace(
                go.Scattergl(
                    x=emb[:, 0], y=emb[:, 1], mode="markers", visible=False,
                    marker=dict(size=4, color=hcup_base_v1, opacity=0.8),
                    name="HCUP",
                    text=hover_raw,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            idx_hcup = len(fig.data) - 1

            total = len(fig.data)
            vis_mort = [False] * total
            for i in range(n_mort):
                vis_mort[i] = True
            vis_cluster = [False] * total
            vis_cluster[idx_cluster] = True
            vis_hcup = [False] * total
            vis_hcup[idx_hcup] = True

            buttons = [
                {"label": "Mortality View", "method": "update", "args": [{"visible": vis_mort}]},
                {"label": "Cluster View", "method": "update", "args": [{"visible": vis_cluster}]},
                {"label": "HCUP View", "method": "update", "args": [{"visible": vis_hcup}]},
            ]
            # HCUP per-label color toggles (separate dropdown), applied to HCUP trace only.
            hcup_dropdown_buttons: List[Dict[str, Any]] = []
            hcup_palette = px.colors.qualitative.Alphabet
            for i, h in enumerate(all_hcup):
                c_hit = hcup_palette[i % len(hcup_palette)]
                c_miss = "lightgray"
                col = []
                for hid in sel["hadm_id"].astype(str).tolist():
                    hs = hadm_hcup.get(hid, [])
                    col.append(c_hit if h in hs else c_miss)
                label_name = hcup_name_map.get(int(h), f"HCUP_{h}")
                hcup_dropdown_buttons.append(
                    {
                        "label": label_name,
                        "method": "restyle",
                        "args": [{"marker.color": [col]}, [idx_hcup]],
                    }
                )

            fig.update_layout(
                title=f"Latent State Alignment ({_viz__display_source_label(source_name)}, {proj_name})",
                template="plotly_white",
                updatemenus=[
                    {
                        "type": "buttons",
                        "direction": "down",
                        "x": 1.02,
                        "y": 1.0,
                        "xanchor": "left",
                        "yanchor": "top",
                        "buttons": buttons,
                    },
                    {
                        "type": "dropdown",
                        "direction": "up",
                        "x": 1.02,
                        "y": 0.02,
                        "xanchor": "left",
                        "yanchor": "bottom",
                        "showactive": True,
                        "buttons": hcup_dropdown_buttons,
                        "name": "HCUP VIEW",
                    },
                ],
                legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(r=230),
            )
            _viz_save_plotly_html(fig, os.path.join(figs_dir, f"{source_name}_{out_name}.html"))
            df_projection = sel.copy()
            df_projection["x"] = emb[:, 0]
            df_projection["y"] = emb[:, 1]
            df_projection["cluster"] = cluster_labels
            # CSV is sufficient and avoids duplicating the same point table in
            # a large parquet artifact.
            df_projection.to_csv(
                os.path.join(tables_dir, f"{source_name}_{out_name}_points.csv"), index=False
            )
        # Cheap sensitivity audit: reuse the same high-dimensional clusters
        # and stratify their admission-level mortality enrichment by time.
        time_bins = pd.cut(
            sel["abs_time"], bins=[-np.inf, 1440.0, 2880.0, np.inf],
            labels=["0_24h", "24_48h", ">48h"], right=True,
        )
        time_bins = pd.cut(
            analysis_df["abs_time"], bins=[-np.inf, 1440.0, 2880.0, np.inf],
            labels=["0_24h", "24_48h", ">48h"], right=True,
        )
        for cid in sorted(np.unique(cluster_labels_full).tolist()):
            for time_bin in ["0_24h", "24_48h", ">48h"]:
                mask = (cluster_labels_full == cid) & (time_bins.to_numpy(dtype=object) == time_bin)
                if not np.any(mask):
                    continue
                admissions = analysis_df.loc[mask, ["hadm_id", "mortality"]].drop_duplicates("hadm_id")
                time_sensitivity_rows.append({
                    "source": source_name,
                    "cluster_id": int(cid),
                    "time_bin": time_bin,
                    "timestamp_count": int(np.sum(mask)),
                    "unique_admission_count": int(len(admissions)),
                "mortality_prevalence": float(admissions["mortality"].mean()),
                })
        return {
            "source": _viz__display_source_label(source_name),
            "source_raw": source_name,
            "clusters": int(len([c for c in np.unique(cluster_labels_full).tolist() if int(c) != -1])),
            "noise_points": int(np.sum(cluster_labels_full == -1)),
            "cluster_fingerprint_path": f"{source_name}_cluster_fingerprints.json",
            "hdbscan": {
                "style": "old_recursive_with_knn_projection",
            },
            "_tsne_emb": tsne_xy,
            "_cluster_labels": cluster_labels,
        }

    per_source = [
        _run_one_source("z_set", z_set_analysis),
        _run_one_source("z_context", z_context_analysis),
    ]
    if "z_event" in bank["latents"] and bank["latents"]["z_event"] is not None:
        z_event = bank["latents"]["z_event"][ridx_all]
        per_source.append(_run_one_source("z_event", z_event))
    panel_rows: Dict[str, Dict[str, np.ndarray]] = {}
    for item in per_source:
        # `source` is the human-readable display label (e.g. ATLAS-EHR
        # (z_set)); the panel lookup uses canonical source keys.
        panel_rows[str(item["source_raw"])] = {
            "emb": np.asarray(item.pop("_tsne_emb")),
            "cluster": np.asarray(item.pop("_cluster_labels")),
        }
    dead_mask_panel = sel["mortality"].to_numpy(dtype=np.int64) == 1
    death_rel_panel = sel["last_observation_rel_min"].to_numpy(dtype=np.float32)
    _save_static_panel_2x2(panel_rows, dead_mask_panel, death_rel_panel)
    fingerprint_table = pd.DataFrame(fingerprint_rows)
    fingerprint_table_path = os.path.join(tables_dir, "latent_cluster_vid_fingerprints.parquet")
    fingerprint_table.to_parquet(fingerprint_table_path, index=False)
    pd.DataFrame(assignment_rows).to_parquet(
        os.path.join(tables_dir, "latent_cluster_assignments.parquet"), index=False
    )
    fingerprint_qc_rows = []
    for (source_name, cluster_id), group in fingerprint_table.groupby(["source", "cluster_id"], sort=True):
        numeric = group[group["feature_type"].astype(str).str.lower().eq("numeric")]
        eligible = numeric[numeric["support_eligible"]].sort_values(
            ["abs_mean_standardized_value", "vid"], ascending=[False, True]
        )
        fingerprint_qc_rows.append({
            "source": source_name,
            "cluster_id": int(cluster_id),
            "total_observed_numeric_vids": int(len(numeric)),
            "support_eligible_numeric_vids": int(len(eligible)),
            "top_10_eligible_vids": ",".join(str(int(v)) for v in eligible["vid"].head(10)),
            "top_10_mean_standardized_values": ",".join(f"{float(v):.8g}" for v in eligible["mean_standardized_value"].head(10)),
            "top_10_unique_admission_counts": ",".join(str(int(v)) for v in eligible["unique_admission_count_within_cluster"].head(10)),
            "top_10_admission_fractions": ",".join(f"{float(v):.8g}" for v in eligible["admission_fraction_within_cluster"].head(10)),
            "top_10_details": [
                {
                    "vid": int(v["vid"]),
                    "feature_name": str(v["feature_name"]),
                    "mean_standardized_value": float(v["mean_standardized_value"]),
                    "unique_admission_count": int(v["unique_admission_count_within_cluster"]),
                    "admission_fraction": float(v["admission_fraction_within_cluster"]),
                }
                for _, v in eligible.head(10).iterrows()
            ],
        })
    pd.DataFrame(fingerprint_qc_rows).to_json(
        os.path.join(tables_dir, "latent_cluster_fingerprint_qc.json"), orient="records", indent=2
    )
    print("latent fingerprint QC: support filtering precedes top-k truncation")
    for row in fingerprint_qc_rows:
        if row["source"] == "z_context":
            print(json.dumps({k: v for k, v in row.items() if k != "top_10_details"}, sort_keys=True))
            for detail in row["top_10_details"]:
                print(json.dumps({"source": row["source"], "cluster_id": row["cluster_id"], **detail}, sort_keys=True))
    cluster_summary_columns = [
        "source", "cluster_id", "timestamp_count", "unique_admission_count",
        "mortality_prevalence", "mean_time_min", "top_vids",
    ]
    pd.DataFrame(cluster_summary_rows, columns=cluster_summary_columns).to_csv(
        os.path.join(tables_dir, "cluster_summary.csv"), index=False
    )
    time_sensitivity_columns = [
        "source", "cluster_id", "time_bin", "timestamp_count",
        "unique_admission_count", "mortality_prevalence",
    ]
    pd.DataFrame(time_sensitivity_rows, columns=time_sensitivity_columns).to_csv(
        os.path.join(tables_dir, "cluster_time_sensitivity.csv"), index=False
    )

    result = {
        "section": "latent_visualization",
        "n_points": int(n_all),
        "n_display_points": int(n),
        "max_patients": int(args.max_patients),
        "formal_run": bool(str(args.split) == "test" and int(args.max_patients) == 0),
        "run_scope": "formal_full_dataset" if str(args.split) == "test" and int(args.max_patients) == 0 else "non_formal_subset_or_split",
        "feature_sources": [_viz__display_source_label(str(item["source_raw"])) for item in per_source],
        "feature_fingerprint_status": (
            "observed_vid_from_processed_data"
            if state_code_rows_sel is not None else
            "latent_dimension_only_without_data_dir"
        ),
        "projection_modes": ["tsne_2d", "umap_2d"],
        "clustering_space": "original_latent_space",
        "projection_space": "t-SNE_2D_display_only",
        "fingerprint_interpretation": "cluster_characterization_not_feature_attribution",
        "views": ["mortality(last_observation_rel_min)", "cluster", "hcup"],
        "cluster_summary_csv": "cluster_summary.csv",
        "cluster_time_sensitivity_csv": "cluster_time_sensitivity.csv",
        "cluster_vid_fingerprints_parquet": "latent_cluster_vid_fingerprints.parquet",
        "cluster_vid_fingerprint_qc_json": "latent_cluster_fingerprint_qc.json",
        "fingerprint_support_rule": "unique_admission_count_within_cluster >= max(20, ceil(0.01 * cluster_unique_admission_count))",
        "fingerprint_ranking_scope": "all observed VIDs within each cluster before support filtering",
        "per_source": per_source,
        "static_panel_2x2": "latent_visualization_2x2.png",
        "timestamp": _viz_now_iso(),
    }
    _viz_save_json(os.path.join(output_dir, "latent_visualization_summary.json"), result)
    return result

def _viz_run_temporal_embeddings(
    args: argparse.Namespace,
    output_dir: str,
    teacher: JEPAModel,
) -> Dict[str, Any]:
    """Visualize Phase-1 signed relative-time embeddings on a synthetic grid.

    Grid: integer minutes 1..max_time_int on the same device as the model.
    Color: time value (log scale for readability).
    """
    figs_dir = _viz_ensure_dir(os.path.join(output_dir, "figures", "temporal_embeddings"))
    tables_dir = _viz_ensure_dir(os.path.join(output_dir, "tables", "temporal_embeddings"))

    device = next(teacher.parameters()).device
    t_vals = torch.arange(1, int(args.max_time_int) + 1, dtype=torch.float32).unsqueeze(-1).to(device)
    t_np = t_vals.squeeze(-1).cpu().numpy()  # (T,) for coloring

    panel_data: Dict[str, pd.DataFrame] = {}

    def _embed_and_plot(module: torch.nn.Module, module_name: str, source_label: str) -> Dict[str, Any]:
        raw = module(t_vals)  # (T, D)
        emb = torch.nn.functional.normalize(raw, p=2, dim=-1)
        emb_np = emb.cpu().numpy().astype(np.float32)
        (emb_sub,), idx = _viz_subsample_points([emb_np], args.max_points, args.seed)
        t_sub = t_np[idx]
        coords = _viz_fit_umap_2d(emb_sub, args.umap_n_neighbors, args.umap_min_dist, args.seed)
        df_emb = pd.DataFrame({
            "t_minutes": t_sub,
            "log_t": np.log1p(t_sub),
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        })
        fig = px.scatter(
            df_emb,
            x="umap_x",
            y="umap_y",
            color="log_t",
            color_continuous_scale="viridis",
            hover_data=["t_minutes"],
            title=f"{source_label} — {module_name} (n={len(df_emb)}, max_t={int(args.max_time_int)})",
            opacity=0.8,
            render_mode="webgl",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        safe = module_name.replace(".", "_")
        _viz_save_plotly_html(fig, os.path.join(figs_dir, f"{safe}.html"))
        _viz_save_dataframe(os.path.join(tables_dir, safe), df_emb)
        panel_data[str(module_name)] = df_emb.copy()
        return {"n_time_steps": int(len(emb_np)), "n_subsampled": int(len(emb_sub)), "embed_dim": int(emb_np.shape[1])}

    summaries: Dict[str, Any] = {}

    # Time exists only in per-layer signed relative-time attention.
    rel_layers = getattr(getattr(teacher, "state_backbone", None), "layers", [])
    for idx, layer in enumerate(rel_layers):
        mod = getattr(layer, "relative_time_embedding", None)
        if mod is not None:
            label = f"P1_relative_time_layer{idx + 1}"
            summaries[label] = _embed_and_plot(mod, label, "JEPAModel")

    # Static 1x2 panel for manuscript-friendly export (non-HTML).
    if len(panel_data) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=300)
        fig.patch.set_facecolor("white")
        keys = list(panel_data)[:2]
        panel_spec = [(keys[0], keys[0], "viridis"), (keys[1], keys[1], "plasma")]
        for ax, (k, ttl, cmap) in zip(axes, panel_spec):
            dff = panel_data[k]
            sc = ax.scatter(
                dff["umap_x"].to_numpy(dtype=np.float32),
                dff["umap_y"].to_numpy(dtype=np.float32),
                c=dff["log_t"].to_numpy(dtype=np.float32),
                cmap=cmap,
                s=8,
                alpha=0.85,
                linewidths=0,
            )
            ax.set_title(ttl, fontsize=8)
            ax.grid(False)
            ax.set_facecolor("white")
            ax.tick_params(labelsize=6, width=0.4, length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("log(1+t)", fontsize=6)
        fig.tight_layout()
        fig.savefig(
            os.path.join(figs_dir, "temporal_embeddings_1x2.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
    result = {
        "section": "temporal_embeddings",
        "max_time_int": int(args.max_time_int),
        "embeddings": summaries,
        "static_panel_1x2": os.path.join("figures", "temporal_embeddings", "temporal_embeddings_1x2.png"),
        "timestamp": _viz_now_iso(),
    }
    _viz_save_json(os.path.join(output_dir, "summary", "temporal_embeddings.json"), result)
    return result

_viz__DEFAULT_ALL_MODES = [
    "latent_visualization",
    "temporal_embeddings",
    "task_primary_performance",
    "task_secondary_metrics",
]

_viz__IMPUTATION_MODES = [
    "imputation_overview",
    "imputation_by_feature_group",
    "imputation_mechanism_bins",
    "imputation_all_features",
    "imputation_representative_features",
]

_viz__OPTIONAL_MODES = [
    "icd_long_tail",
    "icd_retrieval_cardinality",
    "drg_support_bins",
    "hcup_all26_per_label",
    "imputation",
    *_viz__IMPUTATION_MODES,
    "setencoder_attention",
]

_viz__PAPER_PRESET_MODES = ["main", "extended", "supplementary", "all_paper"]

_viz__ALL_MODES = _viz__DEFAULT_ALL_MODES + _viz__OPTIONAL_MODES + _viz__PAPER_PRESET_MODES

_viz__FULL_VISIBLE_MODES = frozenset({"latent_visualization"})

_viz__MODEL_MODES = frozenset({"temporal_embeddings"})

def _viz_expand_modes(mode_str: str) -> List[str]:
    if mode_str.strip().lower() == "all":
        return list(_viz__DEFAULT_ALL_MODES)
    if mode_str.strip().lower() == "main":
        return ["task_primary_performance"]
    if mode_str.strip().lower() == "extended":
        return ["icd_long_tail", "icd_retrieval_cardinality", "drg_support_bins", "hcup_all26_per_label",
                *_viz__IMPUTATION_MODES, "setencoder_attention"]
    if mode_str.strip().lower() == "supplementary":
        return ["task_secondary_metrics", "icd_long_tail", "icd_retrieval_cardinality",
                "drg_support_bins", "hcup_all26_per_label",
                *_viz__IMPUTATION_MODES[:-1], "setencoder_attention"]
    if mode_str.strip().lower() == "all_paper":
        # Paper-facing figures only — excludes model-introspection modes that require
        # full_visible cache / model weights (latent_visualization, temporal_embeddings).
        paper_only = [m for m in _viz__DEFAULT_ALL_MODES
                      if m not in ("latent_visualization", "temporal_embeddings")]
        paper_only += list(_viz__OPTIONAL_MODES)
        return paper_only
    parts = [m.strip() for m in mode_str.split(",") if m.strip()]
    for m in parts:
        if m not in _viz__ALL_MODES:
            raise ValueError(f"Unknown mode '{m}'.  Valid: {_viz__ALL_MODES}")
    expanded: List[str] = []
    for m in parts:
        expanded.extend(_viz__IMPUTATION_MODES if m == "imputation" else [m])
    deduped: List[str] = []
    seen: set = set()
    for m in expanded:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped

def _viz_run(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    modes = _viz_expand_modes(str(args.mode))
    if "latent_visualization" in modes:
        if not args.data_dir or not os.path.isdir(str(args.data_dir)):
            raise ValueError("--data_dir must exist when --mode includes latent_visualization.")
    _viz_ensure_dir(os.path.join(args.output_dir, "summary"))

    teacher: Optional[JEPAModel] = None
    if bool(args.include_z_event):
        if not args.phase1_ckpt or not args.data_dir:
            raise ValueError("--include_z_event requires --phase1_ckpt and --data_dir")
        device = torch.device(_viz_resolve_device(str(args.device)))
        teacher = _viz_load_teacher(args.phase1_ckpt, args.schema_dir, device)

    bank: Optional[Dict[str, Any]] = None
    df_meta: Optional[pd.DataFrame] = None
    if _viz__FULL_VISIBLE_MODES.intersection(modes):
        bank = _viz_load_full_visible_bank(
            cache_root=args.cache_root,
            split=args.split,
            max_patients=int(args.max_patients),
            seed=int(args.seed),
            config=str(args.config),
            schema_dir=args.schema_dir,
            teacher=teacher,
            data_dir=args.data_dir,
            include_z_event=bool(args.include_z_event),
        )
        df_meta = _viz_join_labels(bank["meta"], args.labels_csv)

    if _viz__MODEL_MODES.intersection(modes):
        if teacher is None and not args.phase1_ckpt:
            raise ValueError(
                f"--phase1_ckpt is required for modes: {sorted(_viz__MODEL_MODES.intersection(modes))}"
            )
        if teacher is None:
            device = torch.device(_viz_resolve_device(str(args.device)))
            teacher = _viz_load_teacher(args.phase1_ckpt, args.schema_dir, device)
    all_results: Dict[str, Any] = {}

    if "latent_visualization" in modes:
        all_results["latent_visualization"] = _viz_run_latent_visualization(
            args, args.output_dir, bank, df_meta
        )

    if "temporal_embeddings" in modes:
        all_results["temporal_embeddings"] = _viz_run_temporal_embeddings(
            args, args.output_dir, teacher
        )

    if "task_primary_performance" in modes:
        all_results["task_primary_performance"] = _viz_run_task_primary_performance(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "task_secondary_metrics" in modes:
        all_results["task_secondary_metrics"] = _viz_run_task_secondary_metrics(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "icd_long_tail" in modes:
        all_results["icd_long_tail"] = _viz_run_icd_long_tail(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "icd_retrieval_cardinality" in modes:
        all_results["icd_retrieval_cardinality"] = _viz_run_icd_retrieval_cardinality(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "drg_support_bins" in modes:
        all_results["drg_support_bins"] = _viz_run_drg_support_bins(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "hcup_all26_per_label" in modes:
        all_results["hcup_all26_per_label"] = _viz_run_hcup_all26_per_label(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "imputation_overview" in modes:
        all_results["imputation_overview"] = _viz_run_imputation_overview(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "imputation_by_feature_group" in modes:
        all_results["imputation_by_feature_group"] = _viz_run_imputation_by_feature_group(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "imputation_mechanism_bins" in modes:
        all_results["imputation_mechanism_bins"] = _viz_run_imputation_mechanism_bins(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "imputation_all_features" in modes:
        all_results["imputation_all_features"] = _viz_run_imputation_all_features(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "imputation_representative_features" in modes:
        all_results["imputation_representative_features"] = _viz_run_imputation_representative_features(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
        )
    if "setencoder_attention" in modes:
        all_results["setencoder_attention"] = _viz_run_setencoder_attention(
            output_dir=args.output_dir,
            paper_report_dir=str(args.paper_report_dir),
            attribution_dir=str(args.attribution_dir) if args.attribution_dir else None,
        )

    summary = {
        "visualization_version": _viz_VISUALIZATION_VERSION,
        "modes": modes,
        "split": args.split,
        "cache_root": args.cache_root,
        "output_dir": args.output_dir,
        "max_patients": int(args.max_patients),
        "max_points": int(args.max_points),
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "seed": int(args.seed),
        "sections": all_results,
        "timestamp": _viz_now_iso(),
    }
    _viz_save_json(os.path.join(args.output_dir, "summary", "visualization_summary.json"), summary)
    print(json.dumps({"output_dir": args.output_dir, "modes": modes}, indent=2))

def _viz_parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATLAS-EHR visualization")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configs/params.yaml")
    parser.add_argument("--cache_root", type=str, required=True,
                        help="Pre-built cache_bank root directory")
    parser.add_argument("--schema_dir", type=str, required=True,
                        help="Directory containing schema_processed.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for figures/, tables/, summary/")
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="Optional labels CSV for outcome coloring")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Raw processed Phase-1 data directory; required for --include_z_event")
    parser.add_argument("--phase1_ckpt", type=str, default=None,
                        help="Phase 1 checkpoint; required for temporal_embeddings and model-based analysis")
    parser.add_argument("--include_z_event", action="store_true",
                        help="Also visualize token-level Phase-1 z_event pooled per timestamp; requires --data_dir")
    parser.add_argument(
        "--mode", type=str, default="all",
        help=(
            "Comma-separated list of modes to run, or 'all'. "
            f"Valid: {_viz__ALL_MODES}"
        ),
    )
    parser.add_argument("--split", type=str, default="test",
                        help="Cache split to visualize (default: test)")
    parser.add_argument("--max_patients", type=int, default=1000,
                        help="Max patients to load from FULL_VISIBLE (0 = no limit)")
    parser.add_argument("--max_points", type=int, default=10000,
                        help="Max points to pass to UMAP (0 = no limit)")
    parser.add_argument("--max_time_int", type=int, default=5000,
                        help="Max integer time value for temporal_embeddings synthetic grid")
    parser.add_argument("--umap_n_neighbors", type=int, default=30,
                        help="UMAP n_neighbors")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--paper_report_dir", type=str, default="paper_report",
                        help="Consolidated paper report directory containing figure_data/")
    parser.add_argument("--attribution_dir", type=str, default=None,
                        help="Output directory from the set_encoder_attention mode; required for setencoder_attention mode")
    parser.add_argument("--png-only", action="store_true", default=False,
                        help="Skip interactive/HTML outputs; only produce PNGs.")
    return parser.parse_args(argv)

def _viz_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _viz_parse_args(argv)
    _viz_run(args)

def _viz__figure_mm(mm_w: float, mm_h: float) -> Tuple[float, float]:
    return (float(mm_w) / 25.4, float(mm_h) / 25.4)

def _viz__apply_nature_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.8,
        "axes.facecolor": "#fcfcfc",
    })


_viz__TASK_TITLE = {
    "mortality": "Mortality",
    "los_ge_3d": "LOS ≥3d",
    "los_ge_7d": "LOS ≥7d",
    "readmission": "Readmission",
    "icd": "ICD-10",
    "hcup": "HCUP",
    "drg": "DRG",
}




_viz__MODEL_ORDER = ["ATLAS-EHR (z_set)", "ATLAS-EHR (z_context)", "LR", "XGBoost"]

_viz__OUTCOME_TASKS = ["mortality", "los_ge_3d", "los_ge_7d", "readmission"]

_viz__SECONDARY_OUTCOME = ["brier", "ece"]

_viz__SECONDARY_TOPK = ["precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10"]

_viz__SECONDARY_PHENO_FREQ = ["macro_auprc"]

_viz__SECONDARY_DRG = [
    "cardinality_accuracy",
    "top3_hit",
    "top3_recall",
    "top5_hit",
    "top5_recall",
    "exact_set_match",
    "macro_f1",
]

def _viz__read_primary_figure_table(paper_report_dir: str) -> pd.DataFrame:
    path = os.path.join(paper_report_dir, "figure_data", "figure2_primary_metrics.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _viz__read_report_figure_csv(paper_report_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(paper_report_dir, "figure_data", filename)
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _viz__plot_task_primary_png(df: pd.DataFrame, out_png: str) -> None:
    required = {
        "task_group", "task", "metric", "metric_label", "model", "source",
        "point", "baseline_name", "delta_vs_best_baseline", "include_in_main_figure",
    }
    if df.empty or not required.issubset(set(df.columns)):
        fig = plt.figure(figsize=_viz__figure_mm(220, 160))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No primary figure data available.\nCheck paper_report/figure_data/figure2_primary_metrics.csv.",
            ha="center",
            va="center",
            fontsize=10,
        )
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    plot_df = df.copy()
    plot_df["include_in_main_figure"] = plot_df["include_in_main_figure"].astype(bool)
    plot_df = plot_df[plot_df["include_in_main_figure"]].copy()
    plot_df["display_model"] = plot_df.get("display_model", plot_df["model"])
    plot_df["display_model"] = plot_df["display_model"].astype(str)
    atlas_mask = plot_df["display_model"].astype(str).str.contains("ATLAS-EHR", case=False, na=False)
    atlas_df = plot_df[atlas_mask].copy()
    if atlas_df.empty:
        atlas_df = plot_df[plot_df["source"].astype(str).isin(["ATLAS-EHR (z_set)", "ATLAS-EHR (z_context)", "ATLAS-EHR (z_event)"])].copy()
    if atlas_df.empty:
        raise ValueError("Primary figure data does not contain ATLAS-EHR rows.")
    atlas_df = atlas_df.sort_values(["task_group", "task", "metric_label"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for _, atlas_row in atlas_df.iterrows():
        base_name = str(atlas_row.get("baseline_name", "")).strip()
        if not base_name:
            continue
        sub = plot_df[
            (plot_df["task"].astype(str) == str(atlas_row["task"]))
            & (plot_df["metric"].astype(str) == str(atlas_row["metric"]))
            & (plot_df["display_model"].astype(str) == base_name)
        ]
        if sub.empty:
            continue
        baseline_row = sub.iloc[0]
        rows.append({
            "task_group": str(atlas_row["task_group"]),
            "task": str(atlas_row["task"]),
            "metric_label": str(atlas_row["metric_label"]),
            "atlas_point": float(atlas_row["point"]),
            "baseline_point": float(baseline_row["point"]),
            "baseline_name": base_name,
            "delta": float(atlas_row.get("delta_vs_best_baseline", float(atlas_row["point"]) - float(baseline_row["point"]))),
            "notes": str(atlas_row.get("notes", "")),
        })
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise ValueError("Primary figure data does not contain ATLAS/baseline task pairs.")

    group_order = ["outcome", "phenotype", "case_mix"]
    pair_df["task_group_order"] = pair_df["task_group"].map({k: i for i, k in enumerate(group_order)}).fillna(99)
    pair_df["row_label"] = pair_df["task"].map(_viz__TASK_TITLE).fillna(pair_df["task"]) + " | " + pair_df["metric_label"]
    pair_df = pair_df.sort_values(["task_group_order", "task", "metric_label"]).reset_index(drop=True)

    colors = {"ATLAS-EHR": "#0072B2", "baseline": "#666666", "delta_pos": "#009E73", "delta_neg": "#D55E00"}
    fig = plt.figure(figsize=_viz__figure_mm(244, 178))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.82], wspace=0.02)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    y = np.arange(len(pair_df), dtype=np.float64)
    ax_left.set_xlim(0.0, 1.0)
    ax_left.set_ylim(-0.5, len(pair_df) - 0.5)
    ax_left.axis("off")
    current_group = None
    x_task = 0.00
    x_baseline = 0.985
    n_rows = max(1, int(len(pair_df)))
    for idx, row in pair_df.iterrows():
        y_frac = 1.0 - ((float(idx) + 0.5) / float(n_rows))
        if row["task_group"] != current_group:
            current_group = row["task_group"]
            header_frac = min(0.995, y_frac + (0.40 / float(n_rows)))
            ax_left.text(
                x_task,
                header_frac,
                str(current_group).replace("_", " ").title(),
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
                transform=ax_left.transAxes,
            )
        ax_left.text(x_task, y_frac, str(row["row_label"]), fontsize=7, ha="left", va="center", transform=ax_left.transAxes)
        ax_left.text(x_baseline, y_frac, f"{row['baseline_name']}", fontsize=6.5, ha="right", va="center", color="#666666", transform=ax_left.transAxes)

    ax_right.set_ylim(-0.5, len(pair_df) - 0.5)
    ax_right.set_xlim(0.0, 1.02)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([""] * len(pair_df))
    ax_right.tick_params(axis="y", length=0)
    ax_right.set_xlabel("Point metric")
    ax_right.grid(axis="x", color="#d9d9d9", linewidth=0.4, alpha=0.9)
    ax_right.grid(axis="y", visible=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    for idx, row in pair_df.iterrows():
        delta_color = colors["delta_pos"] if float(row["delta"]) >= 0.0 else colors["delta_neg"]
        ax_right.plot(
            [float(row["baseline_point"]), float(row["atlas_point"])],
            [idx, idx],
            color=delta_color,
            linewidth=1.2,
            alpha=0.85,
            zorder=1,
        )
        ax_right.scatter(float(row["baseline_point"]), idx, color=colors["baseline"], s=20, zorder=2)
        ax_right.scatter(float(row["atlas_point"]), idx, color=colors["ATLAS-EHR"], s=22, zorder=3)
        ax_right.text(
            min(max(float(row["atlas_point"]) + 0.012, 0.02), 0.98),
            idx + 0.18,
            f"{float(row['atlas_point']):.3f}",
            fontsize=6,
            color=colors["ATLAS-EHR"],
        )
        ax_right.text(
            min(max(float(row["baseline_point"]) + 0.012, 0.02), 0.98),
            idx - 0.22,
            f"{float(row['baseline_point']):.3f}",
            fontsize=6,
            color=colors["baseline"],
        )
        ax_right.text(
            1.005,
            idx,
            f"{float(row['delta']):+.3f}",
            fontsize=6.5,
            ha="right",
            va="center",
            color=delta_color,
            clip_on=False,
        )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4.5, markerfacecolor=colors["ATLAS-EHR"], markeredgecolor=colors["ATLAS-EHR"], label="ATLAS-EHR (z_set)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=4.5, markerfacecolor=colors["baseline"], markeredgecolor=colors["baseline"], label="Best fair baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 0.995), ncol=2, frameon=False, fontsize=6.5, handletextpad=0.6, columnspacing=1.4)
    fig.subplots_adjust(top=0.955, left=0.035, right=0.985, bottom=0.075)
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz_run_task_primary_performance(
    output_dir: str,
    paper_report_dir: str,
) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_primary_figure_table(paper_report_dir)
    png_path = os.path.join(out_root, "task_primary_performance.png")
    _viz__plot_task_primary_png(df, png_path)
    data_csv = os.path.join(out_root, "task_primary_performance_data.csv")
    df.to_csv(data_csv, index=False)
    meta = {
        "section": "task_primary_performance",
        "input_csv": os.path.join(paper_report_dir, "figure_data", "figure2_primary_metrics.csv"),
        "output_png_path": png_path,
        "main_figure_design": "paired_slope_plot",
        "uses_confidence_intervals": False,
    }
    meta_path = os.path.join(out_root, "task_primary_performance_metadata.json")
    _viz_save_json(meta_path, meta)
    return {
        "section": "task_primary_performance",
        "png": png_path,
        "data_csv": data_csv,
        "metadata_json": meta_path,
    }

def _viz__plot_icd_long_tail(df: pd.DataFrame, out_png: str) -> None:
    bins = ["common", "intermediate", "rare"]
    fig, ax = plt.subplots(figsize=_viz__figure_mm(170, 95))
    colors = {"ATLAS-EHR (z_set)": "#0072B2", "LR": "#009E73", "XGBoost": "#666666"}
    use = df[df["metric"] == "macro_auprc"].copy()
    atlas = use[use["source"] == "ATLAS-EHR (z_set)"].copy()
    baseline = []
    for b in bins:
        sub = use[(use["bin_name"] == b) & (use["model"].isin(["LR", "XGBoost"]))].copy()
        if sub.empty:
            continue
        baseline.append(sub.sort_values("point", ascending=False).iloc[0])
    base_df = pd.DataFrame(baseline)
    xs = np.arange(len(bins), dtype=np.float64)
    for label, sub in [("ATLAS-EHR (z_set)", atlas), ("Best fair baseline", base_df)]:
        pts = []
        for b in bins:
            mm = sub[sub["bin_name"] == b]
            pts.append(float(mm["point"].iloc[0]) if not mm.empty else float("nan"))
        color = colors["ATLAS-EHR (z_set)"] if "ATLAS" in label else "#666666"
        ax.plot(xs, pts, marker="o", linewidth=1.6, color=color, label=label)
    for i, b in enumerate(bins):
        mm = use[use["bin_name"] == b]
        n_labels = int(mm["n_labels"].dropna().iloc[0]) if not mm.empty else 0
        ax.text(xs[i], ax.get_ylim()[0] if ax.has_data() else 0.0, f"n={n_labels}", ha="center", va="bottom", fontsize=6, color="#555555")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Common", "Intermediate", "Rare"])
    ax.set_ylabel("Macro-AUPRC")
    ax.set_title("ICD Long-Tail Label Frequency", loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz__plot_icd_retrieval_cardinality(df: pd.DataFrame, out_png: str) -> None:
    bins = ["1-5", "6-10", "11-20", ">20"]
    fig, ax = plt.subplots(figsize=_viz__figure_mm(170, 95))
    use = df[df["metric"] == "recall_at_10"].copy()
    xs = np.arange(len(bins), dtype=np.float64)
    atlas_pts = []
    base_pts = []
    oracle_pts = []
    for b in bins:
        atlas = use[(use["cardinality_bin"] == b) & (use["source"] == "ATLAS-EHR (z_set)")]
        base = use[(use["cardinality_bin"] == b) & (use["model"].isin(["LR", "XGBoost"]))]
        atlas_pts.append(float(atlas["point"].iloc[0]) if not atlas.empty else float("nan"))
        oracle_pts.append(float(atlas["oracle_max_recall_at_k"].iloc[0]) if not atlas.empty else float("nan"))
        if base.empty:
            base_pts.append(float("nan"))
        else:
            base_pts.append(float(base.sort_values("point", ascending=False)["point"].iloc[0]))
    ax.plot(xs, atlas_pts, marker="o", linewidth=1.6, color="#0072B2", label="ATLAS-EHR (z_set)")
    ax.plot(xs, base_pts, marker="o", linewidth=1.4, color="#666666", label="Best fair baseline")
    ax.plot(xs, oracle_pts, marker="s", linewidth=1.0, color="#b0b0b0", linestyle="--", label="Oracle max recall@10")
    ax.set_xticks(xs)
    ax.set_xticklabels(bins)
    ax.set_ylabel("Recall@10")
    ax.set_title("ICD Retrieval by True-Label Cardinality", loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz__plot_drg_support_bins(df: pd.DataFrame, out_png: str) -> None:
    bins = ["common", "less_common", "rare"]
    metrics = [("top1_hit", "Top-1 hit"), ("top2_hit", "Top-2 hit"), ("top5_hit", "Top-5 hit")]
    fig, axes = plt.subplots(1, 3, figsize=_viz__figure_mm(220, 80), sharey=True)
    for ax, (metric, title) in zip(list(axes), metrics):
        use = df[df["metric"] == metric].copy()
        xs = np.arange(len(bins), dtype=np.float64)
        atlas_pts = []
        base_pts = []
        for b in bins:
            atlas = use[(use["support_bin"] == b) & (use["source"] == "ATLAS-EHR (z_set)")]
            base = use[(use["support_bin"] == b) & (use["model"].isin(["LR", "XGBoost"]))]
            atlas_pts.append(float(atlas["point"].iloc[0]) if not atlas.empty else float("nan"))
            base_pts.append(float(base.sort_values("point", ascending=False)["point"].iloc[0]) if not base.empty else float("nan"))
        ax.plot(xs, atlas_pts, marker="o", linewidth=1.6, color="#0072B2", label="ATLAS-EHR (z_set)")
        ax.plot(xs, base_pts, marker="o", linewidth=1.4, color="#666666", label="Best fair baseline")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Common", "Less common", "Rare"], rotation=10)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Hit rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz_run_icd_long_tail(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "extended_icd_label_frequency.csv")
    png_path = os.path.join(out_root, "extended_icd_label_frequency.png")
    _viz__plot_icd_long_tail(df, png_path)
    return {"section": "icd_long_tail", "png": png_path}

def _viz_run_icd_retrieval_cardinality(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "extended_icd_retrieval_cardinality.csv")
    png_path = os.path.join(out_root, "extended_icd_retrieval_cardinality.png")
    _viz__plot_icd_retrieval_cardinality(df, png_path)
    return {"section": "icd_retrieval_cardinality", "png": png_path}

def _viz_run_drg_support_bins(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "extended_drg_support_bins.csv")
    png_path = os.path.join(out_root, "extended_drg_support_bins.png")
    _viz__plot_drg_support_bins(df, png_path)
    return {"section": "drg_support_bins", "png": png_path}

def _viz__prepare_hcup_all26(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "source" not in out.columns and "source_name_raw" in out.columns:
        out["source"] = out["source_name_raw"]
    out["source"] = out["source"].apply(_viz__display_source_label)
    if "best_fair_baseline" not in out.columns:
        out["best_fair_baseline"] = "XGBoost"
    if "delta_vs_best_fair_baseline" not in out.columns:
        out["delta_vs_best_fair_baseline"] = np.nan
    return out

def _viz__plot_hcup_all26_per_label(df: pd.DataFrame, out_png: str, dual_metric: bool = False) -> None:
    required = {"label_index", "label_name", "model", "source", "auprc"}
    if df.empty or not required.issubset(set(df.columns)):
        fig = plt.figure(figsize=_viz__figure_mm(240, 160))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No HCUP all-26 data available.\nCheck paper_report/figure_data/hcup_all26_per_label_metrics.csv.", ha="center", va="center", fontsize=10)
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    atlas = df[(df["model"].astype(str) == "ATLAS-EHR") & (df["source"].astype(str) == "ATLAS-EHR (z_set)")].copy()
    if atlas.empty:
        atlas = df[df["source"].astype(str).str.contains("z_set", na=False)].copy()
    if atlas.empty:
        raise ValueError("HCUP all-26 figure requires ATLAS-EHR (z_set) rows.")
    atlas = atlas.sort_values("auprc", ascending=False).reset_index(drop=True)
    order = atlas["label_index"].astype(int).tolist()
    label_names = atlas["label_name"].astype(str).tolist()
    fig, axes = plt.subplots(1, 2 if dual_metric else 1, figsize=_viz__figure_mm(240 if dual_metric else 180, 210), sharey=True if dual_metric else False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    metrics = ["auprc", "auroc"] if dual_metric else ["auprc"]
    titles = ["HCUP all-26 per-label AUPRC", "HCUP all-26 per-label AUROC"] if dual_metric else ["HCUP all-26 per-label AUPRC"]
    colors = {"ATLAS-EHR (z_set)": "#0072B2", "ATLAS-EHR (z_context)": "#E69F00", "XGBoost": "#666666", "LR": "#666666"}
    for ax, metric, title in zip(axes, metrics, titles):
        baseline = df[(df["model"].isin(["LR", "XGBoost"]))].copy()
        y = np.arange(len(order) - 1, -1, -1, dtype=np.float64)
        if not baseline.empty:
            base_rows = []
            for idx in order:
                metric_vals = pd.to_numeric(baseline[metric], errors="coerce")
                sub = baseline[(baseline["label_index"].astype(int) == int(idx)) & metric_vals.notna()].copy()
                if sub.empty:
                    base_rows.append(np.nan)
                else:
                    best = sub.loc[pd.to_numeric(sub[metric], errors="coerce").astype(float).idxmax()]
                    base_rows.append(float(best[metric]))
            ax.scatter(base_rows, y, s=18, color=colors["XGBoost"], alpha=0.75, label="Best fair baseline")
        atlas_vals = pd.to_numeric(atlas[metric], errors="coerce").astype(float).tolist()
        ax.scatter(atlas_vals, y, s=22, color=colors["ATLAS-EHR (z_set)"], label="ATLAS-EHR (z_set)")
        if dual_metric:
            ctx = df[(df["model"].astype(str) == "ATLAS-EHR") & (df["source"].astype(str) == "ATLAS-EHR (z_context)")]
            if not ctx.empty:
                ctx = ctx.set_index("label_index").reindex(order)
                ctx_vals = pd.to_numeric(ctx[metric], errors="coerce").astype(float).tolist()
                ax.scatter(ctx_vals, y, s=16, color=colors["ATLAS-EHR (z_context)"], alpha=0.6, label="ATLAS-EHR (z_context)")
        ax.set_yticks(y)
        ax.set_yticklabels(label_names, fontsize=5.4)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(metric.upper())
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz_run_hcup_all26_per_label(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__prepare_hcup_all26(_viz__read_report_figure_csv(paper_report_dir, "hcup_all26_per_label_metrics.csv"))
    source_csv = os.path.join(out_root, "hcup_all26_per_label_metrics.csv")
    df.to_csv(source_csv, index=False)
    png_auprc = os.path.join(out_root, "hcup_all26_per_label_auprc.png")
    png_dual = os.path.join(out_root, "hcup_all26_per_label_auprc_auroc.png")
    _viz__plot_hcup_all26_per_label(df, png_auprc, dual_metric=False)
    _viz__plot_hcup_all26_per_label(df, png_dual, dual_metric=True)
    manifest = {
        "section": "hcup_all26_per_label",
        "source_data_csv": source_csv,
        "outputs": {
            "hcup_all26_per_label_auprc": png_auprc,
            "hcup_all26_per_label_auprc_auroc": png_dual,
        },
        "source_names_used": sorted(set(df.get("source", pd.Series(dtype=str)).astype(str).tolist())),
    }
    manifest_path = os.path.join(out_root, "hcup_all26_per_label_manifest.json")
    _viz_save_json(manifest_path, manifest)
    return {
        "section": "hcup_all26_per_label",
        "png": png_auprc,
        "png_dual": png_dual,
        "data_csv": source_csv,
        "manifest_json": manifest_path,
    }

def _viz__plot_imputation_overview(df: pd.DataFrame, out_png: str) -> None:
    required = {"metric", "model_or_baseline", "point"}
    if df.empty or not required.issubset(set(df.columns)):
        fig = plt.figure(figsize=_viz__figure_mm(180, 95))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No imputation overview data available.\nCheck paper_report/figure_data/imputation_overview_mse_r2.csv.", ha="center", va="center")
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=_viz__figure_mm(180, 90))
    order = ["ATLAS-EHR", "last-value-carry-forward"]
    titles = [("mse", "MSE"), ("r2", "R²")]
    colors = {"ATLAS-EHR": "#0072B2", "last-value-carry-forward": "#666666"}
    for ax, (metric, ttl) in zip(axes, titles):
        sub = df[df["metric"].astype(str).str.lower() == metric].copy()
        xs = np.arange(len(order), dtype=np.float64)
        vals = [float(sub[sub["model_or_baseline"] == m]["point"].iloc[0]) if not sub[sub["model_or_baseline"] == m].empty else float("nan") for m in order]
        ax.bar(xs, vals, color=[colors[m] for m in order], edgecolor="#333333", linewidth=0.3)
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.set_title(ttl, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Macro metric")
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz__plot_imputation_by_clinical_group(df: pd.DataFrame, out_png: str) -> None:
    if df.empty or "clinical_group" not in df.columns:
        fig = plt.figure(figsize=_viz__figure_mm(180, 120))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No imputation clinical-group data available.", ha="center", va="center")
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    work = df.copy().sort_values("atlas_macro_mse", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=_viz__figure_mm(200, 110), sharey=True)
    metrics = [("delta_mse_vs_last_value_carry_forward", "ΔMSE vs last-value-carry-forward"), ("delta_r2_vs_last_value_carry_forward", "ΔR² vs last-value-carry-forward")]
    colors = "#0072B2"
    y = np.arange(len(work), dtype=np.float64)
    for ax, (col, ttl) in zip(axes, metrics):
        vals = work[col].astype(float).to_numpy()
        ax.barh(y, vals, color=colors, edgecolor="#333333", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r} (n={int(n)})" for r, n in zip(work["clinical_group"].astype(str), work["n_features"].fillna(0))], fontsize=5.5)
        ax.set_title(ttl, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_xlabel("Improvement")
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz__plot_imputation_mechanism_bins(df: pd.DataFrame, out_png: str) -> None:
    if df.empty or "mechanism" not in df.columns:
        fig = plt.figure(figsize=_viz__figure_mm(180, 120))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No imputation mechanism-bin data available.", ha="center", va="center")
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 3, figsize=_viz__figure_mm(230, 100), sharey=True)
    order_map = {
        "frequency": ["low support", "medium support", "high support"],
        "prior_value_availability": ["low prior-value availability", "medium prior-value availability", "high prior-value availability"],
        "cooccurrence_context": ["low co-occurrence context", "medium co-occurrence context", "high co-occurrence context"],
    }
    metric_map = "delta_mse_vs_best_baseline" if "delta_mse_vs_best_baseline" in df.columns else "delta_mse_vs_last_value_carry_forward"
    for ax, mech in zip(axes, ["frequency", "prior_value_availability", "cooccurrence_context"]):
        sub = df[df["mechanism"].astype(str) == mech].copy()
        order = order_map[mech]
        vals = []
        for b in order:
            mm = sub[sub["bin_name"].astype(str) == b]
            vals.append(float(mm[metric_map].iloc[0]) if not mm.empty else float("nan"))
        xs = np.arange(len(order), dtype=np.float64)
        ax.plot(xs, vals, marker="o", color="#0072B2", linewidth=1.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=18, ha="right", fontsize=5.5)
        ax.set_title(mech.replace("_", " "), loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("ΔMSE vs best baseline")
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz__plot_imputation_all_features_landscape(df: pd.DataFrame, out_png: str) -> None:
    if df.empty or not {"atlas_mse", "best_baseline_mse", "clinical_group"}.issubset(set(df.columns)):
        fig = plt.figure(figsize=_viz__figure_mm(180, 120))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No imputation landscape data available.", ha="center", va="center")
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=_viz__figure_mm(180, 120))
    groups = sorted(df["clinical_group"].astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10")
    for i, grp in enumerate(groups):
        sub = df[df["clinical_group"].astype(str) == grp]
        ax.scatter(sub["best_baseline_mse"], sub["atlas_mse"], s=10, alpha=0.75, color=cmap(i % 10), label=grp)
    lim = max(float(np.nanmax(df["best_baseline_mse"])), float(np.nanmax(df["atlas_mse"])))
    ax.plot([0, lim], [0, lim], color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Best baseline MSE")
    ax.set_ylabel("ATLAS MSE")
    ax.set_title("Imputation all-features landscape", loc="left", fontsize=8, fontweight="bold")
    ax.grid(color="#d9d9d9", linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.5, loc="best", ncol=1)
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz__plot_imputation_representative_features(df: pd.DataFrame, out_png: str) -> None:
    if df.empty or not {"feature_id", "feature_name", "true_value", "pred_mu"}.issubset(set(df.columns)):
        fig = plt.figure(figsize=_viz__figure_mm(220, 140))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No representative imputation points available.", ha="center", va="center")
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    reps = sorted(df["feature_name"].astype(str).unique().tolist())[:3]
    fig, axes = plt.subplots(len(reps), 1, figsize=_viz__figure_mm(220, 140), sharex=False)
    if len(reps) == 1:
        axes = [axes]
    for ax, feat in zip(axes, reps):
        sub = df[df["feature_name"].astype(str) == feat].copy()
        ax.scatter(sub["true_value"], sub["pred_mu"], s=7, alpha=0.7, color="#0072B2", label="ATLAS-EHR")
        if "last_value_carry_forward_pred" in sub.columns:
            ax.scatter(sub["true_value"], sub["last_value_carry_forward_pred"], s=7, alpha=0.55, color="#666666", label="last-value-carry-forward")
        lo = float(np.nanmin(sub[["true_value", "pred_mu"]].to_numpy(dtype=float)))
        hi = float(np.nanmax(sub[["true_value", "pred_mu"]].to_numpy(dtype=float)))
        ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(feat, loc="left", fontsize=7, fontweight="bold")
        ax.grid(color="#d9d9d9", linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # simple metric box
        atlas_mse = float(np.mean((sub["pred_mu"] - sub["true_value"]) ** 2))
        locf_mse = float(np.mean((sub["last_value_carry_forward_pred"] - sub["true_value"]) ** 2)) if "last_value_carry_forward_pred" in sub.columns else float("nan")
        atlas_r2 = 1.0 - float(np.sum((sub["pred_mu"] - sub["true_value"]) ** 2)) / float(np.sum((sub["true_value"] - sub["true_value"].mean()) ** 2)) if np.sum((sub["true_value"] - sub["true_value"].mean()) ** 2) > 0 else float("nan")
        locf_r2 = 1.0 - float(np.sum((sub["last_value_carry_forward_pred"] - sub["true_value"]) ** 2)) / float(np.sum((sub["true_value"] - sub["true_value"].mean()) ** 2)) if "last_value_carry_forward_pred" in sub.columns and np.sum((sub["true_value"] - sub["true_value"].mean()) ** 2) > 0 else float("nan")
        ax.text(
            0.02, 0.98,
            f"ATLAS MSE {atlas_mse:.3f}\nlast-value-carry-forward MSE {locf_mse:.3f}\nATLAS R² {atlas_r2:.3f}\nlast-value-carry-forward R² {locf_r2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", linewidth=0.4, alpha=0.9),
        )
    axes[-1].set_xlabel("Observed value")
    fig.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _viz_run_imputation_overview(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "imputation_overview_mse_r2.csv")
    source_csv = os.path.join(out_root, "imputation_overview_mse_r2.csv")
    df.to_csv(source_csv, index=False)
    png_path = os.path.join(out_root, "imputation_overview_mse_r2.png")
    _viz__plot_imputation_overview(df, png_path)
    manifest = {"section": "imputation_overview", "source_data_csv": source_csv, "outputs": {"png": png_path}}
    manifest_path = os.path.join(out_root, "imputation_overview_manifest.json")
    _viz_save_json(manifest_path, manifest)
    return {"section": "imputation_overview", "png": png_path, "data_csv": source_csv, "manifest_json": manifest_path}

def _viz_run_imputation_by_feature_group(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "imputation_by_clinical_group.csv")
    source_csv = os.path.join(out_root, "imputation_by_clinical_group.csv")
    df.to_csv(source_csv, index=False)
    png_path = os.path.join(out_root, "imputation_by_clinical_group.png")
    _viz__plot_imputation_by_clinical_group(df, png_path)
    manifest_path = os.path.join(out_root, "imputation_by_feature_group_manifest.json")
    _viz_save_json(manifest_path, {"section": "imputation_by_feature_group", "source_data_csv": source_csv, "outputs": {"png": png_path}})
    return {"section": "imputation_by_feature_group", "png": png_path, "data_csv": source_csv, "manifest_json": manifest_path}

def _viz_run_imputation_mechanism_bins(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "imputation_mechanism_bins.csv")
    source_csv = os.path.join(out_root, "imputation_mechanism_bins.csv")
    df.to_csv(source_csv, index=False)
    png_path = os.path.join(out_root, "imputation_mechanism_bins.png")
    _viz__plot_imputation_mechanism_bins(df, png_path)
    manifest_path = os.path.join(out_root, "imputation_mechanism_bins_manifest.json")
    _viz_save_json(manifest_path, {"section": "imputation_mechanism_bins", "source_data_csv": source_csv, "outputs": {"png": png_path}})
    return {"section": "imputation_mechanism_bins", "png": png_path, "data_csv": source_csv, "manifest_json": manifest_path}

def _viz_run_imputation_all_features(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "imputation_all_features_landscape.csv")
    source_csv = os.path.join(out_root, "imputation_all_features_landscape.csv")
    df.to_csv(source_csv, index=False)
    png_path = os.path.join(out_root, "imputation_all_features_landscape.png")
    _viz__plot_imputation_all_features_landscape(df, png_path)
    manifest_path = os.path.join(out_root, "imputation_all_features_manifest.json")
    _viz_save_json(manifest_path, {"section": "imputation_all_features", "source_data_csv": source_csv, "outputs": {"png": png_path}})
    return {"section": "imputation_all_features", "png": png_path, "data_csv": source_csv, "manifest_json": manifest_path}

def _viz_run_imputation_representative_features(output_dir: str, paper_report_dir: str) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "imputation_representative_features.csv")
    df = df.rename(columns={
        "locf_pred": "last_value_carry_forward_pred",
        "residual_locf": "residual_last_value_carry_forward",
    })
    source_csv = os.path.join(out_root, "imputation_representative_features.csv")
    df.to_csv(source_csv, index=False)
    png_path = os.path.join(out_root, "imputation_representative_features.png")
    _viz__plot_imputation_representative_features(df, png_path)
    manifest_path = os.path.join(out_root, "imputation_representative_manifest.json")
    _viz_save_json(manifest_path, {"section": "imputation_representative_features", "source_data_csv": source_csv, "outputs": {"png": png_path}})
    return {"section": "imputation_representative_features", "png": png_path, "data_csv": source_csv, "manifest_json": manifest_path}

def _viz__plot_setencoder_attention_top_context(df: pd.DataFrame, out_png: str, metric: str = "lift") -> None:
    """Horizontal bar plot of top context variables per target (one panel per target).

    Caption language: SetEncoder local aggregation behavior — attention assigned to
    same-timestamp observations when the target variable is withheld.
    """
    _viz__apply_nature_style()
    if df.empty or "target_vid" not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No SetEncoder attention data available.", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    target_vids = sorted(df["target_vid"].unique().tolist())
    n_panels = max(1, len(target_vids))
    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(7.0, max(4.0, 4.2 * n_panels)), squeeze=False)
    metric_col = metric if metric in df.columns else ("lift" if "lift" in df.columns else "mean_alpha")
    metric_label = "Attention lift over uniform pooling" if "lift" in metric_col else "Mean gated-pooling weight (alpha)"

    for r, tvid in enumerate(target_vids):
        ax = axes[r, 0]
        sub = df[df["target_vid"] == tvid].copy()
        if "rank_by_" + metric_col in sub.columns:
            sub = sub.nsmallest(10, "rank_by_" + metric_col)
        else:
            sub = sub.nlargest(10, metric_col)
        sub = sub.sort_values(metric_col, ascending=True)
        labels = sub["context_label"].fillna("").astype(str).tolist()
        vals = sub[metric_col].fillna(0).astype(float).tolist()
        cnts = sub["alpha_count"].fillna(0).astype(int).tolist() if "alpha_count" in sub.columns else [0] * len(labels)
        grps = sub["context_clinical_group"].fillna("").astype(str).tolist() if "context_clinical_group" in sub.columns else [""] * len(labels)
        y = np.arange(len(vals))
        cmap = plt.get_cmap("Blues")
        vmax = max(float(max(vals)), 1e-6)
        colors = [cmap(0.35 + 0.55 * v / vmax) for v in vals]
        ax.barh(y, vals, color=colors, edgecolor="none")
        ax.set_yticks(y)
        clipped = [lbl[:37] + "..." if len(lbl) > 40 else lbl for lbl in labels]
        ax.set_yticklabels(clipped, fontsize=7.5)
        for yi, (v, c, g) in enumerate(zip(vals, cnts, grps)):
            ax.text(v + 0.01 * vmax, yi, f"n={c}" + (f"  [{g}]" if g else ""), va="center", ha="left", fontsize=6.5)
        tlabel = str(sub["target_label"].iloc[0]) if "target_label" in sub.columns and len(sub) > 0 else f"VID {tvid}"
        ax.set_title(tlabel, fontsize=9, loc="left", pad=6)
        ax.set_xlabel(metric_label, fontsize=8)
        ax.set_xlim(0, vmax * 1.25)
        ax.grid(axis="x", color="#cccccc", linewidth=0.5, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.01,
             "SetEncoder local aggregation behavior (protocol: local_set_encoder_attention).\n"
             "Attention assigned to same-timestamp observations when the target variable is withheld.",
             ha="center", fontsize=6.5, style="italic")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.06, hspace=0.45)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def _viz__plot_setencoder_attention_group_summary(df_grp: pd.DataFrame, out_png: str, metric: str = "agg_lift") -> None:
    """Target rows × context clinical groups grouped bar chart.

    Caption language: aggregated SetEncoder local attention by clinical group.
    """
    _viz__apply_nature_style()
    if df_grp.empty or "target_vid" not in df_grp.columns or "context_clinical_group" not in df_grp.columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No SetEncoder attention group data available.", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    metric_col = metric if metric in df_grp.columns else "agg_mean_alpha"
    target_labels = (df_grp.groupby("target_vid")["target_label"].first()
                     .reindex(sorted(df_grp["target_vid"].unique())).fillna("").tolist())
    target_vids = sorted(df_grp["target_vid"].unique().tolist())
    groups = sorted(df_grp["context_clinical_group"].dropna().unique().tolist())
    n_targets = len(target_vids)
    n_groups = len(groups)
    if n_targets == 0 or n_groups == 0:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No group data.", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * n_groups), max(3.5, 0.9 * n_targets + 1.5)))
    x = np.arange(n_groups)
    width = 0.8 / max(n_targets, 1)
    palette = plt.get_cmap("tab10")
    for i, (tvid, tlabel) in enumerate(zip(target_vids, target_labels)):
        sub = df_grp[df_grp["target_vid"] == tvid]
        vals = [float(sub.loc[sub["context_clinical_group"] == g, metric_col].mean())
                if g in sub["context_clinical_group"].values else 0.0 for g in groups]
        ax.bar(x + i * width - (n_targets - 1) * width / 2, vals, width=width * 0.92,
               label=tlabel or f"VID {tvid}", color=palette(i % 10), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=7.5)
    metric_label = "Aggregated attention lift" if "lift" in metric_col else "Aggregated mean alpha"
    ax.set_ylabel(metric_label, fontsize=8)
    ax.set_xlabel("Context clinical group", fontsize=8)
    ax.legend(title="Target variable", fontsize=7, title_fontsize=7.5, loc="upper right")
    ax.grid(axis="y", color="#cccccc", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, -0.04,
             "SetEncoder local attention aggregated by clinical group (protocol: local_set_encoder_attention).",
             ha="center", fontsize=6.5, style="italic")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def _viz_run_setencoder_attention(
    output_dir: str,
    paper_report_dir: str,
    attribution_dir: Optional[str] = None,
) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    figure_data_dir = os.path.join(paper_report_dir, "figure_data")

    if attribution_dir is not None:
        raise ValueError(
            "visualization.py no longer builds attribution tables from attribution_dir. "
            "Run: python -m src.eval --include-attribution --attribution-output-dir <dir> "
            "to generate paper_report/figure_data/setencoder_attention_*.csv first."
        )
    top_csv = os.path.join(figure_data_dir, "setencoder_attention_top_context.csv")
    grp_csv = os.path.join(figure_data_dir, "setencoder_attention_by_group.csv")
    if not (os.path.exists(top_csv) and os.path.exists(grp_csv)):
        raise ValueError(
            "setencoder_attention mode requires pre-existing paper_report/figure_data CSVs. "
            "Run: python -m src.eval --include-attribution --attribution-output-dir <dir> first."
        )
    df_top = pd.read_csv(top_csv)
    df_grp = pd.read_csv(grp_csv)

    top_png = os.path.join(out_root, "setencoder_attention_top_context.png")
    grp_png = os.path.join(out_root, "setencoder_attention_group_summary.png")
    _viz__plot_setencoder_attention_top_context(df_top, top_png)
    _viz__plot_setencoder_attention_group_summary(df_grp, grp_png)

    manifest_path = os.path.join(out_root, "setencoder_attention_manifest.json")
    _viz_save_json(manifest_path, {
        "section": "setencoder_attention",
        "protocol": "local_set_encoder_attention",
        "input_top_csv": top_csv,
        "input_grp_csv": grp_csv,
        "outputs": {"top_context_png": top_png, "group_summary_png": grp_png},
    })
    return {
        "section": "setencoder_attention",
        "top_context_png": top_png,
        "group_summary_png": grp_png,
        "manifest_json": manifest_path,
    }

def _viz__secondary_group(task: str, metric: str) -> Optional[str]:
    if task in _viz__OUTCOME_TASKS and metric in _viz__SECONDARY_OUTCOME:
        return "calibration" if metric in {"brier", "ece"} else "threshold"
    if task in {"icd", "hcup"} and metric in _viz__SECONDARY_PHENO_FREQ:
        return "frequency_generalization"
    if task == "hcup" and metric in {"micro_f1", "macro_f1"}:
        return "topk_retrieval"
    if task in {"icd", "hcup"} and metric in _viz__SECONDARY_TOPK:
        return "topk_retrieval"
    if task == "drg" and metric == "cardinality_accuracy":
        return "cardinality"
    if task == "drg" and metric in _viz__SECONDARY_DRG:
        return "drg_auxiliary"
    return None

def _viz_run_task_secondary_metrics(
    output_dir: str,
    paper_report_dir: str,
) -> Dict[str, Any]:
    _viz__apply_nature_style()
    out_root = _viz__ensure_dir(str(output_dir))
    df = _viz__read_report_figure_csv(paper_report_dir, "figure2_secondary_metrics.csv")
    # Adapt eval.py column names to what _plot_task_secondary_combined expects
    if not df.empty and "model" in df.columns and "display_model" not in df.columns:
        df = df.rename(columns={"model": "display_model"})
    if not df.empty:
        df["bin"] = df.get("bin", "")
        df["metric_group"] = df.apply(
            lambda r: _viz__secondary_group(str(r.get("task", "")), str(r.get("metric", ""))) or "", axis=1
        )
    out_big = os.path.join(out_root, "task_secondary_metrics.png")
    freq_warn = _viz__plot_task_secondary_combined(df, out_big)
    csv_path = os.path.join(out_root, "task_secondary_metrics_selected.csv")
    df.to_csv(csv_path, index=False)
    meta = {
        "input_csv": os.path.join(paper_report_dir, "figure_data", "figure2_secondary_metrics.csv"),
        "warnings": freq_warn,
        "outputs": {"task_secondary_metrics_png": out_big, "selected_csv": csv_path},
    }
    meta_path = os.path.join(out_root, "task_secondary_metrics_metadata.json")
    _viz_save_json(meta_path, meta)
    print(f"[task-secondary] wrote: {out_big}")
    return {
        "section": "task_secondary_metrics",
        "secondary_png": out_big,
        "selected_csv": csv_path,
        "metadata_json": meta_path,
    }

def _viz__plot_task_secondary_combined(df: pd.DataFrame, out_png: str) -> List[str]:
    warns: List[str] = []
    df = df.reindex(columns=["task", "display_model", "variant", "metric", "point", "metric_group", "direction", "bin"])
    expected_cols = {"task", "display_model", "metric", "point", "bin"}
    if df.empty or not expected_cols.issubset(set(df.columns)):
        fig = plt.figure(figsize=_viz__figure_mm(220, 190))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No task-secondary data available after variant selection.\nCheck input tables/model names/metrics.",
            ha="center",
            va="center",
            fontsize=10,
        )
        _viz__ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return warns
    colors = {"ATLAS-EHR (z_set)": "#0072B2", "ATLAS-EHR (z_context)": "#E69F00", "LR": "#009E73", "XGBoost": "#666666"}
    fig = plt.figure(figsize=_viz__figure_mm(224, 190))
    gs = fig.add_gridspec(3, 2, hspace=0.58, wspace=0.30)

    def _style_bar_axis(ax: plt.Axes, title: str, ylim: Tuple[float, float] = (0.0, 1.0)) -> None:
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=10)
        ax.set_ylim(*ylim)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", pad=6)
        ax.margins(x=0.05)

    # Row 1: A, B
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(_viz__OUTCOME_TASKS), dtype=np.float64)
    tasks_lbl = [_viz__TASK_TITLE[t] for t in _viz__OUTCOME_TASKS]
    for ax, (metric, title) in zip([ax_a, ax_b], [("brier", "A) Brier score (lower is better)"), ("ece", "B) ECE (lower is better)")]):
        bw = 0.18
        for i, model in enumerate(_viz__MODEL_ORDER):
            vals = []
            for t in _viz__OUTCOME_TASKS:
                mm = df[(df["task"] == t) & (df["display_model"] == model) & (df["metric"] == metric)]
                vals.append(float(mm["point"].iloc[0]) if not mm.empty else float("nan"))
            offs = x + (i - 1.5) * bw
            ax.bar(offs, vals, width=bw, color=colors[model], alpha=0.85, edgecolor="#333333", linewidth=0.25, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_lbl, rotation=20, ha="right")
        _style_bar_axis(ax, title, ylim=(0.0, 0.25))

    # Row 2: C, D (ICD overall retrieval diagnostics)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    task = "icd"
    metric_panels = [
        (ax_c, "precision_at_10", "C) ICD Precision@10", (0.0, 0.45)),
        (ax_d, "recall_at_10", "D) ICD Recall@10", (0.0, 0.30)),
    ]
    xx = np.arange(1, dtype=np.float64)
    for ax, metric_name, title, ylim in metric_panels:
        bw = 0.18
        for i, model in enumerate(_viz__MODEL_ORDER):
            mm = df[(df["task"] == task) & (df["display_model"] == model) & (df["metric"] == metric_name)]
            val = float(mm["point"].iloc[0]) if not mm.empty else float("nan")
            offs = xx + (i - 1.5) * bw
            ax.bar(offs, [val], width=bw, color=colors[model], alpha=0.85, edgecolor="#333333", linewidth=0.25, label=model)
        ax.set_xticks(xx)
        ax.set_xticklabels(["All test admissions"])
        _style_bar_axis(ax, title, ylim=ylim)

    # Row 3: E/F supplementary bars
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    e_candidates = [
        ("precision_at_5", "Precision@5"),
        ("recall_at_5", "Recall@5"),
        ("micro_f1", "Micro-F1"),
        ("macro_f1", "Macro-F1"),
    ]
    e_metrics = [m for m, _ in e_candidates if any(not df[(df["task"] == "hcup") & (df["display_model"] == model) & (df["metric"] == m)].empty for model in _viz__MODEL_ORDER)]
    e_labels = [label for m, label in e_candidates if m in e_metrics]
    xe = np.arange(len(e_metrics), dtype=np.float64)
    bw_e = 0.18
    for i, model in enumerate(_viz__MODEL_ORDER):
        vals = []
        for m in e_metrics:
            mm = df[(df["task"] == "hcup") & (df["display_model"] == model) & (df["metric"] == m)]
            vals.append(float(pd.to_numeric(mm["point"], errors="coerce").mean()) if not mm.empty else float("nan"))
        offs = xe + (i - 1.5) * bw_e
        ax_e.bar(offs, vals, width=bw_e, color=colors[model], alpha=0.85, edgecolor="#333333", linewidth=0.25, label=model)
    ax_e.set_xticks(xe)
    ax_e.set_xticklabels(e_labels, rotation=20, ha="right")
    _style_bar_axis(ax_e, "E) HCUP supplementary retrieval metrics", ylim=(0.0, 0.65))

    f_candidates = [
        ("cardinality_accuracy", "Cardinality"),
        ("top3_hit", "Top-3 hit"),
        ("top5_hit", "Top-5 hit"),
        ("top3_recall", "Top-3 recall"),
        ("top5_recall", "Top-5 recall"),
    ]
    f_metrics = [m for m, _ in f_candidates if any(not df[(df["task"] == "drg") & (df["display_model"] == model) & (df["metric"] == m)].empty for model in _viz__MODEL_ORDER)]
    f_labels = [label for m, label in f_candidates if m in f_metrics]
    xf = np.arange(len(f_metrics), dtype=np.float64)
    bw_f = 0.18
    for i, model in enumerate(_viz__MODEL_ORDER):
        vals = []
        for mt in f_metrics:
            mm = df[(df["task"] == "drg") & (df["display_model"] == model) & (df["metric"] == mt)]
            vals.append(float(mm["point"].iloc[0]) if not mm.empty else float("nan"))
        offs = xf + (i - 1.5) * bw_f
        ax_f.bar(offs, vals, width=bw_f, color=colors[model], alpha=0.85, edgecolor="#333333", linewidth=0.25)
    ax_f.set_xticks(xf)
    ax_f.set_xticklabels(f_labels, rotation=20, ha="right")
    _style_bar_axis(ax_f, "F) DRG supplementary decoding metrics", ylim=(0.0, 1.0))

    handles, labels = ax_a.get_legend_handles_labels()
    bylab = dict(zip(labels, handles))
    fig.legend(bylab.values(), bylab.keys(), loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(top=0.965, bottom=0.12, left=0.06, right=0.985)
    fig.patch.set_facecolor("white")
    _viz__ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, format="png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return warns
