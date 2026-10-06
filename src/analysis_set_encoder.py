"""SetEncoder local-attention analysis workflow."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import PatientSequenceDataset, nested_padding_collate
from src.model import JEPAModel

# ===== set_encoder_attention implementation =====
"""Standalone SetEncoder local attention analysis (protocol: local_set_encoder_attention).

Probes local same-timestamp set aggregation by withholding selected target variables
from a state set and examining the gated-pooling attention (alpha) assigned to the
remaining same-timestamp observations.

This characterizes local co-measurement attention structure, not causal feature
importance for imputation or temporal attribution of any readout prediction.
"""

matplotlib.use("Agg")

def _attr_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _attr_resolve_device(spec: str) -> torch.device:
    x = str(spec).lower().strip()
    if x != "auto":
        return torch.device(x)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _attr_run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)

    device = _attr_resolve_device(args.device)
    teacher = JEPAModel.load_from_checkpoint(
        args.phase1_ckpt,
        map_location=device,
        schema_path=os.path.join(args.schema_dir, "schema_processed.csv"),
        weights_only=False,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    df_schema = pd.read_csv(os.path.join(args.schema_dir, "schema_processed.csv"))
    if "vid" not in df_schema.columns or "normalized_concept_label" not in df_schema.columns:
        raise RuntimeError("schema_processed.csv must include vid and normalized_concept_label")
    label_map = {
        int(r["vid"]): str(r["normalized_concept_label"])
        for _, r in df_schema[["vid", "normalized_concept_label"]].dropna(subset=["vid"]).iterrows()
    }
    group_map: Dict[int, str] = {}
    if "feature_category" in df_schema.columns:
        for _, r in df_schema[["vid", "feature_category"]].dropna(subset=["vid"]).iterrows():
            v = r.get("feature_category")
            if pd.notna(v):
                group_map[int(r["vid"])] = str(v)
    source_map: Dict[int, str] = {}
    if "model_type" in df_schema.columns:
        for _, r in df_schema[["vid", "model_type"]].dropna(subset=["vid"]).iterrows():
            v = r.get("model_type")
            if pd.notna(v):
                source_map[int(r["vid"])] = str(v)

    ds = PatientSequenceDataset(
        data_dir=args.data_dir,
        schema_dir=args.schema_dir,
        split=args.split,
        phase=1,
        mask_target_ratio=0.0,
        mask_min_prob=0.0,
        mask_max_prob=0.0,
        static_mask_prob=0.0,
        static_numeric_vids=[],
    )
    if args.max_patients is not None and args.max_patients > 0:
        ds = Subset(ds, list(range(min(int(args.max_patients), len(ds)))))

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=nested_padding_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    target_vids = [int(x) for x in args.target_vids]

    # Token-conditional statistics:
    # mean_alpha_present = E[alpha_{t,c} | target withheld, context token c visible]
    # alpha_over_uniform = E[alpha_{t,c} * S_t | target withheld, context token c visible]
    alpha_sum: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    alpha_over_uniform_sum: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    alpha_count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # Target-timestamp mass statistics for uniform-baseline lift:
    # mass_lift compares observed alpha mass share vs expected uniform mass over the same T_q.
    target_ts_count: Dict[int, int] = defaultdict(int)
    alpha_mass_sum: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    expected_mass_sum: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    n_context_timestamps: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    n_context_tokens: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    context_patients: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    freq: Dict[int, int] = defaultdict(int)
    set_sizes: Dict[int, List[int]] = defaultdict(list)
    total_context_tokens = 0
    eps = 1.0e-12

    fp = teacher.target_feature_projector
    fn = teacher.target_fusion_norm
    se = teacher.target_set_encoder

    for batch in tqdm(loader, desc="attribution", dynamic_ncols=True):
        st = batch["state_tokens"].to(device)           # (B,T,S)
        sv = batch["state_values"].to(device)           # (B,T,S,1)
        sp = batch["state_pad_mask"].to(device).bool()  # (B,T,S)
        tm = batch["state_times"].to(device)            # (B,T,1)

        B, T, S = st.shape
        D = teacher.embed_dim
        hadm_ids = [str(x) for x in batch.get("hadm_id", [""] * B)]

        z_loc_norm = fn(fp(st, sv))
        z_bt = z_loc_norm.view(B * T, S, D)

        st_bt = st.view(B * T, S)
        sp_bt = sp.view(B * T, S)

        for vid_q in target_vids:
            is_masked = (st == int(vid_q)) & (~sp)
            combined = sp | is_masked
            combined_bt = combined.view(B * T, S)
            masked_any = is_masked.view(B * T, S).any(dim=1)
            if not bool(masked_any.any()):
                continue

            _, alpha = se(z_bt, combined_bt, return_attn=True)  # alpha: (B*T,S)

            rows = torch.where(masked_any)[0]
            for bt_idx in rows.tolist():
                vis_mask = ~combined_bt[bt_idx]
                if not bool(vis_mask.any()):
                    continue
                b_idx = int(bt_idx // T)
                hid = hadm_ids[b_idx] if b_idx < len(hadm_ids) else ""
                ctx_vids = st_bt[bt_idx][vis_mask].detach().cpu().tolist()
                ctx_alpha = alpha[bt_idx][vis_mask].detach().cpu().tolist()
                s_t = int(len(ctx_vids))
                if s_t <= 0:
                    continue
                target_ts_count[int(vid_q)] += 1
                set_sizes[int(vid_q)].append(s_t)
                # Per-timestamp aggregates (multi-occurrence aware).
                tok_count_t: Dict[int, int] = defaultdict(int)
                alpha_mass_t: Dict[int, float] = defaultdict(float)
                for vctx, a in zip(ctx_vids, ctx_alpha):
                    vctx_i = int(vctx)
                    if vctx_i <= 0:
                        continue
                    aval = float(a)
                    alpha_sum[int(vid_q)][vctx_i] += aval
                    alpha_over_uniform_sum[int(vid_q)][vctx_i] += aval * float(s_t)
                    alpha_count[int(vid_q)][vctx_i] += 1
                    tok_count_t[vctx_i] += 1
                    alpha_mass_t[vctx_i] += aval
                    freq[vctx_i] += 1
                    total_context_tokens += 1
                for cvid, n_tok in tok_count_t.items():
                    n_context_tokens[int(vid_q)][int(cvid)] += int(n_tok)
                    n_context_timestamps[int(vid_q)][int(cvid)] += 1
                    alpha_mass_sum[int(vid_q)][int(cvid)] += float(alpha_mass_t[cvid])
                    expected_mass_sum[int(vid_q)][int(cvid)] += float(n_tok) / float(s_t)
                    if hid:
                        context_patients[int(vid_q)][int(cvid)].add(str(hid))

    vid_query_list = sorted(alpha_sum.keys())
    vid_context_list = sorted(freq.keys())

    q2i = {v: i for i, v in enumerate(vid_query_list)}
    c2i = {v: j for j, v in enumerate(vid_context_list)}

    mean_alpha_present = np.zeros((len(vid_query_list), len(vid_context_list)), dtype=np.float32)
    alpha_over_uniform = np.zeros_like(mean_alpha_present)
    alpha_mass_share = np.zeros_like(mean_alpha_present)
    expected_uniform_mass = np.zeros_like(mean_alpha_present)
    mass_lift = np.zeros_like(mean_alpha_present)
    log2_mass_lift = np.zeros_like(mean_alpha_present)
    global_frequency_lift = np.zeros_like(mean_alpha_present)
    cooccurrence_rate = np.zeros_like(mean_alpha_present)
    n_target_timestamps_arr = np.zeros((len(vid_query_list),), dtype=np.int64)
    n_context_tokens_arr = np.zeros_like(mean_alpha_present, dtype=np.int64)
    n_context_timestamps_arr = np.zeros_like(mean_alpha_present, dtype=np.int64)
    n_context_patients_arr = np.zeros_like(mean_alpha_present, dtype=np.int64)
    baseline = np.zeros((len(vid_context_list),), dtype=np.float32)
    if total_context_tokens > 0:
        for cvid, f in freq.items():
            baseline[c2i[cvid]] = float(f) / float(total_context_tokens)

    for qvid in vid_query_list:
        qi = q2i[qvid]
        n_q = int(target_ts_count.get(qvid, 0))
        n_target_timestamps_arr[qi] = n_q
        for cvid, ssum in alpha_sum[qvid].items():
            ci = c2i[cvid]
            cnt = alpha_count[qvid][cvid]
            if cnt <= 0:
                continue
            m = float(ssum) / float(cnt)
            mean_alpha_present[qi, ci] = m
            alpha_over_uniform[qi, ci] = float(alpha_over_uniform_sum[qvid][cvid]) / float(cnt)
            n_tok = int(n_context_tokens[qvid].get(cvid, 0))
            n_ts_ctx = int(n_context_timestamps[qvid].get(cvid, 0))
            n_pt_ctx = int(len(context_patients[qvid].get(cvid, set())))
            n_context_tokens_arr[qi, ci] = n_tok
            n_context_timestamps_arr[qi, ci] = n_ts_ctx
            n_context_patients_arr[qi, ci] = n_pt_ctx
            if n_q > 0:
                a_ms = float(alpha_mass_sum[qvid].get(cvid, 0.0)) / float(n_q)
                e_ms = float(expected_mass_sum[qvid].get(cvid, 0.0)) / float(n_q)
                alpha_mass_share[qi, ci] = a_ms
                expected_uniform_mass[qi, ci] = e_ms
                cooccurrence_rate[qi, ci] = float(n_ts_ctx) / float(n_q)
                ml = a_ms / (e_ms + eps)
                mass_lift[qi, ci] = ml
                log2_mass_lift[qi, ci] = float(np.log2(ml + eps))
            b = float(baseline[ci])
            global_frequency_lift[qi, ci] = float(m / b) if b > 0 else 0.0

    # Lightweight numerical validations.
    alpha_mass_total_mean_by_query: Dict[int, float] = {}
    for qvid in vid_query_list:
        qi = q2i[qvid]
        n_q = int(n_target_timestamps_arr[qi])
        if n_q > 0:
            total_mass_q = float(np.sum(alpha_mass_share[qi, :]))
            alpha_mass_total_mean_by_query[int(qvid)] = total_mass_q
            if not np.isfinite(total_mass_q):
                raise RuntimeError(f"Non-finite alpha mass for query VID {qvid}")
        if np.any(expected_uniform_mass[qi, :] < -1.0e-8):
            raise RuntimeError(f"Negative expected_uniform_mass detected for query VID {qvid}")
    finite_mask = n_context_tokens_arr > 0
    if np.any(~np.isfinite(mass_lift[finite_mask])):
        raise RuntimeError("Non-finite mass_lift values detected on emitted pairs")

    np.savez(
        os.path.join(args.output_dir, "cooccurrence_matrix.npz"),
        mean_alpha_present=mean_alpha_present,
        alpha_over_uniform=alpha_over_uniform,
        alpha_mass_share=alpha_mass_share,
        expected_uniform_mass=expected_uniform_mass,
        mass_lift=mass_lift,
        log2_mass_lift=log2_mass_lift,
        global_frequency_lift=global_frequency_lift,
        cooccurrence_rate=cooccurrence_rate,
        n_target_timestamps=n_target_timestamps_arr,
        n_context_tokens=n_context_tokens_arr,
        n_context_timestamps=n_context_timestamps_arr,
        n_context_patients=n_context_patients_arr,
        vid_query_list=np.asarray(vid_query_list, dtype=np.int64),
        vid_context_list=np.asarray(vid_context_list, dtype=np.int64),
    )

    # Global context frequency for frequency-bias diagnostics.
    global_ctx_freq: Dict[int, float] = {}
    if total_context_tokens > 0:
        for cvid, f in freq.items():
            global_ctx_freq[cvid] = float(f) / float(total_context_tokens)

    # Per-target set size statistics.
    set_size_median: Dict[int, float] = {}
    set_size_mean: Dict[int, float] = {}
    for qvid, sizes in set_sizes.items():
        arr = np.asarray(sizes, dtype=np.float32)
        set_size_median[qvid] = float(np.median(arr)) if arr.size > 0 else float("nan")
        set_size_mean[qvid] = float(np.mean(arr)) if arr.size > 0 else float("nan")

    top_payload: Dict[str, Any] = {}
    top_k = int(args.top_k)
    min_count = int(args.min_count)
    for qvid in vid_query_list:
        qi = q2i[qvid]
        row_m = mean_alpha_present[qi]
        row_l = mass_lift[qi]
        row_lg = global_frequency_lift[qi]
        row_au = alpha_over_uniform[qi]
        row_ms = alpha_mass_share[qi]
        row_ex = expected_uniform_mass[qi]
        row_l2 = log2_mass_lift[qi]
        row_cr = cooccurrence_rate[qi]
        row_ntok = n_context_tokens_arr[qi]
        row_nts = n_context_timestamps_arr[qi]
        row_npt = n_context_patients_arr[qi]
        n_q = int(n_target_timestamps_arr[qi])
        valid_idx = np.where(row_m > 0)[0]
        if valid_idx.size == 0:
            top_payload[str(qvid)] = {
                "target_label": label_map.get(int(qvid), ""),
                "target_clinical_group": group_map.get(int(qvid), ""),
                "n_target_occurrences": int(n_q),
                "median_set_size": set_size_median.get(int(qvid), float("nan")),
                "mean_set_size": set_size_mean.get(int(qvid), float("nan")),
                "top_by_mean_alpha": [],
                "top_by_lift": [],
            }
            continue

        ord_m = valid_idx[np.argsort(-row_m[valid_idx])][:top_k]
        valid_lift_idx = np.asarray(
            [ci for ci in valid_idx.tolist() if int(row_ntok[int(ci)]) >= min_count],
            dtype=np.int64,
        )
        ord_l = valid_lift_idx[np.argsort(-row_l[valid_lift_idx])][:top_k] if valid_lift_idx.size > 0 else np.asarray([], dtype=np.int64)

        def pack(indices: np.ndarray, score_arr: np.ndarray) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for ci in indices.tolist():
                cvid = vid_context_list[ci]
                out.append({
                    "context_vid": int(cvid),
                    "context_label": label_map.get(int(cvid), ""),
                    "context_clinical_group": group_map.get(int(cvid), ""),
                    "source_table": source_map.get(int(cvid), ""),
                    "score": float(score_arr[ci]),
                    "mean_alpha": float(row_m[ci]),
                    "lift": float(row_l[ci]),
                    "alpha_count": int(row_ntok[ci]),
                    "context_frequency": global_ctx_freq.get(int(cvid), 0.0),
                    "alpha_over_uniform": float(row_au[ci]),
                    "alpha_mass_share": float(row_ms[ci]),
                    "expected_uniform_mass": float(row_ex[ci]),
                    "log2_mass_lift": float(row_l2[ci]),
                    "global_frequency_lift": float(row_lg[ci]),
                    "n_target_occurrences": int(n_q),
                    "n_context_timestamps": int(row_nts[ci]),
                    "n_context_patients": int(row_npt[ci]),
                    "cooccurrence_rate": float(row_cr[ci]),
                })
            return out

        top_payload[str(qvid)] = {
            "target_label": label_map.get(int(qvid), ""),
            "target_clinical_group": group_map.get(int(qvid), ""),
            "n_target_occurrences": int(n_q),
            "median_set_size": set_size_median.get(int(qvid), float("nan")),
            "mean_set_size": set_size_mean.get(int(qvid), float("nan")),
            "top_by_mean_alpha": pack(ord_m, row_m),
            "top_by_lift": pack(ord_l, row_l),
        }

    with open(os.path.join(args.output_dir, "top_context_per_query.json"), "w", encoding="utf-8") as f:
        json.dump(top_payload, f, indent=2, ensure_ascii=False)

    # Visualization: one panel per query, ranked by mass_lift only.
    def _display_query_label(s: str) -> str:
        x = (s or "").strip().replace("_", " ")
        lower = x.lower()
        if "creatinine" in lower and "serum" in lower:
            return "Serum creatinine"
        if len(x) > 70:
            x = x[:67] + "..."
        return x if x else "Target variable"

    def _display_context_label(s: str) -> str:
        x = (s or "").strip().replace("_", " ")
        for suffix in [" icu mgdl", " icu", " hosp", " mgdl", " mmol l", " mmhg", " bpm"]:
            if x.lower().endswith(suffix):
                x = x[: -len(suffix)].strip()
        if len(x) > 40:
            x = x[:37] + "..."
        return x if x else "Unknown context"

    for qvid in vid_query_list:
        qkey = str(qvid)
        blob = top_payload.get(qkey, {})
        query_label = _display_query_label(str(blob.get("target_label", "")) or f"VID {qvid}")
        top_l = blob.get("top_by_lift", []) or []

        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        labels_l = [_display_context_label(str(x.get("context_label", f"VID {x.get('context_vid', '')}"))) for x in top_l]
        vals_l = np.asarray([float(x.get("lift", 0.0)) for x in top_l], dtype=np.float32)
        cnts_l = [int(x.get("alpha_count", 0)) for x in top_l]

        if vals_l.size > 0:
            cmap = plt.get_cmap("Blues")
            vmax = float(np.max(vals_l)) if np.max(vals_l) > 0 else 1.0
            colors = [cmap(0.35 + 0.55 * (float(v) / vmax)) for v in vals_l]
            y = np.arange(len(vals_l))
            ax.barh(y, vals_l, color=colors, edgecolor="none")
            ax.set_yticks(y)
            ax.set_yticklabels(labels_l, fontsize=7.5)
            ax.invert_yaxis()
            xmax = float(np.max(vals_l)) if vals_l.size > 0 else 1.0
            ax.set_xlim(0.0, xmax * 1.18 if xmax > 0 else 1.0)
            for yi, (v, c) in enumerate(zip(vals_l.tolist(), cnts_l)):
                ax.text(float(v) + max(0.01 * xmax, 1e-3), yi, f"n={c}", va="center", ha="left", fontsize=7)
        else:
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No entries (count-filtered)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_xlim(0.0, 1.0)

        ax.set_xlabel("SetEncoder attention lift over uniform pooling  (protocol: local_set_encoder_attention)", fontsize=7.5)
        ax.set_ylabel("")
        ax.set_title(query_label, fontsize=10, loc="left", pad=8)
        ax.text(1.0, 1.03, f"n context tokens ≥ {min_count}", transform=ax.transAxes, fontsize=8, va="bottom", ha="right")
        ax.grid(axis="x", color="#cccccc", linewidth=0.5, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        safe_name = str(query_label).replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")[:60]
        fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
        fig.savefig(os.path.join(args.output_dir, f"cooccurrence_vid{qvid}_{safe_name}.png"),
                    dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    # Flat table with full pair-wise metrics for downstream filtering/audit.
    rows_out: List[Dict[str, Any]] = []
    for qvid in vid_query_list:
        qi = q2i[qvid]
        n_q = int(n_target_timestamps_arr[qi])
        for cvid in vid_context_list:
            ci = c2i[cvid]
            n_tok = int(n_context_tokens_arr[qi, ci])
            if n_tok <= 0:
                continue
            rows_out.append({
                "target_vid": int(qvid),
                "target_label": label_map.get(int(qvid), ""),
                "context_vid": int(cvid),
                "context_label": label_map.get(int(cvid), ""),
                "target_clinical_group": group_map.get(int(qvid), ""),
                "context_clinical_group": group_map.get(int(cvid), ""),
                "source_table": source_map.get(int(cvid), ""),
                "n_target_occurrences": int(n_q),
                "median_set_size": set_size_median.get(int(qvid), float("nan")),
                "mean_set_size": set_size_mean.get(int(qvid), float("nan")),
                "alpha_count": n_tok,
                "n_context_timestamps": int(n_context_timestamps_arr[qi, ci]),
                "n_context_patients": int(n_context_patients_arr[qi, ci]),
                "cooccurrence_rate": float(cooccurrence_rate[qi, ci]),
                "mean_alpha": float(mean_alpha_present[qi, ci]),
                "lift": float(mass_lift[qi, ci]),
                "context_frequency": global_ctx_freq.get(int(cvid), 0.0),
                "alpha_over_uniform": float(alpha_over_uniform[qi, ci]),
                "alpha_mass_share": float(alpha_mass_share[qi, ci]),
                "expected_uniform_mass": float(expected_uniform_mass[qi, ci]),
                "log2_mass_lift": float(log2_mass_lift[qi, ci]),
                "global_frequency_lift": float(global_frequency_lift[qi, ci]),
                "protocol": "local_set_encoder_attention",
                "same_timestamp_only": True,
            })
    table_cols = [
        "target_vid", "target_label", "context_vid", "context_label",
        "target_clinical_group", "context_clinical_group", "source_table",
        "n_target_occurrences", "median_set_size", "mean_set_size",
        "alpha_count", "n_context_timestamps", "n_context_patients",
        "cooccurrence_rate", "mean_alpha", "lift", "context_frequency",
        "alpha_over_uniform", "alpha_mass_share", "expected_uniform_mass",
        "log2_mass_lift", "global_frequency_lift",
        "protocol", "same_timestamp_only",
    ]
    df_table = pd.DataFrame(rows_out) if rows_out else pd.DataFrame(columns=table_cols)
    df_table.to_csv(os.path.join(args.output_dir, "cooccurrence_table.csv"), index=False)

    # Add per-target rank columns.
    if not df_table.empty:
        df_table["rank_by_mean_alpha"] = df_table.groupby("target_vid")["mean_alpha"].rank(
            ascending=False, method="min"
        ).astype(int)
        df_table["rank_by_lift"] = df_table.groupby("target_vid")["lift"].rank(
            ascending=False, method="min"
        ).astype(int)
        df_table.to_csv(os.path.join(args.output_dir, "cooccurrence_table.csv"), index=False)

    summary = {
        "timestamp": _attr_now_iso(),
        "protocol": "local_set_encoder_attention",
        "same_timestamp_only": True,
        "scientific_interpretation": (
            "Probes local same-timestamp set aggregation by withholding selected target "
            "variables from a state set and examining the gated-pooling attention (alpha) "
            "assigned to the remaining same-timestamp observations. "
            "This characterizes local co-measurement attention, not causal feature importance "
            "for imputation or temporal attribution of any readout prediction."
        ),
        "phase1_ckpt": os.path.abspath(args.phase1_ckpt),
        "data_dir": os.path.abspath(args.data_dir),
        "schema_dir": os.path.abspath(args.schema_dir),
        "split": str(args.split),
        "target_vids": [int(x) for x in target_vids],
        "max_patients": None if args.max_patients is None else int(args.max_patients),
        "top_k": int(args.top_k),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "n_queries": int(len(vid_query_list)),
        "n_context_vids": int(len(vid_context_list)),
        "total_context_tokens": int(total_context_tokens),
        "plotting": {
            "plotted_metric": "lift",
            "metric_definition": "observed target-specific pooling mass share divided by uniform-pooling expected mass over the same target-masked timestamps",
            "count_filter": f"alpha_count >= {int(min_count)}",
            "figure_layout": "single_panel_lift_per_target",
        },
        "metric_definitions": {
            "mean_alpha": "conditional mean gated-pooling weight when context token is visible and target is withheld",
            "lift": "observed pooling mass share divided by target-specific uniform expected mass over the same target-masked timestamps",
            "alpha_count": "number of times context token was visible at a target-masked timestamp",
            "context_frequency": "global token frequency of context variable across all target-masked timestamps",
            "alpha_over_uniform": "conditional mean of alpha * S_t, where S_t is visible context set size",
            "global_frequency_lift": "mean_alpha / global_context_frequency (exploratory; not used for primary ranking)",
        },
        "validation": {
            "alpha_mass_total_mean_by_query": {str(k): float(v) for k, v in alpha_mass_total_mean_by_query.items()},
            "alpha_mass_expected_close_to_1_note": "Values near 1 indicate pooled alpha mass conservation over visible tokens after target masking.",
        },
        "patch": "SetEncoder.forward has formal return_attn arg (no monkey patch needed)",
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def _attr_parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "SetEncoder local attention analysis (protocol: local_set_encoder_attention). "
            "Probes same-timestamp set aggregation by withholding target variables and "
            "examining gated-pooling attention on remaining observations."
        )
    )
    p.add_argument("--phase1_ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--schema_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--target_vids", type=int, nargs="+", required=True)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--min_count", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    return p.parse_args()

def _attr_main() -> None:
    args = _attr_parse_args()
    out = _attr_run(args)
    print(json.dumps(out, indent=2, ensure_ascii=False))
