"""Phase-1 target-manifold numeric readout.

This module is deliberately non-parametric.  Numeric values are decoded by
inverting the frozen EMA teacher event manifold used by Phase-1.
"""

from __future__ import annotations

import json
import math
import os
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm


READOUT_ARCHITECTURE = "phase1_target_manifold_inverse_v1"




@torch.inference_mode()
def evaluate_readout_streaming(readout: "Phase1TargetManifoldReadout", loader: Any, device: torch.device) -> Dict[str, Any]:
    """Evaluate a query cache loader without materialising a split in RAM."""
    sums = {"sq": 0.0, "abs": 0.0, "n": 0}
    readout = readout.to(device).eval()
    for batch in tqdm(loader, desc="readout_eval", ascii=True, mininterval=5.0):
        z = batch["z_pred"].to(device, non_blocking=True).float()
        fid = batch["feature_id"].to(device, non_blocking=True).long()
        target = batch["target_value"].to(device, non_blocking=True).float()
        with torch.inference_mode():
            out = readout(z, fid)
        err = out["numeric_mean"] - target
        sums["sq"] += float(err.square().sum().detach().cpu())
        sums["abs"] += float(err.abs().sum().detach().cpu())
        sums["n"] += int(err.numel())
    if not sums["n"]:
        raise RuntimeError("readout evaluation loader is empty")
    return {"mse": sums["sq"] / sums["n"], "mae": sums["abs"] / sums["n"], "n": sums["n"]}


def evaluate_readout_per_vid_streaming(readout: "Phase1TargetManifoldReadout", loader: Any, device: torch.device, *, latent_key: str = "z_pred") -> Dict[int, Dict[str, float]]:
    acc: Dict[int, Dict[str, float]] = {}
    for batch in tqdm(loader, desc=f"readout_per_vid/{latent_key}", ascii=True, mininterval=5.0):
        z=batch[latent_key].to(device, non_blocking=True).float(); f=batch["feature_id"].to(device, non_blocking=True).long(); y=batch["target_value"].to(device, non_blocking=True).float(); out=readout(z,f); e=out["numeric_mean"]-y
        for vid in f.unique().tolist():
            m=f==int(vid); a=acc.setdefault(int(vid),{"count":0.,"sq":0.,"abs":0.}); a["count"]+=float(m.sum()); a["sq"]+=float(e[m].square().sum().detach().cpu()); a["abs"]+=float(e[m].abs().sum().detach().cpu())
    return acc


def _stream_vid_stats(loader: Any, latent_key: str) -> Dict[int, Dict[str, Any]]:
    """Accumulate exact sufficient statistics without concatenating a split."""
    stats: Dict[int, Dict[str, Any]] = {}
    for batch in tqdm(loader, desc=f"readout_ridge_stats/{latent_key}", ascii=True, mininterval=5.0):
        x = batch[latent_key].detach().float().cpu().numpy()
        y = batch["target_value"].detach().float().cpu().numpy().reshape(-1)
        f = batch["feature_id"].detach().long().cpu().numpy().reshape(-1)
        if x.ndim != 2 or len(y) != len(f) or x.shape[0] != len(y):
            raise RuntimeError("readout decoder batch identity/latent shape mismatch")
        for vid in np.unique(f):
            m = f == vid
            row = stats.setdefault(int(vid), {"n": 0, "sum_x": np.zeros(x.shape[1], dtype=np.float64), "xtx": np.zeros((x.shape[1], x.shape[1]), dtype=np.float64), "xty": np.zeros(x.shape[1], dtype=np.float64), "y2": 0.0, "sum_y": 0.0})
            xx, yy = x[m].astype(np.float64), y[m].astype(np.float64)
            finite = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
            xx, yy = xx[finite], yy[finite]
            row["n"] += int(len(yy)); row["sum_x"] += xx.sum(axis=0); row["xtx"] += xx.T @ xx; row["xty"] += xx.T @ yy
            row["y2"] += float(yy @ yy); row["sum_y"] += float(yy.sum())
    return stats


def fit_per_vid_ridge_streaming(train_loader: Any, val_loader: Any, *, latent_key: str,
                                 alpha_grid: Sequence[float], min_train_count: int = 100,
                                 min_val_count: int = 20) -> Dict[int, Dict[str, Any]]:
    """Fit per-VID Ridge using train sufficient statistics and validation-only alpha."""
    train = _stream_vid_stats(train_loader, latent_key)
    val = _stream_vid_stats(val_loader, latent_key)
    models: Dict[int, Dict[str, Any]] = {}
    for vid, tr in tqdm(train.items(), desc=f"readout_ridge_fit/{latent_key}", ascii=True, mininterval=5.0):
        va = val.get(vid, {})
        if tr["n"] < int(min_train_count) or va.get("n", 0) < int(min_val_count):
            models[vid] = {"status": "ineligible", "train_count": tr["n"], "val_count": va.get("n", 0), "selected_alpha": float("nan")}
            continue
        d = tr["xtx"].shape[0]
        eye = np.zeros((d + 1, d + 1), dtype=np.float64); eye[1:, 1:] = np.eye(d)
        def augmented(s):
            a = np.zeros((d + 1, d + 1), dtype=np.float64); a[0, 0] = s["n"]; a[0, 1:] = a[1:, 0] = s["sum_x"]; a[1:, 1:] = s["xtx"]; b = np.r_[s["sum_y"], s["xty"]]; return a, b
        tr_xx, tr_xy = augmented(tr); va_xx, va_xy = augmented(va)
        best = None
        for alpha in alpha_grid:
            beta = np.linalg.solve(tr_xx + float(alpha) * eye, tr_xy)
            mse = (beta @ va_xx @ beta - 2.0 * beta @ va_xy + va["y2"]) / max(1, va["n"])
            if best is None or mse < best[0]: best = (float(mse), float(alpha), beta)
        models[vid] = {"status": "ok", "train_count": tr["n"], "val_count": va["n"], "selected_alpha": best[1], "intercept": float(best[2][0]), "coef": best[2][1:], "val_mse": best[0]}
    return models


def clustered_bootstrap_delta(rows: Sequence[Mapping[str, Any]], left: str, right: str, *, iterations: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Admission-clustered paired bootstrap for two prediction columns."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows: groups.setdefault(str(row["hadm_id"]), []).append(row)
    if not groups: return {"status": "not_evaluable", "paired_patient_count": 0}
    rng = np.random.default_rng(int(seed)); ids = list(groups); estimates=[]
    point=[]
    for group in groups.values():
        for r in group:
            a,b=float(r.get(left,np.nan)),float(r.get(right,np.nan))
            if np.isfinite(a) and np.isfinite(b): point.append((a-float(r["target_value"]))**2-(b-float(r["target_value"]))**2)
    if not point: return {"status": "not_evaluable", "paired_patient_count": len(ids)}
    for _ in range(int(iterations)):
        sample=rng.choice(ids,size=len(ids),replace=True); vals=[]
        for gid in sample:
            for r in groups[str(gid)]:
                a,b=float(r.get(left,np.nan)),float(r.get(right,np.nan)); y=float(r["target_value"])
                if np.isfinite(a) and np.isfinite(b): vals.append((a-y)**2-(b-y)**2)
        if vals: estimates.append(float(np.mean(vals)))
    arr=np.asarray(estimates,float); return {"status":"ok","delta":float(np.mean(point)),"ci_low":float(np.quantile(arr,.025)),"ci_high":float(np.quantile(arr,.975)),"win_probability":float(np.mean(arr<0)),"valid_replicates":int(len(arr)),"paired_patient_count":len(ids)}


def audit_readout_parquet(path: str, expected_rows: Optional[int] = None) -> Dict[str, Any]:
    """Validate identity and recompute core MSE metrics from row groups."""
    pf = pq.ParquetFile(path)
    required = {
        "hadm_id", "t_idx", "slot_idx", "feature_id", "query_key", "target_value",
        "numeric_mean", "numeric_sigma", "train_mean_prediction", "locf_value",
        "locf_available", "locf_plus_train_mean_prediction", "strict_locf_prediction",
        "target_learned_decoder_prediction", "prediction_learned_decoder_prediction",
        "target_decoder_transfer_prediction", "hard_oracle_prediction", "sharp_oracle_mean",
        "sharp_oracle_sigma", "calibrated_oracle_mean", "calibrated_oracle_sigma",
        "latent_l1", "latent_l2", "latent_cosine", "latent_norm_ratio",
        "tangent_error", "squared_tangent_error", "orthogonal_error",
        "tangent_total_ratio", "min_energy", "posterior_entropy", "boundary_mass",
    }
    missing = required.difference(pf.schema.names)
    if missing: raise RuntimeError(f"readout parquet missing required columns: {sorted(missing)}")
    seen=set(); n=0; sq=0.; abs_err=0.
    for table in pf.iter_batches():
        frame=table.to_pydict(); m=len(frame["target_value"]); n += m
        for i in range(m):
            key=(str(frame["hadm_id"][i]), int(frame["t_idx"][i]), int(frame["slot_idx"][i]), int(frame["feature_id"][i]))
            if str(frame["query_key"][i]) != str(key): raise RuntimeError(f"query_key identity mismatch in parquet: {key}")
            if key in seen: raise RuntimeError(f"duplicate query key in parquet: {key}")
            seen.add(key); e=float(frame["numeric_mean"][i])-float(frame["target_value"][i]); sq += e*e; abs_err += abs(e)
    if expected_rows is not None and n != int(expected_rows): raise RuntimeError(f"parquet row count mismatch: found={n} expected={expected_rows}")
    return {"row_count": n, "unique_query_keys": len(seen), "mse": sq/n if n else float("nan"), "mae": abs_err/n if n else float("nan"), "columns": pf.schema.names}


@torch.inference_mode()
def run_frozen_readout_evaluator(*, readout: "Phase1TargetManifoldReadout", decoder_payload: Mapping[str, Any], loader: Any, device: torch.device, output_dir: str, support: Mapping[int, Mapping[str, Any]], config: Mapping[str, Any], expected_rows: int) -> Dict[str, Any]:
    """Single frozen test evaluator used by both formal and eval-only runs.

    This is intentionally the only writer for final test diagnostics.  Every
    emitted table is accumulated from the same query rows.
    """
    os.makedirs(output_dir, exist_ok=True)
    readout=readout.to(device).eval(); target_models=decoder_payload.get("target",{}); pred_models=decoder_payload.get("prediction",{})
    rcfg=config.get("readout",{}); sharp=float(rcfg.get("oracle",{}).get("sharp_temperature",1e-4))
    methods=("numeric_mean","train_mean_prediction","locf_plus_train_mean_prediction","strict_locf_prediction","target_learned_decoder_prediction","prediction_learned_decoder_prediction","target_decoder_transfer_prediction","hard_oracle_prediction","sharp_oracle_mean","calibrated_oracle_mean")
    def new_acc(): return {"n":0,"sq":0.,"abs":0.,"y":0.,"y2":0.}
    def new_method_acc(): return {m:new_acc() for m in methods}
    acc={m:new_acc() for m in methods}; acc["numeric_mean"].update({"nll":0.,"cov68":0.,"cov95":0.,"sigma":0.,"width":0.,"entropy":0.,"energy":0.,"boundary":0.})
    per={}; strata={"all_numeric":new_method_acc(),"dynamic_only":new_method_acc(),"locf_available":new_method_acc(),"locf_unavailable":new_method_acc()}; support_acc={"<100":new_method_acc(),"100-999":new_method_acc(),"1000-9999":new_method_acc(),">=10000":new_method_acc()}; admission={}; writer=None; path=os.path.join(output_dir,"readout_predictions_test.parquet")
    static_vids={int(v) for v in rcfg.get("static_numeric_vids", config.get("phase1",{}).get("recipe",{}).get("static_numeric_vids", [5921]))}
    def update(a, pred, target):
        if pred is None or not np.isfinite(pred): return
        e=float(pred)-float(target); a["n"]+=1; a["sq"]+=e*e; a["abs"]+=abs(e); a["y"]+=float(target); a["y2"]+=float(target)*float(target)
    def finish(a):
        n=a["n"]; mse=a["sq"]/n if n else float("nan"); var=a["y2"]-a["y"]*a["y"]/n if n else float("nan")
        return {"count":n,"mse":mse,"rmse":float(np.sqrt(mse)) if np.isfinite(mse) else float("nan"),"mae":a["abs"]/n if n else float("nan"),"r2":1.-a["sq"]/var if var>0 else float("nan")}
    for batch in tqdm(loader, desc="readout_test_export", ascii=True, mininterval=5.0):
        z=batch["z_pred"].float().to(device,non_blocking=True); zt=batch["z_target"].float().to(device,non_blocking=True); fid=batch["feature_id"].long().to(device,non_blocking=True); y=batch["target_value"].float().to(device,non_blocking=True)
        pred=readout(z,fid); geom=readout.tangent_geometry(z,zt,y,fid); hard=readout.hard_decode(zt,fid); sharp_out=readout.decode_with_temperature(zt,fid,sharp); calibrated=readout(zt,fid)
        pred={k:v.detach().cpu() for k,v in pred.items()}; geom={k:v.detach().cpu() for k,v in geom.items()}; hard={k:v.detach().cpu() for k,v in hard.items()}; sharp_out={k:v.detach().cpu() for k,v in sharp_out.items()}; calibrated={k:v.detach().cpu() for k,v in calibrated.items()}; z=z.cpu(); zt=zt.cpu(); fid=fid.cpu(); y=y.cpu(); batch_rows=[]
        z_np_all=z.numpy(); zt_np_all=zt.numpy(); fid_np=fid.numpy(); n_batch=len(y)
        target_dec=np.full((n_batch,), np.nan, dtype=np.float64); pred_dec=np.full((n_batch,), np.nan, dtype=np.float64); transfer_dec=np.full((n_batch,), np.nan, dtype=np.float64)
        for v_unique in np.unique(fid_np):
            mask=fid_np==int(v_unique); tm=target_models.get(int(v_unique),{}); pm=pred_models.get(int(v_unique),{})
            if tm.get("status")=="ok":
                target_dec[mask]=zt_np_all[mask] @ np.asarray(tm["coef"], dtype=np.float64) + float(tm.get("intercept",0.))
                transfer_dec[mask]=z_np_all[mask] @ np.asarray(tm["coef"], dtype=np.float64) + float(tm.get("intercept",0.))
            if pm.get("status")=="ok":
                pred_dec[mask]=z_np_all[mask] @ np.asarray(pm["coef"], dtype=np.float64) + float(pm.get("intercept",0.))
        delta_np=z_np_all-zt_np_all; latent_l1=np.abs(delta_np).mean(axis=1); latent_l2=np.linalg.norm(delta_np, axis=1); z_norm=np.linalg.norm(z_np_all, axis=1); zt_norm=np.linalg.norm(zt_np_all, axis=1); latent_cosine=(z_np_all*zt_np_all).sum(axis=1)/(z_norm*zt_norm+1e-12); latent_norm_ratio=z_norm/(zt_norm+1e-12)
        for i in range(len(y)):
            v=int(fid[i]); tm=target_models.get(v,{}); pm=pred_models.get(v,{}); locf_ok=bool(batch["locf_available"][i]); locf=float(batch["locf_value"][i]) if locf_ok else float("nan"); mean=float(support.get(v,{}).get("mean",float("nan"))); hadm=str(batch["hadm_id"][i]); key=(hadm,int(batch["t_idx"][i]),int(batch["slot_idx"][i]),v); total=float((z[i]-zt[i]).norm())
            row={"hadm_id":hadm,"t_idx":int(batch["t_idx"][i]),"slot_idx":int(batch["slot_idx"][i]),"feature_id":v,"query_key":str(key),"target_value":float(y[i]),"numeric_mean":float(pred["numeric_mean"][i]),"numeric_sigma":float(pred["numeric_sigma"][i]),"train_mean_prediction":mean,"locf_value":locf,"locf_available":locf_ok,"locf_plus_train_mean_prediction":locf if locf_ok else mean,"strict_locf_prediction":locf if locf_ok else float("nan"),"target_learned_decoder_prediction":float(target_dec[i]),"prediction_learned_decoder_prediction":float(pred_dec[i]),"target_decoder_transfer_prediction":float(transfer_dec[i]),"hard_oracle_prediction":float(hard["numeric_mean"][i]),"sharp_oracle_mean":float(sharp_out["numeric_mean"][i]),"sharp_oracle_sigma":float(sharp_out["numeric_sigma"][i]),"calibrated_oracle_mean":float(calibrated["numeric_mean"][i]),"calibrated_oracle_sigma":float(calibrated["numeric_sigma"][i]),"min_energy":float(pred["min_energy"][i]),"posterior_entropy":float(pred["posterior_entropy"][i]),"boundary_mass":float(pred["boundary_mass"][i]),"latent_l1":float(latent_l1[i]),"latent_l2":float(latent_l2[i]),"latent_cosine":float(latent_cosine[i]),"latent_norm_ratio":float(latent_norm_ratio[i]),"tangent_error":float(geom["tangent_error"][i]),"squared_tangent_error":float(geom["squared_tangent_error"][i]),"orthogonal_error":float(geom["orthogonal_error"][i]),"tangent_total_ratio":float(geom["tangent_error"][i]/max(total,1e-12))}
            batch_rows.append(row); a=per.setdefault(v,{"count":0,"target":{m:new_acc() for m in methods},"tangent":[]}); a["count"]+=1; a["tangent"].append((row["tangent_error"],row["squared_tangent_error"],row["orthogonal_error"],row["tangent_total_ratio"]))
            for m in methods:
                update(acc[m], row[m], row["target_value"]); update(per[v]["target"][m], row[m], row["target_value"])
            for key_name, eligible in (("all_numeric", True), ("dynamic_only", v not in static_vids), ("locf_available", locf_ok), ("locf_unavailable", not locf_ok)):
                if eligible:
                    for m in methods: update(strata[key_name][m], row[m], row["target_value"])
            support_key="<100" if float(support.get(v,{}).get("count",0))<100 else "100-999" if float(support.get(v,{}).get("count",0))<1000 else "1000-9999" if float(support.get(v,{}).get("count",0))<10000 else ">=10000"
            for m in methods: update(support_acc[support_key][m], row[m], row["target_value"])
            q=acc["numeric_mean"]; sig=row["numeric_sigma"]; err=abs(row["numeric_mean"]-row["target_value"]); q["nll"] += .5*(err/max(sig,1e-8))**2+np.log(max(sig,1e-8)); q["cov68"] += float(err<=sig); q["cov95"] += float(err<=1.9599639845*sig); q["sigma"] += sig; q["width"] += 2*1.9599639845*sig; q["entropy"] += row["posterior_entropy"]; q["energy"] += row["min_energy"]; q["boundary"] += row["boundary_mass"]
            ag=admission.setdefault(hadm,{"n":0,"manifold_minus_mean":0.,"pred_minus_mean":0.,"pred_minus_manifold":0.,"manifold_minus_locf_mean":0.,"pred_minus_locf_mean":0.,"manifold_minus_locf":0.,"pred_minus_locf":0.,"strict_n":0})
            y_i=row["target_value"]
            if np.isfinite(row["numeric_mean"]) and np.isfinite(row["train_mean_prediction"]): ag["manifold_minus_mean"]+=(row["numeric_mean"]-y_i)**2-(row["train_mean_prediction"]-y_i)**2
            if np.isfinite(row["prediction_learned_decoder_prediction"]) and np.isfinite(row["train_mean_prediction"]): ag["pred_minus_mean"]+=(row["prediction_learned_decoder_prediction"]-y_i)**2-(row["train_mean_prediction"]-y_i)**2
            if np.isfinite(row["prediction_learned_decoder_prediction"]) and np.isfinite(row["numeric_mean"]): ag["pred_minus_manifold"]+=(row["prediction_learned_decoder_prediction"]-y_i)**2-(row["numeric_mean"]-y_i)**2
            if np.isfinite(row["numeric_mean"]) and np.isfinite(row["locf_plus_train_mean_prediction"]): ag["manifold_minus_locf_mean"]+=(row["numeric_mean"]-y_i)**2-(row["locf_plus_train_mean_prediction"]-y_i)**2
            if np.isfinite(row["prediction_learned_decoder_prediction"]) and np.isfinite(row["locf_plus_train_mean_prediction"]): ag["pred_minus_locf_mean"]+=(row["prediction_learned_decoder_prediction"]-y_i)**2-(row["locf_plus_train_mean_prediction"]-y_i)**2
            if locf_ok:
                ag["strict_n"]+=1
                if np.isfinite(row["numeric_mean"]): ag["manifold_minus_locf"]+=(row["numeric_mean"]-y_i)**2-(locf-y_i)**2
                if np.isfinite(row["prediction_learned_decoder_prediction"]): ag["pred_minus_locf"]+=(row["prediction_learned_decoder_prediction"]-y_i)**2-(locf-y_i)**2
            ag["n"]+=1
        table=pa.Table.from_pylist(batch_rows)
        if writer is None: writer=pq.ParquetWriter(path,table.schema)
        writer.write_table(table)
    if writer is not None: writer.close()
    audit=audit_readout_parquet(path,expected_rows=expected_rows)
    metrics={m:finish(q) for m,q in acc.items()}
    q=acc["numeric_mean"]; n=q["n"]; metrics["numeric_mean"].update({"gaussian_nll":q["nll"]/n if n else float("nan"),"coverage68":q["cov68"]/n if n else float("nan"),"coverage95":q["cov95"]/n if n else float("nan"),"mean_sigma":q["sigma"]/n if n else float("nan"),"interval_width":q["width"]/n if n else float("nan"),"entropy":q["entropy"]/n if n else float("nan"),"min_energy":q["energy"]/n if n else float("nan"),"boundary_mass":q["boundary"]/n if n else float("nan")})
    if not np.isclose(float(audit["mse"]), float(metrics["numeric_mean"]["mse"]), rtol=1e-6, atol=1e-8):
        raise RuntimeError("parquet/metrics parity failed for numeric_mean")
    per_rows=[]; oracle_rows=[]
    for v,a in sorted(per.items()):
        n=max(1,a["count"]); geom_rows=np.asarray(a["tangent"],dtype=float); row={"feature_id":v,"count":a["count"],"mean_tangent_error":float(geom_rows[:,0].mean()),"median_tangent_error":float(np.median(geom_rows[:,0])),"mean_squared_tangent_error":float(geom_rows[:,1].mean()),"mean_orthogonal_error":float(geom_rows[:,2].mean()),"mean_tangent_total_ratio":float(geom_rows[:,3].mean())}
        for method in methods:
            fm=finish(a["target"][method]); row[f"{method}_mse"]=fm["mse"]; row[f"{method}_mae"]=fm["mae"]; row[f"{method}_r2"]=fm["r2"]
        per_rows.append(row)
        oracle_rows.append({"feature_id":v,"count":a["count"],**{f"{m}_{k}": value for m in ("hard_oracle_prediction","sharp_oracle_mean","calibrated_oracle_mean") for k,value in finish(a["target"][m]).items()}})
    macro_min_count = int(rcfg.get("macro_min_count", 10))
    for method in methods:
        eligible = [r for r in per_rows if int(r["count"]) >= macro_min_count and np.isfinite(r.get(f"{method}_mse", np.nan))]
        r2_values = [float(r[f"{method}_r2"]) for r in eligible if np.isfinite(r.get(f"{method}_r2", np.nan))]
        metrics[method]["macro"] = {
            "eligible_vid_count": len(eligible),
            "macro_min_count": macro_min_count,
            "mse": float(np.mean([r[f"{method}_mse"] for r in eligible])) if eligible else float("nan"),
            "mae": float(np.mean([r[f"{method}_mae"] for r in eligible])) if eligible else float("nan"),
            "r2": float(np.mean(r2_values)) if r2_values else float("nan"),
            "r2_eligible_vid_count": len(r2_values),
        }
        metrics[method]["macro"]["rmse"] = float(np.sqrt(metrics[method]["macro"]["mse"])) if np.isfinite(metrics[method]["macro"]["mse"]) else float("nan")
    with open(os.path.join(output_dir,"readout_test_metrics.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    pd.DataFrame([{"method":m,**v} for m,v in metrics.items()]).to_csv(os.path.join(output_dir,"readout_baseline_comparison.csv"),index=False)
    pd.DataFrame(per_rows).to_csv(os.path.join(output_dir,"readout_per_feature_test.csv"),index=False)
    pd.DataFrame(per_rows).to_csv(os.path.join(output_dir,"readout_latent_geometry_metrics.csv"),index=False)
    pd.DataFrame([{"feature_id":r["feature_id"],"count":r["count"],"target_decoder_transfer_mse":r.get("target_decoder_transfer_prediction_mse"),"target_decoder_transfer_mae":r.get("target_decoder_transfer_prediction_mae"),"target_decoder_transfer_r2":r.get("target_decoder_transfer_prediction_r2")} for r in per_rows]).to_csv(os.path.join(output_dir,"readout_decoder_transfer_metrics.csv"),index=False)
    pd.DataFrame(oracle_rows).to_csv(os.path.join(output_dir,"oracle_inversion_per_feature_test.csv"),index=False)
    strata_out={name:{"count":a["numeric_mean"]["n"],"metrics":{m:finish(a[m]) for m in methods}} for name,a in strata.items()}
    support_strata={"<100":[],"100-999":[],"1000-9999":[],">=10000":[]}
    for v in per:
        c=float(support.get(v,{}).get("count",0)); support_strata["<100" if c<100 else "100-999" if c<1000 else "1000-9999" if c<10000 else ">=10000"].append(v)
    strata_out["support_strata"]={k:{"feature_ids":vals,"count":support_acc[k]["numeric_mean"]["n"],"metrics":{m:finish(support_acc[k][m]) for m in methods}} for k,vals in support_strata.items()}
    with open(os.path.join(output_dir,"readout_metrics_by_stratum.json"),"w",encoding="utf-8") as f: json.dump(strata_out,f,indent=2)
    with open(os.path.join(output_dir,"readout_oracle_metrics.json"),"w",encoding="utf-8") as f: json.dump({m:metrics[m] for m in ("hard_oracle_prediction","sharp_oracle_mean","calibrated_oracle_mean")},f,indent=2)
    with open(os.path.join(output_dir,"readout_calibration_test.json"),"w",encoding="utf-8") as f: json.dump({"numeric_mean":metrics["numeric_mean"],"selected_on":"validation"},f,indent=2)
    with open(os.path.join(output_dir,"readout_output_audit.json"),"w",encoding="utf-8") as f: json.dump(audit,f,indent=2)
    rng=np.random.default_rng(42); ids=list(admission)
    def bootstrap(field, count_field="n"):
        if not ids or not any(admission[k][count_field] for k in ids): return {"status":"not_evaluable","paired_admission_count":len(ids)}
        estimates=[]
        for _ in range(1000):
            draw=rng.choice(ids,size=len(ids),replace=True); den=sum(admission[k][count_field] for k in draw); estimates.append(sum(admission[k][field] for k in draw)/den if den else np.nan)
        arr=np.asarray(estimates,float); arr=arr[np.isfinite(arr)]
        den=sum(admission[k][count_field] for k in ids); point=sum(admission[k][field] for k in ids)/den
        return {"status":"ok","iterations":1000,"seed":42,"delta":float(point),"ci_low":float(np.quantile(arr,.025)),"ci_high":float(np.quantile(arr,.975)),"win_probability":float(np.mean(arr<0)),"valid_replicates":int(arr.size),"paired_admission_count":len(ids),"paired_query_count":int(den)}
    bootstrap_payload={"manifold_minus_train_mean":bootstrap("manifold_minus_mean"),"prediction_decoder_minus_train_mean":bootstrap("pred_minus_mean"),"manifold_minus_locf_plus_train_mean":bootstrap("manifold_minus_locf_mean"),"prediction_decoder_minus_locf_plus_train_mean":bootstrap("pred_minus_locf_mean"),"prediction_decoder_minus_manifold":bootstrap("pred_minus_manifold"),"manifold_minus_strict_locf":bootstrap("manifold_minus_locf","strict_n"),"prediction_decoder_minus_strict_locf":bootstrap("pred_minus_locf","strict_n")}
    with open(os.path.join(output_dir,"readout_paired_bootstrap.json"),"w",encoding="utf-8") as f: json.dump(bootstrap_payload,f,indent=2)
    with open(os.path.join(output_dir,"readout_mechanism_diagnosis.json"),"w",encoding="utf-8") as f:
        inv=float(metrics["numeric_mean"]["mse"]); pred_m=float(metrics["prediction_learned_decoder_prediction"]["mse"]); transfer=float(metrics["target_decoder_transfer_prediction"]["mse"])
        mechanism="coordinate_mismatch" if pred_m < inv*.5 and transfer > pred_m*2 else "information_loss" if pred_m >= inv*.5 else "manifold_inverse_deficiency" if pred_m < inv*.5 and transfer < inv*.5 else "inconclusive"
        json.dump({"primary_mechanism":mechanism,"supporting_metrics":{"target_manifold_inverse":metrics["numeric_mean"],"prediction_learned_decoder":metrics["prediction_learned_decoder_prediction"],"target_decoder_transfer":metrics["target_decoder_transfer_prediction"]},"limitations":["Predeclared diagnostic rules are associative and not causal."]},f,indent=2)
    pd.DataFrame([{"feature_id":int(v),**{k:value for k,value in row.items() if k not in {"coef"}}} for v,row in pred_models.items()]).to_csv(os.path.join(output_dir,"readout_learned_decoder_metrics.csv"),index=False)
    pd.DataFrame([{"feature_id":int(v),"selected_alpha":row.get("selected_alpha"),"train_count":row.get("train_count"),"val_count":row.get("val_count"),"status":row.get("status")} for v,row in target_models.items()]).to_csv(os.path.join(output_dir,"readout_decoder_selected_alphas.csv"),index=False)
    pd.DataFrame([{"feature_id":int(v),**row} for v,row in support.items()]).to_csv(os.path.join(output_dir,"train_support_summary.csv"),index=False)
    return {"metrics":metrics,"audit":audit,"bootstrap":bootstrap_payload}


def scan_train_numeric_support(dataset: Any, numeric_vids: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """Stream all unmasked train observations and retain only sufficient statistics."""
    wanted = {int(v) for v in numeric_vids}
    stats = {v: {"min": float("inf"), "max": float("-inf"), "count": 0.0, "sum": 0.0, "sum_sq": 0.0} for v in sorted(wanted)}
    wanted_tensor = torch.tensor(sorted(wanted), dtype=torch.long)
    for index in tqdm(range(len(dataset)), desc="readout_train_support", ascii=True, mininterval=5.0):
        item = dataset[index]
        tokens = item.get("state_tokens_flat")
        values = item.get("state_values_flat")
        if tokens is None or values is None:
            for timestamp in item["state"]:
                tokens = timestamp["tokens"].reshape(-1).long()
                values = timestamp["values"].reshape(-1).float()
                finite = torch.isfinite(values)
                if tokens.numel():
                    keep = finite & torch.isin(tokens, wanted_tensor)
                    tokens, values = tokens[keep], values[keep]
                    for vid in tokens.unique().tolist():
                        selected = values[tokens == int(vid)].float()
                        row = stats[int(vid)]
                        row["min"] = min(row["min"], float(selected.min()))
                        row["max"] = max(row["max"], float(selected.max()))
                        row["count"] += float(selected.numel())
                        row["sum"] += float(selected.sum())
                        row["sum_sq"] += float(selected.square().sum())
            continue
        tokens = tokens.reshape(-1).long()
        values = values.reshape(-1).float()
        if not tokens.numel():
            continue
        keep = torch.isfinite(values) & torch.isin(tokens, wanted_tensor)
        tokens, values = tokens[keep], values[keep]
        for vid in tokens.unique().tolist():
            selected = values[tokens == int(vid)].float()
            row = stats[int(vid)]
            row["min"] = min(row["min"], float(selected.min()))
            row["max"] = max(row["max"], float(selected.max()))
            row["count"] += float(selected.numel())
            row["sum"] += float(selected.sum())
            row["sum_sq"] += float(selected.square().sum())
    for row in stats.values():
        if row["count"]:
            row["mean"] = row["sum"] / row["count"]
    return {vid: row for vid, row in stats.items() if row["count"] > 0}


class Phase1TargetManifoldReadout(nn.Module):
    """Invert a frozen Phase-1 teacher manifold for numeric VIDs."""

    def __init__(
        self,
        *,
        grid_values: Tensor,
        grid_latents: Tensor,
        grid_valid_mask: Tensor,
        numeric_vids: Tensor,
        vid_to_row: Tensor,
        temperature: float = 1.0,
        sigma_scale: float = 1.0,
        sigma_floor: float = 1.0e-4,
        query_chunk_size: int = 2048,
        grid_chunk_size: int = 513,
        support_min: Optional[Tensor] = None,
        support_max: Optional[Tensor] = None,
        support_count: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if grid_values.ndim != 2 or grid_latents.ndim != 3:
            raise ValueError("manifold grids must have shapes [F,G] and [F,G,D]")
        if tuple(grid_latents.shape[:2]) != tuple(grid_values.shape):
            raise ValueError("grid value/latent shapes are inconsistent")
        if tuple(grid_valid_mask.shape) != tuple(grid_values.shape):
            raise ValueError("grid_valid_mask shape is inconsistent")
        if numeric_vids.ndim != 1 or vid_to_row.ndim != 1:
            raise ValueError("numeric_vids and vid_to_row must be one-dimensional")
        if numeric_vids.numel() != grid_values.size(0):
            raise ValueError("numeric_vids does not match manifold rows")
        self.register_buffer("grid_values", grid_values.detach().float().contiguous())
        self.register_buffer("grid_latents", grid_latents.detach().float().contiguous())
        self.register_buffer("grid_valid_mask", grid_valid_mask.detach().bool().contiguous())
        self.register_buffer("numeric_vids", numeric_vids.detach().long().contiguous())
        self.register_buffer("vid_to_row", vid_to_row.detach().long().contiguous())
        valid_min = grid_values.masked_fill(~grid_valid_mask, float("inf")).min(dim=1).values
        valid_max = grid_values.masked_fill(~grid_valid_mask, float("-inf")).max(dim=1).values
        self.register_buffer("support_min", (support_min if support_min is not None else valid_min).detach().float())
        self.register_buffer("support_max", (support_max if support_max is not None else valid_max).detach().float())
        self.register_buffer("support_count", (support_count if support_count is not None else grid_valid_mask.long().sum(dim=1)).detach().long())
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))
        self.register_buffer("sigma_scale", torch.tensor(float(sigma_scale), dtype=torch.float32))
        self.sigma_floor = float(sigma_floor)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.grid_chunk_size = max(1, int(grid_chunk_size))

    @property
    def embed_dim(self) -> int:
        return int(self.grid_latents.size(-1))

    def _check_ids(self, feature_id: Tensor) -> Tensor:
        ids = feature_id.long()
        if ids.numel() and ((ids < 0).any() or (ids >= self.vid_to_row.numel()).any()):
            raise ValueError("feature_id contains a VID outside the Phase-1 vocabulary")
        rows = self.vid_to_row.index_select(0, ids.reshape(-1))
        if (rows < 0).any():
            unknown = ids.reshape(-1)[rows < 0].unique().detach().cpu().tolist()
            raise ValueError(f"numeric readout has no train support for VID(s): {unknown[:20]}")
        return rows.reshape(ids.shape)

    def forward(self, z_pred: Tensor, feature_id: Tensor) -> Dict[str, Tensor]:
        """Grouped, bounded-memory manifold inversion preserving query order."""
        if z_pred.ndim != 2 or feature_id.ndim != 1 or feature_id.numel() != z_pred.size(0):
            raise ValueError("z_pred must be [N,D] and feature_id must be [N]")
        rows = self._check_ids(feature_id)
        n = int(z_pred.size(0))
        outputs = [torch.empty((n,), dtype=torch.float32, device=z_pred.device) for _ in range(5)]
        with torch.autocast(device_type=z_pred.device.type, enabled=False):
            z = z_pred.float()
            for row in torch.unique(rows, sorted=True).tolist():
                positions = torch.where(rows == int(row))[0]
                valid = torch.where(self.grid_valid_mask[int(row)])[0]
                if valid.numel() == 0:
                    raise RuntimeError(f"manifold row {int(row)} has no valid grid point")
                values = self.grid_values[int(row)].index_select(0, valid)
                latents = self.grid_latents[int(row)].index_select(0, valid)
                for q0 in range(0, positions.numel(), self.query_chunk_size):
                    qpos = positions[q0:q0 + self.query_chunk_size]
                    q = z.index_select(0, qpos)
                    energy = torch.empty((q.size(0), valid.numel()), dtype=torch.float32, device=z.device)
                    for g0 in range(0, valid.numel(), self.grid_chunk_size):
                        g1 = min(valid.numel(), g0 + self.grid_chunk_size)
                        energy[:, g0:g1] = (q[:, None, :] - latents[g0:g1][None, :, :]).abs().mean(dim=-1)
                    minimum = energy.min(dim=1).values
                    probability = torch.softmax(-(energy - minimum[:, None]) / self.temperature.float().clamp_min(1e-8), dim=1)
                    mean = (probability * values[None, :]).sum(dim=1)
                    variance = (probability * (values[None, :] - mean[:, None]).square()).sum(dim=1)
                    entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=1)
                    boundary = probability[:, 0] + (probability[:, -1] if probability.size(1) > 1 else torch.zeros_like(probability[:, 0]))
                    outputs[0][qpos] = mean
                    outputs[1][qpos] = self.sigma_scale.float().clamp_min(0.0) * torch.sqrt(variance + self.sigma_floor ** 2)
                    outputs[2][qpos], outputs[3][qpos], outputs[4][qpos] = minimum, entropy, boundary
        return {"numeric_mean": outputs[0], "numeric_sigma": outputs[1], "min_energy": outputs[2], "posterior_entropy": outputs[3], "boundary_mass": outputs[4]}

    def compute_energy(self, z: Tensor, feature_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return per-query energy, valid grid values and valid indices for oracle diagnostics."""
        rows = self._check_ids(feature_id); energies=[]; values=[]; indices=[]
        for row in torch.unique(rows, sorted=True).tolist():
            pos = torch.where(rows == int(row))[0]; valid = torch.where(self.grid_valid_mask[int(row)])[0]
            e = (z.index_select(0, pos).float()[:, None, :] - self.grid_latents[int(row)].index_select(0, valid)[None, :, :]).abs().mean(-1)
            energies.append((pos, e)); values.append(self.grid_values[int(row)].index_select(0, valid)); indices.append(valid)
        return energies, values, indices

    def decode_from_energy(self, energies: Tensor, values: Tensor, *, temperature: Optional[float] = None, hard: bool = False) -> Dict[str, Tensor]:
        e = energies.float(); v = values.float(); minimum = e.min(1).values
        if hard:
            idx = e.argmin(1); mu = v[idx]; prob = torch.zeros_like(e).scatter_(1, idx[:, None], 1.0)
        else:
            temp = float(self.temperature.item() if temperature is None else temperature)
            prob = torch.softmax(-(e - minimum[:, None]) / max(temp, 1e-8), 1); mu = (prob * v[None, :]).sum(1)
        sigma_raw = torch.sqrt((prob * (v[None, :] - mu[:, None]).square()).sum(1) + self.sigma_floor ** 2)
        entropy = -(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum(1)
        boundary = prob[:, 0] + (prob[:, -1] if prob.size(1) > 1 else torch.zeros_like(prob[:, 0]))
        return {"numeric_mean": mu, "numeric_sigma": sigma_raw * self.sigma_scale.float(), "min_energy": minimum, "posterior_entropy": entropy, "boundary_mass": boundary}

    def hard_decode(self, z: Tensor, feature_id: Tensor) -> Dict[str, Tensor]:
        rows = self._check_ids(feature_id); out = {k: torch.empty(z.size(0), device=z.device, dtype=torch.float32) for k in ("numeric_mean", "numeric_sigma", "min_energy", "posterior_entropy", "boundary_mass")}
        for row in torch.unique(rows, sorted=True).tolist():
            pos = torch.where(rows == int(row))[0]; valid = torch.where(self.grid_valid_mask[int(row)])[0]
            e = (z.index_select(0, pos).float()[:, None, :] - self.grid_latents[int(row)].index_select(0, valid)[None, :, :]).abs().mean(-1)
            part = self.decode_from_energy(e, self.grid_values[int(row)].index_select(0, valid), hard=True)
            for k in out: out[k][pos] = part[k]
        return out

    def decode_with_temperature(self, z: Tensor, feature_id: Tensor, temperature: float) -> Dict[str, Tensor]:
        """Posterior decode at an explicit frozen temperature (oracle/calibration diagnostics)."""
        rows = self._check_ids(feature_id); out = {k: torch.empty(z.size(0), device=z.device, dtype=torch.float32) for k in ("numeric_mean", "numeric_sigma", "min_energy", "posterior_entropy", "boundary_mass")}
        for row in torch.unique(rows, sorted=True).tolist():
            pos = torch.where(rows == int(row))[0]; valid = torch.where(self.grid_valid_mask[int(row)])[0]
            e = (z.index_select(0, pos).float()[:, None, :] - self.grid_latents[int(row)].index_select(0, valid)[None, :, :]).abs().mean(-1)
            part = self.decode_from_energy(e, self.grid_values[int(row)].index_select(0, valid), temperature=float(temperature))
            for k in out: out[k][pos] = part[k]
        return out

    def tangent_geometry(self, z_pred: Tensor, z_target: Tensor, target_value: Tensor, feature_id: Tensor) -> Dict[str, Tensor]:
        rows = self._check_ids(feature_id); tangent = torch.empty_like(z_pred.float()); nearest = torch.empty(z_pred.size(0), device=z_pred.device, dtype=torch.long)
        for row in torch.unique(rows, sorted=True).tolist():
            pos = torch.where(rows == int(row))[0]; valid = torch.where(self.grid_valid_mask[int(row)])[0]; lat = self.grid_latents[int(row)].index_select(0, valid)
            val = self.grid_values[int(row)].index_select(0, valid); qv = target_value.index_select(0, pos).float(); ni = (qv[:, None] - val[None, :]).abs().argmin(1); global_i = valid.index_select(0, ni); nearest[pos] = global_i
            local = []
            for i in global_i.tolist():
                j = int((valid == int(i)).nonzero()[0]); left, right = max(0, j-1), min(len(valid)-1, j+1); d = lat[right] - lat[left]; local.append(d / d.norm().clamp_min(1e-12))
            tangent[pos] = torch.stack(local)
        err = z_pred.float() - z_target.float(); tangent_error = (err * tangent).sum(-1); orth = err - tangent_error[:, None] * tangent
        return {"tangent_error": tangent_error.abs(), "squared_tangent_error": tangent_error.square(), "orthogonal_error": orth.norm(dim=-1), "nearest_grid_index": nearest}

    def metadata(self) -> Dict[str, Any]:
        return {
            "architecture": READOUT_ARCHITECTURE,
            "grid_size": int(self.grid_values.size(1)),
            "embed_dim": int(self.embed_dim),
            "numeric_vids": self.numeric_vids.detach().cpu().tolist(),
            "support_min": self.support_min.detach().cpu().tolist(),
            "support_max": self.support_max.detach().cpu().tolist(),
            "support_count": self.support_count.detach().cpu().tolist(),
            "temperature": float(self.temperature.item()),
            "sigma_scale": float(self.sigma_scale.item()),
            "sigma_floor": float(self.sigma_floor),
            "query_chunk_size": int(self.query_chunk_size),
            "grid_chunk_size": int(self.grid_chunk_size),
        }


def build_manifold_bank(
    teacher: Any,
    values_by_vid: Mapping[int, Any],
    *,
    grid_size: int = 513,
    checkpoint_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    split: str = "train",
    sigma_floor: float = 1.0e-4,
    query_chunk_size: int = 2048,
    grid_chunk_size: int = 513,
) -> Phase1TargetManifoldReadout:
    if int(grid_size) < 1:
        raise ValueError("grid_size must be positive")
    type_ids = teacher.feature_type_ids.detach().cpu().long()
    numeric = [int(v) for v in type_ids.nonzero(as_tuple=False).flatten().tolist() if int(type_ids[v]) == 2]
    numeric = [v for v in numeric if v in values_by_vid]
    if not numeric:
        raise ValueError("train split contains no supported numeric VIDs")
    d = int(teacher.embed_dim)
    g = int(grid_size)
    grid_values = torch.zeros((len(numeric), g), dtype=torch.float32)
    grid_valid = torch.zeros((len(numeric), g), dtype=torch.bool)
    support_min = torch.empty((len(numeric),), dtype=torch.float32)
    support_max = torch.empty((len(numeric),), dtype=torch.float32)
    support_count = torch.empty((len(numeric),), dtype=torch.long)
    for row, vid in enumerate(numeric):
        support = values_by_vid[vid]
        if isinstance(support, Mapping):
            lo, hi = float(support["min"]), float(support["max"])
            count = int(support["count"])
        else:
            vals = torch.as_tensor(support, dtype=torch.float32).flatten()
            vals = vals[torch.isfinite(vals)]
            if vals.numel() == 0:
                raise ValueError(f"numeric VID {vid} has no finite train support")
            lo, hi, count = float(vals.min()), float(vals.max()), int(vals.numel())
        support_min[row], support_max[row], support_count[row] = lo, hi, count
        if lo == hi:
            grid_values[row, 0] = lo
            grid_valid[row, 0] = True
        else:
            grid_values[row] = torch.linspace(lo, hi, g)
            grid_valid[row] = True
    latents = torch.zeros((len(numeric), g, d), dtype=torch.float32)
    teacher.eval()
    try:
        teacher_device = next(teacher.parameters()).device
    except StopIteration:
        teacher_device = type_ids.device
    with torch.inference_mode():
        for row, vid in enumerate(numeric):
            valid_idx = torch.where(grid_valid[row])[0]
            code = torch.full((valid_idx.numel(), 1), int(vid), dtype=torch.long, device=teacher_device)
            val = grid_values[row, valid_idx].to(code.device).view(-1, 1, 1)
            z = teacher.target_feature_projector(code, val).squeeze(1)
            z = teacher.target_fusion_norm(z)
            latents[row, valid_idx] = z.float().cpu()
    vid_to_row = torch.full((int(type_ids.numel()),), -1, dtype=torch.long)
    vid_to_row[torch.tensor(numeric)] = torch.arange(len(numeric), dtype=torch.long)
    out = Phase1TargetManifoldReadout(
        grid_values=grid_values,
        grid_latents=latents,
        grid_valid_mask=grid_valid,
        numeric_vids=torch.tensor(numeric, dtype=torch.long),
        vid_to_row=vid_to_row,
        sigma_floor=sigma_floor,
        query_chunk_size=query_chunk_size,
        grid_chunk_size=grid_chunk_size,
        support_min=support_min,
        support_max=support_max,
        support_count=support_count,
    )
    out._provenance = {
        "checkpoint_path": os.path.realpath(checkpoint_path) if checkpoint_path else None,
        "schema_path": os.path.realpath(schema_path) if schema_path else None,
        "phase1_checkpoint_sha256": _file_sha256(checkpoint_path) if checkpoint_path else None,
        "schema_sha256": _file_sha256(schema_path) if schema_path else None,
        "phase1_architecture_version": getattr(teacher, "phase1_architecture_version", None),
        "split": str(split),
        "architecture": READOUT_ARCHITECTURE,
    }
    return out


def save_readout_artifact(path: str, readout: Phase1TargetManifoldReadout, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
    payload = {
        "architecture": READOUT_ARCHITECTURE,
        "readout_metadata": readout.metadata(),
        "provenance": getattr(readout, "_provenance", {}),
        "extra_metadata": dict(metadata or {}),
        "state_dict": readout.state_dict(),
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(os.path.realpath(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_readout_artifact(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    expected_phase1_ckpt: Optional[str] = None,
    expected_schema: Optional[str] = None,
    expected_phase1_architecture: Optional[str] = None,
    expected_phase1_sha256: Optional[str] = None,
    expected_schema_sha256: Optional[str] = None,
) -> Phase1TargetManifoldReadout:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("architecture") != READOUT_ARCHITECTURE:
        raise RuntimeError("readout artifact is not a phase1_target_manifold_inverse_v1 artifact")
    provenance = payload.get("provenance", {})
    if expected_phase1_ckpt is not None and os.path.realpath(str(provenance.get("checkpoint_path"))) != os.path.realpath(str(expected_phase1_ckpt)):
        raise RuntimeError("readout artifact Phase-1 checkpoint provenance mismatch")
    if expected_phase1_ckpt is not None and provenance.get("phase1_checkpoint_sha256"):
        if provenance["phase1_checkpoint_sha256"] != _file_sha256(expected_phase1_ckpt):
            raise RuntimeError("readout artifact Phase-1 checkpoint content mismatch")
    if expected_schema is not None and os.path.realpath(str(provenance.get("schema_path"))) != os.path.realpath(str(expected_schema)):
        raise RuntimeError("readout artifact schema provenance mismatch")
    if expected_schema is not None and provenance.get("schema_sha256"):
        if provenance["schema_sha256"] != _file_sha256(expected_schema):
            raise RuntimeError("readout artifact schema content mismatch")
    if expected_phase1_architecture is not None and provenance.get("phase1_architecture_version") != expected_phase1_architecture:
        raise RuntimeError("readout artifact Phase-1 architecture provenance mismatch")
    if expected_phase1_sha256 is not None and provenance.get("phase1_checkpoint_sha256") != expected_phase1_sha256:
        raise RuntimeError("readout artifact Phase-1 checkpoint SHA256 mismatch")
    if expected_schema_sha256 is not None and provenance.get("schema_sha256") != expected_schema_sha256:
        raise RuntimeError("readout artifact schema SHA256 mismatch")
    md = payload["readout_metadata"]
    state = payload["state_dict"]
    obj = Phase1TargetManifoldReadout(
        grid_values=state["grid_values"], grid_latents=state["grid_latents"],
        grid_valid_mask=state["grid_valid_mask"], numeric_vids=state["numeric_vids"],
        vid_to_row=state["vid_to_row"], temperature=float(md["temperature"]),
        sigma_scale=float(md["sigma_scale"]), sigma_floor=float(md["sigma_floor"]),
        query_chunk_size=int(md["query_chunk_size"]), grid_chunk_size=int(md["grid_chunk_size"]),
        support_min=state.get("support_min"), support_max=state.get("support_max"), support_count=state.get("support_count"),
    )
    obj.load_state_dict(state, strict=True)
    obj.eval()
    obj.requires_grad_(False)
    return obj


def calibrate_readout(readout: Phase1TargetManifoldReadout, z_pred: Tensor, feature_id: Tensor, target: Tensor, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Select temperature by macro VID MSE, then sigma scale by Gaussian NLL."""
    search = cfg.get("readout", {})
    tcfg, scfg = search.get("temperature_search", {}), search.get("sigma_scale_search", {})
    def candidates(spec: Mapping[str, Any], default_min: float, default_max: float, default_num: int) -> Tensor:
        lo, hi, num = float(spec.get("min", default_min)), float(spec.get("max", default_max)), int(spec.get("num", default_num))
        if spec.get("scale", "log") == "log":
            return torch.logspace(math.log10(lo), math.log10(hi), num)
        return torch.linspace(lo, hi, num)
    temperatures = candidates(tcfg, 1e-4, 10.0, 81)
    scales = candidates(scfg, 0.05, 20.0, 81)
    min_count = int(search.get("macro_min_count", 10))
    unique = feature_id.long().unique(sorted=True)
    best_t, best_macro = None, float("inf")
    temperature_trace = []
    with torch.no_grad():
        for t in temperatures.tolist():
            readout.temperature.fill_(float(t))
            pred = readout(z_pred, feature_id)["numeric_mean"]
            err = pred - target.float()
            per = [err[feature_id == int(vid)].square().mean() for vid in unique.tolist() if int((feature_id == int(vid)).sum()) >= min_count]
            macro = float(torch.stack(per).mean()) if per else float("inf")
            temperature_trace.append({"temperature": float(t), "macro_mse": macro, "micro_mse": float(err.square().mean()), "macro_mae": float(torch.stack([err[feature_id == int(vid)].abs().mean() for vid in unique.tolist() if int((feature_id == int(vid)).sum()) >= min_count]).mean()) if per else float("nan"), "micro_mae": float(err.abs().mean()), "eligible_vid_count": len(per)})
            if macro < best_macro:
                best_t, best_macro = float(t), macro
        readout.temperature.fill_(best_t)
        base = readout(z_pred, feature_id)
        raw_sigma = base["numeric_sigma"] / readout.sigma_scale.float().clamp_min(1e-8)
        y, mu = target.float(), base["numeric_mean"]
        best_s, best_nll = None, float("inf")
        sigma_trace = []
        for s in scales.tolist():
            readout.sigma_scale.fill_(float(s))
            sigma = (raw_sigma * float(s)).clamp_min(readout.sigma_floor)
            nll = float((0.5 * ((y - mu) / sigma).square() + sigma.log()).mean())
            z = (y - mu).abs() <= sigma
            z2 = (y - mu).abs() <= 1.959963984540054 * sigma
            sigma_trace.append({"sigma_scale": float(s), "nll": nll, "coverage_68": float(z.float().mean()), "coverage_95": float(z2.float().mean()), "mean_interval_width": float((2 * 1.959963984540054 * sigma).mean())})
            if nll < best_nll:
                best_s, best_nll = float(s), nll
        readout.sigma_scale.fill_(best_s)
    return {"temperature": best_t, "sigma_scale": best_s, "val_numeric_mse_macro": best_macro, "val_gaussian_nll": best_nll, "temperature_trace": temperature_trace, "sigma_trace": sigma_trace}


@torch.inference_mode()
def calibrate_readout_streaming(readout: Phase1TargetManifoldReadout, loader: Any, cfg: Mapping[str, Any], *, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Validation-only calibration over a cache loader; no split-wide tensors."""
    if device is None:
        device = next(readout.buffers()).device
    readout = readout.to(device).eval()
    rcfg = cfg.get("readout", {}); tcfg, scfg = rcfg.get("temperature_search", {}), rcfg.get("sigma_scale_search", {})
    def grid(spec, lo, hi, n):
        a,b,k=float(spec.get("min",lo)),float(spec.get("max",hi)),int(spec.get("num",n)); return torch.logspace(math.log10(a),math.log10(b),k) if spec.get("scale","log")=="log" else torch.linspace(a,b,k)
    temps_tensor=grid(tcfg,1e-4,10,81).to(device); temps=temps_tensor.detach().cpu().tolist(); min_count=int(rcfg.get("macro_min_count",10)); traces=[]; best=(None,float("inf"))
    k=len(temps); total_sq=np.zeros(k,dtype=np.float64); total_abs=np.zeros(k,dtype=np.float64); n=0.0; by: Dict[int, List[np.ndarray]] = {}; temp_chunk=int(rcfg.get("temperature_chunk_size",16))
    for batch in tqdm(loader, desc="readout_calibrate_temperature", ascii=True, mininterval=5.0):
        z=batch["z_pred"].to(device, non_blocking=True).float(); f=batch["feature_id"].to(device, non_blocking=True).long(); y=batch["target_value"].to(device, non_blocking=True).float(); rows=readout._check_ids(f)
        for row in torch.unique(rows, sorted=True).tolist():
            positions=torch.where(rows==int(row))[0]; valid=torch.where(readout.grid_valid_mask[int(row)])[0]; values=readout.grid_values[int(row)].index_select(0,valid); latents=readout.grid_latents[int(row)].index_select(0,valid); vid=int(f[positions[0]].detach().cpu())
            bucket=by.setdefault(vid,[np.zeros(k,dtype=np.float64),np.zeros(k,dtype=np.float64),np.zeros(k,dtype=np.float64)])
            for q0 in range(0, positions.numel(), readout.query_chunk_size):
                qpos=positions[q0:q0+readout.query_chunk_size]; q=z.index_select(0,qpos); target=y.index_select(0,qpos); energy=torch.empty((q.size(0), valid.numel()), dtype=torch.float32, device=device)
                for g0 in range(0, valid.numel(), readout.grid_chunk_size):
                    g1=min(valid.numel(), g0+readout.grid_chunk_size); energy[:,g0:g1]=(q[:,None,:]-latents[g0:g1][None,:,:]).abs().mean(-1)
                minimum=energy.min(1).values
                for k0 in range(0,k,temp_chunk):
                    kt=temps_tensor[k0:k0+temp_chunk]; prob=torch.softmax(-(energy[None,:,:]-minimum[None,:,None])/kt[:,None,None].clamp_min(1e-8), dim=2); mu=(prob*values[None,None,:]).sum(2); err=mu-target[None,:]
                    sq=err.square().sum(1).detach().cpu().numpy(); ab=err.abs().sum(1).detach().cpu().numpy(); sl=slice(k0,k0+len(sq)); total_sq[sl]+=sq; total_abs[sl]+=ab; bucket[0][sl]+=float(target.numel()); bucket[1][sl]+=sq; bucket[2][sl]+=ab
                n += float(target.numel())
    for idx,t in enumerate(temps):
        eligible=[r for r in by.values() if r[0][idx]>=min_count]; macro=sum(r[1][idx]/r[0][idx] for r in eligible)/len(eligible) if eligible else float("inf")
        row={"temperature":float(t),"macro_mse":float(macro),"micro_mse":float(total_sq[idx]/n) if n else float("nan"),"macro_mae":float(sum(r[2][idx]/r[0][idx] for r in eligible)/len(eligible)) if eligible else float("nan"),"micro_mae":float(total_abs[idx]/n) if n else float("nan"),"eligible_vid_count":len(eligible)}; traces.append(row)
        if macro<best[1]: best=(float(t),float(macro))
    readout.temperature.fill_(best[0]); scales=grid(scfg,.05,20,81).to(device)
    nll=torch.zeros_like(scales); c68=torch.zeros_like(scales); c95=torch.zeros_like(scales); width=torch.zeros_like(scales); count=0.0
    original_scale = readout.sigma_scale.detach().clone()
    readout.sigma_scale.fill_(1.0)
    for batch in tqdm(loader, desc="readout_calibrate_sigma", ascii=True, mininterval=5.0):
        z=batch["z_pred"].to(device, non_blocking=True).float()
        f=batch["feature_id"].to(device, non_blocking=True).long()
        y=batch["target_value"].to(device, non_blocking=True).float()
        out=readout(z,f)
        raw=out["numeric_sigma"].clamp_min(readout.sigma_floor)
        err=(y-out["numeric_mean"]).abs()
        sig=(raw[:,None] * scales[None,:]).clamp_min(readout.sigma_floor)
        nll += (.5*(err[:,None]/sig).square()+sig.log()).sum(0)
        c68 += (err[:,None] <= sig).sum(0)
        c95 += (err[:,None] <= 1.9599639845*sig).sum(0)
        width += (2*1.9599639845*sig).sum(0)
        count += float(y.numel())
    readout.sigma_scale.copy_(original_scale)
    sigma_trace=[]; bests=(None,float("inf"))
    for idx, s in enumerate(scales.detach().cpu().tolist()):
        row={"sigma_scale":float(s),"nll":float(nll[idx].detach().cpu())/count if count else float("nan"),"coverage_68":float(c68[idx].detach().cpu())/count if count else float("nan"),"coverage_95":float(c95[idx].detach().cpu())/count if count else float("nan"),"mean_interval_width":float(width[idx].detach().cpu())/count if count else float("nan")}; sigma_trace.append(row)
        if row["nll"]<bests[1]: bests=(float(s),row["nll"])
    readout.sigma_scale.fill_(bests[0]); return {"temperature":best[0],"sigma_scale":bests[0],"val_numeric_mse_macro":best[1],"val_gaussian_nll":bests[1],"temperature_trace":traces,"sigma_trace":sigma_trace}
