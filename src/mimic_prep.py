"""
ATLAS-EHR Data Preprocessing Pipeline

Zero-Computation Packer: Transforms BigQuery-exported JSONL buckets into
Dual-Chain JEPA Tensors. All normalization and clipping is performed upstream
in SQL (ehr_model_ready.sql). This script only reads, joins, splits, sorts,
and packs.

Input:
    - buckets_dir:  Directory tree: {split}/bucket={N}/shard-*.jsonl.gz
    - schema_csv:   BigQuery/schema_processed.csv (vid, model_type, stats)
    - freq_csv:     BigQuery/states_frequency.csv (normalized_concept_label, frequency)
    - labels_csv:   sample_data/labels.csv (subject_id, hadm_id, split, targets)

Output:
    - output_dir/{split}/{hadm_id}.pt
    - output_dir/schema.json
    - output_dir/prep_summary.json
"""

import os
import glob
import json
import logging
import argparse
import multiprocessing
import time
from typing import Dict, Any

import torch
import numpy as np
import polars as pl
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PREP] - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("prep.log")],
)
logger = logging.getLogger(__name__)


# ── 1. Schema Artifact Generation ────────────────────────────────────────────

def generate_schema_artifacts(schema_csv: str, output_dir: str, freq_csv: str = None) -> Dict[str, Any]:
    """
    Reads schema_processed.csv and emits schema.json.
    Uses freq_csv (joined by 'vid') to populate state frequencies.
    """
    logger.info(f"Loading schema info from {schema_csv}...")
    df = pl.read_csv(schema_csv)

    required = {"vid", "model_type", "normalized_concept_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schema CSV missing columns: {missing}")

    # Drop rows without valid vid
    df = df.filter(pl.col("vid").is_not_null())

    # Load State Frequencies if provided (STRICT JOIN BY VID)
    freq_dict = {}
    if freq_csv and os.path.exists(freq_csv):
        logger.info(f"Loading frequencies from {freq_csv} (STRICT VID MATCH)...")
        fdf = pl.read_csv(freq_csv)
        
        # Check for vid column presence
        v_col = "vid" if "vid" in fdf.columns else "vocab_id"
        if v_col not in fdf.columns:
            raise ValueError(f"Frequency CSV {freq_csv} must contain 'vid' or 'vocab_id' column for strict matching.")

        # Convert to float and cap at 1.0 (with error checking)
        fdf = fdf.with_columns(
            pl.col("frequency").fill_nan(0.0).clip(0.0, 1.0).alias("freq_clamped")
        )
        freq_dict = dict(zip(fdf[v_col].cast(pl.Int64), fdf["freq_clamped"]))
    elif freq_csv:
         raise FileNotFoundError(f"Frequency file not found at: {freq_csv}")

    allowed_types = {"numeric", "categorical"}

    features: Dict[str, Any] = {}
    for row in df.iter_rows(named=True):
        vid = int(row["vid"])
        if vid == 0:
            raise ValueError(f"Feature '{row['normalized_concept_label']}' has VID=0. "
                             f"VID: 0 is strictly reserved for [PAD] (padding).")
        
        vid_str = str(vid)
        model_type = (row["model_type"] or "categorical").lower()
        
        # Only numeric and categorical allowed
        if model_type not in allowed_types:
            continue
            
        entry: Dict[str, Any] = {
            "item": row["normalized_concept_label"],
            "type": model_type,
            "category": row.get("feature_category", "state"),
        }

        # Frequency for states (STRICT CHECK)
        if entry["category"] == "state":
            if freq_dict and vid not in freq_dict:
                raise ValueError(f"Feature VID={vid} ({entry['item']}) marked as 'state' but missing from frequency CSV.")
            
            f = freq_dict.get(vid, 0.0)
            entry["frequency"] = round(float(f), 6)

        # Stats only for numeric
        if model_type == "numeric":
            entry["mean"] = float(row.get("mean") or 0.0)
            entry["std"] = float(row.get("std") or 1.0)
            entry["p5"] = float(row.get("p5") or 0.0)
            entry["p95"] = float(row.get("p95") or 0.0)

        features[vid_str] = entry

    vocab_size = max(int(k) for k in features) + 1 if features else 1
    schema_out = {"vocab_size": vocab_size, "features": features}

    out_path = os.path.join(output_dir, "schema.json")
    with open(out_path, "w") as f:
        json.dump(schema_out, f, indent=2)

    logger.info(f"schema.json saved → {out_path} ({len(features)} features, vocab_size={vocab_size})")
    return schema_out


# ── 2. Labels Loader ─────────────────────────────────────────────────────────

def load_labels(labels_path: str) -> pl.DataFrame:
    """
    Loads labels.csv and returns a DataFrame with:
        subject_id, hadm_id, split, label_mortality, label_readmission_30d, label_los_days
    Filters to is_standard_cohort == 1 if the column exists.
    """
    logger.info(f"Loading labels from {labels_path}...")
    df = pl.read_csv(labels_path)

    # Standard cohort filter
    if "is_standard_cohort" in df.columns:
        df = df.filter(pl.col("is_standard_cohort") == 1)

    needed = ["subject_id", "hadm_id", "split",
              "label_mortality", "label_readmission_30d", "label_los_days",
              "icd_codes", "drg_codes"]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    df = df.select(needed).with_columns(pl.col("hadm_id").cast(pl.Int64))
    logger.info(f"Labels loaded: {df.height} admissions  "
                f"(train={df.filter(pl.col('split')=='train').height}, "
                f"val={df.filter(pl.col('split')=='val').height}, "
                f"test={df.filter(pl.col('split')=='test').height})")
    return df


# ── 3. Vocab Gate (Schema VID Set) ───────────────────────────────────────────

def load_vocab_set(schema_csv: str) -> set:
    """Returns the set of valid vids from schema_processed.csv."""
    df = pl.read_csv(schema_csv, columns=["vid"])
    df = df.filter(pl.col("vid").is_not_null())
    return set(df["vid"].to_list())


# ── 4. Bucket Processing (Worker) ────────────────────────────────────────────

def process_bucket(
    bucket_path: str,
    output_dir: str,
    labels_path: str,
    schema_csv: str,
    bucket_id: str,
) -> Dict[str, Any]:
    """
    Worker function: processes one JSONL bucket directory.

    Steps:
        1. scan_ndjson  → LazyFrame with trajectory column
        2. join labels  → inject subject_id, split, targets
        3. explode trajectory → flat event rows
        4. vocab gate   → inner-join on vid with schema
        5. sort         → offset_min ASC, cat_priority ASC (state=0, intervention=1)
        6. group_by     → per (subject_id, hadm_id)
        7. pack .pt     → save to output_dir/{split}/{hadm_id}.pt

    Returns:
        Dict with per-bucket statistics for global aggregation.
    """
    worker = multiprocessing.current_process().name
    t0 = time.time()

    # ── Per-bucket stats ─────────────────────────────────────────────────
    stats = {
        "bucket_id": bucket_id,
        "total_hadm_ids": 0,
        "skipped_no_states": 0,
        "kept_with_interventions": 0,
        "kept_no_interventions": 0,
        "filtered_hadm_ids": [],
    }

    try:
        # ── Reference data (loaded per-worker to avoid serialization) ────
        labels_df = load_labels(labels_path)
        vocab_set = load_vocab_set(schema_csv)

        # ── Discover JSONL shards ────────────────────────────────────────
        patterns = [
            os.path.join(bucket_path, "*.jsonl.gz"),
            os.path.join(bucket_path, "*.jsonl"),
            os.path.join(bucket_path, "*.json.gz"),
            os.path.join(bucket_path, "*.json"),
        ]
        shard_files = []
        for pat in patterns:
            shard_files.extend(glob.glob(pat))

        if not shard_files:
            logger.warning(f"[{worker}] No JSONL files in {bucket_path}")
            return stats

        # ── Read JSONL (Eager per shard with empty-file safety) ──────────
        frames = []
        for sf in sorted(shard_files):
            # 1. Quick size check (Skip obviously empty 0-byte or tiny compressed files)
            # BigQuery exports sometimes create tiny 20-30 byte .gz files that are effectively empty.
            if os.path.exists(sf) and os.path.getsize(sf) < 50:
                 continue

            try:
                sdf = pl.read_ndjson(sf)
                if sdf.height > 0:
                    frames.append(sdf)
            except Exception as e:
                # Catch the specific Polars empty reader error and skip silently
                if "empty reader" in str(e).lower() or "no data" in str(e).lower():
                    continue
                logger.warning(f"[{worker}] Failed to read {sf}: {e}")

        if not frames:
            logger.warning(f"[{worker}] All shards failed in {bucket_path}")
            return stats

        raw_df = pl.concat(frames)
        logger.info(f"[{worker}] Loaded {raw_df.height} admissions from {len(shard_files)} shards")

        # ── Ensure hadm_id type ──────────────────────────────────────────
        raw_df = raw_df.with_columns(pl.col("hadm_id").cast(pl.Int64))

        # ── Join labels (inner → cohort filter + split injection) ────────
        raw_df = raw_df.join(labels_df, on="hadm_id", how="inner", suffix="_lbl")

        # Resolve subject_id: prefer the one from labels if both exist
        if "subject_id_lbl" in raw_df.columns:
            raw_df = raw_df.drop("subject_id").rename({"subject_id_lbl": "subject_id"})

        if raw_df.height == 0:
            logger.info(f"[{worker}] No matching admissions after label join")
            return stats

        # ── Explode trajectory → flat event rows ─────────────────────────
        df = raw_df.explode("trajectory")

        # Unnest the struct fields (cast vid to Int64 — JSONL may store as String)
        df = df.with_columns([
            pl.col("trajectory").struct.field("vid").cast(pl.Int64).alias("vid"),
            pl.col("trajectory").struct.field("offset_min").cast(pl.Float64).alias("offset_min"),
            pl.col("trajectory").struct.field("val_norm").cast(pl.Float64).alias("val_norm"),
            pl.col("trajectory").struct.field("cat").cast(pl.Utf8).alias("cat"),
        ]).drop("trajectory")

        # ── Vocab gate (filter to known VIDs) ────────────────────────────
        before = df.height
        df = df.filter(pl.col("vid").is_in(list(vocab_set)))
        after = df.height
        if before != after:
            logger.info(f"[{worker}] Vocab gate removed {before - after} OOV events")

        if df.height == 0:
            logger.info(f"[{worker}] No events after vocab gate")
            return stats

        # ── Sort: offset_min ASC, cat_priority ASC (state=0, interv=1) ───
        df = df.with_columns(
            pl.when(pl.col("cat") == "state")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("cat_priority")
        ).sort(["hadm_id", "offset_min", "cat_priority", "vid"])

        # ── Group by patient and pack .pt ────────────────────────────────
        groups = df.partition_by(["subject_id", "hadm_id"], as_dict=True)
        stats["total_hadm_ids"] = len(groups)

        for key, pdf in groups.items():
            # Unpack key
            if isinstance(key, tuple):
                subject_id, hadm_id = int(key[0]), int(key[1])
            else:
                subject_id, hadm_id = int(pdf["subject_id"][0]), int(pdf["hadm_id"][0])

            split_val = str(pdf["split"][0]).lower().strip()
            if split_val in ("valid", "validation"):
                split_val = "val"
            if split_val not in ("train", "val", "test"):
                logger.warning(f"[{worker}] Unknown split '{split_val}' for hadm_id={hadm_id}")
                continue

            # ── Dual-chain split by cat ──────────────────────────────────
            state_mask = pdf["cat"] == "state"
            interv_mask = pdf["cat"] != "state"

            s_df = pdf.filter(state_mask)
            i_df = pdf.filter(interv_mask)

            # Guard: skip if no state tokens
            if s_df.height == 0:
                stats["skipped_no_states"] += 1
                stats["filtered_hadm_ids"].append(hadm_id)
                logger.debug(f"[{worker}] Filtered hadm_id={hadm_id} (L_s=0)")
                continue

            # ── Pack tensors (float32 precision) ─────────────────────────
            state_tokens = torch.from_numpy(s_df["vid"].to_numpy().astype(np.int64)).long()
            state_values = torch.from_numpy(s_df["val_norm"].to_numpy().astype(np.float32)).float()
            state_times  = torch.from_numpy(s_df["offset_min"].to_numpy().astype(np.float32)).float()

            if i_df.height > 0:
                inter_tokens = torch.from_numpy(i_df["vid"].to_numpy().astype(np.int64)).long()
                inter_values = torch.from_numpy(i_df["val_norm"].to_numpy().astype(np.float32)).float()
                inter_times  = torch.from_numpy(i_df["offset_min"].to_numpy().astype(np.float32)).float()
            else:
                # L_i == 0 is valid (no interventions)
                inter_tokens = torch.zeros(0, dtype=torch.long)
                inter_values = torch.zeros(0, dtype=torch.float32)
                inter_times  = torch.zeros(0, dtype=torch.float32)

            # ── Labels ───────────────────────────────────────────────────
            labels_dict = {
                "mortality":   int(pdf["label_mortality"][0]),
                "readmission": int(pdf["label_readmission_30d"][0]),
                "los_days":    float(pdf["label_los_days"][0]),
                "icd":         str(pdf["icd_codes"][0]) if pdf["icd_codes"][0] is not None else "",
                "drg":         str(pdf["drg_codes"][0]) if pdf["drg_codes"][0] is not None else "",
            }

            # ── Output dict ──────────────────────────────────────────────
            out_dict = {
                "subject_id":   subject_id,
                "hadm_id":      hadm_id,
                "split":        split_val,
                "state_tokens": state_tokens,
                "state_values": state_values,
                "state_times":  state_times,
                "inter_tokens": inter_tokens,
                "inter_values": inter_values,
                "inter_times":  inter_times,
                "labels":       labels_dict,
            }

            # ── Save to flat split dir ────────────────────────────────────
            out_path = os.path.join(output_dir, split_val, f"{hadm_id}.pt")
            torch.save(out_dict, out_path)

            # Track composition
            if i_df.height > 0:
                stats["kept_with_interventions"] += 1
            else:
                stats["kept_no_interventions"] += 1

        elapsed = time.time() - t0
        packed = stats["kept_with_interventions"] + stats["kept_no_interventions"]
        logger.info(
            f"[{worker}] Bucket {bucket_id}: packed={packed}, "
            f"skipped(no_state)={stats['skipped_no_states']}, "
            f"with_inter={stats['kept_with_interventions']}, "
            f"obs_only={stats['kept_no_interventions']}, "
            f"time={elapsed:.1f}s"
        )

        # Log filtered hadm_ids for traceability
        if stats["filtered_hadm_ids"]:
            logger.info(
                f"[{worker}] Bucket {bucket_id} filtered hadm_ids: "
                f"{stats['filtered_hadm_ids']}"
            )

        return stats

    except Exception as e:
        logger.error(f"[{worker}] Critical failure on {bucket_path}: {e}")
        import traceback
        traceback.print_exc()
        return stats


# ── Wrapper for imap_unordered (must be top-level for pickling) ──────────

def _process_bucket_wrapper(args_tuple):
    """Unpacks a tuple of arguments and calls process_bucket."""
    return process_bucket(*args_tuple)


# ── 5. Main Driver ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATLAS-EHR Preprocessing: Zero-Computation Dual-Chain Packer"
    )
    parser.add_argument(
        "--buckets_dir", required=True,
        help="Root directory containing split/bucket subdirectories with JSONL shards"
    )
    parser.add_argument(
        "--schema_csv", required=True,
        help="Path to BigQuery/schema_processed.csv"
    )
    parser.add_argument(
        "--labels_csv", required=True,
        help="Path to sample_data/labels.csv"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output root directory"
    )
    parser.add_argument(
        "--freq_csv",
        help="Path to BigQuery/states_frequency.csv"
    )

    parser.add_argument(
        "--num_workers", type=int,
        default=max(1, multiprocessing.cpu_count() - 2),
        help="Number of parallel workers"
    )
    args = parser.parse_args()

    # ── 1. Create output dirs ────────────────────────────────────────────
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(args.output_dir, s), exist_ok=True)

    # ── 2. Generate schema.json ──────────────────────────────────────────
    logger.info("Step 1: Discovering and Generating schema.json...")
    try:
        generate_schema_artifacts(args.schema_csv, args.output_dir, freq_csv=args.freq_csv)
    except Exception as e:
        logger.error(f"Failed to generate schema.json: {e}")
        # Not fatal, but dataset.py will warn

    # ── 4. Discover buckets ──────────────────────────────────────────────
    logger.info(f"Step 2: Scanning buckets in {args.buckets_dir}...")

    # Actual GCS structure: {buckets_dir}/{split}/bucket={N}/shard-*.jsonl.gz
    #   e.g., mimic_iv/train/bucket=0/shard-000000000000.jsonl.gz
    bucket_dirs = []

    # Pattern 1 (primary): {split}/bucket={N}
    for split_name in ("train", "val", "test"):
        split_root = os.path.join(args.buckets_dir, split_name)
        if not os.path.isdir(split_root):
            continue
        for bd in sorted(glob.glob(os.path.join(split_root, "bucket=*"))):
            bucket_id = os.path.basename(bd)  # e.g., "bucket=0"
            bucket_dirs.append((bd, bucket_id))

    # Pattern 2 (fallback): bucket={N} flat under root
    if not bucket_dirs:
        for bd in sorted(glob.glob(os.path.join(args.buckets_dir, "bucket=*"))):
            bucket_id = os.path.basename(bd)
            bucket_dirs.append((bd, bucket_id))

    # Pattern 3 (legacy): bucket-{N}
    if not bucket_dirs:
        for bd in sorted(glob.glob(os.path.join(args.buckets_dir, "bucket-*"))):
            bucket_id = os.path.basename(bd)
            bucket_dirs.append((bd, bucket_id))

    # Pattern 4 (flat): JSONL files directly in root
    if not bucket_dirs:
        jsonl_check = glob.glob(os.path.join(args.buckets_dir, "*.jsonl*"))
        if jsonl_check:
            bucket_dirs.append((args.buckets_dir, "root"))
            logger.info("Found JSONL files in root dir — treating as single bucket.")

    if not bucket_dirs:
        logger.error("No buckets found! Check --buckets_dir path.")
        return

    logger.info(f"Found {len(bucket_dirs)} buckets to process.")

    # ── 4. Multiprocessing ───────────────────────────────────────────────
    logger.info(f"Step 3: Launching {args.num_workers} workers...")

    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = multiprocessing.get_context("fork")

    task_args = [
        (bd, args.output_dir, args.labels_csv, args.schema_csv, bid)
        for bd, bid in bucket_dirs
    ]

    with ctx.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_bucket_wrapper, task_args),
            total=len(task_args),
            desc="Processing buckets",
            unit="bucket",
        ))

    # ── 5. Aggregate stats and write prep_summary.json ───────────────────
    total_hadm = sum(r["total_hadm_ids"] for r in results if r)
    skipped = sum(r["skipped_no_states"] for r in results if r)
    with_inter = sum(r["kept_with_interventions"] for r in results if r)
    obs_only = sum(r["kept_no_interventions"] for r in results if r)

    summary = {
        "total_input_hadm_ids": total_hadm,
        "filtered_out_no_states": skipped,
        "final_retained_samples": with_inter + obs_only,
        "composition": {
            "with_interventions": with_inter,
            "pure_observation_only": obs_only,
        },
    }

    summary_path = os.path.join(args.output_dir, "prep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"prep_summary.json saved → {summary_path}")
    logger.info(
        f"Final: {summary['final_retained_samples']} retained "
        f"({with_inter} with interventions, {obs_only} observation-only), "
        f"{skipped} filtered (no states)"
    )
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")
    main()
