"""Build canonical eICU samples.

The formal path (``--sample_unit hospital_admission``) streams every ICU-stay
trajectory into a disk-backed SQLite spool and emits one sample per
``patienthealthsystemstayid``.  The legacy ICU-stay path is retained only for
historical reproduction and is rejected by the external-validation preflight.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EICU-PREP] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AGE_LABEL = "age_other_years"
GENDER_LABEL_F = "gender_other_f"
GENDER_LABEL_M = "gender_other_m"
TOL_MIN = 5.0


def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def _load_json_records(path: str) -> Iterable[Dict[str, Any]]:
    with _open_text(path) as handle:
        first = handle.readline()
        if not first:
            return
        if first.lstrip().startswith("["):
            value = json.loads(first + handle.read())
            for row in value:
                if isinstance(row, dict):
                    yield row
            return
        line = first
        while line:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
            line = handle.readline()


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return default
    if str(value).strip() == "> 89":
        return 90
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _infer_split(path: str, row: Mapping[str, Any]) -> str:
    value = str(row.get("split", "")).strip().lower()
    if value in {"valid", "validation"}:
        return "val"
    if value in {"train", "val", "test"}:
        return value
    for part in Path(path).parts:
        if part in {"train", "val", "test"}:
            return part
    raise RuntimeError(f"Cannot infer split for {path}")


def _discover_shards(input_dir: str) -> List[str]:
    return sorted(str(path) for path in Path(input_dir).rglob("*.gz") if path.name != "labels.json")


def _load_labels_map(path: str) -> Dict[str, Dict[str, Any]]:
    rows = _load_json_records(path)
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("patientunitstayid", row.get("hadm_id"))
        if key is None:
            continue
        key = str(key)
        if key in result and result[key] != row:
            raise RuntimeError(f"Duplicate conflicting labels for patientunitstayid={key}")
        result[key] = row
    return result


def _load_static_vids(schema_csv: str) -> Dict[str, int]:
    wanted = {AGE_LABEL, GENDER_LABEL_F, GENDER_LABEL_M}
    result: Dict[str, int] = {}
    with open(schema_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("normalized_concept_label", "")).strip()
            if label in wanted and row.get("vid") not in (None, ""):
                result[label] = int(float(row["vid"]))
    missing = wanted - set(result)
    if missing:
        raise RuntimeError(f"schema_csv missing static VIDs: {sorted(missing)}")
    return result


def _iter_trajectory_events(row: Mapping[str, Any]) -> Iterable[Tuple[float, int, float, int]]:
    trajectory = row.get("trajectory")
    items = trajectory if isinstance(trajectory, list) else [row]
    for order, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        if "mimic_vid" not in item or "converted_value" not in item:
            continue
        if "conversion_status" in item and str(item.get("conversion_status")).lower() != "success":
            continue
        timestamp = _safe_float(item.get("timestamp"))
        vid = _safe_int(item.get("mimic_vid"))
        value = _safe_float(item.get("converted_value"))
        if timestamp is not None and vid is not None and value is not None:
            yield timestamp, vid, value, order


def _trajectory_events(row: Mapping[str, Any]) -> List[Tuple[float, int, float, int]]:
    """Compatibility wrapper; production insertion uses the generator."""
    return list(_iter_trajectory_events(row))


def _hospital_los_days(admit: Any, discharge: Any) -> Optional[float]:
    a, d = _safe_float(admit), _safe_float(discharge)
    if a is None or d is None:
        return None
    value = (d - a) / 1440.0
    if value < 0:
        return None
    return value


def _build_sample(row: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    """Legacy ICU-stay helper retained for old callers/tests."""
    stay = str(row.get("patientunitstayid", "")).strip()
    admit = _safe_float(row.get("hospitaladmitoffset"))
    if not stay or admit is None:
        return None
    events = _trajectory_events(row)
    if not events:
        return None
    converted = [(ts - admit, vid, value, order) for ts, vid, value, order in events]
    if any(ts < -1e-4 for ts, *_ in converted):
        raise RuntimeError(f"Negative hospital-relative timestamp in {stay}")
    converted.sort(key=lambda item: (item[0], item[3], item[1]))
    times = [max(0.0, x[0]) for x in converted]
    return {"subject_id": str(row.get("uniquepid", stay)), "hadm_id": stay, "split": split,
            "time_origin": "hospital_admission", "sample_unit": "patientunitstayid",
            "state_tokens": torch.tensor([x[1] for x in converted], dtype=torch.long),
            "state_values": torch.tensor([x[2] for x in converted], dtype=torch.float32),
            "state_times": torch.tensor(times, dtype=torch.float32),
            "inter_tokens": torch.zeros(0, dtype=torch.long), "inter_values": torch.zeros(0),
            "inter_times": torch.zeros(0), "labels": {"mortality": _safe_int(row.get("hospital_mortality")),
            "hospital_mortality": _safe_int(row.get("hospital_mortality")),
            "hospital_los": _hospital_los_days(row.get("hospitaladmitoffset"), row.get("hospitaldischargeoffset"))}}


def _atomic_replace_dir(staging: str, output: str) -> None:
    backup = f"{output}.previous"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(output):
        os.replace(output, backup)
    os.replace(staging, output)
    if os.path.exists(backup):
        shutil.rmtree(backup)


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _aggregate_from_existing_spool(
    spool_path: str,
    labels_map: Mapping[str, Mapping[str, Any]],
    static_vids: Mapping[str, int],
    materialize_events: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Revalidate/aggregate a completed spool without rereading raw shards."""
    conn = sqlite3.connect(spool_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"stays", "events"}.issubset(tables):
        conn.close()
        raise RuntimeError(f"Invalid or incomplete admission spool: {spool_path}")
    # Resume emission queries events by admission.  The canonical spool writes
    # the admission key on every event and creates this composite index after
    # ingestion, so no additional multi-GB events(stay) index is needed.
    existing_indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(events)")}
    if "events_admission_stay_idx" not in existing_indexes:
        print("[EICU-PREP] building missing admission/stay index for resume spool", flush=True)
        conn.execute("CREATE INDEX events_admission_stay_idx ON events(admission, stay)")
        conn.commit()
        print("[EICU-PREP] admission/stay index ready", flush=True)
    else:
        print("[EICU-PREP] admission/stay index already present", flush=True)
    admissions: Dict[str, Dict[str, Any]] = {}
    seen_stays: Dict[str, str] = {}
    stay_count = int(conn.execute("SELECT COUNT(*) FROM stays").fetchone()[0])
    for row in tqdm(
        conn.execute("SELECT stay, admission, subject, split, meta FROM stays ORDER BY admission, stay"),
        total=stay_count,
        desc="eICU resume: validate admissions",
        unit="stay",
    ):
        stay, admission, subject, split, meta_text = row
        if stay in seen_stays and seen_stays[stay] != admission:
            conn.close()
            raise RuntimeError(f"ICU stay maps to multiple admissions: {stay}")
        seen_stays[stay] = admission
        meta = json.loads(meta_text)
        # Replace label-bearing metadata with the newly supplied labels while
        # retaining the already-spooled trajectory and identity fields.
        meta.update(labels_map.get(str(stay), {}))
        # A corrected labels file may carry the authoritative split.  Reuse it
        # only when explicitly present; otherwise retain the split captured in
        # the spool.  No split is inferred or repaired here.
        effective_split = str(meta.get("split", split)).strip().lower()
        if effective_split == "validation":
            effective_split = "val"
        if effective_split not in {"train", "val", "test"}:
            conn.close()
            raise RuntimeError(f"Invalid split while resuming admission {admission}: {effective_split!r}")
        split = effective_split
        item = admissions.setdefault(admission, {"admission": admission, "subject": subject, "split": split,
            "stays": [], "meta": [], "events": [], "static": {}, "static_seen": set()})
        if item["subject"] != subject or item["split"] != split:
            conn.close()
            raise RuntimeError(f"Admission {admission} has conflicting subject or split")
        item["stays"].append(stay); item["meta"].append(meta)
        admit = _safe_float(meta.get("hospitaladmitoffset"))
        if admit is None:
            item.setdefault("missing_offset_stays", []).append(stay)
            continue
        hospital_los = _hospital_los_days(meta.get("hospitaladmitoffset"), meta.get("hospitaldischargeoffset"))
        if hospital_los is None:
            item.setdefault("invalid_los_stays", []).append(stay)
            item.setdefault("hospital_los", []).append(None)
            continue
        mortality = _safe_int(meta.get("hospital_mortality"))
        if mortality in (0, 1):
            item.setdefault("mortalities", set()).add(mortality)
        else:
            item.setdefault("missing_mortality_stays", []).append(stay)
        item.setdefault("hospital_los", []).append(hospital_los)
        if materialize_events:
            invalid_time = False
            for timestamp, vid, value, order in conn.execute(
                "SELECT ts, vid, value, ord FROM events WHERE admission=? AND stay=?", (admission, stay)
            ):
                relative = float(timestamp) - float(admit)
                if relative < -1e-4:
                    invalid_time = True
                    break
                item["events"].append((max(0.0, relative), -float(admit), str(stay), int(order), int(vid), float(value)))
            if invalid_time:
                item.setdefault("invalid_time_stays", []).append(stay)
                item["events"] = []

    for admission, item in admissions.items():
        if len(item.get("mortalities", set())) > 1:
            conn.close()
            raise RuntimeError(f"Conflicting hospital_mortality for admission {admission}")
        if item.get("invalid_los_stays"):
            item["exclusion_reason"] = "invalid_negative_hospital_los"
            continue
        if item.get("missing_offset_stays"):
            item["exclusion_reason"] = "missing_hospitaladmitoffset"
            continue
        if item.get("invalid_time_stays"):
            item["exclusion_reason"] = "negative_hospital_relative_time"
            continue
        if item.get("missing_mortality_stays") or not item.get("mortalities"):
            item["exclusion_reason"] = "missing_hospital_mortality"
            continue
        los = [value for value in item["hospital_los"] if value is not None]
        if len(los) != len(item["meta"]) or not los or max(los) - min(los) > 1e-6:
            conn.close()
            raise RuntimeError(f"Conflicting or missing hospital LOS for admission {admission}")
        if materialize_events:
            static_values: Dict[int, set[float]] = defaultdict(set)
            for _timestamp, _start, _stay, _order, vid, value in item["events"]:
                if int(vid) in static_vids.values():
                    static_values[int(vid)].add(float(value))
            for vid, values in static_values.items():
                if len(values) > 1:
                    conn.close()
                    raise RuntimeError(f"Conflicting static value vid={vid} admission={admission}")
            item["events"].sort(key=lambda event: (event[0], event[1], event[2], event[3], event[4]))
            seen_static: set[int] = set()
            kept = []
            for event in item["events"]:
                if event[4] in static_vids.values():
                    if event[4] in seen_static:
                        continue
                    seen_static.add(event[4])
                kept.append(event)
            item["static_dedup_count"] = len(item["events"]) - len(kept)
            dynamic_keys = [(event[0], event[4], event[5]) for event in item["events"]]
            item["exact_duplicate_count"] = len(dynamic_keys) - len(set(dynamic_keys))
            item["events"] = kept
        else:
            item["static_dedup_count"] = 0
            item["exact_duplicate_count"] = 0
    conn.close()
    return admissions, {"conflicts": [], "icu_stay_count": len(seen_stays), "spool_path": spool_path,
                         "materialize_events": materialize_events, "static_vids": dict(static_vids),
                         "resumed": True}


def _aggregate_hospital_admissions(input_dir: str, labels_map: Mapping[str, Mapping[str, Any]],
                                   schema_csv: str, spool_path: Optional[str] = None,
                                   materialize_events: bool = True,
                                   resume_spool: bool = False) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    static_vids = _load_static_vids(schema_csv)
    if resume_spool:
        if not spool_path or not os.path.isfile(spool_path):
            raise FileNotFoundError(f"Requested resume spool does not exist: {spool_path}")
        return _aggregate_from_existing_spool(spool_path, labels_map, static_vids, materialize_events)
    conn = sqlite3.connect(spool_path or ":memory:")
    # This database is disposable staging state: WAL/checkpoint durability is
    # unnecessary, while batched writes materially reduce per-event fsync and
    # journaling overhead.  The final processed directory is still published
    # atomically after all validation succeeds.
    if spool_path:
        # The spool is a recovery checkpoint, not disposable scratch space:
        # disabling journaling can leave a multi-GB file with zero committed
        # rows after an interrupted scan. DELETE journaling plus NORMAL sync
        # keeps shard commits recoverable without forcing a full fsync for
        # every inserted event.
        conn.executescript("PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL; PRAGMA temp_store=FILE; PRAGMA cache_size=-65536;")
    # The event payload is kept in SQLite rather than a Python all-dataset list;
    # callers can switch this connection to a disk file for very large exports.
    conn.executescript("CREATE TABLE stays(stay TEXT PRIMARY KEY, admission TEXT, subject TEXT, split TEXT, meta TEXT); CREATE TABLE events(admission TEXT, stay TEXT, ts REAL, vid INTEGER, value REAL, ord INTEGER);")
    seen_stays: Dict[str, str] = {}
    conflict_rows: List[Dict[str, Any]] = []
    shard_paths = _discover_shards(input_dir)
    shard_bar = tqdm(shard_paths, desc="eICU admission spool", unit="shard")
    for path in shard_bar:
        shard_bar.set_postfix(shard=Path(path).name[:32], refresh=False)
        for raw in _load_json_records(path):
            stay = str(raw.get("patientunitstayid", "")).strip()
            admission = str(raw.get("admission_id", raw.get("patienthealthsystemstayid", ""))).strip()
            subject = str(raw.get("uniquepid", "")).strip()
            if not stay or not admission or not subject:
                conflict_rows.append({"reason": "missing_identity", "patientunitstayid": stay, "admission_id": admission})
                continue
            if stay in seen_stays:
                if seen_stays[stay] != admission:
                    raise RuntimeError(f"ICU stay maps to multiple admissions: {stay}")
                raise RuntimeError(f"Duplicate ICU stay encountered across shards: {stay}")
            label = labels_map.get(stay)
            merged = dict(raw)
            merged.update(label or {})
            split = _infer_split(path, merged)
            # Verify the SQL contract when the explicit source field is present.
            if "patienthealthsystemstayid" in merged and str(merged["patienthealthsystemstayid"]) != admission:
                raise RuntimeError(f"admission_id mismatch for ICU stay {stay}")
            seen_stays[stay] = admission
            # Do not serialize the full trajectory into the metadata table:
            # events are already stored row-wise below.  Duplicating this large
            # array caused the disposable spool to grow roughly 2x and made
            # later shards progressively slower as the filesystem cache was
            # exhausted.
            meta_for_spool = {key: value for key, value in merged.items() if key != "trajectory"}
            conn.execute("INSERT INTO stays VALUES(?,?,?,?,?)", (stay, admission, subject, split, json.dumps(meta_for_spool, default=str)))
            conn.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)",
                             ((admission, stay, ts, vid, value, order)
                              for ts, vid, value, order in _iter_trajectory_events(merged)))
        # Commit each shard so an interrupted scan leaves a genuinely usable
        # resume point. A single end-of-scan transaction can produce a very
        # large-looking SQLite file whose committed row counts are still zero
        # after the worker is killed or the host loses the process.
        conn.commit()
    if conflict_rows:
        raise RuntimeError(f"Invalid ICU-stay identity rows: {conflict_rows[:20]}")
    # The subsequent validation reads one stay at a time.  Without this index,
    # SQLite scans the complete event spool for every ICU stay (N+1 full-table
    # scans), which dominates long eICU runs.
    conn.execute("CREATE INDEX IF NOT EXISTS events_admission_stay_idx ON events(admission, stay)")
    conn.commit()
    admissions: Dict[str, Dict[str, Any]] = {}
    for row in conn.execute("SELECT stay, admission, subject, split, meta FROM stays ORDER BY admission, stay"):
        stay, admission, subject, split, meta_text = row
        meta = json.loads(meta_text)
        item = admissions.setdefault(admission, {"admission": admission, "subject": subject, "split": split,
            "stays": [], "meta": [], "events": [], "static": {}, "static_seen": set()})
        if item["subject"] != subject or item["split"] != split:
            raise RuntimeError(f"Admission {admission} has conflicting subject or split")
        item["stays"].append(stay); item["meta"].append(meta)
        admit = _safe_float(meta.get("hospitaladmitoffset"))
        if admit is None:
            item.setdefault("missing_offset_stays", []).append(stay)
            continue
        hospital_los = _hospital_los_days(meta.get("hospitaladmitoffset"), meta.get("hospitaldischargeoffset"))
        if hospital_los is None:
            item.setdefault("invalid_los_stays", []).append(stay)
            item.setdefault("hospital_los", []).append(None)
            continue
        mortality = _safe_int(meta.get("hospital_mortality"))
        if mortality in (0, 1):
            item.setdefault("mortalities", set()).add(mortality)
        else:
            item.setdefault("missing_mortality_stays", []).append(stay)
        item.setdefault("hospital_los", []).append(hospital_los)
        if materialize_events:
            invalid_time = False
            for event in conn.execute("SELECT ts, vid, value, ord FROM events WHERE admission=? AND stay=?", (admission, stay)):
                ts, vid, value, order = event
                relative = float(ts) - float(admit)
                if relative < -1e-4:
                    invalid_time = True
                    break
                item["events"].append((max(0.0, relative), -float(admit), str(stay), int(order), int(vid), float(value)))
                if int(vid) in static_vids.values():
                    item["static_seen"].add((int(vid), float(value)))
            if invalid_time:
                item.setdefault("invalid_time_stays", []).append(stay)
                item["events"] = []
    for admission, item in admissions.items():
        if len(item.get("mortalities", set())) > 1:
            missing = item.get("missing_mortality_stays", [])
            raise RuntimeError(f"Missing or conflicting hospital_mortality for admission {admission}; missing_stays={missing[:20]}")
        if item.get("invalid_los_stays"):
            item["exclusion_reason"] = "invalid_negative_hospital_los"
            continue
        if item.get("missing_offset_stays"):
            item["exclusion_reason"] = "missing_hospitaladmitoffset"
            continue
        if item.get("invalid_time_stays"):
            item["exclusion_reason"] = "negative_hospital_relative_time"
            continue
        if item.get("missing_mortality_stays") or not item.get("mortalities"):
            item["exclusion_reason"] = "missing_hospital_mortality"
            continue
        los = [value for value in item["hospital_los"] if value is not None]
        if len(los) != len(item["meta"]) or not los or max(los) - min(los) > 1e-6:
            raise RuntimeError(f"Conflicting or missing hospital LOS for admission {admission}")
        static_by_vid: Dict[int, set[float]] = defaultdict(set)
        for vid, value in item["static_seen"]:
            static_by_vid[vid].add(value)
        for vid, values in static_by_vid.items():
            if len(values) > 1:
                raise RuntimeError(f"Conflicting static value vid={vid} admission={admission}")
        if materialize_events:
            # Remove repeated static features, retain the earliest occurrence.
            seen_static: set[int] = set()
            kept = []
            for event in sorted(item["events"], key=lambda x: (x[0], x[1], x[2], x[3], x[4])):
                if event[4] in static_vids.values():
                    if event[4] in seen_static:
                        continue
                    seen_static.add(event[4])
                kept.append(event)
            item["static_dedup_count"] = len(item["events"]) - len(kept)
            dynamic_keys = [(event[0], event[4], event[5]) for event in item["events"]]
            item["exact_duplicate_count"] = len(dynamic_keys) - len(set(dynamic_keys))
            item["events"] = kept
        else:
            item["static_dedup_count"] = 0
            item["exact_duplicate_count"] = 0
    if not materialize_events and spool_path:
        # Compute compact audit counts without materializing event payloads.
        count_rows = conn.execute("SELECT admission, COUNT(*) FROM events GROUP BY admission").fetchall()
        for admission, count in count_rows:
            if admission in admissions:
                admissions[admission]["event_count"] = int(count)
        duplicate_rows = conn.execute(
            "SELECT admission, ts, vid, value, COUNT(*) FROM events "
            "GROUP BY admission, ts, vid, value HAVING COUNT(*) > 1"
        ).fetchall()
        duplicate_counts: Dict[str, int] = defaultdict(int)
        for admission, _ts, _vid, _value, count in duplicate_rows:
            duplicate_counts[admission] += int(count) - 1
        for admission, item in admissions.items():
            item["exact_duplicate_count"] = duplicate_counts.get(admission, 0)
    conn.close()
    return admissions, {"conflicts": conflict_rows, "icu_stay_count": len(seen_stays), "spool_path": spool_path,
                        "materialize_events": materialize_events, "static_vids": static_vids}


def _emit_admissions(admissions: Mapping[str, Mapping[str, Any]], staging: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    unique_vids: set[int] = set()
    for admission in sorted(admissions):
        item = admissions[admission]
        if item.get("exclusion_reason"):
            audit.append({"hadm_id": admission, "subject_id": item.get("subject", ""), "split": item.get("split", ""), "patientunitstayids": json.dumps(item.get("stays", [])), "n_icu_stays": len(item.get("stays", [])), "exclusion_reason": item["exclusion_reason"]})
            continue
        events = item["events"]
        if not events:
            continue
        split = item["split"]
        times = [event[0] for event in events]
        tokens = [event[4] for event in events]
        values = [event[5] for event in events]
        out = {"subject_id": item["subject"], "hadm_id": admission, "patienthealthsystemstayid": admission,
               "patientunitstayids": list(item["stays"]), "n_icu_stays": len(item["stays"]), "split": split,
               "time_origin": "hospital_admission", "sample_unit": "patienthealthsystemstayid",
               "state_tokens": torch.tensor(tokens, dtype=torch.long), "state_values": torch.tensor(values, dtype=torch.float32),
               "state_times": torch.tensor(times, dtype=torch.float32), "inter_tokens": torch.zeros(0, dtype=torch.long),
               "inter_values": torch.zeros(0), "inter_times": torch.zeros(0),
               "labels": {"mortality": next(iter(item["mortalities"])), "hospital_mortality": next(iter(item["mortalities"])),
                          "hospital_los": float(item["hospital_los"][0]), "patienthealthsystemstayid": admission}}
        path = os.path.join(staging, split, f"{admission}.pt"); torch.save(out, path)
        unique_vids.update(tokens)
        rows.append({"subject_id": item["subject"], "hadm_id": admission, "patienthealthsystemstayid": admission,
                     "patientunitstayids": json.dumps(item["stays"]), "n_icu_stays": len(item["stays"]), "split": split,
                     "hospital_los": round(float(item["hospital_los"][0]), 6), "hospital_mortality": next(iter(item["mortalities"])),
                     "mortality": next(iter(item["mortalities"])), "icu_mortality": ""})
        audit.append({"hadm_id": admission, "subject_id": item["subject"], "split": split, "patientunitstayids": json.dumps(item["stays"]),
                      "n_icu_stays": len(item["stays"]), "n_events": len(events), "n_unique_timestamps": len(set(times)),
                      "min_time": min(times), "max_time": max(times), "hospital_los": item["hospital_los"][0],
                      "hospital_mortality": next(iter(item["mortalities"])), "static_dedup_count": item.get("static_dedup_count", 0),
                      "exact_duplicate_count": item.get("exact_duplicate_count", 0),
                      "exclusion_reason": ""})
    return rows, {"rows": audit, "unique_vids": sorted(unique_vids)}


def _emit_admissions_from_spool(admissions: Mapping[str, Mapping[str, Any]], spool_path: str,
                                staging: str, static_vids: Mapping[str, int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Emit one admission at a time so event tensors never accumulate globally."""
    conn = sqlite3.connect(spool_path)
    rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    unique_vids: set[int] = set()
    static_values = set(static_vids.values())
    admission_ids = sorted(admissions)
    for admission in tqdm(admission_ids, desc="eICU resume: emit admissions", unit="admission"):
        source = admissions[admission]
        if source.get("exclusion_reason"):
            audit.append({"hadm_id": admission, "subject_id": source.get("subject", ""), "split": source.get("split", ""), "patientunitstayids": json.dumps(source.get("stays", [])), "n_icu_stays": len(source.get("stays", [])), "exclusion_reason": source["exclusion_reason"]})
            continue
        admit_by_stay = {str(meta.get("patientunitstayid", stay)): _safe_float(meta.get("hospitaladmitoffset"))
                         for stay, meta in zip(source["stays"], source["meta"])}
        events = []
        static_by_vid: Dict[int, set[float]] = defaultdict(set)
        # The event admission key is written from the same validated identity
        # as the stays table.  Querying by admission uses the canonical
        # (admission, stay) index and avoids one random lookup per ICU stay.
        stay_ids = [str(stay) for stay in source["stays"]]
        event_rows = conn.execute(
            "SELECT stay, ts, vid, value, ord FROM events WHERE admission=? ORDER BY stay, ord",
            (admission,),
        ).fetchall()
        if not event_rows and stay_ids:
            # Compatibility for an older completed spool whose event-level
            # admission text is stale.  Canonical spools take the indexed
            # admission path above; this fallback is only reached on a
            # malformed/legacy spool and is never the formal hot path.
            placeholders = ",".join("?" for _ in stay_ids)
            event_rows = conn.execute(
                f"SELECT stay, ts, vid, value, ord FROM events WHERE stay IN ({placeholders}) ORDER BY stay, ord",
                stay_ids,
            ).fetchall()
        if not event_rows:
            raise RuntimeError(
                "No events found in resume spool for admission "
                f"{admission}; member patientunitstayids={stay_ids}. "
                "The spool contains stay metadata but its event rows cannot be matched."
            )
        event_stays: set[str] = set()
        post_discharge_stays: set[str] = set()
        filtered_post_discharge_events = 0
        expected_stays = set(stay_ids)
        time_violation: Optional[Dict[str, Any]] = None
        hospital_los_days = _safe_float(source.get("hospital_los", [None])[0] if source.get("hospital_los") else None)
        # Event inclusion is strict: observations after hospital discharge are
        # not part of the admission sequence.  TOL_MIN is reserved for audit
        # validation of source offsets, not for retaining post-discharge data.
        max_hospital_time = None if hospital_los_days is None else hospital_los_days * 1440.0
        for stay, timestamp, vid, value, order in event_rows:
            if str(stay) not in expected_stays:
                raise RuntimeError(
                    f"Spool event admission/stay mismatch for admission={admission}: stay={stay}"
                )
            event_stays.add(str(stay))
            admit = admit_by_stay.get(str(stay))
            if admit is None:
                time_violation = {"stay": str(stay), "reason": "missing_hospitaladmitoffset"}
                break
            relative = float(timestamp) - float(admit)
            if relative < -1e-4:
                time_violation = {
                    "stay": str(stay), "reason": "negative_hospital_relative_time",
                    "raw_timestamp": float(timestamp), "hospitaladmitoffset": float(admit),
                    "relative_time": float(relative),
                }
                break
            if max_hospital_time is not None and relative > max_hospital_time + 1e-4:
                post_discharge_stays.add(str(stay))
                filtered_post_discharge_events += 1
                continue
            if int(vid) in static_values:
                static_by_vid[int(vid)].add(float(value))
            events.append((max(0.0, relative), -float(admit), str(stay), int(order), int(vid), float(value)))
        if time_violation is not None:
            audit.append({
                "hadm_id": admission, "subject_id": source.get("subject", ""),
                "split": source.get("split", ""),
                "patientunitstayids": json.dumps(source.get("stays", [])),
                "n_icu_stays": len(source.get("stays", [])),
                "exclusion_reason": time_violation["reason"],
                **time_violation,
            })
            continue
        missing_stays = sorted(expected_stays.difference(event_stays).difference(post_discharge_stays))
        if missing_stays:
            raise RuntimeError(
                f"Spool admission={admission} is missing events for member stays: {missing_stays[:20]}"
            )
        for vid, values in static_by_vid.items():
            if len(values) > 1:
                raise RuntimeError(f"Conflicting static value vid={vid} admission={admission}")
        events.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        seen_static: set[int] = set()
        kept = []
        for event in events:
            if event[4] in static_values:
                if event[4] in seen_static:
                    continue
                seen_static.add(event[4])
            kept.append(event)
        if not events:
            audit.append({
                "hadm_id": admission, "subject_id": source.get("subject", ""),
                "split": source.get("split", ""),
                "patientunitstayids": json.dumps(source.get("stays", [])),
                "n_icu_stays": len(source.get("stays", [])),
                "exclusion_reason": "no_valid_events_after_hospital_end",
                "filtered_post_discharge_event_count": filtered_post_discharge_events,
            })
            continue
        item = dict(source)
        item["events"] = kept
        item["static_dedup_count"] = len(events) - len(kept)
        item["post_discharge_event_count"] = filtered_post_discharge_events
        batch_rows, batch_audit = _emit_admissions({admission: item}, staging)
        for audit_row in batch_audit["rows"]:
            audit_row["filtered_post_discharge_event_count"] = filtered_post_discharge_events
        rows.extend(batch_rows); audit.extend(batch_audit["rows"]); unique_vids.update(batch_audit["unique_vids"])
    conn.close()
    return rows, {"rows": audit, "unique_vids": sorted(unique_vids)}


def _run_hospital_admission(input_dir: str, output_dir: str, schema_csv: str, labels_path: str, overwrite: bool,
                            resume_spool: Optional[str] = None,
                            spool_path_override: Optional[str] = None) -> None:
    print(f"[EICU-PREP] hospital_admission input={input_dir}", flush=True)
    print(f"[EICU-PREP] loading labels: {labels_path}", flush=True)
    if os.path.exists(output_dir) and not overwrite:
        raise RuntimeError(f"Output exists; pass --overwrite for atomic replacement: {output_dir}")
    labels = _load_labels_map(labels_path)
    if resume_spool:
        print(f"[EICU-PREP] resuming completed admission spool: {resume_spool}", flush=True)
    elif spool_path_override:
        print(f"[EICU-PREP] building persistent admission spool: {spool_path_override}", flush=True)
    else:
        print("[EICU-PREP] building temporary admission spool", flush=True)
    staging = tempfile.mkdtemp(prefix="eicu_prep.", dir=os.path.dirname(output_dir) or None)
    try:
        for split in ("train", "val", "test"): os.makedirs(os.path.join(staging, split), exist_ok=True)
        if resume_spool and spool_path_override:
            raise RuntimeError("Pass only one of --resume_spool or --spool_path")
        persistent_spool = bool(resume_spool or spool_path_override)
        spool_path = os.path.abspath(resume_spool or spool_path_override) if persistent_spool else os.path.join(staging, "admission_spool.sqlite")
        if spool_path_override and os.path.exists(spool_path):
            raise RuntimeError(f"Spool already exists; use --resume_spool to reuse it: {spool_path}")
        if persistent_spool:
            os.makedirs(os.path.dirname(spool_path), exist_ok=True)
        admissions, spool_report = _aggregate_hospital_admissions(input_dir, labels, schema_csv, spool_path,
                                                                  materialize_events=False, resume_spool=bool(resume_spool))
        rows, emitted = _emit_admissions_from_spool(admissions, spool_path, staging, spool_report["static_vids"])
        if not persistent_spool and os.path.exists(spool_path):
            os.remove(spool_path)
        if {row["split"] for row in rows} != {"train", "val", "test"}:
            spool_counts = {}
            if resume_spool:
                with sqlite3.connect(spool_path) as diagnostic_conn:
                    spool_counts = {
                        "stays": int(diagnostic_conn.execute("SELECT COUNT(*) FROM stays").fetchone()[0]),
                        "events": int(diagnostic_conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
                        "admissions": int(diagnostic_conn.execute("SELECT COUNT(DISTINCT admission) FROM stays").fetchone()[0]),
                    }
            raise RuntimeError(
                "hospital-admission output must contain train, val and test; "
                f"observed_split_counts={dict(Counter(row['split'] for row in rows))}. "
                f"resume_spool_counts={spool_counts}. "
                "Check that the input contains all splits or that the supplied labels.json "
                "has an explicit valid split for every ICU stay."
            )
        fields = ["subject_id", "hadm_id", "patienthealthsystemstayid", "patientunitstayids", "n_icu_stays", "split", "hospital_los", "hospital_mortality", "mortality", "icu_mortality"]
        _write_csv(os.path.join(staging, "labels.csv"), rows, fields)
        _write_csv(os.path.join(staging, "hospital_admission_audit.csv"), emitted["rows"], list(emitted["rows"][0].keys()) if emitted["rows"] else ["hadm_id"])
        distribution = []
        for split in ("train", "val", "test", "all"):
            part = [row for row in emitted["rows"] if not row.get("exclusion_reason") and (split == "all" or row["split"] == split)]
            stays = [row["n_icu_stays"] for row in part]
            distribution.append({"split": split, "n_samples": len(part), "n_unique_patients": len({row["subject_id"] for row in part}),
                "n_total_icu_stays": sum(stays), "n_multi_icu_admissions": sum(x > 1 for x in stays),
                "pct_multi_icu_admissions": (100.0 * sum(x > 1 for x in stays) / len(stays)) if stays else 0.0,
                "icu_stays_mean": float(np.mean(stays)) if stays else None, "icu_stays_median": float(np.median(stays)) if stays else None,
                "icu_stays_p95": float(np.percentile(stays, 95)) if stays else None, "icu_stays_max": max(stays) if stays else 0,
                "unique_timestamps_mean": float(np.mean([r["n_unique_timestamps"] for r in part])) if part else None,
                "unique_timestamps_median": float(np.median([r["n_unique_timestamps"] for r in part])) if part else None,
                "unique_timestamps_p95": float(np.percentile([r["n_unique_timestamps"] for r in part], 95)) if part else None,
                "events_mean": float(np.mean([r["n_events"] for r in part])) if part else None,
                "events_median": float(np.median([r["n_events"] for r in part])) if part else None,
                "events_p95": float(np.percentile([r["n_events"] for r in part], 95)) if part else None,
                "observation_span_mean": float(np.mean([r["max_time"] - r["min_time"] for r in part])) if part else None,
                "observation_span_median": float(np.median([r["max_time"] - r["min_time"] for r in part])) if part else None,
                "observation_span_p95": float(np.percentile([r["max_time"] - r["min_time"] for r in part], 95)) if part else None,
                "observed_24h_count": sum(r["min_time"] <= 1440 for r in part),
                "observed_48h_count": sum(r["min_time"] <= 2880 for r in part),
                "mortality_prevalence": float(np.mean([r["hospital_mortality"] for r in part])) if part else None,
                "los3_prevalence": float(np.mean([r["hospital_los"] >= 3 for r in part])) if part else None,
                "los7_prevalence": float(np.mean([r["hospital_los"] >= 7 for r in part])) if part else None,
                "static_dedup_count": sum(r.get("static_dedup_count", 0) for r in part),
                "exact_duplicate_dynamic_events": sum(r.get("exact_duplicate_count", 0) for r in part),
                "time_violation_count": 0})
        _write_csv(os.path.join(staging, "hospital_admission_distribution.csv"), distribution, list(distribution[0].keys()))
        _write_csv(os.path.join(staging, "prep_conflicts.csv"), spool_report["conflicts"], ["reason", "patientunitstayid", "admission_id"])
        excluded_rows = [row for row in emitted["rows"] if row.get("exclusion_reason")]
        summary = {"sample_unit": "patienthealthsystemstayid", "validation_protocol": "hospital_admission_level_external_validation",
                   "time_origin": "hospital_admission", "endpoint_unit": "hospital_admission", "num_written_samples": len(rows),
                   "num_icu_stays": spool_report["icu_stay_count"], "split_counts": Counter(row["split"] for row in rows),
                   "static_dedup_count": sum(row.get("static_dedup_count", 0) for row in emitted["rows"]), "time_violation_count": 0,
                   "excluded_admission_count": len(excluded_rows),
                   "excluded_missing_mortality_count": sum(row.get("exclusion_reason") == "missing_hospital_mortality" for row in excluded_rows),
                   "excluded_negative_hospital_los_count": sum(row.get("exclusion_reason") == "invalid_negative_hospital_los" for row in excluded_rows),
                   "excluded_missing_hospitaladmitoffset_count": sum(row.get("exclusion_reason") == "missing_hospitaladmitoffset" for row in excluded_rows),
                   "excluded_negative_hospital_relative_time_count": sum(row.get("exclusion_reason") == "negative_hospital_relative_time" for row in excluded_rows),
                   "excluded_event_exceeds_hospital_los_count": sum(row.get("exclusion_reason") == "event_exceeds_hospital_los" for row in excluded_rows),
                   "filtered_post_discharge_event_count": sum(row.get("filtered_post_discharge_event_count", 0) for row in emitted["rows"])}
        with open(os.path.join(staging, "prep_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(dict(summary), handle, indent=2, default=lambda value: dict(value))
        with open(os.path.join(staging, "unique_vids.csv"), "w", encoding="utf-8") as handle:
            handle.write("vid\n" + "\n".join(map(str, emitted["unique_vids"])) + "\n")
        _atomic_replace_dir(staging, output_dir); staging = ""
    finally:
        if staging and os.path.exists(staging): shutil.rmtree(staging, ignore_errors=True)


def _run_legacy_icu_stay(input_dir: str, output_dir: str, labels_path: str, overwrite: bool) -> None:
    """Reproduce the historical one-file-per-ICU-stay export.

    This branch is intentionally not accepted by external_validation
    preflight; it exists only so old results can be regenerated explicitly.
    """
    if os.path.exists(output_dir) and not overwrite:
        raise RuntimeError(f"Output exists; pass --overwrite: {output_dir}")
    labels = _load_labels_map(labels_path)
    staging = tempfile.mkdtemp(prefix="eicu_prep_legacy.", dir=os.path.dirname(output_dir) or None)
    rows = []
    try:
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(staging, split), exist_ok=True)
        for path in tqdm(_discover_shards(input_dir), desc="legacy ICU-stay prep", unit="shard"):
            for raw in _load_json_records(path):
                stay = str(raw.get("patientunitstayid", "")).strip()
                merged = dict(raw); merged.update(labels.get(stay, {}))
                sample = _build_sample(merged, _infer_split(path, merged))
                if sample is None:
                    continue
                torch.save(sample, os.path.join(staging, sample["split"], f"{stay}.pt"))
                rows.append({"subject_id": sample["subject_id"], "hadm_id": stay, "split": sample["split"],
                             "hospital_mortality": sample["labels"].get("hospital_mortality"),
                             "hospital_los": sample["labels"].get("hospital_los")})
        _write_csv(os.path.join(staging, "labels.csv"), rows, ["subject_id", "hadm_id", "split", "hospital_mortality", "hospital_los"])
        with open(os.path.join(staging, "prep_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"sample_unit": "patientunitstayid", "time_origin": "hospital_admission",
                       "validation_protocol": "legacy_icu_stay"}, handle, indent=2)
        _atomic_replace_dir(staging, output_dir); staging = ""
    finally:
        if staging and os.path.exists(staging): shutil.rmtree(staging, ignore_errors=True)


def _repair_processed_post_discharge(output_dir: str) -> None:
    """Repair already-emitted hospital samples without rereading raw shards."""
    output_dir = os.path.abspath(output_dir)
    summary_path = os.path.join(output_dir, "prep_summary.json")
    if not os.path.isfile(summary_path):
        raise RuntimeError(f"Missing prep_summary.json: {summary_path}")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("sample_unit") != "patienthealthsystemstayid":
        raise RuntimeError("Post-discharge repair requires hospital-admission processed data.")
    quarantine_root = os.path.join(output_dir, "quarantine", "post_discharge")
    changed_rows: List[Dict[str, Any]] = []
    excluded_ids: set[str] = set()
    for split in ("train", "val", "test"):
        for path in tqdm(sorted(Path(output_dir, split).glob("*.pt")), desc=f"repair {split}", unit="stay"):
            item = torch.load(path, map_location="cpu", weights_only=False)
            labels = item.get("labels", {})
            los = _safe_float(labels.get("hospital_los"))
            if los is None:
                continue
            tokens = torch.as_tensor(item.get("state_tokens", []), dtype=torch.long).reshape(-1)
            values = torch.as_tensor(item.get("state_values", []), dtype=torch.float32).reshape(-1)
            times = torch.as_tensor(item.get("state_times", []), dtype=torch.float32).reshape(-1)
            if not (tokens.numel() == values.numel() == times.numel()):
                raise RuntimeError(f"Invalid processed sample lengths: {path}")
            keep = times <= float(los * 1440.0 + 1e-4)
            removed = int((~keep).sum().item())
            if removed == 0:
                continue
            hadm_id = str(item.get("hadm_id", path.stem))
            kept_count = int(keep.sum().item())
            if kept_count == 0:
                quarantine_path = Path(quarantine_root, split, path.name)
                quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, quarantine_path)
                excluded_ids.add(hadm_id)
                changed_rows.append({"hadm_id": hadm_id, "split": split, "removed_events": removed,
                                     "action": "quarantined_no_valid_events"})
                continue
            item["state_tokens"] = tokens[keep]
            item["state_values"] = values[keep]
            item["state_times"] = times[keep]
            tmp = f"{path}.tmp.{os.getpid()}"
            torch.save(item, tmp)
            os.replace(tmp, path)
            changed_rows.append({"hadm_id": hadm_id, "split": split, "removed_events": removed,
                                 "action": "filtered_post_discharge_events"})
    if excluded_ids:
        labels_path = os.path.join(output_dir, "labels.csv")
        if os.path.isfile(labels_path):
            frame = pd.read_csv(labels_path)
            frame = frame.loc[~frame["hadm_id"].astype(str).isin(excluded_ids)].copy()
            tmp = f"{labels_path}.tmp.{os.getpid()}"
            frame.to_csv(tmp, index=False)
            os.replace(tmp, labels_path)
    audit_path = os.path.join(output_dir, "post_discharge_repair_audit.csv")
    _write_csv(audit_path, changed_rows, ["hadm_id", "split", "removed_events", "action"])
    summary["filtered_post_discharge_event_count"] = int(summary.get("filtered_post_discharge_event_count", 0)) + sum(int(row["removed_events"]) for row in changed_rows)
    summary["post_discharge_repair_excluded_admission_count"] = len(excluded_ids)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=lambda value: dict(value))
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"changed_samples": len(changed_rows), "excluded_admissions": len(excluded_ids),
                      "removed_events": sum(int(row["removed_events"]) for row in changed_rows)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess eICU trajectories.")
    parser.add_argument("--input_dir", required=False); parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schema_csv", required=False); parser.add_argument("--labels_json", required=False)
    parser.add_argument("--sample_unit", choices=("hospital_admission", "icu_stay"), default="hospital_admission")
    parser.add_argument("--labels_only", action="store_true", help="Legacy-only labels regeneration; rejected for formal validation.")
    parser.add_argument("--repair_post_discharge", action="store_true", help="Filter post-discharge events in existing processed samples without reading raw shards.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume_spool", default=None,
                        help="Reuse a completed admission_spool.sqlite and skip raw shard scanning.")
    parser.add_argument("--spool_path", default=None,
                        help="Persist the admission spool at this exact path for crash recovery; existing files are never overwritten.")
    args = parser.parse_args()
    print(f"[EICU-PREP] command sample_unit={args.sample_unit}", flush=True)
    if args.labels_only:
        raise RuntimeError("--labels_only cannot create a formal hospital-admission dataset")
    if args.repair_post_discharge:
        _repair_processed_post_discharge(os.path.abspath(args.output_dir))
        return
    for name in ("input_dir", "schema_csv", "labels_json"):
        if not getattr(args, name):
            parser.error(f"--{name} is required unless --repair_post_discharge is used")
    if args.sample_unit == "icu_stay":
        _run_legacy_icu_stay(os.path.abspath(args.input_dir), os.path.abspath(args.output_dir), os.path.abspath(args.labels_json), bool(args.overwrite))
    else:
        _run_hospital_admission(os.path.abspath(args.input_dir), os.path.abspath(args.output_dir), os.path.abspath(args.schema_csv), os.path.abspath(args.labels_json), bool(args.overwrite), args.resume_spool, args.spool_path)


if __name__ == "__main__":
    main()
