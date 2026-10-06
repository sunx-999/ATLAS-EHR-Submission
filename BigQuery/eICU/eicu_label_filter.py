import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CORE_WHITELIST_PATTERNS = [
    r"\bhear\s*rate\b",
    r"\bheartrate\b",
    r"\bpulse\b",
    r"\brespiratory\s*rate\b",
    r"\brespiration\b",
    r"\bsao2\b",
    r"\bspo2\b",
    r"\bo2\s*saturation\b",
    r"\boxygen\s*saturation\b",
    r"\btemperature\b",
    r"\bsystolic\b",
    r"\bdiastolic\b",
    r"\bmean\s*bp\b",
    r"\bblood\s*pressure\b",
    r"\bnon[-\s]*invasive\s*bp\b",
    r"\binvasive\s*bp\b",
    r"\barterial\s*pressure\b",
    r"\bcvp\b",
    r"\burine\s*output\b",
    r"\bglucose\b",
    r"\bbedside\s*glucose\b",
    r"\bcreatinine\b",
    r"\bbun\b",
    r"\burea\s*nitrogen\b",
    r"\bsodium\b",
    r"\bpotassium\b",
    r"\bchloride\b",
    r"\bbicarbonate\b",
    r"\bco2\b",
    r"\bcalcium\b",
    r"\bionized\s*calcium\b",
    r"\bmagnesium\b",
    r"\bphosphate\b",
    r"\bphosphorus\b",
    r"\blactate\b",
    r"\balbumin\b",
    r"\bbilirubin\b",
    r"\bast\b",
    r"\balt\b",
    r"\balkaline\s*phosphatase\b",
    r"\binr\b",
    r"\bpt\b",
    r"\bptt\b",
    r"\bph\b",
    r"\bpco2\b",
    r"\bpo2\b",
    r"\bfio2\b",
    r"\bpeep\b",
    r"\bpressure\s*support\b",
    r"\bwbc\b",
    r"\bwhite\s*blood\s*cell\b",
    r"\brbc\b",
    r"\bred\s*blood\s*cell\b",
    r"\bhemoglobin\b",
    r"\bhematocrit\b",
    r"\bplatelet\b",
    r"\bplatelets\b",
    r"\brdw\b",
    r"\bgcs\b",
    r"\bbest\s*motor\s*response\b",
    r"\bbest\s*verbal\s*response\b",
    r"\bbest\s*eye\s*response\b",
    r"\bmotor\s*strength\b",
]

RARE_PATTERNS = [
    r"\bab\b",
    r"\bantibody\b",
    r"\bigg\b",
    r"\bigm\b",
    r"\bhsv\b",
    r"\blegionella\b",
    r"\brpr\b",
    r"\bana\b",
    r"\banf\b",
    r"\btiter\b",
]

BODY_FLUID_PATTERNS = [
    r"\bsynovial\b",
    r"\bperitoneal\b",
    r"\bpericardial\b",
    r"\bpleural\b",
    r"\bbody\s*fluid\b",
    r"\bcerebrospinal\b",
    r"\bcsf\b",
]

CORE_PATTERNS = [re.compile(p) for p in CORE_WHITELIST_PATTERNS]
RARE_ASSAY_PATTERNS = [re.compile(p) for p in RARE_PATTERNS]
BODY_FLUID_PATTERNS_COMPILED = [re.compile(p) for p in BODY_FLUID_PATTERNS]


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_missing(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def to_int(value, default=0):
    if is_missing(value):
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def load_records(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8").lstrip()
    if not raw:
        return []
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON array input must contain a top-level list.")
        return data

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def has_any_pattern(text, patterns):
    return any(p.search(text) for p in patterns)


def classify_row(row, min_primary, min_whitelist):
    n_adm = to_int(row.get("n_unique_admissions"))
    source = str(row.get("source", "")).strip()
    label_text = normalize_text(row.get("label"))
    action = str(row.get("suggested_normalization_action", "")).strip()

    core_whitelist_hit = n_adm >= min_whitelist and has_any_pattern(label_text, CORE_PATTERNS)
    rare_pattern_hit = (
        n_adm < min_primary
        and (
            has_any_pattern(label_text, RARE_ASSAY_PATTERNS)
            or has_any_pattern(label_text, BODY_FLUID_PATTERNS_COMPILED)
        )
    )
    manual_review_hit = n_adm >= min_whitelist and action in {
        "mixed_review_needed",
        "numeric_unit_review_needed",
    }

    if n_adm >= min_primary:
        return True, "admission_coverage_rule", core_whitelist_hit, rare_pattern_hit

    if n_adm < min_whitelist:
        return False, "below_min_admissions_whitelist", core_whitelist_hit, rare_pattern_hit

    if source == "customlab_state_core" and n_adm < min_primary:
        return False, "customlab_state_core_below_primary_threshold", core_whitelist_hit, rare_pattern_hit

    if rare_pattern_hit:
        return False, "rare_pattern_filtered", core_whitelist_hit, rare_pattern_hit

    if core_whitelist_hit:
        return True, "core_clinical_whitelist_override", core_whitelist_hit, rare_pattern_hit

    if manual_review_hit:
        return True, "manual_review_preservation", core_whitelist_hit, rare_pattern_hit

    return False, "did_not_meet_keep_criteria", core_whitelist_hit, rare_pattern_hit


def serialize_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def parse_maybe_json(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def format_admission_coverage(n_unique_admissions, total_admissions):
    if total_admissions in (None, 0):
        return "NA"
    n = to_int(n_unique_admissions)
    pct = (n / total_admissions) * 100.0
    return f"{pct:.3f}% ({n}/{total_admissions})"


def format_event_coverage(n_rows, total_event_rows):
    if total_event_rows in (None, 0):
        return "NA"
    n = to_int(n_rows)
    pct = (n / total_event_rows) * 100.0
    return f"{pct:.6f}% ({n}/{total_event_rows})"


def format_numeric_parse(n_numeric_parseable, n_nonmissing_raw_value):
    denom = to_int(n_nonmissing_raw_value, default=0)
    if denom <= 0:
        return "NA"
    num = to_int(n_numeric_parseable, default=0)
    pct = (num / denom) * 100.0
    return f"{pct:.1f}% ({num}/{denom})"


def format_numeric_distribution(row):
    fields = [
        ("numeric_p25", "p25"),
        ("numeric_median", "median"),
        ("numeric_p75", "p75"),
        ("numeric_min", "min"),
        ("numeric_max", "max"),
    ]
    parts = []
    has_any = False
    for key, label in fields:
        value = row.get(key)
        if is_missing(value):
            continue
        has_any = True
        parts.append(f"{label}={value}")
    return "; ".join(parts) if has_any else "NA"


def format_unit_hint(value):
    parsed = parse_maybe_json(value)
    if not isinstance(parsed, list):
        return "NA"
    seen = set()
    items = []
    for entry in parsed:
        if isinstance(entry, dict):
            unit = entry.get("unit_candidate")
        else:
            unit = entry
        if is_missing(unit):
            continue
        unit_text = str(unit).strip()
        if not unit_text:
            continue
        normalized = unit_text.lower()
        if normalized in {"nan", "none", "null"}:
            continue
        if unit_text not in seen:
            seen.add(unit_text)
            items.append(unit_text)
    return "; ".join(items) if items else "NA"


def format_value_examples(value):
    parsed = parse_maybe_json(value)
    if not isinstance(parsed, list):
        return "NA"
    examples = []
    for entry in parsed[:5]:
        if isinstance(entry, dict):
            raw_value = entry.get("raw_value")
        else:
            raw_value = entry
        if is_missing(raw_value):
            continue
        text = str(raw_value).strip()
        if text:
            examples.append(text)
    return " | ".join(examples) if examples else "NA"


def sum_n_rows(rows):
    total = 0
    for row in rows:
        total += to_int(row.get("n_rows"))
    return total


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("source", "")),
            -to_int(r.get("n_unique_admissions")),
            -to_int(r.get("n_rows")),
            str(r.get("label", "")),
        ),
    )


def print_counter(title, rows, key):
    print(title)
    if not rows:
        print("  none")
        return
    counter = Counter(str(row.get(key, "")) for row in rows)
    for item, count in counter.most_common():
        print(f"  {item}: {count}")


def print_top_labels(title, rows):
    print(title)
    if not rows:
        print("  none")
        return
    totals = defaultdict(int)
    for row in rows:
        totals[str(row.get("label", ""))] += to_int(row.get("n_rows"))
    for label, total in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        print(f"  {label}: {total}")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_cell(row.get(k)) for k in fieldnames})


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter eICU state labels before clustering/normalization.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--out-csv", default="state_label_filtered")
    parser.add_argument("--min-admissions-primary", type=int, default=500)
    parser.add_argument("--min-admissions-whitelist", type=int, default=100)
    parser.add_argument("--total-admissions", type=int, default=161175)
    parser.add_argument("--total-event-rows", type=int, default=None)
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise SystemExit("Input file is empty.")

    required_cols = ["source", "label", "n_rows", "n_unique_admissions", "profile_type", "suggested_normalization_action"]
    missing = [col for col in required_cols if any(col not in row for row in records)]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    output_rows = []
    kept_rows = []
    dropped_rows = []

    for row in records:
        keep, reason, core_hit, rare_hit = classify_row(
            row,
            args.min_admissions_primary,
            args.min_admissions_whitelist,
        )
        enriched = dict(row)
        enriched["filter_keep"] = keep
        enriched["filter_reason"] = reason
        enriched["core_whitelist_hit"] = core_hit
        enriched["rare_pattern_hit"] = rare_hit
        output_rows.append(enriched)
        if keep:
            kept_rows.append(enriched)
        else:
            dropped_rows.append(enriched)

    output_rows = sort_rows(output_rows)
    kept_rows = [r for r in output_rows if r["filter_keep"]]
    dropped_rows = [r for r in output_rows if not r["filter_keep"]]
    total_event_rows = args.total_event_rows if args.total_event_rows is not None else sum_n_rows(output_rows)
    kept_event_rows = sum_n_rows(kept_rows)

    total_rows = len(output_rows)
    kept_count = len(kept_rows)
    dropped_count = len(dropped_rows)
    retention_fraction = (kept_event_rows / total_event_rows) if total_event_rows else 0.0

    print(f"total input labels: {total_rows}")
    print(f"kept labels: {kept_count}")
    print(f"dropped labels: {dropped_count}")
    print(f"total event rows before filtering: {total_event_rows}")
    print(f"total event rows after filtering: {kept_event_rows}")
    print(f"fraction of event rows retained: {retention_fraction:.6f}")
    print(f"output CSV path: {args.out_csv}")
    print_counter("kept rows by source:", kept_rows, "source")
    print_counter("kept rows by suggested_normalization_action:", kept_rows, "suggested_normalization_action")
    print_top_labels("top 30 kept labels by n_rows:", kept_rows)

    out_csv = Path(args.out_csv)
    compact_rows = []
    for row in kept_rows:
        compact_rows.append({
            "source": row.get("source"),
            "label": row.get("label"),
            "profile_type": row.get("profile_type"),
            "suggested_normalization_action": row.get("suggested_normalization_action"),
            "admission_coverage": format_admission_coverage(row.get("n_unique_admissions"), args.total_admissions),
            "event_coverage": format_event_coverage(row.get("n_rows"), total_event_rows),
            "numeric_parse": format_numeric_parse(row.get("n_numeric_parseable"), row.get("n_nonmissing_raw_value")),
            "numeric_distribution": format_numeric_distribution(row),
            "unit_hint": format_unit_hint(row.get("unit_candidate_summary")),
            "value_examples": format_value_examples(row.get("raw_value_top50")),
        })
    out_csv = Path(args.out_csv)
    if out_csv.suffix == "":
        out_csv = out_csv.with_suffix(".csv")
    write_csv(out_csv, compact_rows, [
        "source",
        "label",
        "profile_type",
        "suggested_normalization_action",
        "admission_coverage",
        "event_coverage",
        "numeric_parse",
        "numeric_distribution",
        "unit_hint",
        "value_examples",
    ])

    if args.out_jsonl:
        write_jsonl(Path(args.out_jsonl), compact_rows)


if __name__ == "__main__":
    main()
