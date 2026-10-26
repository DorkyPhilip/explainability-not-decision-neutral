"""Build the external replication cohort from MIMIC-IV-ED.

Mirrors ``build_mcmed_cohort`` exactly, so the two datasets are analyzed under an
identical triage-capsule surface and an identical reference comparator:

  gold = "ADMIT"      ED disposition == ADMITTED
  gold = "DISCHARGE"  ED disposition == HOME

All other dispositions (transfer, eloped, left without being seen, left against
medical advice, expired, other) are ambiguous with respect to the
admit-versus-discharge decision and are excluded by design.

Vocabulary is harmonized to MC-MED (sex, arrival mode, temperature in Celsius) so
that the rendered capsule is surface-identical across datasets and the framing
contrast is not confounded by formatting differences.

Requires credentialed PhysioNet access to MIMIC-IV-ED v2.2 (doi:10.13026/5ntk-km72).

Usage:
    python -m cohort.build_mimic_cohort --db mimic.duckdb --out mimic_cohort.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

from .capsule import build_capsule_text

SEED = 42
ESI_LEVEL = 3

# Physiologic plausibility windows. MIMIC-IV-ED triage fields contain sentinel and
# data-entry artifacts (temperatures of 0.1 or 986 F, pain scores above 10). Values
# outside these windows are treated as missing and render as UNKNOWN, which is how
# genuinely absent values are handled for MC-MED as well.
RANGES = {
    "hr": (20.0, 250.0),
    "sbp": (50.0, 300.0),
    "dbp": (20.0, 200.0),
    "rr": (4.0, 60.0),
    "spo2": (50.0, 100.0),
    "temp_f": (90.0, 110.0),
    "pain": (0.0, 10.0),
}

ARRIVAL_MAP = {
    "WALK IN": "Self",
    "AMBULANCE": "EMS",
    "HELICOPTER": "EMS",
    "OTHER": "Other/Unknown",
    "UNKNOWN": "Other/Unknown",
}

QUERY = f"""
    SELECT
        c.stay_id, c.age, c.gender, c.arrival_transport,
        c.triage_hr, c.triage_sbp, c.triage_dbp, c.triage_rr,
        c.triage_spo2, c.triage_temp, c.triage_pain,
        c.cc, c.disposition, c.race, c.med_list, p.pmh_list
    FROM v_mimic_capsule c
    LEFT JOIN v_pmh_prior p ON c.stay_id = p.stay_id
    WHERE c.triage_acuity = {ESI_LEVEL}
      AND c.disposition IN ('ADMITTED', 'HOME')
      AND c.age >= 18
"""
COLS = ["stay_id", "age", "gender", "arrival_transport", "hr", "sbp", "dbp", "rr",
        "spo2", "temp_f", "pain", "cc", "disposition", "race", "med_list", "pmh_list"]


def _clean(value, key):
    """Return a float inside the plausibility window, else None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = RANGES[key]
    return v if lo <= v <= hi else None


def gold_label(row):
    if row["disposition"] == "ADMITTED":
        return "ADMIT"
    if row["disposition"] == "HOME":
        return "DISCHARGE"
    return None


def build_record(row, gold):
    temp_f = _clean(row["temp_f"], "temp_f")
    capsule_row = {
        "age": int(row["age"]) if row["age"] is not None else "UNKNOWN",
        "sex": row["gender"] or "U",
        "arrival_mode": ARRIVAL_MAP.get(
            (row["arrival_transport"] or "UNKNOWN").upper(), "Other/Unknown"),
        "esi_acuity": ESI_LEVEL,
        "hr": _clean(row["hr"], "hr"),
        "sbp": _clean(row["sbp"], "sbp"),
        "dbp": _clean(row["dbp"], "dbp"),
        "rr": _clean(row["rr"], "rr"),
        "spo2": _clean(row["spo2"], "spo2"),
        # MIMIC-IV-ED records triage temperature in Fahrenheit; MC-MED in Celsius.
        "temp": (temp_f - 32.0) * 5.0 / 9.0 if temp_f is not None else None,
        "pain": _clean(row["pain"], "pain"),
        "chiefcomplaint": row["cc"],
    }
    return {
        "csn": int(row["stay_id"]),
        "dataset": "mimic-iv-ed",
        "gold": gold,
        "capsule_text": build_capsule_text(
            capsule_row, row["pmh_list"] or [], row["med_list"] or []),
        "esi": ESI_LEVEL,
        "ed_dispo": row["disposition"],
        "race": row["race"],
        "payor_class": None,
        "sex": row["gender"],
        "age": int(row["age"]) if row["age"] is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path, help="path to the MIMIC-IV-ED DuckDB file")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-per-arm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    rows = [dict(zip(COLS, r)) for r in conn.execute(QUERY).fetchall()]
    conn.close()
    print(f"Eligible ESI-{ESI_LEVEL} encounters: {len(rows):,}")

    pools = {"ADMIT": [], "DISCHARGE": []}
    for r in rows:
        g = gold_label(r)
        if g:
            pools[g].append(r)
    print(f"Gold pool: ADMIT={len(pools['ADMIT']):,}  DISCHARGE={len(pools['DISCHARGE']):,}")

    n_each = min(args.n_per_arm, len(pools["ADMIT"]), len(pools["DISCHARGE"]))
    if n_each < args.n_per_arm:
        print(f"WARNING: pool limits the cohort to {n_each} per arm")

    rng = np.random.RandomState(args.seed)
    records = []
    for gold in ("ADMIT", "DISCHARGE"):
        pool = sorted(pools[gold], key=lambda r: int(r["stay_id"]))
        for i in rng.choice(len(pool), size=n_each, replace=False):
            records.append(build_record(pool[int(i)], gold))
    rng.shuffle(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_admit = sum(1 for r in records if r["gold"] == "ADMIT")
    print(f"Wrote {len(records):,} encounters "
          f"({n_admit} ADMIT / {len(records) - n_admit} DISCHARGE) -> {args.out}")
    print("  Sex:", dict(Counter(r["sex"] for r in records)))
    print("\n--- sample capsule ---")
    print(records[0]["capsule_text"])


if __name__ == "__main__":
    main()
