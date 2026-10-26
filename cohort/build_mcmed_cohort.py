"""Build the primary ED disposition cohort from MC-MED.

Reference standard: the treating clinician's actual ED disposition.

  gold = "ADMIT"      ED disposition in (ICU, Inpatient, Observation)
  gold = "DISCHARGE"  ED disposition == Discharge

All other dispositions are ambiguous with respect to the admit-versus-discharge
decision and are excluded by design.

The cohort is restricted to Emergency Severity Index level 3, the mid-acuity
band nearest the disposition threshold, where the decision is genuinely
discretionary and a framing effect is therefore detectable. It is then sampled
to an exact 1:1 balance of clinician-admitted and clinician-discharged
encounters, so that sensitivity and specificity are estimated with equal
precision and the balanced accuracy of an operating point is interpretable.

Sampling is seeded, and candidate rows are sorted by encounter identifier before
the draw, so the cohort is byte-reproducible regardless of the order the
database returns rows in.

Requires credentialed PhysioNet access to MC-MED v1.0.1 (doi:10.13026/wvyw-g663).

Usage:
    python -m cohort.build_mcmed_cohort --db mcmed.duckdb --out mcmed_cohort.jsonl
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
ADMIT_DISPOSITIONS = ("ICU", "Inpatient", "Observation")

QUERY = """
    SELECT
        c.CSN, c.Age, c.Gender, c.Means_of_arrival, c.Triage_acuity,
        c.Triage_HR, c.Triage_SBP, c.Triage_DBP, c.Triage_RR,
        c.Triage_SpO2, c.Triage_Temp, c.Triage_Pain,
        c.CC, v.ED_dispo, v.Race, v.Payor_class, v.MRN
    FROM v_triage_capsule_raw c
    JOIN visits v ON c.CSN = v.CSN
"""
COLS = ["csn", "age", "sex", "arrival_mode", "esi_acuity", "hr", "sbp", "dbp",
        "rr", "spo2", "temp", "pain", "chiefcomplaint", "ed_dispo", "race",
        "payor_class", "mrn"]


def parse_esi(acuity_str):
    """MC-MED stores acuity as a label such as '3-Urgent'; take the leading level."""
    if not acuity_str:
        return None
    s = str(acuity_str).strip()
    for i in range(1, 6):
        if s.startswith(str(i)):
            return i
    return None


def gold_label(row):
    if row["ed_dispo"] in ADMIT_DISPOSITIONS:
        return "ADMIT"
    if row["ed_dispo"] == "Discharge":
        return "DISCHARGE"
    return None


def load_context(conn):
    pmh = {r[0]: r[1] for r in conn.execute(
        "SELECT MRN, LIST(DISTINCT DescCCS ORDER BY DescCCS) FROM pmh GROUP BY MRN").fetchall()}
    meds = {r[0]: r[1] for r in conn.execute(
        "SELECT MRN, LIST(DISTINCT Generic_name ORDER BY Generic_name) "
        "FROM meds WHERE Active = 'Y' GROUP BY MRN").fetchall()}
    return pmh, meds


def build_record(row, gold, pmh, meds):
    capsule_row = dict(row, esi_acuity=ESI_LEVEL)
    return {
        "csn": int(row["csn"]),
        "dataset": "mcmed",
        "gold": gold,
        "capsule_text": build_capsule_text(
            capsule_row, pmh.get(row["mrn"], []), meds.get(row["mrn"], [])),
        "esi": ESI_LEVEL,
        "ed_dispo": row["ed_dispo"],
        "race": row["race"],
        "payor_class": row["payor_class"],
        "sex": row["sex"],
        "age": row["age"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path, help="path to the MC-MED DuckDB file")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-per-arm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    rows = [dict(zip(COLS, r)) for r in conn.execute(QUERY).fetchall()]
    pmh, meds = load_context(conn)
    conn.close()
    print(f"Loaded {len(rows):,} encounters")

    pools = {"ADMIT": [], "DISCHARGE": []}
    for r in rows:
        if parse_esi(r["esi_acuity"]) != ESI_LEVEL:
            continue
        g = gold_label(r)
        if g:
            pools[g].append(r)
    print(f"ESI-{ESI_LEVEL} pool: ADMIT={len(pools['ADMIT']):,}  "
          f"DISCHARGE={len(pools['DISCHARGE']):,}")

    n_each = min(args.n_per_arm, len(pools["ADMIT"]), len(pools["DISCHARGE"]))
    if n_each < args.n_per_arm:
        print(f"WARNING: pool limits the cohort to {n_each} per arm")

    rng = np.random.RandomState(args.seed)
    records = []
    for gold in ("ADMIT", "DISCHARGE"):
        pool = sorted(pools[gold], key=lambda r: int(r["csn"]))
        for i in rng.choice(len(pool), size=n_each, replace=False):
            records.append(build_record(pool[int(i)], gold, pmh, meds))
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
