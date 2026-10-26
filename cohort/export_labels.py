"""Recover the clinician disposition labels withheld from the released dataset.

The released ``data/decisions.csv.gz`` contains model decisions only. The treating
clinician's actual disposition is derived from MC-MED and MIMIC-IV-ED and is
governed by their data use agreements, so it is not redistributed here.

Credentialed users can regenerate it. The surrogate identifier ``<dataset>-<index>``
is the encounter's position in the seeded cohort file, and the cohort builders are
deterministic, so re-running them reproduces the identical cohort in the identical
order. This script reads those cohort files and writes the label keyed by surrogate
identifier, which restores the clinician-comparator analysis and Fig 3.

Usage:
    python -m cohort.build_mcmed_cohort --db mcmed.duckdb --out mcmed_cohort.jsonl
    python -m cohort.build_mimic_cohort --db mimic.duckdb --out mimic_cohort.jsonl
    python -m cohort.export_labels \
        --mcmed mcmed_cohort.jsonl --mimic mimic_cohort.jsonl \
        --out data/clinician_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = [("mcmed", "mcmed"), ("mimic-iv-ed", "mimic")]


def labels_from(path, dataset):
    """Yield (surrogate_id, clinician_disposition) in cohort-file order."""
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            yield f"{dataset}-{i:05d}", rec["gold"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcmed", help="MC-MED cohort JSONL from build_mcmed_cohort")
    ap.add_argument("--mimic", help="MIMIC-IV-ED cohort JSONL from build_mimic_cohort")
    ap.add_argument("--out", default="data/clinician_labels.csv")
    args = ap.parse_args()

    if not args.mcmed and not args.mimic:
        ap.error("provide --mcmed and/or --mimic")

    rows = []
    for dataset, arg in [("mcmed", args.mcmed), ("mimic-iv-ed", args.mimic)]:
        if not arg:
            continue
        n = 0
        for sid, gold in labels_from(arg, dataset):
            rows.append((sid, gold))
            n += 1
        print(f"  {dataset}: {n:,} labels")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["encounter_id", "clinician_disposition"])
        w.writerows(rows)

    print(f"\nWrote {len(rows):,} labels -> {out}")
    print("The analysis and figures pick this file up automatically.")
    print("Do not commit it: it is derived from credentialed source data.")


if __name__ == "__main__":
    main()
