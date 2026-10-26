"""Reproduce every reported result from the public decision-level dataset.

This module operates entirely on ``data/decisions.csv.gz``, the decision-level
dataset released with the article. It requires no access to MC-MED or
MIMIC-IV-ED and no API key. Every number, table, and figure in the manuscript can
be regenerated from the released file alone, except the comparisons with the
treating clinician's disposition (Fig 3 and S5 Table), which is withheld because
it derives from the credentialed datasets. See ``cohort/export_labels.py``.

Determinism: all interval estimation uses paired bootstrap resampling with
BOOTSTRAP_SEED. Re-running reproduces published values exactly.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 5
N_BOOTSTRAP = 5000

DATA = Path(__file__).resolve().parent.parent / "data" / "decisions.csv.gz"

MODELS = [
    ("gemma-4-31b", "Gemma 4 31B"),
    ("nemotron-120b", "Nemotron Super 120B"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash"),
]
MODEL_KEYS = [m for m, _ in MODELS]

LABELS = {
    "bare": "Disposition only (no confidence)",
    "routine": "Unmonitored",
    "audited": "Observed (record + sign-off context)",
    "neutral_justify": "Neutral justification (reasoning first)",
    "documented": "Documented reasoning and risks (reasoning first)",
    "documented_norisk": "Documented reasoning, risk wording removed",
    "post_justify": "Documented justification (recommendation first)",
    "reasoned": "Private native reasoning",
    "capacity": "Unmonitored + high bed-occupancy cue",
    "mech_discard": "Discarded neutral note",
    "mech_record": "Recorded neutral note",
    "mech_risk": "Documented reasoning and risks",
    "mech_risk1": "Recorded note, one required risk",
    "mech_risk3": "Recorded note, three required risks",
    "mech_risk5": "Recorded note, five required risks",
    "mech_filler": "Length-matched irrelevant transcription",
    "valence_admit": "Risks of admitting",
    "valence_discharge": "Risks of discharging",
    "valence_both": "Risks of both dispositions",
}
LABELS.update({f"nolegal_{k}": f"{LABELS[k]}, audit wording removed"
               for k in ("mech_record", "mech_risk", "valence_admit",
                         "valence_discharge", "valence_both")})

# The contrasts that constitute Table 1 of the manuscript.
TABLE1 = [
    ("A. Requiring a justification is itself a decision input", None, None),
    ("Recorded neutral note vs Unmonitored", "routine", "mech_record"),
    ("Discarded neutral note vs Unmonitored", "routine", "mech_discard"),
    ("Clinical content vs length-matched irrelevant note", "mech_filler", "mech_risk3"),
    ("B. Components of the documentation effect", None, None),
    ("Record and audit the note vs discard it", "mech_discard", "mech_record"),
    ("Documented reasoning and risks vs recorded neutral note", "mech_record", "mech_risk"),
    ("Require one risk vs recorded neutral note", "mech_record", "mech_risk1"),
    ("Require five risks vs recorded neutral note", "mech_record", "mech_risk5"),
    ("C. Risk valence sets the direction", None, None),
    ("Name risks of discharging vs recorded neutral note", "mech_record", "valence_discharge"),
    ("Name risks of admitting vs recorded neutral note", "mech_record", "valence_admit"),
    ("Directional span (discharging minus admitting)", "valence_admit", "valence_discharge"),
    ("Name both harms symmetrically vs recorded neutral note", "mech_record", "valence_both"),
    ("Name both harms symmetrically vs Unmonitored", "routine", "valence_both"),
]

# S7 Table: output-format changes with no clinical content, and the decomposition
# steps judged against the largest of them in each model.
FORMAT_ONLY = [
    ("Add a confidence field to a decision-only request", "bare", "routine"),
    ("Length-matched irrelevant transcription vs Unmonitored", "routine", "mech_filler"),
]
DECOMPOSITION = [
    ("Discarded neutral note vs Unmonitored", "routine", "mech_discard"),
    ("Discarded neutral note vs irrelevant transcription", "mech_filler", "mech_discard"),
    ("Record and audit the note vs discard it", "mech_discard", "mech_record"),
    ("Documented reasoning and risks vs recorded neutral note", "mech_record", "mech_risk"),
    ("Directional span (discharging minus admitting)", "valence_admit", "valence_discharge"),
]

# S8 Table: run 3 repeats these arms with only "subject to quality-assurance audit
# and medico-legal review" removed (condition prefix "nolegal_"), alongside a
# same-batch Unmonitored arm.
ABLATION_ARMS = ["mech_record", "mech_risk", "valence_admit", "valence_discharge", "valence_both"]
ABLATION_WITHIN = [
    ("Documented reasoning and risks vs recorded neutral note", "mech_record", "mech_risk"),
    ("Directional span (discharging minus admitting)", "valence_admit", "valence_discharge"),
    ("Name both harms symmetrically vs recorded neutral note", "mech_record", "valence_both"),
]


# --------------------------------------------------------------------------
LABELS_PATH = DATA.parent / "clinician_labels.csv"


def load_labels():
    """Map encounter_id -> clinician disposition, or {} when unavailable.

    The clinician label is derived from the credentialed source datasets and is
    withheld from the release, so this file is absent by default. Credentialed
    users regenerate it with ``python -m cohort.export_labels``; everything that
    does not involve the clinician comparator runs without it.
    """
    if not LABELS_PATH.exists():
        return {}
    with LABELS_PATH.open(encoding="utf-8", newline="") as fh:
        return {r["encounter_id"]: r["clinician_disposition"] or None
                for r in csv.DictReader(fh)}


def load(path: Path | str = DATA, dataset: str = "mcmed", run: int = 1):
    """Return (by, gold) where by[(model, encounter_id)][condition] = record."""
    by = defaultdict(dict)
    labels = load_labels()
    gold = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] != dataset or int(r["run"]) != run:
                continue
            by[(r["model"], r["encounter_id"])][r["condition"]] = {
                "disposition": r["disposition"],
                "confidence": r["confidence"] or None,
            }
            gold[r["encounter_id"]] = (
                r.get("clinician_disposition") or labels.get(r["encounter_id"]) or None
            )
    return by, gold


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def contrast(by, arm_a, arm_b, model_keys=None):
    """Paired bootstrap change in admission rate, arm_b minus arm_a, in points."""
    rows = []
    for mk in (model_keys or MODEL_KEYS):
        # Deterministic pair order so bootstrap resamples depend only on the data.
        for eid in sorted(eid for (mm, eid) in by if mm == mk):
            fr = by[(mk, eid)]
            if arm_a in fr and arm_b in fr:
                rows.append((fr[arm_a]["disposition"] == "ADMIT",
                             fr[arm_b]["disposition"] == "ADMIT"))
    if not rows:
        return None
    a = np.array(rows, dtype=float)
    n = len(a)
    d = (a[:, 1].mean() - a[:, 0].mean()) * 100
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    bs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        ix = rng.randint(0, n, n)
        bs[i] = (a[ix, 1].mean() - a[ix, 0].mean()) * 100
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0 / N_BOOTSTRAP)
    return {
        "delta": round(float(d), 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1),
        "p": float(p), "star": star(p),
        "flip_pct": round(100 * float(np.mean(a[:, 0] != a[:, 1])), 1),
        "n_pairs": n,
        "rate_a": round(float(a[:, 0].mean() * 100), 1),
        "rate_b": round(float(a[:, 1].mean() * 100), 1),
    }


def did(by, a1, b1, a2, b2, model_keys=None):
    """Paired bootstrap difference in differences, (b2 - a2) - (b1 - a1), in points."""
    rows = []
    for mk in (model_keys or MODEL_KEYS):
        for eid in sorted(eid for (mm, eid) in by if mm == mk):
            fr = by[(mk, eid)]
            if all(k in fr for k in (a1, b1, a2, b2)):
                rows.append([fr[k]["disposition"] == "ADMIT" for k in (a1, b1, a2, b2)])
    if not rows:
        return None
    x = np.array(rows, dtype=float)

    def stat(m):
        return ((m[:, 3].mean() - m[:, 2].mean()) - (m[:, 1].mean() - m[:, 0].mean())) * 100

    n = len(x)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    bs = np.array([stat(x[rng.randint(0, n, n)]) for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0 / N_BOOTSTRAP)
    return {"delta": round(float(stat(x)), 1), "lo": round(float(lo), 1),
            "hi": round(float(hi), 1), "p": float(p), "star": star(p), "n_pairs": n}


def merge_runs(base, other, tag):
    """Return base with another run's conditions added as '<condition>@<tag>'."""
    out = defaultdict(dict)
    for key, fr in base.items():
        out[key].update(fr)
    for key, fr in other.items():
        out[key].update({f"{c}@{tag}": v for c, v in fr.items()})
    return out


def admission_rate(by, condition, model_key=None):
    vals = [fr[condition]["disposition"] == "ADMIT"
            for (mm, _), fr in by.items()
            if condition in fr and (model_key is None or mm == model_key)]
    if not vals:
        return None
    k, n = int(sum(vals)), len(vals)
    lo, hi = wilson(k, n)
    return {"rate": round(100 * k / n, 1), "lo": round(lo, 1), "hi": round(hi, 1), "n": n}


def operating_point(by, gold, condition, model_key):
    tp = fn = fp = tn = 0
    for (mm, eid), fr in by.items():
        if mm != model_key or condition not in fr:
            continue
        pred = fr[condition]["disposition"]
        g = gold.get(eid)
        if g == "ADMIT":
            tp += pred == "ADMIT"
            fn += pred == "DISCHARGE"
        elif g == "DISCHARGE":
            fp += pred == "ADMIT"
            tn += pred == "DISCHARGE"
    if not (tp + fn) or not (fp + tn):
        return None
    sens = 100 * tp / (tp + fn)
    spec = 100 * tn / (fp + tn)
    return {"sens": round(sens, 1), "spec": round(spec, 1),
            "bacc": round((sens + spec) / 2, 1), "n": tp + fn + fp + tn}


def reproducibility(path: Path | str = DATA, dataset: str = "mcmed"):
    """Per-decision agreement and rate stability between the two runs."""
    r1, _ = load(path, dataset, run=1)
    r2, _ = load(path, dataset, run=2)
    out = {}
    conditions = sorted({c for fr in r2.values() for c in fr})
    for cond in conditions:
        for mk, name in MODELS:
            agree = tot = a1 = a2 = 0
            for (mm, eid), fr in r1.items():
                if mm != mk or cond not in fr:
                    continue
                f2 = r2.get((mm, eid))
                if not f2 or cond not in f2:
                    continue
                x, y = fr[cond]["disposition"], f2[cond]["disposition"]
                agree += x == y
                tot += 1
                a1 += x == "ADMIT"
                a2 += y == "ADMIT"
            if tot:
                out[(cond, mk)] = {
                    "agreement_pct": round(100 * agree / tot, 1),
                    "rate_run1": round(100 * a1 / tot, 1),
                    "rate_run2": round(100 * a2 / tot, 1),
                    "abs_rate_diff": round(abs(100 * a1 / tot - 100 * a2 / tot), 1),
                    "n": tot,
                }
    return out


def table1(by):
    """Reproduce Table 1 of the manuscript."""
    out = []
    for label, a, b in TABLE1:
        if a is None:
            out.append({"header": label})
        else:
            out.append({"label": label, "pooled": contrast(by, a, b)})
    return out


def main():
    print(f"Loading {DATA}\n")
    by, gold = load()

    print("=== Table 1: pooled paired contrasts (MC-MED primary cohort) ===")
    for row in table1(by):
        if "header" in row:
            print(f"\n{row['header']}")
            continue
        c = row["pooled"]
        print(f"  {row['label']:<56} {c['delta']:+6.1f} "
              f"({c['lo']:+.1f}, {c['hi']:+.1f}) {c['star']:>3}")

    print("\n=== Admission rate by condition (pooled) ===")
    for cond in LABELS:
        r = admission_rate(by, cond)
        if r:
            print(f"  {LABELS[cond]:<46} {r['rate']:5.1f}%  (n={r['n']})")

    print("\n=== Duplicate-run noise floor ===")
    for (cond, mk), v in reproducibility().items():
        print(f"  {LABELS.get(cond, cond):<44} {mk:<20} "
              f"agree={v['agreement_pct']:5.1f}%  rate diff={v['abs_rate_diff']:.1f}pp")

    print("\n=== Operating points vs clinician disposition ===")
    if not any(gold.values()):
        print("  Skipped: the clinician label is withheld from the public release")
        print("  because it derives from the credentialed source datasets.")
        print("  Credentialed users can restore it, which enables this section")
        print("  and Fig 3:")
        print("    python -m cohort.build_mcmed_cohort --db mcmed.duckdb --out mcmed_cohort.jsonl")
        print("    python -m cohort.export_labels --mcmed mcmed_cohort.jsonl")
    else:
        for cond in ("valence_admit", "mech_record", "mech_risk", "valence_discharge"):
            for mk in MODEL_KEYS:
                op = operating_point(by, gold, cond, mk)
                if op:
                    print(f"  {LABELS.get(cond, cond):<40} {mk:<20} "
                          f"sens={op['sens']:5.1f}%  spec={op['spec']:5.1f}%  "
                          f"balanced acc={op['bacc']:5.1f}%")

    def row(label, c_pooled, cells):
        print(f"  {label:<56} {c_pooled['delta']:+6.1f} |"
              + "".join(f" {c['delta']:+6.1f}" for c in cells))

    print("\n=== S7 Table: decomposition vs output-format effects (pooled | "
          + ", ".join(MODEL_KEYS) + ") ===")
    envelope = {mk: 0.0 for mk in MODEL_KEYS}
    for label, a, b in FORMAT_ONLY:
        cells = [contrast(by, a, b, [mk]) for mk in MODEL_KEYS]
        for mk, c in zip(MODEL_KEYS, cells):
            envelope[mk] = max(envelope[mk], abs(c["delta"]))
        row(label, contrast(by, a, b), cells)
    print("  largest format-only shift: "
          + ", ".join(f"{mk} {v:.1f}" for mk, v in envelope.items()))
    for label, a, b in DECOMPOSITION:
        cells = [contrast(by, a, b, [mk]) for mk in MODEL_KEYS]
        row(label, contrast(by, a, b), cells)
        print(f"  {'':<56}   exceeds format range: " + ", ".join(
            f"{mk} {'yes' if abs(c['delta']) > envelope[mk] else 'no'}"
            for mk, c in zip(MODEL_KEYS, cells)))

    run3, _ = load(run=3)
    if run3:
        m = merge_runs(by, run3, 3)
        print("\n=== S8 Table: audit and medico-legal wording ablation (run 3) ===")
        c = contrast(m, "routine", "routine@3")
        print(f"  Unmonitored drift, run 3 vs run 1: {c['delta']:+.1f} "
              f"({c['lo']:+.1f}, {c['hi']:+.1f}), {c['flip_pct']:.1f}% of decisions differ")
        for arm in ABLATION_ARMS:
            w = contrast(m, "routine", arm)
            wo = contrast(m, "routine@3", f"nolegal_{arm}@3")
            d = did(m, "routine", arm, "routine@3", f"nolegal_{arm}@3")
            print(f"  {LABELS[arm]:<34} vs Unmonitored: with {w['delta']:+5.1f}  "
                  f"removed {wo['delta']:+5.1f}  difference {d['delta']:+5.1f} "
                  f"({d['lo']:+.1f}, {d['hi']:+.1f})")
        for label, a, b in ABLATION_WITHIN:
            w = contrast(m, a, b)
            wo = contrast(m, f"nolegal_{a}@3", f"nolegal_{b}@3")
            print(f"  {label:<56} with {w['delta']:+5.1f}  removed {wo['delta']:+5.1f} "
                  f"({wo['lo']:+.1f}, {wo['hi']:+.1f})")

    mim, mgold = load(dataset="mimic-iv-ed")
    if mim:
        print("\n=== External replication (MIMIC-IV-ED) ===")
        for label, a, b in TABLE1:
            if a is None:
                continue
            c = contrast(mim, a, b)
            if c:
                print(f"  {label:<56} {c['delta']:+6.1f} ({c['lo']:+.1f}, {c['hi']:+.1f}) {c['star']}")
        c = contrast(mim, "routine", "mech_risk")
        if c:
            print(f"  {'Documented reasoning and risks vs Unmonitored':<56} "
                  f"{c['delta']:+6.1f} ({c['lo']:+.1f}, {c['hi']:+.1f}) {c['star']}")


if __name__ == "__main__":
    main()
