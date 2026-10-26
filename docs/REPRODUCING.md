# Reproducing the study

There are two levels of reproduction. Most readers want the first.

## Level 1: verify the reported statistics

Requires only Python and numpy. No data access, no API key, no GPU.

```bash
pip install numpy==2.3.3 matplotlib==3.10.6
python -m elicitation.analysis
python -m elicitation.figures
```

Everything reported in the article is computed from `data/decisions.csv.gz`,
which is the file released as Supporting Information. `elicitation/analysis.py`
prints, in order:

1. Marginal admission rate for each of the 19 elicitation conditions, pooled
   across models, with Wilson 95% intervals.
2. Table 1 of the article: the paired contrasts, each a within-encounter
   comparison across two conditions on the same patients.
3. The duplicate-run noise floor.
4. The MIMIC-IV-ED external replication.
5. S7 Table: the components of the documentation effect beside the effects of
   output format alone.
6. S8 Table: the audit-wording ablation (run 3), as differences in differences
   against the run 3 Unmonitored arm.

The operating-point statistics and Fig 3 are the exception. They compare each
recommendation against the treating clinician's disposition, which is derived
from the credentialed source datasets and is therefore withheld from the
release. `analysis.py` prints an explicit notice where that section would
appear, and `figures.py` skips Fig 3. Level 2 below restores both.

Interval estimates use a paired bootstrap with 5000 resamples,
`BOOTSTRAP_SEED = 5`, and a deterministic pair ordering (sorted by model, then
encounter identifier). Re-running reproduces the published intervals exactly,
not merely to within Monte Carlo error.

### Checking a single number

Every contrast is available programmatically:

```python
from elicitation.analysis import load, contrast
by, gold = load()
contrast(by, "valence_admit", "valence_discharge")
# {'delta': 15.2, 'lo': 14.3, 'hi': 16.2, 'p': 0.0002, 'star': '***',
#  'flip_pct': 16.0, 'n_pairs': 17982, 'rate_a': 67.0, 'rate_b': 82.2}
```

`delta` is the change in admission rate in percentage points when moving from
the first condition to the second, on the encounters answered under both.

## Level 2: rebuild from the source data

Requires credentialed PhysioNet access to both datasets and an inference API
key. See [DATA_ACCESS.md](DATA_ACCESS.md).

### Step 1: build the cohorts

```bash
python -m cohort.build_mcmed_cohort --db mcmed.duckdb --out mcmed_cohort.jsonl
python -m cohort.build_mimic_cohort --db mimic.duckdb --out mimic_cohort.jsonl
```

Both builders restrict to Emergency Severity Index level 3, label the reference
standard from the treating clinician's actual disposition, and sample an exact
1000 admitted / 1000 discharged cohort with `numpy.random.RandomState(42)`.
Candidate rows are sorted by encounter identifier before the draw, so the cohort
does not depend on the order the database returns rows in.

Both write through the same renderer (`cohort/capsule.py`), so the two datasets
present a surface-identical capsule to the model. Every field is either a value
or the literal string `UNKNOWN`; no field is ever omitted, so prompt length does
not covary with data completeness.

### Step 1b: restore the withheld clinician label

Having built a cohort, you can recover the label the release withholds, which
re-enables the operating-point statistics and Fig 3 without re-running inference:

```bash
python -m cohort.export_labels --mcmed mcmed_cohort.jsonl --mimic mimic_cohort.jsonl
```

This writes `data/clinician_labels.csv`, keyed by the same study identifier
used in `data/decisions.csv.gz`. The analysis and figure modules detect it
automatically. The mapping is by cohort position, which is well defined because
the builders are deterministic. Do not commit this file.

To confirm that your rebuilt cohort is the published one, compare the source
identifiers in your cohort file, in order, with `data/cohort_manifest.csv`.

### Step 2: run inference

```bash
export INFERENCE_API_KEY=...
# optional: export INFERENCE_BASE_URL=https://api.deepinfra.com/v1/openai

python -m elicitation.runner \
    --cohort mcmed_cohort.jsonl \
    --out decisions_raw.jsonl \
    --models gemma-4-31b nemotron-120b deepseek-v4-flash \
    --conditions routine bare audited documented documented_norisk neutral_justify \
                 post_justify reasoned capacity \
                 mech_discard mech_record mech_risk mech_risk1 mech_risk3 mech_risk5 \
                 mech_filler valence_admit valence_discharge valence_both
```

The runner is resumable: it reads the output file, skips cells that already have
a decision, and appends only what is missing. An interrupted run can be restarted
with the identical command.

The full primary run is 19 conditions x 3 models x 2000 encounters = 114,000
calls. The replication is 7 conditions x 3 models x 2000 encounters = 42,000
calls.

The audit-wording ablation (run 3) re-runs the Unmonitored condition alongside
the five conditions with the audit and medico-legal phrase removed, 6 x 3 x 2000
= 36,000 calls:

```bash
python -m elicitation.runner --cohort mcmed_cohort.jsonl --out decisions_raw_ablation.jsonl \
    --conditions routine nolegal_mech_record nolegal_mech_risk \
                 nolegal_valence_admit nolegal_valence_discharge nolegal_valence_both
```

### Step 3: cost and runtime

At the token budgets in `elicitation/runner.py`, the full primary run is on the
order of tens of US dollars against a commodity inference provider and completes
in a few hours at 80 to 100 concurrent workers. Throughput is provider-limited,
not client-limited.

## Known non-determinism

Decoding is at temperature 0, but batched GPU inference is not bit-deterministic
at temperature 0: kernel selection and batch composition vary between calls, so
an individual decision can differ between two runs of an identical prompt.

The article does not claim per-decision determinism. It quantifies the
instability directly. Three conditions were re-run in full as an independent
duplicate, giving per-decision agreement of 88 to 98 percent depending on the
model, and condition-level admission rates that moved by at most 1.1 percentage
points between runs. Because every reported effect is a condition-level rate
difference several times larger than that floor, the conclusions do not depend
on per-decision reproducibility. `python -m elicitation.analysis` prints the
floor next to the effects so the comparison is visible.
