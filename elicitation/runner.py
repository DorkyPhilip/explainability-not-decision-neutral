"""Run the elicitation experiment against an OpenAI-compatible inference endpoint.

Reproducing the released decisions requires credentialed access to MC-MED and
MIMIC-IV-ED plus an inference API key; see docs/REPRODUCING.md. Readers who only
want to verify the reported statistics do not need this module, because
``elicitation.analysis`` operates on the released decision-level dataset.

Decoding is temperature 0 with a fixed maximum token budget per condition class.
Sampling at temperature 0 is not bit-deterministic on batched GPU inference, so
the article reports an empirical noise floor from an independent duplicate run
rather than claiming per-decision determinism.

Usage:
    export INFERENCE_API_KEY=...
    python -m elicitation.runner --cohort mcmed_cohort.jsonl \
        --conditions routine mech_record mech_risk --out decisions_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .prompts import (
    FRAMINGS, MECH_SUFFIX, REASONED_FRAMINGS, USER_TEMPLATE, build_system_prompt,
)

BASE_URL = os.environ.get("INFERENCE_BASE_URL", "https://api.deepinfra.com/v1/openai")
ENV_KEY = "INFERENCE_API_KEY"

MODELS = {
    "gemma-4-31b": "google/gemma-4-31B-it",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    "nemotron-120b": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
}

# Models that emit chain-of-thought before the JSON answer need a larger budget
# so the trailing JSON is never truncated. The budget is held constant across
# conditions within a model, so it cannot confound the framing contrast.
REASONS_BEFORE_ANSWER = {"nemotron-120b"}

# Conditions whose output includes generated prose alongside the decision.
_WIDE_OUTPUTS = {"document", "scratchpad", "post_document", "neutral_document", "document_norisk"}

TEMPERATURE = 0
MAX_RETRIES = 6

_DISP_RE = re.compile(r'"?disposition"?\s*[:=]\s*"?\s*(ADMIT|DISCHARGE)\b', re.IGNORECASE)
_CONF_RE = re.compile(r'"?confidence"?\s*[:=]\s*"?\s*(LOW|MED|MEDIUM|HIGH)\b', re.IGNORECASE)


def parse_decision(raw):
    """Extract (disposition, confidence) from a model response.

    Prefers the last well-formed JSON object, which is where the answer sits for
    models that reason before answering; falls back to a regex scan so that a
    malformed wrapper does not discard an otherwise readable decision.
    """
    if not raw:
        return None, None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text)
    for chunk in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "disposition" in obj:
            d = str(obj.get("disposition", "")).strip().upper()
            c = str(obj.get("confidence", "")).strip().upper()
            c = {"MEDIUM": "MED"}.get(c, c)
            return (d if d in {"ADMIT", "DISCHARGE"} else None,
                    c if c in {"LOW", "MED", "HIGH"} else None)
    dm, cm = _DISP_RE.search(text), _CONF_RE.search(text)
    d = dm.group(1).upper() if dm else None
    c = cm.group(1).upper() if cm else None
    c = {"MEDIUM": "MED"}.get(c, c)
    return (d if d in {"ADMIT", "DISCHARGE"} else None,
            c if c in {"LOW", "MED", "HIGH"} else None)


def get_client():
    from openai import OpenAI
    key = os.environ.get(ENV_KEY)
    if not key:
        raise SystemExit(f"Set {ENV_KEY} before running.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def max_tokens_for(model_key, condition):
    wide = (condition in MECH_SUFFIX
            or FRAMINGS[condition][1] in _WIDE_OUTPUTS
            or condition in REASONED_FRAMINGS)
    if model_key in REASONS_BEFORE_ANSWER:
        return 1100 if wide else 768
    return 600 if wide else 64


def call_model(client, model_name, system_prompt, capsule, max_tokens, reasoning_effort=None):
    extra = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_TEMPLATE.format(capsule=capsule)},
                ],
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                extra_body=extra or None,
            )
            raw = (completion.choices[0].message.content or "").strip()
            disp, conf = parse_decision(raw)
            return {"disposition": disp, "confidence": conf, "raw": raw, "error": None}
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                return {"disposition": None, "confidence": None, "raw": None,
                        "error": str(exc)[:300]}
            time.sleep(2 ** attempt + 1)


def worker(client, model_key, condition, rec):
    res = call_model(
        client, MODELS[model_key], build_system_prompt(condition), rec["capsule_text"],
        max_tokens_for(model_key, condition),
        reasoning_effort="medium" if condition in REASONED_FRAMINGS else None,
    )
    return {
        "model": model_key,
        "condition": condition,
        "encounter_id": rec["csn"],
        "clinician_disposition": rec["gold"],
        "disposition": res["disposition"],
        "confidence": res["confidence"],
        "error": res["error"],
    }


def load_completed(path: Path):
    """Cells already answered, so an interrupted run resumes without repeating work."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None and r.get("disposition"):
                    done.add((r["model"], r["condition"], r["encounter_id"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conditions", nargs="+", default=["routine", "mech_record", "mech_risk"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--workers", type=int, default=80)
    args = ap.parse_args()

    unknown = [c for c in args.conditions if c not in FRAMINGS]
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")

    cohort = [json.loads(l) for l in args.cohort.open(encoding="utf-8")]
    completed = load_completed(args.out)
    tasks = [(m, c, r) for m in args.models for c in args.conditions for r in cohort
             if (m, c, r["csn"]) not in completed]
    print(f"{len(cohort):,} encounters x {len(args.models)} models x "
          f"{len(args.conditions)} conditions; {len(tasks):,} calls to make "
          f"({len(completed):,} already done)")
    if not tasks:
        return

    client = get_client()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    with args.out.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, client, m, c, r) for m, c, r in tasks]
        for fut in as_completed(futures):
            fh.write(json.dumps(fut.result()) + "\n")
            fh.flush()
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                print(f"  [{done}/{len(tasks)}] {rate:.1f}/s "
                      f"ETA {(len(tasks) - done) / rate / 60:.1f} min", file=sys.stderr)
    print(f"Wrote {done:,} decisions in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
