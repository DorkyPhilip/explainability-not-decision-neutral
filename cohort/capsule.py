"""Render a triage encounter into the fixed-format text presented to the model.

Both cohort builders emit through this single renderer so that the MC-MED and
MIMIC-IV-ED capsules are surface-identical and the framing contrast cannot be
confounded by formatting differences between datasets.

Every field is either a value or the literal string UNKNOWN; no field is ever
omitted, so prompt length does not covary with data completeness.
"""

from __future__ import annotations

FIELDS = ("age", "sex", "arrival_mode", "esi", "hr", "sbp", "dbp", "rr",
          "spo2", "temp", "pain", "chiefcomplaint")


def build_capsule_text(row, pmh_list=None, med_list=None) -> str:
    age = row.get("age", "UNKNOWN")
    sex = row.get("sex", "UNKNOWN")
    via = row.get("arrival_mode", "UNKNOWN") or "UNKNOWN"
    esi = row.get("esi_acuity") or "UNKNOWN"

    hr, sbp, dbp = row.get("hr"), row.get("sbp"), row.get("dbp")
    rr, spo2, temp, pain = row.get("rr"), row.get("spo2"), row.get("temp"), row.get("pain")

    vitals = [
        f"HR {int(hr)}" if hr is not None else "HR UNKNOWN",
        f"BP {int(sbp)}/{int(dbp)}" if sbp is not None and dbp is not None else "BP UNKNOWN",
        f"RR {int(rr)}" if rr is not None else "RR UNKNOWN",
        f"SpO2 {int(spo2)}%" if spo2 is not None else "SpO2 UNKNOWN",
        f"Temp {temp:.1f}C" if temp is not None else "Temp UNKNOWN",
        f"Pain {int(pain)}/10" if pain is not None else "Pain UNKNOWN",
    ]

    # Derived flags are computed identically for both datasets from the triage
    # values above; they add no information beyond the vitals already shown.
    flags = []
    if sbp is not None and sbp < 90:
        flags.append("hypotension")
    if spo2 is not None and spo2 < 92:
        flags.append("hypoxia")
    if hr is not None and hr > 100:
        flags.append("tachycardia")
    if rr is not None and rr > 22:
        flags.append("tachypnea")
    if temp is not None and temp > 38.0:
        flags.append("fever")
    if temp is not None and temp < 36.0:
        flags.append("hypothermia")

    dx_str = "; ".join(str(x) for x in pmh_list if x) if pmh_list else "none documented"
    med_str = "; ".join(str(x) for x in med_list if x) if med_list else "none active"

    return "\n".join([
        f"PATIENT: {age}yo {sex} via {via} | ESI-{esi}",
        f"VITALS: {' | '.join(vitals)}",
        f"CC: {row.get('chiefcomplaint', 'UNKNOWN') or 'UNKNOWN'}",
        f"FLAGS: {', '.join(flags) if flags else 'none'}",
        f"PMH: {dx_str}",
        f"MEDICATIONS: {med_str}",
    ])
