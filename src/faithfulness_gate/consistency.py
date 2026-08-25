"""Evidence-scoped structured consistency checks.

This layer deliberately avoids naive global number matching. It extracts
attributes only from the retrieved evidence span and the atomic claim, then
compares values that share a semantic slot.
"""

from __future__ import annotations

import re

from .schema import ClaimVerdict, ConsistencyResult


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "twelve": 12,
}


INTRODUCTION_SLOTS = {
    "visit_window",
    "glucose_threshold",
    "hypoglycemia_level",
    "event_count",
    "insulin_treated",
    "diagnosis",
    "fda_indication",
    "training_documented",
    "third_party_assistance",
    "age_threshold",
    "prior_authorization",
}


COMPARABLE_SLOTS = {
    "visit_window",
    "glucose_threshold",
    "hypoglycemia_level",
    "event_count",
    "age_threshold",
}


def _num(value: str) -> int | None:
    value = value.lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def _extract_attributes(text: str) -> dict:
    t = text.lower()
    attrs: dict = {}

    if "six" in t and "month" in t and ("visit" in t or "ordering" in t):
        attrs["visit_window"] = {"value": 6, "unit": "months"}

    glucose = re.search(r"(?:glucose\s*)?(?:less than|<)\s*(\d+)\s*mg/?dl", t)
    if glucose:
        attrs["glucose_threshold"] = {
            "operator": "<",
            "value": int(glucose.group(1)),
            "unit": "mg/dL",
        }

    level = re.search(r"level\s*([23])\s+hypogly", t)
    if level:
        attrs["hypoglycemia_level"] = {"value": int(level.group(1))}

    if "more than one" in t or "recurrent" in t:
        attrs["event_count"] = {"slot": "level2_recurrent", "value": ">1"}
    elif re.search(r"\bone\b", t) and "level 3" in t:
        attrs["event_count"] = {"slot": "level3_single", "value": 1}

    if "insulin-treated" in t or "insulin treated" in t:
        attrs["insulin_treated"] = True

    if "diabetes mellitus" in t:
        attrs["diagnosis"] = "diabetes_mellitus"

    if "fda indication" in t or "fda indications" in t:
        attrs["fda_indication"] = True

    if "sufficient training" in t and "prescription" in t:
        attrs["training_documented"] = True

    if "third-party assistance" in t or "third party assistance" in t:
        attrs["third_party_assistance"] = True

    age = re.search(r"(?:aged|age)\s*(\d+)\s*(?:or older|older|\+)?", t)
    if age:
        attrs["age_threshold"] = {"operator": ">=", "value": int(age.group(1))}

    if "prior authorization" in t:
        attrs["prior_authorization"] = True

    if "not covered" in t or "not eligible for coverage" in t:
        attrs["polarity"] = "not_covered"
    elif "covered" in t or "eligible for coverage" in t or "coverage of" in t:
        attrs["polarity"] = "covered"
    else:
        attrs["polarity"] = None
    attrs["modality"] = "must" if "must" in t else "may" if "may" in t else None
    return attrs


def check_consistency(claim: str, evidence: str) -> ConsistencyResult:
    """Check hard attribute conflicts between an atomic claim and its evidence."""

    claim_attrs = _extract_attributes(claim)
    evidence_attrs = _extract_attributes(evidence)
    flags: list[str] = []

    for slot in COMPARABLE_SLOTS:
        if slot in claim_attrs and slot in evidence_attrs and claim_attrs[slot] != evidence_attrs[slot]:
            flags.append(f"{slot} mismatch: claim={claim_attrs[slot]} evidence={evidence_attrs[slot]}")

    if claim_attrs.get("polarity") and evidence_attrs.get("polarity"):
        if claim_attrs["polarity"] != evidence_attrs["polarity"]:
            flags.append(f"polarity mismatch: claim={claim_attrs['polarity']} evidence={evidence_attrs['polarity']}")

    for slot in sorted(INTRODUCTION_SLOTS):
        if slot in claim_attrs and slot not in evidence_attrs:
            flags.append(
                f"claim introduces structured attribute absent from retrieved evidence: "
                f"{slot}={claim_attrs[slot]}"
            )

    hard_fail = any("mismatch" in f or "introduces" in f for f in flags)
    verdict = ClaimVerdict.FAIL if hard_fail else ClaimVerdict.PASS
    return ConsistencyResult(
        verdict=verdict,
        risk_flags=flags,
        extracted_claim_attributes=claim_attrs,
        extracted_evidence_attributes=evidence_attrs,
    )
