"""Rule-to-atomic-claim conversion.

The gate checks individual claims rather than long conjunctive rules. This
keeps retrieval and verification evidence-scoped: each condition gets its own
source span and its own verdict.
"""

from __future__ import annotations

from src.schema import Rule

from .schema import AtomicClaim


CONDITION_CLAIMS = {
    "diagnosis:diabetes_mellitus": (
        "The beneficiary has diabetes mellitus.",
        False,
    ),
    "training_documented:true": (
        "The treating practitioner has concluded that the beneficiary or caregiver has sufficient training using the prescribed CGM, as evidenced by providing a prescription.",
        False,
    ),
    "fda_indication:true": (
        "The CGM is prescribed in accordance with its FDA indications for use.",
        False,
    ),
    "insulin_treated:true": (
        "The beneficiary is insulin-treated.",
        False,
    ),
    "problematic_hypoglycemia:true": (
        "The beneficiary has a history of problematic hypoglycemia.",
        True,
    ),
    "hypoglycemia_level2_recurrent:true": (
        "The beneficiary has recurrent level 2 hypoglycemic events with glucose less than 54 mg/dL.",
        True,
    ),
    "attempts_adjustment_multiple:true": (
        "The recurrent level 2 hypoglycemic events persist despite multiple attempts to adjust medications or modify the diabetes treatment plan.",
        True,
    ),
    "hypoglycemia_level3_event:true": (
        "The beneficiary has a history of one level 3 hypoglycemic event with glucose less than 54 mg/dL.",
        True,
    ),
    "third_party_assistance:true": (
        "The level 3 hypoglycemic event required third-party assistance for treatment of hypoglycemia.",
        True,
    ),
    "six_month_visit:true": (
        "Within six months prior to ordering the CGM, the treating practitioner had an in-person or Medicare-approved telehealth visit with the beneficiary to evaluate diabetes control and determine that criteria 1 through 4 were met.",
        True,
    ),
    "age_ge:65": (
        "The beneficiary must be aged 65 or older to be eligible for CGM coverage.",
        True,
    ),
    "prior_auth:approved": (
        "Prior authorization must be approved before CGM coverage.",
        True,
    ),
}


def condition_to_claim(token: str) -> tuple[str, bool]:
    """Return a checkable claim for a condition token."""

    if token in CONDITION_CLAIMS:
        return CONDITION_CLAIMS[token]
    readable = token.replace("_", " ").replace(":", " equals ")
    return f"The policy requires {readable}.", True


def rule_to_atomic_claims(rule: Rule) -> list[AtomicClaim]:
    """Convert a structured rule into atomic faithfulness claims."""

    claims: list[AtomicClaim] = []
    for idx, token in enumerate(rule.conditions, start=1):
        text, high_risk = condition_to_claim(token)
        claims.append(
            AtomicClaim(
                rule_id=rule.rule_id,
                claim_id=f"{rule.rule_id}.C{idx}",
                text=text,
                condition_token=token,
                source_hint=rule.source_hint,
                high_risk=high_risk,
            )
        )

    for idx, token in enumerate(rule.exclusions, start=1):
        text, high_risk = condition_to_claim(token)
        claims.append(
            AtomicClaim(
                rule_id=rule.rule_id,
                claim_id=f"{rule.rule_id}.E{idx}",
                text=f"The policy excludes cases where {text[0].lower() + text[1:]}",
                condition_token=token,
                source_hint=rule.source_hint,
                high_risk=True,
            )
        )

    if not claims:
        claims.append(
            AtomicClaim(
                rule_id=rule.rule_id,
                claim_id=f"{rule.rule_id}.C1",
                text=f"The policy states that {rule.service} is {rule.coverage.value}.",
                condition_token="coverage",
                source_hint=rule.source_hint,
                high_risk=True,
            )
        )
    return claims
