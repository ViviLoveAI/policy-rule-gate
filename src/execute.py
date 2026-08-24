"""Stage 4 -- Rule execution.

Applies the rules that passed the faithfulness gate to a handful of sample
claims, producing an adjudication decision for each. This is intentionally a
tiny toy engine -- just enough to show the gated rules actually *do* something.
"""

from __future__ import annotations

from .schema import ClaimDecision, Coverage, Rule


def _rule_matches(rule: Rule, claim: dict) -> bool:
    """A claim matches a rule if the claimed service matches and all of the
    rule's conditions are satisfied by the claim's attributes."""
    if rule.service.split()[0].lower() not in claim.get("service", "").lower():
        return False
    attrs = claim.get("attributes", {})
    for cond in rule.conditions:
        key, _, val = cond.partition(":")
        if key == "age_ge":
            if not (str(attrs.get("age", "")).isdigit() and int(attrs["age"]) >= int(val)):
                return False
        else:
            if str(attrs.get(key)).lower() != val.lower():
                return False
    return True


def adjudicate(claims: list[dict], passed_rules: list[Rule]) -> list[ClaimDecision]:
    decisions: list[ClaimDecision] = []
    for claim in claims:
        matched = next((r for r in passed_rules if _rule_matches(r, claim)), None)
        if matched is None:
            decisions.append(
                ClaimDecision(claim["claim_id"], None, "no_matching_rule",
                              "No gate-approved rule matched this claim.")
            )
            continue
        if matched.coverage == Coverage.COVERED:
            decision = "paid"
        else:
            decision = "denied"
        decisions.append(
            ClaimDecision(
                claim_id=claim["claim_id"],
                matched_rule_id=matched.rule_id,
                decision=decision,
                rationale=f"Matched {matched.rule_id}: {matched.claim_sentence}",
            )
        )
    return decisions
