"""Claim-to-rule verdict aggregation."""

from __future__ import annotations

from .schema import ClaimCheckResult, ClaimVerdict


RISK_ORDER = {
    ClaimVerdict.FAIL: 3,
    ClaimVerdict.NEEDS_SOURCE: 2,
    ClaimVerdict.REVIEW: 1,
    ClaimVerdict.PASS: 0,
}


def aggregate_claim_verdicts(results: list[ClaimCheckResult]) -> tuple[ClaimVerdict, str]:
    if not results:
        return ClaimVerdict.NEEDS_SOURCE, "No atomic claims were available for this rule."

    worst = max((r.verdict for r in results), key=lambda v: RISK_ORDER[v])
    if worst == ClaimVerdict.PASS:
        return worst, "All atomic claims passed the layered gate."

    bad = [r.claim.claim_id for r in results if r.verdict == worst]
    return worst, f"{worst.value} at claim(s): {', '.join(bad)}"
