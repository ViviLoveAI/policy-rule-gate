"""Semantic support layer."""

from __future__ import annotations

import re

from .schema import ClaimVerdict, SemanticResult


_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "for", "with",
    "by", "in", "on", "that", "this", "has", "have", "had", "be", "been",
    "must", "may", "using", "use",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP


def check_semantic_support(claim: str, evidence: str, pass_threshold: float = 0.42) -> SemanticResult:
    """Lightweight support check.

    This is a deterministic stand-in for the richer NLI/adjudication path. It
    keeps the demo offline while preserving the same interface.
    """

    c = _tokens(claim)
    e = _tokens(evidence)
    if not c:
        return SemanticResult(ClaimVerdict.REVIEW, 0.0, "empty claim")
    overlap = len(c & e) / len(c)
    if overlap >= pass_threshold:
        return SemanticResult(ClaimVerdict.PASS, round(overlap, 3), "claim tokens are well supported by the retrieved evidence")
    if overlap >= pass_threshold * 0.55:
        return SemanticResult(ClaimVerdict.REVIEW, round(overlap, 3), "partial lexical-semantic support; escalate if high risk")
    return SemanticResult(ClaimVerdict.REVIEW, round(overlap, 3), "weak support from retrieved evidence")
