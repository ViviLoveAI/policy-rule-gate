"""Shared data structures for the policy -> rule -> faithfulness-gate pipeline.

These schemas are the contracts between pipeline stages. Every stage in the
diagram consumes one of these and produces the next:

    policy text
        -> extract.py   -> list[Rule]            (structured rules)
        -> verbalize.py -> Rule.claim populated   (NL claim sentence)
        -> gate.py      -> list[GateResult]       (support/verdict + evidence)
        -> execute.py   -> list[ClaimDecision]    (adjudication of sample claims)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Coverage(str, Enum):
    """Whether a matched claim should be paid under a rule."""
    COVERED = "covered"
    NOT_COVERED = "not_covered"


class Verdict(str, Enum):
    """Outcome of the faithfulness gate for a single rule."""
    SUPPORTED = "supported"          # grounded in the source policy -> passes gate
    NOT_SUPPORTED = "not_supported"  # not grounded -> blocked, routed to human review


@dataclass
class Rule:
    """A single executable rule extracted from the policy.

    `conditions` and `exclusions` are simple predicates over a claim's fields
    (kept intentionally minimal for the POC). `claim_sentence` is the natural
    language rendering used by the faithfulness gate; it is filled in by
    verbalize.py, not by extraction.
    """
    rule_id: str
    service: str
    coverage: Coverage
    conditions: list[str] = field(default_factory=list)   # e.g. ["age_ge:65"]
    exclusions: list[str] = field(default_factory=list)   # e.g. ["diagnosis_in:Z00"]
    source_hint: str = ""                                  # optional: where in policy
    claim_sentence: Optional[str] = None                  # set by verbalize.py


@dataclass
class GateResult:
    """Result of running one rule's claim sentence through the faithfulness gate."""
    rule_id: str
    claim_sentence: str
    verdict: Verdict
    confidence: float                 # 0..1
    evidence: str                     # the retrieved policy sentence used as source
    passed: bool                      # convenience: verdict == SUPPORTED and conf >= threshold


@dataclass
class ClaimDecision:
    """Adjudication of one sample claim against the set of gate-passed rules."""
    claim_id: str
    matched_rule_id: Optional[str]
    decision: str                     # "paid" | "denied" | "no_matching_rule"
    rationale: str
