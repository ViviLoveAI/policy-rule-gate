"""Schemas for the layered policy-rule faithfulness gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimVerdict(str, Enum):
    """Gate outcome for a claim or whole rule."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    NEEDS_SOURCE = "NEEDS_SOURCE"


@dataclass
class AtomicClaim:
    """One checkable claim derived from a generated structured rule."""

    rule_id: str
    claim_id: str
    text: str
    condition_token: str
    source_hint: str = ""
    high_risk: bool = False


@dataclass
class SourceSpan:
    """Retrieved source evidence for a claim."""

    text: str
    score: float


@dataclass
class ConsistencyResult:
    """Evidence-scoped deterministic consistency result."""

    verdict: ClaimVerdict
    risk_flags: list[str] = field(default_factory=list)
    extracted_claim_attributes: dict = field(default_factory=dict)
    extracted_evidence_attributes: dict = field(default_factory=dict)


@dataclass
class SemanticResult:
    """Semantic support result for a claim against its source span."""

    verdict: ClaimVerdict
    confidence: float
    reason: str = ""


@dataclass
class AdversarialResult:
    """Conditional prosecutor-judge adjudication result."""

    verdict: ClaimVerdict
    confidence: float
    prosecutor_report: str = ""
    judge_reasoning: str = ""
    raw: str = ""


@dataclass
class ClaimCheckResult:
    """Full layered result for one atomic claim."""

    claim: AtomicClaim
    verdict: ClaimVerdict
    source_span: Optional[SourceSpan]
    consistency: Optional[ConsistencyResult]
    semantic: Optional[SemanticResult]
    adversarial: Optional[AdversarialResult]
    reason: str


@dataclass
class RuleGateResult:
    """Aggregated faithfulness result for one generated rule."""

    rule_id: str
    verdict: ClaimVerdict
    claim_results: list[ClaimCheckResult]
    passed: bool
    reason: str


@dataclass
class HumanReviewItem:
    """A rule or claim routed to a human reviewer."""

    rule_id: str
    claim_id: str
    verdict: ClaimVerdict
    reason: str
    claim_text: str
    evidence: str
    recommended_action: str = "Review before allowing this rule to execute."
