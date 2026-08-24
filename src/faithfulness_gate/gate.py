"""Layered faithfulness gate orchestrator."""

from __future__ import annotations

from src.schema import Rule

from .adversarial import AdversarialAdjudicator
from .aggregate import aggregate_claim_verdicts
from .atomic_claims import rule_to_atomic_claims
from .consistency import check_consistency
from .retrieval import SourceRetriever
from .schema import (
    ClaimCheckResult,
    ClaimVerdict,
    HumanReviewItem,
    RuleGateResult,
    SourceSpan,
)
from .semantic import check_semantic_support


class LayeredFaithfulnessGate:
    """Policy-rule gate with source, consistency, semantic, and adversarial layers."""

    def __init__(
        self,
        policy_text: str,
        source_threshold: float = 0.08,
        semantic_pass_threshold: float = 0.42,
        run_adversarial: bool | None = None,
    ):
        self.policy_text = policy_text
        self.source_threshold = source_threshold
        self.semantic_pass_threshold = semantic_pass_threshold
        self.retriever = SourceRetriever(policy_text)
        self.adjudicator = AdversarialAdjudicator(enabled=run_adversarial)
        self.run_adversarial = self.adjudicator.enabled

    def _needs_adversarial(
        self,
        high_risk: bool,
        consistency_verdict: ClaimVerdict,
        semantic_verdict: ClaimVerdict,
    ) -> bool:
        if consistency_verdict == ClaimVerdict.FAIL:
            return False
        return high_risk or semantic_verdict == ClaimVerdict.REVIEW

    def check_claim(self, claim, sibling_claim_texts: list[str] | None = None) -> ClaimCheckResult:
        spans = self.retriever.retrieve(claim.text, top_k=3)
        best = spans[0] if spans else SourceSpan("", 0.0)

        if best.score < self.source_threshold:
            return ClaimCheckResult(
                claim=claim,
                verdict=ClaimVerdict.NEEDS_SOURCE,
                source_span=best,
                consistency=None,
                semantic=None,
                adversarial=None,
                reason="No sufficiently relevant source span was retrieved.",
            )

        consistency = check_consistency(claim.text, best.text)
        if consistency.verdict == ClaimVerdict.FAIL:
            return ClaimCheckResult(
                claim=claim,
                verdict=ClaimVerdict.FAIL,
                source_span=best,
                consistency=consistency,
                semantic=None,
                adversarial=None,
                reason="; ".join(consistency.risk_flags),
            )

        semantic = check_semantic_support(
            claim.text,
            best.text,
            pass_threshold=self.semantic_pass_threshold,
        )
        adversarial = None

        if self.run_adversarial and self._needs_adversarial(claim.high_risk, consistency.verdict, semantic.verdict):
            adversarial = self.adjudicator.run(
                claim=claim.text,
                evidence=best.text,
                policy_text=self.policy_text,
                reason=semantic.reason,
                sibling_claims=sibling_claim_texts or [],
            )
            verdict = adversarial.verdict
            reason = adversarial.judge_reasoning or adversarial.prosecutor_report
        else:
            verdict = semantic.verdict
            reason = semantic.reason

        return ClaimCheckResult(
            claim=claim,
            verdict=verdict,
            source_span=best,
            consistency=consistency,
            semantic=semantic,
            adversarial=adversarial,
            reason=reason,
        )

    def run_rule(self, rule: Rule) -> RuleGateResult:
        claims = rule_to_atomic_claims(rule)
        sibling_claim_texts = [c.text for c in claims]
        results = [self.check_claim(c, sibling_claim_texts) for c in claims]
        verdict, reason = aggregate_claim_verdicts(results)
        return RuleGateResult(
            rule_id=rule.rule_id,
            verdict=verdict,
            claim_results=results,
            passed=verdict == ClaimVerdict.PASS,
            reason=reason,
        )

    def run_rules(self, rules: list[Rule]) -> list[RuleGateResult]:
        return [self.run_rule(rule) for rule in rules]

    def human_review_queue(self, results: list[RuleGateResult]) -> list[HumanReviewItem]:
        queue: list[HumanReviewItem] = []
        for result in results:
            if result.passed:
                continue
            for claim_result in result.claim_results:
                if claim_result.verdict == ClaimVerdict.PASS:
                    continue
                queue.append(
                    HumanReviewItem(
                        rule_id=result.rule_id,
                        claim_id=claim_result.claim.claim_id,
                        verdict=claim_result.verdict,
                        reason=claim_result.reason,
                        claim_text=claim_result.claim.text,
                        evidence=claim_result.source_span.text if claim_result.source_span else "",
                    )
                )
        return queue
