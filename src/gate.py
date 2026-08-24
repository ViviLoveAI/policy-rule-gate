"""Stage 3 -- Faithfulness gate.

For each rule's claim sentence:
    1. retrieve the most relevant policy sentence (source),
    2. run the NLI detector: is the claim entailed by that source?
    3. pass the rule only if SUPPORTED with confidence >= threshold;
       otherwise BLOCK it and route to human review.

This is the differentiating stage: it prevents any rule not grounded in the
source policy from reaching production, and it records the evidence + confidence
for each decision (the audit trail).
"""

from __future__ import annotations

from .nli import NLIDetector
from .retriever import LexicalRetriever
from .schema import GateResult, Rule, Verdict


class FaithfulnessGate:
    def __init__(
        self,
        policy_sentences: list[str],
        detector: NLIDetector,
        pass_threshold: float = 0.45,
    ):
        self.retriever = LexicalRetriever(policy_sentences)
        self.detector = detector
        self.pass_threshold = pass_threshold

    def check(self, rule: Rule) -> GateResult:
        assert rule.claim_sentence is not None, "verbalize the rule before gating"
        source, _ = self.retriever.retrieve(rule.claim_sentence, top_k=1)[0]
        verdict, conf = self.detector.judge(rule.claim_sentence, source)
        passed = verdict == Verdict.SUPPORTED and conf >= self.pass_threshold
        return GateResult(
            rule_id=rule.rule_id,
            claim_sentence=rule.claim_sentence,
            verdict=verdict,
            confidence=conf,
            evidence=source,
            passed=passed,
        )

    def run(self, rules: list[Rule]) -> list[GateResult]:
        return [self.check(r) for r in rules]
