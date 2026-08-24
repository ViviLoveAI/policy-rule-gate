"""Pipeline orchestrator.

Wires the stages together in the order shown in the architecture diagram:

    policy text
      -> extract_rules       (Stage 1)
      -> verbalize_all       (Stage 2)
      -> FaithfulnessGate    (Stage 3)  --blocked--> human review queue
      -> adjudicate          (Stage 4)
      -> console report      (demo output)

Swap the three mock components for real ones in `build_pipeline` to go live.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .execute import adjudicate
from .extract import extract_rules
from .gate import FaithfulnessGate
from .llm import LLMClient, MockLLMClient
from .nli import MockNLIDetector, NLIDetector
from .schema import ClaimDecision, GateResult, Rule
from .verbalize import verbalize_all


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class PipelineResult:
    rules: list[Rule]
    gate_results: list[GateResult]
    passed_rules: list[Rule]
    blocked_rules: list[GateResult]
    decisions: list[ClaimDecision]


def build_pipeline(
    policy_text: str,
    llm: LLMClient | None = None,
    detector: NLIDetector | None = None,
    pass_threshold: float = 0.45,
):
    """Return (llm, gate, policy_sentences). Replace defaults to go live:
        llm=OpenAIClient(), detector=DebertaNLIDetector()
    """
    llm = llm or MockLLMClient()
    detector = detector or MockNLIDetector(support_threshold=pass_threshold)
    policy_sentences = split_sentences(policy_text)
    gate = FaithfulnessGate(policy_sentences, detector, pass_threshold)
    return llm, gate


def run(policy_text: str, claims: list[dict], pass_threshold: float = 0.45) -> PipelineResult:
    llm, gate = build_pipeline(policy_text, pass_threshold=pass_threshold)

    rules = extract_rules(policy_text, llm)          # Stage 1
    rules = verbalize_all(rules)                      # Stage 2
    gate_results = gate.run(rules)                    # Stage 3

    passed_ids = {g.rule_id for g in gate_results if g.passed}
    passed_rules = [r for r in rules if r.rule_id in passed_ids]
    blocked_rules = [g for g in gate_results if not g.passed]

    decisions = adjudicate(claims, passed_rules)     # Stage 4

    return PipelineResult(rules, gate_results, passed_rules, blocked_rules, decisions)
