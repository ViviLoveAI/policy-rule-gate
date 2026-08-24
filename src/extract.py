"""Stage 1 -- Rule extraction.

Reads raw policy text and asks the LLM to emit structured rules. The prompt is
real and usable; with MockLLMClient it returns a fixed extraction so the demo
runs offline. Swap in OpenAIClient to use the live model.
"""

from __future__ import annotations

from .llm import LLMClient
from .schema import Coverage, Rule

_SYSTEM = (
    "You are a healthcare policy analyst. Convert coverage policy text into "
    "structured, executable rules. Only encode what the policy explicitly "
    "states. Do NOT infer thresholds, ages, or conditions that are not written "
    "in the text. Respond with a single JSON object."
)

_USER_TEMPLATE = (
    "Extract coverage rules from the policy below. Return JSON:\n"
    '{{"rules": [{{"rule_id","service","coverage" (covered|not_covered),'
    '"conditions":[...],"exclusions":[...],"source_hint"}}]}}\n\n'
    "POLICY:\n{policy}"
)


def extract_rules(policy_text: str, llm: LLMClient) -> list[Rule]:
    resp = llm.complete_json(_SYSTEM, _USER_TEMPLATE.format(policy=policy_text))
    rules: list[Rule] = []
    for r in resp.get("rules", []):
        rules.append(
            Rule(
                rule_id=r["rule_id"],
                service=r["service"],
                coverage=Coverage(r["coverage"]),
                conditions=list(r.get("conditions", [])),
                exclusions=list(r.get("exclusions", [])),
                source_hint=r.get("source_hint", ""),
            )
        )
    return rules
