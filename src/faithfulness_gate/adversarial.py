"""Conditional adversarial adjudication layer.

The interface mirrors the user's prior long-document faithfulness detector:
a prosecutor raises unsupported-claim accusations and an independent judge
rules on those accusations using only the source evidence.
"""

from __future__ import annotations

import json
import os
import re

from .schema import AdversarialResult, ClaimVerdict


class AdversarialAdjudicator:
    """Dual-agent adjudicator with an offline fallback."""

    def __init__(
        self,
        prosecutor_model: str = "gpt-5-mini",
        judge_model: str = "gpt-5-mini",
        enabled: bool | None = None,
    ):
        self.prosecutor_model = prosecutor_model
        self.judge_model = judge_model
        self.enabled = bool(os.getenv("OPENAI_API_KEY")) if enabled is None else enabled
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def _response_text(self, model: str, prompt: str) -> str:
        resp = self.client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )
        return resp.output_text.strip()

    def _parse_json(self, raw: str) -> dict:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        try:
            return json.loads(match.group(0) if match else cleaned)
        except Exception:
            return {"verdict": "REVIEW", "reason": f"parse error: {raw[:120]}"}

    def run(self, claim: str, evidence: str, policy_text: str, reason: str) -> AdversarialResult:
        if not self.enabled:
            return AdversarialResult(
                verdict=ClaimVerdict.REVIEW,
                confidence=0.5,
                prosecutor_report="Adversarial layer not run because OPENAI_API_KEY is not set.",
                judge_reasoning="Offline fallback routes uncertain/high-risk claims to human review.",
            )

        prosecutor_prompt = f"""You are a prosecutor checking a generated healthcare policy-rule claim.

SOURCE EVIDENCE:
{evidence}

FULL POLICY CONTEXT:
{policy_text}

CLAIM:
{claim}

Trigger reason from earlier gate layers:
{reason}

Accuse the claim if it is unsupported, overgeneralized, contradicted, missing a condition, or introduces a requirement absent from the source. If it is supported, say "No accusations."
"""
        report = self._response_text(self.prosecutor_model, prosecutor_prompt)

        judge_prompt = f"""You are a neutral judge. Use ONLY the source evidence and policy context to rule on the prosecutor's accusation.

SOURCE EVIDENCE:
{evidence}

FULL POLICY CONTEXT:
{policy_text}

CLAIM:
{claim}

PROSECUTOR REPORT:
{report}

Return JSON only:
{{"verdict": "PASS" | "REVIEW" | "FAIL", "confidence": 0.0-1.0, "reason": "one sentence"}}

Use PASS only if the claim is clearly supported. Use FAIL if an unsupported or contradicted claim is upheld. Use REVIEW if ambiguity remains.
"""
        raw = self._response_text(self.judge_model, judge_prompt)
        parsed = self._parse_json(raw)
        verdict_text = str(parsed.get("verdict", "REVIEW")).upper()
        verdict = ClaimVerdict.__members__.get(verdict_text, ClaimVerdict.REVIEW)
        return AdversarialResult(
            verdict=verdict,
            confidence=float(parsed.get("confidence", 0.5)),
            prosecutor_report=report,
            judge_reasoning=str(parsed.get("reason", "")),
            raw=raw,
        )
