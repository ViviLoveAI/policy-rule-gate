"""LLM client abstraction.

The pipeline only depends on the `LLMClient` protocol, so you can swap the
offline mock for the real OpenAI call without touching any other stage.

To go live:
    1. pip install openai
    2. set OPENAI_API_KEY
    3. use OpenAIClient() instead of MockLLMClient() in pipeline.py
"""

from __future__ import annotations

import json
import os
from typing import Protocol


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict:
        """Return the model's response parsed as a JSON object."""
        ...


class MockLLMClient:
    """Offline stand-in for extraction.

    It does NOT call any model. It returns a fixed, hand-authored structured
    extraction for the bundled sample policy so the whole pipeline runs with
    zero dependencies. One rule (R4) is a deliberately over-reaching
    ("soft-fabricated") rule that is NOT grounded in the policy text -- it
    exists so the faithfulness gate has something to catch.
    """

    def complete_json(self, system: str, user: str) -> dict:
        return {
            "rules": [
                {
                    "rule_id": "R1",
                    "service": "continuous glucose monitor (CGM)",
                    "coverage": "covered",
                    "conditions": ["diagnosis:diabetes_mellitus", "insulin_treated:true"],
                    "exclusions": [],
                    "source_hint": "CGM covered for insulin-treated diabetes",
                },
                {
                    "rule_id": "R2",
                    "service": "continuous glucose monitor (CGM)",
                    "coverage": "not_covered",
                    "conditions": ["diagnosis:gestational_diabetes"],
                    "exclusions": [],
                    "source_hint": "gestational diabetes explicitly excluded",
                },
                {
                    "rule_id": "R3",
                    "service": "continuous glucose monitor (CGM)",
                    "coverage": "covered",
                    "conditions": ["prior_auth:approved"],
                    "exclusions": [],
                    "source_hint": "prior authorization required",
                },
                {
                    # Soft fabrication: policy never mentions an age cutoff.
                    # Grounded in nothing -> should be BLOCKED by the gate.
                    "rule_id": "R4",
                    "service": "continuous glucose monitor (CGM)",
                    "coverage": "covered",
                    "conditions": ["age_ge:65"],
                    "exclusions": [],
                    "source_hint": "(model-inferred age threshold)",
                },
            ]
        }


class OpenAIClient:
    """Real LLM client. Left as a thin, ready-to-fill seam.

    Uncomment the body and install `openai` to activate. Kept import-light so
    the mock path never requires the dependency.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        # from openai import OpenAI
        # self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def complete_json(self, system: str, user: str) -> dict:
        raise NotImplementedError(
            "OpenAIClient is a seam. Install `openai`, set OPENAI_API_KEY, "
            "and implement this method (respond_format=json_object)."
        )
        # resp = self._client.chat.completions.create(
        #     model=self.model,
        #     response_format={"type": "json_object"},
        #     messages=[
        #         {"role": "system", "content": system},
        #         {"role": "user", "content": user},
        #     ],
        # )
        # return json.loads(resp.choices[0].message.content)
