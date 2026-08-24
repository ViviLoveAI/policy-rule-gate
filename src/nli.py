"""NLI faithfulness detector.

Judges whether a claim sentence is entailed by (faithful to) a source sentence.
This is the exact role your PHANTOM detector plays -- the pipeline depends only
on the `NLIDetector` protocol, so your real BM25 + DeBERTa-v3 detector plugs in
here with no changes upstream.
"""

from __future__ import annotations

import re
from typing import Protocol

from .schema import Verdict


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_STOP = {"the", "a", "an", "is", "are", "for", "of", "to", "in", "and",
         "or", "with", "be", "by", "on", "if", "when", "that", "this"}

# Threshold / numeric specifier tokens. A claim that introduces one of these
# (or any digit) that is absent from the source is asserting a quantitative
# condition the policy never stated -- exactly the numeric-mismatch class a real
# NLI judge should catch. This mock keys on it explicitly.
_SPECIFIERS = {"aged", "age", "older", "younger", "years", "over", "under",
               "greater", "less", "least", "most", "minimum", "maximum"}


def _is_specifier(tok: str) -> bool:
    return tok in _SPECIFIERS or tok.isdigit()


class NLIDetector(Protocol):
    def judge(self, claim: str, source: str) -> tuple[Verdict, float]:
        """Return (verdict, confidence in 0..1) for `claim` given `source`."""
        ...


class MockNLIDetector:
    """Offline stand-in for the DeBERTa NLI judge.

    Uses content-token overlap between claim and retrieved source as a proxy for
    entailment. This is a lexical approximation ONLY -- it is here so the demo is
    deterministic and dependency-free. The real detector replaces it via the same
    `judge` signature. A claim whose key content tokens are absent from the source
    (e.g. an invented age threshold) scores low and is judged NOT_SUPPORTED.
    """

    def __init__(self, support_threshold: float = 0.45):
        self.support_threshold = support_threshold

    def judge(self, claim: str, source: str) -> tuple[Verdict, float]:
        c = _tokens(claim) - _STOP
        s = _tokens(source) - _STOP
        if not c:
            return Verdict.NOT_SUPPORTED, 0.0

        # Hard fail: any numeric/threshold specifier in the claim that the source
        # does not contain is an ungrounded quantitative assertion.
        ungrounded_specifiers = {t for t in c if _is_specifier(t)} - s
        if ungrounded_specifiers:
            return Verdict.NOT_SUPPORTED, 0.1

        overlap = len(c & s) / len(c)
        if overlap >= self.support_threshold:
            return Verdict.SUPPORTED, round(overlap, 2)
        return Verdict.NOT_SUPPORTED, round(overlap, 2)


class DebertaNLIDetector:
    """Real detector seam. Wrap your existing PHANTOM detector here.

    To activate:
        from your_phantom_pkg import FaithfulnessDetector
        self._detector = FaithfulnessDetector(...)
    and map its output to (Verdict, confidence).
    """

    def __init__(self):
        # self._detector = FaithfulnessDetector(model="deberta-v3-base-mnli")
        pass

    def judge(self, claim: str, source: str) -> tuple[Verdict, float]:
        raise NotImplementedError(
            "DebertaNLIDetector is a seam. Wire in your PHANTOM detector here."
        )
