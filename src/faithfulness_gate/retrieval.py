"""Source-span retrieval for the layered gate."""

from __future__ import annotations

import math
import re
from collections import Counter

from .schema import SourceSpan


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_policy_spans(text: str) -> list[str]:
    """Split policy text into evidence-sized spans.

    CMS pages often lose list numbering during scraping, so this splitter keeps
    semicolon-delimited criteria as separate candidate spans.
    """

    raw_parts = re.split(r"(?<=[.;:])\s+|\n+", text.strip())
    parts = [p.strip(" \t-") for p in raw_parts if len(p.strip()) > 20]
    return parts or [text.strip()]


class SourceRetriever:
    """Small BM25-like retriever with no external dependencies."""

    def __init__(self, policy_text: str):
        self.spans = split_policy_spans(policy_text)
        self.tokens = [tokenize(s) for s in self.spans]
        self.df = Counter()
        for toks in self.tokens:
            for tok in set(toks):
                self.df[tok] += 1
        self.n_docs = max(1, len(self.spans))

    def _idf(self, token: str) -> float:
        return math.log(1 + self.n_docs / (1 + self.df.get(token, 0)))

    def retrieve(self, claim: str, top_k: int = 3) -> list[SourceSpan]:
        q_tokens = tokenize(claim)
        scored: list[SourceSpan] = []
        for span, toks in zip(self.spans, self.tokens):
            tf = Counter(toks)
            score = sum(tf[t] * self._idf(t) for t in q_tokens)
            score = score / (1 + math.log(1 + len(toks)))
            scored.append(SourceSpan(span, round(score, 4)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
