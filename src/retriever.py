"""Evidence retriever: given a claim sentence, find the most relevant policy
sentence to serve as the NLI `source`.

This mirrors the retrieval stage of your existing detector (BM25 + DeBERTa NLI).
Here the retriever is a dependency-free lexical scorer so the demo runs offline;
the real rank-bm25 implementation drops into the same `retrieve` signature.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalRetriever:
    """Offline BM25-lite over the policy's sentences.

    Good enough to pick the right supporting sentence for the demo. Swap for
    rank-bm25 (see BM25Retriever below) when you go live.
    """

    def __init__(self, sentences: list[str]):
        self.sentences = sentences
        self.doc_tokens = [_tokenize(s) for s in sentences]
        self.df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] += 1
        self.n_docs = max(1, len(sentences))

    def _idf(self, term: str) -> float:
        return math.log(1 + self.n_docs / (1 + self.df.get(term, 0)))

    def retrieve(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        scored: list[tuple[str, float]] = []
        for sent, toks in zip(self.sentences, self.doc_tokens):
            tf = Counter(toks)
            score = sum(tf[t] * self._idf(t) for t in q_tokens)
            # length-normalize a little so long sentences don't dominate
            score = score / (1 + math.log(1 + len(toks)))
            scored.append((sent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class BM25Retriever:
    """Real retriever seam using rank-bm25. Same interface as LexicalRetriever.

    To activate:
        pip install rank-bm25
        from rank_bm25 import BM25Okapi
    """

    def __init__(self, sentences: list[str]):
        self.sentences = sentences
        # from rank_bm25 import BM25Okapi
        # self._bm25 = BM25Okapi([_tokenize(s) for s in sentences])

    def retrieve(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        raise NotImplementedError(
            "BM25Retriever is a seam. Install rank-bm25 and implement retrieve()."
        )
        # scores = self._bm25.get_scores(_tokenize(query))
        # ranked = sorted(zip(self.sentences, scores), key=lambda x: x[1], reverse=True)
        # return ranked[:top_k]
