"""Hand-built Okapi BM25 over ``Chunk.index_text``: the baseline retriever for P1.

IDF is the Lucene form ``ln(1 + (N - df + 0.5) / (df + 0.5))``, always positive.
Query terms are the *set* of distinct query tokens (standard BM25: a repeated query
term is not weighted twice). Terms outside the vocabulary contribute nothing, so a
query with no known terms ranks nothing and every returned score is > 0.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from specrag.models import Chunk
from specrag.retrieval.ranking import sort_ranking

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase; keep identifiers (code_verifier, client-id) whole. No stemming or stopwords."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Index:
    k1: float
    b: float
    n_docs: int
    avgdl: float
    doc_len: dict[str, int]
    postings: dict[str, list[tuple[str, int]]]  # term -> [(chunk_id, tf)]

    @classmethod
    def build(cls, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        doc_len: dict[str, int] = {}
        postings: dict[str, list[tuple[str, int]]] = {}
        for chunk in chunks:
            terms = tokenize(chunk.index_text)
            doc_len[chunk.id] = len(terms)
            for term, tf in Counter(terms).items():
                postings.setdefault(term, []).append((chunk.id, tf))
        avgdl = sum(doc_len.values()) / len(doc_len) if doc_len else 0.0
        return cls(k1, b, len(doc_len), avgdl, doc_len, postings)

    def rank(self, query: str, n: int = 1000) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tf in posting:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[chunk_id] / self.avgdl)
                gain = idf * tf * (self.k1 + 1) / (tf + norm)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + gain
        return sort_ranking({k: v for k, v in scores.items() if v > 0})[:n]
