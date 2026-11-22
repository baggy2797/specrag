"""Optional cross-encoder reranking (bge-reranker-base, CPU). Never run in CI."""

from typing import Any, Protocol

from specrag.models import Chunk
from specrag.retrieval.models_pin import RERANK_MODEL, RERANK_REVISION, resolve_pinned

RERANK_DEPTH = 50
MAX_LENGTH = 512


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """The model loads lazily; ``model`` is injectable for tests."""

    def __init__(self, model: Any = None) -> None:
        self._model = model

    def _get(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                resolve_pinned(RERANK_MODEL, RERANK_REVISION), device="cpu", max_length=MAX_LENGTH
            )
        return self._model

    def warm_up(self) -> None:
        """Load the model now (downloads on first use)."""
        self._get()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        preds = self._get().predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in preds]


def rerank(
    query: str,
    ranked: list[tuple[Chunk, float]],
    reranker: Reranker,
    depth: int = RERANK_DEPTH,
) -> list[tuple[Chunk, float]]:
    """Reorder the top ``depth`` by reranker score (ties keep original order).

    The reranked head carries reranker scores; the tail keeps its original scores and order.
    """
    head, tail = ranked[:depth], ranked[depth:]
    if not head:
        return list(ranked)
    scores = reranker.score(query, [c.index_text for c, _ in head])
    scored = [(c, s) for (c, _), s in zip(head, scores, strict=True)]
    scored.sort(key=lambda cs: -cs[1])  # stable: ties keep original order
    return scored + list(tail)
