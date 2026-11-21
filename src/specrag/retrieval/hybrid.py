"""Reciprocal rank fusion of several rankings (hybrid search)."""

from specrag.retrieval.ranking import sort_ranking


def rrf(
    rankings: list[list[tuple[str, float]]], k: int = 60, n: int = 1000
) -> list[tuple[str, float]]:
    """Fuse rankings: score = sum of 1/(k + rank), rank starting at 1; absent IDs add 0."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sort_ranking(scores)[:n]
