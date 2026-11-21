import pytest

from specrag.models import Chunk
from specrag.retrieval.rerank import RERANK_DEPTH, CrossEncoderReranker, rerank


def _chunk(i: int, text: str | None = None) -> Chunk:
    return Chunk(
        id=f"c{i}",
        doc_id="d",
        text=f"body{i}",
        index_text=text if text is not None else f"index{i}",
        sections=["d#1"],
        tokens=[1],
        n_tokens=1,
        span=(i, i + 1),
    )


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [self.scores[t] for t in texts]


def test_rerank_depth_default() -> None:
    assert RERANK_DEPTH == 50


def test_rerank_only_reorders_top_depth() -> None:
    ranked = [(_chunk(i), 10.0 - i) for i in range(5)]
    fake = FakeReranker(
        {"index0": 0.1, "index1": 0.9, "index2": 0.5, "index3": 99.0, "index4": 99.0}
    )
    out = rerank("q", ranked, fake, depth=3)
    assert [c.id for c, _ in out] == ["c1", "c2", "c0", "c3", "c4"]
    assert [s for _, s in out] == [0.9, 0.5, 0.1, 10.0 - 3, 10.0 - 4]
    assert fake.calls == [("q", ["index0", "index1", "index2"])]  # reads index_text


def test_rerank_ties_keep_original_order() -> None:
    ranked = [(_chunk(i), 1.0) for i in range(4)]
    fake = FakeReranker({f"index{i}": 0.5 for i in range(4)})
    out = rerank("q", ranked, fake, depth=4)
    assert [c.id for c, _ in out] == ["c0", "c1", "c2", "c3"]


def test_rerank_depth_larger_than_ranked_and_empty() -> None:
    ranked = [(_chunk(i), 1.0) for i in range(2)]
    fake = FakeReranker({"index0": 0.1, "index1": 0.2})
    out = rerank("q", ranked, fake, depth=50)
    assert [c.id for c, _ in out] == ["c1", "c0"]
    assert rerank("q", [], fake) == []


class FakeCE:
    def __init__(self) -> None:
        self.pairs: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kw: object) -> list[float]:
        self.pairs.append(list(pairs))
        return [float(len(b)) for _, b in pairs]


def test_cross_encoder_injectable_model() -> None:
    ce = FakeCE()
    r = CrossEncoderReranker(model=ce)
    assert r.score("q", ["aa", "b"]) == [2.0, 1.0]
    assert ce.pairs == [[("q", "aa"), ("q", "b")]]
    assert r.score("q", []) == []


@pytest.mark.reranker
def test_cross_encoder_prefers_relevant() -> None:
    r = CrossEncoderReranker()
    scores = r.score(
        "what is PKCE",
        [
            "PKCE, Proof Key for Code Exchange, is an extension to the authorization code "
            "flow that prevents authorization code interception attacks.",
            "A JWS represents content secured with a digital signature or MAC using JSON "
            "data structures and base64url encoding.",
        ],
    )
    assert scores[0] > scores[1]
