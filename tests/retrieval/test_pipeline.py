import time
from pathlib import Path

import numpy as np
import pytest

from specrag.models import Doc, DocStatus, Section
from specrag.retrieval import pipeline, store
from specrag.retrieval.chunkers import CHUNKER_VERSION
from specrag.retrieval.pipeline import RetrievalConfig, Retriever
from specrag.tokens import count_tokens

WORDS = "oauth token grant redirect client secret scope bearer refresh audience".split()


class FakeEmbedder:
    name = "fake"
    revision = "r1"
    max_seq_tokens = 512

    def embed(self, texts: list[str], *, query: bool) -> np.ndarray:
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, sum(map(ord, w)) % 32] += 1.0
        return out + 1e-3

    def truncated(self, texts: list[str]) -> int:
        return 0


class FakeReranker:
    revision = "rr1"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return [float(len(t) % 7) for t in texts]


def _sections() -> list[Section]:
    out = []
    for d in range(3):
        for s in range(1, 9):
            body = " ".join(WORDS[(d + s + j) % len(WORDS)] for j in range(300))
            out.append(
                Section(
                    id=f"rfc{d}#{s}",
                    doc_id=f"rfc{d}",
                    number=str(s),
                    title=f"T{s}",
                    parent_id=None,
                    depth=1,
                    own_text=body,
                    own_tokens=count_tokens(body),
                )
            )
    return out


@pytest.fixture
def fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    secs = _sections()
    (cache / "sections.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in secs))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("docs: []\n")
    monkeypatch.setenv("SPECRAG_CACHE", str(cache))
    monkeypatch.setattr(store, "DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr(pipeline, "read_sections", lambda: secs)
    docs = {
        f"rfc{d}": Doc(doc_id=f"rfc{d}", title=f"Doc {d}", status=DocStatus.BCP, url="https://x")
        for d in range(3)
    }
    monkeypatch.setattr(pipeline, "_load_docs", lambda: docs)
    return cache


def test_retrieve_result_shape(fake_corpus: Path) -> None:
    cfg = RetrievalConfig()
    r = Retriever.open(cfg)
    res = r.retrieve("oauth refresh token")
    assert len(res.ranking) > len(res.packed.chunks)
    assert res.packed.tokens_used == 2500
    assert [c.id for c in res.packed.chunks] == [cid for cid, _ in res.ranking][
        : len(res.packed.chunks)
    ]
    p = res.provenance
    for k in ("corpus_key", "chunker_version", "config_hash", "dense_model", "dense_revision"):
        assert isinstance(p[k], str)
    assert p["chunker_version"] == CHUNKER_VERSION
    assert p["corpus_key"] == store.corpus_key()
    assert p["dense_model"] == "n/a" and p["dense_revision"] == "n/a"
    assert "dense_truncated" not in p and "rerank_revision" not in p
    assert set(r.chunks) >= {cid for cid, _ in res.ranking}
    assert (store.index_dir("section") / "chunks.sqlite").exists()


def test_config_hash_stable_and_sensitive() -> None:
    base = RetrievalConfig()
    assert base.config_hash() == RetrievalConfig().config_hash()
    assert len(base.config_hash()) == 12
    variants = [
        RetrievalConfig(chunker="fixed"),
        RetrievalConfig(retriever="dense"),
        RetrievalConfig(rerank=True),
        RetrievalConfig(budget=2000),
        RetrievalConfig(depth=10),
    ]
    hashes = {v.config_hash() for v in variants} | {base.config_hash()}
    assert len(hashes) == 6


@pytest.mark.parametrize("retriever", ["dense", "hybrid"])
def test_dense_and_hybrid_provenance(fake_corpus: Path, retriever: str) -> None:
    emb = FakeEmbedder()
    r = Retriever.open(RetrievalConfig(retriever=retriever), embedder=emb)  # type: ignore[arg-type]
    res = r.retrieve("bearer scope")
    assert res.provenance["dense_truncated"] == "0"
    assert res.provenance["dense_model"] == "fake"
    assert res.provenance["dense_revision"] == "r1"
    assert res.packed.tokens_used == 2500
    assert list(store.index_dir("section").glob("emb-*.npy"))


def test_rerank_reorders_head(fake_corpus: Path) -> None:
    rr = FakeReranker()
    r = Retriever.open(RetrievalConfig(rerank=True), reranker=rr)
    res = r.retrieve("client secret")
    base = Retriever.open(RetrievalConfig()).retrieve("client secret")
    assert res.ranking[50:] == base.ranking[50:]  # tail keeps BM25 scores and order (M3)
    assert {c for c, _ in res.ranking[:50]} == {c for c, _ in base.ranking[:50]}
    assert res.provenance["rerank_revision"] == "rr1"
    assert rr.calls == 1
    assert res.provenance["rerank_depth"] == "50"
    assert "rerank_revision" in res.provenance
    head = [s for _, s in res.ranking[:50]]
    assert head == sorted(head, reverse=True)
    assert all(s == float(int(s)) for s in head)


def test_warm_store_reused(fake_corpus: Path) -> None:
    Retriever.open(RetrievalConfig())
    path = store.index_dir("section") / "chunks.sqlite"
    mtime = path.stat().st_mtime_ns
    Retriever.open(RetrievalConfig())
    assert path.stat().st_mtime_ns == mtime


def test_open_rebuilds_store_after_key_change(fake_corpus: Path) -> None:
    Retriever.open(RetrievalConfig())
    old = store.index_dir("section")
    with (fake_corpus / "sections.jsonl").open("a") as f:
        f.write("\n")  # same sections, new bytes -> new corpus key (M6)
    Retriever.open(RetrievalConfig())
    new = store.index_dir("section")
    assert new != old and (new / "chunks.sqlite").exists()


@pytest.mark.parametrize("q", ["", "   \n"])
def test_empty_query_rejected(fake_corpus: Path, q: str) -> None:
    r = Retriever.open(RetrievalConfig())
    with pytest.raises(ValueError, match="query must not be empty"):
        r.retrieve(q)


QUERIES = [
    "authorization code grant with PKCE",
    "refresh token rotation and revocation",
    "JWT signature validation algorithm",
    "redirect URI exact string matching",
    "access token audience restriction",
]


@pytest.mark.corpus
@pytest.mark.parametrize("chunker", ["fixed", "section", "section+meta"])
def test_every_chunker_fills_budget_bm25(chunker: str, capsys: pytest.CaptureFixture[str]) -> None:
    t0 = time.perf_counter()
    r = Retriever.open(RetrievalConfig(chunker=chunker))  # type: ignore[arg-type]
    t1 = time.perf_counter()
    for q in QUERIES:
        res = r.retrieve(q)
        assert 2497 <= res.packed.tokens_used <= 2500, (q, res.packed.tokens_used)
        assert res.packed.exhausted is False
    with capsys.disabled():
        print(f"\n[open {chunker}] {t1 - t0:.2f}s")
