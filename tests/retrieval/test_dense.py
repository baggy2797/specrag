import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from specrag.models import Chunk
from specrag.retrieval.dense import (
    QUERY_INSTRUCTION,
    DenseIndex,
    SentenceTransformerEmbedder,
    embedding_path,
)

DIM = 64


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors; counts embed calls."""

    def __init__(self, name: str = "fake", revision: str = "r1", max_seq_tokens: int = 512) -> None:
        self.name = name
        self.revision = revision
        self.max_seq_tokens = max_seq_tokens
        self.calls: list[tuple[list[str], bool]] = []

    def embed(self, texts: list[str], *, query: bool) -> np.ndarray:
        self.calls.append((list(texts), query))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
                out[i, h % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)

    def truncated(self, texts: list[str]) -> int:
        return sum(1 for t in texts if len(t.split()) > self.max_seq_tokens)


def mk(id: str, text: str, index_text: str | None = None) -> Chunk:
    n = len(text.split())
    return Chunk(
        id=id,
        doc_id="d",
        text=text,
        index_text=text if index_text is None else index_text,
        sections=["d#1"],
        tokens=[n],
        n_tokens=n,
        span=(0, n),
    )


CHUNKS = [
    mk("c1", "authorization code grant redirect"),
    mk("c2", "refresh token rotation"),
    mk("c3", "pkce code verifier challenge"),
]


def build(
    chunks: list[Chunk],
    emb: FakeEmbedder,
    cache: Path | None,
    chunker: str = "section",
    corpus_key: str = "ck",
) -> DenseIndex:
    return DenseIndex.build(
        chunks,
        emb,
        cache,
        chunker=chunker,  # type: ignore[arg-type]
        corpus_key=corpus_key,
    )


def test_dense_rank_with_fake_embedder() -> None:
    idx = build(CHUNKS, FakeEmbedder(), None)
    ranked = idx.rank("pkce code verifier")
    assert ranked[0][0] == "c3"
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)
    assert idx.rank("pkce", n=2).__len__() == 2


def test_dense_rank_ties_break_by_chunk_id() -> None:
    idx = build([mk("b", "same words"), mk("a", "same words")], FakeEmbedder(), None)
    assert [cid for cid, _ in idx.rank("same words")] == ["a", "b"]


def test_embeddings_are_float32_normalised() -> None:
    idx = build(CHUNKS, FakeEmbedder(), None)
    assert idx.matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(idx.matrix, axis=1), 1.0, atol=1e-5)


def test_no_cache_dir_writes_nothing(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    build(CHUNKS, emb, None)
    build(CHUNKS, emb, None)
    assert len(emb.calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_embedding_cache_hit_skips_embed(tmp_path: Path) -> None:
    first = FakeEmbedder()
    a = build(CHUNKS, first, tmp_path)
    assert len(first.calls) == 1
    second = FakeEmbedder()
    b = build(CHUNKS, second, tmp_path)
    assert second.calls == []
    np.testing.assert_array_equal(a.matrix, b.matrix)
    assert b.rank("refresh")[0][0] == "c2"
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".npy"]


def test_embedding_cache_invalidates_on_text_change(tmp_path: Path) -> None:
    build(CHUNKS, FakeEmbedder(), tmp_path)
    changed = [*CHUNKS[:2], mk("c3", "pkce code verifier challenge CHANGED")]
    emb = FakeEmbedder()
    idx = build(changed, emb, tmp_path)
    assert len(emb.calls) == 1
    # the rebuilt cache is now valid for the new text
    again = FakeEmbedder()
    build(changed, again, tmp_path)
    assert again.calls == []
    assert idx.rank("changed")[0][0] == "c3"


def test_embedding_cache_invalidates_on_id_change(tmp_path: Path) -> None:
    build(CHUNKS, FakeEmbedder(), tmp_path)
    renamed = [mk("x1", CHUNKS[0].text), *CHUNKS[1:]]
    emb = FakeEmbedder()
    build(renamed, emb, tmp_path)
    assert len(emb.calls) == 1


def test_embedding_cache_key_includes_chunker_version_model_revision(tmp_path: Path) -> None:
    base = embedding_path(tmp_path, "ck", "section", "m", "r")
    assert re.fullmatch(r"emb-[0-9a-f]{16}\.npy", base.name)
    expected = hashlib.sha256(b"ck|section|1|m|r").hexdigest()[:16]
    assert base.name == f"emb-{expected}.npy"
    assert embedding_path(tmp_path, "ck2", "section", "m", "r") != base
    assert embedding_path(tmp_path, "ck", "fixed", "m", "r") != base
    assert embedding_path(tmp_path, "ck", "section", "m2", "r") != base
    assert embedding_path(tmp_path, "ck", "section", "m", "r2") != base

    build(CHUNKS, FakeEmbedder(revision="r1"), tmp_path)
    emb = FakeEmbedder(revision="r2")
    build(CHUNKS, emb, tmp_path)
    assert len(emb.calls) == 1
    emb2 = FakeEmbedder(name="other")
    build(CHUNKS, emb2, tmp_path)
    assert len(emb2.calls) == 1
    emb3 = FakeEmbedder()
    build(CHUNKS, emb3, tmp_path, chunker="fixed")
    assert len(emb3.calls) == 1
    emb4 = FakeEmbedder()
    build(CHUNKS, emb4, tmp_path, corpus_key="other")
    assert len(emb4.calls) == 1


def test_cache_sidecar_holds_ids_and_text_hashes(tmp_path: Path) -> None:
    build(CHUNKS, FakeEmbedder(), tmp_path)
    (side,) = tmp_path.glob("emb-*.json")
    data = json.loads(side.read_text())
    assert data["ids"] == ["c1", "c2", "c3"]
    assert data["sha256"] == [hashlib.sha256(c.index_text.encode()).hexdigest() for c in CHUNKS]
    assert not list(tmp_path.glob("*.tmp*"))


def test_corrupt_cache_rebuilds(tmp_path: Path) -> None:
    build(CHUNKS, FakeEmbedder(), tmp_path)
    (npy,) = tmp_path.glob("emb-*.npy")
    npy.write_bytes(b"garbage")
    emb = FakeEmbedder()
    build(CHUNKS, emb, tmp_path)
    assert len(emb.calls) == 1


def test_embeds_index_text(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    build([mk("c1", "body", index_text="Heading prefix\nbody")], emb, None)
    assert emb.calls[0] == (["Heading prefix\nbody"], False)


class StubModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        self.max_seq_length = 0

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.inputs.append(list(texts))
        return np.array([[3.0, 4.0]] * len(texts), dtype=np.float64)


def test_query_instruction_only_on_queries() -> None:
    stub = StubModel()
    emb = SentenceTransformerEmbedder(model=stub)
    docs = emb.embed(["a passage"], query=False)
    qs = emb.embed(["a question"], query=True)
    assert stub.inputs == [["a passage"], [QUERY_INSTRUCTION + "a question"]]
    assert QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "
    assert docs.dtype == np.float32 and qs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(qs, axis=1), 1.0, atol=1e-6)
    assert stub.max_seq_length == 512


def test_dense_rank_embeds_query_as_query() -> None:
    emb = FakeEmbedder()
    idx = build(CHUNKS, emb, None)
    idx.rank("pkce")
    assert emb.calls[-1] == (["pkce"], True)


def test_truncated_count_reported() -> None:
    emb = FakeEmbedder(max_seq_tokens=10)
    chunks = [
        mk("c1", "short text"),
        mk("c2", " ".join(["word"] * 11)),
        mk("c3", " ".join(["word"] * 10)),
        mk("c4", " ".join(["word"] * 50)),
    ]
    idx = build(chunks, emb, None)
    assert idx.truncated_count == 2


def test_truncated_count_survives_cache_hit(tmp_path: Path) -> None:
    chunks = [mk("c1", "short"), mk("c2", " ".join(["word"] * 50))]
    build(chunks, FakeEmbedder(max_seq_tokens=10), tmp_path)
    emb = FakeEmbedder(max_seq_tokens=10)
    idx = build(chunks, emb, tmp_path)
    assert emb.calls == []
    assert idx.truncated_count == 1


def test_truncated_uses_model_tokenizer() -> None:
    class Tok:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
            return {"input_ids": [[0] * len(t) for t in texts]}

    stub = StubModel()
    stub.tokenizer = Tok()  # type: ignore[attr-defined]
    emb = SentenceTransformerEmbedder(model=stub)
    assert emb.truncated(["x" * 512, "x" * 513]) == 1


@pytest.mark.model
def test_real_model_normalised() -> None:
    emb = SentenceTransformerEmbedder()
    out = emb.embed(["OAuth uses bearer tokens.", "PKCE protects public clients."], query=False)
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    q = emb.embed(["what is pkce"], query=True)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    assert emb.truncated(["short", "word " * 2000]) == 1


def _stored(tmp_path: Path) -> tuple[Path, Path]:
    build(CHUNKS, FakeEmbedder(), tmp_path)
    (npy,) = tmp_path.glob("emb-*.npy")
    return npy, npy.with_suffix(".json")


@pytest.mark.parametrize("bad", ["rows", "nan", "inf", "zero_dim", "dim"])
def test_cache_with_bad_matrix_rebuilds(tmp_path: Path, bad: str) -> None:
    npy, _ = _stored(tmp_path)
    m = np.load(npy)
    if bad == "rows":
        m = m[:2]
    elif bad == "nan":
        m[1, 3] = np.nan
    elif bad == "inf":
        m[0, 0] = np.inf
    elif bad == "zero_dim":
        m = np.zeros((3, 0), dtype=np.float32)
    else:
        m = m[:, : DIM // 2]  # dim differs from what the embedder produces
    np.save(npy, m, allow_pickle=False)
    emb = FakeEmbedder()
    idx = build(CHUNKS, emb, tmp_path)
    assert len(emb.calls) == 1
    assert idx.matrix.shape == (3, DIM)
    assert np.isfinite(idx.matrix).all()
