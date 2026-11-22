from pathlib import Path

import pytest

from specrag.models import Chunk
from specrag.retrieval import store
from specrag.retrieval.store import ChunkStore, corpus_key, index_dir


def mk(i: int) -> Chunk:
    return Chunk(
        id=f"c{i}",
        doc_id="d",
        text=f"text {i}",
        index_text=f"text {i}",
        sections=["d#1"],
        tokens=[2],
        n_tokens=2,
        span=(i, i + 2),
    )


def test_store_roundtrip_preserves_order_and_span(tmp_path: Path) -> None:
    chunks = [mk(i) for i in (5, 1, 3)]
    s = ChunkStore(tmp_path / "chunks.sqlite")
    s.save(chunks)
    loaded = s.load()
    assert loaded == chunks
    assert isinstance(loaded[0].span, tuple)
    assert s.get_many(["c3", "c5"]) == [chunks[2], chunks[0]]
    assert not list(tmp_path.glob("*.tmp"))


def test_store_save_replaces(tmp_path: Path) -> None:
    s = ChunkStore(tmp_path / "chunks.sqlite")
    s.save([mk(1), mk(2)])
    s.save([mk(3)])
    assert s.load() == [mk(3)]


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("docs: []\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sections.jsonl").write_text("{}\n")
    monkeypatch.setenv("SPECRAG_CACHE", str(cache))
    monkeypatch.setattr(store, "DEFAULT_MANIFEST", manifest)
    return manifest


def test_index_key_changes_with_manifest_or_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _setup(tmp_path, monkeypatch)
    k0 = corpus_key()
    assert len(k0) == 12 and k0 == corpus_key()
    manifest.write_text("docs: [x]\n")
    k1 = corpus_key()
    assert k1 != k0
    (tmp_path / "cache" / "sections.jsonl").write_text("{}\n{}\n")
    k2 = corpus_key()
    assert k2 not in (k0, k1)
    assert index_dir("section").name == f"section-v1-{k2}"
    assert index_dir("section").parent == tmp_path / "cache" / "index"


def test_index_key_changes_with_chunker_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(tmp_path, monkeypatch)
    before = index_dir("fixed")
    monkeypatch.setattr(store, "CHUNKER_VERSION", "2")
    after = index_dir("fixed")
    assert before != after and after.name.startswith("fixed-v2-")


def test_corpus_key_missing_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    (tmp_path / "cache" / "sections.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="run `specrag parse` first"):
        corpus_key()
