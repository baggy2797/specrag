import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from typer.testing import CliRunner

from specrag import cli
from specrag.cli import app
from specrag.models import Chunk
from specrag.retrieval.budget import Packed
from specrag.retrieval.pipeline import RetrievalConfig, RetrievalResult


def test_version() -> None:
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0 and "0.1.0" in r.output


def _chunk(cid: str, sections: list[str]) -> Chunk:
    return Chunk(
        id=cid,
        doc_id="rfc1",
        text="t",
        index_text="t",
        sections=sections,
        tokens=[1] * len(sections),
        n_tokens=len(sections),
        span=(0, len(sections)),
    )


def _result(cfg: RetrievalConfig, chunks: list[Chunk], costs: list[int]) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        config=cfg,
        ranking=[(c.id, 1.0) for c in chunks] + [("unpacked", 0.5)],
        packed=Packed(chunks=chunks, costs=costs, tokens_used=sum(costs), exhausted=False),
        provenance={"corpus_key": "k"},
    )


class StubRetriever:
    def __init__(self, cfg: RetrievalConfig, chunks: list[Chunk], costs: list[int]) -> None:
        self.cfg, self.chunks, self.costs = cfg, chunks, costs
        self.dense = None

    def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        return _result(self.cfg, self.chunks, self.costs)


@pytest.fixture(autouse=True)
def _no_preflight(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if "real_preflight" not in request.keywords and "corpus" not in request.keywords:
        monkeypatch.setattr(cli, "_preflight_corpus", lambda: None)


class FakeModel:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc, self.warmed = exc, False

    def warm_up(self) -> None:
        if self.exc is not None:
            raise self.exc
        self.warmed = True


def _patch(
    monkeypatch: pytest.MonkeyPatch, chunks: list[Chunk], costs: list[int]
) -> list[RetrievalConfig]:
    seen: list[RetrievalConfig] = []

    def factory(cfg: RetrievalConfig, embedder: Any = None, reranker: Any = None) -> Any:
        seen.append(cfg)
        return StubRetriever(cfg, chunks, costs)

    monkeypatch.setattr(cli, "_open_retriever", factory)
    return seen


def test_cli_search_empty_query_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [], [])
    r = CliRunner().invoke(app, ["search", "   "])
    assert r.exit_code == 2
    assert "query must not be empty" in r.output


def test_cli_search_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [], [])
    r = CliRunner().invoke(app, ["search", "zzz"])
    assert r.exit_code == 0
    assert "no results" in r.output


def test_cli_search_prints_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, [_chunk("a", ["s1", "s2"]), _chunk("b", ["s3"])], [30, 20])
    monkeypatch.setattr(cli, "_make_reranker", lambda: FakeModel())
    r = CliRunner().invoke(app, ["search", "x", "--budget", "100", "--rerank"])
    assert r.exit_code == 0
    lines = r.output.splitlines()
    assert "1. a  s1,s2  cost=30" in lines
    assert "2. b  s3  cost=20" in lines
    assert "tokens used: 50" in lines
    assert f"config: {seen[0].config_hash()}" in lines
    assert seen[0].budget == 100 and seen[0].rerank


def test_cli_search_json_roundtrips(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [_chunk("a", ["s1"])], [30])
    r = CliRunner().invoke(app, ["search", "x", "--json"])
    assert r.exit_code == 0
    res = RetrievalResult.model_validate_json(r.stdout)
    assert res.packed.costs == [30]
    assert res.ranking and len(res.ranking) > len(res.packed.chunks)
    assert "tokens used" not in r.stdout
    json.loads(r.stdout)


def test_cli_search_model_unavailable_exits_3(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [], [])
    monkeypatch.setattr(cli, "_make_embedder", lambda: FakeModel(OSError("no network")))
    r = CliRunner().invoke(app, ["search", "x", "--retriever", "dense"])
    assert r.exit_code == 3
    assert "model unavailable: BAAI/bge-small-en-v1.5@5c38ec7c" in r.output
    assert "loading BAAI/bge-small-en-v1.5" in r.output


def test_cli_search_reranker_warmup_failure_after_dense_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, [], [])
    dense = FakeModel()
    monkeypatch.setattr(cli, "_make_embedder", lambda: dense)
    monkeypatch.setattr(
        cli, "_make_reranker", lambda: FakeModel(LocalEntryNotFoundError("not cached"))
    )
    r = CliRunner().invoke(app, ["search", "x", "--retriever", "hybrid", "--rerank"])
    assert r.exit_code == 3
    assert dense.warmed
    assert "model unavailable: BAAI/bge-reranker-base@2cfc18c9" in r.output
    assert "model unavailable: BAAI/bge-small" not in r.output


def test_cli_search_bm25_store_error_is_not_a_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(cfg: RetrievalConfig, embedder: Any = None, reranker: Any = None) -> Any:
        raise PermissionError("store not writable")

    monkeypatch.setattr(cli, "_open_retriever", boom)
    r = CliRunner().invoke(app, ["search", "x"])
    assert r.exit_code != 0
    assert isinstance(r.exception, PermissionError)
    assert "model unavailable" not in r.output
    assert "loading" not in r.output


@pytest.mark.real_preflight
def test_cli_search_missing_corpus_exits_3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPECRAG_CACHE", str(tmp_path))  # no sections.jsonl
    r = CliRunner().invoke(app, ["search", "x"])
    assert r.exit_code == 3
    assert "corpus unavailable: not caused by this change" in r.output
    assert "specrag fetch" in r.output


def test_cli_index_all(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, [_chunk("a", ["s1"])], [1])
    r = CliRunner().invoke(app, ["index", "--chunker", "all"])
    assert r.exit_code == 0
    assert [c.chunker for c in seen] == ["fixed", "section", "section+meta"]
    assert all(c.retriever == "bm25" for c in seen)
    assert "section: 1 chunks" in r.output


def test_cli_index_dense_reports_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    class D:
        truncated_count = 4

    def factory(cfg: RetrievalConfig, embedder: Any = None, reranker: Any = None) -> Any:
        s = StubRetriever(cfg, [_chunk("a", ["s1"])], [1])
        s.dense = D()  # type: ignore[assignment]
        return s

    monkeypatch.setattr(cli, "_open_retriever", factory)
    r = CliRunner().invoke(app, ["index", "--chunker", "fixed", "--dense"])
    assert r.exit_code == 0
    assert "dense truncated: 4" in r.output


@pytest.mark.corpus
def test_search_bm25_section_end_to_end() -> None:
    r = CliRunner().invoke(app, ["search", "PKCE code_verifier", "--retriever", "bm25"])
    assert r.exit_code == 0, r.output
    assert "tokens used: 2500" in r.output
