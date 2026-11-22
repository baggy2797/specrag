"""Joins retrieval output with scoring on the real corpus (the seam Plan 5 depends on)."""

import pytest

from specrag import scoring
from specrag.corpus.build import read_sections
from specrag.models import SectionIndex
from specrag.retrieval.pipeline import RetrievalConfig, Retriever
from specrag.retrieval.streams import build_streams

pytestmark = pytest.mark.corpus

CHUNKERS = ["fixed", "section", "section+meta"]
QUERY = "refresh token rotation for public clients"


def _retriever(chunker: str) -> Retriever:
    return Retriever.open(RetrievalConfig(chunker=chunker, retriever="bm25"))  # type: ignore[arg-type]


def test_stream_spans_equal_own_tokens() -> None:
    secs = read_sections()
    idx = SectionIndex(secs)
    streams = build_streams(secs)
    spans = {sid: span for st in streams.values() for sid, span in st.section_spans.items()}
    assert spans
    for sid, (a, b) in spans.items():
        assert b - a == idx.own_tokens(sid), sid
    # every content section with tokens has a span; the rest (0 tokens) must not
    for s in secs:
        assert (s.id in spans) == (s.kind == "content" and s.own_tokens > 0), s.id


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_retrieve_fills_budget_and_costs_add_up(chunker: str) -> None:
    packed = _retriever(chunker).retrieve(QUERY).packed
    assert 2497 <= packed.tokens_used <= 2500  # safe_cut may pull the last chunk back
    assert sum(packed.costs) == packed.tokens_used
    assert packed.exhausted is False


def test_gold_section_is_a_recall_hit() -> None:
    idx = SectionIndex(read_sections())
    packed = _retriever("section").retrieve("code_verifier code_challenge S256").packed
    gold = {"rfc7636#4.2"}
    assert scoring.recall(packed.chunks, gold, idx) == 1.0
    assert any(scoring.hit_set(c, gold, idx) == gold for c in packed.chunks)


def test_heading_only_parent_is_hit_via_descendant() -> None:
    idx = SectionIndex(read_sections())
    assert idx.own_tokens("rfc9700#2.2") == 0
    r = _retriever("section")
    ranked = [r.chunks[cid] for cid, _ in r.retrieve("rfc9700 2.2.2").ranking]
    child = next(c for c in ranked if any(s.startswith("rfc9700#2.2.") for s in c.sections))
    assert scoring.chunk_hits(child, "rfc9700#2.2", idx)
    assert scoring.recall([child], {"rfc9700#2.2"}, idx) == 1.0


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_truncated_last_chunk_matches_stream_attribution(chunker: str) -> None:
    r = _retriever(chunker)
    packed = r.retrieve(QUERY).packed
    original_by_id = r.chunks
    truncated = [c for c in packed.chunks if c.span != original_by_id[c.id].span]
    assert len(truncated) <= 1
    assert truncated, "expected the budget to truncate the last chunk for this query"
    last = truncated[0]
    assert last is packed.chunks[-1]
    sections, tokens = r.streams[last.doc_id].attribute(*last.span)
    assert (last.sections, last.tokens) == (sections, tokens)
    assert last.n_tokens == last.prefix_tokens + (last.span[1] - last.span[0])
    assert last.span[0] == original_by_id[last.id].span[0]
    assert last.span[1] < original_by_id[last.id].span[1]
    # the packed chunks are consumable by scoring without error
    idx = SectionIndex(read_sections())
    for sid in last.sections:
        scoring.hit_set(last, {sid}, idx)
