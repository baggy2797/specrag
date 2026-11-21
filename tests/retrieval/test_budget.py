import pytest

from specrag.corpus.build import read_sections
from specrag.corpus.manifest import load_manifest
from specrag.models import Chunk, Doc, DocStatus, Section, SectionIndex
from specrag.retrieval.budget import BUDGET, Packed, pack
from specrag.retrieval.chunkers import (
    chunk_fixed,
    chunk_section,
    chunk_section_meta,
    make_chunks,
)
from specrag.retrieval.streams import DocStream, build_streams
from specrag.tokens import count_tokens


def sec(id: str, n: int, title: str = "T") -> Section:
    text = " word" * n
    doc, number = id.split("#")
    return Section(
        id=id,
        doc_id=doc,
        number=number,
        title=title,
        parent_id=None,
        depth=number.count(".") + 1,
        own_text=text,
        own_tokens=count_tokens(text),
        kind="content",
    )


def setup(
    sizes: dict[str, int], max_tokens: int = 800
) -> tuple[dict[str, DocStream], SectionIndex, dict[str, list[Chunk]]]:
    """Streams plus section chunks per section id (one doc per section when ids differ)."""
    secs = [sec(sid, n) for sid, n in sizes.items()]
    streams = build_streams(secs)
    idx = SectionIndex(secs)
    chunks = {sid: chunk_section(streams[sid.split("#")[0]], idx, max_tokens) for sid in sizes}
    return streams, idx, chunks


def test_pack_exact_budget() -> None:
    streams, _, ch = setup({"a#1": 1000, "b#1": 1000, "c#1": 1000}, max_tokens=1000)
    ranked = [ch["a#1"][0], ch["b#1"][0], ch["c#1"][0]]
    p = pack(ranked, streams)
    assert BUDGET == 2500
    assert p.tokens_used == 2500
    assert p.costs == [1000, 1000, 500]
    assert p.chunks[-1].span[1] - p.chunks[-1].span[0] == 500
    assert not p.exhausted


def test_pack_truncation_recomputes_sections() -> None:
    streams, _, _ = setup({"d#1": 100, "d#2": 100})
    big = chunk_fixed(streams["d"])[0]
    assert big.sections == ["d#1", "d#2"]
    p = pack([big], streams, budget=50)
    (c,) = p.chunks
    assert c.sections == ["d#1"]
    assert c.tokens == [50]
    assert c.span == (0, 50)
    assert c.text == streams["d"].text(0, 50)
    assert c.n_tokens == 50
    assert p.tokens_used == 50
    assert p.costs == [50]


def test_pack_fixed_overlap_is_free() -> None:
    streams, _, _ = setup({"d#1": 1000})
    a, b, _ = chunk_fixed(streams["d"])
    assert a.span == (0, 512) and b.span == (448, 960)
    p = pack([a, b], streams)
    assert p.costs == [512, 448]
    assert p.tokens_used == 960
    assert b.n_tokens == 512  # n_tokens is the chunk size, not the cost


def test_pack_duplicate_overlap_skipped() -> None:
    streams, _, _ = setup({"d#1": 1000})
    a, _, _ = chunk_fixed(streams["d"])
    p = pack([a, a], streams)
    assert p.chunks == [a]
    assert p.costs == [512]


def test_pack_meta_prefix_counts() -> None:
    secs = [sec("m#1", 100, title="Heading")]
    streams = build_streams(secs)
    doc = Doc(doc_id="m", title="Doc", status=DocStatus.BCP, url="https://x")
    (c,) = chunk_section_meta(streams["m"], SectionIndex(secs), doc)
    assert c.prefix_tokens > 0
    p = pack([c], streams)
    assert p.costs == [c.prefix_tokens + 100]
    assert p.tokens_used == c.prefix_tokens + 100


def test_pack_truncated_meta_keeps_prefix_and_id() -> None:
    secs = [sec("m#1", 100, title="Heading")]
    streams = build_streams(secs)
    doc = Doc(doc_id="m", title="Doc", status=DocStatus.BCP, url="https://x")
    (c,) = chunk_section_meta(streams["m"], SectionIndex(secs), doc)
    prefix = c.index_text.split("\n")[0]
    p = pack([c], streams, budget=c.prefix_tokens + 40)
    (t,) = p.chunks
    assert t.id == c.id
    assert t.prefix_tokens == c.prefix_tokens
    assert t.span == (0, 40)
    assert t.text == streams["m"].text(0, 40)
    assert t.index_text == prefix + "\n" + t.text
    assert t.n_tokens == c.prefix_tokens + 40
    assert p.tokens_used == p.costs[0] == c.prefix_tokens + 40


def test_pack_skips_unfittable_prefix() -> None:
    secs = [
        sec("f#1", 90),
        sec("m#1", 100, title="A rather long heading title for this section"),
        sec("p#1", 100),
    ]
    streams = build_streams(secs)
    idx = SectionIndex(secs)
    doc = Doc(doc_id="m", title="Doc", status=DocStatus.BCP, url="https://x")
    filler = chunk_section(streams["f"], idx)[0]
    (meta,) = chunk_section_meta(streams["m"], idx, doc)
    (plain,) = chunk_section(streams["p"], idx)
    assert meta.prefix_tokens > 10
    p = pack([filler, meta, plain], streams, budget=100)
    assert [c.id for c in p.chunks] == [filler.id, plain.id]
    assert p.tokens_used == 100
    assert p.costs == [90, 10]


def emoji_setup() -> tuple[dict[str, DocStream], list[Chunk]]:
    """Doc a: 1 token. Doc e: 3000 tokens of 2-token emoji (odd cuts split a character).
    Doc c: 1 token. Returns streams and the three section chunks in that order."""
    secs = [sec("a#1", 1), sec("c#1", 1)]
    text = "\U0001f600" * 1500
    e = sec("e#1", 1).model_copy(update={"own_text": text, "own_tokens": count_tokens(text)})
    secs.insert(1, e)
    assert e.own_tokens == 3000
    streams = build_streams(secs)
    idx = SectionIndex(secs)
    chunks = [chunk_section(streams[d], idx, 3000)[0] for d in ("a", "e", "c")]
    return streams, chunks


def test_pack_safe_cut_pullback() -> None:
    streams, (a, e, _) = emoji_setup()
    p = pack([a, e], streams)
    assert 2497 <= p.tokens_used <= 2499
    assert p.tokens_used == 2499  # cut at 2499 splits an emoji, pulled back to 2498 body tokens
    assert p.exhausted is False
    assert p.chunks[-1].span == (0, 2498)
    assert sum(p.costs) == p.tokens_used
    assert "\ufffd" not in p.chunks[-1].text


def test_pack_at_most_one_truncated() -> None:
    # After the pull-back 1 token of budget is left, and chunk c (1 token) would fit.
    # Only the "stop after one truncation" rule keeps it out.
    streams, (a, e, c) = emoji_setup()
    p = pack([a, e, c], streams)
    assert [x.id for x in p.chunks] == [a.id, e.id]
    assert p.tokens_used == 2499
    assert p.exhausted is False


def test_pack_truncation_with_overlap() -> None:
    streams, _, _ = setup({"d#1": 1000})
    a, b, _ = chunk_fixed(streams["d"])
    assert a.span == (0, 512) and b.span == (448, 960)
    p = pack([a, b], streams, budget=612)
    assert p.chunks[1].span == (448, 612)
    assert p.costs == [512, 100]  # 164 positions, 64 already packed
    assert p.tokens_used == 612
    assert p.chunks[1].n_tokens == 164
    assert p.exhausted is False


def test_pack_duplicate_chunk_id_skipped() -> None:
    streams, _, ch = setup({"a#1": 100, "b#1": 100})
    a, b = ch["a#1"][0], ch["b#1"][0]
    p = pack([a, b, a, b], streams)
    assert [c.id for c in p.chunks] == [a.id, b.id]
    assert p.tokens_used == 200


def test_pack_mixed_docs_overlap_tracked_per_doc() -> None:
    streams, _, _ = setup({"x#1": 1000, "y#1": 1000})
    xa, xb, _ = chunk_fixed(streams["x"])
    ya, _, _ = chunk_fixed(streams["y"])
    p = pack([xa, ya, xb], streams)
    assert p.costs == [512, 512, 448]
    assert p.tokens_used == 1472


def test_pack_costs_sum() -> None:
    streams, _, _ = setup({"d#1": 3000})
    ranked = chunk_fixed(streams["d"])
    p = pack(ranked, streams)
    assert sum(p.costs) == p.tokens_used
    assert any(cost < c.n_tokens for c, cost in zip(p.chunks, p.costs, strict=True))


def test_pack_truncated_on_separator() -> None:
    streams, _, _ = setup({"d#1": 10, "d#2": 20})
    st = streams["d"]
    assert st.owner[10] is None  # separator token between the two sections
    sections, tokens = st.attribute(10, 31)
    c = Chunk(
        id="x",
        doc_id="d",
        text=st.text(10, 31),
        index_text=st.text(10, 31),
        sections=sections,
        tokens=tokens,
        n_tokens=21,
        span=(10, 31),
    )
    # room 1 would cut right after the separator: no section tokens, so the chunk is skipped
    p = pack([c], streams, budget=1)
    assert p.chunks == [] and p.tokens_used == 0


def test_pack_exhausted_ranking() -> None:
    streams, _, ch = setup({"a#1": 300, "b#1": 400})
    p = pack([ch["a#1"][0], ch["b#1"][0]], streams)
    assert p.tokens_used == 700
    assert p.exhausted is True


def test_pack_empty() -> None:
    assert pack([], {}) == Packed(chunks=[], costs=[], tokens_used=0, exhausted=True)


# ---- corpus test ----


@pytest.mark.corpus
@pytest.mark.parametrize("name", ["fixed", "section", "section+meta"])
def test_pack_corpus_fills_budget(name: str) -> None:
    secs = read_sections()
    streams = build_streams(secs)
    docs: dict[str, Doc] = {e.doc_id: e for e in load_manifest()}
    chunks = make_chunks(name, streams, SectionIndex(secs), docs)  # type: ignore[arg-type]
    p = pack(chunks, streams)
    assert 2497 <= p.tokens_used <= 2500
    assert p.exhausted is False
    assert sum(p.costs) == p.tokens_used


def test_pack_truncation_of_only_separators_skipped() -> None:
    streams, _, _ = setup({"d#1": 10, "d#2": 20})
    st = streams["d"]
    first = chunk_section(st, SectionIndex([sec("d#1", 10), sec("d#2", 20)]))[0]
    whole = chunk_fixed(st)[0]
    assert first.span == (0, 10) and whole.span[0] == 0 and whole.span[1] == 32
    assert st.owner[10] is None
    # 1 token of room: the only new token it could keep is the separator at 10
    p = pack([first, whole], streams, budget=11)
    assert [c.id for c in p.chunks] == [first.id]
    assert p.tokens_used == 10
