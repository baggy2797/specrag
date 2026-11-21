import statistics
from collections import Counter

import pytest

from specrag.corpus.build import read_sections
from specrag.corpus.manifest import load_manifest
from specrag.models import Doc, DocStatus, Section, SectionIndex
from specrag.retrieval.chunkers import (
    CHUNKER_VERSION,
    DOC_META,
    chunk_fixed,
    chunk_section,
    chunk_section_meta,
    heading_path,
    make_chunks,
    split_section,
)
from specrag.retrieval.streams import DocStream, build_streams
from specrag.tokens import count_tokens, decode, encode


def sec(id: str, text: str, title: str = "T", kind: str = "content") -> Section:
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
        kind=kind,  # type: ignore[arg-type]
    )


def flat_stream(n: int, doc: str = "d") -> DocStream:
    ids = encode(" word" * n)
    assert len(ids) == n
    return DocStream(doc, ids, ["d#1"] * n, {"d#1": (0, n)})


def paragraphs(n_paras: int, words: int) -> str:
    return "\n\n".join(" ".join(["alpha"] * words) + "." for _ in range(n_paras))


def test_fixed_window_and_overlap() -> None:
    chunks = chunk_fixed(flat_stream(1200))
    assert [c.span for c in chunks] == [(0, 512), (448, 960), (896, 1200)]
    assert [c.id for c in chunks] == ["fixed:d:0000", "fixed:d:0001", "fixed:d:0002"]
    assert all(c.index_text == c.text for c in chunks)


def test_fixed_short_and_exact_multiple() -> None:
    assert [c.span for c in chunk_fixed(flat_stream(300))] == [(0, 300)]
    assert [c.span for c in chunk_fixed(flat_stream(960))] == [(0, 512), (448, 960)]


def test_fixed_never_spans_docs() -> None:
    secs = [sec(f"a#{i}", paragraphs(5, 40)) for i in range(1, 6)]
    secs += [sec(f"b#{i}", paragraphs(5, 40)) for i in range(1, 6)]
    streams = build_streams(secs)
    for doc_id, st in streams.items():
        for c in chunk_fixed(st):
            assert c.doc_id == doc_id
            assert all(s.startswith(doc_id + "#") for s in c.sections)


def test_fixed_skips_separator_only_window() -> None:
    ids = encode(" word" * 448)
    sep = encode("\n\n")
    st = DocStream("d", ids + sep * 70, ["d#1"] * 448 + [None] * (70 * len(sep)), {"d#1": (0, 448)})
    chunks = chunk_fixed(st)
    assert len(st.ids) > 512 and [c.span for c in chunks] == [(0, 512)]


def _section_setup(text: str) -> tuple[DocStream, SectionIndex]:
    s = sec("d#1", text)
    return build_streams([s])["d"], SectionIndex([s])


def test_section_one_section_per_chunk() -> None:
    secs = [sec("d#1", paragraphs(3, 30)), sec("d#2", paragraphs(30, 60)), sec("d#3", "")]
    st = build_streams(secs)["d"]
    chunks = chunk_section(st, SectionIndex(secs))
    assert all(len(c.sections) == 1 for c in chunks)
    assert all(c.index_text == c.text for c in chunks)
    assert {c.sections[0] for c in chunks} == {"d#1", "d#2"}


def test_section_split_balanced() -> None:
    text = paragraphs(19, 98)
    st, idx = _section_setup(text)
    a, b = st.section_spans["d#1"]
    assert 1850 <= b - a <= 1950
    chunks = chunk_section(st, idx)
    assert len(chunks) == 3
    assert [c.id for c in chunks] == [f"section:d#1~{i}" for i in range(3)]
    assert chunks[0].span[0] == a and chunks[-1].span[1] == b
    for x, y in zip(chunks, chunks[1:], strict=False):
        assert x.span[1] == y.span[0]
    for c in chunks:
        n = c.span[1] - c.span[0]
        assert n <= 800
        assert 0.75 * 633 <= n <= 1.25 * 633
        assert c.sections == ["d#1"]
    for c in chunks[:-1]:
        assert decode(st.ids[a : c.span[1]]).endswith("\n\n")


def test_section_split_sentence_fallback() -> None:
    text = " ".join("This is a sentence of words." for _ in range(243))
    st, idx = _section_setup(text)
    a, b = st.section_spans["d#1"]
    assert 1650 <= b - a <= 1750
    _, kinds = split_section(st, "d#1")
    assert kinds and all(k == "sentence" for k in kinds)
    chunks = chunk_section(st, idx)
    for c in chunks[:-1]:
        assert st.text(a, c.span[1]).endswith(".")
    assert all(c.span[1] - c.span[0] <= 800 for c in chunks)


def test_section_split_hard_cut() -> None:
    text = "".join(f"x{i}y" for i in range(850))
    st, idx = _section_setup(text)
    a, b = st.section_spans["d#1"]
    assert b - a > 1600
    _, kinds = split_section(st, "d#1")
    assert kinds and all(k == "hard" for k in kinds)
    chunks = chunk_section(st, idx)
    assert all(c.span[1] - c.span[0] <= 800 for c in chunks)
    assert chunks[0].span[0] == a and chunks[-1].span[1] == b
    assert all("�" not in c.text for c in chunks)


def test_section_split_hard_cut_multibyte() -> None:
    text = "é漢字🙂" * 700
    st, idx = _section_setup(text)
    chunks = chunk_section(st, idx)
    assert all(c.span[1] - c.span[0] <= 800 for c in chunks)
    assert all("�" not in c.text for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_heading_path() -> None:
    secs = [
        sec("rfc9700#2", "x", "Best Practices"),
        sec("rfc9700#2.1", "y", "Protecting Redirect-Based Flows"),
        sec("rfc9700#2.1.1", "z", "Authorization Code Grant"),
    ]
    assert heading_path("rfc9700#2.1.1", SectionIndex(secs)) == (
        "2 Best Practices › 2.1 Protecting Redirect-Based Flows › 2.1.1 Authorization Code Grant"
    )


def test_section_meta_prefix_example() -> None:
    body = "The authorization code grant must use PKCE."
    secs = [
        sec("rfc9700#2", "", "Best Practices"),
        sec("rfc9700#2.1", "", "Protecting Redirect-Based Flows"),
        sec("rfc9700#2.1.1", body, "Authorization Code Grant"),
    ]
    doc = Doc(doc_id="rfc9700", title="x", status=DocStatus.BCP, url="https://x")
    st = build_streams(secs)["rfc9700"]
    (c,) = chunk_section_meta(st, SectionIndex(secs), doc)
    prefix = (
        "RFC 9700 · OAuth 2.0 Security BCP › 2 Best Practices"
        " › 2.1 Protecting Redirect-Based Flows › 2.1.1 Authorization Code Grant"
    )
    assert c.id == "meta:rfc9700#2.1.1~0"
    assert c.text == body
    assert c.index_text == prefix + "\n" + body
    assert c.prefix_tokens == count_tokens(prefix + "\n")
    assert c.n_tokens == c.prefix_tokens + (c.span[1] - c.span[0])


def test_doc_meta_examples() -> None:
    assert DOC_META["rfc9700"] == ("RFC 9700", "OAuth 2.0 Security BCP")
    assert DOC_META["oidc-core"] == ("OpenID Connect Core 1.0", "OpenID Connect Core")
    assert DOC_META["oauth-2.1"] == ("draft-ietf-oauth-v2-1-17", "OAuth 2.1 (draft)")
    assert CHUNKER_VERSION == "1"


def test_doc_meta_covers_manifest() -> None:
    assert set(DOC_META) == {e.doc_id for e in load_manifest()}


def test_make_chunks_deterministic() -> None:
    secs = [sec("d#1", paragraphs(30, 60)), sec("d#2", paragraphs(3, 20))]
    doc = Doc(doc_id="d", title="Doc", status=DocStatus.BCP, url="https://x")
    idx = SectionIndex(secs)
    for name in ("fixed", "section", "section+meta"):
        a = make_chunks(name, build_streams(secs), idx, {"d": doc})  # type: ignore[arg-type]
        b = make_chunks(name, build_streams(secs), idx, {"d": doc})  # type: ignore[arg-type]
        assert [c.id for c in a] == [c.id for c in b] and a
        assert len({c.id for c in a}) == len(a)


# ---- corpus tests ----


@pytest.fixture(scope="module")
def corpus() -> dict[str, object]:
    secs = read_sections()
    idx = SectionIndex(secs)
    streams = build_streams(secs)
    docs: dict[str, Doc] = {e.doc_id: e for e in load_manifest()}
    out: dict[str, object] = {"secs": secs, "idx": idx, "streams": streams}
    for name in ("fixed", "section", "section+meta"):
        out[name] = make_chunks(name, streams, idx, docs)  # type: ignore[arg-type]
    return out


@pytest.mark.corpus
def test_no_replacement_chars_in_corpus_chunks(corpus: dict[str, object]) -> None:
    for name in ("fixed", "section", "section+meta"):
        for c in corpus[name]:  # type: ignore[attr-defined]
            assert "�" not in c.index_text, c.id


@pytest.mark.corpus
def test_section_chunks_cover_corpus(corpus: dict[str, object]) -> None:
    secs: list[Section] = corpus["secs"]  # type: ignore[assignment]
    total = sum(s.own_tokens for s in secs if s.kind == "content")
    assert sum(sum(c.tokens) for c in corpus["section"]) == total  # type: ignore[attr-defined]


@pytest.mark.corpus
def test_all_section_pieces_within_800(corpus: dict[str, object]) -> None:
    for name in ("section", "section+meta"):
        chunks = corpus[name]
        for c in chunks:  # type: ignore[attr-defined]
            assert c.span[1] - c.span[0] <= 800, c.id
    ids = {c.sections[0] for c in corpus["section"]}  # type: ignore[attr-defined]
    assert "rfc7515#A.4.1" in ids


@pytest.mark.corpus
def test_no_references_chunked(corpus: dict[str, object]) -> None:
    secs: list[Section] = corpus["secs"]  # type: ignore[assignment]
    refs = {s.id for s in secs if s.kind == "references"}
    for name in ("fixed", "section", "section+meta"):
        for c in corpus[name]:  # type: ignore[attr-defined]
            assert not refs & set(c.sections), c.id


@pytest.mark.corpus
def test_piece_size_report(corpus: dict[str, object]) -> None:
    streams: dict[str, DocStream] = corpus["streams"]  # type: ignore[assignment]
    sizes = [c.span[1] - c.span[0] for c in corpus["section"]]  # type: ignore[attr-defined]
    kinds: Counter[str] = Counter()
    split_sizes: list[int] = []
    for st in streams.values():
        for sid, (a, b) in st.section_spans.items():
            if b - a > 800:
                pieces, ks = split_section(st, sid)
                kinds.update(ks)
                split_sizes += [y - x for x, y in pieces]
    n_cuts = sum(kinds.values())
    print("\n== piece-size report ==")
    for name in ("fixed", "section", "section+meta"):
        print(f"chunks[{name}] = {len(corpus[name])}")  # type: ignore[arg-type]
    print(
        f"section pieces: n={len(sizes)} min={min(sizes)} "
        f"median={statistics.median(sizes)} max={max(sizes)}"
    )
    print(
        f"split-section pieces: n={len(split_sizes)} min={min(split_sizes)} "
        f"median={statistics.median(split_sizes)} max={max(split_sizes)}"
    )
    for k in ("paragraph", "sentence", "hard"):
        print(f"cuts[{k}] = {kinds[k]} ({kinds[k] / n_cuts:.1%})")
    print(f"cuts[total] = {n_cuts}")
    assert max(sizes) <= 800
