import time

import pytest
import tiktoken

from specrag.corpus.build import read_sections
from specrag.models import Section
from specrag.retrieval.ranking import sort_ranking
from specrag.retrieval.streams import DocStream, _clean, build_streams, safe_cut
from specrag.tokens import count_tokens, decode, encode, encoder_bytes


def sec(id: str, text: str, kind: str = "content", doc: str = "d") -> Section:
    return Section(
        id=id,
        doc_id=doc,
        number=id.split("#")[1],
        title="T",
        parent_id=None,
        depth=1,
        own_text=text,
        own_tokens=count_tokens(text),
        kind=kind,  # type: ignore[arg-type]
    )


def fixture() -> list[Section]:
    return [
        sec("d#1", "alpha beta gamma delta epsilon zeta eta theta"),
        sec("d#2", "one two three four five six seven"),
        sec("d#3", "red green blue yellow orange purple"),
    ]


def test_stream_excludes_references_and_empty() -> None:
    secs = [
        sec("d#1", "some real content here"),
        sec("d#2", ""),
        sec("d#3", "reference list entry", kind="references"),
        sec("e#1", "other document text", doc="e"),
    ]
    streams = build_streams(secs)
    assert list(streams) == ["d", "e"]
    assert list(streams["d"].section_spans) == ["d#1"]
    assert len(streams["d"].ids) == len(streams["d"].owner)


def test_attribution_matches_own_tokens() -> None:
    secs = fixture()
    st = build_streams(secs)["d"]
    for s in secs:
        assert st.attribute(*st.section_spans[s.id]) == ([s.id], [s.own_tokens])


def test_attribute_across_boundary() -> None:
    secs = fixture()
    st = build_streams(secs)["d"]
    a_end = st.section_spans["d#1"][1]
    b_start = st.section_spans["d#2"][0]
    assert b_start > a_end  # separator between
    assert st.attribute(a_end - 3, b_start + 5) == (["d#1", "d#2"], [3, 5])
    assert st.owner[a_end] is None


def test_text_roundtrip() -> None:
    secs = fixture()
    st = build_streams(secs)["d"]
    assert st.text(*st.section_spans["d#2"]) == secs[1].own_text


@pytest.mark.corpus
def test_corpus_streams_cover_all_content_tokens() -> None:
    secs = read_sections()
    expected = sum(s.own_tokens for s in secs if s.kind == "content" and s.own_tokens > 0)
    streams = build_streams(secs)
    total = 0
    for st in streams.values():
        _, toks = st.attribute(0, len(st.ids))
        total += sum(toks)
    assert total == expected


def test_safe_cut_multibyte() -> None:
    text = "→é漢字🙂" * 6
    ids = encode(text)
    st = DocStream(
        doc_id="d", ids=ids, owner=["d#1"] * len(ids), section_spans={"d#1": (0, len(ids))}
    )
    for direction in ("forward", "back"):
        for pos in range(len(ids) + 1):
            p = safe_cut(st, pos, direction)  # type: ignore[arg-type]
            assert abs(p - pos) <= 3
            assert "�" not in decode(ids[:p])
            assert "�" not in decode(ids[p:])


def test_sort_ranking_ties() -> None:
    assert sort_ranking({"b": 1.0, "a": 1.0, "c": 2.0}) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


def _stream(text: str) -> DocStream:
    ids = encode(text)
    return DocStream("d", ids, ["d#1"] * len(ids), {"d#1": (0, len(ids))})


def test_safe_cut_with_real_replacement_char_in_stream() -> None:
    text = "a\ufffdb " + "→é漢字🙂" * 6 + " \ufffd end"
    st = _stream(text)
    for direction in ("forward", "back"):
        for pos in range(len(st.ids) + 1):
            p = safe_cut(st, pos, direction)  # type: ignore[arg-type]
            assert abs(p - pos) <= 3
            encoder_ok = True
            try:
                bytes_ = b"".join(encoder_bytes([i]) for i in st.ids[:p])
                bytes_.decode("utf-8")
            except UnicodeDecodeError:
                encoder_ok = False
            assert encoder_ok, (direction, pos, p)


def test_safe_cut_fallback_returns_pos_when_no_clean_cut() -> None:
    # Ten tokens that are each the single raw byte 0x80 (a lone UTF-8 continuation byte):
    # every cut within +-3 of the middle lands between continuation bytes, so none is clean.
    tok = tiktoken.get_encoding("cl100k_base").encode_single_token(b"\x80")
    assert encoder_bytes([tok]) == b"\x80"
    ids = [tok] * 10
    st = DocStream("d", ids, ["d#1"] * 10, {"d#1": (0, 10)})
    assert not _clean(st, 5)
    assert safe_cut(st, 5, "forward") == 5
    assert safe_cut(st, 5, "back") == 5


@pytest.mark.corpus
def test_safe_cut_timing_on_largest_doc() -> None:
    st = max(build_streams(read_sections()).values(), key=lambda d: len(d.ids))
    n = len(st.ids)
    t = time.perf_counter()
    for k in range(1000):
        safe_cut(st, (k * 7919) % n, "forward")
    assert time.perf_counter() - t < 0.2
