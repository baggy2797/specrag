"""Tests for the Plan 1 final-review fix wave (references, STOP guard, dispatch, v3 detect)."""

import pytest
from bs4 import BeautifulSoup

from specrag.corpus import parse as parse_mod
from specrag.corpus.parse import Dialect, ParseError, detect_dialect, parse_doc
from specrag.corpus.parse_htmlized import parse_htmlized
from specrag.corpus.parse_openid import parse_openid
from specrag.corpus.parse_v3 import parse_v3
from specrag.models import Section

TOC = (
    '<table class="TOCbug"><tr><td class="TOCbug"><a href="#toc">&nbsp;TOC&nbsp;</a>'
    "</td></tr></table>"
)


def _openid(extra_before_authors: str) -> str:
    return (
        "<html><body><h3>Abstract</h3><p>Front.</p>"
        f'{TOC}<a name="rfc.section.1"></a><h3>1.&nbsp;Intro</h3><p>Alpha.</p>'
        f'{TOC}<a name="rfc.section.2"></a><h3>2.&nbsp;References</h3>'
        f"{TOC}<h3>2.1.&nbsp;Normative References</h3><table><tr><td>[RFC1] One.</td></tr></table>"
        f"{TOC}<h3>2.2.&nbsp;Informative References</h3>"
        "<table><tr><td>[RFC2] Two.</td></tr></table>"
        f'{TOC}<a name="rfc.section.A"></a><h3>Appendix A.&nbsp;Extras</h3><p>Extra.</p>'
        f"{extra_before_authors}"
        "<h3>Authors' Addresses</h3><p>Somebody Else</p></body></html>"
    )


def test_openid_reference_subsections_emitted_and_flagged() -> None:
    secs = {s.id: s for s in parse_openid("d", BeautifulSoup(_openid(""), "lxml"))}
    assert list(secs) == ["d#1", "d#2", "d#2.1", "d#2.2", "d#A"]
    assert secs["d#2.1"].title == "Normative References"
    assert secs["d#2.1"].parent_id == "d#2"
    assert secs["d#2.1"].number == "2.1"
    assert secs["d#2.1"].own_text == "[RFC1] One."
    assert secs["d#2.2"].own_text == "[RFC2] Two."
    assert {i for i, s in secs.items() if s.kind == "references"} == {"d#2", "d#2.1", "d#2.2"}
    assert secs["d#A"].own_text == "Extra."
    assert "Somebody" not in " ".join(s.own_text for s in secs.values())


def test_openid_stop_guard_raises_on_dropped_content() -> None:
    lost = "<h3>Mystery Heading</h3><p>" + "lost words here " * 30 + "</p>"
    html = _openid("").replace('<a name="rfc.section.A">', lost + '<a name="rfc.section.A">')
    with pytest.raises(ParseError, match=r"d.*Mystery Heading"):
        parse_openid("d", BeautifulSoup(html, "lxml"))


def test_openid_stop_guard_tolerates_short_drop() -> None:
    html = _openid("").replace(
        '<a name="rfc.section.A">', '<h3>Note</h3><p>tiny</p><a name="rfc.section.A">'
    )
    assert [s.id for s in parse_openid("d", BeautifulSoup(html, "lxml"))][-1] == "d#A"


V3 = (
    '<html><body><section id="section-1"><h2><a class="section-number">1. </a>'
    '<a class="section-name">Intro</a></h2><p>Hi.</p><pre>code block</pre></section>'
    '<section id="section-2"><h2><a class="section-number">2. </a>'
    '<a class="section-name">References</a></h2>'
    '<section id="section-2.1"><h3><a class="section-number">2.1. </a>'
    '<a class="section-name">Normative References</a></h3><p>[X] y.</p></section>'
    "</section></body></html>"
)


def test_v3_references_flagged_with_descendants() -> None:
    kinds = {s.id: s.kind for s in parse_v3("d", BeautifulSoup(V3, "lxml"))}
    assert kinds == {"d#1": "content", "d#2": "references", "d#2.1": "references"}


def test_v3_with_pre_still_detected_as_v3() -> None:
    assert detect_dialect(V3) == Dialect.RFC_V3


def test_data_id_alone_is_not_v3() -> None:
    html = (
        '<div data-id="section-1"></div><pre><span class="h2">'
        '<a class="selflink" id="section-1" href="#section-1">1</a>.  Intro</span></pre>'
    )
    assert detect_dialect(html) == Dialect.RFC_HTMLIZED
    assert detect_dialect('<section data-id="section-1">x</section><a name="rfc.section.1">') == (
        Dialect.OPENID_V2
    )


def test_htmlized_references_flagged() -> None:
    html = (
        '<pre><span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>'
        ".  Intro</span>\n   text\n"
        '<span class="h2"><a class="selflink" id="section-2" href="#section-2">2</a>'
        ".  References</span>\n"
        '<span class="h3"><a class="selflink" id="section-2.1" href="#section-2.1">2.1</a>'
        ".  Informative References</span>\n   [X] y\n</pre>"
    )
    kinds = {s.id: s.kind for s in parse_htmlized("d", BeautifulSoup(html, "lxml"))}
    assert kinds == {"d#1": "content", "d#2": "references", "d#2.1": "references"}


def test_section_kind_defaults_to_content() -> None:
    s = Section(
        id="d#1", doc_id="d", number="1", title="t", parent_id=None, depth=1,
        own_text="", own_tokens=0,
    )  # fmt: skip
    assert s.kind == "content"


@pytest.mark.parametrize("dialect", list(Dialect))
def test_every_dialect_dispatches(dialect: Dialect) -> None:
    assert callable(parse_mod._parser_for(dialect))


def test_unknown_dialect_raises() -> None:
    with pytest.raises(ParseError):
        parse_mod._parser_for("bogus")  # type: ignore[arg-type]


def test_parse_doc_flags_openid_refs() -> None:
    assert [s.kind for s in parse_doc("d", _openid(""))][1:4] == ["references"] * 3
