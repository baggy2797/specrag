import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from specrag.corpus.manifest import raw_path
from specrag.corpus.parse import Dialect, detect_dialect, parse_doc
from specrag.corpus.parse_htmlized import parse_htmlized
from specrag.models import Section

FIXTURE = Path(__file__).parent / "fixtures" / "htmlized_min.html"

HTMLIZED_DOCS = [
    "rfc6749",
    "rfc6750",
    "rfc7009",
    "rfc7515",
    "rfc7516",
    "rfc7517",
    "rfc7518",
    "rfc7519",
    "rfc7636",
    "rfc7662",
    "rfc8252",
    "rfc8414",
    "rfc8628",
]


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def secs(html: str) -> dict[str, Section]:
    return {s.id: s for s in parse_htmlized("rfcX", BeautifulSoup(html, "lxml"))}


def test_dispatch(html: str) -> None:
    assert detect_dialect(html) == Dialect.RFC_HTMLIZED
    assert [s.id for s in parse_doc("rfcX", html)] == [
        "rfcX#1",
        "rfcX#2",
        "rfcX#2.1",
        "rfcX#A",
        "rfcX#A.1",
    ]


def test_ids_and_nesting(secs: dict[str, Section]) -> None:
    assert list(secs) == ["rfcX#1", "rfcX#2", "rfcX#2.1", "rfcX#A", "rfcX#A.1"]
    assert secs["rfcX#2"].title == "Main"
    assert secs["rfcX#A"].title == "Extras"
    assert secs["rfcX#A.1"].title == "Sub Extra"
    assert secs["rfcX#2.1"].depth == 2
    assert secs["rfcX#A"].depth == 1
    assert secs["rfcX#2.1"].parent_id == "rfcX#2"
    assert secs["rfcX#A.1"].parent_id == "rfcX#A"
    assert secs["rfcX#1"].parent_id is None
    assert secs["rfcX#1"].own_text == "Alpha beta gamma."
    assert secs["rfcX#2.1"].own_text == "Child text."
    assert secs["rfcX#2.1"].own_tokens > 0


def test_front_matter_and_trailer_excluded(secs: dict[str, Section]) -> None:
    joined = " ".join(s.own_text for s in secs.values())
    assert "Front matter" not in joined
    assert "Somebody Else" not in joined
    assert "Table of Contents" not in joined


def test_page_furniture_removed(secs: dict[str, Section]) -> None:
    for s in secs.values():
        assert "[Page" not in s.own_text
        assert "\f" not in s.own_text
        assert "Standards Track" not in s.own_text
        assert "October 2012" not in s.own_text
        assert "RFC 6749" not in s.own_text


def test_paragraph_joined_across_page_break(secs: dict[str, Section]) -> None:
    assert "split across a page break and then continues." in secs["rfcX#2"].own_text


def test_fallback_regex_path_without_grey_spans(html: str) -> None:
    plain = html.replace('<span class="grey">', "<span>")
    plain = plain.replace('<a href="./rfcX">RFC 6749</a>', "RFC 6749")
    s = {x.id: x for x in parse_htmlized("rfcX", BeautifulSoup(plain, "lxml"))}
    assert "[Page" not in s["rfcX#2"].own_text
    assert "October 2012" not in s["rfcX#2"].own_text
    assert "split across a page break and then continues." in s["rfcX#2"].own_text


def test_bare_final_footer_removed(secs: dict[str, Section]) -> None:
    assert secs["rfcX#A.1"].own_text == "Sub extra text."


def test_mid_document_trailer_line_not_truncated() -> None:
    html = (
        '<pre><span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>'
        ".  A</span>\n   before\nAuthor's Address\n   after\n"
        '<span class="h2"><a class="selflink" id="section-2" href="#section-2">2</a>'
        ".  B</span>\n   tail\nAuthor's Address\n   Someone\n</pre>"
    )
    s = {x.id: x for x in parse_htmlized("d", BeautifulSoup(html, "lxml"))}
    assert s["d#1"].own_text == "before Author's Address after"
    assert s["d#2"].own_text == "tail"


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _header_leak(doc_id: str) -> re.Pattern[str]:
    # Running header is "RFC <this doc's number> <short title> <Month YYYY>". A generic
    # "RFC \d{4} ..." pattern false-positives on reference entries citing other RFCs
    # (e.g. 'RFC 2119 Key Words", BCP 14, RFC 8174, DOI ... May 2017'), so pin the number.
    return re.compile(rf"RFC {doc_id[3:]} .{{0,80}}({_MONTHS}) \d{{4}}")


@pytest.mark.corpus
@pytest.mark.parametrize("doc_id", HTMLIZED_DOCS)
def test_corpus_parses(doc_id: str) -> None:
    out = parse_doc(doc_id, raw_path(doc_id).read_text(encoding="utf-8"))
    assert len(out) > 5
    for s in out:
        assert re.search(r"\[Page \d+\]", s.own_text) is None, s.id
        # own_text is multi-line now; search the collapsed form so a header split
        # across lines still counts as a leak.
        assert _header_leak(doc_id).search(" ".join(s.own_text.split())) is None, s.id
    assert not any("EMail:" in s.own_text or "Email:" in s.own_text for s in out)


@pytest.mark.corpus
def test_corpus_rfc6749_details() -> None:
    out = {s.id: s for s in parse_doc("rfc6749", raw_path("rfc6749").read_text(encoding="utf-8"))}
    assert out["rfc6749#4.1.2"].title == "Authorization Response"
    assert out["rfc6749#10.12"].title == "Cross-Site Request Forgery"
    assert "rfc6749#A.1" in out
    for s in out.values():
        assert "[Page" not in s.own_text
        # "October 2012" alone legitimately appears in reference entries; test the header.
        assert "OAuth 2.0 October 2012" not in s.own_text
        # Whitespace is collapsed, so the footer would read "Hardt Standards Track [Page N]".
        assert "Standards Track [Page" not in s.own_text
        assert "Hardt Standards Track" not in s.own_text
