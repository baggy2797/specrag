from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from specrag.corpus.manifest import raw_path
from specrag.corpus.parse import Dialect, ParseError, detect_dialect, parse_doc, validate_sections
from specrag.corpus.parse_v3 import parse_v3

FIXTURE = Path(__file__).parent / "fixtures" / "v3_min.html"


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_detect_v3(html: str) -> None:
    assert detect_dialect(html) == Dialect.RFC_V3


def test_detect_htmlized() -> None:
    html = (
        '<pre><span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>'
        ".  Intro</span></pre>"
    )
    assert detect_dialect(html) == Dialect.RFC_HTMLIZED


def test_detect_openid() -> None:
    html = '<a name="rfc.section.1"></a><h3>1.&nbsp;Intro</h3>'
    assert detect_dialect(html) == Dialect.OPENID_V2


def test_detect_unknown_raises() -> None:
    with pytest.raises(ParseError):
        detect_dialect("<html><body><p>nothing</p></body></html>")


def test_parse_ids(html: str) -> None:
    secs = parse_doc("rfcX", html)
    assert [s.id for s in secs] == [
        "rfcX#1",
        "rfcX#2",
        "rfcX#2.1",
        "rfcX#2.2",
        "rfcX#A",
        "rfcX#A.1",
    ]


def test_parse_fields(html: str) -> None:
    secs = {s.id: s for s in parse_v3("rfcX", BeautifulSoup(html, "lxml"))}
    assert secs["rfcX#2"].title == "Heading Only"
    assert secs["rfcX#A"].title == "Extras"
    assert secs["rfcX#A.1"].title == "Sub Extra"
    assert secs["rfcX#2.1"].depth == 2
    assert secs["rfcX#A"].depth == 1
    assert secs["rfcX#2.1"].parent_id == "rfcX#2"
    assert secs["rfcX#A.1"].parent_id == "rfcX#A"
    assert secs["rfcX#1"].parent_id is None
    assert secs["rfcX#1"].own_text == "Alpha beta gamma [X]."


def test_own_text_excludes_children(html: str) -> None:
    secs = {s.id: s for s in parse_doc("rfcX", html)}
    two = secs["rfcX#2"]
    assert "Delta" not in two.own_text
    assert two.own_tokens < 50
    assert "item one item two" in secs["rfcX#2.1"].own_text
    assert secs["rfcX#2.1"].own_tokens > 0


def test_validate_duplicate(html: str) -> None:
    secs = parse_doc("rfcX", html)
    with pytest.raises(ParseError, match=r"rfcX#2\.1"):
        validate_sections("rfcX", [*secs, secs[2]])


def test_validate_orphan(html: str) -> None:
    secs = [s for s in parse_doc("rfcX", html) if s.id != "rfcX#2"]
    with pytest.raises(ParseError, match=r"rfcX#2"):
        validate_sections("rfcX", secs)


def test_validate_empty() -> None:
    with pytest.raises(ParseError):
        validate_sections("rfcX", [])


@pytest.mark.corpus
@pytest.mark.parametrize(
    ("doc_id", "dialect"),
    [
        ("rfc9700", Dialect.RFC_V3),
        ("rfc6749", Dialect.RFC_HTMLIZED),
        ("oidc-core", Dialect.OPENID_V2),
        ("oauth-2.1", Dialect.RFC_V3),
    ],
)
def test_corpus_detection(doc_id: str, dialect: Dialect) -> None:
    assert detect_dialect(raw_path(doc_id).read_text(encoding="utf-8")) == dialect


@pytest.mark.corpus
@pytest.mark.parametrize("doc_id", ["rfc9700", "oauth-2.1"])
def test_corpus_parse_v3(doc_id: str) -> None:
    secs = parse_doc(doc_id, raw_path(doc_id).read_text(encoding="utf-8"))
    ids = [s.id for s in secs]
    assert ids[0] == f"{doc_id}#1"
    if doc_id == "rfc9700":
        assert "rfc9700#2.1.1" in ids
