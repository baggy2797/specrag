"""Paragraph breaks and preformatted layout in ``Section.own_text`` (all dialects)."""

import pytest
from bs4 import BeautifulSoup

from specrag.corpus.manifest import load_manifest, raw_path
from specrag.corpus.parse import parse_doc
from specrag.corpus.parse_htmlized import parse_htmlized
from specrag.corpus.parse_openid import parse_openid
from specrag.corpus.parse_v3 import parse_v3
from specrag.models import Section


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# --------------------------------------------------------------------------- v3

_V3 = """<html><body>
<section id="section-1">
<h2><a class="section-number">1. </a><a class="section-name">Intro</a></h2>
<p>First   paragraph
spans lines.<a class="pilcrow">&para;</a></p>
<p>Second&nbsp;paragraph.</p>
<ul><li><p>item one</p></li><li>item
two</li></ul>
<ol start="3" type="A"><li>lettered</li><li>next</li></ol>
<div class="artwork"><pre>
    +-----+     +-----+
    |  A  |---->|  B  |
    +-----+     +-----+
</pre></div>
<table><thead><tr><th>Name</th><th>Value</th></tr></thead>
<tbody><tr><td>alg</td><td>RS256  </td></tr></tbody></table>
</section>
</body></html>"""


@pytest.fixture
def v3() -> Section:
    return parse_v3("d", _soup(_V3))[0]


def test_v3_paragraphs(v3: Section) -> None:
    units = v3.own_text.split("\n\n")
    assert units[0] == "First paragraph spans lines."
    assert units[1] == "Second paragraph."


def test_v3_two_paragraphs_one_break() -> None:
    html = (
        '<section id="section-1"><h2><a class="section-number">1.</a>'
        '<a class="section-name">T</a></h2><p>One.</p>\n<p>Two.</p></section>'
    )
    assert parse_v3("d", _soup(html))[0].own_text == "One.\n\nTwo."


def test_v3_list_items(v3: Section) -> None:
    units = v3.own_text.split("\n\n")
    assert "- item one" in units
    assert "- item two" in units
    assert "C. lettered" in units
    assert "D. next" in units


def test_v3_figure_kept_and_dedented(v3: Section) -> None:
    assert "+-----+     +-----+\n|  A  |---->|  B  |\n+-----+     +-----+" in v3.own_text


def test_v3_table_rows(v3: Section) -> None:
    assert "Name | Value\nalg | RS256" in v3.own_text


# --------------------------------------------------------------------- htmlized

_HTMLIZED = """<html><body><pre>
<span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>.  Intro</span>

   The first paragraph is hard wrapped
   across three lines of text.  It has
   sentence spacing.

   Second paragraph.

     +----------+
     | Resource |
     |   Owner  |
     +----------+
          ^
         (B)

   o  bullet item that wraps
      onto a second line

     GET /authorize?response_type=code HTTP/1.1
     Host: server.example.com

<span class="h2"><a class="selflink" id="section-2" href="#section-2">2</a>.  Two</span>

   One.

   Two.
</pre></body></html>"""


@pytest.fixture
def hz() -> dict[str, Section]:
    return {s.id: s for s in parse_htmlized("d", _soup(_HTMLIZED))}


def test_htmlized_two_paragraphs(hz: dict[str, Section]) -> None:
    assert hz["d#2"].own_text == "One.\n\nTwo."


def test_htmlized_hard_wrap_joined(hz: dict[str, Section]) -> None:
    first = hz["d#1"].own_text.split("\n\n")[0]
    assert first == (
        "The first paragraph is hard wrapped across three lines of text. It has sentence spacing."
    )


def test_htmlized_figure_kept_dedented(hz: dict[str, Section]) -> None:
    fig = "+----------+\n| Resource |\n|   Owner  |\n+----------+\n     ^\n    (B)"
    assert fig in hz["d#1"].own_text.split("\n\n")


def test_htmlized_bullet_is_prose(hz: dict[str, Section]) -> None:
    assert "o bullet item that wraps onto a second line" in hz["d#1"].own_text.split("\n\n")


def test_htmlized_http_example_preformatted(hz: dict[str, Section]) -> None:
    units = hz["d#1"].own_text.split("\n\n")
    assert "GET /authorize?response_type=code HTTP/1.1\nHost: server.example.com" in units


def test_htmlized_leading_figure_keeps_relative_indent() -> None:
    html = (
        '<pre><span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>'
        ".  A</span>\n\n"
        "                    +------+\n"
        "      +----+        | Srv  |\n"
        "      | C  |------->|      |\n"
        "      +----+        +------+\n\n   Prose after.\n</pre>"
    )
    text = parse_htmlized("d", _soup(html))[0].own_text
    assert text.split("\n")[:2] == ["              +------+", "+----+        | Srv  |"]
    assert text.endswith("\n\nProse after.")


def _paged(before: str, after: str) -> str:
    return (
        '<pre><span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>'
        f".  A</span>\n\n   {before}\n\n\n"
        '<span class="grey">Doe                Standards Track                [Page 3]</span></pre>'
        '<pre class="newpage"><span class="grey">RFC 9999         Title         May 2020</span>'
        f"\n\n   {after}\n</pre>"
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("as described in", "(see Section 3)."),
        ("a value that is", "lowercase continues."),
        ("split by a hyphen-", "ated word."),
        ("[1, 2, 3,", "4, 5]"),
    ],
)
def test_htmlized_page_break_joins_mid_sentence(before: str, after: str) -> None:
    text = parse_htmlized("d", _soup(_paged(before, after)))[0].own_text
    assert text == f"{before} {after}"


def test_htmlized_page_break_after_caption_splits() -> None:
    html = _paged("                 Figure 3: Authorization Code Flow", "The flow.")
    text = parse_htmlized("d", _soup(html))[0].own_text
    assert text == "Figure 3: Authorization Code Flow\n\nThe flow."


@pytest.mark.parametrize(
    "before,after",
    [
        ("the request includes the", "Authorization header."),  # uppercase: split
        ('"exp":1300819380}', "The following is prose."),  # rfc7515#A.1.1
        ("callback=package.myCallback", "Successful response:"),  # rfc7009#2.3
        ("o  Specification Document(s): Section 4.1.1 of RFC 7515", "o  Header Parameter Name:"),
        ("1.1. Notational Conventions .................................4", "1.2. Terminology"),
    ],
)
def test_htmlized_page_break_stays_split(before: str, after: str) -> None:
    text = parse_htmlized("d", _soup(_paged(before, after)))[0].own_text
    units = text.split("\n\n")
    assert len(units) == 2, text


@pytest.mark.parametrize("before", ["Ends here.", "Ends with colon:", "Quoted end.)"])
def test_htmlized_page_break_after_terminal_splits(before: str) -> None:
    text = parse_htmlized("d", _soup(_paged(before, "Next paragraph.")))[0].own_text
    assert text == f"{before}\n\nNext paragraph."


def test_htmlized_indented_prose_exemption() -> None:
    html = (
        '<pre><span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>'
        ".  A</span>\n\n   error\n         REQUIRED.  A single code.\n\n"
        "         A second paragraph of the definition body that\n"
        "         wraps onto another line.\n\n"
        "         Host: server.example.com\n</pre>"
    )
    units = parse_htmlized("d", _soup(html))[0].own_text.split("\n\n")
    assert units == [
        "error REQUIRED. A single code.",
        "A second paragraph of the definition body that wraps onto another line.",
        "Host: server.example.com",
    ]


# ----------------------------------------------------------------------- openid

_OPENID = """<html><body>
<a name="rfc.section.1"></a><h3>1.&nbsp;
Intro</h3>
<p>First
	paragraph.
</p>
<p>Second paragraph.
</p>
<ul class="text"><li>alpha
</li><li>beta</li></ul>
<div style="display: table"><pre>
  {
   "issuer":
     "https://server.example.com"
  }
</pre></div>
<table class="full"><tr><th>Property</th><th>Code</th></tr>
<tr><td>Tokens</td><td>yes</td></tr></table>
<a name="rfc.section.2"></a><h3>2.&nbsp;
Two</h3>
<p>One.</p>
<p>Two.</p>
</body></html>"""


@pytest.fixture
def oid() -> dict[str, Section]:
    return {s.id: s for s in parse_openid("d", _soup(_OPENID))}


def test_openid_two_paragraphs(oid: dict[str, Section]) -> None:
    assert oid["d#2"].own_text == "One.\n\nTwo."


def test_openid_layout(oid: dict[str, Section]) -> None:
    units = oid["d#1"].own_text.split("\n\n")
    assert units[:4] == ["First paragraph.", "Second paragraph.", "- alpha", "- beta"]
    assert units[4] == '{\n "issuer":\n   "https://server.example.com"\n}'
    assert units[5] == "Property | Code\nTokens | yes"


# ----------------------------------------------------------------------- corpus


def _all_sections() -> list[Section]:
    out: list[Section] = []
    for e in load_manifest():
        out.extend(parse_doc(e.doc_id, raw_path(e.doc_id).read_text(encoding="utf-8")))
    return out


@pytest.fixture(scope="module")
def corpus() -> list[Section]:
    return _all_sections()


@pytest.mark.corpus
def test_paragraph_breaks_present(corpus: list[Section]) -> None:
    big = [s for s in corpus if s.kind == "content" and s.own_tokens > 200]
    with_breaks = [s for s in big if "\n\n" in s.own_text]
    pct = len(with_breaks) / len(big)
    print(f"paragraph breaks: {len(with_breaks)}/{len(big)} = {pct:.1%}")
    assert pct >= 0.9


@pytest.mark.corpus
def test_figures_preserved(corpus: list[Section]) -> None:
    sec = next(s for s in corpus if s.id == "rfc6749#4.1")
    lines = sec.own_text.split("\n")
    assert any(ln.strip() == "+----------+" for ln in lines)
    assert "| Resource |" in lines
    assert any("(A)" in ln and "Client Identifier" not in ln for ln in lines)
    assert any("+----|-----+" in ln and "Client Identifier" in ln for ln in lines)


@pytest.mark.corpus
def test_no_bad_whitespace(corpus: list[Section]) -> None:
    for s in corpus:
        t = s.own_text
        assert "\xa0" not in t, s.id
        assert "\f" not in t, s.id
        assert "\n\n\n" not in t, s.id
        assert t == t.rstrip(), s.id
        assert not t.startswith("\n"), s.id
        if t[:1] == " ":
            # only a preformatted first unit may keep its relative indent
            first = t.split("\n\n")[0].split("\n")
            assert len(first) > 1 and any(not ln.startswith(" ") for ln in first), s.id


@pytest.mark.corpus
def test_leading_figure_indent_rfc7636(corpus: list[Section]) -> None:
    import html

    raw = raw_path("rfc7636").read_text(encoding="utf-8").split("\n")
    k = next(i for i, ln in enumerate(raw) if "|   Authz Server    |" in ln)
    expected = [html.unescape(ln).rstrip() for ln in raw[k - 1 : k + 3]]
    indent = min(len(ln) - len(ln.lstrip()) for ln in expected)
    sec = next(s for s in corpus if s.id == "rfc7636#1.1")
    got = sec.own_text.split("\n")[:4]
    assert got == [ln[indent:] for ln in expected]
    assert got[0].startswith(" " * 40) and got[2].startswith("+--------+")
