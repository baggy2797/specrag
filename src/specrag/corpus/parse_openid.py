"""Parser for OpenID Foundation specs (xml2rfc v2 HTML).

Every section heading is ``<a name="rfc.section.N"></a><h3>N.&nbsp;Title</h3>`` at
every depth; sections are flat siblings, so a section's text is everything up to
the next marker. Hierarchy comes from the number, never from the tag level.

The "N.1 Normative References" / "N.2 Informative References" headings have no
``rfc.section`` anchor; they are emitted as sections numbered from the heading text.
Other un-anchored headings (front matter, Authors' Addresses) become STOP markers
whose text is dropped; dropping more than ``_MAX_DROP_TOKENS`` of anything else
raises ParseError so content loss is never silent.
"""

import re

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from specrag.corpus.parse import ParseError, flag_references, is_references_title
from specrag.models import Section
from specrag.sections import make_id, parent
from specrag.tokens import count_tokens

_WS = re.compile(r"\s+")
_ANCHOR = re.compile(r"^rfc\.section\.(.+)$")
_MARK = "\x00H{}\x00"
_STOP = "\x00S{}\x00"
_SPLIT_RE = re.compile(r"\x00(H|S)(\d+)\x00")
_NUMBERED_TEXT = re.compile(
    r"^\s*(?:Appendix\s+)?([0-9A-Z]+(?:\.[0-9]+)*)\.?\xa0\s*(.+?)\s*$", re.S
)
_AUTHORS = re.compile(r"^Authors?['\u2019]s? Address(?:es)?$")
_MAX_DROP_TOKENS = 50
_NUM_PREFIX = re.compile(r"^\s*(?:Appendix\s+)?\S+?\.?\xa0\s*")
_BLOCKS = [
    "p",
    "li",
    "ul",
    "ol",
    "blockquote",
    "pre",
    "table",
    "tr",
    "td",
    "th",
    "br",
    "div",
    "dd",
    "dt",
    "dl",
]


def parse_openid(doc_id: str, soup: BeautifulSoup) -> list[Section]:
    for t in soup.select("table.TOCbug"):
        t.decompose()

    headings: list[tuple[str, str]] = []
    numbered_h3: set[int] = set()
    for a in soup.find_all("a", attrs={"name": _ANCHOR}):
        m = _ANCHOR.match(str(a.get("name")))
        h = a.find_next_sibling()
        if m is None or h is None or h.name != "h3":
            continue
        title = _WS.sub(" ", _NUM_PREFIX.sub("", h.get_text())).strip()
        numbered_h3.add(id(h))
        h.replace_with(NavigableString(_MARK.format(len(headings))))
        headings.append((m.group(1), title))
        a.decompose()

    stops: list[str] = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        if id(h) in numbered_h3:
            continue
        m2 = _NUMBERED_TEXT.match(h.get_text())
        if m2 is not None and is_references_title(_WS.sub(" ", m2.group(2))):
            h.replace_with(NavigableString(_MARK.format(len(headings))))
            headings.append((m2.group(1), _WS.sub(" ", m2.group(2)).strip()))
        else:
            h.replace_with(NavigableString(_STOP.format(len(stops))))
            stops.append(_WS.sub(" ", h.get_text().replace("\xa0", " ")).strip())

    for tag in soup.find_all(_BLOCKS):
        assert isinstance(tag, Tag)
        tag.insert_after(NavigableString(" "))
        tag.insert_before(NavigableString(" "))

    parts = _SPLIT_RE.split((soup.body or soup).get_text())
    # re.split with two groups: [front, kind, idx, body, kind, idx, body, ...]
    out: list[Section] = []
    for i in range(1, len(parts), 3):
        kind, idx, body = parts[i], int(parts[i + 1]), parts[i + 2]
        own = _WS.sub(" ", body.replace("\xa0", " ")).strip()
        if kind == "S":
            heading = stops[idx]
            front_matter = not out
            if not front_matter and not _AUTHORS.match(heading):
                if count_tokens(own) > _MAX_DROP_TOKENS:
                    raise ParseError(
                        f"{doc_id}: un-numbered heading {heading!r} would drop "
                        f"{count_tokens(own)} tokens of text"
                    )
            continue
        number, title = headings[idx]
        full_id = make_id(doc_id, number)
        out.append(
            Section(
                id=full_id,
                doc_id=doc_id,
                number=number,
                title=title,
                parent_id=parent(full_id),
                depth=len(number.split(".")),
                own_text=own,
                own_tokens=count_tokens(own),
            )
        )
    return flag_references(out)
