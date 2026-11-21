"""Parser for OpenID Foundation specs (xml2rfc v2 HTML).

Every section heading is ``<a name="rfc.section.N"></a><h3>N.&nbsp;Title</h3>`` at
every depth; sections are flat siblings, so a section's text is everything up to
the next marker. Hierarchy comes from the number, never from the tag level.
"""

import re

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from specrag.models import Section
from specrag.sections import make_id, parent
from specrag.tokens import count_tokens

_WS = re.compile(r"\s+")
_ANCHOR = re.compile(r"^rfc\.section\.(.+)$")
_MARK = "\x00H{}\x00"
_STOP = "\x00S\x00"
_SPLIT_RE = re.compile(r"\x00(?:H(\d+)|S)\x00")
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

    for h in soup.find_all(["h1", "h2", "h3"]):
        if id(h) not in numbered_h3:
            h.replace_with(NavigableString(_STOP))

    for tag in soup.find_all(_BLOCKS):
        assert isinstance(tag, Tag)
        tag.insert_after(NavigableString(" "))
        tag.insert_before(NavigableString(" "))

    parts = _SPLIT_RE.split((soup.body or soup).get_text())
    # re.split with one group: [front, idx|None, body, idx|None, body, ...]
    out: list[Section] = []
    for i in range(1, len(parts), 2):
        idx = parts[i]
        if idx is None:
            continue
        number, title = headings[int(idx)]
        own = _WS.sub(" ", parts[i + 1].replace("\xa0", " ")).strip()
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
    return out
