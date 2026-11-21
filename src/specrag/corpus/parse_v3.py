"""Parser for xml2rfc v3 HTML (``<section id="section-N">`` nesting)."""

import copy
import re

from bs4 import BeautifulSoup, Tag

from specrag.corpus.parse import flag_references
from specrag.models import Section
from specrag.sections import make_id, parent
from specrag.tokens import count_tokens

_ID_RE = re.compile(r"^(?:section-\d|appendix-[A-Z])")
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
_BLOCK_TAGS = ["p", "li", "pre", "div", "dt", "dd", "tr", "td", "th", "br", "table", "ul", "ol"]
_WS = re.compile(r"\s+")


def _normalise_number(raw: str) -> str:
    n = _WS.sub(" ", raw).strip().rstrip(".").strip()
    if n.startswith("Appendix "):
        n = n[len("Appendix ") :]
    return n.strip()


def _own_text(section: Tag) -> str:
    clone = copy.copy(section)
    for sub in clone.find_all("section"):
        sub.decompose()
    for h in clone.find_all(_HEADING_TAGS):
        h.decompose()
    for a in clone.select("a.pilcrow"):
        a.decompose()
    for b in clone.find_all(_BLOCK_TAGS):
        b.insert_after(" ")
    return _WS.sub(" ", clone.get_text("")).strip()


def parse_v3(doc_id: str, soup: BeautifulSoup) -> list[Section]:
    out: list[Section] = []
    for sec in soup.find_all("section"):
        sid = sec.get("id")
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            continue
        heading = sec.find(_HEADING_TAGS)
        if not isinstance(heading, Tag):
            continue
        num_a = heading.select_one("a.section-number")
        name_a = heading.select_one("a.section-name")
        if num_a is None or name_a is None:
            continue  # unnumbered (Acknowledgements, Authors' Addresses, ...)
        number = _normalise_number(num_a.get_text())
        if not number:
            continue
        title = _WS.sub(" ", name_a.get_text()).strip()
        text = _own_text(sec)
        full_id = make_id(doc_id, number)
        out.append(
            Section(
                id=full_id,
                doc_id=doc_id,
                number=number,
                title=title,
                parent_id=parent(full_id),
                depth=len(number.split(".")),
                own_text=text,
                own_tokens=count_tokens(text),
            )
        )
    return flag_references(out)
