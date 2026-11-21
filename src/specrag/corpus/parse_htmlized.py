"""Parser for htmlized plain-text RFCs (content in ``<pre>``, ``span.hN`` headings).

Page furniture (footer and running header) is wrapped in ``span.grey`` and is
removed by element. A regex line-drop (footer ``... [Page N]``, header
``RFC N ... Month YYYY``, both requiring 2+ space gaps) then runs as well: it is
the only path for docs without grey spans and catches the last page's bare footer.
"""

import re

from bs4 import BeautifulSoup, Tag

from specrag.models import Section
from specrag.sections import make_id, parent
from specrag.tokens import count_tokens

_WS = re.compile(r"\s+")
_HEADING_CLASS = re.compile(r"^h[1-6]$")
_FOOTER = re.compile(r"^\S.*\s{2,}\[Page \d+\]\s*$")
_HEADER = re.compile(r"^RFC \d+\s{2,}\S.*\s{2,}\w+ \d{4}\s*$")
_TRAILER = re.compile(r"^Authors?['\u2019]s? Address(?:es)?\s*$", re.MULTILINE)
_MARK = "\x00H{}\x00"
_MARK_RE = re.compile(r"\x00H(\d+)\x00")


def _is_heading(tag: Tag) -> bool:
    classes = tag.get("class")
    return (
        tag.name == "span"
        and isinstance(classes, list)
        and any(_HEADING_CLASS.match(c) for c in classes)
    )


def _drop_furniture_lines(text: str) -> str:
    kept = [
        ln
        for ln in text.replace("\f", "\n").split("\n")
        if not (_FOOTER.match(ln) or _HEADER.match(ln))
    ]
    return "\n".join(kept)


def parse_htmlized(doc_id: str, soup: BeautifulSoup) -> list[Section]:
    pres = soup.find_all("pre")
    grey = [g for pre in pres for g in pre.select("span.grey")]
    for g in grey:
        g.decompose()

    # (number, title) per heading; unnumbered headings get None so they end a section.
    headings: list[tuple[str, str] | None] = []
    for pre in pres:
        for span in pre.find_all(_is_heading):
            anchor = span.select_one('a.selflink[id^="section-"], a.selflink[id^="appendix-"]')
            entry: tuple[str, str] | None = None
            if anchor is not None:
                raw_id = str(anchor.get("id"))
                number = raw_id.split("-", 1)[1]
                title = _WS.sub(
                    " ",
                    anchor.next_sibling.get_text()
                    if isinstance(anchor.next_sibling, Tag)
                    else str(anchor.next_sibling or ""),
                ).strip()
                title = title.lstrip(".").strip()
                entry = (number, title)
            span.replace_with(_MARK.format(len(headings)))
            headings.append(entry)

    text = "\n".join(pre.get_text() for pre in pres)
    # Always run the line-drop too: the final page's footer is not wrapped in span.grey.
    text = _drop_furniture_lines(text)

    parts = _MARK_RE.split(text)  # [front, idx, body, idx, body, ...]
    out: list[Section] = []
    for i in range(1, len(parts), 2):
        entry = headings[int(parts[i])]
        if entry is None:
            continue
        number, title = entry
        body = parts[i + 1]
        if i == len(parts) - 2:  # plain-text trailer only ever follows the last section
            body = _TRAILER.split(body, maxsplit=1)[0]
        own = _WS.sub(" ", body.replace("\f", " ")).strip()
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
