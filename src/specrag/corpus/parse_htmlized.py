"""Parser for htmlized plain-text RFCs (content in ``<pre>``, ``span.hN`` headings).

Page furniture (footer and running header) is wrapped in ``span.grey`` and is
removed by element. A regex line-drop (footer ``... [Page N]``, header
``RFC N ... Month YYYY``, both requiring 2+ space gaps) then runs as well: it is
the only path for docs without grey spans and catches the last page's bare footer.

Layout: the body text is split into paragraph units by
:func:`specrag.corpus.layout.text_units` (prose joined, figures kept verbatim).
Removed furniture leaves a page-break marker. Across a page break, the text is
re-joined into one paragraph only when the line before ends in a lowercase letter,
comma or hyphen and the line after starts lowercase or with "(" (see ``_joins``);
otherwise the break becomes a paragraph break. So a page break may still split a paragraph (or a
figure) into two units -- an accepted limitation.
"""

import re

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from specrag.corpus.layout import collapse, has_layout_marks, join_units, text_units
from specrag.corpus.parse import flag_references
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
_PAGE = "\x01"  # page-break marker left where furniture was removed
_JOIN_BEFORE = re.compile(r"[a-z,\-]$")
_JOIN_AFTER = re.compile(r"^\s*[a-z(]")


def _is_heading(tag: Tag) -> bool:
    classes = tag.get("class")
    return (
        tag.name == "span"
        and isinstance(classes, list)
        and any(_HEADING_CLASS.match(c) for c in classes)
    )


def _drop_furniture_lines(text: str) -> str:
    return "\n".join(
        _PAGE if (_FOOTER.match(ln) or _HEADER.match(ln)) else ln
        for ln in text.replace("\f", "\n").split("\n")
    )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _joins(prev: str, nxt: str) -> bool:
    """Whether a page break between ``prev`` and ``nxt`` falls mid-paragraph.

    Deliberately conservative (a split paragraph is cheaper than a false merge):
    ``prev`` must end in a lowercase letter, comma or hyphen, and ``nxt`` must
    start with a lowercase letter or ``(`` -- or, after a comma, a digit (numeric
    arrays). The two lines must share an indent (within 2 columns), and lines
    holding a heading marker or figure/table marks never join.
    """
    if abs(_indent(prev) - _indent(nxt)) > 2:
        return False  # e.g. a centred caption before the break, prose after it
    prev = prev.rstrip()
    if not prev or "\x00" in prev or "\x00" in nxt:
        return False
    if has_layout_marks(prev) or has_layout_marks(nxt) or not _JOIN_BEFORE.search(prev):
        return False
    if prev.endswith(",") and nxt.lstrip()[:1].isdigit():
        return True  # numeric array continued across the page
    return bool(_JOIN_AFTER.match(nxt))


def _resolve_page_breaks(text: str) -> str:
    """Replace each run of blank/furniture lines that contains a page marker with
    nothing (mid-sentence continuation) or one blank line (paragraph break)."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _PAGE not in lines[i] and lines[i].strip():
            out.append(lines[i])
            i += 1
            continue
        j = i
        page = False
        while j < len(lines) and (_PAGE in lines[j] or not lines[j].strip()):
            page = page or _PAGE in lines[j]
            j += 1
        prev = out[-1] if out else ""
        nxt = lines[j] if j < len(lines) else ""
        if not (page and _joins(prev, nxt)):
            out.append("")
        i = j
    return "\n".join(out)


def parse_htmlized(doc_id: str, soup: BeautifulSoup) -> list[Section]:
    pres = soup.find_all("pre")
    grey = [g for pre in pres for g in pre.select("span.grey")]
    for g in grey:
        g.replace_with(NavigableString(_PAGE))

    # (number, title) per heading; unnumbered headings get None so they end a section.
    headings: list[tuple[str, str] | None] = []
    for pre in pres:
        for span in pre.find_all(_is_heading):
            anchor = span.select_one('a.selflink[id^="section-"], a.selflink[id^="appendix-"]')
            entry: tuple[str, str] | None = None
            if anchor is not None:
                raw_id = str(anchor.get("id"))
                number = raw_id.split("-", 1)[1]
                title = collapse(
                    anchor.next_sibling.get_text()
                    if isinstance(anchor.next_sibling, Tag)
                    else str(anchor.next_sibling or "")
                )
                title = title.lstrip(".").strip()
                entry = (number, title)
            span.replace_with(_MARK.format(len(headings)))
            headings.append(entry)

    text = f"\n{_PAGE}\n".join(pre.get_text() for pre in pres)
    # Always run the line-drop too: the final page's footer is not wrapped in span.grey.
    text = _resolve_page_breaks(_drop_furniture_lines(text))

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
        own = join_units(text_units(body))
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
