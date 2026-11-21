"""Shared layout logic for ``Section.own_text``: paragraph units, preformatted blocks.

Output contract (all dialects): paragraph units are joined by exactly one blank
line. A prose unit has its whitespace collapsed to single spaces. A preformatted
unit (figure, ASCII art, table, code, HTTP/JSON example) keeps its lines verbatim
except that trailing spaces are stripped and the common leading indent is removed.
The final text has no ``\\xa0``, no ``\\f``, no run of 3+ newlines and no
leading/trailing whitespace.

HTML dialects (v3, OpenID) go through :func:`html_units`; the htmlized dialect,
whose whole document is one ``<pre>``, goes through :func:`text_units`.
"""

import re
from dataclasses import dataclass

from bs4 import Tag
from bs4.element import Comment, NavigableString, PageElement

_WS = re.compile(r"\s+")

# Elements that start a new paragraph unit (their content is recursed into).
_BLOCK = {
    "p", "li", "dd", "dt", "dl", "ul", "ol", "div", "blockquote", "figure",
    "figcaption", "aside", "section", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "center", "form", "address",
}  # fmt: skip
_SKIP = {"script", "style", "head", "title", "colgroup", "col"}


@dataclass(frozen=True)
class Unit:
    text: str
    pre: bool = False


def collapse(text: str) -> str:
    """Single-line form: every whitespace run (incl. ``\\xa0``) becomes one space."""
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def dedent_block(text: str) -> str:
    """Verbatim lines with trailing spaces and the common indent removed.

    Leading and trailing blank lines are dropped; tabs are expanded first.
    """
    lines = [ln.rstrip() for ln in text.replace("\xa0", " ").expandtabs(8).split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    indent = min((len(ln) - len(ln.lstrip(" ")) for ln in lines if ln), default=0)
    return "\n".join(ln[indent:] for ln in lines)


def render_table(table: Tag) -> str:
    """One line per row, cells collapsed and joined by ``" | "``; empty rows dropped."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [collapse(c.get_text(" ")) for c in tr.find_all(["td", "th"], recursive=False)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _ol_marker(ol: Tag, index: int) -> str:
    try:
        start = int(str(ol.get("start") or "1"))
    except ValueError:
        start = 1
    n = start + index
    kind = str(ol.get("type") or "1")
    if kind == "A" and 1 <= n <= 26:
        return f"{chr(64 + n)}."
    if kind == "a" and 1 <= n <= 26:
        return f"{chr(96 + n)}."
    return f"{n}."


def _is_class(tag: Tag, name: str) -> bool:
    classes = tag.get("class")
    return isinstance(classes, list) and name in classes


class _Walker:
    def __init__(self) -> None:
        self.units: list[Unit] = []
        self.buf: list[str] = []

    def flush(self) -> None:
        text = collapse("".join(self.buf))
        self.buf = []
        if text:
            self.units.append(Unit(text))

    def add_pre(self, text: str) -> None:
        self.flush()
        block = dedent_block(text)
        if block:
            self.units.append(Unit(block, pre=True))

    def walk(self, node: PageElement) -> None:
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            if "\x00" in node:  # parser marker (heading / STOP): keep it a unit of its own
                self.flush()
                self.units.append(Unit(str(node).strip()))
                return
            self.buf.append(str(node))
            return
        if isinstance(node, Tag) and node.name == "br":
            self.buf.append(" ")
            return
        if not isinstance(node, Tag) or node.name in _SKIP:
            return
        if node.name == "pre":
            self.add_pre(node.get_text())
        elif node.name == "table":
            self.flush()
            rendered = render_table(node)
            if rendered:
                self.units.append(Unit(rendered, pre=True))
        elif node.name in ("ul", "ol"):
            self.flush()
            items = [c for c in node.children if isinstance(c, Tag) and c.name == "li"]
            for i, li in enumerate(items):
                marker = _ol_marker(node, i) if node.name == "ol" else "-"
                self._list_item(li, marker)
            # stray non-<li> content inside the list
            for c in node.children:
                if not (isinstance(c, Tag) and c.name == "li"):
                    self.walk(c)
            self.flush()
        elif node.name in _BLOCK or _is_class(node, "artwork") or _is_class(node, "sourcecode"):
            self.flush()
            for c in list(node.children):
                self.walk(c)
            self.flush()
        else:
            for c in list(node.children):
                self.walk(c)

    def _list_item(self, li: Tag, marker: str) -> None:
        self.flush()
        start = len(self.units)
        for c in list(li.children):
            self.walk(c)
        self.flush()
        if len(self.units) == start:
            return
        first = self.units[start]
        if first.pre:
            self.units.insert(start, Unit(marker))
        else:
            self.units[start] = Unit(f"{marker} {first.text}")


def html_units(root: Tag) -> list[Unit]:
    """Paragraph units of an HTML subtree (block elements each start a unit)."""
    w = _Walker()
    for c in list(root.children):
        w.walk(c)
    w.flush()
    return w.units


# ------------------------------------------------------------------- htmlized

PROSE_INDENT = 3
_GAP = re.compile(r"(\S) {2,}\S")
_BULLET = re.compile(r"^(?:[o*+\-]|\(?[0-9A-Za-z]{1,3}[.)])(?= {2,}\S)")
_SENTENCE_END = set(".?!:;)\"'")
_ART = ("+--", "--+", "-->", "<--")
_CODEY = set("{}<>=|\\")


def _has_layout_gap(line: str) -> bool:
    """2+ spaces between non-space chars, ignoring RFC sentence spacing (".  ")
    and the gap after a leading list marker ("o  text", "(A)  text", "1.  text")."""
    body = _BULLET.sub("", line).lstrip()
    return any(m.group(1) not in _SENTENCE_END for m in _GAP.finditer(body))


def has_layout_marks(line: str) -> bool:
    """Character-level figure/table signal for one line (indent is ignored)."""
    ln = line.strip()
    return _has_layout_gap(ln) or ln[:1] in ("+", "|") or any(a in ln for a in _ART)


def _is_preformatted(lines: list[str]) -> bool:
    """Classify one blank-line-delimited block of htmlized RFC text.

    ``lines`` are the raw lines of the block (no blank lines). After removing the
    block's common indent, the block is preformatted when any line

    * has 2+ consecutive spaces between non-space characters (layout columns),
      except the RFC two-space sentence gap after ``. ? ! : ; ) " '`` and the gap
      after a leading list marker (``o``, ``*``, ``-``, ``(A)``, ``1.``);
    * starts with a box/table character ``+`` or ``|``; or
    * contains ``+--``, ``--+``, ``-->`` or ``<--``.

    Otherwise, a block whose minimum indent exceeds the prose indent
    (:data:`PROSE_INDENT`, 3 in RFC text; figures and examples sit at 6+) is
    preformatted too -- unless it reads as prose: it ends in ``. , ; :``,
    contains none of ``{ } < > = | \\``, and its first character is a letter,
    digit, quote or bracket. That exception keeps indented definition bodies
    and nested list paragraphs joined, while HTTP/JSON examples and captions
    stay verbatim. Anything else is prose and is joined into one line.
    """
    indent = min(len(ln) - len(ln.lstrip(" ")) for ln in lines)
    body = [ln[indent:] for ln in lines]
    if any(has_layout_marks(ln) for ln in body):
        return True
    if indent <= PROSE_INDENT:
        return False
    flat = " ".join(ln.strip() for ln in body)
    prose_like = (
        flat[-1:] in (".", ",", ";", ":")
        and not any(ch in _CODEY for ch in flat)
        and (flat[:1].isalnum() or flat[:1] in "\"'([")
    )
    return not prose_like


def text_units(text: str) -> list[Unit]:
    """Paragraph units of plain RFC text: blocks split at blank lines, each block
    classified by :func:`_is_preformatted`, prose joined with single spaces."""
    lines = [ln.rstrip() for ln in text.replace("\xa0", " ").expandtabs(8).split("\n")]
    units: list[Unit] = []
    block: list[str] = []

    def emit() -> None:
        if not block:
            return
        if _is_preformatted(block):
            units.append(Unit(dedent_block("\n".join(block)), pre=True))
        else:
            units.append(Unit(collapse(" ".join(block))))
        block.clear()

    for ln in lines:
        if ln.strip():
            block.append(ln)
        else:
            emit()
    emit()
    return units


# ----------------------------------------------------------------- final text

_MULTI_NL = re.compile(r"\n{3,}")


def join_units(units: list[Unit]) -> str:
    return finalize("\n\n".join(u.text for u in units if u.text))


def finalize(text: str) -> str:
    """Final normalization applied to every own_text.

    Only newlines are stripped at the ends (lines are already right-stripped), so a
    preformatted block that opens a section keeps its relative indent. Prose
    units are collapsed and never start with a space.
    """
    text = text.replace("\xa0", " ").replace("\f", "\n").replace("\r", "")
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    text = _MULTI_NL.sub("\n\n", text)
    text = text.strip("\n")
    return "" if not text.strip() else text
