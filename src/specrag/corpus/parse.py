"""Dialect detection, shared validation and dispatch for spec HTML parsers.

Recon of the cached corpus (Task 5, Step 1) -- markers per dialect:

* xml2rfc v3 (rfc9700, oauth-2.1, ...): nested ``<section id="section-4.1">``;
  appendices ``<section id="appendix-A">``. The heading (h2..h6) holds
  ``a.section-number`` ("4. ", "Appendix A. ", "A.1. ") and ``a.section-name``.
  Non-content sections have ids ``section-abstract``, ``section-boilerplate.N``,
  ``section-toc.N``, ``section-note.N``. Unnumbered trailing sections
  (Acknowledgements, Authors' Addresses) still get ids like ``appendix-A`` but
  have NO ``a.section-number``, so the number is taken from that anchor, not the id.
* htmlized RFC (rfc6749, ...): content in ``<pre>``; headings
  ``<span class="h2"><a class="selflink" id="section-1">``; appendices
  ``appendix-A``. Page furniture in ``span.grey``. (Parser: Task 6.)
* OpenID (oidc-core, ...): ``<a name="rfc.section.1"></a><h3>``; the anchor uses
  ``name=``, not ``id=``; appendices ``rfc.section.A``. (Parser: Task 7.)
"""

import re
from collections.abc import Callable
from enum import StrEnum

from bs4 import BeautifulSoup

from specrag.models import Section
from specrag.sections import parent


class ParseError(Exception):
    pass


class Dialect(StrEnum):
    RFC_V3 = "rfc_v3"
    RFC_HTMLIZED = "rfc_htmlized"
    OPENID_V2 = "openid_v2"


_V3_RE = re.compile(r"<section\b[^>]*\bid=\"section-\d")
_OPENID_RE = re.compile(r"<a\b[^>]*\bname=\"rfc\.section\.")


def detect_dialect(html: str) -> Dialect:
    if _V3_RE.search(html):
        return Dialect.RFC_V3
    if _OPENID_RE.search(html):
        return Dialect.OPENID_V2
    soup = BeautifulSoup(html, "lxml")
    for pre in soup.find_all("pre"):
        if pre.select_one('a.selflink[id^="section-"]') is not None:
            return Dialect.RFC_HTMLIZED
    raise ParseError("unrecognised document dialect")


def validate_sections(doc_id: str, sections: list[Section]) -> None:
    if not sections:
        raise ParseError(f"{doc_id}: no sections parsed")
    seen: set[str] = set()
    for s in sections:
        if s.id in seen:
            raise ParseError(f"{doc_id}: duplicate section id {s.id}")
        seen.add(s.id)
    for s in sections:
        p = parent(s.id)
        if p is not None and p not in seen:
            raise ParseError(f"{doc_id}: section {s.id} has missing parent {p}")


def _parser_for(dialect: Dialect) -> Callable[[str, BeautifulSoup], list[Section]]:
    if dialect == Dialect.RFC_V3:
        from specrag.corpus.parse_v3 import parse_v3

        return parse_v3
    if dialect == Dialect.RFC_HTMLIZED:
        from specrag.corpus.parse_htmlized import parse_htmlized

        return parse_htmlized
    from specrag.corpus.parse_openid import parse_openid

    return parse_openid


def parse_doc(doc_id: str, html: str) -> list[Section]:
    dialect = detect_dialect(html)
    parser = _parser_for(dialect)
    sections = parser(doc_id, BeautifulSoup(html, "lxml"))
    validate_sections(doc_id, sections)
    return sections
