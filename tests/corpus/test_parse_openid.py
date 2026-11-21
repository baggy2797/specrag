from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from specrag.corpus.manifest import raw_path
from specrag.corpus.parse import Dialect, detect_dialect, parse_doc
from specrag.corpus.parse_openid import parse_openid
from specrag.models import Section

FIXTURE = Path(__file__).parent / "fixtures" / "openid_min.html"
IDS = ["d#1", "d#3", "d#3.1", "d#3.1.2", "d#A", "d#A.1"]


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def secs(html: str) -> dict[str, Section]:
    return {s.id: s for s in parse_openid("d", BeautifulSoup(html, "lxml"))}


def test_dispatch(html: str) -> None:
    assert detect_dialect(html) == Dialect.OPENID_V2
    assert [s.id for s in parse_doc("d", html)] == IDS


def test_ids_titles_nesting(secs: dict[str, Section]) -> None:
    assert list(secs) == IDS
    assert secs["d#3"].title == "Main"
    assert secs["d#A"].title == "Extras"
    assert secs["d#3.1.2"].title == "Deep"
    assert secs["d#3.1.2"].depth == 3
    assert secs["d#3.1.2"].parent_id == "d#3.1"
    assert secs["d#3.1"].parent_id == "d#3"
    assert secs["d#3"].parent_id is None
    assert secs["d#A.1"].parent_id == "d#A"


def test_own_text_excludes_children_and_joins_lists(secs: dict[str, Section]) -> None:
    assert secs["d#3"].own_text == "Main text. first item second item"
    assert secs["d#3.1"].own_text == "Sub text."
    assert secs["d#1"].own_tokens > 0


def test_front_matter_toc_and_trailer_excluded(secs: dict[str, Section]) -> None:
    joined = " ".join(s.own_text for s in secs.values())
    for bad in ("Front matter", "Toc listing", "TOC", "Somebody Else", "Authors"):
        assert bad not in joined
    assert secs["d#A.1"].own_text == "Last text."


EXPECTED = {"oidc-core": (183, "C"), "oidc-discovery": (31, "B"), "oidc-registration": (32, "B")}


@pytest.mark.corpus
@pytest.mark.parametrize("doc_id", list(EXPECTED))
def test_corpus_parses(doc_id: str) -> None:
    out = parse_doc(doc_id, raw_path(doc_id).read_text(encoding="utf-8"))
    count, last = EXPECTED[doc_id]
    assert len(out) == count
    assert out[0].number == "1"
    assert out[-1].number == last
    for s in out:
        assert "\xa0TOC\xa0" not in s.own_text, s.id
        assert not s.own_text.startswith("TOC"), s.id
        assert not s.own_text.endswith("TOC"), s.id
    assert "Authors' Addresses" not in out[-1].own_text


@pytest.mark.corpus
def test_corpus_core_details() -> None:
    out = {
        s.id: s for s in parse_doc("oidc-core", raw_path("oidc-core").read_text(encoding="utf-8"))
    }
    assert out["oidc-core#3.1.2.1"].title == "Authentication Request"
    assert out["oidc-core#3.1.2.1"].parent_id == "oidc-core#3.1.2"
