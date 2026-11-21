import pytest
from pydantic import ValidationError

from specrag.models import Chunk, DocStatus, Section, SectionIndex


def _sec(number: str, tokens: int = 1) -> Section:
    sid = f"rfc6749#{number}"
    par = number.rsplit(".", 1)[0] if "." in number else None
    return Section(
        id=sid,
        doc_id="rfc6749",
        number=number,
        title="t",
        parent_id=f"rfc6749#{par}" if par else None,
        depth=number.count(".") + 1,
        own_text="x",
        own_tokens=tokens,
    )


def test_chunk_requires_parallel_sections_tokens() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            id="c",
            doc_id="rfc6749",
            text="t",
            index_text="t",
            sections=["rfc6749#1"],
            tokens=[],
            n_tokens=1,
        )
    with pytest.raises(ValidationError):
        Chunk(id="c", doc_id="d", text="t", index_text="t", sections=[], tokens=[], n_tokens=0)


def test_chunk_valid_and_frozen() -> None:
    c = Chunk(
        id="c", doc_id="d", text="t", index_text="t", sections=["d#1"], tokens=[3], n_tokens=3
    )
    with pytest.raises(ValidationError):
        c.text = "u"


def test_doc_status_values() -> None:
    assert DocStatus.STANDARDS_TRACK == "standards_track"
    assert DocStatus.OPENID_FINAL == "openid_final"


def test_section_index_ancestors_and_lookup() -> None:
    idx = SectionIndex([_sec("4"), _sec("4.1"), _sec("4.1.2", 7)])
    assert idx.ancestors("rfc6749#4.1.2") == ["rfc6749#4.1", "rfc6749#4"]
    assert idx.ancestors("rfc6749#4") == []
    assert "rfc6749#4.1" in idx
    assert "rfc6749#9" not in idx
    assert idx.own_tokens("rfc6749#4.1.2") == 7
    with pytest.raises(KeyError, match="rfc6749#9"):
        idx.get("rfc6749#9")


def test_section_index_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="rfc6749#4"):
        SectionIndex([_sec("4"), _sec("4")])
