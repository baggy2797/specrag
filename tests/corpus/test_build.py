from pathlib import Path

import pytest
from typer.testing import CliRunner

from specrag.cli import app
from specrag.corpus import build
from specrag.corpus.manifest import ManifestEntry
from specrag.models import DocStatus, Section


def _secs() -> list[Section]:
    a = Section(
        id="x:1",
        doc_id="x",
        number="1",
        title="Intro",
        parent_id=None,
        depth=0,
        own_text="a",
        own_tokens=3,
    )
    b = Section(
        id="x:1.1",
        doc_id="x",
        number="1.1",
        title="Sub",
        parent_id="x:1",
        depth=1,
        own_text="b",
        own_tokens=5,
    )
    return [a, b]


def _entry() -> ManifestEntry:
    return ManifestEntry(doc_id="x", title="X", status=DocStatus.INFORMATIONAL, url="http://x")


def _patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPECRAG_CACHE", str(tmp_path))
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(build, "parse_doc", lambda doc_id, html: _secs())


def test_build_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, tmp_path)
    idx = build.build_sections([_entry()])
    assert "x:1.1" in idx
    assert (tmp_path / "sections.jsonl").read_text().count("\n") == 2
    loaded = build.load_section_index()
    assert loaded.get("x:1.1") == _secs()[1]
    assert loaded.own_tokens("x:1") == 3


def test_load_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECRAG_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="specrag parse"):
        build.load_section_index()


def test_sections_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, tmp_path)
    build.build_sections([_entry()])
    r = CliRunner().invoke(app, ["sections", "x"])
    assert r.exit_code == 0
    assert "x:1  Intro  (3)" in r.output
    assert "  x:1.1  Sub  (5)" in r.output
    bad = CliRunner().invoke(app, ["sections", "nope"])
    assert bad.exit_code == 2
    assert "nope" in bad.output
