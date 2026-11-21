import hashlib
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

import specrag.cli as cli
from specrag.corpus.fetch import FetchError, fetch_all
from specrag.corpus.manifest import ManifestEntry, load_manifest, raw_path
from specrag.models import DocStatus

BODY = b"<html>spec</html>"
SHA = hashlib.sha256(BODY).hexdigest()


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SPECRAG_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def entry(sha: str | None = SHA, doc_id: str = "rfc1") -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id,
        title="T",
        status=DocStatus.STANDARDS_TRACK,
        url=f"https://example.test/{doc_id}.html",
        sha256=sha,
    )


def client(handler: httpx.MockTransport | None = None) -> httpx.Client:
    return httpx.Client(
        transport=handler or httpx.MockTransport(lambda r: httpx.Response(200, content=BODY))
    )


def test_fetch_writes_atomically_and_verifies_sha() -> None:
    report = fetch_all([entry()], client=client())
    p = raw_path("rfc1")
    assert p.read_bytes() == BODY
    assert report.fetched == ["rfc1"]
    assert not list(p.parent.glob("*.tmp"))
    assert (p.parent.parent / "manifest.lock.json").exists()


def test_cached_with_matching_sha_is_not_redownloaded() -> None:
    fetch_all([entry()], client=client())

    def boom(r: httpx.Request) -> httpx.Response:
        raise AssertionError("should not download")

    report = fetch_all([entry()], client=client(httpx.MockTransport(boom)))
    assert report.skipped == ["rfc1"] and report.fetched == []


def test_unpinned_is_downloaded_and_marked() -> None:
    report = fetch_all([entry(None)], client=client())
    assert report.fetched == ["rfc1"] and report.unpinned == ["rfc1"]


def test_hash_drift_raises_and_leaves_no_file() -> None:
    with pytest.raises(FetchError, match="sha256 mismatch") as ei:
        fetch_all([entry("0" * 64)], client=client())
    assert ei.value.doc_id == "rfc1"
    assert not raw_path("rfc1").exists()
    assert not list(raw_path("rfc1").parent.glob("*.tmp"))


def test_404_raises_named_error() -> None:
    c = client(httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(FetchError) as ei:
        fetch_all([entry()], client=c)
    assert "rfc1" in str(ei.value) and "404" in str(ei.value)


def test_retries_then_succeeds() -> None:
    codes = iter([503, 503, 200])

    def handler(r: httpx.Request) -> httpx.Response:
        code = next(codes)
        return httpx.Response(code, content=BODY if code == 200 else b"")

    sleeps: list[float] = []
    fetch_all([entry()], client=client(httpx.MockTransport(handler)), sleep=sleeps.append)
    assert raw_path("rfc1").exists() and len(sleeps) == 2


def test_transport_errors_retried_then_succeeds() -> None:
    calls = iter([True, True, False])

    def handler(r: httpx.Request) -> httpx.Response:
        if next(calls):
            raise httpx.ConnectError("boom", request=r)
        return httpx.Response(200, content=BODY)

    sleeps: list[float] = []
    fetch_all([entry()], client=client(httpx.MockTransport(handler)), sleep=sleeps.append)
    assert raw_path("rfc1").exists() and len(sleeps) == 2


def test_transport_errors_exhausted_names_doc_and_cause() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns down", request=r)

    with pytest.raises(FetchError) as ei:
        fetch_all(
            [entry()], client=client(httpx.MockTransport(handler)), retries=2, sleep=lambda s: None
        )
    assert "rfc1" in str(ei.value) and "dns down" in str(ei.value)
    assert not raw_path("rfc1").exists()


def test_retries_exhausted_raises() -> None:
    c = client(httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(FetchError, match="503"):
        fetch_all([entry()], client=c, retries=2, sleep=lambda s: None)


def test_pin_writes_hashes_back(tmp_path: Path) -> None:
    m = tmp_path / "m.yaml"
    m.write_text(
        yaml.safe_dump(
            {
                "docs": [
                    {
                        "doc_id": "rfc1",
                        "title": "T",
                        "status": "standards_track",
                        "url": "https://example.test/rfc1.html",
                        "sha256": None,
                    }
                ]
            }
        )
    )
    fetch_all(load_manifest(m), client=client(), pin=True, manifest_path=m)
    assert load_manifest(m)[0].sha256 == SHA


def test_unsafe_doc_id_rejected(tmp_path: Path) -> None:
    m = tmp_path / "m.yaml"
    m.write_text(
        yaml.safe_dump(
            {
                "docs": [
                    {
                        "doc_id": "../evil",
                        "title": "T",
                        "status": "standards_track",
                        "url": "https://example.test/x.html",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="invalid doc_id"):
        load_manifest(m)
    with pytest.raises(ValueError, match="escapes"):
        raw_path("../evil")


def test_manifest_has_23_unique_docs() -> None:
    entries = load_manifest(Path(__file__).parents[2] / "corpus" / "manifest.yaml")
    ids = [e.doc_id for e in entries]
    assert len(ids) == 23 and len(set(ids)) == 23
    by = {e.doc_id: e for e in entries}
    assert by["oauth-2.1"].status is DocStatus.DRAFT
    assert by["rfc9700"].status is DocStatus.BCP


def test_cli_drift_exit_code_and_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "m.yaml"
    m.write_text(
        yaml.safe_dump(
            {
                "docs": [
                    {
                        "doc_id": "rfc1",
                        "title": "T",
                        "status": "standards_track",
                        "url": "https://example.test/rfc1.html",
                        "sha256": "0" * 64,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(cli, "_make_client", lambda: client())
    r = CliRunner().invoke(cli.app, ["fetch", "--manifest", str(m)])
    assert r.exit_code == 3
    assert "corpus unavailable: not caused by this change — rfc1: sha256 mismatch" in r.stderr
