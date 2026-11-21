"""Download and verify the corpus into the local cache."""

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel

from specrag.corpus.manifest import ManifestEntry, corpus_cache_dir, raw_path, write_pins

MAX_BYTES = 20 * 1024 * 1024


class FetchError(Exception):
    def __init__(self, doc_id: str, reason: str) -> None:
        super().__init__(f"{doc_id}: {reason}")
        self.doc_id = doc_id
        self.reason = reason


class FetchReport(BaseModel):
    fetched: list[str] = []
    skipped: list[str] = []
    unpinned: list[str] = []


def _download(
    entry: ManifestEntry,
    client: httpx.Client,
    retries: int,
    sleep: Callable[[float], None],
) -> bytes:
    reason = "no attempts made"
    for attempt in range(max(retries, 1)):
        if attempt:
            sleep(float(2 ** (attempt - 1)))
        try:
            with client.stream("GET", entry.url, follow_redirects=True) as resp:
                if resp.status_code >= 500:
                    reason = f"HTTP {resp.status_code}"
                    continue
                if resp.status_code >= 400:
                    raise FetchError(entry.doc_id, f"HTTP {resp.status_code}")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > MAX_BYTES:
                        raise FetchError(entry.doc_id, "response exceeds 20 MB")
                return bytes(buf)
        except httpx.TransportError as e:
            reason = f"network error: {e}"
            continue
    raise FetchError(entry.doc_id, reason)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def fetch_all(
    entries: Sequence[ManifestEntry],
    *,
    client: httpx.Client,
    pin: bool = False,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    manifest_path: Path | None = None,
) -> FetchReport:
    report = FetchReport()
    hashes: dict[str, str] = {}
    for e in entries:
        path = raw_path(e.doc_id)
        if e.sha256 and path.exists():
            cached = hashlib.sha256(path.read_bytes()).hexdigest()
            if cached == e.sha256:
                report.skipped.append(e.doc_id)
                hashes[e.doc_id] = cached
                continue
        data = _download(e, client, retries, sleep)
        digest = hashlib.sha256(data).hexdigest()
        if e.sha256 and digest != e.sha256:
            raise FetchError(e.doc_id, f"sha256 mismatch (expected {e.sha256}, got {digest})")
        _write_atomic(path, data)
        hashes[e.doc_id] = digest
        report.fetched.append(e.doc_id)
        if not e.sha256:
            report.unpinned.append(e.doc_id)
    lock = corpus_cache_dir() / "manifest.lock.json"
    _write_atomic(lock, json.dumps(hashes, indent=2, sort_keys=True).encode())
    if pin and manifest_path is not None:
        write_pins(manifest_path, hashes)
        report.unpinned = []
    return report
