"""Download and verify the corpus into the local cache."""

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel

from specrag.corpus.manifest import ManifestEntry, corpus_cache_dir, raw_path, write_pins

MAX_BYTES = 20 * 1024 * 1024


_CF_MARKER = b"__CF$cv$params"
_OPEN_TAG = re.compile(rb"<script(?:[\s/][^<>]{0,2048})?>", re.IGNORECASE)
# Alternation order matters: a well-formed closer wins over a bare ``</script``.
_TOKEN = re.compile(rb"<script|</script\s{0,32}>|</script|" + re.escape(_CF_MARKER), re.IGNORECASE)


def normalize(data: bytes) -> bytes:
    """Remove Cloudflare's injected challenge-platform beacon scripts.

    Cloudflare inserts a per-request ``<script>`` block containing
    ``window.__CF$cv$params={r:'<random>',t:'<ts>'}`` into some pages, so the
    same document hashes differently on every fetch. Hashing and storing the
    bytes with that block removed makes the pin stable. Removal is anchored on
    the ``__CF$cv$params`` marker: a block is cut only when a well-formed
    ``<script ...>`` opening tag, the marker and a ``</script>`` closer occur in
    that order with no other script tag boundary between them. Otherwise the
    data is left unchanged. Every other byte, including ordinary scripts, is
    untouched. A ``<script`` seen while a block is open (an unterminated earlier
    tag, or one inside a string) restarts the block, so for
    ``<script>var s='<script>';MARK</script>`` the outer prefix
    ``<script>var s='`` is kept and only the inner block is removed. A single
    forward pass over the tokens, so linear time.
    """
    out: list[bytes] = []
    pos = 0
    open_start = -1  # start of the current ``<script`` candidate, or -1
    open_end = -1  # end of its validated open tag, or -1 if not yet validated
    seen = False  # marker seen inside the current block
    for m in _TOKEN.finditer(data):
        tok = m.group(0)
        if tok == _CF_MARKER:
            if open_start < 0 or seen:
                continue
            if open_end < 0:  # validate the open tag lazily, once per block
                tag = _OPEN_TAG.match(data, open_start)
                if tag is None:
                    open_start = -1
                    continue
                open_end = tag.end()
            seen = m.start() >= open_end
        elif tok[:2] == b"</":
            if seen and tok.endswith(b">"):
                out.append(data[pos:open_start])
                pos = m.end()
            open_start, open_end, seen = -1, -1, False
        else:
            # Any new ``<script`` (including an unterminated earlier one) restarts.
            open_start, open_end, seen = m.start(), -1, False
    if not out:
        return data
    out.append(data[pos:])
    return b"".join(out)


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
            raw = path.read_bytes()
            clean = normalize(raw)
            cached = hashlib.sha256(clean).hexdigest()
            if cached == e.sha256:
                if clean != raw:
                    _write_atomic(path, clean)
                report.skipped.append(e.doc_id)
                hashes[e.doc_id] = cached
                continue
        data = normalize(_download(e, client, retries, sleep))
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
