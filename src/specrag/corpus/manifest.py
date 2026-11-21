"""Corpus manifest: the pinned list of specification documents."""

import os
import re
from pathlib import Path

import yaml
from pydantic import field_validator

from specrag.models import Doc

# src/specrag/corpus/manifest.py -> parents[3] is the repo root.
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "corpus" / "manifest.yaml"
_DOC_ID_RE = re.compile(r"[a-z0-9][a-z0-9.-]*")


class ManifestEntry(Doc):
    sha256: str | None = None

    @field_validator("doc_id")
    @classmethod
    def _safe_doc_id(cls, v: str) -> str:
        if not _DOC_ID_RE.fullmatch(v) or ".." in v:
            raise ValueError(f"invalid doc_id: {v!r}")
        return v


def corpus_cache_dir() -> Path:
    env = os.environ.get("SPECRAG_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "specrag"


def raw_path(doc_id: str) -> Path:
    base = (corpus_cache_dir() / "raw").resolve()
    path = (base / f"{doc_id}.html").resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"doc_id escapes the cache directory: {doc_id!r}")
    return path


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[ManifestEntry]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [ManifestEntry(**d) for d in data["docs"]]


def write_pins(path: Path, hashes: dict[str, str]) -> None:
    """Write sha256 values into the YAML, preserving entry order."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for d in data["docs"]:
        if d["doc_id"] in hashes:
            d["sha256"] = hashes[d["doc_id"]]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
