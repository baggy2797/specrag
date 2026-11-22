"""SQLite chunk store plus the corpus/index keys (review C2)."""

import hashlib
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from specrag.corpus.build import SECTIONS_FILE
from specrag.corpus.manifest import DEFAULT_MANIFEST, corpus_cache_dir
from specrag.models import Chunk
from specrag.retrieval.chunkers import CHUNKER_VERSION, ChunkerName

STORE_FILE = "chunks.sqlite"


def corpus_key() -> str:
    """sha256(manifest bytes + sections.jsonl bytes)[:12]: re-pinning or re-parsing invalidates."""
    sections = corpus_cache_dir() / SECTIONS_FILE
    if not sections.exists():
        raise FileNotFoundError(f"{sections} not found; run `specrag parse` first")
    h = hashlib.sha256()
    h.update(DEFAULT_MANIFEST.read_bytes())
    h.update(sections.read_bytes())
    return h.hexdigest()[:12]


def index_dir(chunker: ChunkerName, key: str | None = None) -> Path:
    key = corpus_key() if key is None else key
    return corpus_cache_dir() / "index" / f"{chunker}-v{CHUNKER_VERSION}-{key}"


class ChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, chunks: list[Chunk]) -> None:
        """Write atomically: build a temp database, then os.replace it into place."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".chunks-", suffix=".tmp")
        os.close(fd)
        try:
            with closing(sqlite3.connect(tmp)) as db:
                db.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, ord INTEGER, json TEXT)")
                db.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?)",
                    [(c.id, i, c.model_dump_json()) for i, c in enumerate(chunks)],
                )
                db.commit()
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load(self) -> list[Chunk]:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT json FROM chunks ORDER BY ord").fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def get_many(self, ids: list[str]) -> list[Chunk]:
        """Chunks for ``ids`` in the input order; an unknown ID raises KeyError."""
        if not ids:
            return []
        with closing(sqlite3.connect(self.path)) as db:
            marks = ",".join("?" * len(ids))
            rows = db.execute(f"SELECT id, json FROM chunks WHERE id IN ({marks})", ids).fetchall()
        by_id = {i: Chunk.model_validate_json(j) for i, j in rows}
        return [by_id[i] for i in ids]
