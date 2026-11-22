"""CPU dense retrieval: bge-small embeddings, an on-disk embedding cache, cosine ranking."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from specrag.models import Chunk
from specrag.retrieval.chunkers import CHUNKER_VERSION, ChunkerName
from specrag.retrieval.models_pin import DENSE_MODEL, DENSE_REVISION, resolve_pinned
from specrag.retrieval.ranking import sort_ranking

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQ_TOKENS = 512
BATCH_SIZE = 32


class Embedder(Protocol):
    name: str
    revision: str
    max_seq_tokens: int

    def embed(self, texts: list[str], *, query: bool) -> np.ndarray:
        """L2-normalised float32 vectors, shape (n, d)."""
        ...

    def truncated(self, texts: list[str]) -> int:
        """How many texts exceed the model's own token limit."""
        ...


def _normalise(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(out / norms, dtype=np.float32)


class SentenceTransformerEmbedder:
    """bge-small-en-v1.5 on CPU. The model loads lazily; ``model`` is injectable for tests."""

    def __init__(self, model: Any = None) -> None:
        self.name = DENSE_MODEL
        self.revision = DENSE_REVISION
        self.max_seq_tokens = MAX_SEQ_TOKENS
        self._model = model

    def _get(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                resolve_pinned(DENSE_MODEL, DENSE_REVISION), device="cpu"
            )
        self._model.max_seq_length = self.max_seq_tokens
        return self._model

    def warm_up(self) -> None:
        """Load the model now (downloads on first use)."""
        self._get()

    def embed(self, texts: list[str], *, query: bool) -> np.ndarray:
        model = self._get()
        inputs = [QUERY_INSTRUCTION + t for t in texts] if query else list(texts)
        vecs = model.encode(
            inputs,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalise(vecs)

    def truncated(self, texts: list[str]) -> int:
        if not texts:
            return 0
        enc = self._get().tokenizer(texts, add_special_tokens=True, truncation=False)
        return sum(1 for ids in enc["input_ids"] if len(ids) > self.max_seq_tokens)


def embedding_path(
    cache_dir: Path, corpus_key: str, chunker: str, model: str, revision: str
) -> Path:
    raw = "|".join([corpus_key, chunker, CHUNKER_VERSION, model, revision])
    return cache_dir / f"emb-{hashlib.sha256(raw.encode()).hexdigest()[:16]}.npy"


def _sidecar(npy: Path) -> Path:
    return npy.with_suffix(".json")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _atomic_write(path: Path, write: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_cache(npy: Path, ids: list[str], hashes: list[str]) -> tuple[np.ndarray, int] | None:
    side = _sidecar(npy)
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
        if meta["ids"] != ids or meta["sha256"] != hashes:
            return None
        matrix = np.load(npy, allow_pickle=False)
        truncated = int(meta["truncated"])
        dim = int(meta["dim"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape != (len(ids), dim):
        return None
    if dim <= 0 or not np.isfinite(matrix).all():
        return None
    return matrix, truncated


def _store_cache(
    npy: Path, matrix: np.ndarray, ids: list[str], hashes: list[str], truncated: int
) -> None:
    npy.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(npy, lambda f: np.save(f, matrix, allow_pickle=False))
    meta = {"ids": ids, "sha256": hashes, "truncated": truncated, "dim": int(matrix.shape[1])}
    _atomic_write(_sidecar(npy), lambda f: f.write(json.dumps(meta).encode("utf-8")))


class DenseIndex:
    def __init__(
        self, ids: list[str], matrix: np.ndarray, embedder: Embedder, truncated_count: int
    ) -> None:
        self.ids = ids
        self.matrix = matrix
        self.embedder = embedder
        self.truncated_count = truncated_count

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        cache_dir: Path | None,
        chunker: ChunkerName,
        corpus_key: str,
    ) -> "DenseIndex":
        ids = [c.id for c in chunks]
        texts = [c.index_text for c in chunks]
        hashes = [_text_hash(t) for t in texts]
        npy = (
            None
            if cache_dir is None
            else embedding_path(cache_dir, corpus_key, chunker, embedder.name, embedder.revision)
        )
        if npy is not None:
            hit = _load_cache(npy, ids, hashes)
            if hit is not None:
                return cls(ids, hit[0], embedder, hit[1])
        matrix = _normalise(embedder.embed(texts, query=False))
        truncated = embedder.truncated(texts)
        if npy is not None:
            _store_cache(npy, matrix, ids, hashes, truncated)
        return cls(ids, matrix, embedder, truncated)

    def rank(self, query: str, n: int = 1000) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = _normalise(self.embedder.embed([query], query=True))[0]
        scores = self.matrix @ q
        return sort_ranking({cid: float(s) for cid, s in zip(self.ids, scores, strict=True)})[:n]
