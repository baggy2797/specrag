"""Retriever: rank -> (Chunk, score) -> optional rerank -> ranking -> pack (re-review N5)."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from specrag.corpus.build import read_sections
from specrag.corpus.manifest import load_manifest
from specrag.models import Chunk, Doc, SectionIndex
from specrag.retrieval.bm25 import BM25Index
from specrag.retrieval.budget import Packed, pack
from specrag.retrieval.chunkers import CHUNKER_VERSION, ChunkerName, make_chunks
from specrag.retrieval.dense import DenseIndex, Embedder, SentenceTransformerEmbedder
from specrag.retrieval.hybrid import rrf
from specrag.retrieval.models_pin import RERANK_REVISION
from specrag.retrieval.rerank import RERANK_DEPTH, CrossEncoderReranker, Reranker, rerank
from specrag.retrieval.store import STORE_FILE, ChunkStore, corpus_key, index_dir
from specrag.retrieval.streams import DocStream, build_streams


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunker: ChunkerName = "section"
    retriever: Literal["bm25", "dense", "hybrid"] = "bm25"
    rerank: bool = False
    budget: int = 2500
    depth: int = 1000

    def config_hash(self) -> str:
        canon = json.dumps(json.loads(self.model_dump_json()), sort_keys=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    config: RetrievalConfig
    ranking: list[tuple[str, float]]
    packed: Packed
    provenance: dict[str, str]


def _load_docs() -> dict[str, Doc]:
    return {e.doc_id: e for e in load_manifest()}


class Retriever:
    def __init__(
        self,
        cfg: RetrievalConfig,
        chunks: list[Chunk],
        streams: dict[str, DocStream],
        key: str,
        bm25: BM25Index,
        dense: DenseIndex | None,
        reranker: Reranker | None,
    ) -> None:
        self.cfg = cfg
        self.chunks: dict[str, Chunk] = {c.id: c for c in chunks}
        self.streams = streams
        self.corpus_key = key
        self.bm25 = bm25
        self.dense = dense
        self._reranker = reranker

    @classmethod
    def open(
        cls,
        cfg: RetrievalConfig,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> "Retriever":
        key = corpus_key()
        sections = read_sections()
        streams = build_streams(sections)
        d = index_dir(cfg.chunker, key)
        store = ChunkStore(d / STORE_FILE)
        if store.exists():
            chunks = store.load()
        else:
            chunks = make_chunks(cfg.chunker, streams, SectionIndex(sections), _load_docs())
            store.save(chunks)
        dense = None
        if cfg.retriever in ("dense", "hybrid"):
            emb = embedder if embedder is not None else SentenceTransformerEmbedder()
            dense = DenseIndex.build(chunks, emb, d, cfg.chunker, key)
        return cls(cfg, chunks, streams, key, BM25Index.build(chunks), dense, reranker)

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _rank(self, query: str) -> list[tuple[str, float]]:
        n = self.cfg.depth
        if self.cfg.retriever == "bm25":
            return self.bm25.rank(query, n)
        assert self.dense is not None
        if self.cfg.retriever == "dense":
            return self.dense.rank(query, n)
        return rrf([self.bm25.rank(query, n), self.dense.rank(query, n)], n=n)

    def _provenance(self) -> dict[str, str]:
        p = {
            "corpus_key": self.corpus_key,
            "chunker_version": CHUNKER_VERSION,
            "config_hash": self.cfg.config_hash(),
            "dense_model": "n/a",  # no dense index took part (review I1)
            "dense_revision": "n/a",
        }
        if self.dense is not None:
            p["dense_model"] = self.dense.embedder.name
            p["dense_revision"] = self.dense.embedder.revision
            p["dense_truncated"] = str(self.dense.truncated_count)
        if self.cfg.rerank:
            p["rerank_revision"] = str(getattr(self._reranker, "revision", RERANK_REVISION))
            p["rerank_depth"] = str(RERANK_DEPTH)
        return p

    def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        scored = [(self.chunks[cid], s) for cid, s in self._rank(query)]
        if self.cfg.rerank:
            scored = rerank(query, scored, self.reranker, RERANK_DEPTH)
        ranking = [(c.id, s) for c, s in scored]
        packed = pack([c for c, _ in scored], self.streams, self.cfg.budget)
        return RetrievalResult(
            query=query,
            config=self.cfg,
            ranking=ranking,
            packed=packed,
            provenance=self._provenance(),
        )
