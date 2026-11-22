import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol, cast

import httpx
import typer
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from specrag import __version__
from specrag.corpus.build import build_sections, read_sections
from specrag.corpus.fetch import FetchError, fetch_all
from specrag.corpus.manifest import DEFAULT_MANIFEST, load_manifest
from specrag.models import Section
from specrag.retrieval.chunkers import ChunkerName
from specrag.retrieval.dense import SentenceTransformerEmbedder
from specrag.retrieval.models_pin import (
    DENSE_MODEL,
    DENSE_REVISION,
    RERANK_MODEL,
    RERANK_REVISION,
)
from specrag.retrieval.pipeline import RetrievalConfig, Retriever
from specrag.retrieval.rerank import CrossEncoderReranker
from specrag.retrieval.store import corpus_key

USER_AGENT = "specrag/0.1 (+https://github.com/baggy2797/specrag)"

app = typer.Typer(help="specrag: RAG and evaluation over technical specifications.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """specrag command line interface."""


def _make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0)


@app.command()
def fetch(
    pin: Annotated[
        bool, typer.Option("--pin", help="Write sha256 hashes into the manifest.")
    ] = False,
    manifest: Annotated[
        Path, typer.Option("--manifest", help="Manifest YAML path.")
    ] = DEFAULT_MANIFEST,
) -> None:
    """Download the corpus into the local cache."""
    entries = load_manifest(manifest)
    try:
        with _make_client() as client:
            report = fetch_all(entries, client=client, pin=pin, manifest_path=manifest)
    except FetchError as e:
        typer.echo(f"corpus unavailable: not caused by this change — {e}", err=True)
        raise typer.Exit(3) from e
    typer.echo(f"fetched {len(report.fetched)}, cached {len(report.skipped)}")
    if report.unpinned:
        typer.echo(f"unpinned: {', '.join(report.unpinned)}")


@app.command()
def parse(
    manifest: Annotated[
        Path, typer.Option("--manifest", help="Manifest YAML path.")
    ] = DEFAULT_MANIFEST,
) -> None:
    """Parse every cached document and write sections.jsonl."""
    entries = load_manifest(manifest)
    try:
        build_sections(entries)
    except FileNotFoundError as e:
        typer.echo(f"{e} (run `specrag fetch` first)", err=True)
        raise typer.Exit(3) from e
    by_doc: dict[str, list[Section]] = {e.doc_id: [] for e in entries}
    for s in read_sections():
        by_doc[s.doc_id].append(s)
    typer.echo(f"{'doc_id':<22}{'sections':>9}{'own_tokens':>12}")
    for doc_id, secs in by_doc.items():
        typer.echo(f"{doc_id:<22}{len(secs):>9}{sum(s.own_tokens for s in secs):>12}")
    total_s = sum(len(v) for v in by_doc.values())
    total_t = sum(s.own_tokens for v in by_doc.values() for s in v)
    typer.echo(f"{'TOTAL':<22}{total_s:>9}{total_t:>12}")


@app.command()
def sections(doc_id: str) -> None:
    """Print a document's section tree: id, title, own tokens."""
    try:
        secs = [s for s in read_sections() if s.doc_id == doc_id]
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(3) from e
    if not secs:
        typer.echo(f"unknown doc_id: {doc_id}", err=True)
        raise typer.Exit(2)
    for s in secs:
        typer.echo(f"{'  ' * s.depth}{s.id}  {s.title}  ({s.own_tokens})")


class _RetrieverKind(StrEnum):
    bm25 = "bm25"
    dense = "dense"
    hybrid = "hybrid"


class _ChunkerChoice(StrEnum):
    fixed = "fixed"
    section = "section"
    section_meta = "section+meta"


class _IndexChunkerChoice(StrEnum):
    fixed = "fixed"
    section = "section"
    section_meta = "section+meta"
    all = "all"


_LOADING = "loading {id} (first run downloads ~130 MB for dense, ~1.1 GB for the reranker) …"


class _Warmable(Protocol):
    def warm_up(self) -> None: ...


def _preflight_corpus() -> None:
    """Phase A: the corpus must be parsed before anything else runs."""
    try:
        corpus_key()
        read_sections()
    except FileNotFoundError as e:
        typer.echo(
            "corpus unavailable: not caused by this change — "
            "run `specrag fetch` and `specrag parse`",
            err=True,
        )
        raise typer.Exit(3) from e


def _make_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def _make_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def _warm(model: _Warmable, model_id: str, revision: str) -> None:
    """Phase B: load one model; only load failures map to "model unavailable"."""
    typer.echo(_LOADING.format(id=model_id), err=True)
    try:
        model.warm_up()
    except (LocalEntryNotFoundError, HfHubHTTPError, OSError) as e:
        typer.echo(
            f"model unavailable: {model_id}@{revision} — "
            "run once with network access or populate the HF cache",
            err=True,
        )
        raise typer.Exit(3) from e


def _open_retriever(
    cfg: RetrievalConfig,
    embedder: SentenceTransformerEmbedder | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> Retriever:
    """Phase C seam for tests: build the Retriever. No error mapping here."""
    return Retriever.open(cfg, embedder=embedder, reranker=reranker)


def _prepare(cfg: RetrievalConfig) -> Retriever:
    _preflight_corpus()
    embedder = reranker = None
    if cfg.retriever != "bm25":
        embedder = _make_embedder()
        _warm(embedder, DENSE_MODEL, DENSE_REVISION)
    if cfg.rerank:
        reranker = _make_reranker()
        _warm(reranker, RERANK_MODEL, RERANK_REVISION)
    return _open_retriever(cfg, embedder, reranker)


@app.command()
def index(
    chunker: Annotated[
        _IndexChunkerChoice, typer.Option("--chunker", help="Chunker to build.")
    ] = _IndexChunkerChoice.all,
    dense: Annotated[
        bool, typer.Option("--dense", help="Also build the dense embedding index.")
    ] = False,
) -> None:
    """Build the chunk store (and optionally the dense index) for a chunker."""
    names: list[ChunkerName] = (
        ["fixed", "section", "section+meta"]
        if chunker is _IndexChunkerChoice.all
        else [cast(ChunkerName, chunker.value)]
    )
    for name in names:
        cfg = RetrievalConfig(chunker=name, retriever="dense" if dense else "bm25")
        t0 = time.perf_counter()
        r = _prepare(cfg)
        line = f"{name}: {len(r.chunks)} chunks in {time.perf_counter() - t0:.1f}s"
        if r.dense is not None:
            line += f", dense truncated: {r.dense.truncated_count}"
        typer.echo(line)


@app.command()
def search(
    query: str,
    chunker: Annotated[_ChunkerChoice, typer.Option("--chunker")] = _ChunkerChoice.section,
    retriever: Annotated[_RetrieverKind, typer.Option("--retriever")] = _RetrieverKind.bm25,
    rerank: Annotated[bool, typer.Option("--rerank", help="Cross-encoder rerank.")] = False,
    budget: Annotated[int, typer.Option("--budget", help="Token budget.")] = 2500,
    as_json: Annotated[bool, typer.Option("--json", help="Print the RetrievalResult.")] = False,
) -> None:
    """Retrieve and pack chunks for QUERY within a token budget."""
    if not query.strip():
        typer.echo("query must not be empty", err=True)
        raise typer.Exit(2)
    cfg = RetrievalConfig(
        chunker=chunker.value,
        retriever=retriever.value,
        rerank=rerank,
        budget=budget,
    )
    res = _prepare(cfg).retrieve(query)
    if as_json:
        typer.echo(res.model_dump_json(indent=2))
        return
    packed = res.packed
    if not packed.chunks:
        typer.echo("no results")
        return
    for i, (c, cost) in enumerate(zip(packed.chunks, packed.costs, strict=True), start=1):
        typer.echo(f"{i}. {c.id}  {','.join(c.sections)}  cost={cost}")
    typer.echo(f"tokens used: {packed.tokens_used}")
    typer.echo(f"config: {cfg.config_hash()}")
