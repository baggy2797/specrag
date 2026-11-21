from pathlib import Path
from typing import Annotated

import httpx
import typer

from specrag import __version__
from specrag.corpus.build import build_sections, read_sections
from specrag.corpus.fetch import FetchError, fetch_all
from specrag.corpus.manifest import DEFAULT_MANIFEST, load_manifest
from specrag.models import Section

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
