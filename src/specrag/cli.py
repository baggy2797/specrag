from pathlib import Path
from typing import Annotated

import httpx
import typer

from specrag import __version__
from specrag.corpus.fetch import FetchError, fetch_all
from specrag.corpus.manifest import DEFAULT_MANIFEST, load_manifest

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
