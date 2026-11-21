"""Local-only report: embed every chunk for each chunker; print wall time and truncated_count.

Not a test (CI never embeds the whole corpus). Run:
    SPECRAG_MODELS=1 uv run python scripts/embed_report.py
"""

import os
import sys
import time
from typing import get_args

from specrag.corpus.build import read_sections
from specrag.corpus.manifest import load_manifest
from specrag.models import Doc, SectionIndex
from specrag.retrieval.chunkers import ChunkerName, make_chunks
from specrag.retrieval.dense import DenseIndex, SentenceTransformerEmbedder
from specrag.retrieval.streams import build_streams


def main() -> int:
    if os.environ.get("SPECRAG_MODELS") != "1":
        print("set SPECRAG_MODELS=1 to run this report", file=sys.stderr)
        return 2
    sections = read_sections()
    streams = build_streams(sections)
    docs: dict[str, Doc] = {e.doc_id: e for e in load_manifest()}
    idx = SectionIndex(sections)
    embedder = SentenceTransformerEmbedder()
    print(f"{'chunker':<14}{'chunks':>8}{'truncated':>11}{'pct':>8}{'seconds':>10}")
    for name in get_args(ChunkerName):
        chunks = make_chunks(name, streams, idx, docs)
        start = time.perf_counter()
        index = DenseIndex.build(chunks, embedder, None, chunker=name, corpus_key="report")
        wall = time.perf_counter() - start
        pct = 100 * index.truncated_count / max(len(chunks), 1)
        print(f"{name:<14}{len(chunks):>8}{index.truncated_count:>11}{pct:>7.1f}%{wall:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
