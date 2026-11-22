import math

import pytest

from specrag.corpus.build import read_sections
from specrag.corpus.manifest import load_manifest
from specrag.models import Chunk, Doc, SectionIndex
from specrag.retrieval.bm25 import BM25Index, tokenize
from specrag.retrieval.chunkers import make_chunks
from specrag.retrieval.streams import build_streams


def mk(id: str, text: str, index_text: str | None = None) -> Chunk:
    n = len(text.split())
    return Chunk(
        id=id,
        doc_id="d",
        text=text,
        index_text=text if index_text is None else index_text,
        sections=["d#1"],
        tokens=[n],
        n_tokens=n,
        span=(0, n),
    )


def test_tokenize_keeps_identifiers() -> None:
    assert tokenize("The code_verifier MUST use client-id") == [
        "the",
        "code_verifier",
        "must",
        "use",
        "client-id",
    ]


def test_bm25_prefers_rare_term() -> None:
    idx = BM25Index.build(
        [
            mk("c1", "token endpoint request"),
            mk("c2", "token endpoint response"),
            mk("c3", "token endpoint pkce"),
        ]
    )
    assert idx.rank("token pkce")[0][0] == "c3"


def test_bm25_indexes_index_text_not_text() -> None:
    idx = BM25Index.build([mk("c1", "body", index_text="Heading prefix\nbody")])
    assert [cid for cid, _ in idx.rank("heading")] == ["c1"]


def test_bm25_hand_computed_score() -> None:
    idx = BM25Index.build([mk("a", "alpha beta beta"), mk("b", "gamma")], k1=1.2, b=0.75)
    ((cid, score),) = idx.rank("beta")
    assert cid == "a"
    # N = 2 chunks, df("beta") = 1  ->  idf = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(2)
    idf = math.log(2.0)
    # avgdl = (3 + 1) / 2 = 2; dl = 3; tf = 2
    # norm = k1 * (1 - b + b * dl / avgdl) = 1.2 * (0.25 + 0.75 * 1.5) = 1.2 * 1.375 = 1.65
    # tf part = tf * (k1 + 1) / (tf + norm) = 2 * 2.2 / (2 + 1.65) = 4.4 / 3.65
    expected = idf * 4.4 / 3.65
    assert score == pytest.approx(expected, abs=1e-9)


def test_bm25_repeated_query_terms_count_once() -> None:
    idx = BM25Index.build([mk("a", "alpha beta"), mk("b", "gamma")])
    assert idx.rank("beta beta beta") == idx.rank("beta")


def test_bm25_no_match_returns_empty() -> None:
    idx = BM25Index.build([mk("a", "alpha beta")])
    assert idx.rank("zzz qqq") == []
    assert idx.rank("") == []


def test_bm25_ties_break_by_chunk_id() -> None:
    idx = BM25Index.build([mk("c3", "same words"), mk("c1", "same words"), mk("c2", "same words")])
    assert [cid for cid, _ in idx.rank("same")] == ["c1", "c2", "c3"]


def test_bm25_rank_truncates_to_n() -> None:
    idx = BM25Index.build([mk(f"c{i}", "same") for i in range(5)])
    assert len(idx.rank("same", n=2)) == 2


@pytest.mark.corpus
def test_bm25_pkce_smoke() -> None:
    secs = read_sections()
    docs: dict[str, Doc] = {e.doc_id: e for e in load_manifest()}
    chunks = make_chunks("section", build_streams(secs), SectionIndex(secs), docs)  # type: ignore[arg-type]
    top = BM25Index.build(chunks).rank("code_verifier code_challenge S256", n=5)
    by_id = {c.id: c for c in chunks}
    wanted = {"rfc7636#4.1", "rfc7636#4.2"}
    assert any(wanted & set(by_id[cid].sections) for cid, _ in top), top


_SEED_SCRIPT = """
import sys
from specrag.models import Chunk
from specrag.retrieval.bm25 import BM25Index

def mk(i):
    words = [f"w{(i * 7 + j * 3) % 23}" for j in range(40)]
    text = " ".join(words)
    return Chunk(id=f"c{i:03d}", doc_id="d", text=text, index_text=text,
                 sections=["d#1"], tokens=[40], n_tokens=40, span=(0, 40))

idx = BM25Index.build([mk(i) for i in range(60)])
q = " ".join(f"w{k}" for k in range(23))
print([(cid, repr(s)) for cid, s in idx.rank(q)])
"""


def test_bm25_rank_is_hash_seed_independent() -> None:
    import os
    import subprocess
    import sys

    outs = set()
    for seed in ("1", "2", "3", "4", "5"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, "-c", _SEED_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outs.add(r.stdout)
    assert len(outs) == 1


def test_bm25_sums_terms_in_sorted_order() -> None:
    chunks = [mk(f"c{i}", " ".join(f"t{(i + j) % 9}" for j in range(5))) for i in range(12)]
    idx = BM25Index.build(chunks)
    query = "t8 t1 t5 t0 t3 t7 t2 t6 t4"
    expected: dict[str, float] = {}
    for term in sorted(set(tokenize(query))):
        posting = idx.postings[term]
        df = len(posting)
        idf = math.log(1 + (idx.n_docs - df + 0.5) / (df + 0.5))
        for cid, tf in posting:
            norm = idx.k1 * (1 - idx.b + idx.b * idx.doc_len[cid] / idx.avgdl)
            expected[cid] = expected.get(cid, 0.0) + idf * tf * (idx.k1 + 1) / (tf + norm)
    assert dict(idx.rank(query)) == expected
