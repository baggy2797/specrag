import math

import pytest

from specrag.models import Chunk, Section, SectionIndex
from specrag.scoring import (
    chunk_hits,
    condensed,
    hit_rate_precision,
    hit_set,
    hit_threshold,
    mrr,
    ndcg_at,
    recall,
)

DOC = "rfc6749"


def _sec(number: str, own: int) -> Section:
    parts = number.split(".")
    parent_id = f"{DOC}#{'.'.join(parts[:-1])}" if len(parts) > 1 else None
    return Section(
        id=f"{DOC}#{number}",
        doc_id=DOC,
        number=number,
        title=f"S{number}",
        parent_id=parent_id,
        depth=len(parts),
        own_text="x",
        own_tokens=own,
    )


def mk(sections: list[str], tokens: list[int]) -> Chunk:
    return Chunk(
        id="c",
        doc_id=DOC,
        text="t",
        index_text="t",
        sections=sections,
        tokens=tokens,
        n_tokens=sum(tokens),
    )


@pytest.fixture
def idx() -> SectionIndex:
    return SectionIndex(
        [_sec("4", 10), _sec("4.1", 0), _sec("4.1.1", 200), _sec("4.1.2", 40), _sec("4.10", 120)]
    )


@pytest.fixture
def idx2() -> SectionIndex:
    return SectionIndex(
        [_sec("4", 200), _sec("4.1", 0), _sec("4.1.1", 200), _sec("4.1.2", 40), _sec("4.10", 120)]
    )


def test_hit_threshold() -> None:
    assert hit_threshold(200) == 50
    assert hit_threshold(40) == pytest.approx(12)
    assert hit_threshold(0) == 0


def test_direct_hit_needs_threshold(idx: SectionIndex) -> None:
    assert chunk_hits(mk(["rfc6749#4.1.1"], [60]), "rfc6749#4.1.1", idx)
    assert not chunk_hits(mk(["rfc6749#4.1.1"], [49]), "rfc6749#4.1.1", idx)


def test_small_section_threshold_is_30pct(idx: SectionIndex) -> None:
    assert chunk_hits(mk(["rfc6749#4.1.2"], [12]), "rfc6749#4.1.2", idx)
    assert not chunk_hits(mk(["rfc6749#4.1.2"], [11]), "rfc6749#4.1.2", idx)


def test_heading_only_gold_credited_via_descendant(idx: SectionIndex) -> None:
    assert chunk_hits(mk(["rfc6749#4.1.1"], [80]), "rfc6749#4.1", idx)
    assert not chunk_hits(mk(["rfc6749#4.1.1"], [10]), "rfc6749#4.1", idx)


def test_parent_never_hits_child(idx: SectionIndex) -> None:
    assert not chunk_hits(mk(["rfc6749#4.1"], [0]), "rfc6749#4.1.1", idx)


def test_sibling_prefix_is_not_descendant(idx: SectionIndex) -> None:
    assert not chunk_hits(mk(["rfc6749#4.10"], [120]), "rfc6749#4.1", idx)


def test_non_heading_gold_gets_no_descendant_credit(idx2: SectionIndex) -> None:
    assert not chunk_hits(mk(["rfc6749#4.1.1"], [80]), "rfc6749#4", idx2)


def test_gold_counted_once_and_unanswerable_excluded(idx: SectionIndex) -> None:
    r = [mk(["rfc6749#4.1.1"], [100]), mk(["rfc6749#4.1.1"], [100])]
    assert recall(r, {"rfc6749#4.1.1", "rfc6749#4.1.2"}, idx) == 0.5
    assert recall(r, set(), idx) is None


def test_chunk_hits_two_gold_sections(idx: SectionIndex) -> None:
    c = mk(["rfc6749#4.1.1", "rfc6749#4.1.2"], [100, 40])
    gold = {"rfc6749#4.1.1", "rfc6749#4.1.2"}
    assert hit_set(c, gold, idx) == gold
    assert recall([c], gold, idx) == 1.0


def test_recall_ignored_shrinks_gold(idx: SectionIndex) -> None:
    r = [mk(["rfc6749#4.1.1"], [100])]
    gold = {"rfc6749#4.1.1", "rfc6749#4.1.2"}
    assert recall(r, gold, idx, ignored={"rfc6749#4.1.2"}) == 1.0
    assert recall(r, gold, idx, ignored=gold) is None


def test_ndcg_binary_dedup(idx: SectionIndex) -> None:
    r = [mk(["rfc6749#4.10"], [120]), mk(["rfc6749#4.1.1"], [100]), mk(["rfc6749#4.1.1"], [100])]
    assert ndcg_at(r, {"rfc6749#4.1.1"}, idx, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_perfect_and_partial_and_none(idx: SectionIndex) -> None:
    a, b = "rfc6749#4.1.1", "rfc6749#4.1.2"
    r = [mk([a], [100]), mk([b], [40])]
    assert ndcg_at(r, {a, b}, idx) == pytest.approx(1.0)
    # only one of two gold found at rank 2: DCG=1/log2(3); IDCG=1+1/log2(3)
    r2 = [mk(["rfc6749#4.10"], [120]), mk([a], [100])]
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at(r2, {a, b}, idx) == pytest.approx(expected)
    assert ndcg_at(r, set(), idx) is None


def test_ndcg_respects_k(idx: SectionIndex) -> None:
    r = [mk(["rfc6749#4.10"], [120]), mk(["rfc6749#4.1.1"], [100])]
    assert ndcg_at(r, {"rfc6749#4.1.1"}, idx, k=1) == 0.0


def test_ignored_removed_before_scoring(idx: SectionIndex) -> None:
    r = [mk(["rfc6749#4.1.2"], [40]), mk(["rfc6749#4.1.1"], [100])]
    assert mrr(r, {"rfc6749#4.1.1", "rfc6749#4.1.2"}, idx, ignored={"rfc6749#4.1.2"}) == 1.0


def test_mrr_rank_none_and_zero(idx: SectionIndex) -> None:
    r = [mk(["rfc6749#4.10"], [120]), mk(["rfc6749#4.1.1"], [100])]
    assert mrr(r, {"rfc6749#4.1.1"}, idx) == 0.5
    assert mrr(r, {"rfc6749#4.1.2"}, idx) == 0.0
    assert mrr(r, set(), idx) is None


def test_chunk_with_ignored_and_real_hit_is_kept(idx: SectionIndex) -> None:
    a, b = "rfc6749#4.1.1", "rfc6749#4.1.2"
    r = [mk([a, b], [100, 40])]
    assert mrr(r, {a, b}, idx, ignored={b}) == 1.0
    assert hit_rate_precision(r, {a, b}, idx, ignored={b}) == 1.0


def test_hit_rate_precision(idx: SectionIndex) -> None:
    a = "rfc6749#4.1.1"
    r = [mk([a], [100]), mk(["rfc6749#4.10"], [120]), mk([a], [100]), mk(["rfc6749#4.10"], [1])]
    assert hit_rate_precision(r, {a}, idx) == 0.5
    assert hit_rate_precision(r, set(), idx) is None
    only_ignored = [mk(["rfc6749#4.1.2"], [40])]
    assert (
        hit_rate_precision(only_ignored, {a, "rfc6749#4.1.2"}, idx, ignored={"rfc6749#4.1.2"})
        is None
    )


def test_condensed(idx: SectionIndex) -> None:
    keep = mk(["rfc6749#4.1.1", "rfc6749#4.10"], [100, 5])
    drop = mk(["rfc6749#4.10"], [120])
    assert condensed([drop, keep], {"rfc6749#4.1.1"}, idx) == [keep]


def test_zero_token_entry_never_hits(idx: SectionIndex) -> None:
    assert not chunk_hits(mk(["rfc6749#4.1"], [0]), "rfc6749#4.1", idx)
    assert chunk_hits(mk(["rfc6749#4.1"], [1]), "rfc6749#4.1", idx)


def test_zero_token_descendant_entry_never_hits(idx: SectionIndex) -> None:
    assert not chunk_hits(mk(["rfc6749#4.1"], [0]), "rfc6749#4", idx)
    assert chunk_hits(mk(["rfc6749#4.1"], [1]), "rfc6749#4", idx)


def _idx12() -> tuple[SectionIndex, list[str]]:
    secs = [_sec(str(n), 200) for n in range(1, 13)]
    return SectionIndex(secs), [s.id for s in secs]


def test_ndcg_idcg_capped_at_k() -> None:
    idx12, ids = _idx12()
    gold = set(ids)
    full = [mk([i], [60]) for i in ids[:10]]
    assert ndcg_at(full, gold, idx12, k=10) == pytest.approx(1.0)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, 11))
    first_only = [mk([ids[0]], [60])]
    assert ndcg_at(first_only, gold, idx12, k=10) == pytest.approx(1 / idcg)


def test_empty_ranking(idx: SectionIndex) -> None:
    g = {"rfc6749#4.1.1"}
    assert recall([], g, idx) == 0.0
    assert mrr([], g, idx) == 0.0
    assert ndcg_at([], g, idx) == 0.0
    assert hit_rate_precision([], g, idx) is None


def test_chunk_hitting_two_new_gold_counts_once_in_ndcg(idx: SectionIndex) -> None:
    a, b = "rfc6749#4.1.1", "rfc6749#4.1.2"
    c = mk([a, b], [100, 40])
    assert ndcg_at([c], {a, b}, idx) == pytest.approx(1 / (1 + 1 / math.log2(3)))
