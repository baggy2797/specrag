import pytest

from specrag.retrieval.hybrid import rrf


def test_rrf_hand_computed() -> None:
    a = [("A", 9.0), ("B", 5.0), ("C", 1.0)]
    b = [("B", 0.9), ("D", 0.5), ("A", 0.1)]
    out = dict(rrf([a, b]))
    assert out["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert out["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert [cid for cid, _ in rrf([a, b])][0] == "B"


def test_rrf_missing_contributes_zero() -> None:
    out = dict(rrf([[("A", 1.0)], [("B", 1.0)]]))
    assert out == {"A": pytest.approx(1 / 61), "B": pytest.approx(1 / 61)}
    assert dict(rrf([[("A", 1.0), ("B", 0.5)]]))["B"] == pytest.approx(1 / 62)


def test_rrf_ties_break_by_chunk_id() -> None:
    out = rrf([[("z", 1.0)], [("a", 1.0)], [("m", 1.0)]])
    assert [cid for cid, _ in out] == ["a", "m", "z"]


def test_rrf_empty_and_n_and_k() -> None:
    assert rrf([]) == []
    assert len(rrf([[("A", 1.0), ("B", 0.5), ("C", 0.1)]], n=2)) == 2
    assert dict(rrf([[("A", 1.0)]], k=10))["A"] == pytest.approx(1 / 11)
