"""Hit rule and retrieval metrics (spec 7.2-7.3).

Rules:
- A section entry counts only if its tokens are ``> 0`` AND ``>= threshold``
  (text must actually come from the section; spec 7.2).
- A chunk hits gold section ``g`` if ``g in chunk.sections`` with that entry's
  tokens ``> 0`` and ``>= hit_threshold(own_tokens(g))``.
- Otherwise, only if ``own_tokens(g) < 50`` (a heading-only section), it hits
  when some entry ``s`` is a strict descendant of ``g`` with tokens
  ``> 0`` and ``>= hit_threshold(own_tokens(s))``.
- A parent never hits a child.
- Each gold section is counted once across the ranking.
- ``ignored`` sections are removed from ``gold``, and chunks whose only hits
  are ignored are removed from the ranking before nDCG, MRR and precision.
- nDCG uses binary relevance: rank i is relevant if its chunk hits an
  effective-gold section not hit by an earlier rank.
  IDCG = ``min(|gold|, k)`` relevant ranks.
- Metrics return ``None`` when the effective gold set is empty (unanswerable
  questions are excluded from averages).
"""

import math
from collections.abc import Set as AbstractSet

from specrag.models import Chunk, SectionIndex
from specrag.sections import is_descendant

HEADING_ONLY_TOKENS = 50
_NO_IGNORED: frozenset[str] = frozenset()


def hit_threshold(own_tokens: int) -> float:
    """Tokens of a section a chunk must carry to hit it: min(50, 0.3 * own_tokens)."""
    return min(50.0, 0.3 * own_tokens)


def chunk_hits(chunk: Chunk, gold_id: str, idx: SectionIndex) -> bool:
    """Whether ``chunk`` hits ``gold_id`` (direct, or via a descendant if heading-only)."""
    own = idx.own_tokens(gold_id)
    for sec, tok in zip(chunk.sections, chunk.tokens, strict=True):
        if sec == gold_id and tok > 0 and tok >= hit_threshold(own):
            return True
    if own < HEADING_ONLY_TOKENS:
        for sec, tok in zip(chunk.sections, chunk.tokens, strict=True):
            if (
                is_descendant(sec, gold_id)
                and tok > 0
                and tok >= hit_threshold(idx.own_tokens(sec))
            ):
                return True
    return False


def hit_set(chunk: Chunk, gold: AbstractSet[str], idx: SectionIndex) -> set[str]:
    """All gold sections hit by ``chunk``."""
    return {g for g in gold if chunk_hits(chunk, g, idx)}


def _effective(gold: AbstractSet[str], ignored: AbstractSet[str]) -> set[str]:
    return set(gold) - set(ignored)


def _drop_ignored(
    ranked: list[Chunk], gold: AbstractSet[str], idx: SectionIndex, ignored: AbstractSet[str]
) -> list[Chunk]:
    """Remove chunks whose hits against the original gold are all ignored."""
    kept = []
    for c in ranked:
        hits = hit_set(c, gold, idx)
        if hits and hits <= set(ignored):
            continue
        kept.append(c)
    return kept


def recall(
    ranked: list[Chunk],
    gold: AbstractSet[str],
    idx: SectionIndex,
    ignored: AbstractSet[str] = _NO_IGNORED,
) -> float | None:
    """Fraction of effective gold sections hit by any chunk (each counted once)."""
    eff = _effective(gold, ignored)
    if not eff:
        return None
    found: set[str] = set()
    for c in ranked:
        found |= hit_set(c, eff, idx)
    return len(found) / len(eff)


def mrr(
    ranked: list[Chunk],
    gold: AbstractSet[str],
    idx: SectionIndex,
    ignored: AbstractSet[str] = _NO_IGNORED,
) -> float | None:
    """1/rank of the first chunk hitting effective gold (after ignored-dropping); 0.0 if none."""
    eff = _effective(gold, ignored)
    if not eff:
        return None
    for i, c in enumerate(_drop_ignored(ranked, gold, idx, ignored), start=1):
        if hit_set(c, eff, idx):
            return 1.0 / i
    return 0.0


def ndcg_at(
    ranked: list[Chunk],
    gold: AbstractSet[str],
    idx: SectionIndex,
    k: int = 10,
    ignored: AbstractSet[str] = _NO_IGNORED,
) -> float | None:
    """Binary-gain nDCG@k; a gold section only earns credit at its first hit."""
    eff = _effective(gold, ignored)
    if not eff:
        return None
    seen: set[str] = set()
    dcg = 0.0
    for i, c in enumerate(_drop_ignored(ranked, gold, idx, ignored)[:k], start=1):
        new = hit_set(c, eff, idx) - seen
        if new:
            dcg += 1.0 / math.log2(i + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(eff), k) + 1))
    return dcg / idcg


def hit_rate_precision(
    ranked: list[Chunk],
    gold: AbstractSet[str],
    idx: SectionIndex,
    ignored: AbstractSet[str] = _NO_IGNORED,
) -> float | None:
    """Chunks hitting effective gold / chunks retained after ignored-dropping."""
    eff = _effective(gold, ignored)
    if not eff:
        return None
    kept = _drop_ignored(ranked, gold, idx, ignored)
    if not kept:
        return None
    return sum(1 for c in kept if hit_set(c, eff, idx)) / len(kept)


def condensed(ranked: list[Chunk], judged: AbstractSet[str], idx: SectionIndex) -> list[Chunk]:
    """Keep chunks with at least one judged section; drop all-unjudged chunks."""
    return [c for c in ranked if any(s in judged for s in c.sections)]
