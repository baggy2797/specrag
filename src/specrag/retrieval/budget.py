"""The token-budget packer: turns any ranking into (at most) BUDGET tokens of context.

Cost of a chunk is its prefix tokens plus the body tokens not already packed from the
same document (overlap is free). At most one chunk is truncated, and packing stops there.
"""

from pydantic import BaseModel, ConfigDict

from specrag.models import Chunk
from specrag.retrieval.streams import DocStream, safe_cut

BUDGET = 2500


class Packed(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk]
    costs: list[int]  # parallel to chunks; sum(costs) == tokens_used
    tokens_used: int
    exhausted: bool  # the ranking ran out before the budget filled


def _truncate(chunk: Chunk, stream: DocStream, cut: int) -> Chunk:
    start = chunk.span[0]
    sections, tokens = stream.attribute(start, cut)
    text = stream.text(start, cut)
    prefix = chunk.index_text[: len(chunk.index_text) - len(chunk.text)]  # "" or "<prefix>\n"
    return Chunk(
        id=chunk.id,
        doc_id=chunk.doc_id,
        text=text,
        index_text=prefix + text,
        sections=sections,
        tokens=tokens,
        n_tokens=chunk.prefix_tokens + (cut - start),
        span=(start, cut),
        prefix_tokens=chunk.prefix_tokens,
    )


def pack(ranked: list[Chunk], streams: dict[str, DocStream], budget: int = BUDGET) -> Packed:
    covered: dict[str, set[int]] = {}
    chunks: list[Chunk] = []
    costs: list[int] = []
    used = 0
    stopped = False  # filled the budget or truncated a chunk
    for chunk in ranked:
        if used >= budget:
            stopped = True
            break
        seen = covered.setdefault(chunk.doc_id, set())
        start, end = chunk.span
        new = [p for p in range(start, end) if p not in seen]
        if not new:
            continue
        cost = chunk.prefix_tokens + len(new)
        if used + cost <= budget:
            chunks.append(chunk)
            costs.append(cost)
            used += cost
            seen.update(new)
            continue
        room = budget - used - chunk.prefix_tokens
        if room <= 0:
            continue
        stream = streams[chunk.doc_id]
        cut = safe_cut(stream, new[room - 1] + 1, "back")
        cut = max(cut, start)
        kept = [p for p in new if p < cut]
        if not kept:
            continue
        if all(stream.owner[p] is None for p in kept):  # new tokens are only separators
            continue
        chunks.append(_truncate(chunk, stream, cut))
        costs.append(chunk.prefix_tokens + len(kept))
        used += costs[-1]
        stopped = True
        break
    return Packed(
        chunks=chunks, costs=costs, tokens_used=used, exhausted=not (stopped or used >= budget)
    )
