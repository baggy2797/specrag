from dataclasses import dataclass
from typing import Literal

from specrag.models import Section
from specrag.tokens import decode, encode, encoder_bytes

SEPARATOR = "\n\n"
MAX_CUT_SHIFT = 3


@dataclass(frozen=True)
class DocStream:
    doc_id: str
    ids: list[int]
    owner: list[str | None]  # same length as ids; None for separator tokens
    section_spans: dict[str, tuple[int, int]]

    def attribute(self, start: int, end: int) -> tuple[list[str], list[int]]:
        """Section IDs in first-appearance order with their token counts in [start, end)."""
        counts: dict[str, int] = {}
        for o in self.owner[start:end]:
            if o is not None:
                counts[o] = counts.get(o, 0) + 1
        return list(counts), list(counts.values())

    def text(self, start: int, end: int) -> str:
        return decode(self.ids[start:end])


def build_streams(sections: list[Section]) -> dict[str, DocStream]:
    """One stream per document: content sections with tokens, each followed by a separator."""
    sep = encode(SEPARATOR)
    ids: dict[str, list[int]] = {}
    owner: dict[str, list[str | None]] = {}
    spans: dict[str, dict[str, tuple[int, int]]] = {}
    for s in sections:
        if s.kind != "content" or s.own_tokens <= 0:
            continue
        body = encode(s.own_text)
        d_ids = ids.setdefault(s.doc_id, [])
        d_owner = owner.setdefault(s.doc_id, [])
        start = len(d_ids)
        d_ids.extend(body)
        d_owner.extend([s.id] * len(body))
        spans.setdefault(s.doc_id, {})[s.id] = (start, len(d_ids))
        d_ids.extend(sep)
        d_owner.extend([None] * len(sep))
    return {d: DocStream(d, ids[d], owner[d], spans[d]) for d in ids}


def _clean(stream: DocStream, pos: int) -> bool:
    """True if a cut at `pos` falls on a UTF-8 character boundary (local byte check)."""
    if pos <= 0 or pos >= len(stream.ids):
        return True
    tail = encoder_bytes(stream.ids[max(0, pos - 4) : pos])[-4:]
    for i in range(len(tail) - 1, -1, -1):
        b = tail[i]
        if 0x80 <= b <= 0xBF:  # continuation byte, keep looking for the lead byte
            continue
        need = 1 if b < 0x80 else 2 if b < 0xE0 else 3 if b < 0xF0 else 4
        return len(tail) - i == need
    return False


def safe_cut(stream: DocStream, pos: int, direction: Literal["forward", "back"]) -> int:
    """Nearest cut position (moving in `direction`, at most 3 tokens) that splits no character.

    Fallback: if no clean position exists within 3 tokens (including a stream whose
    bytes are not valid UTF-8 there), returns `pos` unchanged. Callers must tolerate that.
    """
    step = 1 if direction == "forward" else -1
    for shift in range(MAX_CUT_SHIFT + 1):
        p = pos + step * shift
        if 0 <= p <= len(stream.ids) and _clean(stream, p):
            return p
    return pos
