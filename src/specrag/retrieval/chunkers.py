"""The three chunkers compared in experiment P1: fixed, section and section+meta.

Every chunk is a span of a document token stream (see ``streams.py``). ``text`` is the
body only; ``index_text`` adds the heading prefix for ``section+meta`` chunks.
"""

import math
from bisect import bisect_left, bisect_right
from typing import Literal

from specrag.models import Chunk, Doc, SectionIndex
from specrag.retrieval.streams import DocStream, safe_cut
from specrag.sections import split_id
from specrag.tokens import count_tokens, encoder_bytes

CHUNKER_VERSION = "1"

ChunkerName = Literal["fixed", "section", "section+meta"]
CutKind = Literal["paragraph", "sentence", "hard"]

TOLERANCE = 0.25  # a soft cut must lie within ±25% of the target piece length

# doc_id -> (label, short title) used in the section+meta prefix.
DOC_META: dict[str, tuple[str, str]] = {
    "rfc6749": ("RFC 6749", "OAuth 2.0 Authorization Framework"),
    "rfc6750": ("RFC 6750", "OAuth 2.0 Bearer Token Usage"),
    "rfc7636": ("RFC 7636", "PKCE"),
    "rfc7009": ("RFC 7009", "OAuth 2.0 Token Revocation"),
    "rfc7662": ("RFC 7662", "OAuth 2.0 Token Introspection"),
    "rfc8414": ("RFC 8414", "OAuth 2.0 Authorization Server Metadata"),
    "rfc8252": ("RFC 8252", "OAuth 2.0 for Native Apps"),
    "rfc8628": ("RFC 8628", "OAuth 2.0 Device Authorization Grant"),
    "rfc9126": ("RFC 9126", "OAuth 2.0 Pushed Authorization Requests"),
    "rfc9449": ("RFC 9449", "OAuth 2.0 DPoP"),
    "rfc8705": ("RFC 8705", "OAuth 2.0 Mutual-TLS"),
    "rfc9068": ("RFC 9068", "JWT Profile for OAuth 2.0 Access Tokens"),
    "rfc9700": ("RFC 9700", "OAuth 2.0 Security BCP"),
    "rfc7515": ("RFC 7515", "JSON Web Signature (JWS)"),
    "rfc7516": ("RFC 7516", "JSON Web Encryption (JWE)"),
    "rfc7517": ("RFC 7517", "JSON Web Key (JWK)"),
    "rfc7518": ("RFC 7518", "JSON Web Algorithms (JWA)"),
    "rfc7519": ("RFC 7519", "JSON Web Token (JWT)"),
    "rfc8725": ("RFC 8725", "JWT Best Current Practices"),
    "oidc-core": ("OpenID Connect Core 1.0", "OpenID Connect Core"),
    "oidc-discovery": ("OpenID Connect Discovery 1.0", "OpenID Connect Discovery"),
    "oidc-registration": (
        "OpenID Connect Dynamic Client Registration 1.0",
        "OpenID Connect Dynamic Client Registration",
    ),
    "oauth-2.1": ("draft-ietf-oauth-v2-1-17", "OAuth 2.1 (draft)"),
}


def _chunk(id: str, stream: DocStream, start: int, end: int, prefix: str | None = None) -> Chunk:
    sections, tokens = stream.attribute(start, end)
    text = stream.text(start, end)
    prefix_tokens = 0 if prefix is None else count_tokens(prefix + "\n")
    return Chunk(
        id=id,
        doc_id=stream.doc_id,
        text=text,
        index_text=text if prefix is None else prefix + "\n" + text,
        sections=sections,
        tokens=tokens,
        n_tokens=prefix_tokens + (end - start),
        span=(start, end),
        prefix_tokens=prefix_tokens,
    )


def chunk_fixed(stream: DocStream, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Windows of `size` tokens, step `size - overlap`, within one document."""
    n = len(stream.ids)
    step = size - overlap
    out: list[Chunk] = []
    start = 0
    while start < n:
        end = n if start + size >= n else safe_cut(stream, start + size, "back")
        if any(o is not None for o in stream.owner[start:end]):
            out.append(_chunk(f"fixed:{stream.doc_id}:{len(out):04d}", stream, start, end))
        if end >= n:
            break
        nxt = safe_cut(stream, start + step, "forward")
        start = nxt if nxt > start else start + step
    return out


def _boundaries(stream: DocStream, a: int, b: int) -> tuple[list[int], list[int]]:
    """Absolute paragraph and sentence cut positions strictly inside (a, b)."""
    pieces = [encoder_bytes([t]) for t in stream.ids[a:b]]
    data = b"".join(pieces)
    paras: list[int] = []
    sents: list[int] = []
    off = 0
    for i, piece in enumerate(pieces[:-1]):
        off += len(piece)
        before, after = data[max(0, off - 2) : off], data[off : off + 2]
        if before.endswith(b"\n\n"):
            paras.append(a + i + 1)
        elif (before.endswith(b".") and after[:1] == b" " and after[1:2].isupper()) or (
            before == b". " and after[:1].isupper()
        ):
            sents.append(a + i + 1)
    return paras, sents


def _nearest(cands: list[int], lo: float, hi: float, target: float) -> int | None:
    window = cands[bisect_left(cands, math.ceil(lo)) : bisect_right(cands, math.floor(hi))]
    return min(window, key=lambda p: (abs(p - target), p)) if window else None


def split_section(
    stream: DocStream, section_id: str, max_tokens: int = 800
) -> tuple[list[tuple[int, int]], list[CutKind]]:
    """Balanced contiguous pieces of a section's span, each at most `max_tokens`.

    Returns the piece spans and the kind of each cut (one fewer than pieces).
    """
    a, b = stream.section_spans[section_id]
    if b - a <= max_tokens:
        return [(a, b)], []
    paras, sents = _boundaries(stream, a, b)
    n = math.ceil((b - a) / max_tokens)
    while True:
        cuts: list[int] = []
        kinds: list[CutKind] = []
        prev = a
        for k in range(n - 1):
            length = (b - prev) / (n - k)  # target recomputed from what remains
            target = prev + length
            lo = max(prev + 1, target - TOLERANCE * length)
            hi = min(prev + max_tokens, target + TOLERANCE * length)
            kind: CutKind
            pos = _nearest(paras, lo, hi, target)
            kind = "paragraph"
            if pos is None:
                pos, kind = _nearest(sents, lo, hi, target), "sentence"
            if pos is None:
                hard = min(round(target), prev + max_tokens)
                pos, kind = safe_cut(stream, hard, "back"), "hard"
                if pos <= prev:
                    pos = hard
            cuts.append(pos)
            kinds.append(kind)
            prev = pos
        if b - prev <= max_tokens:
            bounds = [a, *cuts, b]
            return list(zip(bounds, bounds[1:], strict=False)), kinds
        n += 1


def chunk_section(stream: DocStream, idx: SectionIndex, max_tokens: int = 800) -> list[Chunk]:
    """One section per chunk; sections over `max_tokens` become balanced pieces."""
    out: list[Chunk] = []
    for sid in stream.section_spans:
        idx.get(sid)  # every streamed section must be indexed
        pieces, _ = split_section(stream, sid, max_tokens)
        for i, (s, e) in enumerate(pieces):
            out.append(_chunk(f"section:{sid}~{i}", stream, s, e))
    return out


def heading_path(section_id: str, idx: SectionIndex) -> str:
    """Ancestors root-first, then the section, each as "<number> <title>"."""
    ids = [*reversed(idx.ancestors(section_id)), section_id]
    return " › ".join(f"{split_id(i)[1]} {idx.get(i).title}" for i in ids)


def meta_prefix(section_id: str, idx: SectionIndex, doc: Doc) -> str:
    label, short = DOC_META.get(doc.doc_id, (doc.doc_id, doc.title))
    return f"{label} · {short} › {heading_path(section_id, idx)}"


def chunk_section_meta(
    stream: DocStream, idx: SectionIndex, doc: Doc, max_tokens: int = 800
) -> list[Chunk]:
    """`section` chunks with the doc label, title and heading path prepended to index_text."""
    out: list[Chunk] = []
    for sid in stream.section_spans:
        prefix = meta_prefix(sid, idx, doc)
        pieces, _ = split_section(stream, sid, max_tokens)
        for i, (s, e) in enumerate(pieces):
            out.append(_chunk(f"meta:{sid}~{i}", stream, s, e, prefix))
    return out


def make_chunks(
    name: ChunkerName,
    streams: dict[str, DocStream],
    idx: SectionIndex,
    docs: dict[str, Doc],
) -> list[Chunk]:
    """All chunks for one chunker, documents in sorted doc_id order (deterministic)."""
    out: list[Chunk] = []
    for doc_id in sorted(streams):
        st = streams[doc_id]
        if name == "fixed":
            out.extend(chunk_fixed(st))
        elif name == "section":
            out.extend(chunk_section(st, idx))
        else:
            out.extend(chunk_section_meta(st, idx, docs[doc_id]))
    return out
