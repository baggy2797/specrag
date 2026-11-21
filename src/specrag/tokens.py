from functools import cache

import tiktoken


@cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def encode(text: str) -> list[int]:
    return _encoder().encode(text)


def decode(ids: list[int]) -> str:
    return _encoder().decode(ids)


def encoder_bytes(ids: list[int]) -> bytes:
    """Raw UTF-8 bytes of the given tokens (may start or end mid-character)."""
    return _encoder().decode_bytes(ids)
