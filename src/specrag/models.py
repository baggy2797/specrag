"""Core frozen data models."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from specrag.sections import parent


class DocStatus(StrEnum):
    STANDARDS_TRACK = "standards_track"
    BCP = "bcp"
    INFORMATIONAL = "informational"
    DRAFT = "draft"
    OPENID_FINAL = "openid_final"


class Doc(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    status: DocStatus
    url: str
    updates: list[str] = []
    obsoletes: list[str] = []


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    number: str
    title: str
    parent_id: str | None
    depth: int
    own_text: str
    own_tokens: int
    kind: Literal["content", "references"] = "content"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    text: str
    index_text: str
    sections: list[str]
    tokens: list[int]
    n_tokens: int

    @model_validator(mode="after")
    def _check_parallel(self) -> Self:
        if len(self.sections) < 1 or len(self.sections) != len(self.tokens):
            raise ValueError("sections and tokens must be the same length and non-empty")
        return self


class SectionIndex:
    def __init__(self, sections: list[Section]) -> None:
        self._by_id: dict[str, Section] = {}
        for s in sections:
            if s.id in self._by_id:
                raise ValueError(f"duplicate section id: {s.id}")
            self._by_id[s.id] = s

    def get(self, id: str) -> Section:
        try:
            return self._by_id[id]
        except KeyError:
            raise KeyError(f"unknown section id: {id}") from None

    def own_tokens(self, id: str) -> int:
        return self.get(id).own_tokens

    def __contains__(self, id: object) -> bool:
        return id in self._by_id

    def ancestors(self, id: str) -> list[str]:
        """Ancestor IDs, nearest first, derived from the ID (not looked up in the index).

        ``validate_sections`` guarantees every parsed section's parents exist.
        """
        out: list[str] = []
        p = parent(id)
        while p is not None:
            out.append(p)
            p = parent(p)
        return out
