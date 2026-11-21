"""Parse the cached corpus into a section index persisted as JSONL."""

import os
import tempfile

from specrag.corpus.manifest import ManifestEntry, corpus_cache_dir, raw_path
from specrag.corpus.parse import parse_doc
from specrag.models import Section, SectionIndex

SECTIONS_FILE = "sections.jsonl"


def build_sections(entries: list[ManifestEntry]) -> SectionIndex:
    sections: list[Section] = []
    for e in entries:
        html = raw_path(e.doc_id).read_text(encoding="utf-8")
        sections.extend(parse_doc(e.doc_id, html))
    index = SectionIndex(sections)  # rejects duplicate ids before anything is written
    cache = corpus_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache, prefix=".sections-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for s in sections:
                f.write(s.model_dump_json() + "\n")
        os.replace(tmp, cache / SECTIONS_FILE)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return index


def read_sections() -> list[Section]:
    path = corpus_cache_dir() / SECTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `specrag parse` first")
    with path.open(encoding="utf-8") as f:
        return [Section.model_validate_json(line) for line in f if line.strip()]


def load_section_index() -> SectionIndex:
    return SectionIndex(read_sections())
