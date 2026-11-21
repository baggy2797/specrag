import hashlib
from pathlib import Path

import pytest
import yaml

from specrag.corpus.manifest import load_manifest, raw_path
from specrag.corpus.parse import parse_doc

SNAPSHOTS = yaml.safe_load((Path(__file__).parent / "snapshots.yaml").read_text(encoding="utf-8"))

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize("doc_id", list(SNAPSHOTS))
def test_snapshot(doc_id: str) -> None:
    snap = SNAPSHOTS[doc_id]
    secs = parse_doc(doc_id, raw_path(doc_id).read_text(encoding="utf-8"))
    assert len(secs) == snap["n_sections"]
    assert secs[0].id == snap["first_id"]
    assert secs[-1].id == snap["last_id"]
    assert sum(s.own_tokens for s in secs) == snap["own_tokens_total"]
    ids_sha = hashlib.sha256("\n".join(s.id for s in secs).encode()).hexdigest()
    assert ids_sha == snap["ids_sha256"]
    assert sum(s.kind == "references" for s in secs) == snap["n_references"]
    titles = {s.id: s.title for s in secs}
    for sample in snap["samples"]:
        assert titles[sample["id"]] == sample["title"]


def test_every_doc_parses_and_validates() -> None:
    entries = load_manifest()
    assert set(SNAPSHOTS) == {e.doc_id for e in entries}
    for e in entries:
        secs = parse_doc(e.doc_id, raw_path(e.doc_id).read_text(encoding="utf-8"))
        assert len(secs) >= 10, e.doc_id
