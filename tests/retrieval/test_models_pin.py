from pathlib import Path

import pytest

from specrag.retrieval import models_pin
from specrag.retrieval.models_pin import (
    ALLOW_PATTERNS,
    DENSE_MODEL,
    DENSE_REVISION,
    resolve_pinned,
)


def test_resolve_pinned_rejects_shadowing_local_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "BAAI" / "bge-small-en-v1.5").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="local path shadows pinned model id"):
        resolve_pinned(DENSE_MODEL, DENSE_REVISION)


def test_resolve_pinned_passes_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake(model_id: str, **kw: object) -> str:
        calls.append((model_id, kw))
        return "/snap/path"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    assert resolve_pinned(DENSE_MODEL, DENSE_REVISION) == "/snap/path"
    assert calls == [(DENSE_MODEL, {"revision": DENSE_REVISION, "allow_patterns": ALLOW_PATTERNS})]
    assert not any(p in ("*.bin", "*.onnx") for p in models_pin.ALLOW_PATTERNS)


@pytest.mark.model
def test_dense_model_loads_on_cpu() -> None:
    from sentence_transformers import SentenceTransformer

    path = resolve_pinned(DENSE_MODEL, DENSE_REVISION)
    assert Path(path).is_absolute()
    model = SentenceTransformer(path, device="cpu")
    assert model.device.type == "cpu"
    assert model.get_embedding_dimension() == 384
