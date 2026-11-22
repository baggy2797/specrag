from pathlib import Path

import pytest

from specrag.retrieval import models_pin
from specrag.retrieval.models_pin import (
    ALLOW_PATTERNS,
    DENSE_MODEL,
    DENSE_REVISION,
    ShadowedModelError,
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


@pytest.mark.parametrize("bad", ["/abs/model", "./BAAI/x", "../x", "~/x", "."])
def test_resolve_pinned_rejects_absolute_and_explicit_relative_ids(
    bad: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/snap")
    with pytest.raises(ShadowedModelError, match="local path shadows pinned model id"):
        resolve_pinned(bad, "rev")


def test_resolve_pinned_shadow_check_is_cwd_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "elsewhere" / "org" / "m").mkdir(parents=True)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/snap")
    monkeypatch.chdir(tmp_path / "a")  # org/m exists only under the parent: no shadow here
    assert resolve_pinned("org/m", "rev") == "/snap"
    monkeypatch.chdir(tmp_path / "elsewhere")
    with pytest.raises(ShadowedModelError):
        resolve_pinned("org/m", "rev")
