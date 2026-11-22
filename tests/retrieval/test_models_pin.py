from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from specrag.retrieval import models_pin
from specrag.retrieval.models_pin import (
    ALLOW_PATTERNS,
    DENSE_MODEL,
    DENSE_REVISION,
    REQUIRED_FILES,
    ShadowedModelError,
    resolve_pinned,
)


def _make_snapshot(root: Path, groups: tuple[tuple[str, ...], ...]) -> str:
    for group in groups:
        f = root / group[0]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    return str(root)


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
    snap = _make_snapshot(tmp_path / "snap", REQUIRED_FILES[DENSE_MODEL])

    def fake(model_id: str, **kw: object) -> str:
        calls.append((model_id, kw))
        return snap

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    assert resolve_pinned(DENSE_MODEL, DENSE_REVISION) == snap
    assert calls == [
        (
            DENSE_MODEL,
            {
                "revision": DENSE_REVISION,
                "allow_patterns": ALLOW_PATTERNS,
                "local_files_only": True,
            },
        )
    ]
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
    assert resolve_pinned("org/m", DENSE_REVISION) == "/snap"
    monkeypatch.chdir(tmp_path / "elsewhere")
    with pytest.raises(ShadowedModelError):
        resolve_pinned("org/m", "rev")


def test_resolve_pinned_cache_miss_falls_back_to_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []

    def fake(model_id: str, **kw: object) -> str:
        calls.append(kw)
        if kw.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return "/net/snap"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    assert resolve_pinned(DENSE_MODEL, DENSE_REVISION) == "/net/snap"
    assert calls == [
        {"revision": DENSE_REVISION, "allow_patterns": ALLOW_PATTERNS, "local_files_only": True},
        {"revision": DENSE_REVISION, "allow_patterns": ALLOW_PATTERNS},
    ]


def test_resolve_pinned_partial_snapshot_falls_back_to_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    calls: list[dict[str, object]] = []

    def fake(model_id: str, **kw: object) -> str:
        calls.append(kw)
        return str(partial) if kw.get("local_files_only") else "/net/snap"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    assert resolve_pinned(DENSE_MODEL, DENSE_REVISION) == "/net/snap"
    assert [bool(c.get("local_files_only")) for c in calls] == [True, False]
    assert all(
        c["revision"] == DENSE_REVISION and c["allow_patterns"] == ALLOW_PATTERNS for c in calls
    )


@pytest.mark.parametrize("model", list(REQUIRED_FILES))
def test_resolve_pinned_complete_snapshot_makes_no_network_call(
    model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    snap = _make_snapshot(tmp_path / "full", REQUIRED_FILES[model])
    calls: list[dict[str, object]] = []

    def fake(model_id: str, **kw: object) -> str:
        calls.append(kw)
        return snap

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake)
    assert resolve_pinned(model, DENSE_REVISION) == snap
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["main", "rev", "5c38ec7", "G" * 40, DENSE_REVISION.upper()])
def test_resolve_pinned_rejects_non_sha_revision(
    bad: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: calls.append("x"))
    with pytest.raises(ValueError, match="40-hex"):
        resolve_pinned(DENSE_MODEL, bad)
    assert calls == []


def test_resolve_pinned_shadow_check_runs_before_any_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "BAAI" / "bge-small-en-v1.5").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: calls.append("x"))
    with pytest.raises(ShadowedModelError):
        resolve_pinned(DENSE_MODEL, DENSE_REVISION)
    assert calls == []


@pytest.mark.model
def test_resolve_pinned_cached_smoke_fast_without_offline_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import time

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.chdir(tmp_path)
    t = time.monotonic()
    path = resolve_pinned(DENSE_MODEL, DENSE_REVISION)
    assert time.monotonic() - t < 5
    assert Path(path).name == DENSE_REVISION
