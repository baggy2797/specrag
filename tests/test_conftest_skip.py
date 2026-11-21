import os
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).parent / "conftest.py"


def test_unmarked_test_under_corpus_dir_not_skipped_without_cache(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    (proj / "corpus").mkdir(parents=True)
    (proj / "conftest.py").write_text(CONFTEST.read_text())
    (proj / "corpus" / "test_a.py").write_text(
        "import pytest\ndef test_unmarked(): pass\n@pytest.mark.corpus\ndef test_marked(): pass\n"
    )
    env = {**os.environ, "SPECRAG_CACHE": str(tmp_path / "empty-cache")}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-W", "ignore"],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "1 passed, 1 skipped" in r.stdout, r.stdout + r.stderr


def _run_marked(tmp_path: Path, marker: str, **env_over: str) -> str:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "conftest.py").write_text(CONFTEST.read_text())
    (proj / "test_a.py").write_text(
        f"import pytest\n@pytest.mark.{marker}\ndef test_marked(): pass\ndef test_plain(): pass\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("SPECRAG_MODELS", "SPECRAG_RERANKER")}
    env.update(env_over)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-W", "ignore"],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
    )
    return r.stdout + r.stderr


def test_model_marker_skipped_by_default(tmp_path: Path) -> None:
    out = _run_marked(tmp_path, "model")
    assert "1 passed, 1 skipped" in out
    assert "SPECRAG_MODELS=1" in out


def test_model_marker_runs_with_env(tmp_path: Path) -> None:
    assert "2 passed" in _run_marked(tmp_path, "model", SPECRAG_MODELS="1")


def test_reranker_marker_skipped_by_default(tmp_path: Path) -> None:
    out = _run_marked(tmp_path, "reranker", SPECRAG_MODELS="1")
    assert "1 passed, 1 skipped" in out
    assert "SPECRAG_RERANKER=1" in out


def test_reranker_marker_runs_with_env(tmp_path: Path) -> None:
    assert "2 passed" in _run_marked(tmp_path, "reranker", SPECRAG_RERANKER="1")
