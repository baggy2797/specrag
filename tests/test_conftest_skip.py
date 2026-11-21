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
