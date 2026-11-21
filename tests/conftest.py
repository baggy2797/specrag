import os

import pytest

from specrag.corpus.manifest import corpus_cache_dir


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    rules: list[tuple[str, bool, str]] = [
        (
            "corpus",
            (corpus_cache_dir() / "manifest.lock.json").exists(),
            "corpus cache not available (run `specrag fetch`)",
        ),
        (
            "model",
            os.environ.get("SPECRAG_MODELS") == "1",
            "needs the dense model (SPECRAG_MODELS=1)",
        ),
        (
            "reranker",
            os.environ.get("SPECRAG_RERANKER") == "1",
            "needs the reranker model (SPECRAG_RERANKER=1)",
        ),
    ]
    for marker, enabled, reason in rules:
        if enabled:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)
