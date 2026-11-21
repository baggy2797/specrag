import pytest

from specrag.corpus.manifest import corpus_cache_dir


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if (corpus_cache_dir() / "manifest.lock.json").exists():
        return
    skip = pytest.mark.skip(reason="corpus cache not available (run `specrag fetch`)")
    for item in items:
        if item.get_closest_marker("corpus") is not None:
            item.add_marker(skip)
