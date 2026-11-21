import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # TODO(Task 4): skip only when corpus_cache_dir() has no manifest.lock.json.
    skip = pytest.mark.skip(reason="corpus cache not available")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)
