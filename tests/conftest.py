"""Shared fixtures. Tests marked ``data`` need the real data and the force store; they are
skipped when either is absent (run ``rulfuse build-force-store`` first)."""

from __future__ import annotations

import pytest

from rulfuse import paths


def _real_data_ready() -> bool:
    return (paths.raw_dir() / "labels.csv").is_file() and (
        paths.force_store_dir() / "meta.json"
    ).is_file()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _real_data_ready():
        return
    skip = pytest.mark.skip(reason="real data or force store absent (rulfuse build-force-store)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def metas():
    from rulfuse.data.nonastreda import load_labels

    return load_labels(paths.raw_dir())


@pytest.fixture(scope="session")
def dataset():
    from rulfuse.data.dataset import load_dataset

    return load_dataset()


@pytest.fixture(scope="session")
def dataset_recorded():
    from rulfuse.data.dataset import DatasetOptions, load_dataset

    return load_dataset(DatasetOptions(axis="recorded"))
