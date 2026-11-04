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
    from rulfuse.adapters.nonastreda.load import load_labels

    return load_labels(paths.raw_dir())


@pytest.fixture(scope="session")
def data():
    """The Nonastreda adapter's own view (passes, runs, labels), default options."""
    from rulfuse.adapters.nonastreda import load_data

    return load_data()


@pytest.fixture(scope="session")
def data_recorded():
    from rulfuse.adapters.nonastreda import NonastredaOptions, load_data

    return load_data(NonastredaOptions(axis="recorded"))


@pytest.fixture(scope="session")
def dataset(data):
    """The real dataset in the generic schema."""
    from rulfuse.adapters.nonastreda import to_dataset

    return to_dataset(data)


@pytest.fixture(scope="session")
def real_store(dataset):
    """The default store (built once, then reused by signature)."""
    from rulfuse.core.store import SegmentStore, build_segment_store

    return SegmentStore.open(build_segment_store(dataset))
