"""Shared fixtures. Tests marked ``data`` need the real data and the force store; they are
skipped when either is absent (run ``rulfuse build-force-store`` first)."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from rulfuse import paths
from tests.catalog import COMPATIBILITY_TESTS, LEGACY, LEVELS, legacy_level

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_ITEMS: dict[str, dict] = {}
_SKIPS: list[str] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rulfuse verification")
    group.addoption(
        "--requirements-report", type=Path, help="write collected requirements and outcomes"
    )
    group.addoption(
        "--fail-on-skip", action="store_true", help="fail if any selected test is skipped"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Classify before pytest applies -m, so categories also select legacy tests."""
    ready = _real_data_ready()
    for item in items:
        filename = item.path.name
        name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        if name in COMPATIBILITY_TESTS:
            item.add_marker(pytest.mark.compatibility)
        levels = [m.name for m in item.iter_markers() if m.name in LEVELS]
        if not levels and filename in LEGACY:
            dependency = bool(item.get_closest_marker("data") or item.get_closest_marker("weights"))
            item.add_marker(getattr(pytest.mark, legacy_level(filename, name, dependency)))
        if not list(item.iter_markers("requirements")) and filename in LEGACY:
            item.add_marker(pytest.mark.requirements(*LEGACY[filename][1]))
        levels = [m.name for m in item.iter_markers() if m.name in LEVELS]
        if len(levels) != 1:
            raise pytest.UsageError(
                f"{item.nodeid}: declare exactly one unit/integration/e2e level"
            )
        requirements = sorted({str(v) for m in item.iter_markers("requirements") for v in m.args})
        if not requirements or any(not re.fullmatch(r"(?:P|D|O)\d+", r) for r in requirements):
            raise pytest.UsageError(f"{item.nodeid}: declare valid requirements(P*, D*, O*)")
        if levels[0] == "e2e":
            item.add_marker(pytest.mark.slow)
        if filename in {"test_audit_fixes.py", "test_padded_images.py"}:
            item.add_marker(pytest.mark.regression)
        if "monte_carlo" in name or "learns" in name or "coverage_exchangeable" in name:
            item.add_marker(pytest.mark.scientific)
        if name == "test_status_counts_inner_fits":
            item.add_marker(pytest.mark.system)
        if item.get_closest_marker("data") and not ready:
            item.add_marker(pytest.mark.skip(reason="real data or force store absent"))
        if item.get_closest_marker("gpu"):
            import torch

            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="CUDA device unavailable"))
        _ITEMS[item.nodeid] = {
            "level": levels[0],
            "requirements": requirements,
            "markers": sorted({m.name for m in item.iter_markers()}),
            "outcome": "not_run",
        }


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:
        _SKIPS.append(report.nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    entry = _ITEMS.get(report.nodeid)
    if entry is not None and (report.when == "call" or report.failed or report.skipped):
        entry["outcome"] = report.outcome
        if report.failed:
            entry["failure"] = str(report.longrepr)
    if report.skipped:
        _SKIPS.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if session.config.getoption("fail_on_skip") and _SKIPS and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    target = session.config.getoption("requirements_report")
    if target is None:
        return
    selected = {item.nodeid for item in session.items}
    entries = {key: value for key, value in _ITEMS.items() if key in selected}
    report = {
        "exit_code": int(session.exitstatus),
        "collected": session.testscollected,
        "selected": len(entries),
        "outcomes": dict(Counter(v["outcome"] for v in entries.values())),
        "levels": dict(Counter(v["level"] for v in entries.values())),
        "skipped_nodes": _SKIPS,
        "tests": entries,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def bounded_test_threads():
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("RULFUSE_TEST_THREADS", "2")))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def micro_bundle(tmp_path_factory):
    from tests.factories import build_micro_bundle

    return build_micro_bundle(tmp_path_factory.mktemp("micro"))


def _real_data_ready() -> bool:
    return (paths.raw_dir() / "labels.csv").is_file() and (
        paths.force_store_dir() / "meta.json"
    ).is_file()


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
