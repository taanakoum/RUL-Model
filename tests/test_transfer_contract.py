"""P22 Transfer contract (PROJECT_PLAN.md D18, §6.11, §15).

1. The dataset-agnostic packages import nothing from ``rulfuse.adapters`` and use none of
   Nonastreda's vocabulary.
2. A toy dataset unlike Nonastreda (2 accelerometer channels at 2 kHz, one image type,
   H = 600 s, no gaps, no duplicates) runs through the unchanged core: schema, store,
   features, folds, probes, readouts, leakage probe and baselines.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from rulfuse import paths
from rulfuse.adapters import toy
from rulfuse.baselines import score_usage_baselines
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.schema import validate
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import extract_image_features, extract_signal_features
from rulfuse.features.check import feature_check
from rulfuse.features.encoders import PixelStatsEncoder, ProjectionEncoder

GENERIC = ("core", "features", "models")  # rulfuse.models arrives in Stage 4
GENERIC_FILES = ("audit.py", "baselines.py", "metrics.py")
VOCABULARY = ("nonastreda", "flank", "tool_label", "image_label", "passmeta", "fz_rel", "ttf")


def _generic_sources() -> list[Path]:
    package = paths.REPO_ROOT / "rulfuse"
    files = [f for name in GENERIC for f in (package / name).rglob("*.py")]
    return files + [package / name for name in GENERIC_FILES]


def test_generic_code_never_imports_an_adapter():
    offenders = []
    for path in _generic_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.startswith("rulfuse.adapters")]
    assert offenders == []


def _code_words(tree: ast.AST) -> list[tuple[int, str]]:
    """Identifiers and string constants, docstrings excluded (they may explain where an idea
    came from; the code may not depend on it)."""
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    words = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            words.append((line, node.id))
        elif isinstance(node, ast.Attribute):
            words.append((line, node.attr))
        elif isinstance(node, ast.arg):
            words.append((line, node.arg))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            words.append((line, node.name))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                words.append((line, node.value))
    return words


def test_generic_code_has_no_nonastreda_vocabulary():
    hits = []
    for path in _generic_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, word in _code_words(tree):
            hits += [f"{path.name}:{line}: {w}" for w in VOCABULARY if w in word.lower()]
    assert hits == []


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    ds = toy.build(root / "images")
    validate(ds, check_signals=True)
    store = SegmentStore.open(build_segment_store(ds, root / "store"))
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=root / "f", log=lambda *_: None
    )
    image = extract_image_features(store, PixelStatsEncoder(), root=root / "f", log=lambda *_: None)
    return ds, store, signal, image


def test_toy_store_follows_the_settings(toy_run):
    ds, store, *_ = toy_run
    assert store.settings.horizon_s == 600.0 and store.settings.sample_rate_hz == 2000.0
    assert [c.name for c in store.channels] == ["acc_x", "acc_y"]
    seg = store.segments
    assert seg["signal_present"].all()  # no duplicates, no missing signals
    point = seg["is_prediction_point"].astype(bool)
    np.testing.assert_array_less(seg["rul_s"][point & seg["window"]], 600.0 + 1e-6)
    for unit in ds.units:
        starts = [r.start_s for r in unit.recordings]
        ends = [r.end_s for r in unit.recordings]
        np.testing.assert_allclose(starts[1:], ends[:-1])  # no gaps
        assert sum(seg["is_prediction_point"][seg["unit"] == unit.name]) > 0


def test_toy_features_run_through_the_unchanged_core(toy_run):
    _, store, signal, image = toy_run
    assert signal.complete and image.complete
    n_distinct = int(store.segments["distinct_segment"].max()) + 1
    assert signal.embeddings.shape == (n_distinct, 2, 2 * 16)  # 2 channels, no magnitude
    assert signal.descriptors.shape == (n_distinct, 2, 7)
    assert image.tokens.shape == (len(store.images["unit"]), 9, 6)
    assert np.isfinite(np.asarray(signal.embeddings)).all()


def test_toy_feature_check_and_baselines(toy_run):
    _, store, signal, image = toy_run
    report = feature_check(store, signal, image)
    assert report["signals"]["go"]  # vibration amplitude follows the synthetic wear
    assert report["images"]["surface"]["informative"]  # so does image brightness
    shuffled = report["signals"]["probe"]["encoder+descriptors"]["shuffled_control"]
    assert shuffled["mae"] > 0.8 * shuffled["mae_mean_predictor"]  # the control collapses
    result = score_usage_baselines(store, leave_one_unit_out(store.unit_names))
    assert result["horizon_s"] == 600.0
    assert result["summary"]["B0"]["mae"] > 0
