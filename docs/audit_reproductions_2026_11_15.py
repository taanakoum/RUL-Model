"""Reproduce audit findings with synthetic data; no real test results are read.

Run from the repository: python docs/audit_reproductions_2026_11_15.py
The results describe the audited implementation. After fixes, the corresponding
observations should change. Temporary files are confined to the repository.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold, leave_one_unit_out
from rulfuse.core.schema import (
    ArraySignals,
    ChannelSpec,
    Dataset,
    Recording,
    Settings,
    SignalRef,
    Unit,
    validate,
)
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.gate3 import RELIABILITY_LEVELS, decide_calibration, freeze
from rulfuse.models.fusion import RULFuse
from rulfuse.sequences import FoldTensors, Sequence, collate
from rulfuse.study import Study, find_identical_fit
from rulfuse.tokens import build_fold_data
from rulfuse.train import fit_key


def image_padding() -> dict:
    torch.manual_seed(0)
    cfg = FitConfig(
        players=("signal", "photo"),
        fusion="F0",
        d_model=16,
        rank=4,
        layers=1,
        heads=2,
        ffn=32,
        dropout=0.0,
        regularize=False,
    )
    units, seqs = {}, []
    for name, count in (("short", 2), ("long", 4)):
        units[name] = SimpleNamespace(
            signal=np.ones((3, 1, 2), np.float32),
            seg_present=np.ones(3, bool),
            images=np.arange(count * 2, dtype=np.float32).reshape(count, 1, 2),
            img_kind=np.zeros(count, np.int64),
            img_readout=np.zeros(count, np.float32),
        )
        seqs.append(
            Sequence(
                unit=name,
                seg=np.arange(3),
                tau=np.array([5.0, 15.0, 25.0]),
                elapsed=np.array([5.0, 15.0, 25.0]),
                entry=np.zeros(3, np.int64),
                present=np.ones(3, bool),
                point=np.ones(3, bool),
                rul=np.array([95.0, 85.0, 75.0]),
                y=np.array([95.0, 85.0, 75.0]),
                window=np.ones(3, bool),
                weight=np.ones(3) / 3,
                hi=np.zeros(3),
                oracle=np.zeros(3),
                img=np.arange(count),
                img_time=np.arange(1, count + 1) * 10.0,
                img_entry=np.zeros(count, np.int64),
                failure_s=100.0,
            )
        )
    data = SimpleNamespace(units=units, image_tokens=1, image_dim=2)
    tensors = FoldTensors(data, ["short", "long"], "cpu")
    model = RULFuse(cfg, 100.0, 2, [("sensor", "x", "raw")], 2, 1, 1, False).eval()
    batch = collate(seqs, tensors, cfg, None, 1)
    single = collate(seqs[:1], tensors, cfg, None, 1)
    with torch.no_grad():
        original, alone = model(batch, tensors), model(single, tensors)
        tensors.images[1] += 50.0  # short unit's photo available at 20 seconds
        changed = model(batch, tensors)
    return {
        "padded_image_times": batch["img_time"][0].tolist(),
        "image_present_before_first_photo": bool(original["experts"]["present"][0, 0, 2]),
        "single_prediction_at_5s": float(alone["mu"][0, 0]) * 100,
        "batched_prediction_at_5s": float(original["mu"][0, 0]) * 100,
        "change_at_5s_after_editing_photo_at_20s": float(
            (changed["mu"][0, 0] - original["mu"][0, 0]).abs()
        )
        * 100,
    }


def artifact_checks(root: Path) -> dict:
    fold = Fold("T1", "T2", ("T3", "T4"))
    old = Study(
        "source",
        {
            "store": "toy/same",
            "signal_encoder": "projection",
            "image_encoder": "old",
            "arms": {"design": {}},
            "decision": {"type": "none"},
        },
    )
    new = Study("target", {**old.spec, "signal_encoder": "mantis8m", "image_encoder": "new"})
    source = root / "source/design/fold-T1/seed-0"
    source.mkdir(parents=True)
    (root / "source/study.json").write_text(json.dumps(old.spec), encoding="utf-8")
    (source / "selection.json").write_text(
        json.dumps(
            {
                "config_key": fit_key(old.config("design", root)),
                "fold_validation": "T2",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    (source / "DONE").write_text("ok", encoding="utf-8")
    found = find_identical_fit(
        root,
        "toy/same",
        fit_key(new.config("design", root)),
        fold,
        0,
        root / "target/design/fold-T1/seed-0",
    )
    study = Study(
        "g3",
        {
            "store": "toy/same",
            "signal_encoder": "projection",
            "image_encoder": "pixelstats",
            "arms": {"design": {}},
            "inner": {"n": 2},
            "decision": {"type": "calibration"},
        },
    )
    cells = [
        {
            "done": True,
            "validation": f"V{i}",
            "fold_test": f"T{i}",
            "methods": {
                m: {"coverage": 0.3 if m == "c" else 0.2, "winkler": 100.0}
                for m in ("raw", "a", "b", "c")
            },
            "filter": {
                "coverage": 0.2,
                "winkler": 100.0,
                "window_mae_s": 10.0,
                "unfiltered_window_mae_s": 10.0,
            },
            "reliability": {m: [0.2] * len(RELIABILITY_LEVELS) for m in ("b", "c")},
        }
        for i in range(10)
    ]
    decision = decide_calibration(study, cells)
    study.run_dir(root).mkdir()
    (study.run_dir(root) / "decision.json").write_text(
        json.dumps(decision, default=lambda v: sorted(v) if isinstance(v, set) else float(v)),
        encoding="utf-8",
    )
    frozen = freeze(study, root, root / "final.json")
    return {
        "reused_despite_different_encoders": found == source,
        "coverage_constraint_met": decision["constraint_met"],
        "calibration_choice": decision["calibration_choice"],
        "freeze_accepted": bool(frozen["hash"]),
    }


def no_images(root: Path) -> dict:
    units = tuple(
        Unit(
            f"U{i}",
            (Recording(f"r{i}", 0.0, 2.0, SignalRef(f"s{i}", 0, 20)),),
            (),
            2.0,
        )
        for i in range(3)
    )
    ds = Dataset(
        "signalonly",
        units,
        (ChannelSpec("x", "force", "x"),),
        Settings(
            horizon_s=2.0,
            sample_rate_hz=10.0,
            model_rate_hz=10.0,
            baseline_recordings=1,
            segment_s=1.0,
            min_remainder_s=0.5,
        ),
        ArraySignals({f"s{i}": np.ones((20, 1), np.float32) for i in range(3)}),
    )
    validate(ds, check_signals=True)
    store = SegmentStore.open(build_segment_store(ds, root / "signalonly"))
    result = {"schema_valid": True, "image_columns": list(store.images)}
    try:
        build_fold_data(
            store, None, None, leave_one_unit_out(store.unit_names)[0], FitConfig(players=())
        )
        result["fold_build"] = "passed"
    except KeyError as error:
        result["fold_build"] = f"KeyError: {error}"
    return result


def b6_seeds(root: Path) -> dict:
    from rulfuse.omnifuser_rul.aggregate import ensemble, load_predictions
    from rulfuse.omnifuser_rul.train import ArmSpec

    spec = ArmSpec(players=("force",), seeds=(0,))
    folder = root / "b6"
    for seed, value in ((0, 100.0), (1, 300.0)):
        fit_dir = folder / "fold-01" / f"seed-{seed}"
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit.json").write_text(
            json.dumps({"spec": spec.signature()}),
            encoding="utf-8",
        )
        pq.write_table(
            pa.table(
                {
                    "role": ["validation"],
                    "fold": [1],
                    "row": [0],
                    "unit": ["U1"],
                    "on_grid": [True],
                    "seed": [seed],
                    "prediction_s": [value],
                }
            ),
            fit_dir / "predictions.parquet",
        )
    loaded = load_predictions(folder, spec.signature())
    return {
        "requested_seeds": list(spec.seeds),
        "loaded_seeds": np.unique(loaded["seed"]).tolist(),
        "expected_prediction": 100.0,
        "actual_prediction": ensemble(loaded, "validation")["prediction_s"].tolist(),
    }


def main() -> None:
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix=".audit-probes-", dir=repo) as folder:
        root = Path(folder).resolve()
        assert root.is_relative_to(repo)
        results = {
            "f0_image_padding": image_padding(),
            "artifacts": artifact_checks(root),
            "no_images": no_images(root),
            "b6_seeds": b6_seeds(root),
            "cuda_available": torch.cuda.is_available(),
        }
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
