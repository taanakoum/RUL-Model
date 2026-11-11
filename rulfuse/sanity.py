"""The seven sanity checks that end Stage 4 (PROJECT_PLAN.md §7.7; parts P11, P14).

Each check is a short fit or a direct test on fold 1 (fold 1 of the toy dataset for check
5). Each reads only the validation and training units (D15). They run as Ruche job J4
(``slurm/stage4.sh``); on the laptop, run single checks with ``--checks`` and short
``--steps``.

| # | Check | Pass when |
|---|---|---|
| 1 | Overfit one unit, regularization off | training MAE < 10 s |
| 2 | Failure times permuted among the training units | validation MAE no better than B2 − 10 s |
| 3 | Usage-only model B3, no trajectory augmentation | within 20 s of B2 |
| 4 | Oracle input: the measured causal HI of the previous recording | validation MAE < 0.8 × B2 |
| 5 | Toy data (known RUL) | validation MAE below B2 and 90 % coverage in [0.8, 0.98] |
| 6 | Truncation invariance | outputs up to a cut unchanged when later tokens are removed |
| 7 | The same seed twice on a CPU | identical weights |

Check 5's band is wider than the plan's 0.88-0.92: one toy validation unit has few points.
The band is logged as an amendment in PROJECT_LOG.
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rulfuse.config import FitConfig
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.store import SegmentStore
from rulfuse.sequences import FoldTensors, collate, make_sequence
from rulfuse.tokens import build_fold_data
from rulfuse.train import fit, predict


def _diag(out: Path) -> dict:
    return json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))


def check_overfit(store, sig, img, base: FitConfig, out: Path, device: str) -> dict:
    fold = leave_one_unit_out(store.unit_names)[0]
    cfg = base.with_overrides(
        {"regularize": False, "dropout": 0.0, "weight_decay": 0.0, "phase_a": False,
         "patience": 10_000, "copies": 1, "max_steps": base.phase_a_steps + 300}
    )
    sel = fit(store, sig, img, fold, 0, cfg, out, device, train_units=[fold.train[0]])
    train_mae = min(e["train_mae_s"] for e in sel["evaluations"])
    return {"train_mae_s": train_mae, "pass": train_mae < 10.0}


def check_permuted(store, sig, img, base: FitConfig, out: Path, device: str) -> dict:
    fold = leave_one_unit_out(store.unit_names)[0]
    data = build_fold_data(store, sig, img, fold, base)
    rng = np.random.default_rng(0)
    train = list(fold.train)
    lives = {u: data.units[u].failure_s - data.units[u].rec_start[0] for u in train}
    perm = rng.permutation(train)
    for u, v in zip(train, perm, strict=True):
        unit = data.units[u]
        new = unit.rec_start[0] + lives[str(v)]
        lo = unit.rec_end[unit.rec_baseline].max() + 1.0 if unit.rec_baseline.any() else unit.rec_end[0]
        unit.failure_s = float(np.clip(new, lo, unit.rec_end[-1]))
    fit(store, sig, img, fold, 0, base, out, device, data=data)
    d = _diag(out)
    return {
        "validation_mae_s": d["validation_window_mae_s"],
        "b2_mae_s": d["b2_window_mae_s"],
        "pass": d["validation_window_mae_s"] >= d["b2_window_mae_s"] - 10.0,
    }


def check_usage_only(store, sig, img, base: FitConfig, out: Path, device: str) -> dict:
    fold = leave_one_unit_out(store.unit_names)[0]
    cfg = base.with_overrides({"players": (), "trajectory_aug": False})
    fit(store, sig, img, fold, 0, cfg, out, device)
    d = _diag(out)
    gap = d["validation_window_mae_s"] - d["b2_window_mae_s"]
    return {"validation_mae_s": d["validation_window_mae_s"], "b2_mae_s": d["b2_window_mae_s"],
            "pass": abs(gap) <= 20.0}


def check_oracle(store, sig, img, base: FitConfig, out: Path, device: str) -> dict:
    fold = leave_one_unit_out(store.unit_names)[0]
    fit(store, sig, img, fold, 0, base.with_overrides({"oracle_hi": True}), out, device)
    d = _diag(out)
    return {"validation_mae_s": d["validation_window_mae_s"], "b2_mae_s": d["b2_window_mae_s"],
            "pass": d["validation_window_mae_s"] < 0.8 * d["b2_window_mae_s"]}


def check_toy(base: FitConfig, out: Path, device: str) -> dict:
    from rulfuse.adapters import toy
    from rulfuse.core.store import build_segment_store
    from rulfuse.features.cache import extract_image_features, extract_signal_features
    from rulfuse.features.encoders import PixelStatsEncoder, ProjectionEncoder

    root = Path(tempfile.mkdtemp())
    ds = toy.build(root / "images", n_units=6)
    store = SegmentStore.open(build_segment_store(ds, root / "store"))
    sig = extract_signal_features(store, ds.signals, ProjectionEncoder(), root=root / "f", log=lambda *_: None)
    img = extract_image_features(store, PixelStatsEncoder(), root=root / "f", log=lambda *_: None)
    fold = leave_one_unit_out(store.unit_names)[0]
    cfg = base.with_overrides({"players": ("signal", "surface")})
    fit(store, sig, img, fold, 0, cfg, out, device)
    d = _diag(out)
    cov = d["coverage90_by_eval"][-1] if d["coverage90_by_eval"] else float("nan")
    return {"validation_mae_s": d["validation_window_mae_s"], "b2_mae_s": d["b2_window_mae_s"],
            "coverage90": cov,
            "pass": d["validation_window_mae_s"] < d["b2_window_mae_s"] and 0.8 <= cov <= 0.98}


def check_truncation(store, sig, img, base: FitConfig, device: str) -> dict:
    from rulfuse.models.fusion import build_model

    fold = leave_one_unit_out(store.unit_names)[0]
    worst = 0.0
    for fusion in ("F1", "F0"):
        cfg = base.with_overrides({"fusion": fusion})
        data = build_fold_data(store, sig, img, fold, cfg, units=[fold.validation, *fold.train[:1]])
        torch.manual_seed(0)
        model = build_model(cfg, data, store.settings.horizon_s).to(device).eval()
        unit = data.units[fold.validation]
        tensors = FoldTensors(data, [fold.validation], device)
        full = make_sequence(unit)
        cut = len(full.seg) // 2
        short = copy.copy(full)
        for name in ("seg", "tau", "elapsed", "entry", "present", "point", "rul", "y", "window",
                     "weight", "hi", "oracle"):
            setattr(short, name, getattr(full, name)[:cut])
        t_cut = full.tau[cut - 1]
        keep = full.img_time < t_cut
        short.img, short.img_time, short.img_entry = full.img[keep], full.img_time[keep], full.img_entry[keep]
        a = predict(model, full, tensors, cfg, len(data.channels))
        b = predict(model, short, tensors, cfg, len(data.channels))
        worst = max(worst, float(np.max(np.abs(a.mu_s[:cut] - b.mu_s))))
    return {"max_abs_difference_s": worst, "pass": worst < 1e-3}


def check_determinism(store, sig, img, base: FitConfig, out: Path) -> dict:
    fold = leave_one_unit_out(store.unit_names)[0]
    cfg = base.with_overrides({"max_steps": base.phase_a_steps + 30, "patience": 10_000})
    states = []
    for k in (0, 1):
        target = out / f"run{k}"
        fit(store, sig, img, fold, 0, cfg, target, "cpu")
        states.append(torch.load(target / "model.pt", weights_only=True))
    same = all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    return {"identical": same, "pass": same}


def run_checks(
    store: SegmentStore,
    signal_features,
    image_features,
    base: FitConfig,
    out_dir: Path,
    checks: list[int],
    device: str = "cpu",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for k in checks:
        target = out_dir / f"check-{k}"
        if k == 1:
            report[k] = check_overfit(store, signal_features, image_features, base, target, device)
        elif k == 2:
            report[k] = check_permuted(store, signal_features, image_features, base, target, device)
        elif k == 3:
            report[k] = check_usage_only(store, signal_features, image_features, base, target, device)
        elif k == 4:
            report[k] = check_oracle(store, signal_features, image_features, base, target, device)
        elif k == 5:
            report[k] = check_toy(base, target, device)
        elif k == 6:
            report[k] = check_truncation(store, signal_features, image_features, base, device)
        elif k == 7:
            report[k] = check_determinism(store, signal_features, image_features, replace(base), target)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    report["all_pass"] = all(v["pass"] for v in report.values() if isinstance(v, dict))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return report
