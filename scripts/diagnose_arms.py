"""Where does a study's error come from? Validation tools only (D15), no training.

For every validation tool it reports, from the finished fits of a study:

* the window MAE of each arm and of the **equal-weight ensemble of the arms** (arms of one
  study are different models, so their average shows how much of the error is training noise);
* the mean signed error (prediction minus truth; negative = too pessimistic, positive = too
  optimistic);
* the MAE in three bands of true remaining life: near the end (<= 500 s), middle
  (500-1000 s) and far (1000-1500 s).

    python scripts/diagnose_arms.py --study gate2b-fix              # on Ruche, CPU, seconds
    python scripts/diagnose_arms.py --study gate2b-fix --out diag.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulfuse.study import (  # noqa: E402
    Study,
    fit_dir,
    open_store,
    read_validation_rows,
    study_folds,
)

BANDS = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0))


def tool_rows(study: Study, arm: str, fold, runs: Path | None) -> dict | None:
    """The validation rows of the arm's first seed, window points, in time order."""
    seed = study.arm_seeds(arm)[0]
    folder = fit_dir(study.run_dir(runs), arm, fold, seed)
    if not (folder / "DONE").is_file():
        return None
    rows = read_validation_rows(folder / "predictions.parquet")
    keep = rows["window"].astype(bool)
    order = np.argsort(rows["tau_s"][keep], kind="stable")
    return {k: np.asarray(v)[keep][order] for k, v in rows.items()}


def band_mae(rul: np.ndarray, err: np.ndarray) -> list[float]:
    out = []
    for lo, hi in BANDS:
        sel = (rul > lo) & (rul <= hi) if lo else (rul <= hi)
        out.append(float(np.mean(np.abs(err[sel]))) if sel.any() else float("nan"))
    return out


def diagnose(study: Study, runs: Path | None = None) -> list[dict]:
    store = open_store(study.spec)
    table = []
    for fold in study_folds(study, store.unit_names):
        members = {a: tool_rows(study, a, fold, runs) for a in study.arms}
        members = {a: m for a, m in members.items() if m is not None}
        if not members:
            continue
        first = next(iter(members.values()))
        rul = first["rul_s"].astype(float)
        for m in members.values():
            if not np.array_equal(m["tau_s"], first["tau_s"]):
                raise ValueError(f"{fold.validation}: arms predict different points")
        preds = {a: m["mu_s"].astype(float) for a, m in members.items()}
        preds["ENSEMBLE"] = np.mean(list(preds.values()), axis=0)
        for arm, mu in preds.items():
            err = mu - rul
            near, mid, far = band_mae(rul, err)
            table.append(
                {
                    "validation": fold.validation,
                    "arm": arm,
                    "points": len(rul),
                    "mae_s": float(np.mean(np.abs(err))),
                    "bias_s": float(np.mean(err)),
                    "mae_near_le500": near,
                    "mae_mid_500_1000": mid,
                    "mae_far_1000_1500": far,
                }
            )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Where does a study's error come from?")
    parser.add_argument("--study", required=True)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="also write a CSV")
    args = parser.parse_args()
    study = Study.load(args.study)
    table = diagnose(study, args.runs_root)
    if not table:
        print("no finished fit found")
        return 1
    print(
        f"{'tool':5s} {'arm':18s} {'pts':>4s} {'MAE':>6s} {'bias':>7s} {'<=500':>6s} "
        f"{'500-1000':>9s} {'1000-1500':>10s}"
    )
    for r in table:
        if r["arm"] not in ("ENSEMBLE", "default"):
            continue
        print(
            f"{r['validation']:5s} {r['arm']:18s} {r['points']:4d} {r['mae_s']:6.0f} "
            f"{r['bias_s']:7.0f} {r['mae_near_le500']:6.0f} {r['mae_mid_500_1000']:9.0f} "
            f"{r['mae_far_1000_1500']:10.0f}"
        )
    for arm in ("default", "ENSEMBLE"):
        mine = [r for r in table if r["arm"] == arm]
        if mine:
            print(f"mean over tools, {arm}: MAE {np.mean([r['mae_s'] for r in mine]):.1f} s")
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
