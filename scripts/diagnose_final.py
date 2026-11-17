"""Why does the frozen design fail on the test tools? CPU only, no training.

Two parts:

* ``--features`` (needs only the store and the feature caches, runs anywhere they exist):
  for the tool photos (DINOv2-B) and the force (Mantis-8M + level descriptors),
  1. **tool identity**: centroids of each tool from the first third of its life classify its
     photos/segments from the middle third (chance = 1/10). High accuracy means the features
     say *which tool* this is, beyond its wear;
  2. **transferable wear**: a leave-one-tool-out ridge from the features to the capped RUL,
     scored on window points of the held-out tool, against a constant.
* ``--fits`` (needs a study's finished fits, on Ruche): per fold of one arm,
  1. test window MAE, mean signed error (bias), the MAE left after removing each tool's own
     constant offset, and the slope of μ against τ (the truth falls with slope −1);
  2. the tool's lifetime against the mean lifetime of its training tools, and the B2 error;
  3. F0's gate weights α (usage, signal, image) on the test tool;
  4. window MAE on the training tools, the validation tool and the test tool;
  5. selection: the validation window MAE at the kept evaluation, at the last one and the
     median over evaluations, the evaluation-to-evaluation noise, and the out-of-fold window
     MAE of the inner fits (never selected on the tools they predict).

    python scripts/diagnose_final.py --features --out diag_features.md
    python scripts/diagnose_final.py --fits --study gate3 --arm design --out diag_gate3.md
    python scripts/diagnose_final.py --fits --study shapley --arm T --out diag_T.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulfuse.study import (  # noqa: E402
    Study,
    fit_dir,
    inner_fit_dir,
    inner_split,
    open_store,
    study_folds,
)

# ----------------------------------------------------------------------------- features


def _standardize(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return [(x - mu) / sd for x in (train, *others)]


def _pca(train: np.ndarray, k: int, *others: np.ndarray):
    _, _, vt = np.linalg.svd(train - train.mean(0), full_matrices=False)
    w = vt[: min(k, vt.shape[0])].T
    m = train.mean(0)
    return [(x - m) @ w for x in (train, *others)]


def identity_accuracy(x: np.ndarray, unit: np.ndarray, frac: np.ndarray) -> float:
    """Centroids from life fraction < 1/3, classify 1/3 <= fraction < 2/3."""
    early, mid = frac < 1 / 3, (frac >= 1 / 3) & (frac < 2 / 3)
    tools = sorted(set(unit[early]) & set(unit[mid]))
    keep_e = early & np.isin(unit, tools)
    keep_m = mid & np.isin(unit, tools)
    xe, xm = _standardize(x[keep_e], x[keep_m])
    xe, xm = _pca(xe, 50, xm)
    cent = np.stack([xe[unit[keep_e] == t].mean(0) for t in tools])
    d = ((xm[:, None, :] - cent[None]) ** 2).sum(-1)
    pred = np.array(tools)[d.argmin(1)]
    return float(np.mean(pred == unit[keep_m]))


def loto_ridge(x, unit, rul_capped, window, horizon, lam=10.0) -> dict:
    """Leave-one-tool-out ridge to the capped RUL; window MAE per held-out tool."""
    out_model, out_const = [], []
    for t in sorted(set(unit)):
        tr, te = unit != t, (unit == t) & window
        if not te.any():
            continue
        a, b = _standardize(x[tr], x[te])
        a, b = _pca(a, 50, b)
        y = rul_capped[tr]
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (y - y.mean()))
        pred = np.clip(b @ w + y.mean(), 0, horizon)
        out_model.append(float(np.mean(np.abs(pred - rul_capped[te]))))
        out_const.append(float(np.mean(np.abs(horizon / 2 - rul_capped[te]))))
    return {"ridge": float(np.mean(out_model)), "constant": float(np.mean(out_const))}


def features_report(study_name: str) -> list[str]:
    from rulfuse.features.cache import ImageFeatures, SignalFeatures, find_features

    store = open_store(Study.load(study_name).spec)
    horizon = store.settings.horizon_s
    fail = {u: v["failure_s"] for u, v in store.units.items() if v.get("failure_s")}
    lines = ["## Features: tool identity and transferable wear", ""]
    lines += [
        "| Modality | Identity accuracy (chance 0.10) | LOTO ridge window MAE (s) "
        "| Constant H/2 (s) |",
        "|---|---|---|---|",
    ]

    # tool photos
    rows = store.image_rows(kind="tool")
    imgs = ImageFeatures.open(find_features(store, "image", store_encoder(study_name, "image")))
    x = np.asarray(imgs.tokens[rows], float).mean(1)
    unit = store.images["unit"][rows].astype(str)
    t = store.images["available_s"][rows].astype(float)
    ok = np.isin(unit, list(fail))
    x, unit, t = x[ok], unit[ok], t[ok]
    life = np.array([fail[u] for u in unit])
    alive = t <= life  # photos after the failure have no RUL target
    x, unit, t, life = x[alive], unit[alive], t[alive], life[alive]
    rul = np.clip(life - t, 0, None)
    acc = identity_accuracy(x, unit, t / life)
    r = loto_ridge(x, unit, np.minimum(rul, horizon), rul <= horizon, horizon)
    lines.append(f"| Tool photos (DINOv2-B) | {acc:.2f} | {r['ridge']:.0f} | {r['constant']:.0f} |")

    # force
    sig = SignalFeatures.open(find_features(store, "signal", store_encoder(study_name, "signal")))
    seg = store.segments
    keep = seg["signal_present"].astype(bool) & np.isin(seg["unit"].astype(str), list(fail))
    keep &= seg["is_prediction_point"].astype(bool)
    d = seg["distinct_segment"][keep].astype(int)
    unit = seg["unit"][keep].astype(str)
    tau = seg["tau_s"][keep].astype(float)
    life = np.array([fail[u] for u in unit])
    rul = np.clip(life - tau, 0, None)
    for name, feats in (
        ("Force embeddings (Mantis-8M)", np.asarray(sig.embeddings[d], float).mean(1)),
        ("Force level descriptors", np.asarray(sig.descriptors[d], float).reshape(len(d), -1)),
    ):
        acc = identity_accuracy(feats, unit, tau / life)
        r = loto_ridge(feats, unit, np.minimum(rul, horizon), rul <= horizon, horizon)
        lines.append(f"| {name} | {acc:.2f} | {r['ridge']:.0f} | {r['constant']:.0f} |")
    lines.append("")
    return lines


def store_encoder(study_name: str, kind: str) -> str:
    spec = Study.load(study_name).spec
    return spec["image_encoder"] if kind == "image" else spec["signal_encoder"]


# ----------------------------------------------------------------------------- fits


def _rows(path: Path, role: str) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("role", "=", role)])
    out = {c: table.column(c).to_numpy(zero_copy_only=False) for c in table.column_names}
    order = np.lexsort((out["tau_s"], out["unit"].astype(str))) if len(out["unit"]) else []
    return {k: np.asarray(v)[order] for k, v in out.items()}


def _window_mae(r: dict) -> float:
    w = r["window"].astype(bool)
    return float(np.mean(np.abs(r["mu_s"][w] - r["rul_s"][w]))) if w.any() else float("nan")


def fits_report(study_name: str, arm: str, runs: Path | None) -> tuple[list[str], list[dict]]:
    from rulfuse.baselines import b2

    study = Study.load(study_name)
    store = open_store(study.spec)
    horizon = store.settings.horizon_s
    fail = {u: v["failure_s"] for u, v in store.units.items()}
    seed = study.arm_seeds(arm)[0]
    table = []
    for fold in sorted(study_folds(study, store.unit_names), key=lambda f: f.test):
        folder = fit_dir(study.run_dir(runs), arm, fold, seed)
        if not (folder / "DONE").is_file():
            print(f"skip {fold.test}: not finished")
            continue
        pred = folder / "predictions.parquet"
        te = _rows(pred, "test")
        w = te["window"].astype(bool)
        err = te["mu_s"][w] - te["rul_s"][w]
        tau = te["tau_s"][w]
        train_life = np.array([fail[u] for u in fold.train if fail[u]])
        b2_pred = b2(train_life, tau, horizon)
        slope = float(np.polyfit(tau, te["mu_s"][w], 1)[0]) if w.sum() > 2 else float("nan")
        tr = _rows(pred, "train")
        va = _rows(pred, "validation")
        sel = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        ev = [e["val_window_mae_s"] for e in sel["evaluations"]]
        k = sel["selected_eval"]
        oof = []
        if study.inner:
            for i, (held, _) in enumerate(inner_split(fold, **study.inner)):
                f = inner_fit_dir(study.run_dir(runs), arm, fold, i, seed)
                if (f / "DONE").is_file():
                    h = _rows(f / "predictions.parquet", "heldout")
                    for u in held:
                        m = h["unit"].astype(str) == u
                        oof.append(_window_mae({c: v[m] for c, v in h.items()}))
        row = {
            "test": fold.test,
            "points": int(w.sum()),
            "test_mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "mae_after_offset": float(np.mean(np.abs(err - err.mean()))),
            "slope_mu_vs_tau": slope,
            "life_s": float(fail[fold.test]),
            "train_mean_life_s": float(train_life.mean()),
            "life_gap_s": float(train_life.mean() - fail[fold.test]),
            "b2_mae": float(np.mean(np.abs(b2_pred - te["rul_s"][w]))),
            "corr_mu_b2": float(np.corrcoef(te["mu_s"][w], b2_pred)[0, 1])
            if w.sum() > 2
            else np.nan,
            "train_mae": _window_mae(tr),
            "val_mae": _window_mae(va),
            "val_at_kept": float(ev[k]) if 0 <= k < len(ev) else float("nan"),
            "val_at_last": float(ev[-1]) if ev else float("nan"),
            "val_median_evals": float(np.median(ev)) if ev else float("nan"),
            "eval_noise_sd": float(np.std(np.diff(ev))) if len(ev) > 2 else float("nan"),
            "n_evals": len(ev),
            "oof_inner_mae": float(np.mean(oof)) if oof else float("nan"),
        }
        for name in ("alpha_usage", "alpha_signal", "alpha_image"):
            if name in te:
                row[name] = float(np.mean(te[name][w]))
        table.append(row)
    return render_fits(study_name, arm, table), table


def render_fits(study_name: str, arm: str, t: list[dict]) -> list[str]:
    if not t:
        return [f"## Fits {study_name}/{arm}: none finished", ""]

    def m(k):
        return float(np.nanmean([r.get(k, np.nan) for r in t]))

    gap = np.array([r["life_gap_s"] for r in t])
    bias = np.array([r["bias"] for r in t])
    corr = float(np.corrcoef(gap, bias)[0, 1])
    share = 1 - m("mae_after_offset") / m("test_mae")
    lines = [
        f"## Fits {study_name}/{arm} ({len(t)} folds)",
        "",
        "| Quantity | Mean over folds |",
        "|---|---|",
        f"| Test window MAE (s) | {m('test_mae'):.1f} |",
        f"| Test MAE minus each tool's own offset (s) | {m('mae_after_offset'):.1f} |",
        f"| Share of test MAE that is a per-tool offset | {share:.0%} |",
        f"| Corr(bias, training-mean life − tool life) across tools | {corr:.2f} |",
        f"| Slope of μ against τ on the test tool (truth: −1) | {m('slope_mu_vs_tau'):.2f} |",
        f"| Corr(μ, B2 prediction) on the test tool | {m('corr_mu_b2'):.2f} |",
        f"| B2 test MAE (s) | {m('b2_mae'):.1f} |",
        f"| Window MAE on training tools (s) | {m('train_mae'):.1f} |",
        f"| Window MAE on the validation tool (s) | {m('val_mae'):.1f} |",
        f"| Validation MAE at kept / last / median evaluation (s) | {m('val_at_kept'):.1f} / "
        f"{m('val_at_last'):.1f} / {m('val_median_evals'):.1f} |",
        f"| Evaluation-to-evaluation noise sd (s) | {m('eval_noise_sd'):.1f} |",
        f"| Out-of-fold MAE of the inner fits (s) | {m('oof_inner_mae'):.1f} |",
    ]
    if "alpha_usage" in t[0]:
        lines.append(
            f"| F0 gate weights on test tool: usage / signal / image | {m('alpha_usage'):.2f} / "
            f"{m('alpha_signal'):.2f} / {m('alpha_image'):.2f} |"
        )
    lines += [
        "",
        "| Tool | Life (s) | Train mean life | Bias | Test MAE | After offset | B2 "
        "| Train MAE | Val MAE | OOF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t:
        lines.append(
            f"| {r['test']} | {r['life_s']:.0f} | {r['train_mean_life_s']:.0f} | {r['bias']:+.0f} "
            f"| {r['test_mae']:.0f} | {r['mae_after_offset']:.0f} | {r['b2_mae']:.0f} "
            f"| {r['train_mae']:.0f} | {r['val_mae']:.0f} | {r['oof_inner_mae']:.0f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", action="store_true")
    p.add_argument("--fits", action="store_true")
    p.add_argument("--study", default="gate3")
    p.add_argument("--arm", default="design")
    p.add_argument("--runs-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    lines = ["# Diagnosis of the final result", ""]
    if a.features:
        lines += features_report(a.study)
    if a.fits:
        fl, table = fits_report(a.study, a.arm, a.runs_root)
        lines += fl
        if a.out:
            a.out.with_suffix(".json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
