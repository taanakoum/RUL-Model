"""B4, B5 and R1 on the real store, scored on the validation units (PROJECT_PLAN.md §8, §15.1
row P16). The test units stay sealed until 2026-12-01.

    D:\\RULModel\\.venv\\Scripts\\python.exe scripts/feature_baselines.py

Light: ridge fits and importance sampling on cached features, no weights. Writes
``D:\\RULModel\\runs\\baselines\\feature-nonastreda-3e357b5a23ff.json``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from rulfuse.baselines import score_feature_baselines  # noqa: E402
from rulfuse.core.folds import leave_one_unit_out  # noqa: E402
from rulfuse.core.store import SegmentStore  # noqa: E402
from rulfuse.features.cache import ImageFeatures, SignalFeatures, find_features  # noqa: E402

STORE = Path(r"D:\RULModel\interim\segments\nonastreda\3e357b5a23ff")
OUT = Path(r"D:\RULModel\runs\baselines\feature-nonastreda-3e357b5a23ff.json")
IMAGE_KIND = "tool"


def main() -> None:
    start = time.time()
    store = SegmentStore.open(STORE)
    signal = SignalFeatures.open(find_features(store, "signal", "mantis8m"))
    image = ImageFeatures.open(find_features(store, "image", "dinov2b"))
    result = score_feature_baselines(
        store, leave_one_unit_out(store.unit_names), signal, image, IMAGE_KIND
    )
    result["store"] = store.signature
    result["signal_features"] = str(signal.folder)
    result["image_features"] = str(image.folder)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"validation-unit MAE (s), window points, {time.time() - start:.0f} s")
    for name, summary in result["summary"].items():
        per_fold = [r["mae"] for r in result["rows"] if r["baseline"] == name]
        print(f"{name}: {summary['mae']:7.1f}   folds: " + " ".join(f"{m:6.1f}" for m in per_fold))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
