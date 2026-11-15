"""GPU check of the checkpoint fix (audit F05): an interrupted and resumed fit must equal an
uninterrupted one on CUDA, as the tests show on the CPU.

Run on a Ruche GPU node (a few minutes, toy data only, nothing is read from the real store):

    sbatch --job-name=cuda-resume --time=00:30:00 slurm/job.sbatch \\
        python scripts/check_cuda_resume.py

It trains the tiny toy configuration three times on one fold:

1. uninterrupted;
2. uninterrupted again: the control, which shows whether the GPU run is deterministic at all;
3. stopped after 12 steps, then resumed from its checkpoint.

PASS needs 3 equal to 1. If 2 differs from 1 the GPU kernels are not deterministic and the
check cannot tell; the script says so. Exit code 0 on PASS, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS, as in the array

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulfuse.train import fit  # noqa: E402
from tests.toy_engine import TINY, build_toy  # noqa: E402


def _weights(folder: Path) -> dict:
    return torch.load(folder / "model.pt", map_location="cpu", weights_only=True)


def _max_diff(a: dict, b: dict) -> float:
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device: run this on a GPU node")
        return 1
    quiet = lambda *_: None  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        toy = build_toy(root / "toy")
        fold = toy.folds[0]
        args = (toy.store, toy.signal, toy.image, fold, 0, TINY)
        fit(*args, root / "a", "cuda", quiet)
        fit(*args, root / "b", "cuda", quiet)
        stopped = fit(*args, root / "c", "cuda", quiet, stop_after_steps=12)
        assert stopped == {"interrupted_at": 12}, stopped
        assert (root / "c" / "checkpoint.pt").is_file()
        fit(*args, root / "c", "cuda", print)  # resumes from the checkpoint
        a, b, c = _weights(root / "a"), _weights(root / "b"), _weights(root / "c")
        control, resumed = _max_diff(a, b), _max_diff(a, c)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"max |uninterrupted - uninterrupted again| = {control:.3g}")
    print(f"max |uninterrupted - resumed|             = {resumed:.3g}")
    if control > 0:
        print("INCONCLUSIVE: two uninterrupted GPU runs differ, so the GPU is not deterministic")
        return 1
    if resumed == 0:
        print("PASS: the resumed fit equals the uninterrupted one")
        return 0
    print("FAIL: the resumed fit differs from the uninterrupted one")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
