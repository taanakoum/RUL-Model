#!/bin/bash
# Job J0 on a CPU node: verify the data, build the stores, score the usage baselines, prove
# that the encoder weights load offline, then run the whole test suite (PROJECT_PLAN.md §15).
# Submit from the repository folder:  sbatch slurm/checks.sbatch
#SBATCH --job-name=J0-checks
#SBATCH --output=runs/logs/%x-%j.out
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see docs/RUCHE_RUNBOOK.md." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
echo "data root: $RULFUSE_DATA_ROOT; weights: $HF_HOME (offline: ${HF_HUB_OFFLINE:-0})"
python -c "import sys, torch, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__)"
rulfuse resources

echo "=== P1 verify the raw copy"
rulfuse verify-data
echo "=== force store (reads the MAT file once; skipped if it exists)"
if [ ! -f "$RULFUSE_DATA_ROOT/interim/forces/meta.json" ]; then
    rulfuse build-force-store
fi
echo "=== segment store"
rulfuse build-store
echo "=== usage baselines B0-B2 (PROJECT_PLAN.md §2.7: 412 / 513 / 468 s)"
rulfuse baselines
echo "=== offline weights: Mantis-8M and DINOv2-B load from \$HF_HOME without the network"
python - <<'PY'
from rulfuse.features.encoders import DinoV2Encoder, MantisEncoder

for encoder in (MantisEncoder(device="cpu", batch_size=4), DinoV2Encoder(device="cpu")):
    signature = encoder.signature()
    assert signature["revision"], f"{signature['repo']}: no cached revision"
    print("ok", signature)
PY
echo "=== tests (synthetic, real data, real weights)"
python -m pytest -o addopts= -q -rs tests
echo "finished $(date)"
