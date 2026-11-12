#!/bin/bash
# Runs one rulfuse command on a Ruche GPU node (OmniFuser's slurm/job.sbatch).
#
# Submit it from the repository folder, with the command after the script name:
#   sbatch --job-name=NAME [more sbatch options] slurm/job.sbatch COMMAND [ARGUMENTS...]
# Example:
#   sbatch --job-name=J2-features --time=02:00:00 slurm/job.sbatch bash slurm/features.sh
#
# The lines below are defaults; options given to sbatch on the command line override them.
#SBATCH --job-name=rulfuse
#SBATCH --output=runs/logs/%x-%j.out
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see docs/RUCHE_RUNBOOK.md." >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "No command given after slurm/job.sbatch." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail
# Keep CPU thread pools inside the cores SLURM granted (Ruche gives 4 GB per core).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS (rulfuse.train.seed_everything)

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION, started $(date)"
echo "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
echo "command: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
# Fail fast, naming the node, if its GPU cannot be used (a GPU in exclusive mode still held
# by another process gives "CUDA-capable device(s) is/are busy or unavailable"). Resubmit
# with that node excluded (docs/RUCHE_RUNBOOK.md §15.4).
if ! python -c "import torch; torch.zeros(1, device='cuda'); print('GPU ok:', torch.cuda.get_device_name(0))"; then
    echo "BAD GPU NODE: $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
    nvidia-smi >&2 || true
    exit 3
fi
"$@"
echo "finished $(date)"
