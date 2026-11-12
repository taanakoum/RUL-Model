#!/bin/bash
# One task of a RUL-Fuse study array (PROJECT_PLAN.md §7.11): the seeds of one manifest line
# (arm x fold), on one GPU. Finished fits are skipped and an interrupted one resumes, so a
# resubmission completes a study.
#   sbatch --array=0-N%4 slurm/array.sbatch STUDY        (N + 1 = lines of runs/STUDY/manifest.tsv)
# Usually submitted by slurm/study.sh, which writes the manifest first.
#SBATCH --job-name=rulfuse-array
#SBATCH --output=runs/logs/%x-%A_%a.out
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Submit this script with sbatch --array; see docs/RUCHE_RUNBOOK.md §15." >&2
    exit 2
fi
if [ "$#" -ne 1 ]; then
    echo "usage: sbatch --array=0-N slurm/array.sbatch STUDY" >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS (rulfuse.train.seed_everything)
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ($SLURM_JOB_NAME) on $(hostname), started $(date)"
echo "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
# Fail fast, naming the node, if its GPU cannot be used (a GPU in exclusive mode still held
# by another process gives "CUDA-capable device(s) is/are busy or unavailable"). Resubmit
# with that node excluded (docs/RUCHE_RUNBOOK.md §15.4).
if ! python -c "import torch; torch.zeros(1, device='cuda'); print('GPU ok:', torch.cuda.get_device_name(0))"; then
    echo "BAD GPU NODE: $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
    echo "$(hostname)  # $(date +%F) job ${SLURM_JOB_ID:-?}" >> "$RULFUSE_DATA_ROOT/bad_gpu_nodes.txt"
    nvidia-smi >&2 || true
    exit 3
fi
python -m rulfuse.cli run-task --study "$1" --task "$SLURM_ARRAY_TASK_ID" --device cuda
echo "finished $(date)"
