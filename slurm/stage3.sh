#!/bin/bash
# Submits Stage 3 as one dependency chain (docs/RUCHE_RUNBOOK.md §11):
#   J0   checks (cpu_short)  ┐
#   J0g  GPU check (gpu_test) ┴─> J2 features (gpua100) ─> J3 feature check (cpu_short)
# A job starts only if the jobs it waits for succeeded (afterok); if one fails, the later
# ones stay pending with "DependencyNeverSatisfied" and must be cancelled (scancel).
# Usage, from the repository folder on the login node:  bash slurm/stage3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs

j0=$(sbatch --parsable slurm/checks.sbatch)
j0g=$(sbatch --parsable --job-name=J0g-gpu-check --partition=gpu_test --time=00:10:00 \
    slurm/job.sbatch python -c "import torch; x = torch.randn(256, 256, device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'matmul ok', float((x @ x).sum()))")
j2=$(sbatch --parsable --dependency=afterok:${j0%%;*}:${j0g%%;*} --job-name=J2-features \
    --time=02:00:00 slurm/job.sbatch bash slurm/features.sh)
j3=$(sbatch --parsable --dependency=afterok:${j2%%;*} --job-name=J3-feature-check \
    slurm/cpu.sbatch rulfuse feature-check)
echo "J0 checks        ${j0%%;*}"
echo "J0g GPU check    ${j0g%%;*}"
echo "J2 features      ${j2%%;*} (waits for J0 and J0g)"
echo "J3 feature check ${j3%%;*} (waits for J2)"
echo "Follow with: bash slurm/status.sh    Cancel all: scancel ${j0%%;*} ${j0g%%;*} ${j2%%;*} ${j3%%;*}"
