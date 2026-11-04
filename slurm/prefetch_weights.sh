#!/bin/bash
# Run on the Ruche LOGIN node (it has internet), once, before any GPU job:
#   source slurm/env.sh && bash slurm/prefetch_weights.sh
# Downloads only the files the loaders read (config + safetensors) of the two encoders into
# $HF_HOME, about 380 MB, so jobs run with HF_HUB_OFFLINE=1 (set by slurm/env.sh in jobs).
# MOMENT-1-base, the fallback of PROJECT_PLAN.md §6.2, is fetched only if the force
# go/no-go check fails: bash slurm/prefetch_weights.sh AutonLab/MOMENT-1-base
set -euo pipefail
: "${HF_HOME:?source slurm/env.sh first}"
python - "$@" <<'PY'
import sys

from huggingface_hub import snapshot_download

repos = sys.argv[1:] or ["paris-noah/Mantis-8M", "facebook/dinov2-base"]
for repo in repos:
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])
    print("ok", repo, path)
PY
du -sh "$HF_HOME"
echo "Offline loading is tested on a compute node by job J0 (sbatch slurm/checks.sbatch)."
