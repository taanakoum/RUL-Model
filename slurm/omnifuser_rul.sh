#!/bin/bash
# Baseline B6, OmniFuser-RUL (PROJECT_PLAN.md D20; docs/OMNIFUSER_RUL_PLAN.md §8, §10).
# Submits one cache job, then one GPU job per arm of the modality study (10 folds x 3 seeds
# each), every arm job waiting for the caches. Run on the login node, from the repository:
#   source slurm/env.sh
#   bash slurm/omnifuser_rul.sh force tool chip    # force + the photo types Stage 3 admitted
#   bash slurm/omnifuser_rul.sh force tool chip -- --pairs-per-epoch 1024   # study settings
# Options after "--" go to every arm job; give the same ones to slurm/omnifuser_results.sh.
# A rerun skips finished fits, so resubmitting an arm resumes it.
#
# Guards (docs/RUCHE_RUNBOOK.md §12.1): Stage 3 must have built the stores (job J0), and
# every photo type given must be admitted by the Stage-3 feature check (job J3), because
# B6's configuration is "force + every admitted photo type" (O5). B6_ALLOW_UNCHECKED=1
# skips the second guard (for a pilot before J3 has run; never for the study).
set -euo pipefail
players=()
extra=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do players+=("$1"); shift; done
if [ "$#" -gt 0 ]; then shift; extra=("$@"); fi
if [ "${#players[@]}" -eq 0 ]; then
    echo "usage: bash slurm/omnifuser_rul.sh force IMAGE_TYPE [IMAGE_TYPE ...] [-- OPTIONS]" >&2
    exit 2
fi
set -- "${players[@]}"
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
cd "$(dirname "$0")/.."
mkdir -p runs/logs

if [ ! -f "$RULFUSE_DATA_ROOT/interim/forces/meta.json" ]; then
    echo "no force store under $RULFUSE_DATA_ROOT: run Stage 3 first (bash slurm/stage3.sh)" >&2
    exit 2
fi
if [ "${B6_ALLOW_UNCHECKED:-0}" != "1" ]; then
    python - "$RULFUSE_DATA_ROOT" "$@" <<'PY'
import glob
import json
import os
import sys

root, players = sys.argv[1], [p for p in sys.argv[2:] if p != "force"]
reports = sorted(glob.glob(os.path.join(root, "runs", "feature-check", "*.json")), key=os.path.getmtime)
if not reports:
    sys.exit("no Stage-3 feature-check report: wait for job J3 (or set B6_ALLOW_UNCHECKED=1 for a pilot)")
decisions = json.load(open(reports[-1], encoding="utf-8"))["decisions"]
kept = decisions["image_types_kept"]
refused = [p for p in players if p not in kept]
if refused:
    sys.exit(f"photo types {refused} are not admitted by {os.path.basename(reports[-1])} "
             f"(kept: {kept}; excluded: {decisions['image_types_excluded']})")
missing = [k for k in kept if k not in players]
if missing:
    print(f"note: admitted photo types not given: {missing} (B6's configuration uses all of them)")
print(f"photo types {players} admitted by {os.path.basename(reports[-1])}")
PY
fi

# The arms first (no data read), so a failure stops the script before anything is submitted.
arms=$(python -m rulfuse.cli omnifuser-study --players "$@" --list ${extra[@]+"${extra[@]}"})
if [ -z "$arms" ]; then
    echo "no arms listed" >&2
    exit 2
fi

prepare=$(sbatch --parsable --job-name=b6-prepare --time=02:00:00 slurm/job.sbatch \
    python -m rulfuse.cli omnifuser-prepare --feature-device cuda ${extra[@]+"${extra[@]}"})
prepare=${prepare%%;*}
echo "caches: job $prepare"

jobs="$prepare"
while IFS=$'\t' read -r name options; do
    # shellcheck disable=SC2086  # the options are one word per argument
    job=$(sbatch --parsable --dependency=afterok:"$prepare" --job-name="b6-$name" \
        --time=08:00:00 slurm/job.sbatch python -m rulfuse.cli omnifuser-arm $options \
        ${extra[@]+"${extra[@]}"})
    job=${job%%;*}
    jobs="$jobs $job"
    echo "arm $name: job $job"
done <<< "$arms"
echo "Follow with: bash slurm/status.sh    Cancel all: scancel $jobs"
tail=""
if [ "${#extra[@]}" -gt 0 ]; then tail=" -- ${extra[*]}"; fi
echo "When they finish: sbatch --job-name=b6-results slurm/cpu.sbatch bash slurm/omnifuser_results.sh $*$tail"
