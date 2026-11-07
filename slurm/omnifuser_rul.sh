#!/bin/bash
# Baseline B6, OmniFuser-RUL: the single full run (docs/OMNIFUSER_RUL_PLAN.md version 5;
# docs/RUCHE_RUNBOOK.md §12.1). One configuration, no comparison of settings:
#   players = force + the photo types Stage 3 admitted, with usage;
#   GRU memory on; loss = point-weighted MSE + 1/15 x monotonic penalty + 0.05 x slope term
#   (both move only the earlier prediction of a pair); the report has both inference modes;
#   10 folds x 3 seeds = 30 fits, in one GPU job.
# It submits that job and, chained after it, the report job (slurm/omnifuser_results.sh),
# which starts only if the training succeeded. Run on the login node, from the repository:
#   source slurm/env.sh
#   bash slurm/omnifuser_rul.sh force tool chip
#   bash slurm/omnifuser_rul.sh force tool chip -- --pairs-per-epoch 1024   # other settings
# Options after "--" go to the training and the report jobs alike (the report selects the
# fits made with them). A rerun skips finished fits, so resubmitting resumes the run.
#
# Guards: Stage 3 must have built the stores (job J0), and every photo type given must be
# admitted by the Stage-3 feature check (job J3): B6's configuration is "force + every
# admitted photo type" (O5). B6_ALLOW_UNCHECKED=1 skips the second guard.
set -euo pipefail
players=()
extra=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do players+=("$1"); shift; done
if [ "$#" -gt 0 ]; then shift; extra=("$@"); fi
if [ "${#players[@]}" -eq 0 ]; then
    echo "usage: bash slurm/omnifuser_rul.sh force IMAGE_TYPE [IMAGE_TYPE ...] [-- OPTIONS]" >&2
    exit 2
fi
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
cd "$(dirname "$0")/.."
mkdir -p runs/logs

if [ ! -f "$RULFUSE_DATA_ROOT/interim/forces/meta.json" ]; then
    echo "no force store under $RULFUSE_DATA_ROOT: run Stage 3 first (bash slurm/stage3.sh)" >&2
    exit 2
fi
if [ "${B6_ALLOW_UNCHECKED:-0}" != "1" ]; then
    python - "$RULFUSE_DATA_ROOT" "${players[@]}" <<'PY'
import glob
import json
import os
import sys

root, players = sys.argv[1], [p for p in sys.argv[2:] if p != "force"]
reports = sorted(glob.glob(os.path.join(root, "runs", "feature-check", "*.json")), key=os.path.getmtime)
if not reports:
    sys.exit("no Stage-3 feature-check report: wait for job J3 (or set B6_ALLOW_UNCHECKED=1)")
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

train=$(sbatch --parsable --job-name=b6-train --time=08:00:00 slurm/job.sbatch \
    python -m rulfuse.cli omnifuser-arm --players "${players[@]}" --feature-device cuda \
    ${extra[@]+"${extra[@]}"})
train=${train%%;*}
report=$(sbatch --parsable --dependency=afterok:"$train" --job-name=b6-report \
    slurm/cpu.sbatch bash slurm/omnifuser_results.sh "${players[@]}" \
    ${extra[@]+-- "${extra[@]}"})
report=${report%%;*}
echo "B6 training: job $train (30 fits; builds the 1 s store and caches first)"
echo "B6 report:   job $report (waits for the training; writes REPORT.md and the archive)"
echo "Follow with: bash slurm/status.sh    Cancel both: scancel $train $report"
