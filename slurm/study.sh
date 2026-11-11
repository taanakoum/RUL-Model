#!/bin/bash
# Submits one RUL-Fuse study (PROJECT_PLAN.md §7.11, docs/RUCHE_RUNBOOK.md §15):
#   1. writes runs/STUDY/manifest.tsv (on the login node; fails if an inherited gate has no
#      decision.json yet);
#   2. submits the array (slurm/array.sbatch), at most 4 tasks at once;
#   3. submits the aggregation (cpu_short), which waits for the whole array (afterok).
# Usage, from the repository, after `source slurm/env.sh`:
#   bash slurm/study.sh pilot | gate2a | gate2b | sweep
# Gates run one after the other: gate2b inherits gate2a's decision and the sweep both, so
# submit the next one only after reading the previous REPORT.md.
set -euo pipefail
study="${1:?usage: bash slurm/study.sh STUDY}"
cd "$(dirname "$0")/.."
mkdir -p runs/logs
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
python -m rulfuse.cli manifest --study "$study"
n=$(( $(wc -l < "$RULFUSE_DATA_ROOT/runs/$study/manifest.tsv") - 1 ))
array=$(sbatch --parsable --array=0-$((n - 1))%4 --job-name="$study" slurm/array.sbatch "$study")
array=${array%%;*}
agg=$(sbatch --parsable --dependency=afterok:"$array" --job-name="$study-aggregate" \
    slurm/cpu.sbatch python -m rulfuse.cli aggregate --study "$study")
agg=${agg%%;*}
echo "$study: array $array ($n tasks), aggregation $agg (waits for the array)"
echo "Result: \$RULFUSE_DATA_ROOT/runs/$study/REPORT.md and decision.json"
echo "Follow with: bash slurm/status.sh    Cancel: scancel $array $agg"
