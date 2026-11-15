#!/bin/bash
# Submits one RUL-Fuse study (PROJECT_PLAN.md §7.11, docs/RUCHE_RUNBOOK.md §15):
#   1. writes runs/STUDY/manifest.tsv (on the login node; fails if an inherited gate has no
#      decision.json yet);
#   2. submits the array (slurm/array.sbatch), at most 4 tasks at once;
#   3. submits the aggregation (cpu_short), which waits for the whole array (afterok).
# Usage, from the repository, after `source slurm/env.sh`:
#   bash slurm/study.sh pilot | gate2a | gate2b | gate3  [SBATCH OPTIONS...]
# Options after the study go to the array, e.g. --exclude=NODE1,NODE2 to avoid bad GPU
# nodes. Resubmitting a study skips finished fits and resumes interrupted ones.
# Gates run one after the other: gate2b inherits gate2a's decision (the sweep was dropped, D28), so
# submit the next one only after reading the previous REPORT.md. gate3 adds the inner fits
# (one task per fold and inner split) that calibrate the intervals (PROJECT_PLAN.md D29).
set -euo pipefail
study="${1:?usage: bash slurm/study.sh STUDY [SBATCH OPTIONS...]}"
shift
opts=("$@")
exclude=$(bash slurm/exclude.sh)  # known bad GPU nodes (slurm/bad_nodes.txt)
cd "$(dirname "$0")/.."
mkdir -p runs/logs
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
set +e
python -m rulfuse.cli manifest --study "$study"
status=$?
set -e
if [ "$status" -eq 3 ]; then  # the study is not needed (its "requires" is not met): no job
    echo "$study: nothing submitted; \$RULFUSE_DATA_ROOT/runs/$study/decision.json says why"
    exit 0
fi
[ "$status" -eq 0 ] || exit "$status"
n=$(( $(wc -l < "$RULFUSE_DATA_ROOT/runs/$study/manifest.tsv") - 1 ))
# A study with inner fits (gate3) has a fifth manifest column, `inner`: -1 for an outer fit. An
# inner fit trains for the step its outer fit kept, so the inner tasks wait for the outer array.
n_outer=$n
if head -1 "$RULFUSE_DATA_ROOT/runs/$study/manifest.tsv" | grep -q "inner"; then
    n_outer=$(awk -F'\t' 'NR > 1 && $5 == -1' "$RULFUSE_DATA_ROOT/runs/$study/manifest.tsv" | wc -l)
fi
# One seed per line (D31): pack PACK lines into one array task, side by side on one GPU
# (default 3; RULFUSE_PACK overrides). With several seeds the seeds already fill the GPU.
n_seeds=$(awk -F'\t' 'NR == 2 {print split($4, s, ",")}' "$RULFUSE_DATA_ROOT/runs/$study/manifest.tsv")
pack=1
[ "$n_seeds" -eq 1 ] && pack="${RULFUSE_PACK:-3}"
chunks() { echo $(( ($2 - $1 + pack) / pack )); }  # array tasks for lines $1..$2
array=$(sbatch --parsable --array=0-$(( $(chunks 0 $((n_outer - 1))) - 1 ))%4 --job-name="$study" \
    ${exclude:+"$exclude"} ${opts[@]+"${opts[@]}"} slurm/array.sbatch "$study" "$pack" 0 $((n_outer - 1)))
array=${array%%;*}
wait_for="$array"
if [ "$n_outer" -lt "$n" ]; then
    inner_array=$(sbatch --parsable --array=0-$(( $(chunks "$n_outer" $((n - 1))) - 1 ))%4 \
        --dependency=afterok:"$array" --job-name="$study-inner" ${exclude:+"$exclude"} \
        ${opts[@]+"${opts[@]}"} slurm/array.sbatch "$study" "$pack" "$n_outer" $((n - 1)))
    inner_array=${inner_array%%;*}
    wait_for="$array:$inner_array"
    echo "$study: inner fits: array $inner_array (tasks $n_outer-$((n - 1)); waits for the outer array)"
fi
agg=$(sbatch --parsable --dependency=afterok:"$wait_for" --job-name="$study-aggregate" \
    slurm/cpu.sbatch python -m rulfuse.cli aggregate --study "$study")
agg=${agg%%;*}
echo "$study: array $array ($n manifest lines, $pack per GPU task), aggregation $agg (waits for the array)"
echo "Excluded nodes: ${exclude:-none}"
echo "Result: \$RULFUSE_DATA_ROOT/runs/$study/REPORT.md and decision.json"
echo "Follow with: bash slurm/status.sh    Cancel: scancel ${wait_for//:/ } $agg"
