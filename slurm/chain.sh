#!/bin/bash
# Submits the same job several times in a row, for work longer than the 24 h limit
# (OmniFuser's slurm/chain.sh). Each part starts when the previous one has ended (finished,
# failed or out of time) and resumes where it stopped: rulfuse caches and fits skip what is
# already done.
#
# Usage, from the repository folder:
#   bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND [ARGUMENTS...]
set -euo pipefail

if [ "$#" -lt 2 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND ..." >&2
    exit 2
fi
parts=$1
shift
mkdir -p runs/logs

job=$(sbatch --parsable "$@")
job=${job%%;*}
jobs=$job
echo "part 1/$parts: job $job"
for ((part = 2; part <= parts; part++)); do
    job=$(sbatch --parsable --dependency=afterany:"$job" "$@")
    job=${job%%;*}
    jobs="$jobs $job"
    echo "part $part/$parts: job $job (waits for the previous part)"
done
echo "To cancel this chain: scancel $jobs"
