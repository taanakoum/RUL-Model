#!/bin/bash
# Baseline B6, OmniFuser-RUL (PROJECT_PLAN.md D20; docs/OMNIFUSER_RUL_PLAN.md §8, §10).
# Submits one cache job, then one GPU job per arm of the modality study (10 folds x 3 seeds
# each), every arm job waiting for the caches. Run on the login node, from the repository:
#   source slurm/env.sh
#   bash slurm/omnifuser_rul.sh force tool chip    # force + the photo types Stage 3 admitted
# A rerun skips finished fits, so resubmitting an arm resumes it.
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo "usage: bash slurm/omnifuser_rul.sh force IMAGE_TYPE [IMAGE_TYPE ...]" >&2
    exit 2
fi
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
mkdir -p runs/logs

prepare=$(sbatch --parsable --job-name=b6-prepare --time=02:00:00 slurm/job.sbatch \
    python -m rulfuse.cli omnifuser-prepare --feature-device cuda)
prepare=${prepare%%;*}
echo "caches: job $prepare"

jobs="$prepare"
while IFS=$'\t' read -r name options; do
    # shellcheck disable=SC2086  # the options are one word per argument
    job=$(sbatch --parsable --dependency=afterok:"$prepare" --job-name="b6-$name" \
        --time=08:00:00 slurm/job.sbatch python -m rulfuse.cli omnifuser-arm $options)
    job=${job%%;*}
    jobs="$jobs $job"
    echo "arm $name: job $job"
done < <(python -m rulfuse.cli omnifuser-study --players "$@" --list)
echo "To cancel all: scancel $jobs"
