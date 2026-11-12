#!/bin/bash
# Stage 4 on Ruche (PROJECT_PLAN.md §17.2, docs/RUCHE_RUNBOOK.md §15):
#   J4  the seven sanity checks (gpua100; writes runs/sanity/report.json)
#   J1  the pilot: fold T1, F1 at the defaults, 3 seeds (times a fit; slurm/study.sh pilot)
# Both run at once. Stage 5 (bash slurm/study.sh gate2a) starts once every check passes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
# Options given to this script go to both jobs, e.g. --exclude=NODE.
exclude=$(bash slurm/exclude.sh)  # known bad GPU nodes (slurm/bad_nodes.txt)
j4=$(sbatch --parsable --job-name=J4-sanity --time=06:00:00 ${exclude:+"$exclude"} "$@" slurm/job.sbatch \
    python -m rulfuse.cli sanity --device cuda)
echo "J4 sanity checks ${j4%%;*}"
bash slurm/study.sh pilot "$@"
