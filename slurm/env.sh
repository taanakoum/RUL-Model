# Loads the rulfuse environment on Ruche, inside a job or in a terminal:
#   source slurm/env.sh
# Edit the two names below once, after the installation (docs/RUCHE_RUNBOOK.md §7).
# Based on OmniFuser's and Project 14's slurm/env.sh.

CONDA_MODULE="anaconda3/2023.09-0/none-none"  # a conda module listed by: module avail anaconda miniconda
CONDA_ENV="rulfuse"                           # the environment created in §7.2

# Jobs submitted with --export=NONE start with an empty environment; load the system
# profile so that the module command exists.
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module purge
module load "$CONDA_MODULE"
# Ruche asks for "source activate", not "conda activate" (which edits ~/.bashrc).
source activate "$CONDA_ENV"

export RULFUSE_DATA_ROOT="$WORKDIR/rul_data"  # raw/, interim/ (stores, caches), runs/
export HF_HOME="$WORKDIR/hf_cache"            # model weights, downloaded on the login node
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export HF_HUB_OFFLINE=1                   # compute nodes read weights from the cache only
    export TRANSFORMERS_OFFLINE=1
fi
export PYTHONUNBUFFERED=1  # logs show progress while a job runs
export PYTHONIOENCODING=utf-8
