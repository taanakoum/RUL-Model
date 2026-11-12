#!/bin/bash
# Prints "--exclude=node1,node2" for every known bad GPU node, or nothing if there is none:
# slurm/bad_nodes.txt (committed) plus $RULFUSE_DATA_ROOT/bad_gpu_nodes.txt (appended on
# Ruche by the GPU probe of array.sbatch and job.sbatch).
#   sbatch $(bash slurm/exclude.sh) ...
cd "$(dirname "$0")/.." || exit 1
nodes=$(cat slurm/bad_nodes.txt "${RULFUSE_DATA_ROOT:-/nonexistent}/bad_gpu_nodes.txt" 2>/dev/null \
    | tr -d '\r' | sed 's/#.*//' | tr -s ' \t' '\n' | grep -v '^$' | sort -u | paste -sd, -)
if [ -n "$nodes" ]; then echo "--exclude=$nodes"; fi
