#!/bin/bash
# Job J2 body (run by slurm/job.sbatch on a GPU node): the two Stage-3 feature caches, with
# the Mantis scale and determinism checks (PROJECT_PLAN.md §5.8, §6.2). Resumable: a rerun
# encodes only what is missing.
set -euo pipefail
echo "=== signals: Mantis-8M on every 512-sample sub-window of every distinct segment"
rulfuse extract-features --kind signal --encoder mantis8m --device cuda --chunk 64
echo "=== images: DINOv2-B tokens of every tool, chip and workpiece photo"
rulfuse extract-features --kind image --encoder dinov2b --device cuda --batch 16
echo "=== caches"
du -sh "$RULFUSE_DATA_ROOT"/interim/features/*/* 2>/dev/null || true
