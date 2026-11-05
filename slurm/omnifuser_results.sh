#!/bin/bash
# Body of the B6 results job (docs/RUCHE_RUNBOOK.md §12.2). Run it on a CPU node:
#   sbatch --job-name=b6-results slurm/cpu.sbatch bash slurm/omnifuser_results.sh force tool chip
#   sbatch --job-name=b6-results slurm/cpu.sbatch bash slurm/omnifuser_results.sh --after-freeze force tool chip
# The players are "force" and the admitted photo types, as given to slurm/omnifuser_rul.sh.
# It writes runs/omnifuser_rul/<store>/results/ (RESULTS.md, progress, scores, choices,
# curves) and packs everything a later analysis needs into one archive in $WORKDIR: every
# fit's predictions and fit.json, the results folder and the B6 job logs. Test scores are
# sealed until the design freeze (D15): --after-freeze is refused before 2026-12-01.
set -euo pipefail
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
cd "$(dirname "$0")/.."

flags=()
players=()
for arg in "$@"; do
    if [ "$arg" = "--after-freeze" ]; then flags+=("$arg"); else players+=("$arg"); fi
done
if [ "${#players[@]}" -eq 0 ]; then
    echo "usage: bash slurm/omnifuser_results.sh [--after-freeze] force IMAGE_TYPE ..." >&2
    exit 2
fi

rulfuse omnifuser-results --players "${players[@]}" ${flags[@]+"${flags[@]}"}

results="$RULFUSE_DATA_ROOT/runs/omnifuser_rul"
mkdir -p "$results/logs"
cp runs/logs/b6-*.out "$results/logs/" 2>/dev/null || true
stamp=$(date +%Y%m%d-%H%M)
archive="$WORKDIR/b6-results-$stamp.tar.gz"
tar -czf "$archive" -C "$RULFUSE_DATA_ROOT/runs" omnifuser_rul
(cd "$WORKDIR" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
echo "archive: $archive ($(du -h "$archive" | cut -f1))"
echo "copy home from the laptop (PowerShell, docs/RUCHE_RUNBOOK.md §13):"
echo "  scp \"\${Ruche}:$archive\" \"\${Ruche}:$archive.sha256\" D:\\RULModel\\transfer\\"
