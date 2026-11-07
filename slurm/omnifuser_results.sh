#!/bin/bash
# Body of the B6 report job (docs/RUCHE_RUNBOOK.md §12.2), run on a CPU node. Submitted by
# slurm/omnifuser_rul.sh after the training; to run it again by hand:
#   sbatch --job-name=b6-report slurm/cpu.sbatch bash slurm/omnifuser_results.sh force tool chip
#   sbatch --job-name=b6-report slurm/cpu.sbatch bash slurm/omnifuser_results.sh --unseal force tool chip
# The players are those of the run; options after "--" must be the run's own (they select
# its fits). Test results are included on or after the design freeze (2026-12-01, D15);
# --unseal includes them earlier, and REPORT.md records that it was used.
# It writes runs/omnifuser_rul/<store>/<arm>/report/ (REPORT.md, run_summary.json, CSV
# tables, figures) and packs the run (every fit's predictions and fit.json, the report and
# the B6 job logs) into one archive in $WORKDIR, with its sha256.
set -euo pipefail
: "${RULFUSE_DATA_ROOT:?source slurm/env.sh first}"
cd "$(dirname "$0")/.."

flags=()
players=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    if [ "$1" = "--unseal" ]; then flags+=("$1"); else players+=("$1"); fi
    shift
done
if [ "$#" -gt 0 ]; then shift; flags+=("$@"); fi
if [ "${#players[@]}" -eq 0 ]; then
    echo "usage: bash slurm/omnifuser_results.sh [--unseal] force IMAGE_TYPE ... [-- OPTIONS]" >&2
    exit 2
fi

rulfuse omnifuser-report --players "${players[@]}" ${flags[@]+"${flags[@]}"}

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
