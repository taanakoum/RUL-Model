#!/bin/bash
# One-screen overview: your jobs, the stores and feature caches, the Stage-3 report and logs
# that contain errors. It only reads files, so it may run on the login node.
# Usage, from anywhere:  bash slurm/status.sh
cd "$(dirname "$0")/.." || exit 1
ROOT="${RULFUSE_DATA_ROOT:-$WORKDIR/rul_data}"

echo "== Your jobs (ST: PD waiting, R running)"
if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" -o "%.10i %.20j %.2t %.10M %.11l %R"
else
    echo "  (squeue is not available here)"
fi

echo
echo "== Stores and caches under $ROOT"
"${PYTHON:-python3}" - "$ROOT" <<'EOF'
import glob
import json
import os
import sys

root = sys.argv[1]
force = os.path.join(root, "interim", "forces", "meta.json")
print("  force store:", "present" if os.path.isfile(force) else "missing")
for meta in sorted(glob.glob(os.path.join(root, "interim", "segments", "*", "*", "meta.json"))):
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    c = m["counts"]
    print(f"  segment store {m['payload']['dataset']}/{m['signature']}: "
          f"{c['segments']} segments, {c['distinct_segments']} to encode, {c['images']} images")
for meta in sorted(glob.glob(os.path.join(root, "interim", "features", "*", "*", "meta.json"))):
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    shape = next(iter(m["shapes"].values()))
    done = m.get("n_done", 0)
    state = "complete" if m.get("complete") else f"{done}/{shape[0]}"
    print(f"  cache {os.path.basename(os.path.dirname(meta))}: {state}")
for report in sorted(glob.glob(os.path.join(root, "runs", "feature-check", "*.json"))):
    with open(report, encoding="utf-8") as f:
        r = json.load(f)
    print(f"  feature check {os.path.basename(report)}: {json.dumps(r.get('decisions'))}")
EOF

echo
echo "== Logs with an error (time-limit stops are listed separately; read a log with: tail -n 40 FILE)"
found=0
for log in runs/logs/*.out; do
    [ -f "$log" ] || continue
    last=$(grep -E "Traceback|Error:|error:|FAILED|CANCELLED|oom-kill" "$log" | grep -v "DUE TO TIME LIMIT" | tail -n 1)
    if [ -n "$last" ]; then
        echo "  $log: $(echo "$last" | cut -c1-110)"
        found=1
    fi
    if grep -q "DUE TO TIME LIMIT" "$log"; then echo "  $log: stopped at its time limit (resubmit to resume)"; fi
done
if [ "$found" -eq 0 ]; then echo "  none"; fi
echo
echo "== Disk use: data $(du -sh "$ROOT" 2>/dev/null | cut -f1), weights $(du -sh "${HF_HOME:-$WORKDIR/hf_cache}" 2>/dev/null | cut -f1)"
