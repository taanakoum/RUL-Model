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
    shapes = m.get("shapes")  # B6's own caches have no "shapes"
    done = m.get("n_done", 0)
    total = next(iter(shapes.values()))[0] if shapes else "?"
    state = "complete" if m.get("complete") or not shapes else f"{done}/{total}"
    print(f"  cache {os.path.basename(os.path.dirname(meta))}: {state}")
for report in sorted(glob.glob(os.path.join(root, "runs", "feature-check", "*.json"))):
    with open(report, encoding="utf-8") as f:
        r = json.load(f)
    print(f"  feature check {os.path.basename(report)}: {json.dumps(r.get('decisions'))}")

# RUL-Fuse studies: fits done (DONE files) against the manifest
for manifest in sorted(glob.glob(os.path.join(root, "runs", "*", "manifest.tsv"))):
    study = os.path.dirname(manifest)
    with open(manifest, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines()[1:] if line.strip()]
    expected = sum(len(line.split("\t")[3].split(",")) for line in lines)
    done = len(glob.glob(os.path.join(study, "*", "fold-*", "seed-*", "DONE")))
    extra = "  decision.json present" if os.path.isfile(os.path.join(study, "decision.json")) else ""
    print(f"  RUL-Fuse study {os.path.basename(study)}: {done}/{expected} fits{extra}")

# B6 OmniFuser-RUL: fits done per arm (fold x seed folders with a fit.json)
for study in sorted(glob.glob(os.path.join(root, "runs", "omnifuser_rul", "*-*"))):
    if not os.path.isdir(study) or os.path.basename(study) in ("logs",):
        continue
    dataset, _, sig = os.path.basename(study).rpartition("-")
    units = 10
    meta = os.path.join(root, "interim", "segments", dataset, sig, "meta.json")
    if os.path.isfile(meta):
        with open(meta, encoding="utf-8") as f:
            units = json.load(f)["counts"]["units"]
    arms = sorted(glob.glob(os.path.join(study, "*", "arm.json")))
    total = expected_all = 0
    print(f"  B6 study {os.path.basename(study)}: {len(arms)} arms")
    for arm in arms:
        with open(arm, encoding="utf-8") as f:
            seeds = len(json.load(f)["spec"]["seeds"])
        done = len(glob.glob(os.path.join(os.path.dirname(arm), "fold-*", "seed-*", "fit.json")))
        expected = units * seeds
        total, expected_all = total + done, expected_all + expected
        state = "complete" if done >= expected else f"{done}/{expected}"
        print(f"    {os.path.basename(os.path.dirname(arm)):30s} {state}")
    if arms:
        print(f"    all arms: {total}/{expected_all} fits")
    if os.path.isfile(os.path.join(study, "results", "RESULTS.md")):
        print(f"    results: {os.path.join(study, 'results', 'RESULTS.md')}")
EOF

echo
echo "== Logs with an error (time-limit stops are listed separately; read a log with: tail -n 40 FILE)"
found=0
for log in runs/logs/*.out; do
    [ -f "$log" ] || continue
    last=$(grep -E "Traceback|Error:|error:|FAILED|CANCELLED|oom-kill" "$log" | grep -v "DUE TO TIME LIMIT" | tail -n 1)
    if [ -n "$last" ]; then
        node=$(grep -m1 -oE " on [A-Za-z0-9_-]+" "$log" | cut -c5-)
        echo "  $log [${node:-?}]: $(echo "$last" | cut -c1-100)"
        found=1
    fi
    if grep -q "DUE TO TIME LIMIT" "$log"; then echo "  $log: stopped at its time limit (resubmit to resume)"; fi
done
if [ "$found" -eq 0 ]; then echo "  none"; fi
echo
echo "== Disk use: data $(du -sh "$ROOT" 2>/dev/null | cut -f1), weights $(du -sh "${HF_HOME:-$WORKDIR/hf_cache}" 2>/dev/null | cut -f1)"
