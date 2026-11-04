# RUL-Fuse: project log

The running record of what was built, how it was checked, what was decided and why. The
plan is [PROJECT_PLAN.md](PROJECT_PLAN.md) (version 2) and its review is
[PLAN_REVIEW.md](PLAN_REVIEW.md). Newest stage last; the status table is kept current.

## Status at a glance (updated 2026-11-04)

| Stage | State | Evidence |
|---|---|---|
| S0 Foundations | **Done** (2026-11-04) | `rulfuse verify-data` ok; tests, ruff and pyright clean |
| S1 Data audit and RUL EDA | **Done** (2026-11-04) | [docs/EDA_RUL.md](docs/EDA_RUL.md); `scripts/eda_rul.py` (47-63 s); 55 tests pass |
| S2 Data pipeline | Next. Already built in S1: trim, timeline, segments, labels, folds. Still to build: the segment store, photo availability and fold-level normalization | – |
| S3-S9 | Not started | – |

**Answered 2026-11-04:**

1. The gap-aware time axis is confirmed: T2 and T4 kept cutting through their missing
   runs (D12 is final).
2. The Nonastreda paper does not explain the workpiece-photo change, so the default
   stands: per-image standardization plus the Gate-1 leakage probe.

**Needs you:** nothing blocking. Plan §19 questions 3-7 run on their defaults.

## How to reproduce

```powershell
# environment (once)
uv venv --python 3.11 D:\RULModel\.venv
uv pip install --python D:\RULModel\.venv\Scripts\python.exe -e ".[test,eda,dev,hub]"
# data (once): copy D:\Project14\data\raw\nonastreda (without spec\) to D:\RULModel\raw\nonastreda
D:\RULModel\.venv\Scripts\rulfuse.exe verify-data
D:\RULModel\.venv\Scripts\rulfuse.exe build-force-store
D:\RULModel\.venv\Scripts\rulfuse.exe hub-check
# checks
D:\RULModel\.venv\Scripts\python.exe -m pytest
D:\RULModel\.venv\Scripts\ruff.exe check .
D:\RULModel\.venv\Scripts\pyright.exe
```

---

## Stage 0: Foundations (2026-11-04)

### What was built

| Item | Where | Notes |
|---|---|---|
| Data root | `D:\RULModel` | `raw\nonastreda` (force MAT file, labels, 512 tool, 512 chip and 512 workpiece photos, `MANIFEST.json`), `interim\`, `runs\`, `hf_cache\`. Outside OneDrive (plan §14.1) |
| Python environment | `D:\RULModel\.venv` | CPython 3.11.14 (uv-managed, as Project 14); numpy, scipy, Pillow, matplotlib, pytest, ruff, pyright, huggingface_hub. No torch yet: Stage 3 adds the `encoders` extra |
| Repository | `RUL Model/` | `git init -b main`; `.gitignore` excludes `Data/`, `runs/`, arrays; **nothing committed** (plan §19 question 2 default) |
| Package `rulfuse` | `rulfuse/` | `pyproject.toml` (Python 3.11-3.12; extras test, eda, dev, hub; `rulfuse` command) |
| `rulfuse/__init__.py` | | Windows WMI guard, vendored from OmniFuser |
| `rulfuse/paths.py` | | `RULFUSE_DATA_ROOT` (default `D:\RULModel`) and the folders under it |
| `rulfuse/resources.py` | | Idle CPUs (vendored from OmniFuser) and free memory (new); `check_resources` refuses heavy steps on a busy machine |
| `rulfuse/data/nonastreda.py` | | Labels, photo paths, the MAT table, duplicate groups. OmniFuser's checks vendored; all 512 passes kept |
| `rulfuse/data/force_store.py` | `D:\RULModel\interim\forces` | Reads the MAT file **once** and writes `signals.npy` (378 distinct recordings, float32, memory-mapped, 452 MB), `index.csv`, `groups.json`, `meta.json` |
| `rulfuse/data/trim.py` | | OmniFuser's span rule (vendored unchanged) and the causal variant (5 s reference) |
| `rulfuse/data/segments.py` | | 10 s segments, remainder rule (D7) |
| `rulfuse/data/timeline.py` | | Gap-aware and recorded axes; duplicate attachment `latest`, `every`, `dedup` |
| `rulfuse/data/labels.py` | | The five TTF rules; isotonic wear (full and causal); prediction points; RUL targets in passes and seconds |
| `rulfuse/data/folds.py` | | Leave-one-tool-out folds (vendored) |
| `rulfuse/data/dataset.py` | | `load_dataset(options)`: one call for labels, store, trims, timelines and RUL labels; every option is a §12 sensitivity axis |
| `rulfuse/baselines.py`, `rulfuse/metrics.py` | | B0-B2 with causal elapsed time; MAE, MSE, RMSE, bias, MAE_eq, coverage, Winkler |
| `rulfuse/verify.py`, `rulfuse/hub.py`, `rulfuse/cli.py` | | `rulfuse resources`, `verify-data`, `build-force-store`, `hub-check` |
| Tests | `tests/` | 51 tests: 38 synthetic (2.7 s), 13 on the real data (marked `data`, skipped without the store) |

### Verification

| Check (plan §15 part) | Result |
|---|---|
| P1 `verify-data`: every publisher-manifest file (1,539, spectrograms excluded) present with the publisher's size and sha256 | **0 missing, 0 mismatched** |
| P1 repository `Data/` against the publisher (515 files) | Identical, except `labels.csv`, whose only change is CRLF line endings (513 lines identical). `verify-data` now recognises line-ending-only differences |
| P1 labels | 512 passes; per tool T1 40, T2 56, T3 52, T4 40, T5 48, T6 60, T7 56, T8 52, T9 52, T10 56 |
| P1 MAT file against labels.csv (OmniFuser's checks) | All 512 labels agree; shapes (3, samples); no non-finite value |
| P1 force store | 512 records, **378 distinct recordings, 132 duplicate groups** (127 B3+B4, 4 B1+B2, 1 B1-B4), 134 non-latest members. The same groups as OmniFuser's (test against its `records.csv`). Built in 11 s with 1.8 GB free memory |
| P2 vendored OmniFuser trim | Reproduces OmniFuser's `trim_start`/`trim_end` on **512 of 512** passes |
| P3 gaps | T1 leading run 1; T2 interior run 10; T4 interior runs 6-9 |
| P6 folds | 10 disjoint, cyclic folds |
| Stage-0 access check (`hub-check`): the 10 planned checkpoints | All exist. Not gated: Mantis-8M, MOMENT-1 base/large, Chronos-T5-small, AST, DINOv2 S/B, SigLIP 2. **Gated (manual approval): DINOv3 S and B** |
| Chronos API | `chronos-forecasting` 2.3.2: `embed()` exists on the T5 `ChronosPipeline` (`chronos/chronos.py`), not on the Bolt or Chronos-2 pipelines. The plan's `amazon/chronos-t5-small` with `pipeline.embed` is right |
| Code quality | ruff: all checks passed; pyright: 0 errors |

### Decisions taken (defaults and educated guesses)

| Decision | Taken | Why |
|---|---|---|
| Python version | 3.11.14 | The same as Project 14's venv; the encoder libraries support it; Ruche uses 3.11 too |
| Raw data source | Copied from `D:\Project14\data\raw\nonastreda` | It holds every file with the publisher's sha256 in `MANIFEST.json`; the repository's `Data/` lacks the chip and workpiece photos. Both copies are verified identical |
| Commits | Repository initialised; nothing committed | Plan §19 question 2 default |
| Duplicates detection | Hash candidates, confirmed by exact equality | Same groups as OmniFuser's exact comparison, faster |
| Gap duration | Median of the tool's recorded passes **before** the gap; 97 s for a leading gap | Causal (D14). Found while building: the plan's "tool median" would use later pass durations for the cumulative time after a gap. Plan updated |
| Elapsed passes as a baseline input | Completed passes + elapsed seconds / mean duration of the tool's earlier passes | Causal (D14): the current pass's duration is unknown until it ends |
| DINOv3 | Kept optional; no licence requested | It is gated; §6.2 lists it as optional. Requesting it needs your Hugging Face account |

### Problems found and fixed

1. **Line endings** made `labels.csv` look different from the publisher's. Content is
   identical; `verify-data` now reports line-ending-only differences separately.
2. **Gap duration used the future** (the median of all the tool's passes). It now uses
   the passes before the gap.
3. A bare `python` hung in this session (the Windows Store alias or the WMI issue noted in
   OmniFuser). Every command uses `D:\RULModel\.venv\Scripts\python.exe`.

### Machine notes

- About 1.8-2.3 GB of free memory, held mostly by VS Code windows and memory compression.
- The MAT file is read once (the force store); everything else memory-maps it.
- CPUs were idle (6-7 of 8) during the heavy steps.

---

## Stage 1: Data audit and RUL EDA (2026-11-04)

Report: [docs/EDA_RUL.md](docs/EDA_RUL.md), with 8 figures in `docs/eda_rul/`. The
numbers are in `D:\RULModel\runs\eda\eda_summary.json` and its CSV tables.

### What was built

| Item | Notes |
|---|---|
| `scripts/eda_rul.py` | Every Stage-1 analysis and figure; checks free memory first. Runtime 47-63 s |
| `rulfuse/audit.py` | `gap_wear_jumps` (are run gaps unrecorded cutting?), `stage_steps` (does a statistic encode the stage label?), `mean_luminance`. Reused in Gate 1 as the leakage probe |
| `tests/test_audit.py` | Synthetic: the jump ratio counts missing runs; the step probe separates a label-tied statistic from a wear trend. Real data: T4's ratio in [4.5, 6.5] and T2's in [1.5, 3]; workpiece brightness steps at stage 2 in ≥ 9 of 10 tools |
| Figures | Built with the dataviz skill's reference palette (slots 1-3, documented as all-pairs safe). Node.js is not installed, so its validator script could not run; the palette values were used unchanged. Every figure was rendered and inspected; two layout fixes were made (shared x-label; legend moved off the data) |

### Results, and the decisions they caused

| # | Finding | Decision | Status |
|---|---|---|---|
| E1 | **T2 and T4 kept cutting through their missing runs.** Wear-jump ratio 5.6 for T4 (expected 5 if cut, 1 if not) and 2.4 at the local rate for T2 (expected 2). T1 starts at 46 µm against a 32 µm median first run (ambiguous) | Gap-aware axis D12 kept. T1's leading run not counted (no RUL change either way) | **Confirmed by you (2026-11-04)** |
| E2 | **Causal trim agrees with OmniFuser's:** starts within 200 samples on 99.4 % of passes. The vendored OmniFuser rule reproduces OmniFuser's bounds 512/512. The 3 disagreements are slow entry ramps, 0.26-0.32 s apart. The 4 passes flagged in `EDA.md` §5 are unchanged | P2 accepted; the causal trim is the default | Done |
| E3 | **The workpiece photos encode the stage label.** Brightness 0.27-0.28 in stage 1, 0.50-0.52 from the first run of stage 2, in every tool. The stage-1→2 boundary is the largest run-boundary step in 10 of 10 tools (96 % of the range); texture also steps at stage 3. Tool photos and force show no such step (0-1 of 10) | Workpiece photos: per-image standardization; a **Gate-1 stage-step leakage probe for every photo type** (pass: ≤ 3 of 10 tools); excluded if they fail. Chip blue tint (6 of 10 at stage 3) gets the same probe | Default stands: the paper does not explain the change (you, 2026-11-04); plan updated |
| E4 | Raw Fz has no wear information (ρ ≈ 0); Fz_rel has some (ρ 0.28 with wear). RMS of \|F\|, Fx, Fy and the 0-50 Hz band follow wear strongly within tools (median ρ with RUL −0.92 for \|F\|), less across tools (−0.47 to −0.59) | Confirms D4 (all channels + Fz_rel) | – |
| E5 | The absolute force level relates to wear across tools at least as well as the relative one (\|F\|: 0.70 against 0.59) | Default (relative tokens) unchanged until Gate 2b; noted as a serious candidate | Plan note §5.9 |
| E6 | Exact baselines with the causal trim (gap axis, segment ends): B0 4.13 / 412 s; B1 5.17 / 513 s; **B2 3.82 / 468 s**; B2's 90 % intervals cover only 0.42 (passes) and 0.50 (seconds) | S1 bar 3.82 / 468 s; S2 bar Winkler 49.4 / 5,388 s | Plan updated |
| E7 | 134 passes force-missing (128 in block 3); **951 of 3,747 pre-TTF points (25 %)** | Reported separately in every evaluation | Plan updated |
| E8 | Workpiece photos are single-channel greyscale; a first "blue share" statistic was an artifact of an epsilon | Colour statistics only for colour images | Fixed in the script |

### Plan changes caused by Stage 1

- **D4:** workpiece standardization and the photo leakage probe.
- **D12:** gap duration from the tool's earlier passes.
- **S1:** 468 s instead of 467 s.
- **§2.2 and §2.7:** exact numbers with the causal trim (cutting times, seconds windows,
  B1, B2, the B2 interval coverage).
- **§2.4:** 951 force-missing points.
- **§2.5:** the expected jump ratios.
- **§2.6:** the workpiece-leakage trap.
- **§5.7:** workpiece standardization.
- **§5.9:** the absolute-vs-relative note.
- **§6.2, item 8:** the leakage probe.
- **§15 P9:** probe acceptance.
- **§17.3:** decision 17.
- **§19:** question 1b.

### Verification at the end of Stage 1

| Check | Result |
|---|---|
| `pytest` | 55 passed in 16 s: 40 synthetic, 15 on the real data |
| `ruff check .`, `ruff format` | All checks passed |
| `pyright` | 0 errors |
| Plan numbers reproduced by code | TTF tables (5 rules × 2 axes), points 3,747, windows 1,338 / 1,361, B0-B2: all asserted in `tests/` |

### Problems found and fixed

1. **A key-filtering bug in the photo statistics:** the `tool` column was taken for the
   tool-photo kind. Fixed.
2. **The greyscale "blue share" artifact (E8).** Fixed.
3. **Figure layout:** overlapping x-labels in the force panels, and a legend covering
   points. Fixed and re-inspected.

### Next: Stage 2

**Already in place:** trim, timeline, segments, labels and folds, with tests.

**Still to build:**

- the segment store (`interim/segments/<signature>/` with `segments.parquet`, or `.npz` if
  pyarrow is not added);
- photo availability times in the store;
- fold-level standardization and PCA fitting (needs features, so its tests use synthetic
  features until Stage 3);
- the causal tool baseline b_u(t);
- `rulfuse build-store`.

**Exit:** P3-P7 accepted on the store.

---

## Repository note (2026-11-04)

- **You commit personally.** You made commits `c4a8718` ("Project Skeliton") and
  `fb849dc` ("Data Audit and Tests") during this work, which answers plan §19 question 2.
  I do not commit.
- **A `.gitignore` bug, fixed.** The unanchored pattern `Data/` also matched the package
  folder `rulfuse/data/`, because git on Windows ignores case (`core.ignorecase=true`).
  The whole data layer was therefore **missing from both commits**. The patterns are now
  anchored (`/Data/`, `/runs/`); `rulfuse/data/` shows as untracked and the raw `Data/`
  stays ignored.

Suggested commits for the current working tree:

1. "Fix .gitignore anchoring; add the data layer":
   - `.gitignore`;
   - `rulfuse/data/` (`nonastreda`, `force_store`, `trim`, `segments`, `timeline`,
     `labels`, `folds`, `dataset`).
2. "Stage 1: audit module, EDA script, tests":
   - `rulfuse/audit.py`, `scripts/eda_rul.py`;
   - `tests/test_audit.py`, `tests/test_loader.py` (line-ending test fix).
3. "Stage 1 docs: EDA report, figures, plan sync, project log":
   - `docs/EDA_RUL.md`, `docs/eda_rul/`;
   - `PROJECT_PLAN.md`, `PROJECT_LOG.md`.

---

## Decisions received (2026-11-04)

| Question (plan §19) | Your answer | Effect |
|---|---|---|
| 1. Were T2's run 10 and T4's runs 6-9 cut but not recorded? | **Yes** | The gap-aware axis (D12) is final. TTFs stay T2 = 44 and T4 = 38; the recorded axis remains only a tier-1 sensitivity arm. T1's leading run is still not counted (it changes no RUL target) |
| 1b. Does the Nonastreda paper explain the workpiece-photo change at the stage boundaries? | **No** | The cause stays unknown, so the default stands: per-image standardization, the Gate-1 stage-step leakage probe for every photo type (pass: ≤ 3 of 10 tools), and exclusion if it fails |
| 2. Commits | **Only you commit** | I never commit or push; suggested commit groups go at the end of each stage's entry |

Updated: `PROJECT_PLAN.md` (banner, D12, §2.5, §18, §19), `docs/EDA_RUL.md` (§2, §6,
§10), `PLAN_REVIEW.md` (§3). No code or result changes: the pipeline already used these
defaults.
