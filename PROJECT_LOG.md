# RUL-Fuse: project log

The running record of what was built, how it was checked, what was decided and why. The
plan is [PROJECT_PLAN.md](PROJECT_PLAN.md) (version 2) and its review is
[PLAN_REVIEW.md](PLAN_REVIEW.md). Newest stage last; the status table is kept current.

## Status at a glance (updated 2026-11-04)

| Stage | State | Evidence |
|---|---|---|
| S0 Foundations | **Done** (2026-11-04) | `rulfuse verify-data` ok; tests, ruff and pyright clean |
| S1 Data audit and RUL EDA | **Done** (2026-11-04) | [docs/EDA_RUL.md](docs/EDA_RUL.md); `scripts/eda_rul.py` (47-63 s); 55 tests pass |
| S2 Data pipeline | **Done** (2026-11-04) | Segment store `8d2b69c23453`; `scripts/validate_stage2.py`; [docs/END_TO_END.md](docs/END_TO_END.md); 78 tests pass |
| S3 Features, Ruche, Gate 1 | Next | – |
| S4-S9 | Not started | – |

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

---

## Stage 2: Data pipeline (2026-11-04)

### What was built

| Item | Notes |
|---|---|
| `rulfuse/data/segment_store.py` | `build_segment_store` and `SegmentStore`: `passes.parquet` (532 rows), `segments.parquet` (5,125 rows), `tools.json`, `meta.json`, written atomically. **Signature** = hash of the force-store sha256, the label files' sha256, every `DatasetOptions` field and the segment settings; a store with the same signature is reused. `photo_visible` (D9). The store points into the force store and copies no samples |
| `rulfuse/data/images.py` | OmniFuser's `load_photo` (crop) and the `ImageBank` idea, vendored; per-type preparation (tool edge band → 224 × 672; chip → 448²; work → 224², luminance-standardized to mean 0.5, sd 0.2); ImageNet normalization. numpy, not torch (torch arrives in Stage 3) |
| `rulfuse/data/normalization.py` | `Scaler`, `PCA` (deterministic signs), `BlockTransform` (per-channel blocks, `output_slices` for channel masking), `train_rows`, `tool_shift`, `causal_baseline`, `relative_to_baseline`, `relative_to_first`, `baseline_rows` |
| `rulfuse/data/channels.py` | Fz_rel (offset = mean Fz over the pass's first second, causal) and \|F\| (D4) |
| `rulfuse/features/physics.py` | The first part of encoder H: mean and RMS of the 5 channels per segment, computed once per distinct segment |
| `rulfuse/data/force_store.py` | Refactored: `write_force_store` (from arrays) split from `build_force_store` (from the MAT file), so tests can build small stores |
| CLI | `rulfuse build-store` (every `DatasetOptions` field is a flag, for the §12 arms), `rulfuse store-info` |
| `scripts/validate_stage2.py` | P7 scientific check (tool shift, tool identifiability) and the workpiece standardization check |
| `scripts/end_to_end_trace.py`, `docs/END_TO_END.md` | One prediction point (T5, pass 10, segment 5) traced through every function with real numbers |
| Dependency | `pyarrow` 26 (Parquet tables) |

### The store (signature recorded)

| | Default store `8d2b69c23453` (`D:\RULModel\interim\segments\8d2b69c23453`) |
|---|---|
| Passes | 532 (512 recorded + 20 unrecorded) |
| Segments | 5,125, of which 3,747 are prediction points; 1,378 after the TTF are history and wear targets |
| Windows | 1,338 (passes) and 1,361 (seconds) points |
| Force-missing prediction points | 951 |
| Distinct segments to encode | 3,777 (duplicated recordings shared) |
| Build time | 5.7 s |

### Verification (parts P3-P7)

| Part | Tests | Result |
|---|---|---|
| P3 Timeline | `test_timeline.py`, plus the store's pass table (gap rows have no segments and photo position −1) | Pass |
| P4 Segmentation | `test_segment_store.py`: in the synthetic and the real store, segments **tile every recorded pass exactly** and every inner boundary is a whole 10 s from the trimmed start | Pass (5,125 segments, 0 errors) |
| P5 Labels | The store's targets equal `labels.tool_labels` row by row; rows after the TTF carry no target; real counts asserted (3,747 / 1,338 / 1,361 / 951) | Pass |
| P6 Folds | `test_folds.py` | Pass |
| P7 Normalization: code | `test_normalization.py`: the **poison test** (extreme validation/test rows leave the Scaler, PCA and BlockTransform unchanged); the causal baseline is a running mean that the future never changes; force-missing rows excluded; channel blocks independent | Pass |
| P7 Normalization: science | `validate_stage2.py`: test-tool shift and tool identifiability, absolute against relative features | **Premise not confirmed** (see below); reported, as P7 now specifies |
| Images | `test_images.py`: crop geometry, sizes, standardization (mean 0.5, sd 0.2; exposure invariance), bank = direct preparation; real workpiece photos standardized | Pass |
| Store mechanics | Signature stable across rebuilds and sensitive to options; identical tables on rebuild; reuse without rewrite; duplicates encoded once (`attach="every"` shares distinct ids) | Pass |
| All | 78 tests (61 synthetic in 3.2 s, 17 real-data); ruff and pyright clean | Pass |

### Findings, and the decisions they caused

| # | Finding | Decision | Status |
|---|---|---|---|
| S2-1 | **Relative tokens do not remove tool identity.** On per-segment channel statistics they remove static offsets (T1: 0.97 → 0.34 sd; T4, T6, T10 also lower), but raise the test tool's shift in 6 of 10 folds (median over folds 0.35 against 0.31 sd absolute). Tools are *easier* to tell apart from relative features (nearest-centroid 35 % against 23 %, chance 10 %). With Stage 1's finding (absolute \|F\| relates to wear across tools at 0.70 against 0.59), three signals agree | Educated guess: **default token mode "both"** (absolute + relative halves at the same size: 8 + 8 PCA per channel, 16 + 16 per photo). Gate 2b compares relative, absolute and both as peers (Rule B). P7's science check is now reported, not a gate | Plan updated: §2.6, §5.4, §5.9, §6.1, §6.3, §7.1, §7.5, §7.6, §8, §13, §15 P7, §17.3, App. A |
| S2-2 | **Standardizing the workpiece photos does not remove the leak.** After the per-image luminance standardization the stage-1→2 boundary is still the largest step in 9 of 10 tools for edge strength (share 0.74) and luminance sd; the texture itself changes at the stage change | No change now: the plan's formal test is the Gate-1 probe on encoder features. Expected outcome: the workpiece photos fail and leave the headline model and the Shapley study (the study would then have 3 players: force, tool, chip) | Recorded |
| S2-3 | The plan's worked example used estimated seconds (τ ≈ 923 s, RUL ≈ 904 s) | Replaced with the traced values: τ = 909.7 s, RUL = 917.3 s | Plan §3.2 updated |

### Problems found and fixed

1. **Stale docstring behaviour.** `relative_to_first` returned unchanged values before
   the first valid row instead of 0. Fixed before the tests ran.
2. **Weak test assertion.** A segment-count assertion mixed positions of different tools;
   it was replaced by a per-pass check.
3. **Typing.** Four pyright errors (annotations); fixed.

### Suggested commits (you commit)

1. "Stage 2: segment store, images, normalization, channels":
   - `pyproject.toml`, `rulfuse/cli.py`;
   - `rulfuse/data/{dataset,force_store,segment_store,images,normalization,channels}.py`;
   - `rulfuse/features/`.
2. "Stage 2 tests":
   - `tests/helpers.py`;
   - `tests/test_{segment_store,images,normalization,channels}.py`.
3. "Stage 2 validation and docs":
   - `scripts/{validate_stage2,end_to_end_trace}.py`;
   - `docs/END_TO_END.md`, `PROJECT_PLAN.md`, `PROJECT_LOG.md`.

### Next: Stage 3 (features, Ruche bring-up, Gate 1)

1. The rest of encoder H (band energies, spindle-line energy, spectral centroid and
   entropy, synchronous average).
2. The `encoders` extra (torch, transformers, momentfm, mantis-tsfm, chronos-forecasting)
   and the Project 14 wrappers.
3. The feature cache keyed by distinct segment (3,777) and photo.
4. The Ruche setup and jobs J0-J4.
5. Gate 1: VB-probe ranking, the derived channels, the cross-fitted readouts, and the
   stage-step leakage probe for every photo type.
