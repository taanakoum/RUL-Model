# RUL-Fuse: project log

The running record of what was built, how it was checked, what was decided and why. The
plan is [PROJECT_PLAN.md](PROJECT_PLAN.md) (version 2) and its review is
[PLAN_REVIEW.md](PLAN_REVIEW.md). Newest stage last; the status table is kept current.

## Status at a glance (updated 2026-11-07)

| Stage | State | Evidence |
|---|---|---|
| S0 Foundations | **Done** (2026-11-04) | `rulfuse verify-data` ok; tests, ruff and pyright clean |
| S1 Data audit and RUL EDA | **Done** (2026-11-04) | [docs/EDA_RUL.md](docs/EDA_RUL.md); `scripts/eda_rul.py` (47-63 s); 55 tests pass |
| S2 Data pipeline | **Done** (2026-11-04) | Segment store `8d2b69c23453`; `scripts/validate_stage2.py`; [docs/END_TO_END.md](docs/END_TO_END.md); 78 tests pass |
| S3 Core/adapter refactor, features, feature check | **Code done** (2026-11-05); **Ruche jobs J0-J3 waiting for you** | Refactored store identical to Stage 2's; 111 tests (109 pass, 2 need the weights: they run in J0); ruff and pyright clean; [docs/RUCHE_RUNBOOK.md](docs/RUCHE_RUNBOOK.md) |
| B6 OmniFuser-RUL (D20) | **Version 6 ready** (2026-11-08): one fixed setting (GRU memory over OmniFuser's encodings of each 96-step block; monotonic penalty moving only the earlier prediction), one full run of 10 folds × 3 seeds, then a report with per-tool test results in two inference modes, compared; **waiting for Ruche** after Stage 3's photo decision | [docs/OMNIFUSER_RUL_PLAN.md](docs/OMNIFUSER_RUL_PLAN.md) O9, O12, O13, §7, §14; runbook §12.1-§12.2 |
| S4-S9 | Not started | – |

**Answered 2026-11-04:**

1. The gap-aware time axis is confirmed: T2 and T4 kept cutting through their missing
   runs (D12 is final).
2. The Nonastreda paper does not explain the workpiece-photo change, so the default
   stands: per-image standardization plus the Gate-1 leakage probe.

**Needs you:**

1. Commit and push (suggested groups at the end of the Stage 3 section), then run Stage 3 on
   Ruche: [docs/RUCHE_RUNBOOK.md](docs/RUCHE_RUNBOOK.md) Part A once, then
   `bash slurm/stage3.sh` (§11). I cannot log in to Ruche: it needs your password.
2. Copy the report and the caches home (§13) and tell me; I then log the decisions.
3. B6 (after J3): the pilot, then `bash slurm/omnifuser_rul.sh force <admitted types>`;
   when every arm is complete and before 2026-12-01, the results job (runbook §12.2); copy
   the archive home (§13) and tell me. The test read follows from 2026-12-01.

Plan §19 questions 3-7 run on their defaults.

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

---

## Plan change: fixed encoders and a transferable design (2026-11-04, no code yet)

Your request: one time-series and one image encoder without screening runs, and a model
that generalizes beyond Nonastreda as a versatile backbone. Plan sections changed: D16,
D17, §4.2, §4.4, §5.4, §5.7-§5.9, §6.1-§6.3, §6.8-§6.10 (new), §7, §11, §12, §13, §14,
§15 P8-P9, §16, §17, §18, §19, App. A.

| Change | Before | Now | Why |
|---|---|---|---|
| Encoders (D16) | Gate 1 screened 7 force and 5 photo encoders (Ruche jobs J3, J4) | **Mantis-8M** for every force channel, **DINOv2-B/14** for every photo type, frozen; alternatives documented, not run; MOMENT-1-base only as a fallback if a go/no-go check fails | Your instruction; Mantis is built for frozen classification features and is channel-independent and small; DINOv2 keeps local texture and is not gated |
| PCA (D17) | Per-fold PCA (16 per channel, 32 per photo) fitted on the training tools | **No PCA in the model.** LayerNorm + a learned, shared low-rank projection (rank 16), part of the reusable backbone. PCA kept only for analysis and baseline B4 | A PCA basis is a property of one dataset's training tools; refitting it elsewhere changes every input's meaning, so a trained model could not transfer |
| Physics features (D17) | Up to ~20 per channel, some Nonastreda-specific (66.7 Hz spindle line, period-15 average, fixed-Hz bands) | **6 generic level descriptors** per channel (log RMS and log peak of the AC part, kurtosis, crest, mean, slope; Project 14's functions), absolute and relative to the healthy start. Machine-specific ones dropped; the full set only for B4 | Normalized foundation encoders cannot see absolute level, and the level is Stage 1's strongest wear signal (ρ 0.70); generic descriptors transfer, machine parameters do not |
| Channels | 5 fixed PCA blocks | A **channel set** with descriptor embeddings (sensor type, axis, derivation), pooled per segment | Any sensor set (accelerometer, microphone) fits without changing the model |
| Stage 3 | Features on Ruche, Gate-1 encoder ranking, derived-channel decision (Nov 10-17) | Features on the **laptop CPU**, a pass/fail feature check and the leakage probe (Nov 10-14); Ruche setup moves to Stage 4 | No screening, no Ruche feature jobs |
| LoRA, per-fold PatchTST | Stretch goals | Documented alternatives | Fewer runs; no encoder training |
| Decisions | 17 | 10 | Encoder, derived-channel and LoRA decisions removed |
| Ruche budget | 65-80 A100-hours | 45-55 | No feature jobs; Shapley with 3 players if the workpiece fails (7 coalitions) |
| Stage 6 | – | A plain-language explanation added to §17.2 | Your request |

**Code impact (for later).**

- Stage 2's `normalization.py` keeps `PCA` and `BlockTransform`, used now only by
  baseline B4 and the analysis.
- `Scaler` is now used for the level descriptors; `causal_baseline` is unchanged.
- No code was changed in this step.

---

## Plan change: seconds only, 3 seeds, a generic core with a dataset adapter (2026-11-04, no code yet)

Your request: 3 random seeds instead of 5; RUL predicted only in seconds, because other
datasets have no passes; as few dataset-specific models and functions as possible. Plan
sections changed: D3, D6, D7, D9, D11, D18 and D19 (new), §1, §2.6, §2.7, §3, §4.2-§4.5,
§5, §6.1-§6.10, §6.11 (new), §7, §8, §9, §10, §11, §12, §13 (L20 new), §14.2-§14.4, §15
(P22 new), §16 J10, §17 Stages 1, 3, 5 and 6, App. A. Backup of the previous version:
the session scratchpad, `PROJECT_PLAN.before-generic.md`.

| Change | Before | Now | Why |
|---|---|---|---|
| Output (D6) | Two Student-t heads, RUL in seconds and in passes | **One head, RUL in seconds of operation** | Passes exist only in Nonastreda |
| Horizon (D3) | k = 15 passes and 1,500 s | **H = 1,500 s**, a dataset setting; window RUL ≤ H (1,361 points) | Same window in one unit |
| Selection and gate metric | MAE_eq = ½ (MAE_pass + MAE_sec / 100 s) | **MAE in seconds**; Rule A threshold 10 s (H/150), the "neither unit worsens" clause dropped; also reported as MAE / H | One unit |
| Bar to beat (§2.7) | B2 3.82 passes / 468 s | **B2 468 s**, B0 412 s, B1 513 s; B2 intervals 0.50 coverage, Winkler 5,388 s | Seconds only |
| Seeds (D19) | 3 for gates, 5 for the final model | **3 everywhere** | Your instruction; J10 falls from 250 to 150 fits, the Ruche total to about 40-50 A100-hours |
| Architecture (D18, §6.11) | Nonastreda logic spread through the pipeline | **Generic core + one Nonastreda adapter.** The adapter gives units with recordings (start, duration, channels with descriptors), observations (images with type and availability time, optional health indicator, optional stage labels) and a failure time. The core never refers to passes, runs, blocks, photo types or flank wear | Transfer to other datasets |
| Fz_rel channel | Fz minus its mean over the pass's first second | **Removed.** Every channel gets a 7th generic descriptor: the mean change since the recording started | The same information without a Nonastreda-only channel |
| Channels | Fx, Fy, Fz, Fz_rel, \|F\| | **Fx, Fy, Fz, \|F\|** (magnitude of every 3-axis group: a generic rule) | As above |
| Usage input | Completed passes, τ, seconds into the pass | **Operating time τ and seconds since the recording started** | No pass count |
| F0 experts | Usage, force, and one GRU per photo type (5) | **Usage, signals, images (3)**; one image GRU with type embeddings | Expert count no longer depends on the dataset's photo types |
| Wear-class head (CORAL), oracle R2, single-unit control | Planned | **Removed** | Tied to Nonastreda's three classes or to passes |
| Wear target | Flank wear VB̃ | **Optional health indicator (HI)**; here VB̃. Phase A is skipped when a dataset has no HI | Not every dataset measures wear |
| Thresholds | In passes (late > 1 pass, RUL bins of 3 passes, ±2 passes) | **Fractions of H** (late > H/15 = 100 s, 300 s bins, ±200 s) | Comparable across datasets |
| Proof of transfer | – | **P22:** an import-graph test (the core never imports the adapter) and a synthetic toy adapter (2 accelerometer channels at 2 kHz, one image type, H = 600 s) run one fold end to end | Makes "generic" testable |
| Stage 3 | Features and Gate 1 | **First task: refactor the Stage 0-2 code** into `rulfuse/core/` and `rulfuse/adapters/nonastreda/` with the toy adapter and P22; the 78 tests move and stay green | The code still follows the old structure |

**Interpretation.** "Not folds" in your message was read as "not passes" (the fold
protocol, leave-one-tool-out, is unchanged).

**What stays Nonastreda-specific, inside the adapter:** MAT/CSV loading and verification,
the span-rule trim, duplicate groups (D8), the gap-aware timeline (D12), the five TTF
rules, the photo crops and the workpiece standardization, flank-wear smoothing, and the
`tool_label` stages for the leakage probe.

**Code impact (for Stage 3).**

- `labels.py` computes RUL in passes and seconds; the passes columns become unused and
  are dropped in the refactor. The segment store `8d2b69c23453` gets a new signature then.
- `channels.py` drops `Fz_rel` and `fz_offset`; `features/physics.py` gains the drift
  descriptor.
- `metrics.mae_eq` and the passes baselines are removed; `baselines.py` keeps seconds.
- No code was changed in this step.


---

## Stage 3: core/adapter refactor, features and the feature check (2026-11-05)

Your instructions this stage: start implementing Stage 3; download the encoders and run the
experiments **on Ruche, not on the laptop**; delete any weights already downloaded that
are not needed, and verify; write a comprehensive Ruche runbook learning from OmniFuser.

### Weights on the laptop: none were downloaded

Checked on 2026-11-05: no Mantis-8M or DINOv2-B snapshot exists on this machine from this
project. `D:\RULModel\hf_cache` holds only a 5.8 KB metadata file left by the Stage-0
`hub-check`; `~\.cache\huggingface` and `~\.cache\torch` hold other projects' models
(dated before this project). Project 14's cache (`D:\Project14\hf_cache`, which includes
Mantis-8M) belongs to Project 14 and was not touched. Nothing needed deleting.

What *was* installed on the laptop: the `encoders` packages into `D:\RULModel\.venv`
(torch 2.15.0+cpu, transformers 5.22.1, mantis-tsfm 1.1.0). They are code, not weights;
the tests use them to check the wrappers on tiny random networks, and Stage 4's sanity
checks need torch. They can be removed with
`uv pip uninstall --python D:\RULModel\.venv\Scripts\python.exe transformers mantis-tsfm`
if you prefer (the two wrapper tests then skip).

### What was built

| Part | Files | Content |
|---|---|---|
| Generic schema (D18) | `rulfuse/core/schema.py` | `Dataset`, `Unit`, `Recording` (start, duration, `SignalRef`, HI, stage, group, `attrs`), `Image` (type, availability time), `ChannelSpec`, `ImageSpec`, `Settings` (H, rates, B, S), `SignalSource`; `validate()` lists every broken rule |
| Generic core | `rulfuse/core/{segments,targets,channels,folds,normalization,images,store}.py` | Segmentation and sub-windows; RUL in seconds, cap, window; magnitude of 3-axis groups and the 7 level descriptors (drift new); leave-one-unit-out in the adapter's order; normalization; image preparation from a spec; the generic segment store (`recordings`, `segments`, `images` tables) |
| Nonastreda adapter | `rulfuse/adapters/nonastreda/{load,force_store,trim,timeline,failure,images,dataset,verify,audit}.py` | Everything Nonastreda-specific, moved from `rulfuse/data/`; `to_units()` maps passes to recordings, photos to images, flank wear to the HI, `tool_label` to the stage, runs to groups |
| Toy adapter | `rulfuse/adapters/toy.py` | 2 accelerometer channels at 2 kHz, one image type, H = 600 s, no gaps or duplicates |
| Encoders | `rulfuse/features/encoders.py` | Mantis-8M and DINOv2-B wrappers vendored from Project 14 (dinov2-base instead of -small, one 2 × 4 grid for all types, injectable networks); weight-free stand-ins for tests |
| Caches | `rulfuse/features/cache.py` | Signal cache per distinct segment (embeddings mean+sd, descriptors, spectral H) and image cache per photo; resumable; foreign signatures refused |
| Handcrafted H | `rulfuse/features/handcrafted.py` | 11 spectral features relative to the model rate (8 octave bands, centroid, entropy, RMS of the difference); the generic mean/RMS diagnostics |
| Probes | `rulfuse/features/probe.py` | Ridge with α by inner leave-one-unit-out (SVD path); cross-fitted readouts; fold-wise go/no-go with the shuffled control |
| Feature check | `rulfuse/features/check.py` | Scale and determinism checks, go/no-go per feature set, readouts, leakage probe (LOO PCA + readout), HI reference, decisions |
| Generic audit, baselines, metrics | `rulfuse/audit.py`, `baselines.py`, `metrics.py` | `stage_steps` on any (unit, group, stage) rows; B0-B2 in seconds from the store; `mae_eq` removed |
| CLI | `rulfuse/cli.py` | `--dataset` everywhere; new `extract-features`, `feature-check`, `baselines`, `--features-root` |
| Ruche | `slurm/{env.sh,job.sbatch,cpu.sbatch,checks.sbatch,features.sh,stage3.sh,chain.sh,status.sh,prefetch_weights.sh}`, `scripts/pack_for_ruche.ps1`, `.gitattributes` | OmniFuser's and Project 14's patterns; one command submits the Stage-3 chain |
| Runbook | `docs/RUCHE_RUNBOOK.md` | Setup, Stage 3 step by step with the expected log lines, reading the report, copying home, troubleshooting, OmniFuser lessons |
| Tests | `tests/test_{schema,transfer_contract,store,targets,failure,encoders,cache,probe}.py` (new); the others ported | 111 tests |
| Scripts | `scripts/{eda_rul,end_to_end_trace,validate_stage2}.py` | Ported to the adapter and the generic store; [docs/END_TO_END.md](docs/END_TO_END.md) regenerated |

### Verification

| Check | Result |
|---|---|
| Refactor regression: the new store (`nonastreda/3e357b5a23ff`) against Stage 2's (`8d2b69c23453`), row by row | All 5,125 segments equal in unit, recording, segment, signal presence, e, τ, prediction flag, RUL and capped targets (max difference 0.0), window, baseline flag, distinct id, sample bounds and source; failure times equal; image visibility agrees on 400 random points |
| Counts | 512 recordings, 5,125 segments, 3,747 prediction points, 1,361 window points, 951 signal-missing points, 3,777 distinct segments, 1,536 images |
| Baselines from the generic store (P16) | B0 412.0 s, B1 512.5 s, B2 467.8 s, B2 coverage 0.503, Winkler 5,388: equal to plan §2.7 |
| P22 transfer contract | The core, `features`, `audit`, `baselines` and `metrics` import nothing from `rulfuse.adapters` (AST test) and use none of the Nonastreda vocabulary in code (identifiers and strings, docstrings excluded). The toy adapter runs schema → store → features → probes → readouts → leakage probe → baselines without any core change; the probe beats the mean on the toy and the shuffled control collapses |
| P8 encoders and caches | Mantis and DINOv2 wrappers on random networks: shapes, determinism, CLS + 2 × 4 grid pooling equal to a manual computation; vectorized descriptors equal Project 14's functions; known RMS, crest, kurtosis, slope, drift; level survives scaling (log RMS moves by log 2) while kurtosis does not; resume gives the same cache; a foreign signature is refused; duplicates share a row |
| P9 probes | Ridge recovers known coefficients; a cross-fitted readout is unchanged when its own unit's labels change, and labels outside training are never read; shuffled control collapses; the leakage PCA is fitted without the projected unit |
| Real-data smoke run of J2's commands with the weight-free stand-ins (40 segments, 6 photos, into a scratch folder, deleted after) | Works; about 77 segments/s of CPU-side work (reading, descriptors, spectral) on the laptop |
| `docs/END_TO_END.md` trace | Same numbers as Stage 2 (T5R3B2: trim 430-96,478, τ = 909.7 s, RUL = 917.3 s); the Fz drift descriptor equals the old Fz_rel mean (−10.37 N) |
| Tests, lint, types | 111 tests: 109 pass, 2 skip (`weights`: they load the real checkpoints and run in Ruche job J0); ruff and pyright clean |

**Not yet run (needs Ruche):** the Mantis and DINOv2 caches, the scale and determinism
checks, the go/no-go probes, readouts and leakage probe on the real features. Their
expected log lines and acceptance rules are in the runbook (§11.2, §11.4).

### Decisions taken (defaults and educated guesses)

| Decision | Choice | Why |
|---|---|---|
| Where Stage 3 runs | Ruche jobs J0 (checks, `cpu_short`), J0g (GPU check, `gpu_test`), J2 (features, `gpua100`), J3 (feature check, `cpu_short`), chained by `slurm/stage3.sh` | Your instruction; the CPU work goes to CPU partitions so no GPU waits on it |
| What goes to Ruche | The raw copy only (1.2 GB tar with sha256); Ruche rebuilds the force store, the segment store and the caches | Results do not depend on files made on Windows; the store signature must match the laptop's, which J0 shows |
| Weights | Prefetched on the login node, config + safetensors only (about 380 MB; DINOv2's duplicate `pytorch_model.bin` skipped); none on the laptop; MOMENT only if the force check fails | Your instruction; offline jobs |
| Drift descriptor | Segment mean − mean of the recording's first second | The causal generic version of Fz_rel; equals it on Fz |
| Image tokens | One 2 × 4 grid for every image type (9 tokens), DINOv2-B (768) | A fixed token count for any image type (generic); plan §5.8 |
| Leakage rule | A type leaks if a stage boundary is its largest group step in more than 3 of 10 units, over its first 3 PCs and its readout | Plan §6.2; PCA fitted without the projected unit, so no unit's own data shapes its basis |
| Go/no-go | The probe's mean validation MAE over the 10 folds below the training-mean predictor's | Plan §6.2 |
| Determinism tolerance | Relative 1e-3 | A100 kernels and TF32 convolutions change the last digits between batch sizes |
| Handcrafted H | Generic spectral features relative to the model rate; the spindle-line and synchronous-average features dropped | D17, D18; H is for B4 and diagnostics only |
| Store location | `interim/segments/<dataset>/<signature>/` | Several datasets can coexist |

### Problems found and fixed

1. **Float32 drift reference.** A test that recomputed the drift in float32 differed from
   the cache by 1e-4 N; the cache accumulates the head mean in float64, which is right.
   The test now does the same.
2. **Vocabulary test caught docstrings.** The first version flagged docstring lines that
   explain where an idea came from; it now inspects identifiers and string constants
   through the AST.
3. **`Dinov2Config` has no `intermediate_size`** in transformers 5; the test uses
   `mlp_ratio`.
4. **The old END_TO_END.md said T5R3B2 has `tool_label` 1**; `labels.csv` says 2. Corrected.
5. Two unspecified-behaviour risks in the cache (fancy-assignment order for the first row
   of each distinct segment; the sub-window axis order) were rewritten before any test
   ran.

### Plan changes this stage

- §6.2 costs, §12 arm 2, §14.1, §14.2, §14.3, §16.1-§16.4, §17.1, Stages 3 and 4: the
  features and the feature check run on Ruche (J0, J0g, J2, J3 in Stage 3); Stage 4 keeps
  pilot J1 and the laptop sanity checks on the copied caches.

### Suggested commits (you commit)

1. "Stage 3: generic core and Nonastreda adapter":
   - `rulfuse/core/`, `rulfuse/adapters/`, `rulfuse/audit.py`, `rulfuse/baselines.py`,
     `rulfuse/metrics.py`, `rulfuse/cli.py`, `pyproject.toml`;
   - the removal of `rulfuse/data/`, `rulfuse/verify.py`, `rulfuse/features/physics.py`.
2. "Stage 3: encoders, caches and the feature check":
   - `rulfuse/features/{encoders,cache,handcrafted,probe,check}.py`.
3. "Stage 3 tests":
   - `tests/` (new and ported files; `test_labels.py` and `test_segment_store.py`
     removed).
4. "Ruche runbook and job scripts":
   - `slurm/`, `scripts/pack_for_ruche.ps1`, `.gitattributes`, `docs/RUCHE_RUNBOOK.md`.
5. "Stage 3 docs and scripts":
   - `scripts/{eda_rul,end_to_end_trace,validate_stage2}.py`, `docs/END_TO_END.md`,
     `PROJECT_PLAN.md`, `PROJECT_LOG.md`.

### Next

1. You: push, then Part A of the runbook and `bash slurm/stage3.sh` (§11).
2. Copy home the report and caches (§13); I log decision 1 (photo types kept) and the
   force go/no-go, and update the plan if a type is excluded.
3. Stage 4 (training engine) can start in parallel: it needs only the schema and the cache
   layout, which are fixed.

## Baseline B6: OmniFuser-RUL (2026-11-05)

A separate model and an additional baseline (D20): OmniFuser's architecture on this
project's targets, inputs, folds, weights, selection and metrics. Plan:
[docs/OMNIFUSER_RUL_PLAN.md](docs/OMNIFUSER_RUL_PLAN.md) (version 2, reviewed); §14 there
is the full implementation record.

### Built

| Part | Where |
|---|---|
| OmniFuser v1.1: `encode()` (outputs bit-identical to v1.0) and `image_views` | OmniFuser repository, branch `release-1.1` (tag `v1.1` to push) |
| Step tables, descriptor and photo-feature caches, windows | `rulfuse/omnifuser_rul/data.py` |
| Model: OmniFuser `encode()` + an RUL head; input switches for force, photos, usage | `rulfuse/omnifuser_rul/model.py` |
| One fit and one arm (resumable), study arms, analysis | `rulfuse/omnifuser_rul/{train,study}.py` |
| Scores, seed ensembles, Shapley values, two-versus-one tests, test seal (D15) | `rulfuse/omnifuser_rul/aggregate.py` |
| Commands `omnifuser-prepare`, `omnifuser-arm`, `omnifuser-study`, `omnifuser-analyse` | `rulfuse/cli.py` |
| Ruche: one cache job and one job per arm | `slurm/omnifuser_rul.sh`; `docs/RUCHE_RUNBOOK.md` §12.1 |
| Dependencies | the `omnifuser` extra in `pyproject.toml`; `TORCH_HOME` in `slurm/env.sh`; EfficientNet-B0 in `slurm/prefetch_weights.sh` |

### Verification

All checks V-1 to V-11 of the plan pass (plan §14). The project's suite: 130 tests pass,
2 skip (the Mantis and DINOv2 weight tests); ruff and pyright are clean. The laptop
environment needed torchvision: 0.29.2 is the newest published, and it runs EfficientNet-B0
correctly with this environment's torch 2.15.

### Problems found and fixed

1. Finished fits were never reused: the saved settings came back from JSON with lists
   where the settings had tuples. Now compared in JSON form.
2. The same in the photo-feature cache's key (the tool crop), found by the real-data run.
3. Photos switched off still depended on the data passing an all-zero first slot; the model
   now zeroes them itself.
4. Epoch selection let a 2-value average at the first epoch win; now only complete triples
   of 3 consecutive values count.

### Suggested commits (you commit)

1. OmniFuser: "v1.1: encode() and image views" (branch `release-1.1`, staged), merge to
   `main`, then `git tag -a v1.1 -m "OmniFuser v1.1" && git push origin main v1.1`.
2. This project: "B6 OmniFuser-RUL baseline": `rulfuse/omnifuser_rul/`, `rulfuse/cli.py`,
   `tests/test_omnifuser_rul.py`, `tests/test_transfer_contract.py`, `slurm/omnifuser_rul.sh`,
   `slurm/env.sh`, `slurm/prefetch_weights.sh`, `pyproject.toml`, `docs/RUCHE_RUNBOOK.md`,
   `docs/OMNIFUSER_RUL_PLAN.md`, `PROJECT_PLAN.md`, `PROJECT_LOG.md`.



---

## Review: the OmniFuser-RUL baseline and the runbook (2026-11-05)

Your request: review the changes that accommodate the OmniFuser model (commit `7a3d274`)
and the new runbook; make the results of training and testing extractable when the runs
finish on Ruche; update the docs if everything is ready for Ruche training.

### Verdict

The B6 design and code are sound and faithful to `docs/OMNIFUSER_RUL_PLAN.md`:

- the store's strict photo-availability rule, and windows that only look back;
- the shared point weights, standardization on the training tools, and the
  centre-of-triple epoch selection;
- the test seal (D15) and exact Shapley values;
- OmniFuser `v1.1` is published (tag `v1.1` = `eac0fed` on `main`), so Ruche installs the
  code tested here.

It was **not ready for Ruche** because of one blocker and three gaps, all fixed now (R-1
to R-4 below). After the fixes, the Stage-3 and B6 jobs are ready to submit.

### Findings and fixes

| # | Severity | Finding | Fix |
|---|---|---|---|
| R-1 | Blocker | `tests/test_omnifuser_rul.py` imported OmniFuser at collection. The runbook's setup did not install it, so Stage-3 job J0 (which runs the whole suite) would fail at collection, and J2 and J3 would never start. Reproduced by hiding OmniFuser: exit code 2, zero tests run | The module skips without OmniFuser or torchvision; the setup (§7.2) installs the `omnifuser` extra, so J0 tests B6 on Ruche too |
| R-2 | Major | torchvision was to be installed as `torchvision==<the version matching your torch>` (a placeholder) | torch and torchvision installed together from the CUDA 12.6 index; one `pip install -e ".[test,encoders,omnifuser]"`; an upgrade block for an existing environment |
| R-3 | Major | Nothing enforced that B6 uses exactly the photo types Stage 3 admitted (O5), or that Stage 3 had run | Guards in `slurm/omnifuser_rul.sh`, tested on fake Stage-3 outputs (an excluded type refused; admitted types submitted; no store refused) |
| R-4 | Major | Results stopped at a validation JSON, with no completeness check, no comparisons recorded before the test read, no report and no way home | `rulfuse omnifuser-results` (`rulfuse/omnifuser_rul/report.py`) and `slurm/omnifuser_results.sh`; runbook §12.2 and §13 |
| R-5 | Minor | J0 did not prove that EfficientNet-B0 loads offline | J0 checks `$TORCH_HOME` and loads it |
| R-6 | Minor | `status.sh` did not show B6 | A `B6 study` block with fits per arm |
| R-7 | Minor | A failed arm listing still submitted the cache job | Arms listed first; an empty list stops the script |
| R-8 | Note | The data loader runs in the main process; a fit may take longer than the 3-minute estimate | The pilot measures it; the runbook says when to report |
| R-9 | Note | The B6 entry said "130 tests pass, 2 skip"; the commit had 130 tests (128 pass) | Corrected in the B6 plan §14 |

### Extracting the results (what you run on Ruche)

1. When `bash slurm/status.sh` shows every B6 arm `complete`:
   `sbatch --job-name=b6-results slurm/cpu.sbatch bash slurm/omnifuser_results.sh force tool chip`.
2. It writes `runs/omnifuser_rul/nonastreda-<store>/results/`:
   - `RESULTS.md`;
   - `progress.json`, `curves.csv`;
   - the validation scores per tool against B2;
   - `choices.json`, the comparisons fixed on validation.

   It also packs everything into `$WORKDIR/b6-results-<date>.tar.gz` with its sha256.
3. Copy it home and unpack it into `D:\RULModel\runs` (runbook §13); the report can be
   rebuilt on the laptop in seconds.
4. **Test results:** the test predictions are in the archive, but scoring them is sealed
   until 2026-12-01 (D15, your decision O3). From that date the same job with
   `--after-freeze` adds:
   - the test table;
   - the Shapley values and the paired tests;
   - a check against `choices.json`.

   It refuses an incomplete study.

### Verification

| Check | Result |
|---|---|
| Suite | 132 tests: 130 pass, 2 skip (the Mantis and DINOv2 checkpoint tests, which run in J0); ruff and pyright clean |
| Without OmniFuser installed (Ruche before §7.2's extra) | The suite passes with the B6 module skipped (exit 0) |
| `report.py` tests | Fake studies with known errors: exact scores, progress, choices, seal before the freeze, refusal of an incomplete study, Shapley identities after it |
| End to end on the real store | An 11-arm fake study (330 fits, one missing) through `rulfuse omnifuser-results`: the missing fit is named, scores and choices are written, the test read is refused before the freeze with a clean message (exit 2); the fake runs were deleted |
| Scripts | `bash -n` on every SLURM script; LF line endings; `status.sh` and the `omnifuser_rul.sh` guards dry-run on fake trees |
| Weights on the laptop | None added: the EfficientNet-B0 checkpoint in `~\.cache\torch` dates from 2026-10-14 (the OmniFuser project); the B6 caches under `D:\RULModel\interim\features` (8 MB) are features |

### Suggested commit (you commit)

"B6 review: Ruche readiness and results extraction":
- `tests/test_omnifuser_rul.py`;
- `rulfuse/omnifuser_rul/report.py`, `rulfuse/cli.py`;
- `slurm/checks.sbatch`, `slurm/omnifuser_rul.sh`, `slurm/omnifuser_results.sh`,
  `slurm/status.sh`;
- `docs/RUCHE_RUNBOOK.md`, `docs/OMNIFUSER_RUL_PLAN.md`, `PROJECT_PLAN.md`,
  `PROJECT_LOG.md`.


---

## B6 version 3: more training examples, a memory and a temporal loss (2026-11-06)

Your request:

- check whether B6 has enough training examples, and if not, shorten the segments,
  with RUL and TTF labels that match and nothing broken, including OmniFuser's input
  dimensions;
- a GRU or LSTM memory, so each new segment uses the previous ones, in B6 and in the new
  model;
- a loss penalty when a predicted RUL is higher than the previous segment's;
- test, validate, document and iterate until B6 is ready for Ruche.

### Findings

| Question | Answer |
|---|---|
| Training examples under version 2 | 2,823-3,343 prediction points per fold (median 2,956) at 10 s, all from 8 trajectories. Low for a deep fusion model |
| With shorter segments | 5 s: 5,802; 2 s: 14,572; **1 s: 29,123** per fold (median). The 1 s store has 50,378 segments and 36,919 prediction points; its longest tool has 6,830 steps |
| Do the labels need changes? | No. At 5, 2 and 1 s, every one of the 5,125 points of the 10 s store is found with identical τ and RUL (largest difference 0.0 s), and the failure times are equal. The TTF is the end of the TTF pass for every segment length, and RUL = τ* − τ follows from each segment's end |
| OmniFuser's input dimensions | OmniFuser requires exactly 96 steps per window and takes any number of channels; B6 always builds (B, 96, 31) windows, so nothing changes with S. Its own loss optimizes only the force and class terms (C2F alignment and orthogonality are logged, not trained), so B6's head replacement stays faithful; RevIN is off |
| Memory in the new model (RUL-Fuse) | Its model is Stage 4, not built yet. F1 attends to the full history by design; the plan now adds the shared GRU memory to F1 and to F0's signal expert (D21). The memory module is built and tested now |

### What was built

| Part | Where |
|---|---|
| Shared causal memory (GRU or LSTM over block means; step t reads only complete blocks) | `rulfuse/models/memory.py` (D21) |
| Shared temporal losses: monotonic hinge relu(ŷ_later − ŷ_earlier), slope consistency below the cap | `rulfuse/models/losses.py` (D22) |
| Temporal stability metrics: rising share, mean rise, \|Δŷ + Δτ\| | `rulfuse/metrics.py` |
| Grid mask: which points of a fine store are the 10 s points | `rulfuse/core/store.py` |
| Segment length as a setting of the adapter, the toy and the CLI (`--segment-s`; 10 s default, 1 s for B6) | `rulfuse/adapters/`, `rulfuse/cli.py` |
| B6 version 3: 1 s store, pairs sampling, device-side batching, memory with dropout variants, temporal loss, grid selection and scoring, `on_grid` predictions, loss terms in `fit.json`, 13 arms (two ablations), stability columns, scoring only fits with matching settings, photo cache shared across segment lengths | `rulfuse/omnifuser_rul/` |
| Study options passed through to every job (`-- --pairs-per-epoch 1024`) | `slurm/omnifuser_rul.sh`, `slurm/omnifuser_results.sh` |
| Docs | B6 plan version 3 (O7-O11, §14); `PROJECT_PLAN.md` D20-D22, §6.4, §6.7, Gate 2b controls, job table; runbook §11.2, §12.1, §12.2 |

### Better ideas adopted (documented in the B6 plan)

1. **Score on the 10 s points.** B6 trains at 1 s but is scored and selected on the
   project's 10 s points. A 1 s store contains them exactly, so the comparison with
   RUL-Fuse and B0-B5 is unchanged.
2. **A slope term with the hinge.** Below the cap the true RUL falls by exactly Δτ, so a
   slope-consistency term is added to your hinge. RUL-Fuse's plan already had it. The two
   models now share one temporal loss with the same weights (λ_mono 1.0, λ_cons 0.5).
3. **Pairs with mixed lags.** Each pair is the previous segment half the time, and up to
   200 s apart otherwise, so both immediate and longer-range rises are penalized.
4. **A memory over blocks.** The memory runs over 10 s blocks of the 1 s steps, so its cost
   stays linear (about 700 blocks per tool instead of 6,800 steps). The steps of the
   unfinished block are inside OmniFuser's 96-step window.
5. **A fixed pair budget.** Each epoch draws 2,048 pairs, so 10× more points do not cost
   10× more time.
6. **Ablations.** Two arms (without the memory; without the temporal terms) measure what
   each change adds.

### Verification

| Check | Result |
|---|---|
| Labels at 1 s (V-13) | As in the findings; real-data test: B2 on the 1 s store's grid = 467.8 s, as on the 10 s store |
| Input dimensions (V-14) | Toy windows (B, 96, 17) at 10 s and 2 s; OmniFuser's parameter count identical; real data (B, 96, 31) |
| Memory causality (V-15) | GRU and LSTM, blocks 1 and 4: later steps never change earlier states; only complete blocks are read; padding is never averaged. **Found and fixed:** a LayerNorm on the block means cancelled a rise shared by all inputs, which is the common rise of the force descriptors with wear. It was removed, and a test now checks that such a rise registers |
| Switches reach the memory (V-16) | Force, photos and usage switched off change neither the window output nor the memory |
| Temporal loss (V-17) | Hinge and slope by hand; on the toy, the fits with the terms rise less than the same fits without them (2 seeds) |
| Scoring only matching fits (V-18) | **Found and fixed:** scores read every prediction file in an arm folder, whatever its settings, so a partly rerun arm would have mixed fits. Scores now use only fits whose `fit.json` matches |
| Real data, laptop CPU (V-11) | `omnifuser-prepare` builds the 1 s store and descriptor cache in 38 s. A 2-epoch fit of force + tool + chip on fold 1 takes 38 s, trains on 30,103 points, has 296,181 trainable parameters, and predicts all 3,818 validation points of T2, flagging exactly its 393 grid points (116 in the window). B2 on that grid is 228.2 s, as on the 10 s store. The validation MAE after 2 short epochs was 445 s, a smoke value |
| Job scripts | `bash -n` and LF on all; `omnifuser_rul.sh` dry-run with a fake `sbatch`: 14 submissions (the cache job and 13 arms), the study options on every job, the printed results command carrying them; the guards as before |
| Suite | 151 tests: 149 pass, 2 skip (the Mantis and DINOv2 checkpoint tests, which run in J0); the B6 tests take about 3 minutes on the laptop CPU; ruff and pyright clean |

### Cost and the pilot

390 fits (13 arms × 10 folds × 3 seeds), expected 3-6 minutes each on an A100: about 20-40
GPU-hours, against 16 for version 2. The pilot's time per fit decides, before the study and
by cost only, whether the pair budget is halved (runbook §12.1).

### Laptop files

- The 1 s store (`nonastreda/6e0cb095967b`, 1.2 MB) and its descriptor cache (4 MB) were
  built for the smoke run and kept under `D:\RULModel\interim`, so results copied home can
  be scored on the laptop.
- The photo features of the earlier smoke run were reused under the new key (same
  encoder, specs and image table, asserted), and the copy under the old key was deleted.
- No model weights were added.

### Suggested commit (you commit)

Your commit `f46453b` ("Memory Component for Model") already holds the memory, the losses,
the grid mask, the segment-length setting and B6's data, model and training. What remains
is one commit, "B6 version 3: matching-fit scoring, study options, docs":

- `rulfuse/omnifuser_rul/{aggregate,report,study}.py`;
- `slurm/omnifuser_rul.sh`, `slurm/omnifuser_results.sh`;
- `tests/test_models.py`, `tests/test_omnifuser_rul.py`;
- `docs/OMNIFUSER_RUL_PLAN.md`, `docs/RUCHE_RUNBOOK.md`, `PROJECT_PLAN.md`,
  `PROJECT_LOG.md`.


---

## B6 version 4: one full run and a report (2026-11-07)

**Your request:** no experiments on the OmniFuser implementation, and no switching of
settings for comparison (for example GRU against LSTM). Choose one setting, run one full
training, extract meaningful test results for a report, and include the loss with the
monotonic penalty. Update the runbook.

**The setting** (fixed; decision O12 of the B6 plan):

| Part | Value |
|---|---|
| Inputs | Force (7 level descriptors × 4 channels, a signal flag), every photo type admitted by Stage 3, usage |
| Steps and points | 1 s segments, 96-step windows; scored on the project's 10 s points (RUL ≤ 1,500 s) |
| Model | OmniFuser v1.1's encoder + a **GRU** memory (hidden 64, 10-step blocks) + a linear head |
| Loss | Point-weighted MSE + **1.0 × monotonic penalty** + 0.5 × slope consistency |
| Training | Adam 1e-4, 30 epochs of 2,048 pairs, 10 folds × 3 seeds = 30 fits |

**Built:**

- **Report command and figures.** `rulfuse omnifuser-report`
  (`rulfuse/omnifuser_rul/run_report.py`). It writes `REPORT.md`, `run_summary.json`, the
  CSV tables and three figures. The figures follow the chart guidance: the reference
  palette, B6 in blue, B2 in orange, truth in ink.
- **Report metrics** in `rulfuse/metrics.py`: the asymmetric score of plan §10.2, the late
  and early rates, the share within ±20 %, error by true-RUL range, and a bootstrap
  interval over tools.
- **One submission.** `slurm/omnifuser_rul.sh` now submits one training job (`b6-train`)
  and the report job (`b6-report`) chained after it with `afterok`. It keeps the Stage-3
  guards.
- **Report job body.** `slurm/omnifuser_results.sh` runs the report and packs the archive.
- **Documentation.**
  - The runbook's §12.1-§12.2 are rewritten for the single run, as are the job table, the
    cheat sheet, the troubleshooting rows and §13.
  - §7.2 installs `eda` for the figures.
  - The B6 plan reaches version 4 (O12; §8 marked not run).
  - The project plan's D20, its B6 baseline row and the job table are updated.

**Test results and the seal.** You decided earlier (O3, D15) that B6's test scores stay
sealed until the design freeze on 2026-12-01. The report keeps that by default: before the
freeze it has the run and the validation results. From 2026-12-01 the same report includes
the test results. To read them earlier for your report, run it with `--unseal`; it then
opens with a note recording the date. That choice is yours.

**Verification:**

- **Report tests.** A fake run checks every number: per-tool MAE and bias, the seed spread,
  B2 on the same points, the gain and tools improved, error by range, the seal, `--unseal`,
  the read after the freeze, and the figures.
- **Metrics by hand.** The new metrics are checked against hand computations.
- **Real store.** On the real 1 s store, the report's baselines are B2 467.8 s, B0 412.0 s
  and B1 512.5 s, exactly plan §2.7.
- **Figures.** They were rendered and inspected.
- **Submission.** A dry run with a fake `sbatch` submits 2 jobs, chained, with the options
  passed to both.
- **Suite.** 154 tests: 152 pass, 2 skip (the Mantis and DINOv2 checkpoint tests, which
  run in J0); ruff and pyright are clean.

**Cost:** 30 fits at an expected 3-6 minutes each, about 1.5-3 GPU-hours, in one 8-hour
job; the report takes minutes on `cpu_short`.

**Suggested commit (you commit):** "B6 version 4: one full run and its report":

- `rulfuse/omnifuser_rul/run_report.py`, `rulfuse/metrics.py`, `rulfuse/cli.py`;
- `tests/test_metrics.py`, `tests/test_omnifuser_rul.py`;
- `slurm/omnifuser_rul.sh`, `slurm/omnifuser_results.sh`;
- `docs/RUCHE_RUNBOOK.md`, `docs/OMNIFUSER_RUL_PLAN.md`, `PROJECT_PLAN.md`,
  `PROJECT_LOG.md`.

Uncommitted changes of the previous step (matching-fit scoring, study options) are included
if not yet committed.

## B6 version 5: the penalty fix and two inference modes (2026-11-07)

**Your question.** Could the penalty on rising predictions lock later predictions below a
too-low early one?

- **At prediction time: no.** Each prediction comes from its own inputs. The model never
  sees its past predictions, and nothing clamps them.
- **In training: yes, with version 4's loss.** Weight 1.0 acted on both predictions of a
  pair. When the model could not correct an earlier prediction, the later one settled
  exactly at the stuck value.

**What you asked for:**

1. Apply the fix and make sure training behaves correctly.
2. Two inference modes: the current one, and one where previous segments guide the new
   prediction.
3. Whether the second mode is valid in production.
4. A way to compare the two modes.

**What was done:**

- **Training (D22, O9).**
  - In both temporal terms the later prediction is the fixed reference: detached in
    `rulfuse.models.losses`, shared with RUL-Fuse.
  - The weights are 1/15 (hinge) and 0.05 (slope), down from 1.0 and 0.5.
  - The batch loss is `train.pair_loss`, called by `fit()`.
- **Inference modes (D23, O13).**
  - `rulfuse/inference.py` is generic. The context mode ages the running estimate by the
    cutting time elapsed and blends in each new prediction: a prediction made Δ seconds ago
    keeps the weight e^(−Δ/120 s). Near the cap, ageing never goes below the new
    prediction.
  - `aggregate.with_mode` applies it to each fit's predictions of a tool.
  - The current mode stays the primary result. Both come from the same fits.
- **Comparison.**
  - `run_report.compare_modes`: context − current per metric, with bootstrap intervals; the
    window-MAE gain with tools better and worse; a two-sided Wilcoxon test.
  - Verdict rule, fixed in advance: the context mode is better if the gain is at least
    10 s and its interval lies above 0; the current mode under the mirror condition;
    otherwise neither.
  - Outputs: a section in REPORT.md for test and validation, `mode_comparison.csv`, a
    `mode` column in the per-unit CSVs, both modes in the three figures, and a CLI line.
- **Production.** The context mode is valid: causal, one number per tool. It needs a reset
  at a tool change and must persist across machine stops (it ages by cutting time). Its
  limits:
  - real sudden changes reach the output with a 120 s time constant;
  - it removes noise, not a whole-tool bias;
  - where the model's predictions below the cap fall more slowly than 1 s per s, the
    context sits up to 120 s lower: pessimistic, and costly on the whole-life MAE.

  B6 plan §7 has the full discussion.

**Verification** (the code `fit()` runs):

| Check | Version 4's loss | Version 5 |
|---|---|---|
| Earlier prediction stuck 450 s / 600 s too low → the later one (truth 700 s) | dragged to 450 s / 300 s | stays at 700.0 s |
| Later prediction stuck 300 s too high → the earlier one | +112.5 s | +73.6 s |
| Noise the model cannot remove → level shift | 0 | +9.7 to +14.1 s |

- **Toy fits, by hinge weight (15 epochs × 512 pairs).** Mean rise / window MAE / bias:
  - 0: 2.22 / 32.9 / +27.5 s;
  - 1/15: 2.11 / 34.6 / +30.5 s;
  - 0.2: 1.66 / 41.7 / +40.8 s;
  - 1.0: 0.62 / 62.4 / +62.4 s.

  A strong penalty removes rises at a large accuracy cost; the run's weight costs almost
  nothing and removes few. The context mode handles rises at inference instead.
- **Two tests changed, from these measurements.**
  - The shuffled-target control's limit: 0.8 → 0.7 × the constant predictor, plus at least
    1.5 × the fit. With the terms off it scores 0.78 ×. The old strong penalty had raised
    every fit above 0.8 ×, the real one included: 89.6 s against 69.9 s.
  - The "fewer rises" fit comparison is replaced by a spy inside a real fit: on every
    batch, the temporal gradient reaches earlier predictions only, with the spec's weights.
- **The context mode.**
  - Unit tests: causal; an exact trajectory is unchanged, cap and uneven steps included;
    it recovers from a low start as e^(−t/T); it removes jitter without sliding off the
    cap; each fit, tool and role is filtered on its own.
  - Choices made before any result, on synthetic errors over the real tools' time axes:
    T = 120 s; the cap rule costs a perfect model 0 s, against 7.4 s for pausing.
- **Fake run on the real 1 s store.**
  - B0, B1 and B2 on the test points: 412.0, 512.5 and 467.8 s (plan §2.7).
  - Window MAE 78.9 → 74.1 s, verdict "neither"; RMSE 88.8 → 79.4 s; whole-life MAE
    73.1 → 97.5 s (the flat-prediction cost).
  - The figures were inspected.
- **Suite.** 164 tests: 162 pass, 2 skip (the Mantis and DINOv2 checkpoint
  tests, which run in J0); ruff and pyright are clean.

**The run on Ruche is unchanged:** `bash slurm/omnifuser_rul.sh force <admitted types>`.
No fit exists yet, so no result is affected. The new weights change the fits' settings
signature, so old fits would not be mixed in.

**Commits (you commit).** Most of version 5 is in your commit e77a0d3 ("Revising Loss
Function for Training Stability"). What followed it is still uncommitted; suggested
commit "B6 version 5: test fixes from the toy measurements, docs":

- `tests/test_omnifuser_rul.py`: the control's limit and the in-fit spy test;
- `rulfuse/inference.py`: the flat-prediction note;
- `docs/OMNIFUSER_RUL_PLAN.md` (version-5 notes, V-7, V-17, the flat-prediction limit),
  `docs/RUCHE_RUNBOOK.md` (test count), `PROJECT_LOG.md`.

## B6 version 6: the GRU memory reads OmniFuser's encodings (2026-11-08)

**Your request.** The GRU should take the encoded time series and encoded images, in
training and inference. Before, it read the raw step inputs: force descriptors, its own
projection of the photo features, and a photo flag, in 10-step means.

**Design** (B6 plan O8, §5):

- **Input.** One GRU input per block of 96 steps (one OmniFuser window): OmniFuser's encoded
  force series and encoded photos for that block, from the same encoders and weights as the
  head's window. Each is averaged over the block, giving 64 numbers.
- **Causality.** OmniFuser mixes all steps inside its window, so a block is encoded only
  when it is complete. Step t reads the memory of the blocks complete at t, and the head's
  96-step window covers the newest steps.
- **No fusion on the past.** The fusion runs only on the current window, in the head.
- **Training and inference.** Gradients reach the encoders through the memory. Training
  and inference use the same `memory_states`.

**Why 96-step blocks and not 10.** 10-step blocks would need about 2,900 encoded windows per
batch instead of about 300, too slow for the run. 96-step blocks tile the history exactly
once.

**Changes:**

- `OmniFuserRUL.encoded_streams` and a new `memory_states`;
- `CausalMemory.forward_blocks`, shared; the block-mean `forward` is unchanged for RUL-Fuse;
- block default 96 in `MemorySpec`, `ArmSpec` and `--memory-block`; the memory's own photo
  projection is removed;
- `b6-train` asks for 16 h (was 8);
- docs: B6 plan version 6 (O8, §5, §10, §14), project plan D21, its job table and Appendix
  A, runbook §12.1;
- tests (below).

**Verification:**

- **The input is OmniFuser's encoding.** `encoded_streams` equals the two streams
  `OmniFuser.encode` fuses (hooks). Each block's GRU input equals their mean over the head's
  own window ending at the block's last step.
- **Causal.** Changes after step t never alter the memory up to t (three values of t).
- **Gradients.** They reach the force embedding, the photo projection and the GRU through
  the memory; the fusion is not on that path.
- **Existing checks.** Input switches, modality dropout and per-tool independence still
  hold, now tested with 16-step blocks so the short toy tools complete several blocks.
- **Found by the suite: the toy learning test.** With 96-step blocks the short toy tools
  (82-109 points) barely complete a block. The toy fit stalled at 280.5 s (no memory:
  160.7 s; 16-step blocks: 128.5 s). The toy fit now uses 16-step blocks and 256 pairs per
  epoch: fit 80.3 s, 0.44 × the constant predictor. Its shuffled control is 0.96 ×; the old
  raw-input memory had given it a usage shortcut (0.78 ×). Real tools complete 19-56 blocks
  each, so this does not carry over. Check the first Ruche fits for a validation MAE stuck
  at its first value.
- **Real data, laptop CPU.** The 2-epoch smoke fit (force + tool + chip, fold 1, 256 pairs
  per epoch) runs end to end in 135 s, against 38 s with version 3's memory. Expected on
  Ruche: 5-15 minutes per fit, 2.5-7.5 GPU-hours for the run.
- **Suite.** 167 tests: 165 pass, 2 skip (the Mantis and DINOv2 checkpoint
  tests, which run in J0); ruff and pyright are clean.

**Suggested commit (you commit):** "B6 version 6: the GRU memory reads OmniFuser's
encodings":

- `rulfuse/omnifuser_rul/model.py`, `rulfuse/models/memory.py`,
  `rulfuse/omnifuser_rul/train.py`, `rulfuse/omnifuser_rul/run_report.py`,
  `rulfuse/cli.py`;
- `tests/test_omnifuser_rul.py`;
- `slurm/omnifuser_rul.sh`;
- `docs/OMNIFUSER_RUL_PLAN.md`, `docs/RUCHE_RUNBOOK.md`, `PROJECT_PLAN.md`,
  `PROJECT_LOG.md`.

## Ruche J0 failed: the weight revision lookup (2026-11-11)

**Symptom.** Three J0 attempts stopped with `AssertionError: paris-noah/Mantis-8M: no cached
revision`. J2 stayed `DependencyNeverSatisfied` and J3 kept waiting. J0 had already built
the force store and the segment store (5,125 segments, 1,536 images).

**Cause.** The weights were cached (Mantis-8M snapshot `bc7d5ab`), and the encoders loaded
offline. `hf_revision` called `snapshot_download(repo, local_files_only=True)` without
patterns. Ruche's huggingface_hub then raised `IncompleteSnapshotError`, because the
prefetch caches only `*.json` and `*.safetensors` (README, `.gitattributes` and the logo
are not needed). The function swallowed the error and returned None.

**Fix.**

- `rulfuse/features/encoders.py`: `hf_revision` looks up the same patterns as the prefetch
  (`WEIGHT_PATTERNS`). It falls back to the cache's `refs/main`, and requires a
  `*.safetensors` file in the snapshot.
- Test: `tests/test_encoders.py::test_a_weights_only_cache_still_gives_the_revision`.
- Runbook §15: a troubleshooting row.

**Ruche end of life.** Ruche's login banner says it will be replaced: back up data by the
end of November 2026. Results must be copied home (runbook §13) before then.

**Suggested commit (you commit):** "Fix the weight revision lookup for weights-only caches":
`rulfuse/features/encoders.py`, `tests/test_encoders.py`, `docs/RUCHE_RUNBOOK.md`,
`PROJECT_LOG.md`.

## Stage 3 finished; tool photos readmitted (D24, 2026-11-11)

**Stage 3 on Ruche.** With the revision-lookup fix, J0, J2 and J3 ran.

- The Mantis-8M and DINOv2-B caches are complete.
- Force: go.
- The feature check found all three photo types informative and excluded all three for
  leaking the stage label (report `nonastreda-3e357b5a23ff.json`).

**Leakage counts:** in how many of 10 tools a `tool_label` stage boundary is the largest
run-to-run step; the limit is 3.

| | PC1 | PC2 | PC3 | Wear readout | Probe MAE (mean predictor 35.8) | Shuffled |
|---|---|---|---|---|---|---|
| True wear (`hi_reference`) | – | – | – | 3 | – | – |
| Force signals | 2 | 2 | 1 | 1 | – | – |
| Workpiece | 10 | 10 | 0 | 10 | 14.9 | 32.7 |
| Chip | 6 | 3 | 2 | 7 | 19.4 | 34.1 |
| Tool | 1 | 4 | 2 | 3 | 21.7 | 35.9 |

**Your decision (D24): readmit tool photos only.**

- Workpiece and chip are stage-locked, well beyond the true wear's own count, and stay out.
- Tool crossed the limit by one tool on one of four statistics, and its readout behaves
  like the true wear.
- The decision was made before any RUL result.

**Implemented:**

- `configs/feature_check_overrides.json` records the readmission: who, when, which report,
  and why.
- `rulfuse.features.check.admitted_image_types` returns the kept types plus the readmitted
  ones. It refuses types the report never examined.
- B6's submit guard reads both.
- Tests: `tests/test_feature_overrides.py`, against the real decisions file.
- Dry run with a fake `sbatch`: `force tool` submits `b6-train` and `b6-report` with
  `--players force tool`; `force tool chip` is refused.

**Docs:**

- Project plan D24.
- B6 plan O5: force + tool + usage, OmniFuser's original pairing.
- Runbook §11.5 (the narrow-exclusion case), §12.1, §12.2, §13 and the cheat sheet; the
  run's arm folder is now `force+tool`.

**Next on Ruche:** `bash slurm/omnifuser_rul.sh force tool`.

**Suggested commit (you commit):** "D24: readmit tool photos; record feature-check overrides":

- new: `configs/feature_check_overrides.json`, `tests/test_feature_overrides.py`;
- `rulfuse/features/check.py`, `slurm/omnifuser_rul.sh`, `slurm/omnifuser_results.sh`;
- `PROJECT_PLAN.md`, `docs/OMNIFUSER_RUL_PLAN.md`, `docs/RUCHE_RUNBOOK.md`, `PROJECT_LOG.md`.

## B6 full run: results (2026-11-11)

**Jobs.** Commit `71da7b1`, store `nonastreda/6e0cb095967b`, `force tool`.

- `b6-train` 2493824: 30 of 30 fits, median 448 s each.
- `b6-report` 2493825: sealed.
- `b6-report` 2494379: run by you with `--unseal`, so the test was read before the freeze.
  B6 is a stand-alone baseline; under D15, no RUL-Fuse design decision may use these
  numbers before 2026-12-01.
- Archives `b6-results-20261111-2040` and `-2120`: both hashes match. The second is
  unpacked in `D:\RULModel\runs\omnifuser_rul\`.

**Test results** (current mode, mean over the 10 test tools):

| | B6 | B2 | B0 (constant) | B1 |
|---|---|---|---|---|
| Window MAE (s) | 587.4 (CI 411.7 to 801.7) | 467.8 | 412.0 | 512.5 |

- Against B2, B6 is worse by 119.6 s (CI −281.6 to +33.1) and better on 3 of 10 tools.
- Asymmetric score 196.3 against 46.7 for B2.
- 35 % of 1 s steps rise.
- Validation window MAE 540.1 (B2 464.7).

**Inference modes:** neither mode is clearly better, on test and on validation.

- Test, context − current: +0.4 s (CI −20.2 to 18.7).
- Validation: −8.6 s (CI −35.5 to 12.6).
- The context mode lowers the bias (−35 s) and the asymmetric score, but more of its
  1 s steps rise (0.48 against 0.35).

**What the trajectories show:**

- The decline is tracked on T3, T6–T9, but starts about 5–10 min early (pessimistic).
- Wear is missed on T4 and T5: the prediction stays at the cap.
- On T1 and T10 the prediction falls to only about 1,200 s.
- T2 is far too pessimistic from minute 15.
- The tools split into strongly late and strongly early errors, so the mean bias is small
  but meaningless.

**Why: over-fitting to the 8 training tools.**

- Median training MSE falls 50× (0.092 → 0.0018).
- The validation MAE barely moves: median 402 s at epoch 1, best 310 s, last 376 s. The
  curve is noisy, so the epoch picked on one validation tool is optimistic (validation 540
  against test 587).
- Expected for an end-to-end OmniFuser with ten tools.
- B6 stays one fixed setting (O-decisions): this is its result. It is not tuned.
