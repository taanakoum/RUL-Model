# End to end: from the raw files to a labelled, causal prediction point

This document follows the data through the pipeline built so far, function by function,
with real numbers. It traces the plan's worked example: tool T5, pass 10, segment 5. It is
regenerated after every substantial pipeline change:

```powershell
D:\RULModel\.venv\Scripts\python.exe scripts/end_to_end_trace.py   # writes runs\stage3\trace.json
```

Last verified: 2026-11-05, after the core/adapter refactor (D18). Segment store
`nonastreda/3e357b5a23ff`. Default options: causal trim, gap-aware axis, duplicates attached
to the latest member, confirmed TTF, horizon H = 1,500 s.

**Two layers.** Steps 1-5 are the Nonastreda adapter (`rulfuse/adapters/nonastreda/`):
everything that knows about passes, runs, blocks and labels. It hands the core a list of
units with recordings, images and failure times (`rulfuse/core/schema.py`). Steps 6-10 are
the generic core: it never sees a pass number.

## 1. Raw files → verified copy (adapter)

| Step | Function | Input → output | Real numbers |
|---|---|---|---|
| Copy | `robocopy` from `D:\Project14\data\raw\nonastreda` | Publisher files → `D:\RULModel\raw\nonastreda` | 1,539 files (spectrograms excluded), 1.2 GB |
| Verify | `adapters.nonastreda.verify.verify_data` | Files → report | Every sha256 equal to the publisher's; 512 passes; gaps T1 run 1, T2 run 10, T4 runs 6-9 |
| Labels | `adapters.nonastreda.load.load_labels` | `labels.csv`, `labels_reg.csv` → 512 `PassMeta` | T5R3B2: `image_label` sharp, `tool_label` 2, flank wear 52.29 µm |

## 2. Force file → force store (adapter, read once)

`adapters.nonastreda.force_store.build_force_store` is the only reader of the 270 MB MAT
file. It loads the file, applies OmniFuser's checks (labels agree, shape (3, samples),
finite values), finds the bit-identical recordings and writes the store.

| Output | Value |
|---|---|
| `interim/forces/signals.npy` | 37,691,053 samples × 3, float32 (452 MB), memory-mapped |
| Distinct recordings / records / duplicate groups | 378 / 512 / 132 |
| T5R3B2 | 97,586 samples (97.6 s), its own recording (not in a duplicate group) |

## 3. Trim (adapter, causal)

`adapters.nonastreda.trim.span_bounds_causal`: the envelope √(20-sample moving mean of
\|F\|²); the reference, the median envelope over the first 5 s; the pass runs from the
first to the last sample with envelope ≥ 50 % of the reference.

| | Causal | OmniFuser (5 % of the pass) |
|---|---|---|
| T5R3B2 bounds | 430 to 96,478 | 429 to 96,479 |
| Duration | **96.048 s** | 96.050 s |

The causal start is within 200 samples of OmniFuser's on 99.4 % of passes
(`docs/EDA_RUL.md` §4).

## 4. Timeline (adapter, gap-aware)

`adapters.nonastreda.timeline.build_tool_timeline`: positions follow (run, block); interior
run gaps count as unrecorded cutting at the median duration of the tool's earlier passes;
duplicates are attached to their latest member (D8).

| | Value |
|---|---|
| T5 | 48 recorded passes, no gaps; T5R3B2 is position 10 |
| Whole dataset | 512 recordings; 20 unrecorded passes (T2: 4, T4: 16) only shift τ; 134 recordings without a signal (non-latest duplicate members) |
| Operating time at the start of T5R3B2 | 859.7 s (the 9 earlier trimmed passes) |

## 5. Failure time and health indicator (adapter)

`adapters.nonastreda.failure.tool_failure` applies the confirmed rule: the first `dulled`
pass followed by ≥ 2 `dulled` passes among the next 3. The failure time τ* is the end of
that pass.

| T5 | Value |
|---|---|
| Failure recording | T5R5B3 (pass 19), τ* = 1,827.0 s |
| HI at the failure (isotonic flank wear) | 109.7 µm |
| HI of T5R3B2 | Measured 52.3 µm; smoothed 63.4 µm; causal smoothed 63.4 µm |

`adapters.nonastreda.dataset.to_units` then writes each pass as a generic `Recording`
(start 859.7 s, duration 96.048 s, a `SignalRef` to samples 430-96,478, HI, stage 2,
group `R3`), its three photos as `Image`s available at the pass's end (955.8 s), and pass,
run, block and labels into `attrs`, which the core passes through but never reads.

## 6. Segments → prediction points (core)

`core.segments.segment_sample_bounds` cuts S = 10 s windows from the recording's start; a
remainder under S/2 is merged.

| T5R3B2, segment 5 | Value | Note |
|---|---|---|
| Segments in the recording | 10 | 96.05 s: 9 × 10 s + 6.05 s |
| Samples | 40,430 to 50,430 | Recording start + 40-50 s |
| Seconds since the recording started, e | 50.0 s | Input |
| Operating time, τ | 909.7 s | Input |

## 7. Targets (core, seconds only)

`core.targets.rul_targets`: RUL = τ* − τ = 1,827.0 − 909.7 = **917.3 s**, about 15 minutes
of cutting. The capped target is the same (917.3 < H = 1,500), and the point is in the
window.

**Store-wide:** 3,747 prediction points (segments up to the failure), 1,361 in the window.
The 1,378 segments after the failure are history tokens and HI targets only.

## 8. Segment store (core)

`core.store.build_segment_store` validates the dataset against the schema, then writes
`interim/segments/nonastreda/3e357b5a23ff/` (about 15 s, mostly reading the trims).

| Table | Rows | Holds |
|---|---|---|
| `recordings.parquet` | 512 | Unit, index, key, times, signal reference, HI (target and causal), stage, group, baseline and failure flags, `attr_*` columns |
| `segments.parquet` | 5,125 | Sample bounds, distinct-segment id (3,777 distinct to encode), τ and e, RUL targets, baseline flag |
| `images.parquet` | 1,536 | Unit, type, key, path, availability time |
| `units.json` | 10 | Failure time and recording, baseline recordings (T5: R1B1-R1B4) |

**Image rule** (`image_visible`, D9): an image is readable only when its availability time
lies strictly before τ. At τ = 909.7 s the point sees 9 tool photos (passes 1-9, the last
T5R3B1), never T5R3B2's own (available at 955.8 s).

The refactored store was checked against Stage 2's (`8d2b69c23453`): every segment's times,
targets, sample bounds, distinct ids and baseline flags are identical, and image
visibility agrees on 400 random points.

## 9. Channels and level descriptors (core)

`core.channels.model_channels` adds one magnitude channel per 3-axis sensor group (here
\|force\|). `core.channels.descriptors` gives each channel 7 level descriptors; `drift` is
the segment mean minus the mean of the recording's first second.

| T5R3B2, segment 5 | Fx | Fy | Fz | \|force\| |
|---|---|---|---|---|
| log10 RMS of the AC part | 0.599 | 0.486 | −0.124 | 0.607 |
| Kurtosis | 1.98 | 2.06 | 2.23 | 1.97 |
| Crest | 2.99 | 2.86 | 3.49 | 3.01 |
| Mean (N) | 106.34 | −12.44 | −21.23 | 109.19 |
| Slope (N/s) | −0.097 | 0.291 | −0.190 | −0.091 |
| Drift since the first second (N) | +47.38 | −30.04 | **−10.37** | +46.58 |

Fz's drift (−10.37 N) is exactly the old Fz_rel channel's mean: the generic descriptor
carries what the Nonastreda-specific channel carried. The large Fx drift comes from the
entry transient that the first second still contains.

**Relative to T5's healthy start** (`core.normalization.relative_to_baseline`, the mean of
its first 4 recordings): mean Fx +21.6 N, RMS of \|force\| +23.8 N. T5's force has risen by
about 22 N by pass 10, with 917 s of cutting left.

## 10. Fold normalization (core)

`core.normalization`: `Scaler` (and `PCA`, kept for B4 and analysis) are fitted on the
training units' rows only; a poison test checks that validation and test rows never move
them. `causal_baseline` is a running mean over the unit's first 4 recordings seen so far;
a perturbation test checks that later rows never change earlier baselines.

**Stage-2 finding** (`scripts/validate_stage2.py`): relative features remove static offsets
but raise the cross-tool shift in 6 of 10 folds and make tools easier to identify, so the
default token mode is "both"; Gate 2b decides.

## 11. Encoders and caches (Stage 3, on Ruche)

`features.cache.extract_signal_features` reads each distinct segment once, resamples it to
the model rate (1 kHz here, unchanged), computes the 7 descriptors and the 11 spectral
features of H, and encodes its 512-sample sub-windows with Mantis-8M (20 for a 10 s
segment), mean- and sd-pooled to 512 values per channel. `extract_image_features` prepares
each photo with its type's spec (tool crop, chip resize, workpiece standardization) and
keeps DINOv2-B's class token and a 2 × 4 patch grid. Both run in Ruche job J2
(`docs/RUCHE_RUNBOOK.md` §11); the feature check (J3) then decides which photo types stay.
