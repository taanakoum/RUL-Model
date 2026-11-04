# End to end: from the raw files to a labelled, causal prediction point

This document follows the data through the pipeline built so far (Stages 0-2), function by
function, with real numbers. It traces the plan's worked example: tool T5, pass 10,
segment 5. It is regenerated after every substantial pipeline change:

```powershell
D:\RULModel\.venv\Scripts\python.exe scripts/end_to_end_trace.py   # writes runs\stage2\trace.json
```

Last verified: 2026-11-04. Segment store `8d2b69c23453`; default options: causal trim,
gap-aware axis, duplicates attached to the latest member, confirmed TTF, k = 15 passes and
1,500 s.

## 1. Raw files → verified copy

| Step | Function | Input → output | Real numbers |
|---|---|---|---|
| Copy | `robocopy` from `D:\Project14\data\raw\nonastreda` | Publisher files → `D:\RULModel\raw\nonastreda` | 1,539 files (spectrograms excluded), 1.2 GB |
| Verify | `rulfuse.verify.verify_data` | Files → report | Every sha256 equal to the publisher's; 512 passes; gaps T1 run 1, T2 run 10, T4 runs 6-9 |
| Labels | `rulfuse.data.nonastreda.load_labels` | `labels.csv`, `labels_reg.csv` → 512 `PassMeta` | T5R3B2: `image_label` sharp, `tool_label` 1, flank wear 52.29 µm |

## 2. Force file → force store (read once)

`rulfuse.data.force_store.build_force_store` runs the following, and is the only reader of
the 270 MB MAT file:

1. `read_datastore` loads the file.
2. `parse_datastore` applies OmniFuser's checks: the labels agree, the shape is
   (3, samples), every value is finite.
3. `duplicate_groups` finds the bit-identical recordings.
4. `write_force_store` writes the store.

| Output | Value |
|---|---|
| `interim/forces/signals.npy` | 37,691,053 samples × 3, float32 (452 MB), memory-mapped |
| Distinct recordings / records / duplicate groups | 378 / 512 / 132 |
| T5R3B2 | 97,586 samples (97.6 s), its own recording (not in a duplicate group) |

## 3. Trim (causal)

`rulfuse.data.trim.span_bounds_causal` works in three steps:

1. The envelope: √(20-sample moving mean of \|F\|²).
2. The reference: the median envelope over the first 5 s.
3. The active samples: envelope ≥ 50 % of the reference.

The pass runs from the first to the last active sample.

| | Causal | OmniFuser (5 % of the pass) |
|---|---|---|
| T5R3B2 bounds | 430 to 96,478 | 429 to 96,479 |
| Duration | **96.048 s** | 96.050 s |

The causal start is within 200 samples of OmniFuser's on 99.4 % of passes
(`docs/EDA_RUL.md` §4).

## 4. Timeline (gap-aware)

`rulfuse.data.timeline.build_tool_timeline` builds each tool's timeline:

- **positions** follow (run, block);
- **interior run gaps** are inserted as unrecorded passes, with the median duration of
  the tool's earlier passes;
- **duplicates** are attached to their latest member (D8).

| | Value |
|---|---|
| T5 | 48 recorded passes, no gaps; T5R3B2 is position 10 |
| Whole dataset | 532 passes = 512 recorded + 20 unrecorded (T2: 4, T4: 16); 134 force-missing |
| Cumulative cutting time at the start of T5's pass 10 | 859.7 s (the 9 earlier trimmed passes) |

## 5. Labels

`rulfuse.data.labels.tool_labels` applies the confirmed rule: the first `dulled` pass
followed by ≥ 2 `dulled` passes among the next 3.

| T5 | Value |
|---|---|
| TTF pass | 19 (T5R5B3), the end of its pass at τ = 1,827.0 s |
| Smoothed wear at the TTF | 109.7 µm (isotonic over T5's recorded passes) |
| Pass 10 wear | Measured 52.3 µm; smoothed 63.4 µm; causal smoothed 63.4 µm |

## 6. Segments → prediction points

`rulfuse.data.segments.segment_sample_bounds` cuts 10 s windows from the trimmed start;
a remainder under 5 s is merged.

| T5, pass 10, segment 5 | Value | Note |
|---|---|---|
| Segments in the pass | 10 | 96.05 s: 9 × 10 s + 6.05 s |
| Samples | 40,430 to 50,430 | Trim start + 40-50 s |
| Elapsed in the pass, e | 50.0 s | Input |
| Completed passes | 9 | Input |
| Cumulative cutting time, τ | 909.7 s | Input |
| Mean duration of earlier passes | 95.5 s | Input (for causal elapsed passes) |
| Label time, t = 9 + 50/96.05 | 9.521 passes | Label only: it uses the pass's own duration |

## 7. Targets

`rulfuse.data.labels.rul_targets` computes the targets.

| | Passes | Seconds |
|---|---|---|
| RUL | 19 − 9.521 = **9.479** | 1,827.0 − 909.7 = **917.3** |
| Capped target y | 9.479 (< 15) | 917.3 (< 1,500) |
| In the window | yes | yes |

**Store-wide:** 3,747 prediction points (segments up to the TTF), 1,338 in the passes
window and 1,361 in the seconds window. The 1,378 segments after the TTF are history and
wear targets only.

## 8. Segment store

`rulfuse.data.segment_store.build_segment_store` writes the store to
`interim/segments/8d2b69c23453/` (built in 5.7 s).

| Table | Rows | Holds |
|---|---|---|
| `passes.parquet` | 532 | Identity, trim bounds, duration, τ, force attachment, photo position, labels, smoothed and causal wear |
| `segments.parquet` | 5,125 | Sample bounds, distinct-segment id (3,777 distinct to encode), input times, label time, targets, baseline flag |
| `tools.json` | 10 | TTF, first recorded passes (T5: 1-4) |

**Photo rule** (`photo_visible`, D9): the point in pass 10 may read the photos of passes
1-9 only, never pass 10's own.

## 9. Channels and features (Stage-2 subset of H)

1. `rulfuse.data.channels.fz_offset` takes the mean Fz over the pass's first second:
   −10.87 N for T5R3B2.
2. `derived_channels` adds Fz_rel = Fz − offset and \|F\|.
3. `rulfuse.features.physics.channel_stats` gives the mean and RMS per channel.

| Feature | Absolute | Relative to T5's first 4 passes (`relative_to_baseline`) |
|---|---|---|
| mean Fx (N) | 106.34 | +21.61 |
| RMS of \|F\| (N) | 109.27 | +23.79 |
| mean Fy (N) | −12.44 | −12.18 |
| mean Fz (N) | −21.23 | −13.99 |
| mean Fz_rel (N) | −10.37 | −8.41 |

The relative values are the change since the tool's healthy start. T5's force has risen
by about 22 N by pass 10, with 9.5 passes left.

## 10. Fold normalization

`rulfuse.data.normalization` provides the fold-level transforms:

- **`Scaler`, `PCA` and `BlockTransform`** (per channel) are fitted on the training tools'
  rows only (`train_rows`). A poison test checks that validation and test rows never move
  the fit.
- **`causal_baseline`** is a running mean over the tool's first 4 recorded passes seen so
  far. A perturbation test checks that later rows never change earlier baselines.

**Stage-2 finding** (`scripts/validate_stage2.py`, `runs/stage2/validation.json`):

- Relative features remove static offsets (T1: 0.97 → 0.34 sd), but raise the cross-tool
  shift in 6 of 10 folds (median 0.35 against 0.31 sd).
- They also make tools easier to identify (35 % against 23 %).
- The default token mode is therefore "both" (absolute + relative halves); Gate 2b decides.

## 11. What comes next

**Stage 3** adds the rest of H (spectral features), the frozen pretrained encoders and the
feature caches, keyed by distinct segment and photo. It then runs Gate 1, including the
stage-step leakage probe for every photo type.
