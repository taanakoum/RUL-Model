# OmniFuser-RUL: an OmniFuser baseline for the RUL task

Version 2, 2026-11-05. Version 1 (the same day) was a draft; §12 lists what
the review of it found and how version 2 resolves each point. **Implemented the
same day; §14 records the implementation, its verification and the three
places where it refines this plan.**

OmniFuser-RUL turns the finished OmniFuser reproduction
([github.com/taanakoum/OmniFuser](https://github.com/taanakoum/OmniFuser), tag
`v1.0`) into a remaining-useful-life predictor with as few model changes as
possible. It is a **separate model and an additional baseline (B6)**. Its
results are compared with RUL-Fuse, this project's model. The labels, the
processing and the evaluation come from this project, so the comparison
isolates the architecture. It also answers two questions with a second,
independent architecture:

- which modality carries RUL information;
- whether two modalities are justified or one is enough.

## Contents

1. [Decisions](#1-decisions)
2. [Summary](#2-summary)
3. [What is shared with RUL-Fuse, and what is OmniFuser's](#3-what-is-shared-with-rul-fuse-and-what-is-omnifusers)
4. [Mapping the RUL task onto OmniFuser](#4-mapping-the-rul-task-onto-omnifuser)
5. [Model changes (the minimal set)](#5-model-changes-the-minimal-set)
6. [Data module](#6-data-module)
7. [Training, selection and evaluation](#7-training-selection-and-evaluation)
8. [The modality study](#8-the-modality-study)
9. [Verification](#9-verification)
10. [Stages, timeline and cost](#10-stages-timeline-and-cost)
11. [Risks](#11-risks)
12. [Review of version 1](#12-review-of-version-1)
13. [Code layout](#13-code-layout)
14. [Implementation status](#14-implementation-status)

## 1. Decisions

| # | Decision | Value | Source |
|---|---|---|---|
| O1 | Role | A **separate model, baseline B6**, against which RUL-Fuse is compared. Not a fusion candidate of Gate 2, and no RUL-Fuse decision reads it | You (2026-11-05) |
| O2 | Photo encoder | **EfficientNet-B0, ImageNet weights, frozen**, as in OmniFuser | You (2026-11-05) |
| O3 | Test reads | **After the design freeze of 2026-12-01 (D15)**. Test predictions are written by every fit but not scored before then | You (2026-11-05) |
| O4 | Shared task settings | The targets, prediction points, folds, point weights, selection metric, seeds, ensembling and metrics are RUL-Fuse's (§3). Only the architecture is OmniFuser's | Review (§12, V2) |
| O5 | B6's configuration, fixed now | Force + **every photo type admitted by the Stage-3 leakage probe** (the same inputs as RUL-Fuse) + usage. The secondary row B6-FT (force + tool photo) is OmniFuser's original pairing | Review (V7) |
| O6 | Architecture values | OmniFuser `v1.0` defaults: d = 32, 4 RTD blocks, 4 GAT layers × 4 heads, k = 4, 2 C2F rounds; input of 96 steps; dropout 0.1; Adam, learning rate 1e-4, batch 32, 30 epochs. No search | Minimal change |

## 2. Summary

- **What changes.** OmniFuser's two outputs (force forecast, wear class) are
  replaced by one RUL output from the same last fused step. The encoders, C2F
  fusion and refinement stay as they are.
- **What it reads.** This project's segment store gives everything:
  - the gap-aware time axis (D12) and the confirmed TTF (D2);
  - the causal trim, 10 s segments, and RUL capped at H = 1,500 s;
  - force-missing passes (D8), and photo availability times (D9).
- **Model changes.**
  - In OmniFuser `v1.1`, about 25 lines:
    - an `encode()` method that returns the fused sequence (a refactor with
      identical outputs);
    - a photo-view count for the image projection.
  - Here, a 60-line wrapper: an RUL head, and a switch that trains with one
    stream zeroed.
- **The modality study.**
  - Players: force (F), tool photo (T), chip photo (C), and the workpiece
    photo (W) only if it passes the leakage probe. Usage is in every arm.
  - Every subset is trained and Shapley values are computed.
  - A paired test of the best two-modality arm against the best
    single-modality arm, over the 10 test tools, answers "is two justified?"
    (§8).
- **Cost and dates.**
  - About 15 GPU-hours on Ruche with cached features.
  - Implementation and the validation-side runs by 2026-11-25; test reads
    after 2026-12-01, alongside RUL-Fuse's final runs.

## 3. What is shared with RUL-Fuse, and what is OmniFuser's

**Shared, so that B6 and RUL-Fuse differ only in the model:**

| Item | Source |
|---|---|
| Recordings, gap-aware axis, TTF, duplicated-force rule, causal trim, photo availability | `rulfuse/adapters/nonastreda/` |
| Segment store: τ (operating time), e (seconds into the recording), RUL, y = min(RUL, H), window, prediction points, signal presence; image table with `kind`, `path`, `available_s` | `rulfuse/core/store.py` |
| Photo preparation: tool edge band at 224 × 672, chip 448 × 448, workpiece 224 × 224 with per-image standardization | The adapter's `ImageSpec`s, `rulfuse/core/images.prepared_pixels` |
| Force descriptors per segment and channel: log RMS, log peak, kurtosis, crest, mean, slope, drift | `rulfuse/core/channels.descriptors` |
| Standardization of the descriptors on the 8 training tools | Plan §5.9 |
| Folds; 3 seeds; ensemble = mean of the 3 seeds' predictions | `rulfuse/core/folds.py`; D19 |
| Point weights: each tool weighs the same; points with RUL > H get w_far = 0.3 | Plan §7.3 |
| Selection: the weighted whole-life validation MAE, mean of 3 consecutive evaluations | Plan §4.3 |
| Metrics, aggregation, statistics | Plan §10.2-§10.3; `rulfuse/metrics.py`, `rulfuse/baselines.py` |
| Test-score policy | D15 |

**OmniFuser's own, unchanged:**

- the temporal embedding (value MLP and position encoding), MRTE, the image
  projection and temporal GAT, C2F × 2 with refinement, and the read-out of
  the last fused step;
- frozen EfficientNet-B0 photo features;
- optimizer, epochs, batch and dropout.

**Deliberately not taken from RUL-Fuse:**

- trajectory augmentation, channel and token dropout;
- the relative ("both") token mode;
- the full causal history;
- the interval heads and calibration;
- Mantis and DINOv2.

Those are RUL-Fuse's contributions. A baseline that borrowed them would no
longer measure what they add.

## 4. Mapping the RUL task onto OmniFuser

| OmniFuser concept | OmniFuser-RUL | Why |
|---|---|---|
| Time step: 1 s of pooled force | **One 10 s segment** | Prediction points are segment ends (D7), so one step is one prediction point |
| Input: 96 steps | **The 96 segments ending at the prediction point** (about 16 min, about 10 passes). Shorter histories at a tool's start are left-padded. | A causal history covering the recent wear trend; padding keeps every prediction point |
| Force per step: 3 pooled channels | **7 descriptors × 4 channels (Fx, Fy, Fz, \|F\|) = 28 values, plus a signal-present flag = 29** | This project's processing. The value MLP takes any width. Force-missing segments (D8) and padding are zeros with the flag 0. |
| Run, block and position codes | **Off** (`metadata="none"`) | Run and block codes are the known shortcut. The position code uses the pass's total length, which is unknown during the pass (review R2 of the project plan). |
| Usage | **log(1 + τ/H) and e/100 s**, appended to the force vector (31 values); switchable | Plan §3.3 allows them. Usage is in every arm of the study, as in RUL-Fuse. |
| Photo per step | **Per view, the latest photo with `available_s` < the step's τ** | Enforces D9 with the store's own times; equals OmniFuser's lag 1 |
| Several photo types | **One composite vector per pass: the views' frozen features side by side** (`image_views` = V; a missing or excluded view is zeros) | All photos of a pass share one availability time, so one index per step suffices (asserted, §9) |
| Outputs | **One value, ŷ/H**, from the last fused step | D6. The force-forecast and class heads are not built. |
| Loss | **Point-weighted MSE on y/H** (§3) | D10's mean objective, with the shared point weights |
| Epoch selection | **Weighted whole-life validation MAE**, best mean of 3 consecutive epochs | Shared with RUL-Fuse (plan §4.3) |
| Prediction at evaluation | Clipped to [0, H] | The target's range |

## 5. Model changes (the minimal set)

**In OmniFuser (`v1.1`, about 25 lines; all 175 existing tests must pass):**

1. **`OmniFuser.encode(force, images, image_indices, cycle, batch, relative,
   drop_modality)`.** It returns the fused sequence F⁽ⁿ⁾. `forward()` becomes
   `encode()` followed by the existing heads, so every current output is
   bit-identical (a new test asserts it).
2. **`ModelConfig.image_views: int = 1`.**
   - The image projection becomes `Linear(views × feature_width, d)`, and the
     precomputed-feature width check uses the same product.
   - With one view nothing changes.

**In this project (`rulfuse/omnifuser_rul/model.py`, about 60 lines):**

3. **`OmniFuserRUL(nn.Module)`.**
   - It builds `OmniFuser(config)` with `prediction_head="direct"`,
     `metadata="none"`, `force_channels=31`, `freeze_image_encoder=True` and
     `image_views=V`.
   - It returns `Linear(d, 1)(encode(...)[:, -1])`, that is ŷ/H.
   - The unused force and class heads of the direct head are deleted after
     construction, so they hold no parameters and get no gradient.
4. **Input switches `use_force`, `use_photos`, `use_usage`** (changed in the
   implementation, §14):
   - They act at the input level, in training **and** testing:
     - force off zeroes the descriptors and the signal flag, the state of a
       force-missing segment (D8);
     - photos off zeroes the photo features and points every step at the
       "no photo yet" slot;
     - usage off zeroes the usage columns.
   - Every arm keeps the same architecture and parameter count, and usage
     stays available when force or photos are off. Zeroing OmniFuser's whole
     sensor stream would also have removed usage, which travels in the force
     vector.

## 6. Data module

`rulfuse/omnifuser_rul/data.py`:

1. **Step tables per tool**, built once from the segment store and the force
   store and cached:
   - the 28 descriptors per segment, with the recording's head mean as the
     drift reference (`channels.head_mean`);
   - the signal flag and the usage values;
   - per view, the image row of the latest visible photo (`available_s` < τ);
   - y, RUL, the window flag and the prediction-point flag.
2. **Photo features:** each image row is prepared with its kind's `ImageSpec`,
   ImageNet-normalized, and passed through the frozen EfficientNet-B0. Its
   pooled 1,280 features are cached once for the whole dataset: they do not
   depend on the fold.
3. **Samples:** one per prediction point (3,747). Each is the 96 steps ending
   at it, with y and the point weight at that point.
4. **Normalization:** the descriptor and usage columns are standardized on
   the fold's 8 training tools; the flags stay 0/1. Photo features enter raw,
   as in OmniFuser; the projection learns their scale.
5. **Batches:** windows, a per-window list of distinct composite photo
   vectors, and per-step indices. This is exactly OmniFuser's
   `(images, image_indices)` interface with precomputed features.

## 7. Training, selection and evaluation

**One fit** (fold *i*, seed *s*):

1. Build the samples of the 8 training tools and of the validation tool.
2. Train for 30 epochs with the point-weighted MSE. Use Adam with learning
   rate 1e-4 and batch 32, and clip gradients at 1.0 (OmniFuser's settings).
3. After each epoch, compute the weighted whole-life MAE on the validation
   tool. Keep the epoch with the best mean of 3 consecutive values.
4. Predict every pre-failure point of the validation and test tools. Write
   one row per point (`unit`, `segment row`, `fold`, `seed`, `ŷ`) to
   `runs/omnifuser_rul/<arm>/fold-<i>/seed-<s>/predictions.parquet`.

**Aggregation** (`rulfuse.omnifuser_rul.aggregate`):

- validation-tool scores at any time;
- test-tool scores only with `--after-freeze` on or after 2026-12-01 (D15),
  as `rulfuse aggregate` does for RUL-Fuse.

**Reported** (the same tables as RUL-Fuse):

- the plan §10.2 groups **point, safety, prognostic and stability**, on the
  window (headline), whole-life and recording-end points;
- force-missing points and the points after T4's and T2's gaps, separately;
- the mean ± sd over the 10 test tools of each tool's 3-seed ensemble metric,
  and the pooled value;
- tool-bootstrap 95 % intervals.

B6 gives points, not intervals, so its uncertainty and decision rows show "–".

**Comparison with RUL-Fuse:**

- paired Wilcoxon signed-rank tests over the 10 test tools (window MAE),
  with Holm correction across the rows of the table (plan §10.3);
- B6 sits in every table next to B0-B5.

## 8. The modality study

**Arms.** Usage is on in all arms. Each arm runs 10 folds × 3 seeds.

| Arm (folder name) | `--players` | Other options |
|---|---|---|
| U (`usage-only`) | (none) | |
| F (`force`) | `force` | |
| T (`tool`) | `tool` | |
| C (`chip`) | `chip` | |
| F+T, F+C, T+C | two of them | |
| F+T+C | `force tool chip` | |
| W arms | + `work`, only if the workpiece passes the Stage-3 leakage probe | |
| F, usage off (`force-nousage`) | `force` | `--no-usage` |
| T, usage off (`tool-nousage`) | `tool` | `--no-usage` |
| F+T+C with modality dropout (`force+tool+chip-dropout0.2`) | `force tool chip` | `--modality-dropout 0.2`: per sample, with p/2 the force is missing and with p/2 no photo is visible yet |

`rulfuse omnifuser-study --players force tool chip --list` prints the 11 arms
with their options.

**Q1. Which modality contributes most?**

- Shapley values over the admitted players (F, T, C; W only if admitted),
  with v(S) = −window MAE of the 3-seed ensemble on the test tools.
- v(∅) is the better, on the validation tools, of B2 and arm U (as plan §11.1
  chooses between B2 and B3).
- Reported:
  - φ per tool;
  - the mean with tool-bootstrap 95 % intervals;
  - the share of the total gain over v(∅);
  - pairwise interactions (for example F × T).

**Q2. Are two modalities justified, or is one enough?**

- The comparison is fixed in advance: the best two-player arm against the
  best single-player arm. "Best" is chosen on the **validation** tools, so the
  test read cannot choose its own comparison.
- Two modalities are judged **justified** if both hold on the test tools:
  - a one-sided paired Wilcoxon test over the 10 tools gives p < 0.05;
  - the mean window MAE improves by at least 10 s (H/150, the project's
    threshold).
- **"One is enough"** if the improvement is below 10 s, or the test is not
  significant. In that case, the per-tool wins and the 95 % interval of the
  difference say whether the two are equivalent or the study is
  underpowered.
- The same test compares:
  - the three-player arm with the best two-player arm;
  - photos only (T+C) with force plus a photo.

**Q3. Reliance.** The F+T+C model trained with modality dropout is masked at
test time, one subset at a time (plan §11.2). A modality can be used when
present but still be replaceable.

**Q4. Sensing without usage.** F and T with usage off, against B2. This
measures what the sensors know without the elapsed time.

**Cross-architecture check.** RUL-Fuse's own Shapley study (plan §11) uses the
same players and points. If both architectures rank the modalities the same
way, the ranking is a property of the data, not of one model.

**Expectation, stated in advance so it can be wrong:**

- The photos should carry most of the information near failure, because they
  show the wear itself.
- Force should matter most when the latest photo is old or missing.
- Usage is strong on its own (B2 468 s), so sensing gains are measured
  against v(∅), not zero.
- In the OmniFuser forecasting runs the fusion did not help; Q2 tests whether
  it helps here.

## 9. Verification

| # | Check | How |
|---|---|---|
| V-1 | `encode()` changes nothing | OmniFuser's 175 tests pass; new test: `forward()` outputs before and after are identical; `image_views=1` gives the old parameter count (4,183,600) |
| V-2 | Causality | For random samples: no step after the prediction point; no photo with `available_s` ≥ its step's τ; no input that uses a recording's total length (plan P10, P11) |
| V-3 | Composite photos are valid | All photos of a recording share one `available_s` (asserted on the real store) |
| V-4 | Targets and weights | Each sample's y equals the store's `y_s`; window flags equal; each tool's weights sum to 1 |
| V-5 | Baselines on the same points | B0-B2 reproduce 412, 513 and 468 s with this module's sample set |
| V-6 | Single-stream switch | With `streams="force"`, the output is unchanged when the photos change, and the other way round; `none` ignores both |
| V-7 | Learns | On the toy adapter, the window MAE is well below the mean predictor's; a shuffled-target control collapses to it |
| V-8 | Determinism and resume | The same seed gives the same predictions on the CPU; an interrupted fit resumes to the same result |
| V-9 | Test seal | `aggregate` refuses test scores before the freeze without `--after-freeze` |
| V-10 | Transfer contract | The core imports nothing from `omnifuser_rul`; photo kinds come from the store, never hard-coded (plan P22) |
| V-11 | Real-data smoke | One fold, 2 epochs, laptop CPU, cached features: runs, writes predictions, scores validation |

## 10. Stages, timeline and cost

| Stage | Dates | Output | Exit criterion |
|---|---|---|---|
| A. Release | Nov 5-6 | OmniFuser merged to `main` and tagged `v1.0`; `v1.1` with `encode()` and `image_views` | V-1 |
| B. Data, model, training | Nov 7-12 | `rulfuse/omnifuser_rul/`, its tests, photo-feature cache | V-2 to V-11 |
| C. Pilot on Ruche | Nov 13 | One 10-fold F+T run, 1 seed; time per fit | Validation MAE below B2, or a diagnosis |
| D. Study runs | Nov 14-20 | The arms of §8 × 3 seeds (with W only if admitted) | Every arm complete; validation scores |
| E. Validation analysis | Nov 21-25 | Best single and two-player arms chosen on validation; report skeleton | Choices recorded before the freeze |
| F. Test reads | after Dec 1 | B6 in the RUL-Fuse tables; Q1-Q4 | D15 respected |

**Cost.**

- One fit trains on about 3,000 samples for 30 epochs: about 3 minutes on an
  A100 with cached features. The pilot measures it.
- 11 arms × 10 folds × 3 seeds = 330 fits, about 16 GPU-hours. The 30 fits of
  an arm form one job, so the study is 11 jobs.
- It fits the 4-GPU limit in about half a day of wall time.
- It is scheduled around RUL-Fuse's Stage-4/5 jobs.
- The EfficientNet-B0 weights must be prefetched for the offline jobs
  (`slurm/prefetch_weights.sh`).

## 11. Risks

| Risk | Mitigation |
|---|---|
| The model learns elapsed time, not wear | Usage-off arms (Q4); v(∅) from B2 or U; the usage sensitivity s_U of plan §7.8 as a diagnostic |
| 96 segments are a short history for RUL | Usage gives the position in life. The window length stays OmniFuser's 96, a stated limitation and part of what the baseline measures |
| Descriptors read wear worse than learned embeddings | Accepted: B6 is OmniFuser with this project's generic processing. RUL-Fuse's Mantis inputs are its own contribution |
| 10 tools give low power for Q2 | The interval of the difference and per-tool wins; "underpowered" is a possible, stated outcome |
| T4 has 20 window points | Whole-life selection; per-tool and pooled reporting |
| The workpiece leaks the stage | Admitted only if it passes the Stage-3 probe; decision recorded when the J3 report arrives |
| GPU time competes with RUL-Fuse | 16 GPU-hours in 11 jobs, run in gaps; the arms are independent and resumable |

## 12. Review of version 1

| # | Severity | Finding in version 1 | Resolution in version 2 |
|---|---|---|---|
| V1 | Major | It treated OmniFuser-RUL as a possible Gate-2 fusion candidate, which would let a baseline steer RUL-Fuse's design | Separate model, baseline B6 (O1); no RUL-Fuse decision reads it |
| V2 | Major | Unweighted MSE and unweighted selection, while RUL-Fuse uses point weights. Long-lived tools and the healthy plateau would dominate B6's training and confound the comparison | Shared task settings (O4): point weights, weighted whole-life selection, ensembles, metrics |
| V3 | Major | OmniFuser's 128 × 640 tool crop, not this project's photo preparation, so the two models would see different pixels | The adapter's `ImageSpec`s for every kind (§3) |
| V4 | Major | B6's configuration was left open, so it could have been chosen after the test read | Fixed now (O5): force + every admitted photo type + usage; B6-FT as a secondary row |
| V5 | Major | The Q2 rule borrowed Rule A ("≥ 7 of 10 wins"), which governs design gates on validation tools, not a scientific comparison on test tools | A one-sided paired Wilcoxon test with p < 0.05 and a ≥ 10 s effect; the arms compared are chosen on validation (§8) |
| V6 | Minor | Compute estimate of 1.5 min per fit (7.5 GPU-h) was optimistic for 3,000 samples × 30 epochs | About 3 min per fit, 16 GPU-h; the pilot measures it |
| V7 | Minor | A 192-step history arm "if the pilot shows a gap" added an unplanned degree of freedom | Removed; 96 steps, as in OmniFuser |
| V8 | Minor | An optional HI auxiliary head (OmniFuser has none) | Removed: B6 stays minimal |
| V9 | Minor | Composite photo vectors assume all views of a pass become visible together, unchecked | Check V-3 on the real store |
| V10 | Minor | Padding without a mask makes padded steps look like average force | Flag 0 and blank photos on padded steps; usage marks early life; test V-2 covers it |
| V11 | Minor | No test seal for B6's own predictions | V-9: `aggregate --after-freeze` |
| V12 | Minor | The unused force and class heads would still exist in the model | Deleted in the wrapper (§5.3) |
| V13 | Minor | Offline Ruche jobs need the EfficientNet weights | Added to the prefetch step (§10) |

**Checks of the plan against the project plan:**

- **Same targets, points and folds:** D2, D3, D6, D7, D11 and D12, through
  the store.
- **Causality:** D9 and D14, through `available_s` and causal descriptors.
  Run, block and position codes are off.
- **Force-missing passes (D8):** zeros with the flag 0, scored separately.
- **Seeds (D19), test-score policy (D15)** and the transfer contract (P22)
  hold.
- **What does not apply:** D16 (encoders) and D17 (no PCA, low-rank
  projection) govern RUL-Fuse only. B6 uses EfficientNet-B0 (O2) and
  OmniFuser's projection.

## 13. Code layout

```text
rulfuse/omnifuser_rul/
  __init__.py
  data.py        step tables, photo-feature cache, samples, weights
  model.py       OmniFuserRUL wrapper (encode + RUL head, input switches)
  train.py       one fit and one arm: training, selection, predictions, resume
  study.py       arm definitions (§8) and the analysis
  aggregate.py   scores, ensembles, Shapley, two-versus-one tests, test seal
  report.py      results: progress, scores, choices fixed on validation, RESULTS.md
rulfuse/cli.py   omnifuser-prepare | omnifuser-arm | omnifuser-study | omnifuser-analyse |
                 omnifuser-results
tests/test_omnifuser_rul.py
slurm/omnifuser_rul.sh       guards, then one cache job and one job per arm (10 folds × 3 seeds)
slurm/omnifuser_results.sh   the results job: RESULTS.md and the archive to copy home
```

The dependency is the `omnifuser` extra of `pyproject.toml`
(`omnifuser-reproduction @ git+https://github.com/taanakoum/OmniFuser@v1.1`).
`rulfuse.omnifuser_rul` is generic like the core: it is in the transfer
contract's list (P22), imports nothing from the adapters, and reads image types
from the store. Only `rulfuse/cli.py` loads the dataset.

## 14. Implementation status

2026-11-05. Implemented in OmniFuser `v1.1` and in this project
(`rulfuse/omnifuser_rul/`). Committed: this project `7a3d274`; OmniFuser `v1.1` is
commit `eac0fed` on `main`, tag `v1.1` pushed to GitHub (checked 2026-11-05 with
`git ls-remote`). Reviewed the same day (below, "Review of the implementation").

**Refinements of the plan, found while implementing:**

1. **Input switches instead of stream masking** (§5.4). Usage travels in the
   force vector, so zeroing OmniFuser's sensor stream would have removed usage
   from the photo-only arms. The switches act on the inputs instead. Modality
   dropout uses the same two natural states: force missing, no photo yet.
2. **Epoch selection.** "Best mean of 3 consecutive values" is implemented as
   the centre of the complete triple with the lowest mean, so a single good
   epoch at the start cannot win on a 2-value average.
3. **Seeds** are 0, 1 and 2.

**Verification** (check numbers from §9):

| # | Result |
|---|---|
| V-1 | OmniFuser v1.1: `forward()` is bit-identical to v1.0 in 48 configurations (8 model variants × training/evaluation × three masking states), compared against the code of tag `v1.0`; the default parameter count stays 4,183,600; all 180 OmniFuser tests pass |
| V-2 | Each step's photo is the latest one with `available_s` < τ, checked at every step of every toy unit; windows end at their point, padding is zero, and every photo slot holds the features of a visible photo |
| V-3 | On the real store, all photos of one recording share one availability time (real-data test) |
| V-4 | y and the window flags equal the store's; each unit's weights sum to 1, with far points at 0.3 of window points |
| V-5 | The samples are exactly the store's prediction points; B2 computed by this module equals `rulfuse baselines` on every fold |
| V-6 | With force, photos or usage switched off, changing that input leaves the output unchanged; with all on, it changes |
| V-7 | On the toy dataset a fit reaches a validation MAE below 0.7 × the constant predictor's; with shuffled training targets it stays above 0.8 × |
| V-8 | The same seed gives identical predictions on the CPU; a rerun skips finished fits. This caught a bug: JSON lists never compared equal to tuples, so no fit was ever reused. The real-data run found the same bug in the photo cache's key (the tool crop); both are fixed, with tests |
| V-9 | Test scores raise `TestSealed` before 2026-12-01 and without `--after-freeze` |
| V-10 | The transfer-contract tests pass with `omnifuser_rul` in the generic list |
| V-11 | Real data, laptop CPU (3 threads): `omnifuser-prepare` builds the caches (3,747 prediction points over 10 tools, 1,536 photo features); one fit of force + tool + chip on fold 1 for 2 epochs takes 111 s, keeps epoch 2, and writes 393 validation points for T2 (116 in the window, as in plan §2.2). Its validation window MAE after 2 epochs is 515 s against B2's 228 s on that tool: a smoke run, not a result. Test scoring is refused (sealed) |
| V-12 | Results extraction (added in the review): on fake studies with known errors, `write_results` counts fits and names the missing ones, scores each arm exactly, fixes the comparisons on validation, seals the test units before 2026-12-01, refuses a test read of an incomplete study, and after the freeze reproduces the Shapley identities (φ_a − φ_b = v(a) − v(b); efficiency) and confirms the recorded choices. Run end to end on the real store with an 11-arm fake study (one fit missing) through `rulfuse omnifuser-results` |

The RUL project's suite at the commit: 130 tests, of which 128 pass and 2 skip (the Mantis
and DINOv2 weight tests, which run on Ruche); ruff and pyright report no issues. After the
review: 132 tests, 130 pass, the same 2 skip.

**Review of the implementation (2026-11-05).** The model, data and training follow §3-§8:

- photo visibility uses the store's strict `available_s < τ` rule;
- windows only look back;
- weights, standardization, selection and the seal are as planned;
- Shapley values and the paired tests are computed exactly.

Findings and fixes:

| # | Severity | Finding | Fix |
|---|---|---|---|
| R-1 | Blocker | `tests/test_omnifuser_rul.py` imported OmniFuser at collection. On Ruche, Stage-3 job J0 runs the whole suite before B6 is installed, so collection failed (simulated: exit code 2, zero tests run), J0 failed and J2, J3 never started | The module skips without OmniFuser or torchvision (`pytest.importorskip`); and the runbook's one-time setup (§7.2) now installs the `omnifuser` extra, so J0 also tests B6 |
| R-2 | Major | The runbook installed torchvision as `torchvision==<the version matching your torch>`, a placeholder, after the setup | torch and torchvision installed together from the CUDA 12.6 index (pip pairs them), then `pip install -e ".[test,encoders,omnifuser]"`; an upgrade block for an existing environment |
| R-3 | Major | Nothing checked that the B6 photo types are the ones Stage 3 admitted (O5), or that Stage 3 had run | `slurm/omnifuser_rul.sh` refuses without the force store, without a feature-check report, or with a type the report excluded (override only for a pilot) |
| R-4 | Major | Results stopped at a validation JSON: no completeness check, no record of the comparisons before the test read, no report, no way home | `rulfuse omnifuser-results` (`report.py`) and `slurm/omnifuser_results.sh` (§12.2 of the runbook): progress, curves, per-unit scores against B2, `choices.json` written before the freeze and checked after it, `RESULTS.md`, and one archive with its sha256 |
| R-5 | Minor | J0 proved offline loading for Mantis and DINOv2 but not for EfficientNet-B0, which `b6-prepare` needs offline | J0 checks the file in `$TORCH_HOME` and loads it, when OmniFuser is installed |
| R-6 | Minor | `status.sh` did not show B6 | A `B6 study` block: fits per arm |
| R-7 | Minor | If the arm listing failed, `omnifuser_rul.sh` still submitted the cache job and no arm | The arms are listed first; an empty list stops the script |
| R-8 | Note | The data loader runs in the main process (`num_workers=0`), so a fit may be slower than the 3-minute estimate on a GPU node | Measured by the pilot (the runbook asks to report a fit far above 3 min) |
| R-9 | Note | The log said "130 tests pass, 2 skip"; the commit had 130 tests, 128 passing | Corrected here |

**What remains:**

1. Install on Ruche (`docs/RUCHE_RUNBOOK.md` §7.2, which now includes B6) and fetch the
   EfficientNet-B0 weights (§8).
2. After the Stage-3 photo decision (J3): the pilot, then
   `bash slurm/omnifuser_rul.sh force tool chip` (with `work` only if it is admitted);
   runbook §12.1.
3. When every arm is complete, and before 2026-12-01: the results job (runbook §12.2),
   which records the comparisons in `choices.json`; copy the archive home (§13).
4. From 2026-12-01: the same job with `--after-freeze` for the test read.

