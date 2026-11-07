# OmniFuser-RUL: an OmniFuser baseline for the RUL task

**Version 6, 2026-11-08: the GRU memory reads OmniFuser's own encodings.** For each past
block of 96 steps, the GRU takes OmniFuser's encoded force series and encoded photos,
averaged over the block. Before, it read the raw step inputs (O8, §5). Training and
inference compute it the same way. §14 records what changed.

**Version 5, 2026-11-07: the penalty can no longer drag predictions down, and the report
compares two inference modes.**

- The temporal terms now move only the earlier prediction of a pair. Their weights are
  1/15 and 0.05, down from 1.0 and 0.5 (O9).
- Every prediction is reported in the current mode and in a context mode, where the
  tool's earlier predictions guide the new one (O13, §7).

The single run of version 4 is unchanged otherwise. §14 records what changed and why.

**Version 4, 2026-11-07: B6 is trained once, with one fixed setting, and reported (O12).**
No comparison of settings is run:

- not the modality study of §8;
- not the ablations of O11;
- not GRU against LSTM.

The run uses the version-3 design. §14 records what changed.

Version 3, 2026-11-06. Version 1 (2026-11-05) was a draft; §12 lists what its
review found and how version 2 resolves each point; version 2 was implemented and
reviewed the same day (§14). **Version 3** (your request of 2026-11-06) adds:

- 1 s steps, OmniFuser's own (O7);
- a GRU memory of the whole history (O8);
- a penalty on rising RUL predictions, with a slope term (O9);
- a fixed pair budget per epoch, with selection on the 10 s points (O10);
- two ablation arms (O11).

§14 records its implementation and verification.

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
| O7 | Time step | **1 s segments** (OmniFuser's own pooled step): B6 has its own store cut at S = 1 s. A fold trains on about 29,000 prediction points instead of about 3,000 at 10 s, and 96 steps are 96 s, about one pass. **Scored and selected on the 10 s grid** (D7): a 1 s store contains every 10 s point with the same τ, RUL and window, so B6, RUL-Fuse and B0-B5 are compared on the same 3,747 points | You (2026-11-06): more training examples; checked in §14 |
| O8 | Memory | A **GRU over the tool's whole history** (`rulfuse.models.memory.CausalMemory`, shared with RUL-Fuse, D21). **Its input for each block of 96 steps is OmniFuser's encoding of that block** (version 6): the block's window goes through OmniFuser's force encoder and its image branch, the same modules and weights as the head's window, and each stream is averaged over the window: [encoded force ; encoded photos], 64 numbers. Step t reads the state after the last block complete at t (causal); the head reads [last fused step ; memory state]. Hidden size 64. Gradients reach the encoders through both paths. (Versions 3-5: block means of the raw step inputs, in 10-step blocks) | You (2026-11-06): every new segment uses the information of all previous ones. You (2026-11-08): the GRU takes the encoded time series and encoded images, in training and inference |
| O9 | Temporal loss | Training samples are **pairs** of points of one tool, the earlier at lag 1 (the previous segment) half the time and up to 200 steps otherwise. The loss adds **1/15 × relu(ŷ_later − ŷ_earlier)** (a predicted RUL never rises) and **0.05 × ((ŷ_earlier − ŷ_later) − Δτ)²** where the earlier target is below the cap (true RUL falls by exactly Δτ), all on y/H (`rulfuse.models.losses`, D22). **Both terms move only the earlier prediction; the later one is the fixed reference** (version 5) | You (2026-11-06), plus the slope term (a stronger statement of the same fact, already in RUL-Fuse's plan). Version 5 (2026-11-07), after your question: with weight 1.0 on both predictions, an earlier prediction the model cannot correct dragged later ones down to it (§14) |
| O10 | Epochs | **2,048 pairs (4,096 windows) per epoch**, 30 epochs: a fixed training budget, close to version 2's, despite 10× more points | Cost (§10) |
| O11 | Ablations | Two extra arms with all players: **without the memory** and **without the temporal terms** | Measures what O8 and O9 add. **Not run (O12)** |
| O12 | One full run | **One configuration, trained once:** force + every admitted photo type + usage; 1 s steps; the GRU memory (O8); the loss with the monotonic penalty and the slope term (O9); 10 folds × 3 seeds = 30 fits. Its report gives each tool's test result (each tool is the test tool of one fold) against B0-B2 on the same points. **The modality study (§8), the ablations (O11) and any GRU/LSTM comparison are not run.** Test results appear from the freeze (D15), or earlier with `--unseal`, which the report records | You (2026-11-07): no experiments on the OmniFuser implementation, one setting, one full run, test results for a report |
| O13 | Inference modes | **Every fit's predictions are reported in two modes** (`rulfuse.inference`, D23): **current**, each prediction as the model makes it (the primary result); **context**, where the tool's earlier predictions, aged by the operating time since, are blended with the new one (time constant 120 s, fixed in advance). The report compares the two on the same tools, with a decision rule fixed in advance (§7). It is computed from the same 30 fits; nothing is trained twice | You (2026-11-07): a second mode where previous segments guide the new prediction, and a way to compare the two |

## 2. Summary

- **What changes.** OmniFuser's two outputs (force forecast, wear class) are
  replaced by one RUL output from the same last fused step, joined by a GRU
  memory of the whole history (O8). The encoders, C2F fusion and refinement stay
  as they are. Steps are 1 s (O7); the loss penalizes rising predictions (O9).
- **What it reads.** This project's segment store gives everything:
  - the gap-aware time axis (D12) and the confirmed TTF (D2);
  - the causal trim, 1 s segments (scored on the 10 s points), and RUL capped at
    H = 1,500 s;
  - force-missing passes (D8), and photo availability times (D9).
- **Model changes.**
  - In OmniFuser `v1.1`, about 25 lines:
    - an `encode()` method that returns the fused sequence (a refactor with
      identical outputs);
    - a photo-view count for the image projection.
  - Here, a wrapper: an RUL head on [last fused step ; memory], the input
    switches, and the shared memory and temporal-loss modules.
- **The modality study.**
  - Players: force (F), tool photo (T), chip photo (C), and the workpiece
    photo (W) only if it passes the leakage probe. Usage is in every arm.
  - Every subset is trained and Shapley values are computed.
  - A paired test of the best two-modality arm against the best
    single-modality arm, over the 10 test tools, answers "is two justified?"
    (§8).
- **Cost and dates.**
  - About 20-40 GPU-hours on Ruche with cached features (13 arms; §10); the
    pilot measures it.
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
| Time step: 1 s of pooled force | **One 1 s segment** (O7) | OmniFuser's own step; every segment end is a prediction point, so 10× the training samples of 10 s segments |
| Input: 96 steps | **The 96 segments ending at the prediction point** (96 s, about one pass). Shorter histories at a tool's start are left-padded. **Plus the memory of everything before** (O8) | The window is OmniFuser's; the memory carries the wear trend over the tool's life |
| Force per step: 3 pooled channels | **7 descriptors × 4 channels (Fx, Fy, Fz, \|F\|) = 28 values, plus a signal-present flag = 29** | This project's processing. The value MLP takes any width. Force-missing segments (D8) and padding are zeros with the flag 0. |
| Run, block and position codes | **Off** (`metadata="none"`) | Run and block codes are the known shortcut. The position code uses the pass's total length, which is unknown during the pass (review R2 of the project plan). |
| Usage | **log(1 + τ/H) and e/100 s**, appended to the force vector (31 values); switchable | Plan §3.3 allows them. Usage is in every arm of the study, as in RUL-Fuse. |
| Photo per step | **Per view, the latest photo with `available_s` < the step's τ** | Enforces D9 with the store's own times; equals OmniFuser's lag 1 |
| Several photo types | **One composite vector per pass: the views' frozen features side by side** (`image_views` = V; a missing or excluded view is zeros) | All photos of a pass share one availability time, so one index per step suffices (asserted, §9) |
| Outputs | **One value, ŷ/H**, from the last fused step | D6. The force-forecast and class heads are not built. |
| Loss | **Point-weighted MSE on y/H** (§3) **+ monotonic hinge + slope consistency on pairs** (O9) | D10's mean objective, with the shared point weights; RUL can only fall |
| Epoch selection | **Weighted whole-life validation MAE on the 10 s points**, best mean of 3 consecutive epochs | Shared with RUL-Fuse (plan §4.3): the same points and weights |
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
   - It returns `Linear(d + 64, 1)([encode(...)[:, -1] ; m_t])`, that is ŷ/H, with
     m_t the memory state (version 3; `Linear(d, 1)` without the memory).
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
   - The switches act on the memory's inputs in the same way. Modality dropout
     draws one code per sample (force missing, no photo yet), and the sample
     reads the memory computed with the same input missing.
5. **Memory (O8; version 6).** `rulfuse.models.memory.CausalMemory` (`forward_blocks`):
   a GRU, hidden 64, with one input per block of 96 steps (one OmniFuser window).
   - **Input:** the block's window goes through OmniFuser's force encoder (temporal
     embedding and multi-resolution extractor) and its image branch (photo
     projection and graph attention), the modules and weights the head's window
     uses (`OmniFuserRUL.encoded_streams`). Each stream is averaged over the
     window's 96 steps, and the GRU reads [encoded force ; encoded photos]: 2 × 32
     numbers. The cross-modal fusion is not run on past blocks: the head already
     fuses the current window, and fusion is the costliest part.
   - **Causal by construction.** OmniFuser mixes all 96 steps of a window, so a
     block is encoded only once all its steps exist. Step t reads the state after
     the last block complete at t; before the first one (step 95), zeros.
   - **Coverage.** The 96-step blocks tile the history exactly once. The head's
     window covers the newest steps, so nothing falls between the two.
   - **One computation.** Gradients reach the encoders through the memory as well
     as the head, and `memory_states` is the same in training and inference.
   - **Versions 3-5.** The GRU read 10-step block means of [force vector ;
     Linear(views × 1280 → 32) of the photo composite ; photo flag]. A LayerNorm on
     the block means was found to cancel a rise shared by all descriptors and was
     removed (§14).
6. **Temporal loss (version 3, O9).** `rulfuse.models.losses`: `monotonic_hinge`
   and `slope_consistency` on each training pair.

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
3. **Samples:** pairs of prediction points of one tool (version 3); B6's store
   has 36,919 points at 1 s, and about 29,000 train each fold. A point's window
   is the 96 steps ending at it, with y and the point weight at that point.
   The 10 s grid points are flagged (`core.store.grid_mask`). Selection and
   scoring use them, with their own point weights, which equal the 10 s store's.
4. **Normalization:** the descriptor and usage columns are standardized on
   the fold's 8 training tools; the flags stay 0/1. Photo features enter raw,
   as in OmniFuser; the projection learns their scale.
5. **Batches** (`FitTensors`, version 3): every unit's force vectors and
   per-step photo rows are held on the device, padded. Windows are read by tensor
   indexing; each sample passes its tool's table of photo composites (about 50
   rows) and per-step rows into it. This is OmniFuser's `(images, image_indices)`
   interface with precomputed features, without building samples one at a time.
6. **Photo-feature cache key** (version 3): the image table and the specs, not
   the store's signature, so stores cut at 1 s and 10 s share one EfficientNet
   pass.

## 7. Training, selection and evaluation

**One fit** (fold *i*, seed *s*):

1. Standardize on the 8 training tools; put every tool's inputs on the device.
2. Train for 30 epochs of 2,048 pairs (O10). Per batch of 16 pairs (32 windows):
   - run the memory over the training tools' whole histories;
   - encode the windows;
   - minimize the point-weighted MSE + 1/15 × hinge + 0.05 × slope term (O9; both terms
     move only the earlier prediction of each pair).

   Use Adam with learning rate 1e-4, and clip gradients at 1.0 (OmniFuser's
   settings).
3. After each epoch, compute the weighted whole-life MAE on the validation
   tool's 10 s points. Keep the epoch with the best mean of 3 consecutive
   values.
4. Predict every pre-failure point (1 s) of the validation and test tools.
   Write one row per point (`unit`, `segment row`, `fold`, `seed`, `on_grid`,
   `ŷ`) to `runs/omnifuser_rul/<store>/<arm>/fold-<i>/seed-<s>/predictions.parquet`,
   and the per-epoch validation MAE and loss terms to `fit.json`.

**Aggregation** (`rulfuse.omnifuser_rul.aggregate`):

- validation-tool scores at any time;
- test-tool scores only with `--after-freeze` on or after 2026-12-01 (D15),
  as `rulfuse aggregate` does for RUL-Fuse.

**Reported** (the same tables as RUL-Fuse):

- the plan §10.2 groups **point, safety, prognostic and stability**, on the
  window (headline), whole-life and recording-end points of the 10 s grid;
- the temporal stability over every 1 s point: the share of steps where ŷ rises,
  the mean rise, and |Δŷ + Δτ| (`metrics.temporal_stability`);
- force-missing points and the points after T4's and T2's gaps, separately;
- the mean ± sd over the 10 test tools of each tool's 3-seed ensemble metric,
  and the pooled value;
- tool-bootstrap 95 % intervals.

B6 gives points, not intervals, so its uncertainty and decision rows show "–".

**Inference modes (O13, D23; `rulfuse.inference`).** Every fit's predictions are
reported twice:

- **Current mode:** each prediction as the model makes it. It already uses the tool's
  past *signals*: the 96 s input window and the GRU memory of the whole history. It does
  not use the model's own past *predictions*. This is the primary result, fixed before
  any result.
- **Context mode:** the model's earlier predictions of the same tool also guide the new
  one. A running estimate is aged by the cutting time elapsed (the remaining life falls
  by one second per second) and blended with the new prediction: a prediction made Δ
  seconds ago keeps the weight e^(−Δ/120 s). Near the cap (a prediction within H/15 of
  H, which only says "at least about H"), ageing never takes the estimate below the new
  prediction. Each fit keeps its own context, and the seeds are then averaged, as in the
  current mode.

Properties, each tested (`tests/test_inference.py`):

- **Causal:** output t uses predictions 0..t of the same tool only.
- **Exact:** a trajectory that is right, cap included, is left unchanged.
- **It recovers:** after an early prediction that was too low, the output rises back.
  The leftover error falls as e^(−t/120 s), to 5 % after 6 minutes whatever its size. A
  "never rise" rule (running minimum) can only wait for the true RUL to come down to it:
  an error of e seconds lasts e seconds.
- **It removes jitter:** on synthetic noise, but not a bias that lasts a whole tool.

**Is the context mode valid in production?** Yes: it is causal, and it needs one number
per tool, which a deployed system has. It needs three operating rules:

- reset the context when the tool is changed;
- keep it across machine stops (it ages by cutting time, like the rest of the project, so
  idle time costs no life);
- feed it every prediction in time order.

Its limits:

- **Lag.** A real sudden change (a chipped edge) reaches the output with a 120 s time
  constant, 95 % after 6 minutes.
- **No bias correction.** It removes noise, not bias: if the model is wrong for a whole
  tool, so is the context.
- **Correlated inputs.** Neighbouring predictions share most of their inputs, so
  averaging them gains less than averaging independent ones would.
- **Flat predictions below the cap.** Once the model predicts more than 100 s below H, the
  context assumes the RUL falls one second per second. Where the model's predictions fall
  more slowly, or stay flat, the context settles up to 120 s × (1 − their slope) below
  them, on the pessimistic side. An example is a model that under-predicts the capped
  stretch by more than 100 s. This can raise the whole-life MAE, which includes the capped
  stretch, while the window MAE falls. A fake run on the real store showed both: window MAE
  78.9 → 74.1 s, whole-life MAE 73.1 → 97.5 s. The decision rule uses the window MAE, the
  project's headline (plan §10.1); the report shows both.

A learned context, which feeds past predictions into the network, was not chosen. It
would need a second model trained on its own outputs. That model would be trained on
true or teacher-forced past values and deployed on its own, which risks feedback of its
own errors (the lock-in that the version-5 loss removes). It would also be a second B6
setting (O12).

**How the two modes are compared** (`run_report.compare_modes`):

- the same points, tools and metrics for both modes;
- for each metric, context − current with a 95 % bootstrap interval over the tools;
- for the window MAE, the tools better and worse, and a two-sided Wilcoxon test.

**Decision rule, fixed in advance:** the context mode is better if it lowers the mean
window MAE by at least 10 s (H/150, plan §4.4) and the interval of that gain lies above
0. The current mode is better under the mirror condition. Otherwise neither is clearly
better.

The time constant was fixed before any B6 result, so the comparison on the validation
tools is fair too (those tools selected the epochs, for both modes alike). The test
comparison follows the seal (D15).

**Comparison with RUL-Fuse:**

- paired Wilcoxon signed-rank tests over the 10 test tools (window MAE),
  with Holm correction across the rows of the table (plan §10.3);
- B6 sits in every table next to B0-B5.

## 8. The modality study

> **Not run (O12, 2026-11-07).** B6 is trained once in its single configuration. This
> section stays as the design of a study that could be run later. Its code remains
> (`study.py`, `report.py`, `omnifuser-study`, `omnifuser-results`) but is not used.
> RUL-Fuse's Shapley study (project plan §11) remains the project's modality study.

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
| F+T+C without the memory (`force+tool+chip-nomemory`) | `force tool chip` | `--no-memory` (O11) |
| F+T+C without the temporal terms (`force+tool+chip-notemporal`) | `force tool chip` | `--mono-weight 0 --slope-weight 0` (O11) |

`rulfuse omnifuser-study --players force tool chip --list` prints the 13 arms
with their options. Shapley values and the paired tests use only the arms with
usage on, no dropout and the default memory and loss.

**Q5. Do the memory and the temporal terms help?** F+T+C against its two ablations,
on validation (any time) and on test (after the freeze): the window MAE, and the
stability (rising steps, |Δŷ + Δτ|). This is reported, not used to change B6
before the freeze.

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
| V-7 | Learns | On the toy adapter, the window MAE ends below 0.7 × the constant predictor's; a shuffled-target control stays above 0.7 × and above 1.5 × the real fit (version 5; the margin was 0.8, see §14). Since version 6 the toy fit uses 16-step memory blocks and 256 pairs per epoch: fit 0.44 ×, control 0.96 × |
| V-8 | Determinism and resume | The same seed gives the same predictions on the CPU; an interrupted fit resumes to the same result |
| V-9 | Test seal | `aggregate` refuses test scores before the freeze without `--after-freeze` |
| V-10 | Transfer contract | The core imports nothing from `omnifuser_rul`; photo kinds come from the store, never hard-coded (plan P22) |
| V-11 | Real-data smoke | One fold, 2 epochs, laptop CPU, cached features: runs, writes predictions, scores validation |
| V-13 | Labels at 1 s (O7) | A 1 s store contains every 10 s point with the same τ, RUL and window; failure times are unchanged; B2 on the grid points of the 1 s store equals B2 on the 10 s store |
| V-14 | Input dimensions (O7) | Windows are (B, 96, 31) force vectors and (B, 96) photo rows whatever S; OmniFuser's parameter count is the same at 10 s and finer steps |
| V-15 | Memory causality (O8) | The memory state at step t is unchanged when any step after t changes; only complete blocks are read; padding is never averaged; a rise shared by all inputs registers |
| V-16 | Switches reach the memory (O8) | With force, photos or usage off, changing that input changes neither the window output nor the memory |
| V-17 | Temporal loss (O9) | The hinge is zero for non-increasing pairs and lifts the earlier prediction of a rising pair; the slope term is zero when the decrease equals Δτ; **neither term reaches the later prediction**: in `pair_loss`, the later half's gradient equals the MSE's, so a stuck earlier prediction cannot drag a later one; a stuck later prediction moves the earlier one by at most about 1/15 × H plus a small slope share; inside a real toy fit, every batch's temporal gradient reaches earlier predictions only, with the spec's weights |
| V-18 | Scoring only the right fits | Scores use only fits whose `fit.json` records the arm's settings (a partly rerun arm never mixes) |
| V-19 | Inference modes (O13) | The context mode is causal, leaves an exact trajectory unchanged (cap and uneven steps included), recovers from a low start as e^(−t/T), removes jitter without sliding off the cap; each fit, unit and role is filtered on its own; the report's comparison finds no difference for constant offsets and finds the context mode better for independent noise |

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

- One fit (version 3) trains on 30 × 4,096 windows, plus the memory over
  the 8 training tools for each of its 3,840 batches. Expected 3-6 minutes on an
  A100 with cached features; the pilot measures it.
- **Version 6:** the memory encodes each training tool's complete 96-step blocks in
  every batch, about 300 extra windows next to the batch's 32. A laptop-CPU smoke fit
  took 135 s against 38 s with version 3's memory. Expected 5-15 minutes per fit on an
  A100, 2.5-7.5 GPU-hours for the 30 fits; the job asks for 16 hours.
- **Version 4 (O12): one run of 30 fits, about 1.5-3 GPU-hours (version 6: 2.5-7.5),
  in one job, followed by a CPU report job.** The rest of this list was the version-3
  study's cost:
- 13 arms × 10 folds × 3 seeds = 390 fits, about 20-40 GPU-hours. The 30 fits of
  an arm form one job, so the study is 13 jobs of at most 8 hours each.
- If the pilot's fit takes more than 6 minutes, halve `--pairs-per-epoch` (1,024)
  for the whole study before it starts. That is a cost decision, made before any
  result. If it takes more than 15 minutes, also split each arm's folds over two
  jobs (`--folds 1 2 3 4 5` and `--folds 6 7 8 9 10`).
- It fits the 4-GPU limit in about a day of wall time.
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
| GPU time competes with RUL-Fuse | 20-40 GPU-hours in 13 jobs, run in gaps; the arms are independent and resumable; the pair budget can be halved after the pilot |
| 1 s descriptors are noisier than 10 s ones | Each is still 1,000 samples (about 66 spindle turns); the memory averages 10 s blocks; the 10 s grid scores the same points as before |
| The memory's backpropagation through about 500 blocks per tool is slow or unstable | Blocks of 10 steps; gradient clipping at 1; the no-memory ablation bounds the effect |
| The temporal terms over-smooth (a model that always predicts slowly falling values) | The MSE term dominates on the targets; the no-temporal ablation measures the effect |

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
rulfuse/models/  (shared with RUL-Fuse)
  memory.py      CausalMemory: GRU/LSTM over block means, causal (D21, O8)
  losses.py      monotonic_hinge, slope_consistency (D22, O9)
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

### Version 3 (2026-11-06)

**Training examples.** Under version 2 (10 s steps) a fold trained on 2,823-3,343
prediction points (median 2,956), all from 8 trajectories. With finer segments:

| Segment length | Segments | Prediction points | Window points | Training points per fold (median) | Longest tool (steps) |
|---|---|---|---|---|---|
| 10 s (version 2, RUL-Fuse) | 5,125 | 3,747 | 1,361 | 2,956 | 670 |
| 5 s | 10,014 | 7,350 | 2,647 | 5,802 | 1,364 |
| 2 s | 25,181 | 18,467 | 6,673 | 14,572 | 3,414 |
| **1 s (version 3)** | 50,378 | 36,919 | 13,310 | 29,123 | 6,830 |

At every length, all 5,125 points of the 10 s store are found with identical τ and RUL
(largest difference 0.0 s), and the failure times are equal. The RUL and failure-time
labels therefore need no change: the failure time is the end of the TTF pass for every S,
and RUL = τ* − τ follows from each segment's end. 1 s was chosen as OmniFuser's own step;
the samples are strongly correlated (the information is still 8 failure trajectories), so
the gain is more distinct windows for a deep model, not more independent tools.

**OmniFuser's input dimensions.** OmniFuser requires exactly `input_len` = 96 steps per
window and any number of force channels; B6 always builds 96-step windows of 31 values, so
nothing in OmniFuser changes with S (V-14). OmniFuser's own loss uses only the force and
class terms (its C2F alignment and orthogonality terms are logged, not optimized), so B6's
head-only replacement remains faithful; RevIN is off by default and stays off (it would
remove the absolute force level).

**Implemented:** `rulfuse/models/{memory,losses}.py`; `core.store.grid_mask`;
the adapter's and the toy's `segment_s`; B6's data (`UnitInputs`, `FitTensors`,
`sample_pairs`, the segment-independent photo key), model (memory, dropout variants),
training (pairs, temporal loss, grid selection, `on_grid`, loss terms in `fit.json`),
aggregation (grid scoring, stability metrics, B2 on the grid, only matching fits),
study (13 arms), report (stability columns) and CLI (`--segment-s`, `--eval-grid-s`,
`--no-memory`, `--memory-hidden`, `--memory-block`, `--mono-weight`, `--slope-weight`,
`--max-lag`, `--pairs-per-epoch`).

**Verification:**

| # | Result |
|---|---|
| V-13 | The table above; real-data test: B6's 1 s store has 36,919 points, its grid is exactly the 5,125 segments of the 10 s store, and B2 on it equals 467.8 s |
| V-14 | Toy windows (B, 96, 17) at 10 s and at 2 s; OmniFuser's parameter count identical; real data: (B, 96, 31), 296,181 trainable parameters with the memory |
| V-15 | GRU and LSTM, blocks 1 and 4: changing any later step leaves earlier states unchanged; zero state before the first complete block. **Found and fixed:** a LayerNorm on the block means cancelled a rise shared by all inputs (the common rise of the force descriptors with wear); removed |
| V-16 | Force, photos and usage switched off: neither the output nor the memory changes when that input changes |
| V-17 | Hinge and slope terms by hand; on the toy (2 seeds), the fits with the terms have a lower mean rise than the same fits without them |
| V-18 | A smoke fit with other settings is not scored for the default arm (and is, 1 of 10, with its own settings) |
| V-11 (version 3) | Real data, laptop CPU: `omnifuser-prepare` builds the 1 s store (36,919 points) and its descriptor cache in 38 s; a 2-epoch fit of force + tool + chip on fold 1 (256 pairs per epoch) takes 38 s, trains on 30,103 points, predicts all 3,818 validation points of T2 and flags exactly its 393 grid points (116 in the window); B2 on that grid is 228.2 s, as on the 10 s store; the stability metrics are reported (rising steps 0.46 after 2 epochs, a smoke value) |

The project's suite: 151 tests, 149 pass, 2 skip (the Mantis and DINOv2 checkpoint tests,
which run in Ruche job J0); the B6 tests take about 3 minutes on the laptop CPU; ruff and
pyright report no issues.

**What remains:** the same four steps as above. The pilot (runbook §12.1) now also
decides the pair budget (§10).



### Version 4 (2026-11-07): one full run and its report

**Your decision.** No experiments on the OmniFuser implementation and no comparison of
settings (for example GRU against LSTM): one setting, one full training run, and test
results for a report. The run uses the GRU memory and the loss with the monotonic penalty.

**What changed:**

| Part | Version 4 |
|---|---|
| Run | One arm: force + the admitted photo types + usage, GRU memory, monotonic penalty 1.0 and slope term 0.5, 30 epochs of 2,048 pairs, 10 folds × 3 seeds (`slurm/omnifuser_rul.sh` submits it as `b6-train`, with `b6-report` chained after it) |
| Report | `rulfuse omnifuser-report` (`rulfuse/omnifuser_rul/run_report.py`): settings and fits; per test tool and in summary: window MAE, RMSE, bias, whole-life MAE, late and early rates (100 s), the asymmetric score of plan §10.2, the share within ±20 %, rising steps, the seed spread; B0, B1, B2 on the same points; the paired comparison with B2 (mean gain, 95 % bootstrap interval, tools improved, one-sided Wilcoxon); error by true-RUL range; three figures (RUL trajectories, per-tool MAE, error by RUL range); validation results |
| Test seal | Test results are included on or after 2026-12-01 (D15). `--unseal` includes them earlier, and the report opens with a note recording the date |
| Not run | The modality study (§8), the ablations (O11), the pilot of version 3, and any memory-cell comparison (the GRU is fixed; `CausalMemory` still accepts `cell="lstm"` as a generic option, unused) |
| Metrics added | `rulfuse.metrics`: `phm_score`, `late_rate`, `early_rate`, `relative_accuracy`, `error_by_bin`, `bootstrap_mean_ci` |

**Verification:**

- **Report tests.** A fake run on the toy store has predictions = target + a per-tool bias
  ± 10 s by seed. They check:
  - each tool's window MAE and bias equal its bias;
  - the seed spread equals the spread of the two seeds;
  - B2 equals `b2_by_unit` on the same points;
  - the gain over B2 and the tools improved;
  - the error-by-range points sum to the window points;
  - the seal before the freeze, the `--unseal` note, and the read after the freeze;
  - the figures are written.
- **Metrics by hand.** The new metrics are checked against hand computations.
- **Real store.** A fake run on the real 1 s store gives B2 = 467.8 s, B0 = 412.0 s and
  B1 = 512.5 s on the report's points, exactly plan §2.7. This shows the report scores the
  right points.
- **Figures.** They were rendered and inspected.
- **Submission.** A dry run of `slurm/omnifuser_rul.sh` with a fake `sbatch` submits
  `b6-train`, then `b6-report` with `afterok`. Options after `--` reach both jobs, and an
  excluded photo type is refused.

### Version 5 (2026-11-07): the penalty fix and two inference modes

**Your question.** Could the penalty on rising predictions trap the model? If its first
predictions are too low, later ones might be unable to come back up.

- **At prediction time: no.** Each prediction is computed from its own inputs. The model
  never sees its own past predictions, and nothing clamps them.
- **In training: yes, with version 4's loss.** The penalty pushed both predictions of a
  rising pair towards each other, with weight 1.0. When the model could not correct an
  earlier prediction, the later one settled exactly at that stuck value (table below).

**The fix (O9):**

- **The later prediction is the reference.** In both temporal terms it is detached, so the
  terms move only the earlier prediction.
- **Smaller weights:** 1/15 on the hinge and 0.05 on the slope term. The hinge weight acts
  as a tolerance: it can hold a prediction at most about 1/15 × H = 100 s from what the MSE
  wants. With the detach the weights must stay small. The hinge then only lifts, so noise
  the model cannot remove shifts the level up: about +10 to +15 s at these weights, +160 to
  +215 s at 1.0.

**Your second request (O13).** A second inference mode, in which the tool's earlier
predictions guide the new one, and a way to compare the two modes (§7). It is computed from
the same 30 fits, so nothing is trained twice.

**What changed:**

| Part | Version 5 |
|---|---|
| Temporal loss | `rulfuse.models.losses`: the later prediction is detached in both terms (shared with RUL-Fuse). `train.pair_loss` is a batch's loss, called by `fit()`. `ArmSpec` and CLI defaults: 1/15 and 0.05 |
| Inference | `rulfuse/inference.py` (generic): `context_filter`, `apply_mode`, T = 120 s, cap margin H/15. `aggregate.with_mode` applies a mode to each fit's predictions of one unit and role, in time order |
| Report | Both modes per role. `compare_modes` applies the rule fixed in advance. New `mode_comparison.csv`; a `mode` column in the per-unit CSVs; `b6_context_s` in the trajectories; `B6 context` rows in `error_by_bin.csv`. All three figures show both modes, in categorical slots 1-3. The CLI prints both modes and the verdict |
| Tests | `tests/test_inference.py` (6). In `tests/test_omnifuser_rul.py`: the training loss moves only the earlier prediction; a stuck later prediction moves the earlier one only a little; the CLI and spec defaults agree; inside a real fit, every batch's temporal gradient reaches earlier predictions only; the mode comparison finds a real difference; the report tests cover both modes |

**Choices made before any B6 result:**

- **T = 120 s.** On the real tools' time axes with synthetic errors:
  - jitter: 46.6 → 8.5 s window MAE;
  - errors that change pass by pass: 72.6 → 45.3 s;
  - a constant per-tool bias: not removed (111.8 → 113.8 s);
  - a mix of all four: 142.8 → 133.5 s, the best of T = 30-480 s for the mix.
- **The cap rule.** Near the cap, ageing never goes below the new prediction. Pausing the
  ageing instead cost a perfect model 7.4 s of window MAE at T = 120 s; this rule costs 0
  at every T. Ageing always drifted on the capped stretch (110 s there).

**Verification.** Every check uses the code `fit()` runs (`pair_loss`, `sample_pairs`).

| Check | Version 4's loss (1.0, 0.5, both predictions) | Version 5 (1/15, 0.05, earlier only) |
|---|---|---|
| Earlier prediction stuck 450 s too low (truth 900 → 700 s, in a 16-pair batch) | later settles at 450 s (250 s too low) | later stays at 700.0 s |
| Earlier stuck 600 s too low | later at 300 s (400 s too low) | later stays at 700.0 s |
| Later prediction stuck 300 s too high | earlier 112.5 s too high | earlier 73.6 s too high |
| Level shift from noise the model cannot remove (sd 30 s) | 0 | +9.7 s (smooth noise) to +14.1 s (white) |

Toy fits (one fold, 15 epochs × 512 pairs, 2 seeds):

| Hinge weight | Mean rise | Window MAE | Bias |
|---|---|---|---|
| 0 | 2.22 s | 32.9 s | +27.5 s |
| 1/15 (the run) | 2.11 s | 34.6 s | +30.5 s |
| 0.2 | 1.66 s | 41.7 s | +40.8 s |
| 1.0 | 0.62 s | 62.4 s | +62.4 s |

A stronger penalty removes more rises, but costs accuracy through the late bias. The run's
weight removes few rises in training, at almost no cost. The context mode removes rises at
inference instead.

**Two tests changed after the toy measurements:**

- **The shuffled-target control.** Its limit fell from 0.8 to 0.7 × the constant predictor;
  the fit must still be below 0.7 × and the control at least 1.5 × the fit. With the terms
  off the control scores 0.78 ×. Only version 4's strong penalty had kept every fit above
  0.8 ×, the real fit included (89.6 s against 69.9 s).
- **The "fewer rises" fit comparison.** It is replaced by the check inside a real fit
  above: at the run's weights the effect is too small to see in a 5-epoch toy fit.

**Context mode, fake run on the real store** (per-tool bias, noise per pass and per step):

- B0, B1 and B2 on the test points: 412.0, 512.5 and 467.8 s, as plan §2.7.
- Window MAE 78.9 s current, 74.1 s context (gain 4.8 s, interval −1.3 to 12.0 s). Verdict:
  neither mode is clearly better.
- RMSE 88.8 → 79.4 s; rising steps 0.454 → 0.365.
- Whole-life MAE 73.1 → 97.5 s: the flat-prediction cost described in §7.
- The figures were rendered and inspected.

### Version 6 (2026-11-08): the memory reads OmniFuser's encodings

**Your request.** The GRU should take the encoded time series and the encoded images, in
training and in inference. In versions 3-5 it read the raw step inputs: the force
descriptors, a separate projection of the photo features, and a photo flag.

**Design (O8, §5):**

- **Input.** One input per block of 96 steps: OmniFuser's encoded force series and encoded
  photos for that block, each averaged over the block's 96 steps.
- **Causality.** OmniFuser mixes all steps of its window (the multi-resolution extractor,
  graph attention over the photos, attention in the fusion), so its per-step outputs cannot
  be read step by step. A block is encoded only once it is complete.
- **Coverage and cost.** 96-step blocks tile the history exactly once. 10-step blocks would
  have meant about 2,900 encoded windows per batch instead of about 300, likely more than a
  day of A100 time for the run.
- **No fusion on the past.** The fusion is not run on past blocks: the head fuses the
  current window, and fusion is the costliest part.
- **One computation.** The same code serves training and inference, and gradients reach the
  encoders through the memory.

**What changed:**

| Part | Version 6 |
|---|---|
| Model | `OmniFuserRUL.encoded_streams` (OmniFuser's force encoder and image branch on a window); `memory_states` encodes each complete block and feeds `CausalMemory.forward_blocks`. The memory's own photo projection is removed |
| Shared memory | `CausalMemory.forward_blocks`: one input per block, the same reading rule as before (`forward`, the block means used by RUL-Fuse, is unchanged) |
| Settings | `MemorySpec.block`, `ArmSpec.memory_block` and `--memory-block` default to 96 (was 10). The settings signature changes; no fit existed |
| Ruche | `b6-train` asks for 16 hours (was 8): about 5-15 minutes per fit expected |

**Verification:**

- **The input is OmniFuser's encoding.** `encoded_streams` returns exactly the two streams
  that `OmniFuser.encode` fuses (checked with hooks). The GRU's input for each block equals
  the mean of those streams over the head's own window ending at the block's last step;
  blocks a tool has not completed are padding.
- **Causal.** Changing any force value or photo after step t leaves the memory up to t
  unchanged, at three values of t. The first state appears at the first block's last step.
- **Gradients.** They reach the force embedding, the photo projection and the GRU through
  the memory; the fusion is not on that path.
- **Unchanged checks.** Switched-off inputs reach neither the output nor the memory;
  modality dropout reads the matching memory; a tool's memory does not depend on the other
  tools in the tensors. These tests now use 16-step blocks (each still a 96-step encoded
  window), because the toy tools are only 82-109 points long.
- **Real data, laptop CPU.** A 2-epoch smoke fit of force + tool + chip on fold 1 (256
  pairs per epoch) runs, trains on 30,103 points, and writes the validation and test
  predictions: 135 s against 38 s in version 3.
- **Toy learning test (V-7), found by the suite.** The toy tools are only 82-109 points
  long, so with 96-step blocks most of them complete none. Validation tool U2 never does,
  and two training tools get memory only at their last ~12 points, where it flags the end
  of life. After 8 epochs of 128 pairs:

  | Memory | Validation MAE |
  |---|---|
  | None | 160.7 s |
  | 96-step blocks | 280.5 s (stuck at 335 s for 4 epochs) |
  | 16-step blocks | 128.5 s |

  The test passed before because versions 3-5's raw-input memory gave the toy an easy usage
  signal. The toy fit now uses 16-step blocks and 256 pairs per epoch (fit 80.3 s, 0.44 ×
  the constant predictor). Its shuffled control now really collapses: 175.2 s, 0.96 ×, where
  the raw-input memory had given it a usage shortcut (0.78 ×). Real tools are 1,800-5,400
  points long, 19-56 complete blocks each, so this toy effect does not carry over. The
  first fits' logs should still be checked for a validation MAE stuck at its first value.
