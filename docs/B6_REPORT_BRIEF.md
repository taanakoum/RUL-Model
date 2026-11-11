# B6 "OmniFuser-RUL": source pack for a report

This file holds what you need to write a thesis chapter or paper section on baseline B6
and to explain its results. It assumes you know machine learning and have not seen this
repository.

**How to read the numbers.**

- Every number comes from a project file (paths in section 8) or from a computation on
  those files. Numbers marked **†** are computed for this brief and are not in
  `REPORT.md`; Appendix A gives each method.
- "Data shows" and "Interpretation" are kept apart in section 6. The data statements can
  be quoted. The interpretations are explanations that fit the data. No experiment in
  this repository tests them.
- Units are seconds (s) of cutting time. "Late" or "optimistic" means the model predicts
  more remaining life than is left (the dangerous error). "Early" or "pessimistic" means
  it predicts less.

---

## 1. Purpose of B6 and its place in the project

**The project.** RUL-Fuse is a multimodal model that predicts the remaining useful life
(RUL) of milling tools from cutting forces and photos (`PROJECT_PLAN.md`).

**B6 is a baseline, decision D20.** B6 ("OmniFuser-RUL") takes a published multimodal
architecture, OmniFuser (arXiv:2511.01320, reproduced in the author's own repository,
tag `v1.1`), and gives it an RUL output. It then runs on this project's targets,
preprocessing, folds, point weights, epoch selection and metrics. Only the architecture
differs from RUL-Fuse. The question it answers is: what does a published fusion
architecture reach on the same task and the same inputs?

**What B6 is not** (`OMNIFUSER_RUL_PLAN.md` O1, `PROJECT_PLAN.md` D20):

- It is not a candidate for RUL-Fuse's fusion gate (Gate 2).
- No RUL-Fuse design decision reads it. Under D15 this holds for the test numbers until
  the design freeze of 2026-12-01.
- It is not tuned. Decision O12 (2026-11-07) fixes one configuration, trained once
  (10 folds x 3 seeds = 30 fits). The modality study, the ablations and a GRU against
  LSTM comparison were designed and then deliberately not run.

**Version history in one paragraph** (`OMNIFUSER_RUL_PLAN.md`). v3 (2026-11-06) added 1 s
steps, a GRU memory and a penalty on rising predictions. v4 (2026-11-07) fixed the single
full run. v5 (2026-11-07) made the temporal loss move only the earlier prediction of a
pair, cut its weights to 1/15 and 0.05, and added the second inference mode. **v6
(2026-11-08) made the GRU read OmniFuser's own encodings of 96-step blocks. The run
reported here is v6.**

**Position in the baseline ladder** (`PROJECT_PLAN.md` section 8):

| ID | Model | Deployable | Window MAE (s, mean over the 10 test tools) |
|---|---|---|---|
| B0 | Constant H/2 = 750 s | No (it knows the point is in the window) | 412.0 |
| B1 | Mean training failure time minus elapsed time, capped | Yes | 512.5 |
| B2 | E[min(tau* - tau, H) given tau* > tau] over the training tools (usage rule) | Yes | 467.8 |
| **B6** | **OmniFuser-RUL, force + tool photo + usage** | Yes | **587.4** |

---

## 2. Data and task

### 2.1 Dataset

- **Nonastreda milling** (public dataset). 10 tools, T1 to T10, each run from new until
  well past dulled. 512 recorded passes. Per tool, 40 to 60 passes
  (`PROJECT_PLAN.md` section 2.1).
- **Force:** Fx, Fy, Fz at 1 kHz. The trimmed pass is 97 s long (median).
- **Tool photo:** one per pass, taken after the pass. About 1550 x 500 px; T1 and T2 are
  3100 x 1000.
- **Labels used only for the target and checks:** `tool_label` (sharp, used, dulled),
  `flank_wear` (um).
- **Force-missing passes (D8):** a duplicated force recording is attached to the latest
  pass of its group. 134 of the 512 passes (26 %) have missing force. They enter as
  zeros with a signal flag of 0.

### 2.2 Inputs used by B6, and why chip and workpiece photos are excluded (D24)

The B6 configuration is **force + tool photo + usage** (`OMNIFUSER_RUL_PLAN.md` O5).

- **Force:** per 1 s segment and per channel (Fx, Fy, Fz, |F|), 7 level descriptors (log
  RMS, log peak, kurtosis, crest, mean, slope, drift): 28 values, plus a signal-present
  flag.
- **Usage:** log(1 + tau/H) and e/100 s, where tau is the cutting time so far and e the
  seconds into the current recording. Total per step: 31 values. Usage lets the model
  use elapsed time, as B2 does.
- **Tool photo:** frozen EfficientNet-B0 (ImageNet weights), 1,280 pooled features. A
  photo is visible only to predictions made after its pass ends (D9, `available_s < tau`).
  Preparation: tool edge band resized to 224 x 672.

**D24 (2026-11-11, decided before any RUL result).** The Stage-3 feature check ran a
stage-step leakage probe on every photo type. The probe counts, in how many of the 10
tools a `tool_label` stage boundary is the largest run-to-run step of a photo statistic.
The limit is 3 tools. Table (`PROJECT_LOG.md`, section "Stage 3 finished"; report
`nonastreda-3e357b5a23ff.json`):

| Source | PC1 | PC2 | PC3 | Wear readout | Probe MAE (mean predictor 35.8) | Shuffled-control MAE |
|---|---|---|---|---|---|---|
| True wear (`hi_reference`) | - | - | - | 3 | - | - |
| Force signals | 2 | 2 | 1 | 1 | - | - |
| Workpiece photo | 10 | 10 | 0 | 10 | 14.9 | 32.7 |
| Chip photo | 6 | 3 | 2 | 7 | 19.4 | 34.1 |
| Tool photo | 1 | 4 | 2 | 3 | 21.7 | 35.9 |

- **Workpiece and chip stay out.** Their photo statistics step at the annotators'
  stage boundaries in 10 of 10 tools (workpiece PC1, PC2, readout) and 7 of 10 (chip
  readout), far above the 3 of 10 of the true wear. A model could read the stage label
  from the photo.
- **Tool is readmitted.** Only its PC2 crossed the limit (4 of 10; limit 3). Its readout
  scores 3 of 10, equal to the true wear. Its probe beats the mean predictor (21.7 against
  35.8) and the shuffled control collapses (35.9). Recorded in
  `configs/feature_check_overrides.json`.
- **Consequence:** B6's inputs equal OmniFuser's original force + tool pairing.

### 2.3 Target

- **RUL in seconds of cutting only (D6).** Idle time between passes is not recorded, so
  wall-clock time is not predicted.
- **Failure time tau\* (TTF rule, D2):** the first recorded pass labelled `dulled` that is
  followed by at least 2 more `dulled` passes among its next 3 recorded passes. Smoothed
  flank wear at that point is 108.5 to 122.6 um on every tool, matching the dataset's
  110 um "dulled" threshold.
- **RUL = tau\* - tau; target y = min(RUL, H), H = 1,500 s (D3)** (about 25 min, about 15
  median passes). Training uses every pre-failure point with the capped target.
- **Gap-aware time axis (D12):** runs missing between a tool's first and last recorded
  run count as cutting at the median duration of the tool's recorded passes. T2 run 10
  (4 passes) and T4 runs 6-9 (16 passes). Confirmed by the author on 2026-11-04.
- **Evaluation window:** points with RUL <= H. **1,361 points** over the 10 tools.
  Inside the window the target equals the true RUL.

| Tool | Cutting time to failure (s) | Pre-failure points (10 s grid) | Window points |
|---|---|---|---|
| T1 | 3,000 | 308 | 155 |
| T2 | 4,203 | 393 | 116 |
| T3 | 3,463 | 355 | 156 |
| T4 | 3,657 | 216 | **20** |
| T5 | 1,827 | 188 | 156 |
| T6 | 4,472 | 458 | 154 |
| T7 | 4,543 | 465 | 155 |
| T8 | 4,474 | 459 | 154 |
| T9 | 3,837 | 379 | 148 |
| T10 | 5,367 | 526 | 147 |
| All | | 3,747 | 1,361 |

Source: `PROJECT_PLAN.md` section 2.2. T4's window holds 20 points because 14 of its
last 16 passes before failure are unrecorded (runs 6-9). In B6's test file its 20 window
points have a true RUL between 0 and 183.7 s **†**. T5 has the shortest life (1,827 s);
the next shortest is T1 at 3,000 s.

### 2.4 Time step and scoring grid (O7, D7)

- B6 has its own store cut at **1 s** segments (OmniFuser's native step): store
  `nonastreda/6e0cb095967b`, 36,919 prediction points. A fold trains on about 29,000
  points (27,711 to 32,976 per fit; median 29,123 **†**).
- **Scoring is on the project's 10 s grid** (3,747 points, 1,361 in the window), which a
  1 s store contains exactly with the same tau and RUL. B6, RUL-Fuse and B0-B2 are
  compared on the same points. Stability metrics use every 1 s point.
- B2 on the grid points of the 1 s store reproduces 467.8 s, and B0 and B1 reproduce
  412.0 and 512.5 s (`OMNIFUSER_RUL_PLAN.md` V-13).

### 2.5 Folds (D1)

10-fold leave-one-tool-out. In fold i, test tool = Ti, validation tool = T(i+1) (T1 after
T10), training on the other 8 tools. Each tool is therefore the test tool of one fold and
the validation tool of another. The validation tool picks the epoch; the test tool is only
predicted.

| Fold | Test | Validation |
|---|---|---|
| 1 | T1 | T2 |
| 2 | T2 | T3 |
| 3 | T3 | T4 |
| 4 | T4 | T5 |
| 5 | T5 | T6 |
| 6 | T6 | T7 |
| 7 | T7 | T8 |
| 8 | T8 | T9 |
| 9 | T9 | T10 |
| 10 | T10 | T1 |

---

## 3. Method

### 3.1 OmniFuser as adapted

OmniFuser (original task: forecast 96 s of force, classify wear) is used with these parts
unchanged (`OMNIFUSER_RUL_PLAN.md` sections 3 and 5; OmniFuser `README.md` section 2):

- **Force encoder:** temporal embedding (value MLP + position encoding), then MRTE, four
  multi-resolution blocks (`num_rtd_layers` 4, stride 3, rank 8).
- **Photo branch:** frozen EfficientNet-B0 features, a projection to d = 32, and a
  temporal graph attention network (4 layers x 4 heads) over the photo nodes.
- **C2F cross-modal fusion:** shared/private split, adaptive landmarks (k = 4), gates,
  2 rounds plus refinement.
- **Read-out:** the last fused step of a 96-step window.

Changes made for RUL (about 25 lines in OmniFuser v1.1; about 60 lines of wrapper in
`rulfuse/omnifuser_rul/model.py`):

- OmniFuser v1.1 adds `OmniFuser.encode()` (returns the fused sequence; `forward()` output
  is bit-identical to v1.0 in 48 tested configurations) and `image_views` (the projection
  takes views x 1,280 features).
- The force-forecast and class heads are deleted. One **RUL head**,
  `Linear([last fused step (32) ; memory state (64)]) -> y/H`, replaces them. Predictions
  are clipped to [0, H] at evaluation.
- **Input window:** the 96 one-second steps ending at the prediction point (96 s, about
  one pass); shorter histories at a tool's start are left-padded with zeros and flag 0.
- **Metadata codes off** (`metadata="none"`): run, block and position codes are the known
  shortcut and the position code needs the pass's unknown total length.
- **Not borrowed from RUL-Fuse:** trajectory augmentation, channel/token dropout, interval
  heads, Mantis and DINOv2 encoders. Those are RUL-Fuse's contributions.
- **Trainable parameters per fit: 173,269** (`fit.json`). The EfficientNet-B0 weights are
  frozen and not counted.

### 3.2 GRU memory over encoded 96-step blocks (v6, decisions O8, D21)

The 96-step window forgets everything older. A GRU (hidden 64, `CausalMemory`) carries the
tool's whole history.

1. The history is cut into consecutive blocks of 96 steps. Block j ends at step
   (j + 1) x 96 - 1.
2. Each complete block's 96-step window goes through OmniFuser's own force encoder and
   photo branch (same modules and weights as the head's window). Each stream is averaged
   over the block: GRU input = [encoded force ; encoded photos] = 2 x 32 = 64 numbers.
3. At step t the head reads the GRU state after the last block complete at t. Before the
   first complete block (step 95) the state is zero. The unfinished block is the newest
   data, which the head's window sees.
4. **Causal by construction.** OmniFuser mixes all 96 steps of a window, so a block is
   encoded only when all its steps exist. Tested: changing any step after t leaves the
   memory up to t unchanged.
5. Cross-modal fusion is not run on past blocks (the head already fuses the current
   window; fusion is the costliest part). Gradients reach the encoders through both the
   head path and the memory path. Training and inference compute the memory identically.

Before v6 (v3-v5) the GRU read 10-step block means of the raw descriptors. A test on the
toy dataset found that 96-step blocks are too coarse for toy tools of 82-109 points; real
tools have 1,800-5,400 points and 19-56 complete blocks (`OMNIFUSER_RUL_PLAN.md` v6).

### 3.3 Loss (D22, O9)

Each training sample is a pair (earlier, later) of prediction points of one tool. The
earlier point is one step before the later one half the time, and 2 to 200 steps before it
otherwise. Everything is in units of y/H.

    loss = point-weighted MSE (both points of every pair)
         + (1/15)  x  mean relu( y_hat_later  - y_hat_earlier )            monotonic hinge
         + 0.05    x  mean ( (y_hat_earlier - y_hat_later) - dtau/H )^2    slope consistency

- The hinge says a predicted RUL never rises. The slope term applies where the earlier
  target is below the cap: there the true RUL falls by exactly the elapsed time dtau.
- **The later prediction is detached in both terms.** The terms move only the earlier
  prediction. Why (`PROJECT_LOG.md`, 2026-11-07; `losses.py`): with weight 1.0 on both
  predictions, an earlier prediction the model could not correct dragged the later one
  down to it. In a test, an earlier prediction stuck 450 s too low made the later one
  settle 250 s too low; with the detach the later one stays right.
- **The weights act as a tolerance.** With the detach, the hinge can only lift earlier
  predictions, so noise the model cannot remove shifts the level up. The design notes
  estimate +10 to +15 s at these weights and +160 to +215 s at weight 1.0 (toy
  measurements; not re-measured on the real run).
- **Point weights (shared with RUL-Fuse, plan section 7.3):** each tool weighs the same;
  points with RUL > H weigh 0.3 (`w_far`).

### 3.4 Training and selection

- 30 epochs of 2,048 pairs (4,096 windows) per epoch; Adam, learning rate 1e-4, batch of 32
  windows (16 pairs), gradient clipping at 1.0 (OmniFuser's settings). No learning-rate
  schedule, no early stopping, no augmentation.
- After each epoch, the **weighted whole-life MAE** on the validation tool's 10 s points
  is computed (the tool's grid weights; points with RUL > H weigh 0.3). **The kept epoch
  is the centre of the 3 consecutive epochs with the lowest mean** (so a single lucky
  epoch cannot win; if the first three epochs have the lowest mean, epoch 2 is kept).
  No EMA weights.
- **3 seeds** (0, 1, 2) per fold, 30 fits in all. A tool's prediction is the **mean of the
  3 seeds' predictions** (ensemble). Seeds also differ in the training draws.
- The test tool's predictions are written by every fit but scored only by the report.

### 3.5 Two inference modes (D23, O13)

Both modes come from the same 30 fits; nothing is trained twice.

- **Current (primary, fixed in advance):** each prediction as the model makes it. The model
  already reads past signals (window and memory) but never its own past predictions.
- **Context:** a running estimate is aged by the cutting time elapsed (remaining life
  falls 1 s per s) and blended with the new prediction r_t:

      aged_t = x_{t-1} - dtau_t                         (not below r_t while r_t >= H - 100 s)
      x_t    = aged_t + a_t (r_t - aged_t),   a_t = 1 - exp(-dtau_t / 120 s)

  A prediction made D seconds ago keeps weight exp(-D / 120 s). The time constant 120 s
  and the cap margin H/15 = 100 s were fixed before any B6 result, using synthetic errors
  on the real time axes. Each fit has its own context; seeds are averaged afterwards. The
  design notes list its limits: it removes noise, not a bias that lasts a whole tool, and
  on flat predictions below the cap it settles up to 120 s x (1 - slope) lower than the
  predictions.
- **Comparison rule, fixed before any result:** the context mode is better if it lowers
  the mean window MAE over the tools by at least 10 s (H/150, the project's smallest
  meaningful gain) **and** the 95 % bootstrap interval (over tools) of that gain lies above
  0. The current mode is better under the mirror condition. Otherwise neither is clearly
  better. A two-sided Wilcoxon test over the tools is reported alongside.

### 3.6 Hyperparameter table

| Group | Setting | Value | Source |
|---|---|---|---|
| Task | Horizon H | 1,500 s | D3 |
| Task | Segment / time step | 1 s | O7 |
| Task | Scoring grid | 10 s | D7 |
| Task | Far-point weight w_far | 0.3 | plan section 7.3 |
| Input | Window | 96 steps (96 s) | OmniFuser |
| Input | Force vector per step | 31 = 28 descriptors + 1 flag + 2 usage | O5, plan section 4 |
| Input | Photo features | frozen EfficientNet-B0, 1,280 (ImageNet) | O2 |
| Model | d (model width) | 32 | OmniFuser v1.0 default |
| Model | Multi-resolution blocks | 4 (stride 3, rank 8) | OmniFuser default |
| Model | GAT layers x heads | 4 x 4 | OmniFuser default |
| Model | C2F rounds, landmarks k, rank | 2, 4, 8 | OmniFuser default |
| Model | Dropout | 0.1 | OmniFuser default |
| Model | Trainable parameters | 173,269 | `fit.json` |
| Memory | Cell, hidden size | GRU, 64 | O8 |
| Memory | Block length | 96 steps (v6) | O8 |
| Loss | Monotonic hinge weight | 1/15 (0.0667) | O9 |
| Loss | Slope consistency weight | 0.05 | O9 |
| Loss | Pair lag | 1 (half the time), else uniform 2 to 200 | `max_lag` 200 |
| Optimizer | Adam learning rate | 1e-4 | OmniFuser |
| Optimizer | Batch | 32 windows (16 pairs) | OmniFuser |
| Optimizer | Gradient clip | 1.0 | OmniFuser |
| Training | Epochs x pairs per epoch | 30 x 2,048 | O10 |
| Training | Windows drawn per fit | 122,880 **†** (about 4.2 per training point) | 30 x 2,048 x 2 |
| Selection | Metric / rule | weighted whole-life MAE on validation tool; centre of lowest 3-epoch mean | plan section 4.3 |
| Ensemble | Seeds | 0, 1, 2; mean of predictions | D19 |
| Inference | Context time constant / cap margin | 120 s / 100 s | D23 |
| Cost | Fit time | median 447.9 s (range 440.2 to 477.4 **†**) on A100 | `run_summary.json` |

---

## 4. Baselines

All are computed at the same points (the 10 s grid; window RUL <= H) with each fold's
training tools (`PROJECT_PLAN.md` section 2.7 and 8).

| ID | Definition | Window MAE, mean over tools (s) | Pooled over 1,361 points (s) |
|---|---|---|---|
| B0 | Constant 750 s (H/2). It knows the point is in the window, so it is not deployable | 412.0 | 386.8 **†** |
| B1 | Mean training failure time minus elapsed time, capped | 512.5 | not reported |
| B2 | E[min(tau\* - tau, H) given tau\* > tau] over the training tools still alive at tau. The best usage-only rule | 467.8 | 443.9 |
| B6 | This model | 587.4 | 502.6 |

Per-tool window MAE of the baselines (`test_per_unit.csv`):

| Tool | B0 | B1 | B2 | B6 |
|---|---|---|---|---|
| T1 | 377.9 | 650.8 | 625.1 | 526.4 |
| T2 | 446.7 | 262.9 | 262.4 | 756.1 |
| T3 | 377.2 | 417.5 | 521.7 | 216.6 |
| T4 | 659.3 | 512.6 | 775.3 | 1403.6 |
| T5 | 376.7 | 752.1 | 747.5 | 726.3 |
| T6 | 375.8 | 555.8 | 302.0 | 352.4 |
| T7 | 378.0 | 590.8 | 327.5 | 431.1 |
| T8 | 376.0 | 510.0 | 228.8 | 469.3 |
| T9 | 376.5 | 126.0 | 256.8 | 324.9 |
| T10 | 375.4 | 746.9 | 631.2 | 667.2 |
| Mean | 412.0 | 512.5 | 467.8 | 587.4 |

Note on B0: on eight of ten tools B0 is between 375.4 and 378.0 s, because the window
points are spread evenly over RUL 0 to 1,500 s and a constant at the middle is wrong by
375 s on average. T4 (659.3, only 20 points, all with RUL under 184 s) and T2 (446.7)
raise its mean to 412.0. This is why B0 is hard to beat with a model that is wrong by
hundreds of seconds.

---

## 5. Results

### 5.0 Read of the test set before the freeze

> **The test was read on 2026-11-11, before the design freeze of 2026-12-01, with
> `--unseal`.** Job 2493825 (report, sealed) ran first and printed only validation
> results. The author then ran job 2494379 with `--unseal`, which wrote the test results
> (REPORT.md, `written` 2026-11-11T20:20:14+00:00). `REPORT.md` records this in its
> header.
>
> **What this implies (D15):** B6 is a stand-alone baseline. No RUL-Fuse design decision
> may use these test numbers before the freeze. In the report you should say that B6's
> test result was seen early, that it was not tuned afterwards, and that the setting was
> fixed before the read (O12). This brief found no sign that any B6 setting was changed
> after the read (`arm.json` holds the one setting; the logs show one training job). The
> early read would bias RUL-Fuse only if its design were influenced by these numbers,
> which D15 forbids.

All test tables below are the **current mode** (primary) unless stated. Numbers are the
mean over the 10 test tools, each tool scored with its 3-seed ensemble, with the 95 %
bootstrap interval of the mean over tools in brackets.

### 5.1 Test results

| Metric | B6 | B2 (usage rule) |
|---|---|---|
| Window MAE (s) | 587.4 +- 337.3 (411.7 to 801.7) | 467.8 +- 215.5 (346.9 to 597.4) |
| Window RMSE (s) | 656.3 +- 340.4 (467.4 to 868.3) | - |
| Bias (s; positive = too optimistic) | 108.9 +- 688.6 (-254.4 to 536.5) | - |
| Whole-life MAE of the capped target (s) | 325.1 +- 225.3 (212.3 to 468.8) | - |
| Late predictions (> 100 s too optimistic) | 0.413 +- 0.444 (0.162 to 0.685) | 0.612 +- 0.303 (0.432 to 0.780) |
| Early predictions (> 100 s too pessimistic) | 0.462 +- 0.443 (0.196 to 0.714) | - |
| Asymmetric score (plan section 10.2; lower is better) | 196.316 +- 425.850 (27.981 to 475.322) | 46.740 +- 79.215 (11.232 to 100.126) |
| Within +-20 % of the true RUL | 0.119 +- 0.146 (0.037 to 0.207) | - |
| Rising steps (share of 1 s steps where y_hat rises) | 0.352 +- 0.104 (0.295 to 0.416) | 0 |

- **Pooled over all 1,361 window points:** B6 MAE 502.6 s, B2 443.9 s.
- **Other baselines (mean window MAE):** B0 412.0 s, B1 512.5 s.
- **Paired against B2 over the 10 tools:** B6 changes the window MAE by **-119.6 s on
  average, meaning B6 is worse** (the report's "gain" is B2 minus B6; 95 % interval
  -281.6 to +33.1 s). B6 is better on 3 of 10 tools (T1, T3, T5). One-sided Wilcoxon
  p = 0.92.
- **Against B0 and B1 (computed †):** B6 beats B0 on 3 tools (T3, T6, T9) and B1 on 7
  tools (T1, T3, T5, T6, T7, T8, T10). On 39.2 % of the 1,361 window points B6's error is
  smaller than B0's.
- **Sensitivity to T4 †:** without T4, the mean window MAE is B6 496.7 s, B2 433.7 s,
  B0 384.5 s, B1 512.5 s. The median over the 10 tools is B6 497.9, B2 424.6, B0 377.0,
  B1 534.2 s. The asymmetric score is dominated by T4 (1,385.4): B6's mean without T4 is
  64.2, B2's 46.1.

### 5.2 Per test tool

| Tool | Window points | B6 MAE (s) | B2 MAE (s) | Gain over B2 (s) | Seed sd (s) | Bias (s) | Late | Early | Rising steps |
|---|---|---|---|---|---|---|---|---|---|
| T1 | 155 | 526.4 | 625.1 | 98.7 | 224.6 | 519.2 | 0.77 | 0.00 | 0.531 |
| T2 | 116 | 756.1 | 262.4 | -493.7 | 68.7 | -751.8 | 0.00 | 0.89 | 0.504 |
| T3 | 156 | 216.6 | 521.7 | 305.1 | 78.4 | 109.8 | 0.61 | 0.19 | 0.327 |
| T4 | 20 | 1403.6 | 775.3 | -628.3 | 10.0 | 1403.6 | 1.00 | 0.00 | 0.355 |
| T5 | 156 | 726.3 | 747.5 | 21.2 | 15.8 | 725.8 | 0.91 | 0.00 | 0.203 |
| T6 | 154 | 352.4 | 302.0 | -50.4 | 131.6 | -344.1 | 0.01 | 0.81 | 0.414 |
| T7 | 155 | 431.1 | 327.5 | -103.6 | 59.5 | -429.6 | 0.00 | 0.88 | 0.317 |
| T8 | 154 | 469.3 | 228.8 | -240.5 | 16.7 | -466.3 | 0.00 | 0.93 | 0.322 |
| T9 | 148 | 324.9 | 256.8 | -68.0 | 93.4 | -323.7 | 0.00 | 0.89 | 0.254 |
| T10 | 147 | 667.2 | 631.2 | -36.0 | 20.0 | 646.0 | 0.83 | 0.04 | 0.292 |

"Gain" = B2 MAE minus B6 MAE (positive: B6 better). "Seed sd" = standard deviation of the
window MAE of the three single-seed fits. "Early" column from `test_per_unit.csv`.

### 5.3 Error by true RUL (pooled window points)

| True RUL (s) | Points | B6 MAE | B6 bias | B6 context MAE | B6 context bias | B2 MAE | B2 bias |
|---|---|---|---|---|---|---|---|
| 0-300 | 304 | 561.3 | 455.2 | 551.7 | 415.8 | 647.3 | 617.7 |
| 300-600 | 250 | 522.6 | 193.5 | 518.9 | 140.9 | 504.2 | 399.8 |
| 600-900 | 258 | 503.8 | -14.5 | 502.0 | -41.1 | 399.0 | 184.7 |
| 900-1200 | 276 | 465.5 | -233.0 | 466.3 | -274.8 | 339.5 | -48.9 |
| 1200-1500 | 273 | 455.1 | -407.9 | 470.9 | -437.1 | 310.2 | -255.3 |

Source: `error_by_bin.csv`. B6 is better than B2 only in the 0-300 s bin.

### 5.4 Inference modes: current against context

Test (means over tools; last column = context minus current, with 95 % bootstrap interval):

| Metric | Current | Context | Context - current |
|---|---|---|---|
| Window MAE (s) | 587.4 | 587.8 | 0.4 (-20.2 to 18.7) |
| Window RMSE (s) | 656.3 | 654.1 | -2.2 (-22.5 to 15.5) |
| Bias (s) | 108.9 | 74.0 | -34.9 (-50.1 to -20.8) |
| Whole-life MAE (s) | 325.1 | 345.4 | 20.3 (7.8 to 35.6) |
| Late (> 100 s) | 0.413 | 0.399 | -0.014 (-0.028 to -0.002) |
| Early (> 100 s) | 0.462 | 0.494 | 0.033 (0.013 to 0.054) |
| Asymmetric score | 196.316 | 184.587 | -11.729 (-26.198 to -0.849) |
| Within +-20 % | 0.119 | 0.117 | -0.002 (-0.013 to 0.011) |
| Rising steps | 0.352 | 0.481 | 0.129 (-0.024 to 0.275) |

Validation tools (same layout):

| Metric | Current | Context | Context - current |
|---|---|---|---|
| Window MAE (s) | 540.1 | 531.5 | -8.6 (-35.5 to 12.6) |
| Window RMSE (s) | 614.8 | 604.2 | -10.6 (-41.1 to 14.2) |
| Bias (s) | 113.4 | 67.6 | -45.8 (-67.0 to -27.6) |
| Whole-life MAE (s) | 265.8 | 276.0 | 10.2 (-5.6 to 27.4) |
| Late (> 100 s) | 0.403 | 0.362 | -0.041 (-0.068 to -0.016) |
| Early (> 100 s) | 0.433 | 0.461 | 0.029 (0.006 to 0.052) |
| Asymmetric score | 191.698 | 176.551 | -15.147 (-33.139 to -1.797) |
| Within +-20 % | 0.228 | 0.207 | -0.021 (-0.060 to 0.011) |
| Rising steps | 0.345 | 0.482 | 0.137 (0.013 to 0.257) |

**Verdict under the pre-fixed rule: neither mode is clearly better, on test and on
validation.**

- Test: window MAE change +0.4 s (interval -20.2 to 18.7; the rule needs a gain of at
  least 10 s with the interval above 0). Context is better on 5 and worse on 5 of 10
  tools; two-sided Wilcoxon p = 0.846.
- Validation: change -8.6 s (interval -35.5 to 12.6; below the 10 s bar). Better on 5,
  worse on 5; p = 1.
- The context mode lowers the bias (-34.9 s on test) and the asymmetric score, raises the
  whole-life MAE, the early rate and the share of rising steps.

Per tool (test and validation), window MAE (s) in both modes (`mode_comparison.csv`):

| Tool | Test current | Test context | Test gain | Test bias, current | Test bias, context |
|---|---|---|---|---|---|
| T1 | 526.4 | 466.8 | 59.6 | 519.2 | 460.8 |
| T2 | 756.1 | 785.9 | -29.8 | -751.8 | -785.9 |
| T3 | 216.6 | 213.1 | 3.4 | 109.8 | 44.0 |
| T4 | 1403.6 | 1399.0 | 4.5 | 1403.6 | 1399.0 |
| T5 | 726.3 | 719.9 | 6.4 | 725.8 | 717.2 |
| T6 | 352.4 | 397.8 | -45.4 | -344.1 | -395.3 |
| T7 | 431.1 | 455.6 | -24.5 | -429.6 | -455.6 |
| T8 | 469.3 | 476.5 | -7.2 | -466.3 | -476.5 |
| T9 | 324.9 | 343.4 | -18.5 | -323.7 | -343.4 |
| T10 | 667.2 | 619.5 | 47.7 | 646.0 | 575.2 |

| Tool (validation role) | Fold's test tool | Val current | Val context | Val gain | Val bias, current |
|---|---|---|---|---|---|
| T1 | T10 | 498.6 | 413.8 | 84.8 | 485.8 |
| T2 | T1 | 646.2 | 656.7 | -10.5 | -562.4 |
| T3 | T2 | 276.6 | 302.1 | -25.6 | -198.0 |
| T4 | T3 | 1408.0 | 1407.8 | 0.1 | 1408.0 |
| T5 | T4 | 696.5 | 672.1 | 24.4 | 694.4 |
| T6 | T5 | 155.3 | 172.1 | -16.7 | -93.7 |
| T7 | T6 | 121.0 | 111.5 | 9.6 | -77.7 |
| T8 | T7 | 568.1 | 601.2 | -33.1 | -568.1 |
| T9 | T8 | 476.2 | 498.7 | -22.5 | -476.1 |
| T10 | T9 | 554.5 | 478.9 | 75.5 | 522.2 |

### 5.5 Validation summary

- Validation window MAE (current): **540.1 +- 363.4 (349.5 to 772.6) s**; B2 on the same
  validation tools 464.7 +- 211.9 (345.4 to 592.8) s. B6 is better than B2 on 6 of 10
  validation tools (T1, T3, T5, T6, T7, T10 as validation tools **†**).
- Validation B6 means: whole-life MAE 265.8 s, bias 113.4 s, late 0.403, early 0.433,
  rising steps 0.345, mean seed sd 53.4 s **†**.
- The validation tool selects the epoch, so its score is the optimistic one (section 6.1).
  On the same ten tools, B6's test mean (587.4) is 47.3 s above its validation mean
  (540.1) **†**; B2, which selects nothing, moves 3.1 s (464.7 to 467.8) **†**.

### 5.6 Run statistics

| Item | Value |
|---|---|
| Fits | 30 of 30 complete (10 folds x 3 seeds); 0 stale |
| Fit time | median 447.9 s; range 440.2 to 477.4 s **†**; total 13,551 s = 3.76 h of A100 time |
| Job wall time | 16:49:39 to 20:38:58 CET (job 2493824) |
| Selected epoch | median 14, range 2 to 29 |
| Fits whose best triple is at the end (still improving at epoch 30) | 4 (selected epoch 29 in folds 1, 4, 4, 9) |
| Training points per fit | 27,711 to 32,976 (median 29,123 **†**) |
| Trainable parameters | 173,269 |

---

## 6. Explanation of the results

**Headline.** B6 is worse than the usage rule B2 (587.4 against 467.8 s), worse than the
window-aware constant B0 (412.0 s), and only better than B1 (512.5 s) on 7 of 10 tools
while worse in the mean. Its errors are huge but mostly **one-sided per tool**: the tool is
predicted to last too long or too short for its whole window. The sections below separate
what the data shows from interpretation.

### 6.1 Over-fitting to the 8 training tools

**Data shows** (all from the 30 `fit.json` files; **†**):

| Quantity | Median over the 30 fits | Mean over the 30 fits |
|---|---|---|
| Training weighted MSE (y/H), epoch 1 | 0.0923 | |
| Training weighted MSE (y/H), epoch 30 | 0.00178 | |
| Ratio epoch 1 / epoch 30 | 51.9 (ratio of medians); 53.1 (median of per-fit ratios) | |
| Same MSE as RMS error in seconds | 455.8 s -> 63.3 s | |
| Training hinge term, epoch 1 -> 30 (in s) | 32.9 -> 14.3 | |
| Validation weighted whole-life MAE, epoch 1 | 402.1 s | 423.0 s |
| Validation MAE at the best epoch of each fit | 309.7 s | 318.0 s |
| Validation MAE, epoch 30 | 376.4 s | 376.8 s |
| Validation MAE at the selected epoch | 364.1 s | 365.7 s |

- **Training error falls about 52 times; validation error barely moves.** The median
  validation MAE goes from 402.1 s (epoch 1) to 376.4 s (epoch 30), 26 s or 6 %. Across the
  fits, the median validation curve stays between 358.9 and 408.9 s for all 30 epochs.
  24 of 30 fits end below their epoch-1 value, and 6 end above it.
- **Validation is noisy.** The per-fit standard deviation of `validation_mae_s` over its 30
  epochs has median 37.4 s (range 15.3 to 56.3 s). The median absolute change between
  consecutive epochs is 22.1 s. The noise is larger than the 26 s of improvement.
- **Many fits peak at once.** In 6 of 30 fits the lowest validation MAE is at epoch 1 or 2
  (5 at epoch 1); in 8 of 30, within the first 3. In 27 of 30 the last epoch is above the
  fit's best epoch, by a median of 32.4 s. The median gap of epoch 1 above the best epoch
  is 81.4 s.
- **Selected epochs split in two groups.** 11 fits select epoch 7 or earlier, 11 select
  epoch 25 or later, and 8 are between. This fits a flat noisy curve in which the minimum
  falls where the noise puts it.
- **Selection optimism inside a fit.** The selected epoch's validation value is on average
  27.0 s (median 19.0 s) below the same fit's mean over its 30 epochs; it is below the
  fit's own median in 25 of 30 fits. The best epoch of a fit is 74.7 s (mean over fits)
  below the mean of its 30 epochs.
- **Validation against test on the same ten tools.** Validation window MAE 540.1 s,
  test 587.4 s (difference 47.3 s). Per tool, the difference ranges from -151 to +310 s
  (mean absolute 110.2 s; standard deviation of the difference 139.6 s; a bootstrap
  interval of the mean difference over tools is -30.6 to 132.0 s). So the direction is
  consistent with selection optimism, but with 10 tools it is **not statistically
  established**. A tool is scored in two different folds (different training tools), and
  the two scores can be far apart: T6 155 against 352 s, T7 121 against 431 s, T9 476
  against 325 s.
- **Per-fold detail** (medians over the 3 seeds; weighted whole-life validation MAE in s):

| Fold | Test / validation tool | Epoch 1 | Best epoch | Last epoch | At selected epoch | Selected epochs (seeds 0,1,2) | Train MSE epoch 1 -> 30 |
|---|---|---|---|---|---|---|---|
| 1 | T1 / T2 | 466 | 466 | 672 | 611 | 2, 29, 2 | 0.087 -> 0.0017 |
| 2 | T2 / T3 | 285 | 220 | 250 | 255 | 27, 9, 5 | 0.093 -> 0.0019 |
| 3 | T3 / T4 | 436 | 367 | 377 | 373 | 21, 13, 28 | 0.079 -> 0.0018 |
| 4 | T4 / T5 | 697 | 618 | 618 | 654 | 6, 29, 29 | 0.075 -> 0.0017 |
| 5 | T5 / T6 | 277 | 91 | 200 | 99 | 9, 19, 6 | 0.090 -> 0.0019 |
| 6 | T6 / T7 | 292 | 75 | 91 | 90 | 27, 13, 15 | 0.102 -> 0.0019 |
| 7 | T7 / T8 | 408 | 327 | 396 | 404 | 25, 25, 2 | 0.095 -> 0.0017 |
| 8 | T8 / T9 | 331 | 277 | 368 | 360 | 4, 25, 27 | 0.096 -> 0.0022 |
| 9 | T9 / T10 | 363 | 287 | 302 | 338 | 7, 24, 29 | 0.090 -> 0.0018 |
| 10 | T10 / T1 | 608 | 366 | 489 | 391 | 5, 7, 3 | 0.100 -> 0.0018 |

(The fold-1 median "selected" value is high because the validation tool there, T2, is
badly predicted: B6's validation window MAE on T2 is 646.2 s.)

**Interpretation.** The network (173,269 trainable parameters, plus a GRU that sees the
whole history) learns the 8 training trajectories almost exactly. The training weighted
RMS error falls from about 456 s to about 63 s. What it learns does not carry to a ninth
tool. About 29,000 training points per fit come from 8 trajectories, so they are strongly
correlated. Each fit draws about 4.2 windows per training point, which is enough to
memorize them. The conclusion is the one in `PROJECT_LOG.md`: expected for an end-to-end
OmniFuser with ten tools. This is a statement about this run, not a test of regularization;
B6 was deliberately not tuned (O12), so no dropout, augmentation or early-stopping variant
was tried, and the results do not show that a tuned OmniFuser could not do better.

### 6.2 Failure patterns per tool

**Data shows.** The figure `rul_trajectories.png` (section 8) and `test_per_unit.csv`.
Facts from `test_trajectories.csv` **†** unless stated:

| Tool | Pattern in the figure | B6 mean prediction when true RUL is 1,200-1,500 s | B6 mean prediction when true RUL <= 300 s (true mean about 149 s) | Lowest B6 prediction in the window |
|---|---|---|---|---|
| T1 | Falls to about 830 s at the start, recovers to about 1,450 s, then falls only to about 1,150-1,250 s | 1,361 | 1,241 | 1,146 |
| T2 | Below 1,200 s from minute 2.5 (first dip to about 900 s), about 300 s at minute 20, 300-600 s until about minute 35, near 0 from about minute 50; true RUL stays at 1,500 s until minute 45 | 77 | 27 | 5 |
| T3 | Follows the decline; crosses 1,200 s about 10.5 min before the truth | 1,101 | 487 | 391 |
| T4 | Stays at the cap through its whole life | n/a (T4 has no window point above 183.7 s) | 1,494 | 1,484 |
| T5 | Stays near the cap until minute 25, falls to about 1,330 s at the end; true RUL starts falling at minute 5.5 | 1,484 | 1,400 | 1,327 |
| T6 | Declines gradually from minute 5 and sits near 1,000 s from minute 20 to 40, while truth is at the cap until about minute 50 | 721 | 129 | 91 |
| T7 | Follows the decline; crosses 1,200 s about 19 minutes early | 729 | 61 | 39 |
| T8 | Follows the decline; crosses 1,200 s about 8.5 minutes early | 705 | 13 | 0 |
| T9 | Follows the decline; crosses 1,200 s about 9.5 minutes early | 983 | 44 | 21 |
| T10 | Stays near the cap (1,450 s) with dips to about 1,200 s in the last 25 minutes; true RUL falls from minute 65 | 1,348 | 1,460 | 1,209 |

"Minutes early" here = the first operating minute (after the first 2 minutes) at which
B6's mean prediction goes below 1,200 s, compared with the minute at which the true RUL
does (T3 27.2 against 37.7; T6 8.0 against 54.5; T7 36.9 against 55.7; T8 46.1 against
54.6; T9 34.5 against 44.0) **†**.

Five groups, with the numbers from `test_per_unit.csv`:

1. **Early (pessimistic) decline: T3, T6, T7, T8, T9.** The prediction tracks the shape of
   the decline (correlation with the true RUL inside the window: T3 0.96, T6 0.98, T7 0.95,
   T8 0.94, T9 0.96 **†**) but starts it too early. B6's bias is negative on T6 to T9
   (-344.1, -429.6, -466.3, -323.7 s) with early rates 0.81 to 0.93. T3 is the mixed
   case: early near the start of its window, late near its end (B6 predicts 487 s when
   the truth is about 149 s), net bias +109.8 s, the lowest MAE (216.6 s).
2. **Cap saturation: T4 and T5.** The prediction stays at or near H whatever the tool does:
   T4 MAE 1403.6 s (late rate 1.00; its predictions in the window span 1,484 to 1,500 s
   **†**), T5 726.3 s (late 0.91; predictions 1,327 to 1,500 s). T5 is also the
   shortest-lived tool (1,827 s).
3. **Partial drop: T1 and T10.** The prediction falls only part way (T1 to about 1,150 to
   1,250 s, T10 to about 1,200 s) while the truth falls to 0. Bias +519.2 and +646.0 s;
   late rates 0.77 and 0.83. T1 also has the largest seed spread (224.6 s).
4. **Far too pessimistic: T2.** The prediction is near 0 for most of the tool's life.
   Bias -751.8 s, early rate 0.89, MAE 756.1 s. B2 scores 262.4 s on T2, so T2 costs B6
   -493.7 s against B2.
5. **The bias splits the tools into two camps.** Five tools are strongly late (T1, T3, T4,
   T5, T10; mean bias **+680.9 s** **†**) and five strongly early (T2, T6, T7, T8, T9;
   mean bias **-463.1 s** **†**). The mean over the tools, +108.9 s, is therefore
   meaningless as a summary: its standard deviation over tools is 688.6 s, the mean of
   the absolute bias is **572.0 s** **†**, and the 95 % interval of the mean bias
   (-254.4 to 536.5) contains 0.
   Per tool, MAE minus |bias| is below 22 s on nine tools (T3: 106.7 s) **†**. So nearly
   all of the error is a signed offset that lasts across the tool's window. The pooled
   bias by true-RUL bin runs from +455.2 s (RUL 0-300) to -407.9 s (RUL 1,200-1,500).

**Interpretation.** The per-tool offsets are what you would expect when a model has
learned each training tool's own "level" and has no information that places a new tool on
the common scale. Tools T3 and T6-T9 have a within-window correlation near 0.95 with the
truth, so the model does read a monotone wear-like signal there, and fails mostly on
when the decline starts. T4 and T5 are special cases: T4's last 14 of 16 passes before
failure were never recorded, and T5 fails at 1,827 s, so the elapsed-time input and the
signals look like early life for every other tool. These readings are plausible but
untested. Other possible contributors that this repository does not test: T1 and T2 have
photos of twice the resolution of the others; T1 cuts at a force level above the
training tools and T5 at a more spread level (OmniFuser README, forecasting analysis);
T9 and T10 have different pass lengths (about 81 s and 122 s against a median 97 s,
`PROJECT_PLAN.md` section 2.1). None of these was isolated by an experiment.

### 6.3 Why B6 loses even to the constant B0

**Data shows.**

- The evaluation window contains only the last 1,500 s of life. Inside it the true RUL
  is spread evenly between 0 and 1,500 s (pooled mean 742.5, sd 444.2 s **†**). A constant
  at 750 s is wrong by about 375 s on average on every tool (B0 per tool 375.4 to 378.0 s on
  eight tools; section 4).
- B6's window MAE is below that on only three tools (T3 216.6, T6 352.4, T9 324.9 s) and
  far above on T4 (+744 s over B0), T5 (+350 s), T2 (+309 s) and T10 (+292 s) **†**.
- Pooled over the window, the correlation of B6's prediction with the true RUL is 0.23 and
  the fitted slope of prediction on truth is 0.28 **†**. For B2 these are 0.29 and 0.27
  **†**. So pooled over tools, B6 carries no more information about the position within the
  window than the usage rule does.
- B6 is accurate where it matters least for the headline: on the plateau (RUL above H,
  not scored in the window) its pooled MAE is 203.7 s against B2's 129.0 s **†**; the
  whole-life MAE of the capped target is 325.1 s, lower than the window MAE of 587.4 s.

**Interpretation.**

- The window metric asks the model to be right on the decline, where the true RUL moves
  from 1,500 to 0. A model that mostly stays on a plateau (T4, T5, T10) or begins the
  decline too early (T2, T6-T9) misses the whole window by the size of its timing error.
  The constant B0 never mislocates the decline, because it makes no claim about it;
  its error is a fixed 375 s.
- B6's error is mostly a signed level offset, not noise (6.2). B6 has to get the level of
  an unseen tool right to within a few hundred seconds, and with 8 training tools it
  does not.
- Because errors have opposite signs on different tools, B6's pooled slope on the truth is
  small (0.28) even though its within-tool correlation is high on five tools.
- B1 and B2 use only elapsed time and fall steadily with it. B6 has the same usage input
  and also sees signals and photos. In this run the extra inputs add per-tool offsets and
  no pooled information (an interpretation of the pooled slope and correlation above).

### 6.4 Rising steps despite the hinge, and why context cannot fix level errors

**Data shows (rising steps).**

- 35.2 % of 1 s steps rise on average (0.203 to 0.531 per tool), against 0 for B2, even
  though the loss has a hinge on rising pairs.
- The mean rise is small: the mean of max(delta y_hat, 0) over all steps is 0.32 s per
  step, so a rising step rises by **0.90 s on average** **†** (per tool 0.52 to 1.39 s).
  The slope error |delta y_hat + delta tau| is 1.44 s per step on average; a perfectly flat
  prediction would score 1.0 s at 1 s steps. T4, whose prediction is flat at the cap, scores 0.985.
- At the end of training the hinge term is 0.0095 in y/H, about 14.3 s per pair on
  average **†**, down from 32.9 s at epoch 1.
- The count treats any rise above 1e-6 s as a rise.

**Interpretation (rising steps).** The hinge weight 1/15 acts as a tolerance: an earlier
prediction is moved only by a small push of at most about 1/15 x H (100 s) against a
squared-error pull (`losses.py` and plan v5), and the later prediction is the fixed
reference, so the penalty can lift the earlier prediction but never lowers the later one.
It reduces large reversals but does not forbid second-to-second jitter in a model with
noise from 1 s windows. The 35 % are mostly small movements of about 1 s, which is
negligible against the 587 s of error. Reducing them was never the goal of the loss; the
check of v5 on the toy dataset found a stronger weight removes more rises but raises the
bias (hinge weight 1.0: mean rise 0.62 s, window MAE 62.4 s; weight 1/15: 2.11 s, 34.6 s).

**Data shows (context mode).**

- The context mode changes the window MAE by +0.4 s on test and -8.6 s on validation, both
  inside the "neither" band.
- It lowers the mean bias by 34.9 s on test (45.8 s on validation) **and shifts the bias
  of every one of the 10 tools downwards** (10 of 10 on test and on validation **†**).
- On the test tools, the sign of the gain equals the sign of the bias on all 10. The five
  tools it helps are exactly the five late-biased tools (T1, T3, T4, T5, T10); the five it
  hurts are the five early-biased tools (T2, T6, T7, T8, T9) **†**. On validation this
  holds for 9 of 10 tools (T7, bias -77.7 s, gains 9.6 s) **†**.
- Its whole-life MAE rises by 20.3 s, its early rate by 0.033, and its share of rising
  steps from 0.352 to 0.481.
- The tool with the largest bias, T4 (+1,403.6 s), changes by 4.5 s (1,399.0 s).

**Interpretation (context mode).**

- A context filter averages a tool's own past predictions. If the model is wrong by the
  same amount for the whole tool, so is the average. The design notes say this in advance
  ("no bias correction", synthetic constant bias 111.8 to 113.8 s). The data agree: T4
  barely changes, and every tool's bias moves down, by between 4.5 s (T4) and 70.8 s (T10) **†**.
- The downward shift matches the documented side effect: once the model's prediction is
  more than 100 s below H, the filter ages its estimate by 1 s per s, so where the model's
  predictions fall more slowly than that, the context settles below them. That helps the
  late tools and hurts the early ones. This mechanism is taken from the design notes and
  is consistent with the per-tool pattern, but the mechanism itself was not isolated.
- The higher share of rising steps in context mode is also consistent with the filter's
  design: its estimate drops 1 s per step by construction, and any new prediction more than
  about 120 s above that drops it back up. Neither claim was measured step by step here
  (the trajectory file has only 10 s points).
- In short: the context mode averages noise away and cannot supply the missing level. The
  noise it removes is a small part of the error.

### 6.5 Seed spread

**Data shows** (`test_per_unit.csv`; seed sd = sd of the 3 single-seed window MAEs):

| Tool | Seed 0 | Seed 1 | Seed 2 | Max minus min | Seed sd | Ensemble MAE |
|---|---|---|---|---|---|---|
| T1 | 299.2 | 740.7 | 591.5 | 441 | 224.6 | 526.4 |
| T2 | 792.8 | 803.1 | 679.3 | 124 | 68.7 | 756.1 |
| T3 | 250 | 276 | 129 | 147 | 78.4 | 216.6 |
| T4 | 1409 | 1409 | 1392 | 17 | 10.0 | 1403.6 |
| T5 | 732 | 709 | 739 | 30 | 15.8 | 726.3 |
| T6 | 510 | 252 | 333 | 257 | 131.6 | 352.4 |
| T7 | 403 | 500 | 392 | 108 | 59.5 | 431.1 |
| T8 | 471 | 453 | 487 | 33 | 16.7 | 469.3 |
| T9 | 292 | 431 | 254 | 178 | 93.4 | 324.9 |
| T10 | 664 | 652 | 691 | 39 | 20.0 | 667.2 |

(Single-seed values for T3 to T10 are rounded to whole seconds.)

- The mean seed sd over tools is 71.9 s (median 64.1 s). It is small on T4, T5, T8, T10
  (10.0 to 20.0 s) and large on T1, T6, T9 (224.6, 131.6, 93.4 s). T1's three fits give
  299, 741 and 591 s.
- The three seeds' tool-averaged window MAE are 582.3, 622.6 and 568.7 s **†** (sd 28.0 s
  over seeds). The 3-seed ensemble (587.4 s) is lower than the mean single-seed MAE
  (591.2 s) **†** on every tool but by only 3.8 s on average.
- Seed spread relative to the MAE is small on the biased tools (T4 0.01, T5 0.02, T10 0.03,
  T8 0.04) and large on T1 (0.43), T6 (0.37), T3 (0.36), T9 (0.29) **†**.

**Interpretation.** Where the error is a large level offset (T4, T5, T8, T10), all three
seeds make the same mistake, so the ensemble cannot fix it. The seeds differ mainly on
tools whose prediction is moving (T1, T3, T6, T9). The error is mostly a systematic property of
the 8-tool training set, not random variation between initialisations. Averaging more
seeds would not move the headline number much.

---

## 7. Threats to validity and limitations

1. **Ten tools.** Everything is a mean over 10 tools; per-tool values range from 216.6 to
   1,403.6 s. The bootstrap interval of the mean window MAE is wide (411.7 to 801.7), and
   the comparison with B2 is not significant (interval -281.6 to +33.1 s; Wilcoxon one-sided
   p = 0.92). The data support "B6 is not better than B2". "B6 is worse than B2" is
   supported only as a point estimate.
2. **One validation tool per fold.** The epoch is chosen on a single tool. With a noisy
   curve (median per-fit sd 37.4 s) this is a weak selector, which the selected epochs
   (spread over 2 to 29) show. The validation score (540.1 s) is optimistic by an amount
   this study cannot pin down (47.3 s, interval -30.6 to 132.0 s **†**).
3. **T4 has 20 window points** (RUL under 184 s). It weighs little in a pooled score but a
   tenth of the tool mean. Its B6 MAE of 1,403.6 s raises B6's tool mean from 496.7 s (the
   other nine tools) to 587.4 s, by 90.7 s **†**. It also
   dominates the asymmetric score (1,385.4 of a mean of 196.3). Report results with and
   without T4 (section 5.1).
4. **Test read early.** The test was unsealed on 2026-11-11, 20 days before the freeze.
   B6 was not changed afterwards, but RUL-Fuse's design must not use these numbers (D15).
5. **Single fixed setting, not tuned.** By design (O12) there is no search over
   regularization, memory, loss weights, learning rate or number of epochs. The results
   characterise this setting. They do not say what a tuned OmniFuser would reach.
6. **Ablations not run.** The effect of the GRU memory, of the temporal loss, of usage
   inputs, and of each modality are not measured on B6. Statements about their role are
   interpretation.
7. **The photo type was chosen after a leakage probe.** Tool photos were readmitted by the
   author's decision (D24) although one statistic crossed the limit. The decision is
   recorded and was made before any RUL result. Photo-derived stage leakage is reduced,
   not shown to be absent.
8. **Cross-fold dependence.** Fold i's validation tool is fold i+1's test tool. Every tool
   is both, in different folds with different training sets. Per-tool scores for the same
   tool differ by 110 s on average between its validation and test role. Ten per-tool
   scores are not independent samples of "a new tool".
9. **Gap-aware time axis (D12) and force-missing passes (D8)** are project choices shared
   by every model. T4's window is mostly unrecorded cutting.
10. **Baseline asymmetry.** B0 is not deployable. B1 and B2 use only elapsed time.
    B6 uses force, tool photos and elapsed time.
11. **Same data, same metrics as RUL-Fuse, a different architecture:** conclusions about
    "fusion" apply to this architecture on this dataset only.

---

## 8. Suggested report structure and file list

### 8.1 Outline

1. **Introduction.** Why a published-architecture baseline (D20, O1); what B6 is not
   (section 1 of this brief).
2. **Task and data.** Nonastreda, 10 tools, target RUL capped at H = 1,500 s, TTF rule,
   gap-aware axis, window of 1,361 points, folds. **Table 1:** per-tool failure time and
   window points (section 2.3). **Table 2:** the D24 leakage counts (section 2.2).
3. **Method.** OmniFuser adapted, the GRU memory over 96-step block encodings, the loss,
   training and selection, the two inference modes and the pre-fixed comparison rule.
   **Table 3:** hyperparameters (section 3.6). OmniFuser's README has a flow chart; draw one for B6 with force and photo branches, C2F fusion, the GRU over 96-step blocks, and
   the RUL head.
4. **Baselines.** B0, B1, B2 (section 4), **Table 4** with per-tool baselines.
5. **Results.**
   - **Table 5:** test results, mean +- sd and bootstrap interval, B6 against B2 (5.1).
   - **Table 6:** per-tool test table (5.2).
   - **Figure 1:** `window_mae_per_tool.png` next to Table 6.
   - **Figure 2:** `rul_trajectories.png`, the main figure; place it before the analysis.
   - **Table 7 and Figure 3:** error by true RUL, `error_by_rul.png` (5.3).
   - **Table 8:** current against context, test and validation (5.4).
   - **Table 9:** validation summary (5.5); **Table 10:** run statistics (5.6).
   - A boxed note: the test was read early (5.0).
6. **Analysis.** Section 6.1 (over-fitting, with the per-fold table, and a plot of the median
   validation curve and training MSE over epochs if you want a figure: it must be drawn
   from the `fit.json` files, no such figure exists), 6.2 (failure patterns on Figure 2),
   6.3 (why B0 wins), 6.4 (rising steps and context), 6.5 (seeds).
7. **Limitations** (section 7).
8. **Conclusion.** B6 is a stand-alone baseline; it does not reach the usage rule; the
   dominant failure is a per-tool level offset consistent with over-fitting to 8
   training tools.

### 8.2 Files and captions

All paths are under
`D:\RULModel\runs\omnifuser_rul\nonastreda-6e0cb095967b\force+tool\report\`
(call it `REPORT_DIR`), unless noted.

| File | Content | Suggested caption |
|---|---|---|
| `REPORT_DIR\figures\rul_trajectories.png` | 2 x 5 panels, one per test tool T1-T10. Black: true RUL capped at H. Orange: B2. Blue: B6 current mode (mean of seeds), with a light-blue band for the range of the three seeds. Green: B6 context mode. x: operating time in minutes. y: RUL in seconds (0 to 1,500) | "True and predicted RUL of each test tool over its life. B6 (blue, mean of 3 seeds; band: seed range; green: context mode) against the usage rule B2 (orange)." |
| `REPORT_DIR\figures\window_mae_per_tool.png` | Dot plot, one row per tool, window MAE in s (x axis 0 to 1,400+). Orange B2, blue B6 current, green B6 context, joined by a grey line. B6 is to the right of B2 on 7 of 10 tools. T4 is far right (1,403.6 s) | "Window MAE of each test tool: B6 in both inference modes and B2." |
| `REPORT_DIR\figures\error_by_rul.png` | Grouped bars for five true-RUL bins (0-300, 300-600, 600-900, 900-1,200, 1,200-1,500 s). Blue B6 current, green B6 context, orange B2. B6 bars are flat (561 down to 455 s); B2 falls from 647 to 310 s; B2 is lower in the four right bins | "MAE by true-RUL range, pooled over the 10 test tools (1,361 points): B6 in both modes and B2." |
| `REPORT_DIR\REPORT.md` | The generated report: all tables of section 5 | (source of Tables 5, 6, 7, 8, 9) |
| `REPORT_DIR\run_summary.json` | Settings, progress, per-unit metrics (validation and test), selected epochs, fit seconds, inference modes | (source of settings and of 5.6) |
| `REPORT_DIR\test_per_unit.csv` | Per tool and mode: window MAE, RMSE, bias, whole-life MAE, late and early rates, asymmetric score, within +-20 %, rising-step share, mean rise, slope error, seed sd and per-seed MAEs, B0, B1 and B2 values | Table 6 and Table 4 |
| `REPORT_DIR\validation_per_unit.csv` | Same, for the validation tools | Section 5.4, 5.5 |
| `REPORT_DIR\mode_comparison.csv` | Per tool and role: current and context window MAE, gain, bias, rising share | Table 8 per-tool parts |
| `REPORT_DIR\error_by_bin.csv` | B6, B6 context and B2 by true-RUL bin: points, MAE, bias | Table 7 |
| `REPORT_DIR\curves.csv` | Per-epoch validation weighted MAE of each fit (`arm, fold, seed, epoch, validation_weighted_mae_s, selected`) | Source for a validation-curve figure |
| `REPORT_DIR\test_trajectories.csv` | Per test point (10 s grid): `unit, tau_s, y_s, b6_s, b6_context_s, seed_min_s, seed_max_s, b2_s` (3,747 rows) | Source for custom trajectory figures |
| `D:\RULModel\runs\omnifuser_rul\nonastreda-6e0cb095967b\force+tool\fold-NN\seed-S\fit.json` | Per fit: `validation_mae_s` and `train_loss` (mse, hinge, slope) per epoch, `selected_epoch`, `seconds`, parameters, training points | Section 6.1 numbers |
| `...\force+tool\fold-NN\seed-S\predictions.parquet` | Every 1 s prediction of the validation and test tools | |
| `...\force+tool\arm.json` | The run's settings and keys (`store 6e0cb095967b`, `key 57532f80fb96`) | Table 3 |
| `D:\RULModel\runs\omnifuser_rul\logs\b6-train-2493824.out` | Training log, one line per epoch per fit | |
| `D:\RULModel\runs\omnifuser_rul\logs\b6-report-2493825.out`, `...-2494379.out` | Report job logs (sealed; unsealed) | |

---

## 9. Reproduction

**Code:** commit `71da7b1` (full hash 71da7b13d68162fb50d9410986187fea45d15986; message
"Added only toll images to OmniFuser Study"), as printed in the job logs.

**Stores and keys:**

| Item | Value |
|---|---|
| B6's 1 s store | `nonastreda/6e0cb095967b` (36,919 prediction points; 37,105 distinct segments cached for descriptors) |
| The project's 10 s store | `nonastreda/3e357b5a23ff` |
| Stage-3 feature-check report | `nonastreda-3e357b5a23ff.json` |
| Arm settings key | `57532f80fb96`; arm folder `force+tool` |
| Photo-feature cache | 1,536 EfficientNet-B0 features (all three photo kinds; B6 reads the tool ones) |

**Running** (on Ruche; the laptop must not run weights or heavy jobs; the author submits
the jobs, `RUCHE_RUNBOOK.md` section 12):

    source slurm/env.sh
    bash slurm/omnifuser_rul.sh force tool

This submits `b6-train` (16 h limit; runs
`python -m rulfuse.cli omnifuser-arm --players force tool --feature-device cuda`) and,
chained after it, `b6-report`. A guard refuses photo types not admitted by the feature
check and `configs/feature_check_overrides.json`. A rerun skips finished fits.

**Report** (CPU):

    python -m rulfuse.cli omnifuser-report --players force tool            # test sealed until 2026-12-01
    python -m rulfuse.cli omnifuser-report --players force tool --unseal   # reads the test; REPORT.md says so

On Ruche the same is `bash slurm/omnifuser_results.sh [--unseal] force tool`, which also
builds the archive and its sha256. Use the repository's own Python (`D:\RULModel\.venv`)
on the laptop. The report reads
`$RULFUSE_DATA_ROOT\runs\omnifuser_rul\<store>\<arm>` (default `D:\RULModel`).

**Jobs** (from `D:\RULModel\runs\omnifuser_rul\logs\`):

| Job | Name | Where | Time (CET, 2026-11-11) | Outcome |
|---|---|---|---|---|
| 2493824 | `b6-train` | `ruche-gpu17.cluster`, partition `gpua100`, NVIDIA A100-SXM4-40GB | 16:49:39 to 20:38:58 | 30 of 30 fits |
| 2493825 | `b6-report` | `node054`, partition `cpu_short` | 20:39:08 to 20:40:22 | validation only; test sealed; archive `b6-results-20261111-2040.tar.gz` |
| 2494379 | `b6-report` (`--unseal`) | `node030`, partition `cpu_short` | started 21:18:59 | test and validation; archive `b6-results-20261111-2120.tar.gz` (per `PROJECT_LOG.md`; the copy of this job's log holds no archive line) |

**Archives** (in `D:\RULModel\transfer\`, copied with `scp`; the second is unpacked into
`D:\RULModel\runs\omnifuser_rul\`):

| Archive | Size (bytes) | sha256 |
|---|---|---|
| `b6-results-20261111-2040.tar.gz` | 1,292,284 | `40f51b2c91e3d7fa84bbbe4159dec2f1648cc12d4be58bd001d68da990c79005` |
| `b6-results-20261111-2120.tar.gz` | 1,783,471 | `a0c57b0dd84f27b82850f8e8866298e9a59ebfec9101c1e59809a1f03e6aade0` |

**The hashes matched.** `PROJECT_LOG.md` records that both hashes match; they were
recomputed with `sha256sum` for this brief and equal the values in the `.sha256` files
**†**.

---

## Appendix A. Numbers computed for this brief (†) and how

All from the files in section 8.2, using the repository's Python.

| Number | Method |
|---|---|
| Training MSE epoch 1 0.0923, epoch 30 0.00178; ratio 51.9 | Median over the 30 `fit.json` of `train_loss[0].mse` and `train_loss[29].mse`. Ratio of the two medians; the median of the per-fit ratios is 53.1 |
| RMS 455.8 s -> 63.3 s | sqrt(median MSE) x 1,500 |
| Hinge 32.9 -> 14.3 s | Median `train_loss.hinge` (0.0219, 0.0095) x 1,500 |
| Validation MAE epoch 1 402.1, best 309.7, last 376.4, selected 364.1 (medians); 423.0, 318.0, 376.8, 365.7 (means) | Over the 30 fits: `validation_mae_s[0]`, its minimum, `[29]`, and the value at `selected_epoch` - 1 |
| 6 fits peak at epoch 1 or 2; 8 within 3 | argmin of each `validation_mae_s` (epoch 1: 5 fits; epoch 2: 1; epoch 3: 2) |
| Per-fit sd, median 37.4 s (range 15.3 to 56.3) | `numpy.std(validation_mae_s, ddof=1)` per fit |
| Median abs epoch-to-epoch change 22.1 s | Median of `abs(diff(validation_mae_s))` over all fits and epochs |
| Selected epoch 27.0 s (median 19.0 s) below the fit's mean over epochs | value at the selected epoch minus the mean (median) of the 30 values of the same fit; below the median in 25 of 30 fits |
| 27 of 30 fits end above their best; median 32.4 s | last epoch minus minimum, per fit |
| Median curve range 358.9 to 408.9 | Median over fits of `validation_mae_s` at each epoch; min and max over epochs |
| Validation to test difference 47.3 s; per-tool sd 139.6; mean abs 110.2; interval -30.6 to 132.0 | Per-tool difference of `window_mae` (current) between `test_per_unit.csv` and `validation_per_unit.csv`; bootstrap of the mean over tools, 20,000 resamples, seed 0 |
| B2 validation to test 3.1 s | 467.8 minus 464.7 (mean `B2_window_mae`) |
| Late group mean bias +680.9, early group -463.1; mean abs bias 572.0 | Mean of `window_bias` over T1, T3, T4, T5, T10, over T2, T6-T9, and mean of its absolute value |
| MAE minus abs bias below 22 s on nine tools | `window_mae` - abs(`window_bias`) per tool (7.2, 4.3, 106.7 for T3, 0.0, 0.5, 8.3, 1.5, 3.0, 1.2, 21.2) |
| Pooled slope 0.28, correlation 0.23 (B2: 0.27, 0.29) | Least squares of the prediction on the true RUL over the 1,361 window points of `test_trajectories.csv` (window = `y_s` < 1,500) |
| B0 pooled 386.8 s; B6 beats B0 on 39.2 % of window points | mean of abs(`y_s` - 750), and the share of points with abs(`b6_s` - `y_s`) < abs(750 - `y_s`) |
| Plateau MAE B6 203.7, B2 129.0 s | Over the 2,386 test points with true RUL above 1,500 s |
| Without T4: B6 496.7, B2 433.7, B0 384.5, B1 512.5; medians 497.9, 424.6, 377.0, 534.2 | Mean and median of the per-tool window MAE columns |
| T4 contribution to the B6 mean: 1,403.6 / 10 = 140.4 s of 587.4 s; the mean of the other nine is 496.7 s, so T4 raises the mean by 90.7 s | Arithmetic from the same two numbers |
| Asymmetric score without T4: B6 64.2, B2 46.1 | Mean of `phm_score` and `B2_phm_score` over the other nine tools |
| Tool-level lead times (first minute below 1,200 s) | First `tau_s` > 120 s with `b6_s` < 1,200, against (failure time - 1,200 s), in `test_trajectories.csv` |
| Within-window correlation per tool (T3 0.96, T6 0.98, T7 0.95, T8 0.94, T9 0.96; T1 0.61, T2 0.75, T4 0.51, T5 0.61, T10 -0.31) | Pearson r of `b6_s` and `y_s` over each tool's window points |
| Rise per rising step 0.90 s | `mean_increase_s` / `increase_rate`, mean over tools of the ratio is 0.92; ratio of the means 0.896 |
| Context shifts all 10 biases downward; gain sign equals bias sign on 10 of 10 tools (test), 9 of 10 (validation) | `mode_comparison.csv`: `context_window_bias` < `current_window_bias`; `gain_s` > 0 against `current_window_bias` > 0 |
| Seeds: tool-mean per seed 582.3, 622.6, 568.7 (sd 28.0); mean seed MAE 591.2 | From `seed0/1/2_window_mae` in `test_per_unit.csv` |
| Fit time range 440.2 to 477.4 s; median training points 29,123; windows per fit 122,880 | `fit.json` `seconds`, `train_points`; 30 x 2,048 x 2 |
| sha256 match | `sha256sum` of the two archives against the `.sha256` files |

The 10 most important computed numbers are listed in the hand-off message that came with
this file.
