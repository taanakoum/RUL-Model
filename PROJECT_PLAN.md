# RUL-Fuse: multimodal remaining-useful-life prediction on Nonastreda

**Project plan, version 2.** 2026-11-04. Target completion 2026-12-16.

Version 2 follows a full review of version 1. [PLAN_REVIEW.md](PLAN_REVIEW.md)
lists every finding, the change it caused and why. Version 1 is kept in
[docs/archive/PROJECT_PLAN_v1.md](docs/archive/PROJECT_PLAN_v1.md).

Status: **Stages 0-2 done** (2026-11-04; see [PROJECT_LOG.md](PROJECT_LOG.md),
[docs/EDA_RUL.md](docs/EDA_RUL.md) and [docs/END_TO_END.md](docs/END_TO_END.md)). Every number below was measured
on `Data/` (identical to OmniFuser's `Nonastreda_Data/`), on OmniFuser's
`experiment_results/eda/records.csv`, and on the chip and workpiece photos in
`D:\Project14\data\raw\nonastreda`. Appendix B lists how.

> **Resolved 2026-11-04.** You confirmed the gap-aware time axis (D12): T2's
> run 10 and T4's runs 6-9 were cut but not recorded. The Nonastreda paper does
> not explain the workpiece-photo change (question 1b), so its default stands.

## Contents

0. [Decisions](#0-decisions)
1. [Goal and success criteria](#1-goal-and-success-criteria)
2. [What the data gives for RUL](#2-what-the-data-gives-for-rul)
3. [Task definition](#3-task-definition)
4. [Protocol: folds, selection, decisions](#4-protocol-folds-selection-decisions)
5. [Preprocessing pipeline](#5-preprocessing-pipeline)
6. [Model architecture](#6-model-architecture)
7. [Training strategy](#7-training-strategy)
8. [Baselines and reference models](#8-baselines-and-reference-models)
9. [Uncertainty and calibration](#9-uncertainty-and-calibration)
10. [Evaluation](#10-evaluation)
11. [Modality-importance study](#11-modality-importance-study)
12. [Sensitivity and robustness studies](#12-sensitivity-and-robustness-studies)
13. [Reuse from OmniFuser and Project 14, and lessons learned](#13-reuse-from-omnifuser-and-project-14-and-lessons-learned)
14. [Repository and engineering](#14-repository-and-engineering)
15. [Verification and validation, part by part](#15-verification-and-validation-part-by-part)
16. [Training on Ruche](#16-training-on-ruche)
17. [Stages](#17-stages)
18. [Risks and mitigations](#18-risks-and-mitigations)
19. [Open questions for you](#19-open-questions-for-you)
- [Appendix A: default settings](#appendix-a-default-settings)
- [Appendix B: how the numbers were measured](#appendix-b-how-the-numbers-were-measured)
- [Appendix C: references](#appendix-c-references)

---

## 0. Decisions

| # | Decision | Value | Source |
|---|---|---|---|
| D1 | Validation protocol | 10-fold leave-one-tool-out, as in OmniFuser. Fold *i*: test T*i*, validation T*i+1* (cyclic; T1 after T10), training on the other 8 tools | You |
| D2 | Time to failure (TTF) | **Confirmed first dulled**: the first recorded pass labelled `dulled` that is followed by at least 2 more `dulled` passes among its next 3 recorded passes | You |
| D3 | Horizon | **H = 1,500 s of cutting** (25 min, about 15 median passes). Headline metrics use the points with RUL ≤ H. Training uses every pre-TTF point, with its target capped at H. H is a dataset setting (§6.11) | You (your k = 15 passes, expressed in seconds since 2026-11-04) |
| D4 | Modalities | Force with **all three channels** Fx, Fy, Fz at 1 kHz, plus the magnitude \|F\| (§5.4); tool photo; chip photo; workpiece photo. Workpiece photos are standardized per image and every photo type must pass the Stage-3 stage-step leakage probe (§6.2; `docs/EDA_RUL.md` §6) | You; leakage probe from Stage 1 |
| D5 | Timeline | 6 weeks, to 2026-12-16. Design freeze on 2026-12-01 (D15) | You |
| D6 | Output | **RUL in seconds of operation only, at every segment,** with its interval. Passes are a Nonastreda concept that other datasets lack (your instruction of 2026-11-04). Here operation means spindle-engaged cutting time; idle time between passes is not recorded, so wall-clock time cannot be predicted | You |
| D7 | Segment | 10 s of raw signal from the (causally trimmed) start of each recording. A remainder shorter than 5 s joins the previous segment; a recording shorter than 10 s is one segment | Default; §12 tests 5 s and 20 s |
| D8 | Duplicated force | A duplicated recording (§2.4) is attached to the **latest** pass of its group; the earlier members have missing force. This affects 134 of the 512 passes (26 %), almost always block 3 | Default (causally safe) |
| D9 | Observation availability | An observation (here a photo) taken after recording r is input only to predictions made after r ends | Default; removes OmniFuser's lag-0 leakage |
| D10 | Loss | MSE on the predicted mean (your request), plus a likelihood term for the spread with the mean detached, plus auxiliary wear and consistency terms (§6.7) | Default |
| D11 | Sign | RUL = τ* − τ ≥ 0, with τ* the failure time in seconds of operation | Convention |
| D12 | Time axis | **Gap-aware.** Positions follow (run, block). Runs missing *between* a tool's first and last recorded run count as unrecorded cutting, at the median duration of the tool's recorded passes before the gap: T2 run 10 (4 passes), T4 runs 6-9 (16 passes). T1's missing *leading* run 1 is not counted | Review (§2.5); supported by the Stage-1 wear-jump test (`docs/EDA_RUL.md` §2); **confirmed by you (2026-11-04)** |
| D13 | Fusion | **Two candidates: F1 token fusion and F0 late fusion (generalized product of experts).** The others are documented as alternatives (§6.5) | You |
| D14 | Causal inputs | Every input at a prediction point is computable from data recorded up to that point. This covers trimming, segmentation and the time features | Review |
| D15 | Test-score policy | No test-tool score is read before the design freeze (end of Stage 6, 2026-12-01). After the freeze nothing about the model changes; the headline, importance and sensitivity results are read from then on | Review |
| D16 | Encoders | **One fixed encoder per modality, no screening:** Mantis-8M for every force channel, DINOv2-B/14 for every photo type, both frozen (§6.2). Alternatives are documented, not run | You (2026-11-04) |
| D17 | Built to transfer | No PCA and no dataset-specific physics features inside the model. Frozen embeddings enter through a LayerNorm and a learned, shared low-rank projection. Channels are a set with descriptor embeddings. Only generic level descriptors complement the encoder (§6.2, §6.10) | You (2026-11-04): a versatile backbone |
| D18 | Core and adapter | Everything dataset-specific lives in one **adapter** that turns a dataset into generic units, recordings and observations with a failure time per unit. The **core** (segments, targets, features, tokens, model, training, calibration, evaluation) never refers to passes, runs, blocks, named photo types or flank wear (§6.11) | You (2026-11-04) |
| D19 | Seeds | **3 seeds everywhere:** gate decisions and the final model are 3-seed ensembles | You (2026-11-04) |
| D20 | OmniFuser baseline | **B6, OmniFuser-RUL:** the OmniFuser architecture (frozen EfficientNet-B0 photos, C2F fusion, an RUL head) on this project's targets, processing, folds, weights, selection and metrics. A separate model: no RUL-Fuse decision reads it. Test reads follow D15. Plan: `docs/OMNIFUSER_RUL_PLAN.md` Version 3 (2026-11-06): 1 s steps scored on the 10 s points, a GRU memory and the temporal loss (D21, D22). **Version 4 (2026-11-07): one configuration trained once (10 folds × 3 seeds) and reported; no modality study and no comparison of settings** | You (2026-11-05, 2026-11-06) |
| D21 | Memory | **A causal recurrent memory of each unit's whole history** (`rulfuse.models.memory.CausalMemory`, a GRU over block means of the step inputs; step t reads only complete blocks). B6 uses it now. In RUL-Fuse, F1 and F0's signal expert concatenate its state (over the force tokens, block 1) with their output at every force token before the head; Gate 2b reports a no-memory control (§7.5). F1's attention already reaches the whole history; the memory adds a running summary that is cheap at any length | You (2026-11-06): each new segment uses the information of all previous ones |
| D22 | Rising predictions | **Penalized in the loss of both models** (`rulfuse.models.losses`): λ_mono · relu(μ_later − μ_earlier) on pairs of points of one unit (RUL can only fall), with the slope-consistency term where the target is below the cap; λ_mono = 1/15 and λ_cons = 0.05 on y/H in both models (§6.7). **Both terms move only the earlier prediction of a pair; the later one is the fixed reference** | You (2026-11-06). Weights and reference changed 2026-11-07, after your question: with 1.0 on both predictions, an earlier prediction the model cannot correct dragged later ones down to it |
| D23 | Inference modes | **Two modes for every model's predictions** (`rulfuse.inference`): **current**, each prediction as the model makes it (the primary result); **context**, where the unit's earlier predictions, aged by the operating time since, are blended with the new one (time constant 120 s and cap margin H/15, fixed in advance). Both are reported side by side with a paired comparison and a decision rule fixed in advance (§10.3). B6 reports both now; RUL-Fuse's evaluation uses the same function | You (2026-11-07): a mode where previous segments guide the new prediction, and a way to compare the two |

---

## 1. Goal and success criteria

**Goal.** At the end of every 10 s force segment, predict how many **seconds of
cutting** the tool has left before it is (confirmed) dulled. The model reads
everything recorded so far: every past force segment and every photo already
taken. It outputs a calibrated interval as well as a point value. The project
also measures how much each modality and each force channel contributes. The
core architecture stays free of anything specific to Nonastreda (D18).

**Success criteria.** These are fixed in advance and measured on the test
tools with the 3-seed ensemble of the frozen design.

| # | Criterion | Threshold |
|---|---|---|
| S1 | Accuracy | Window MAE below the best *deployable* baseline B2: **468 s** (§2.7). One-sided paired Wilcoxon test over the 10 test tools, p < 0.05. With 10 tools this needs about 8 or more tools improved, with small losses on the rest. The window-aware constant B0 (412 s) is reported alongside |
| S2 | Intervals | Pooled coverage of the nominal 90 % interval within [0.85, 0.95]; per-tool coverage reported. Winkler interval score below B2's empirical intervals (5,388 s; §8) |
| S3 | Modality importance | Shapley values for the modalities that pass the leakage probe (force, tool, chip; the workpiece only if it passes), with 95 % bootstrap intervals over tools. The sign of the most important modality is resolved |
| S4 | Integrity | Every check of §15 passes. No decision reads a test score (D15) |
| S5 | Reproducibility | Every reported number regenerates from the saved predictions with one command |

**Non-goals.** There is no published RUL benchmark on Nonastreda to match.
External datasets (MATWI, PHM 2010) are out of scope for 6 weeks.

---

## 2. What the data gives for RUL

### 2.1 Inventory

| Item | Content |
|---|---|
| Tools | 10 (T1-T10), each run from new until well past dulled |
| Passes | 512 recorded (`T#R#B#`), 4 blocks per run, 128 recorded runs. Per tool: 40-60. Five runs are missing from the record (§2.5) |
| Force | Fx, Fy, Fz at 1 kHz. Trimmed pass length: 97 s median. T9's passes are about 81 s and T10's about 122 s, so the cutting conditions likely differ by tool (§19) |
| Tool photo | 1 per pass, taken **after** the pass. About 1550 × 500 px (T1 and T2: 3100 × 1000). The cutting edge lies between 0.51 and 0.68 of the photo height (OmniFuser photo probe) |
| Chip photo | 1 per pass, 1024 × 1024 JPG, in `D:\Project14\data\raw\nonastreda\chip`. Shows chip shape and heat tint |
| Workpiece photo | 1 per pass, 512 × 512 PNG, in `...\work`. Grey surface texture |
| Publisher spectrograms | `...\spec\{x,y,z}`: 512, 512 and 506 whole-pass images. Not used: a whole-pass input is only available after the pass |
| Labels | `image_label` (sharp, used, dulled; the annotators' visual judgement), `tool_label` (1/2/3, monotone), `flank_wear` (µm), `gaps`, `overhang` |

### 2.2 Per tool (gap-aware axis D12, TTF rule D2)

| Tool | Recorded passes | Unrecorded passes before TTF | TTF (pass) | Smoothed wear at TTF (µm) | Cutting time to TTF (s) | Pre-TTF points | Window points, passes | Window points, seconds |
|---|---|---|---|---|---|---|---|---|
| T1 | 40 | 0 (run 1 not counted) | 31 | 118 | 3,000 | 308 | 150 | 155 |
| T2 | 56 | 4 (run 10) | 44 | 112 | 4,203 | 393 | 111 | 116 |
| T3 | 52 | 0 | 36 | 123 | 3,463 | 355 | 151 | 156 |
| T4 | 40 | 16 (runs 6-9) | 38 | 113 | 3,657 | 216 | **20** | **20** |
| T5 | 48 | 0 | 19 | 110 | 1,827 | 188 | 151 | 156 |
| T6 | 60 | 0 | 47 | 109 | 4,472 | 458 | 151 | 154 |
| T7 | 56 | 0 | 47 | 109 | 4,543 | 465 | 151 | 155 |
| T8 | 52 | 0 | 46 | 119 | 4,474 | 459 | 151 | 154 |
| T9 | 52 | 0 | 44 | 111 | 3,837 | 379 | 121 | 148 |
| T10 | 56 | 0 | 44 | 111 | 5,367 | 526 | 181 | 147 |
| **All** | 512 | 20 | mean 39.6, sd 8.5 | median 112 | | **3,747** | **1,338** | **1,361** |

- **The confirmed rule agrees with physics.** Smoothed flank wear at the TTF
  is 108.5-122.6 µm on every tool, which matches the dataset's 110 µm "dulled"
  threshold.
- **T4's window is mostly unrecorded.** 14 of its last 16 passes before the
  TTF are runs 6-9, so only 20 observed points fall in its window. As a
  test tool it weighs little; as the validation tool of fold 3 it is too thin
  for model selection, which is why selection uses the whole life (§4.3).
- "Points" are segment ends of recorded passes, including force-missing
  passes (D8). Unrecorded passes have no points.

### 2.3 TTF definitions compared (pass number, gap-aware)

| Rule | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | Mean ± sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Confirmed first dulled (D2)** | 31 | 44 | 36 | 38 | 19 | 47 | 47 | 46 | 44 | 44 | 39.6 ± 8.5 |
| Literal first dulled | 31 | 44 | **20** | 38 | **16** | **24** | 47 | **27** | 44 | 42 | 33.3 ± 10.6 |
| Smoothed wear ≥ 110 µm | 31 | 44 | 28 | 38 | 23 | 50 | 49 | 45 | 44 | 44 | 39.6 ± 8.8 |
| `tool_label` = 3 | 29 | 45 | 25 | 37 | 21 | 49 | 49 | 45 | 45 | 45 | 39.0 ± 9.8 |
| Dulled from here on | 31 | 48 | 39 | 38 | 31 | 54 | 49 | 50 | 44 | 46 | 43.0 ± 7.5 |

The literal rule puts T6's failure at 76 µm and T8's at 65 µm: isolated label
flips (`EDA.md` §2). The other rules agree within a few passes, except on T3
and T5. Every tool reaches the TTF; none is censored.

### 2.4 Duplicated force recordings

OmniFuser found 132 groups of bit-identical force arrays within a run:

- 127 groups of B3 + B4;
- 4 of B1 + B2;
- 1 of B1-B4.

Their tool, chip and workpiece photos are **all distinct** (0 of 132 groups
share a photo hash), and so are their flank-wear values. So the tool did 4
physical passes per run, and only the force file was copied.

- **Default (D8).** Attach each recording to the latest member of its group;
  the earlier members are force-missing, with the copy's duration.
  - If the recording really belongs to the later pass, attaching it to the
    earlier one would show the future by one pass. Attaching it to the later
    one at worst delays real information.
- **Consequence.** 134 passes (26 %), nearly all block 3, have no force of
  their own.
  - Predictions inside them rely on the history and the photos.
  - Their points (951 of the 3,747 before the TTF, 25 %) are scored and also reported
    separately (§10).
  - The missing-force pattern reveals the block position, which is minor
    because usage is an input anyway.
- **Sensitivity arms (§12):** (i) attach the recording to every member; (ii)
  OmniFuser's deduplication (378 passes; time counted in retained passes;
  TTF recomputed with D2).

### 2.5 Unrecorded runs: the tools kept cutting

| Tool | Missing runs | Evidence from the measured wear | Reading |
|---|---|---|---|
| T1 | Run 1 (leading) | First recorded run (run 2) has mean wear 46 µm; the other tools' first runs have 27-35 µm, except T4 (47 µm) | Ambiguous; it does not change any RUL target |
| T2 | Run 10 | Wear rises 14.3 µm across the gap, against a median 8.7 µm between consecutive runs (1.6×; 2.4× at the local rate). Expected if cut: 2; if not: 1 | About one run of cutting |
| T4 | Runs 6-9 | Wear rises 41.7 µm across the gap, against a median 7.4 µm (5.6×). Expected if cut: 5; if not: 1 | About four runs of cutting |

**What goes wrong on the recorded-pass axis:**

- Every T4 point before the gap would carry an RUL 16 passes too short; T2's
  points would be 4 passes too short. These are wrong labels, and T4 and T2
  train in 8 of the 10 folds.
- T4 would look like a fast-wearing tool (TTF 22) when it is typical
  (TTF 38).

**D12 (default):** a gap-aware axis.

- Interior gaps count as unrecorded passes, at the tool's median pass
  duration. They have no inputs and no prediction points.
- The model sees the time jump through its time features.
- T1's leading run 1 is not counted. Counting it would only shift T1's
  elapsed time by 4 passes, not its RUL.

**Checks:**

- Sensitivity arms: the recorded axis, and T1's run 1 counted (§12).
- You confirmed this reading on 2026-11-04 (§19 question 1).

### 2.6 Known traps, carried over from OmniFuser and Project 14

| Trap | Evidence | Consequence here |
|---|---|---|
| Same-pass photo leaks the label | Copying the lag-0 photo's label scores 0.91-0.95 (`EDA.md` §7) | D9, plus a test |
| Run number is a shortcut | Run number alone gives 71 % accuracy (OmniFuser F2) | No run or block codes; usage enters only as explicit, switchable time features; schedule baselines B1-B3 in every table |
| Fz drifts | Sawtooth that resets every 1-3 passes; its level correlates with wear at −0.04 (F3) | All channels input (D4); a generic descriptor, the mean change since the recording started, captures the drift (§6.2); the channel study (§11.5) measures what Fz adds |
| Tool-to-tool force offset | T1's Fx sits at +1.15 sd of the training distribution (`EDA.md` §8) | Tokens carry the absolute level and the change from each tool's healthy start (§5.9). Relative-only removes static offsets such as T1's but not the cross-tool shift (Stage 2) |
| Frozen ImageNet features read wear poorly | Photo probe 0.61-0.67 accuracy against 0.95 possible; the crop helps; resolution alone does not | Edge crop, DINOv2-B features, wear-regression readout; LoRA fine-tuning documented as an alternative |
| Absolute embeddings plus elapsed time memorize units | Project 14 RUL GRU peaked after 1 epoch (`pdmfm/rul.py`) | Trajectory augmentation, small models, a low-rank projection; the token mode is decided in Gate 2b |
| Wear is not monotone | Flank wear decreases at 189 of 502 steps (F12) | Isotonic-smoothed wear targets; causal smoothing wherever wear is an *input* (R1) |
| Workpiece photos encode the stage label (found in Stage 1) | Mean brightness steps at the first run of `tool_label` stage 2 in 10 of 10 tools (96 % of its range); texture also at stage 3 (`docs/EDA_RUL.md` §6) | Per-image standardization; Stage-3 stage-step leakage probe for every photo type (§6.2); exclusion if it fails |
| Whole-pass preprocessing leaks within the pass | OmniFuser's trim uses 5 % of the pass length | Causal trim (§5.2) |
| Unrecorded runs | §2.5 | D12 |

### 2.7 The bar to beat

Settings: segment ends; gap-aware axis; fold protocol D1; the 8 training tools
of each fold; the window RUL ≤ H = 1,500 s.

| Predictor | Uses | MAE (s) | RMSE (s) |
|---|---|---|---|
| B0: constant H/2 = 750 s | That the point lies in the window, which is the truth itself | **412** | 463 |
| B1: mean training failure time − elapsed time, capped | Elapsed time | 513 | 562 |
| B2: E[min(τ* − τ, H) \| τ* > τ] over the training tools | Elapsed time | 468 | 526 |

- **Same points as the model.** These baselines are computed at the segment
  ends where the model predicts, with the cap inside the expectation (the
  MSE-optimal usage-only predictor of the capped target).
- **B0 beats B2.** The window-aware constant (412 s) beats the usage rule
  (468 s). No deployable method has B0's knowledge.
- **B2's empirical 90 % intervals under-cover:** 0.50 coverage (Winkler
  5,388 s), the S2 reference.
- **Recorded axis, for reference:** B2 scores 480 s.

---

## 3. Task definition

### 3.1 Notation (generic)

The core sees every dataset the same way (§6.11).

- **Unit.** A unit u (here a tool) runs from new to failure.
- **Recordings.** Its recordings r = 1…R_u (here passes) are contiguous sensor
  captures. Recording r starts at τ_r seconds of operation and lasts D_r
  seconds. Unrecorded operation between recordings (here the missing runs,
  D12) only shifts τ.
- **Segments.** Each recording is cut into segments (D7). Segment (r, s) ends
  e_{r,s} seconds into the recording, at operating time τ_{r,s} = τ_r + e_{r,s}.
- **Observations.** Observations (here the photos taken after each pass) carry
  the time from which they may be read (D9).
- **Failure time.** The failure time τ*_u comes from the adapter. For Nonastreda
  it is the end of the confirmed TTF pass (D2), when the dulled state is first
  observed.

### 3.2 Target

$$
\mathrm{RUL}_{r,s} = \tau^\ast - \tau_{r,s}\ \text{(seconds of operation)},\qquad
y_{r,s} = \min(\mathrm{RUL}_{r,s},\,H),\ H = 1{,}500\ \text{s}.
$$

- **One target, predicted at every segment:** "how long can I keep cutting".
  Inside a recording it falls by exactly the elapsed operating time.
- **Prediction points:** every segment end of the recordings up to the
  failure: 3,747 points. Points after the failure carry no RUL target.
- **Evaluation window:** RUL ≤ H (1,361 points). Inside it the target equals
  the true RUL.
- **Training:** every pre-failure point, with the capped (piecewise-linear)
  target.
- **Auxiliary target (optional, generic):** a **health indicator** (HI), when
  the dataset measures one. Here it is VB̃, the isotonic-smoothed flank wear of
  the current pass, over all recorded passes including those after the TTF.

**Worked examples.**

- **T5** (τ* = 1,827.0 s). At the end of segment 5 of pass 10 (record
  T5R3B2), τ = 909.7 s, so RUL = 917.3 s, about 15 min of cutting.
  `scripts/end_to_end_trace.py` reproduces these numbers.
- **T4** (τ* = 3,657.2 s). At the end of R5B4 (pass 20, τ = 1,922.1 s),
  RUL = 1,735.1 s. The recorded axis would say 193.7 s.

### 3.3 What the model may read at point (r, s)

| Input | Available | Never |
|---|---|---|
| Signals | Segments of recordings < r, plus segments 1…s of recording r | Later segments |
| Observations (photos) | Those whose availability time has passed: here the photos of recordings < r | Photos of recording r or later |
| Usage | Operating time τ_{r,s}; seconds since the current recording started, e_{r,s} | The current recording's total duration before it ends; any count of passes, runs or blocks |
| Labels | None | `image_label`, `tool_label`, flank wear, gaps, overhang |

The causality tests of §15 (P10, P11) enforce this table.

---

## 4. Protocol: folds, selection, decisions

### 4.1 Folds (OmniFuser's `leave_one_tool_out_folds`)

| Fold | Test | Validation | Training |
|---|---|---|---|
| 1 | T1 | T2 | T3-T10 |
| 2 | T2 | T3 | T1, T4-T10 |
| 3 | T3 | T4 | T1, T2, T5-T10 |
| … | … | … | … |
| 10 | T10 | T1 | T2-T9 |

### 4.2 What is fitted where

| Object | Fitted on |
|---|---|
| Level-descriptor standardization; ridge probes and HI readouts (α by inner leave-one-tool-out) | The 8 training tools |
| Wear readouts used as training inputs | Cross-fitted on the training tools (§7.2) |
| Tool baseline (§5.9) | Each tool's own first B recordings, causally, including the test tool |
| HI threshold θ and degradation prior (B5, R1) | The training tools |
| Step selection | The validation tool |
| Interval calibration | Out-of-fold predictions on the training tools (default), or the validation tool (§9) |
| Test tool | Prediction only |

### 4.3 Metric for selection and decisions

**MAE in seconds.** It is also reported relative to the horizon (MAE / H), so
that datasets with other horizons can be compared.

**In-fit step selection.**

- Every 10 steps, compute the validation tool's MAE over its whole pre-failure
  life, with the training-loss weights (§7.3), using the EMA weights.
- Keep the step with the lowest mean of 3 consecutive evaluations.
- Why the whole life and not only the window: T4, the validation tool of fold
  3, has only 20 window points.

### 4.4 Design decisions: gates with rules fixed in advance

All gate scores come from the 3-seed ensemble, on the validation tools.

- **Rule A: adding complexity or switching an option on.** Adopt only if both
  hold:
  - the mean validation window MAE (s) over the 10 folds improves by at
    least the gate's threshold;
  - the option wins on at least 7 of the 10 validation tools (the chance
    probability of 7 or more is 0.17).
- **Rule B: choosing among peers of similar cost.** Take the best mean. Within
  half the gate's threshold, take the cheaper or more interpretable one.

| Gate | Metric | Rule-A threshold |
|---|---|---|
| Stage-3 feature check (no ranking, §6.2) | Pass/fail: the VB probe beats the training mean; the photo leakage probe | – |
| Gate 2 (fusion, options, sweep) | Window MAE (s) | 10 s (H/150) |
| Gate 3 (head, calibration, filter) | Winkler 90 % interval score (s), subject to validation coverage within [0.85, 0.95] | 3 % relative |

**Caveat.** Fold *i*'s validation tool is fold *i+1*'s test tool, so
cross-fold choices touch every tool once. This is the same as in OmniFuser
and Project 14. The plan keeps the decisions few (10, listed in §17.3) and
states the count in the report.

### 4.5 Seeds and ensembles

- Gates and the final runs both use the 3-seed ensemble (D19).
- The seed spread is a diagnostic (§7.8), not a reported score.

### 4.6 Test-score policy (D15)

1. Test predictions are written by every fit but are not aggregated or read
   before the freeze. `rulfuse aggregate` refuses to score test tools for
   studies dated before the freeze unless given `--after-freeze`.
2. The freeze commits the final configuration (`configs/final.yaml`) and its
   hash.
3. After the freeze, the final runs (Stage 7) come first, then importance and
   sensitivity (Stage 8). None of them may change the model.

---

## 5. Preprocessing pipeline

Every step is a pure function with a unit test (§15). The pipeline is cached
by a signature hash: data manifest + settings.

**Adapter or core (D18).**

- **Nonastreda adapter:** loading, trimming and the timeline with its gaps and
  duplicates (§5.1-§5.3), the TTF rules (§5.6), and the photo preparation
  (§5.7).
- **Generic core:** channels and descriptors (§5.4), segmentation (§5.5), the
  RUL target (§5.6) and normalization (§5.9).

### 5.1 Load: `rulfuse/data/nonastreda.py`, vendored from OmniFuser

- Reuse OmniFuser's checks: ID parsing (`RECORD_ID`, `_decode_record_id`),
  CSV reading (`_read_csv`), the MAT label against the CSV label, finite
  forces, present images, the `(3, samples)` shape.
- Load **all 512 recorded passes**. Detect the duplicate groups (bit-identical
  arrays within a run) and return them as metadata.
- Add the chip and workpiece paths and verify them against `MANIFEST.json`'s
  sha256.

### 5.2 Trim: OmniFuser's span rule, made causal

OmniFuser's rule (`_nonastreda_span_bounds`) works in three steps:

1. A 20-sample envelope of the force magnitude.
2. A reference level: the median envelope over the first **5 % of the pass**.
3. A threshold at 50 % of that reference.

Step 2 needs the pass length, which is unknown while the pass runs.

- **Causal variant (D14).** The reference is the median envelope over the
  first **5 s** (5,000 samples), about 5 % of a median pass. Steps 1 and 3 are
  unchanged.
- The end is the last active sample. It is known when the pass ends, which is
  also when the last segment's prediction is made.
- **Validation (§15, P2):** agreement with OmniFuser's trims on the real
  passes.

### 5.3 Timeline: `rulfuse/data/timeline.py`

1. Order the recorded passes by (run, block).
2. Insert the unrecorded passes of interior gaps (D12).
3. Compute D_p and τ_p.
4. Apply D8 to the duplicates.
5. Record each photo's availability time: the end of its pass.

### 5.4 Channels

**All three measured channels are always input:** Fx, Fy and Fz. One derived
channel is added by a generic rule: the magnitude of every 3-axis sensor group.

| Channel | Definition | Why |
|---|---|---|
| Fx, Fy, Fz | Raw, trimmed | Every measured channel (your instruction) |
| \|F\| | √(Fx² + Fy² + Fz²) | The resultant force, the classic wear indicator. Stage 1: ρ 0.70 with wear across tools |

**Fz_rel is no longer a channel.** Its information was Fz's drift since the
pass started (Stage 1: ρ 0.28 with wear). It is now a generic level descriptor
of every channel, the mean change since the recording started (§6.2), and needs
no Nonastreda-specific rule.

**One channel, one vector,** built the same way for any channel of any dataset:

1. The Mantis-8M embedding of each 512-sample sub-window, mean- and
   sd-pooled over the segment: 512 values.
2. The channel's 7 **level descriptors** (§6.2): what a scale-normalized
   encoder cannot see.

**The channels form a set, not fixed slots.**

- Each channel vector gets a *channel-descriptor embedding* (sensor type, axis,
  derivation; §6.3), and the segment token pools over the set.
- A missing channel is simply absent, and another dataset with other sensors
  (for example accelerometers or a microphone) uses the same machinery.
- Channel dropout (§6.8) and the channel study (§11.5) remove channels from the
  set.

### 5.5 Segment

- S = 10 s windows from the (trimmed) recording start; a remainder under S/2
  joins the previous segment. On Nonastreda this gives 8 segments per pass on
  T9, 12 on T10 and 10 elsewhere.
- Encoder sub-windows: 512 samples, the last aligned to the segment end so
  every sample is covered.
- Each segment stores: unit, recording, index, start and end sample, e_{r,s},
  τ_{r,s}, `signal_present`.

### 5.6 Labels

- **Adapter (Nonastreda).**
  - The five TTF rules of §2.3, selected by config key `ttf_rule`. Each is
    applied to the recorded label sequence; a TTF can never fall in a gap.
  - It returns each unit's failure time τ* (the end of the TTF pass).
  - Optionally it returns the health indicator (VB̃: isotonic flank wear, full
    and causal) and the stage labels used by the leakage probe.
- **Core.** RUL in seconds from τ*; the capped target; the evaluation and
  pre-failure masks. The causal HI is used where a measured HI is an *input*
  (oracle R1).

### 5.7 Images: `rulfuse/data/images.py`

Reuse OmniFuser's `load_photo` (crop) and `ImageBank`. All photos are
normalized with the ImageNet statistics.

| Photo | Preparation |
|---|---|
| Tool | Crop rows 0.30-0.85 at full width (the OmniFuser probe's best band). Resize to 224 × 672 (Project 14's setting) |
| Chip | Resize to 448² |
| Workpiece | Resize to 224²; standardize each image's luminance to mean 0.5, sd 0.2, to remove the per-stage exposure step (`docs/EDA_RUL.md` §6; Stage 2 showed the texture step remains) |

### 5.8 Stores and caches

- **Location:** everything generated lives under `RULFUSE_DATA_ROOT`, default
  `D:\RULModel`, outside OneDrive (§14.1). This covers the raw copy, the
  stores, the caches and the runs.
- **Force store (built in Stage 0):** `interim/forces/`, holding:
  - `signals.npy`: the 378 distinct recordings, float32, memory-mapped (452 MB);
  - `index.csv`, `groups.json`, `meta.json` (with the MAT file's sha256).
- **Segment store (built in Stage 2, `rulfuse build-store`):**
  `interim/segments/<signature>/`. It points into the force store and copies no
  samples. It holds:
  - `passes.parquet`: one row per pass, unrecorded ones included. Trim bounds,
    durations, cumulative time, force attachment, the photo position (D9),
    labels, smoothed and causal wear;
  - `segments.parquet`: one row per 10 s segment of every recorded pass,
    including those after the TTF. Sample bounds, the distinct-segment id for
    encoding, the input times, the label time, the RUL targets of prediction
    points, and the baseline flag;
  - `tools.json` (TTF, baseline passes) and `meta.json` (signature, inputs,
    counts).
- **Store signature.** A hash of the force store's sha256, the label files' sha256,
  every `DatasetOptions` field and the segment settings. The default store is
  `8d2b69c23453`: 532 passes, 5,125 segments, 3,747 prediction points, 3,777
  distinct segments to encode.
- **Feature caches (Stage 3), two in total:**
  - `interim/features/force-mantis8m-<signature>/`: per distinct segment and
    channel, the pooled Mantis-8M embedding (512 values) and the 7 level
    descriptors (3,777 segments × 4 channels);
  - `interim/features/photo-dinov2b-<signature>/`: per photo and type, the CLS
    and 2 × 4 patch-grid tokens (9 × 768 values; 1,536 photos).
  - The signature names the checkpoint revision and the preprocessing (Project
    14's `FeatureStore`). Duplicated recordings are encoded once.

### 5.9 Normalization

- **No PCA in the model (changed 2026-11-04, D17).**
  - The frozen embeddings reach the model through a LayerNorm and a learned
    low-rank projection (§6.3), which are part of the reusable backbone (§6.10).
  - The only statistic fitted on the training tools is the standardization of
    the 7 level descriptors, whose units differ between datasets: the
    "dataset adapter".
  - PCA remains only as an analysis tool (inside the leakage probe and the
    diagnostics) and in baseline B4.
- **Tool level (causal baseline).**
  - The relative half of each token is taken against the tool's healthy start:
    $$\tilde z = z - b_u(t),\qquad b_u(t) = \text{mean of } z \text{ over the tool's force segments in its first } B=4 \text{ recordings, among those seen by } t .$$
  - Photos are taken relative to the tool's first photo.
  - The HI readouts (§6.2) stay **absolute**: a dull edge looks dull on any
    tool.
  - **Token mode (changed in Stage 2): "both" by default.** Every token is built
    from the absolute values and from their change since the tool's healthy
    start: two inputs to the same projection.
    - The plan assumed relative tokens would remove tool identity.
    - Stage 1 found absolute force levels relate to wear across tools at least as
      well (`docs/EDA_RUL.md` §7).
    - Stage 2 measured, on per-segment channel statistics
      (`runs/stage2/validation.json`):
      - relative features *raised* the test tool's shift in 6 of 10 folds
        (median 0.35 sd against 0.31 absolute), and only static offsets shrank
        (T1: 0.97 → 0.34);
      - tools were *easier* to identify from relative features (35 % against
        23 %, chance 10 %).
    - Gate 2b compares relative, absolute and both as peers (Rule B).

---

## 6. Model architecture

### 6.1 Overview

```
 pass 1 ......... pass p-1 | pass p: segments 1..s                     time →
 ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ force seg│...│ force seg│...│ force seg│  raw 1 kHz Fx, Fy, Fz (+ |F|), 10 s
 └────┬─────┘   └────┬─────┘   └────┬─────┘
      ▼ per channel: frozen Mantis-8M on 512-sample sub-windows (mean+sd) ⊕ 7 level descriptors
      ▼ absolute ⊕ change from the healthy start → LayerNorm → shared low-rank projection
      ▼ + channel-descriptor embedding → pooled over the channel set → MLP → d=64
 images available before the point (here tool | chip | workpiece, each if it passes the leakage probe)
      ▼ frozen DINOv2-B: CLS + 2×4 patch tokens; absolute ⊕ change from the first image
      ▼ LayerNorm → shared low-rank projection + image-type embedding → attention pool → d=64
      ▼ + HI readout when the dataset has an HI (cross-fitted ridge probe)
 ─────────────────────────────────────────────────────────────────────────────
 tokens ordered by availability time + modality embedding (+ usage features, switchable)
      ▼
 causal fusion model with a relative-time attention bias:
      F1 token fusion (one transformer)   |   F0 late fusion (experts + gPoE)
      ▼ at every force-segment token (= prediction point)
 heads: RUL in seconds, Student-t (μ, σ, ν) · health-indicator nowcast (optional)
      ▼
 3-seed ensemble → calibration (out-of-fold) → optional causal RUL filter
      → "≈ 640 s (90 %: 420-900 s) of cutting left"
```

Why a hierarchy:

- **Too much raw data.** "All previous data" at 1 kHz is up to 47 passes ×
  100 s × 1000 × 3, about 14 M values.
- **Tokens make it small.** Frozen encoders turn each 10 s segment into one
  token. A tool's full history becomes about 470 force tokens plus about 140
  photo tokens: small for a causal model, and no truncation is needed.
- **Little to train on.** Only 8 trajectories train each fold, so the
  learnable part stays small (about 0.1-0.15 M parameters) on top of fixed,
  general representations.

### 6.2 Encoders and level descriptors (fixed; no screening)

**One encoder per modality, chosen by reasoning, not by a search (D16, your
instruction of 2026-11-04).** This removes Gate 1's encoder screening and every
Ruche feature job.

| Modality | Encoder | Why this one | Cost |
|---|---|---|---|
| Force (every channel) | **Mantis-8M** (`paris-noah/Mantis-8M`), frozen | A time-series foundation model pretrained for *classification* on diverse public series, whose frozen features are its intended use. Channel-independent, so any sensor set fits. Small (8 M parameters). Already wrapped and cached in Project 14 | 3,777 segments × 4 channels × 20 sub-windows ≈ 300 k windows: minutes on one A100 (Ruche job J2) |
| Photos (tool, chip, workpiece) | **DINOv2-B/14** (`facebook/dinov2-base`), frozen | Self-supervised general features that keep fine local texture (what wear changes). Not gated. Project 14 wraps the same family | 1,536 photos: minutes on one A100 (job J2) |

**Level descriptors: the physics features that stay.**

- **Why they are needed.** Foundation encoders are built to be
  scale-invariant: MOMENT normalizes every input window (reversible instance
  normalization), and Project 14 treated Mantis the same way. Its
  `preprocess.py` describes level descriptors as "what per-input normalized
  encoders cannot see (absolute level, DC trend)". Without them, the absolute
  force level, Stage 1's strongest wear signal (RMS of \|F\|, ρ 0.70 with wear
  across tools), would be lost. Stage 3 verifies Mantis's scale invariance on
  the real segments.
- **What they are.** Per channel and segment, 7 generic descriptors (Project 14's
  `level_descriptors` and `slow_summary`, plus one):
  - log10 RMS and log10 peak of the AC part;
  - Pearson kurtosis;
  - crest factor;
  - mean;
  - least-squares slope;
  - the mean change since the recording started: the drift a recording
    accumulates. It replaces the Nonastreda-specific Fz_rel channel.

  They exist for any signal, need no machine parameters, and enter both
  absolute (standardized per dataset) and relative to the tool's healthy start
  (log ratios: unit-free).
- **Dropped from the model:** the Nonastreda-specific features. Those are the
  energy at the inferred 66.7 Hz spindle line and its harmonics, and the
  period-15 synchronous average: they need the machine's spindle speed and tooth
  count. The fixed-Hz band energies also go; they depend on the sampling rate
  and the process.
- **Kept outside the model:** the full handcrafted set (H) feeds baseline B4
  and the diagnostics only.

**Alternatives (documented, not run).**

- **Force:** MOMENT-1 base or large (the fallback if Mantis fails the feature
  check), Chronos-T5 embeddings, AST, segment spectrograms into DINOv2, own
  PatchTST.
- **Photos:** EfficientNet-B0, SigLIP 2, DINOv3 (gated), handcrafted colour and
  texture.
- **Fine-tuning:** LoRA on DINOv2 with the wear target.

**Stage-3 feature check (replaces Gate 1's screening; Ruche job J3 on `cpu_short`;
no ranking).** It only checks that the chosen features work and are clean.

1. **Scale check.** Is a window's Mantis embedding (nearly) unchanged when the
   window is multiplied by 2? If so, the level descriptors alone carry the
   level, as expected. They are kept either way.
2. **Information check (go/no-go).**
   - A ridge probe (α by inner leave-one-tool-out) maps pass-level pooled
     features to VB̃, scored on each fold's validation tool. Force (Mantis +
     descriptors) and each photo type must beat the training-mean predictor.
   - The descriptors alone and Mantis alone are reported too, to show what each
     adds.
   - Only a force no-go triggers the fallback encoder (MOMENT-1-base).
3. **HI readouts.** The cross-fitted ridge readout of the HI (here VB̂) from
   each image type (§7.2).
4. **Leakage probe for every photo type.**
   - `rulfuse.audit.stage_steps` runs on the photo features' first 3 principal
     components (PCA as an analysis tool only) and on the HI readout.
   - A type passes if a `tool_label` stage boundary is its largest run-boundary
     step in **at most 3 of 10 tools**.
   - Failing types leave the headline model and the Shapley study and are
     reported as a leakage demonstration. The Stage-2 preview says the
     workpiece photos will fail (texture steps in 9 of 10 tools after
     standardization).

### 6.3 Tokens and time

| Token | Content | Position |
|---|---|---|
| Force segment (r, s) | For each of the 4 channels: LayerNorm of [Mantis embedding ⊕ descriptors, absolute ⊕ change from the healthy start] → shared low-rank projection (rank 16) + channel-descriptor embedding; mean over the channel set → MLP → d = 64. `force_present` flag; a force-missing segment uses a learned mask vector | Its end time τ_{r,s} |
| Image (r, m), m = an image type supplied by the adapter (here tool, chip, work) | 9 DINOv2-B tokens (CLS + 2 × 4 grid), absolute ⊕ change from the unit's first image of that type → LayerNorm → shared low-rank projection (rank 16) + image-type and patch-position embeddings → attention pooling → d = 64; plus its HI readout (here VB̂) when the dataset has an HI | At its availability time: here after the last segment of its pass, so only from the next pass on (D9) |

- **Usage features (U, switchable):** Fourier features of the operating time
  τ_{r,s} (scaled by H) and of e_{r,s}, the seconds since the current recording
  started. They are added to every force token. The absolute time embedding is
  part of U and switches off with it.
- **Relative time:** each attention head adds a learned bias
  b_h(Δτ) = MLP_h(log(1 + Δτ / 10 s)), where Δτ is the cutting time between
  the query and the key. This gives every token, photos included, its age
  relative to the prediction point, without revealing absolute time.
- **Modality embedding** on every token.
- **Channel descriptors (for transfer).** A channel's embedding is the sum of
  three learned embeddings: sensor type (force, acceleration, sound, …), axis
  (x, y, z, none) and derivation (raw, magnitude).
  - Nonastreda uses force × {x, y, z} × raw and force × none × magnitude.
  - A new dataset reuses the descriptors it shares and learns the new ones.
- **Sensing-only control (§11.4):** a causal model over the full history can
  infer elapsed time from the token count, even with U off. The control
  therefore reads only the last 4 recordings, with U off.

### 6.4 Fusion: the two candidates (D13)

| ID | Design | Why it is a finalist |
|---|---|---|
| **F1 Token fusion** | One causal transformer over the interleaved sequence of §6.3: 2 layers, d = 64, 4 heads, FFN 128, pre-LN, dropout 0.2, relative-time bias, U features. The prediction is read at every force token | The most flexible. Attention can learn cross-modal timing, for example "the force has risen since the last photo" |
| **F0 Late fusion (generalized product of experts)** | Three experts, each with a Student-t head (details below) | Robust to a missing modality. The weights α_m read directly as time-resolved modality importance. Project 14 found late fusion hard to beat on Nonastreda |

**F0's experts.**

- **U:** an MLP on the usage features.
- **S (signals):** a causal transformer over the force tokens, whatever the
  channel set.
- **I (images):** one causal GRU (d = 32) over all image tokens, whatever their
  type; a type embedding tells them apart. Between images its mean falls with
  the operating time since the last image.
- One expert per photo type would be Nonastreda-specific. The Shapley study
  (§11) still separates the photo types by retraining.
- Sensing experts get **relative time only**: no absolute usage, so usage is
  not counted several times.

**F0's fusion.** Each expert's Student-t variance is used as a Gaussian
precision π_m.

$$
\pi = \textstyle\sum_m \alpha_m \pi_m,\qquad \mu = \textstyle\sum_m \alpha_m \pi_m \mu_m / \pi,\qquad \nu = \textstyle\sum_m \alpha_m \nu_m,
$$

- α_m is a softmax over the *present* experts, from a small gate MLP on the
  experts' σ and the usage features.
- Absent modalities simply drop out of the sum.

**Memory (D21).** Both candidates read a recurrent summary of the unit's whole
history, the shared `CausalMemory` (a GRU, hidden 64) over the force tokens:

- F1 concatenates its state with the transformer output at each force token
  before the heads;
- F0 adds it to the signal expert;
- the image expert is a GRU already.

The memory is causal (it reads only steps up to the token), and its cost grows
linearly with the history. That matters if the segment-length arm (§12) or a
larger dataset makes the token history long. Gate 2b runs the winner without the
memory as a reported control.

**Together** the two finalists bracket the design space: F1 has full
interaction, F0 has none beyond the weighting. Gate 2 compares them, each at
its best of three learning rates (Rule B), then tests the options on the
winner (§7.5).

### 6.5 Alternatives, documented but not run

| Alternative | Idea | Why not now | Revisit if |
|---|---|---|---|
| F2 Gated cross-attention | Force-stream transformer; per photo type h ← h + tanh(α_m)·CrossAttn(h, photos); α_m starts at 0 (Flamingo) | Sits between F0 and F1 and adds little information for its cost; treats force as the main stream, while the photos carry most of the wear signal (OmniFuser) | F1 wins but its attention maps show photos ignored |
| F4 Pass-level GRU | GRU over per-pass vectors [force mean and sd, photos], plus attention pooling of the current pass | A coarser F1; prediction inside a pass is weaker | F1 overfits and smaller models are needed |
| Early concatenation | Append the last photos' features to every force token, then one sequence model | A special case of F1 without photo history | Never as a main model; useful as a debugging reference |
| C2F (OmniFuser), MBT, Perceiver | Shared/private decomposition; bottleneck tokens; latent cross-attention | OmniFuser: C2F did not beat concatenation; Project 14's Gate-2 pilot: no candidate beat late fusion | A larger dataset (MATWI) is added |
| MulT | Bidirectional cross-modal transformers per modality pair | 12 cross-attention streams for 4 modalities; too large for 8 trajectories | — |
| F5 Health indicator + extrapolation | Fused HI nowcast (here VB̂(t)), then a degradation curve with a prior from the training tools; RUL = crossing of θ ≈ 112 µm | Not a fusion strategy. Its machinery is kept and used by baselines B5 and R1 (§8) | F0 and F1 lose to B5; then the fused nowcast + F5 becomes a candidate |
| TFT-style variable selection | Learned per-token feature gating | Overlaps with F0's weights | — |

### 6.6 Heads

| Head | Parameterization | Status |
|---|---|---|
| **Student-t (RUL in seconds)** | μ = H·sigmoid(w_μᵀh); σ = softplus(w_σᵀh) + 0.005·H; ν = 3 + softplus(w_νᵀh) (finite variance) | Default |
| Quantile | 7 quantiles 0.05…0.95, non-crossing by cumulative softplus increments; pinball loss | Gate-3 comparator |
| Health-indicator nowcast | Linear on h; used when the dataset has an HI (here flank wear) | Auxiliary task |
| NIG (evidential), discrete hazard | Project 14 `losses.nig_*`; hazard over time bins | Alternatives, not run: the ensemble already gives the epistemic part |

- Intervals are clipped to [0, H].
- **Removed (dataset-specific):** the passes head, the single-unit control, and
  the ordinal wear-class head (CORAL), which depended on Nonastreda's three
  classes.

### 6.7 Loss

$$
\mathcal L = \underbrace{\textstyle\sum_i w_i(\mu_i-y_i)^2/H^2}_{\text{MSE (primary)}}
+ \lambda_{\text{nll}}\,\textstyle\sum_i w_i\,\mathrm{NLL}_t\!\big(y_i \mid \mathrm{sg}(\mu_i),\sigma_i,\nu_i\big)
+ \lambda_{\text{cons}}\!\!\sum_{(i,j)}\!\frac{\big(\mu_i-\mu_j-(\tau_j-\tau_i)\big)^2}{H^2}\,\mathbb 1[y_i<H]
+ \lambda_{\text{mono}}\!\!\sum_{(i,j)}\!\frac{\mathrm{relu}(\mu_j-\mu_i)}{H}
+ \lambda_{\text{hi}}\,\mathrm{Huber}\big(\widehat{\mathrm{HI}}-\mathrm{HI}\big)/s_{\mathrm{HI}}
$$

- **Point weights** w_i: defined in §7.3.
- **Stop-gradient.** sg stops the gradient, so the mean is driven by MSE as
  you asked, and the spread learns from the likelihood without pulling the
  mean.
- **Consistency term.** True RUL falls by exactly the operating time between
  two points of the same unit (pairs at most 0.13·H = 200 s apart). For F0 it
  applies to the fused output.
- **Monotonic hinge (D22).** For the same pairs (i earlier than j), and for every
  pair of consecutive force tokens, a predicted RUL that rises is penalized:
  relu(μ_j − μ_i). It holds in the capped region too, where the consistency
  term does not apply. The functions are shared with B6
  (`rulfuse.models.losses`).
- **The later point is the reference.** In both temporal terms the later
  prediction is detached: they move only the earlier prediction. So an earlier
  prediction the model cannot correct never drags later ones down with it
  (D22, 2026-11-07).
- **Defaults:** λ_nll = 0.5, λ_hi = 0.3 (0 without an HI), λ_cons = 0.05,
  λ_mono = 1/15 (D22). The hinge weight acts as a tolerance: it can hold a
  prediction at most about λ_mono × H = 100 s away from what the MSE wants.
  With the detach, a larger weight would shift every prediction up under noise
  the model cannot remove (about 160-215 s at λ_mono = 1). History: λ_cons was
  0.1 before 2026-11-06, then 0.5 with λ_mono = 1.0 until 2026-11-07. MSE and MAE
  are always both reported (§10).

### 6.8 Regularization and augmentation (8 trajectories per fold)

1. Frozen encoders, a rank-16 shared projection, d = 64, dropout 0.2, weight decay 0.05.
2. **Trajectory augmentation (the main guard against the usage shortcut).**
   - Delete recordings (probability ρ ~ U[0, ρ_max], ρ_max = 0.2) or duplicate
     them, between recording B+1 and the failure recording. Unrecorded
     stretches may be shortened too.
   - Recompute positions, times, the TTF and every target.
   - With cached tokens this is index manipulation only. The tools then wear
     faster or slower per recording, which breaks the link between elapsed time
     and RUL.
3. **Modality dropout.**
   - F1: each sensing modality is dropped for the whole sequence with
     p = 0.3, independently; all of them together with p = 0.05, so the U-only
     state is seen.
   - F0: the same probabilities drop experts.
4. Token dropout 0.1; Gaussian feature noise, sd 0.1 in standardized units.
5. **Channel dropout:** each force channel is removed from the channel set with
   p = 0.1 per sequence.
6. Batching, optimizer, curriculum, selection and diagnostics: §7.

### 6.9 Size

| Part | F1 | F0 |
|---|---|---|
| Force: shared projection 2 × 519 → 16 → 64, channel-descriptor embeddings, MLP | about 22 k | about 22 k |
| Photos: shared projection 2 × 768 → 16 → 64, type and position embeddings, attention pooling | about 30 k | about 30 k |
| Backbone(s) | about 70 k | signals about 70 k + one image GRU about 7 k + U MLP about 5 k |
| Heads, gate | about 2 k | about 10 k |
| **Trainable total** | **about 0.12 M** | **about 0.14 M** |

The rank-16 bottleneck plays the part PCA played: few trainable parameters for 8
training trajectories. The projection is learned, shared and reusable.

### 6.10 Designed as a reusable backbone (D17)

The goal is a model that generalizes beyond Nonastreda and can serve as a
backbone for other machining datasets. The design separates what transfers
from what each dataset must supply.

| Layer | Transfers to a new dataset? | Why |
|---|---|---|
| Frozen encoders (Mantis-8M, DINOv2-B) | Yes, unchanged | Pretrained on broad data; channel-independent (force) and resolution-flexible (photos) |
| Level descriptors | Yes | Defined for any signal, in log or relative form; no machine parameters |
| LayerNorm + shared low-rank projection, channel-set pooling, descriptor embeddings | Yes, and fine-tunable | Learned once, shared across channels and photo types. A PCA basis would be refitted per dataset and every input dimension would change meaning, so a trained backbone could not be reused |
| Temporal fusion model (F1 or F0), relative-time bias | Yes | Reads tokens and cutting-time differences, not dataset ids |
| Health-indicator head | When the new dataset measures an HI | A supervised readout |
| RUL head | Reused; only the horizon H is set per dataset | RUL is always in seconds of operation |
| **Dataset adapter** | Supplied per dataset (§6.11) | Loading, trimming, the timeline, the failure time, channel and image descriptors, the descriptor standardization, H |

**Plugging in a new dataset** (for example MATWI: accelerometer, microphone,
Fx, Fy, Fz, insert photos):

1. Map its channels to descriptors.
2. Compute the same features.
3. Fit the adapter's descriptor standardization.
4. Set H, then fine-tune (or freeze) the projection and backbone.

Nothing in the model refers to a Nonastreda tool, pass, run, spindle speed or
PCA basis.

**The cost, stated.** PCA was a strong, free regularizer for 8 trajectories. The
learned projection has more parameters (about 50 k against about 12 k). Its
safeguards:

- the rank-16 bottleneck;
- dropout and weight decay;
- phase A's dense HI supervision;
- the projection rank in the sweep ({8, 32}).

### 6.11 Core and dataset adapter (D18)

**The contract.** The adapter turns a dataset into a list of units. Each unit
gives the core:

| Field | Content | Nonastreda adapter |
|---|---|---|
| Recordings | Operating start τ_r (s), duration D_r (s), sample rate, channels with descriptors (sensor type, axis, derivation) | Passes on the gap-aware axis (D12), causally trimmed (§5.2), duplicates attached to the latest member (D8); Fx, Fy, Fz = force × x/y/z × raw |
| Observations | Images (type, availability time); optional health indicator (value, unit, time); optional stage labels | Tool, chip and workpiece photos after each pass, with their crop and standardization; flank wear (µm); `tool_label` |
| Failure time τ* | Seconds of operation, or censored | The end of the confirmed TTF pass (D2) |
| Settings | Horizon H, baseline recordings B, segment length S, model sample rate | H = 1,500 s, B = 4, S = 10 s, 1 kHz |

**What stays in the adapter (Nonastreda only):**

- loading the MAT and CSV files, and their verification;
- the span-rule trim;
- the duplicate groups and their attachment;
- the gap-aware timeline;
- the five TTF rules;
- the photo crops and the workpiece standardization flag;
- flank-wear smoothing;
- the stage labels.

**Everything else is core and dataset-agnostic:** segmentation, RUL targets and
windows, channel magnitude, level descriptors, the encoders and caches,
tokens, the causal baseline, F1 and F0, heads, losses, augmentation, training,
calibration, evaluation, importance and statistics. None of them knows what a
pass, run, block, tool photo or flank wear is.

**Removed because they were Nonastreda-specific:**

| Removed | Replaced by |
|---|---|
| RUL in passes: its head, window, MAE_eq and the single-unit control | RUL in seconds only (D6) |
| "Completed passes" as a usage input | Operating time and seconds since the recording started |
| The Fz_rel channel | The generic drift descriptor of every channel |
| The ordinal wear-class head (CORAL) and oracle R2 | Nothing: they depended on Nonastreda's three classes |
| One F0 expert per photo type | One image expert over all image types |
| Physics features tied to the spindle (§6.2) | The generic level descriptors |

**Proof of transfer (test P22).** A synthetic toy adapter runs one fold end to
end through the unchanged core: 2 accelerometer channels at 2 kHz, one image
type, a different H, no gaps and no duplicates.

---

---

## 7. Training strategy

How one model is trained, step by step, and why the result can be trusted
when each fold has only 8 training tools.

### 7.1 The learning problem, and seven principles

| Supervision per fold (8 training tools) | Count | Character |
|---|---|---|
| RUL targets | About 3,000 prediction points, but only **8 failure events** | Points of one tool are strongly correlated; the effective sample size is closer to 8 than to 3,000 |
| Health indicator (here flank wear VB̃) | About 410 recorded passes, including those after the TTF | Dense, physical, comparable across tools |

A flexible model can memorize 8 life curves; Project 14's first RUL model did
so within one epoch.

| # | Principle | Implemented by |
|---|---|---|
| P1 | Learn little; borrow most | Frozen encoders, a shared low-rank projection, about 0.12 M trainable parameters |
| P2 | Dense supervision before sparse | Phase A learns the HI (here flank wear on about 410 recorded passes) before any RUL target (§7.4) |
| P3 | Give the model nothing that identifies a tool | No run or block codes; no per-dataset fitted basis; equal loss weight per tool; the token mode tested in Gate 2b (Stage 2: relative tokens did not make tools harder to identify) |
| P4 | Break the link between elapsed time and RUL | Trajectory augmentation (§6.8) |
| P5 | Judge only on held-out tools | Selection on the validation tool, calibration out-of-fold, test read after the freeze (D15) |
| P6 | Average away training noise | EMA weights, smoothed selection, seed ensembles, paired gate rules |
| P7 | Prove that it learns before trusting it | Sanity checks (§7.7) and per-fit diagnostics (§7.8) |

### 7.2 The lifecycle of one fit (fold *i*, seed *s*)

Each step writes into `runs/<study>/<arm>/fold-T<i>/seed-<s>/`.

1. **Fold setup.** Read the manifest; assert that test, validation and
   training are disjoint; seed Python, NumPy and PyTorch from (fold, seed).
   Writes `config.yaml` and `signature.json`.
2. **Transforms on the training tools:** the level-descriptor standardization;
   the VB̃ scale. Writes `transforms.npz`.
3. **Wear readouts with cross-fitting.**
   - Validation and test tools get a ridge probe fitted on all 8 training
     tools.
   - Each training tool gets a probe fitted on the other 7.
   - An in-sample readout would be nearly perfect: the network would learn to
     trust it and then fail on the noisier held-out readouts (stacking
     leakage).
   - Writes `readouts.parquet`.
4. **Tensors.** One sequence per tool, holding:
   - tokens, modality ids, availability times, masks;
   - targets y, VB̃ and the point weights.

   All 10 tools fit in GPU memory at once (about 10 × 600 tokens; tens of
   megabytes of features).
5. **Phase A, wear warm-up:** steps 1-100.
6. **Phase B, joint training:** steps 101-500 at most, with the EMA restarted
   at step 101.
   - Every 10 steps, evaluate the selection metric (§4.3).
   - Stop after 10 evaluations without improvement of the smoothed score.
7. **Restore the selected EMA weights.** Writes `model.pt` and
   `selection.json`.
8. **Predict** every point of the validation, test and training tools.
   Writes `predictions.parquet` (test rows sealed until the freeze, §4.6).
9. **Diagnostics** (§7.8). Writes `diagnostics.json`, then `DONE`.

Per fold, after all seeds, `rulfuse aggregate` does four more steps:

10. **Inner fits for calibration** (when the calibration method needs them,
    §9). 4 inner folds of 2 training tools each. Each inner fit trains on 6
    tools for the outer fit's selected number of steps (no inner selection),
    then predicts the 2 held-out tools.
11. **Ensemble:** mix the seeds' predictive distributions (§9).
12. **Calibrate** the ensemble (§9). The optional causal filter is applied
    after calibration.
13. **Score** the validation tool (and the test tool after the freeze).
    Writes `fold-summary.json`.

### 7.3 One optimizer step

1. **Assemble the batch:** 8 training tools × 4 augmented copies = 32
   sequences. Each copy gets trajectory augmentation, modality dropout,
   channel dropout, token dropout and feature noise (§6.8).
2. **Forward** through the causal model. One pass gives μ, σ, ν and the HI
   nowcast at every force token: about 300-500 predictions per sequence.
3. **Point weights** for point i of tool j:

   $$w_{i}=\frac{v_{i}}{\sum_{i'\in j} v_{i'}},\qquad v_{i}=\begin{cases}1 & \mathrm{RUL}_{i}\le H\\ w_{\mathrm{far}}=0.3 & \text{otherwise.}\end{cases}$$

   - Every tool weighs the same whatever its life length: T10 has 526
     pre-TTF points and T5 188.
   - The healthy plateau does not dominate.
   - Points after the failure carry only the HI loss.
4. **Loss** (§6.7) with the phase weights of §7.4.
5. **Update.**
   - Backward; clip the gradient norm at 1.0; AdamW step; update the EMA
     (decay 0.99).
   - Log every loss term, the gradient norm, the learning rate and F0's
     weights α_m.

Cost: about 0.05-0.3 s per step on a CPU, so a fit takes 1-3 min; the pilot
(J1) measures it.

### 7.4 Curriculum and schedules

| Phase | Steps | Losses (weights) | Learning rate | Purpose |
|---|---|---|---|---|
| A: HI warm-up | 1-100 | HI 1.0 (skipped when a dataset has no HI) | Linear warm-up over 20 steps to the base rate | The backbone learns a unit-invariant health state before it sees the 8 failure events |
| B: joint | 101-500 | MSE 1.0, consistency 0.1, NLL ramped 0 → 0.5 over steps 101-150 (mean detached); HI 0.3 | Cosine, base rate → 1e-5 | RUL from the wear state and the history |
| C: calibration | After training | None | – | §9 |

- The base learning rate is 1e-3, or the Gate-2 choice.
- Selection happens only in phase B. Phase A has a fixed length.
- Option in Gate 2: no phase A (Rule A).
- **F0:** deep supervision. The loss is the fused loss plus 0.5 × the mean of
  the experts' own losses, so each expert stands alone (for the weights and
  for a missing modality).

### 7.5 Hyperparameters

| Step | What | Fits |
|---|---|---|
| Defaults | Fixed in Appendix A before any run | – |
| Gate 2a | Learning rate {3e-4, 1e-3, 3e-3} for each of F1 and F0 (Rule B within a family); then F1 against F0 at their best (Rule B) | 2 × 3 × 10 folds × 3 seeds = 180 |
| Gate 2b | Options on the winner, no trajectory augmentation and no phase A (each by Rule A); token mode relative only or absolute only against the default both (peers, Rule B) | 4 × 30 = 120 |
| Gate 2b control | Last recording only (reported, not a decision: your specification is the full history) | 30 |
| Gate 2b control | Without the memory (D21) and without the monotonic hinge (D22), each reported, not a decision: both are your specification | 2 × 30 = 60 |
| Sweep | One factor at a time on the winner, each by Rule A: d {32, 96}; dropout {0.1, 0.3}; projection rank {8, 32}; ρ_max {0.1, 0.3}; w_far {0.1, 1.0} | 10 × 30 = 300 |

There is no per-fold tuning: one validation tool per fold is too noisy to
tune on (OmniFuser F4).

### 7.6 Making the result reliable

| Source of unreliability | Countermeasure |
|---|---|
| Seed noise | 3-seed ensembles for decisions and for the final model (D19) |
| Noisy selection of one step | EMA weights; smoothed whole-life selection; patience. A fixed-step variant (no selection; the median selected step) is reported |
| Long lives dominating the loss | Equal weight per tool |
| The healthy plateau dominating | w_far = 0.3 |
| Unstable variance learning | Mean detached in the NLL; NLL ramp; σ floor; ν ≥ 3; calibration after training |
| Tool-to-tool shift | Token mode chosen in Gate 2b; trajectory, modality and channel dropout |
| One validation tool per fold | Decisions from 10 folds × 3 seeds with paired rules (§4.4) |
| Calibrating on the tool used for selection | Out-of-fold calibration on the 8 training tools; the validation tool then checks coverage independently (§9) |
| Stacking leakage of the HI readout | Cross-fitted readouts |

### 7.7 Sanity checks (exit of Stage 4; laptop, cached features)

| Check | Expected result | What it shows |
|---|---|---|
| 1. Overfit one tool, regularization off | Training MAE < 10 s within 300 steps | Plumbing and capacity |
| 2. TTFs permuted among the training tools | Validation MAE no better than B2 | No leakage path to the labels |
| 3. Usage-only model B3 (no trajectory augmentation) | Within 20 s of B2 | The loop reproduces a known optimum |
| 4. Oracle token: the causally smoothed measured HI of the last recording as an input | Validation MAE well below B2, toward R1 | The model uses a strong signal when it exists |
| 5. Synthetic trajectories: simulated wear with known RUL and noise | MAE near the noise floor; 90 % intervals cover 0.88-0.92 | Learning and uncertainty end to end |
| 6. Truncation invariance (P11) | Identical outputs | Causality |
| 7. The same seed twice on a CPU | Identical weights | Determinism |

### 7.8 Per-fit diagnostics: learning or memorizing?

Written to `diagnostics.json` and plotted by `rulfuse report --diagnostics`.

1. **Learning curves** per loss term, training against validation, with the
   selected step. A warning fires when the optimum lies in the first 3
   evaluations while the training loss keeps falling (memorization).
2. **Validation MAE against B2** on the same tool; folds that do not beat B2
   are flagged.
3. **Usage sensitivity s_U.**
   - Delete 20 % of the validation tool's recordings *before* each evaluation
     point, keeping the baseline recordings.
   - The true RUL at that point does not change, but the elapsed time drops by
     Δ.
   - s_U = (increase of the predicted RUL) / Δ. A value of 0 means the model
     reads the sensors only; 1 means it counts time only (B1 has s_U = 1).
   - Reported for the model and for B3.
4. **Modality reliance** during training: validation MAE with each
   modality masked, every 50 steps.
5. **Calibration over steps:** validation coverage of the raw intervals.
6. **Head health:** the share of μ saturated at 0 or H; the share of σ at
   its floor; gradient norms; F0's weights.
7. **Seed spread:** the sd of validation MAE across seeds per fold. Above
   30 s, the regularization gets a closer look.

### 7.9 Failure modes and responses

Responses are gate options decided on validation tools, never reactions to
test scores.

| Symptom | Likely cause | Response |
|---|---|---|
| s_U near 1; predictions track B1 or B2 | Usage shortcut | ρ_max 0.3 (sweep); check the force probe of the Stage-3 feature check |
| Validation optimum within the first 30 steps of phase B while training falls | Memorizing 8 lives | Smaller d or projection rank, more dropout, phase A on |
| T5 (shortest life) predicted at the cap | Plateau dominates; few fast-wearing training tools | Lower w_far; trajectory deletion |
| σ at its floor; coverage far below nominal | Over-confident spread | Ensemble, out-of-fold calibration |
| Validation coverage fine, one test tool poor | That tool is unlike the others | Report it per tool |
| F0's photo weights near 0 | Photos not informative to the model | Check the readout quality (LoRA fine-tuning is a documented alternative, §6.2) |
| NaN or exploding loss | Learning rate; the NLL too early | Clipping, the NLL ramp, the Gate-2a learning rate |

### 7.10 Optional components

None remain.

- The former stretch goals, per-fold PatchTST pretraining and LoRA fine-tuning of
  the photo encoder, moved to the documented alternatives (§6.2) with the
  encoder simplification (D16).
- No encoder is trained in this project.

### 7.11 How a study runs, end to end

1. Write the study's configs (`configs/<study>/*.yaml`); each inherits the
   decisions so far.
2. `rulfuse manifest --study <study>` writes the TSV of (arm, fold, seeds).
3. `sbatch --array=0-N%4 slurm/array.sbatch runs/<study>/manifest.tsv`. Each
   task runs §7.2 for its seeds and skips finished fits.
4. `bash slurm/status.sh` until every fit is `DONE`.
5. `rulfuse aggregate --study <study>` writes the ensembles, calibration,
   validation scores, the paired comparison and `decision.json` (§4.4).
6. The decision and its numbers go into `PROJECT_LOG.md`; the next study
   inherits it.

Local runs use `--local`: sequential, with threads from `rulfuse resources`.

---

## 8. Baselines and reference models

Every table shows these, in seconds.

| ID | Model | Deployable | Purpose |
|---|---|---|---|
| B0 | Window-aware constant H/2 | No | How easy the window is (412 s) |
| B1 | Mean training TTF − elapsed time, capped | Yes | Pure usage (513 s) |
| B2 | E[min(τ* − τ, H) \| τ* > τ] over the training tools. Intervals from the empirical quantiles of min(τ*_j − τ, H) over the training tools still alive at τ | Yes | Best usage-only rule (468 s); the S1 and S2 reference |
| B3 | Learned usage-only model: the chosen head on U features, no trajectory augmentation | Yes | A learned B2; v(∅) candidate (§11) |
| B4 | Ridge on the level descriptors (absolute and relative; mean of the last 3 segments) + usage features → capped RUL | Yes | A generic condition-monitoring regression |
| B5 | The image HI readout (here the tool photos' wear readout) + degradation extrapolation (F5 machinery, §6.5) | Yes | "Read the image and extrapolate" |
| B6 | OmniFuser-RUL (D20): force + admitted photo types + usage through OmniFuser's encoders and C2F fusion, a GRU memory and the loss with the monotonic penalty; one full run, 3-seed ensemble, points only | Yes | What a published multimodal fusion architecture achieves on the same task and inputs |
| R1 | Oracle: the *measured* HI, causally isotonic-smoothed, + the same extrapolation | No | Upper reference: what a perfect image reader would give |

**Degradation extrapolation (B5, R1)** (generic, for any dataset with an HI).

- Fit HI(τ) = α + β·τ^γ (γ ≥ 1) to each training unit's smoothed HI against
  operating time. The prior over (α, log β, log γ) is Gaussian, fitted to the
  training units' triples.
- At time τ, importance-weight 5,000 prior draws by the likelihood of the unit's
  HI history (noise sd from the training residuals).
- RUL samples = the first crossing of θ − τ, capped at H. θ is the training
  units' median HI at failure (here VB̃ ≈ 112 µm).

---

## 9. Uncertainty and calibration

1. **Predictive distribution.** Aleatoric: the Student-t heads. Epistemic:
   the seed ensemble.
   - The mixture has mean avg μ_m and variance
     avg(σ_m² ν_m / (ν_m − 2)) + var(μ_m).
   - Its quantiles are found numerically (bisection on the mixture CDF), not
     by a Gaussian approximation.
2. **Calibration methods** (Gate 3, Rule B), one scale *c* applied to every
   member's σ:

   | Method | Fitted on | Note |
   |---|---|---|
   | (a) Validation scale | The validation tool's window | Simple, but the same tool chose the step, so it is optimistic |
   | **(b) Out-of-fold scale (default candidate)** | Out-of-fold predictions of the 8 training tools from the 4 inner fits (§7.2 step 10) | 8 tools of calibration data. The validation tool becomes an independent coverage check before any test score |
   | (c) Cross-conformal | The same out-of-fold residuals, normalized \|y − μ\|/σ; the 90 % quantile | Distribution-free under exchangeability between tools. The coverage on the test tools is measured, not assumed |

3. **Causal RUL filter** (Gate 3, Rule A).
   - A Kalman filter per unit. The state is RUL; the dynamics are
     x_{j+1} = x_j − Δ_j + w_j, with Var w_j = q·Δ_j. The measurement is the
     calibrated ensemble's μ_j, with its variance.
   - q is fitted on the out-of-fold predictions.
   - It is fully causal, and smooths the flicker that true RUL never shows
     (slope −1).
   - Predictions at the cap are passed through unfiltered.
4. **Gate 3** compares the head (Student-t, quantile) × the calibration
   method (a, b, c) by the validation Winkler score. The constraint is
   validation coverage within [0.85, 0.95]. The filter is then
   tested by Rule A.

---

## 10. Evaluation

### 10.1 Where and how

- Every model predicts at every point of a tool in one causal pass. The
  truncation-invariance test proves that this equals predicting point by
  point.
- The headline uses the window (RUL ≤ H). Whole-life metrics (capped targets)
  and recording-end-only metrics come second.
- Force-missing passes (D8) and the points right after T4's and T2's gaps are
  also shown separately.

### 10.2 Metrics (in seconds; also divided by H, to compare datasets)

| Group | Metrics |
|---|---|
| Point | MAE, MSE, RMSE; bias (mean of ŷ − y); MAE / H |
| Safety | Late-prediction rate: share of points with ŷ − y > H/15 (100 s). Asymmetric PHM-style score s = e^{−d/0.17} − 1 for d < 0, e^{d/0.13} − 1 for d ≥ 0, with d = (ŷ − y)/H |
| Prognostic (Saxena et al.) | α-λ accuracy: share within ±20 % of the true RUL at λ = 0.25, 0.5, 0.75 of the window. Prognostic horizon: the earliest time after which predictions stay within ±2H/15 (±200 s) until failure |
| Uncertainty | Coverage (PICP) and width (MPIW) at 50, 80, 90, 95 %; Winkler score; CRPS; NLL; reliability curve |
| Stability | Monotonicity violation: mean \|Δμ + Δτ\| between consecutive points |
| Decision | Replace the unit when the lower 90 % bound ≤ r₀. Over a grid of r₀: missed failures (units run past τ*) against wasted life. Compared with fixed-schedule replacement at quantiles of the training failure times |

### 10.3 Aggregation and statistics

- **Two summaries, always labelled:**
  - the mean ± sd over the 10 test tools of each tool's ensemble metric;
  - the pooled value over all window points.
- T4 has only 20 window points, so it weighs little pooled and is noisy per
  tool. Both views matter.
- **95 % intervals** by bootstrap over tools (Project 14
  `stats.unit_bootstrap`).
- **Comparisons:** paired Wilcoxon signed-rank over the 10 test tools, Holm
  across the comparisons of one table. With n = 10, the smallest possible
  one-sided p is 0.001; the 0.05 level needs W ≤ 10.
- **Breakdowns:** error by RUL bin (0-300 s, …, 1,200-1,500 s) and by position
  inside the recording (OmniFuser's phase-locking lesson).
- **Inference modes (D23).** Every table is computed in the current mode (primary)
  and in the context mode (`rulfuse.inference`). The modes are compared on the same
  tools:
  - context − current per metric, with a tool-bootstrap 95 % interval;
  - the window-MAE gain, with tools better and worse and a two-sided Wilcoxon test.

  The context mode is better if it lowers the mean window MAE by at least 10 s
  (§4.4) with an interval above 0, the current mode under the mirror condition,
  otherwise neither. `docs/OMNIFUSER_RUL_PLAN.md` §7 gives the definition and its
  validity in production.

### 10.4 Figures

1. Per tool: true RUL, predicted mean, 50 % and 90 % bands;
   the window shaded; photo labels as a strip; gaps marked.
2. Error and coverage against true RUL.
3. Reliability curves.
4. α-λ plot.
5. Decision curve.
6. Shapley bars with intervals.
7. Modality and channel importance over the horizon.

---

## 11. Modality-importance study

**Players:** F (force), T (tool photo), C (chip photo), and W (workpiece photo) only if it passes the leakage probe. Stage 2 expects it not to: then 3 players and 7 non-empty coalitions.
**Usage U is in every coalition** (a factory always knows the usage).

### 11.1 Retraining Shapley (primary)

- Train the frozen design on each non-empty subset: 7 × 10 folds × 3 seeds =
  210 fits with 3 players (450 with 4).
- **The empty coalition** is the better, on validation, of B2 and B3 (no
  trajectory augmentation). Trajectory augmentation deliberately destroys
  usage information, so a U-only model trained with it would understate
  v(∅) and inflate every sensing modality.
- Value: v(S) = −window MAE (s) on the test tools (after the freeze, D15).

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\,\big[v(S\cup\{i\})-v(S)\big].
$$

- **Report:**
  - φ_i per fold, the mean with tool-bootstrap intervals, and the share of
    the total gain over v(∅);
  - pairwise Shapley interactions (for example F × T);
  - the same for the interval score;
  - leave-one-out and add-one values, which come free from the 16 coalitions.

### 11.2 Test-time masking (the model's reliance)

- On the final model (trained with modality dropout, which includes the
  all-masked state), mask each subset at test time.
- Compare with §11.1: a modality can be used when present yet be replaceable.

### 11.3 Time-resolved importance

- **F0:** the weights α_m (signals, images, usage) over the RUL horizon.
- **F1:** integrated gradients and occlusion over history tokens. How far
  back does the model look? Does the force matter more near failure?
- **Image staleness:** replace the latest images with ones 1, 2 or 3
  observations older (Project 14's staleness test).

### 11.4 Sensing without usage

- A recent-window variant (last 4 recordings, U off) with F, with T, and
  with all four.
- Compared with B2. This separates what the sensors know from how long the
  tool has been used. Its usage sensitivity s_U should be near 0.

### 11.5 Force channels

All four force channels are input (§5.4).

1. **Test-time channel masking** on the final model (trained with channel
   dropout): the change in MAE when each channel, or each pair,
   is masked.
2. **Retraining check:** the final design without Fz, under the
   paired test. Does Fz add wear information or only drift?

Reported per channel: the mean change with tool-bootstrap intervals, and its
dependence on the RUL horizon.

---

## 12. Sensitivity and robustness studies

Each arm reruns the frozen design, 10 folds × 3 seeds, after the freeze.

| Tier | Study | Arms |
|---|---|---|
| 1 | TTF rule | Literal first dulled; smoothed wear ≥ 110 µm |
| 1 | Time axis | Recorded passes only; T1's leading run counted |
| 1 | Duplicates | Attach the recording to every member |
| 1 | Force channels | Without Fz (shared with §11.5) |
| 1 | Missing modality at test time | Each modality absent; 50 % of photos missing at random (no refits) |
| 2 | TTF rule | `tool_label` = 3; dulled from here on |
| 2 | Horizon | H = 1,000 s; H = 2,000 s (T5's window is then nearly its whole life) |
| 2 | Duplicates | OmniFuser's deduplication |
| 2 | Tokens | Absolute; both (only if not already settled in Gate 2b) |
| 2 | Trimming | OmniFuser's non-causal 5 %-of-pass reference |
| 2 | Segment length | 5 s, 20 s (new Mantis extraction, Ruche job J2) |

Tier 1 must run. Tier 2 runs in the order listed, as time allows.

---

## 13. Reuse from OmniFuser and Project 14, and lessons learned

### 13.1 What to reuse

Vendor the code, with a provenance header naming the file and the commit
(OmniFuser `e3b9985`, branch `Ruche-OmniFuser-Training`; Project 14
`master`). Do not import it as an editable dependency: OmniFuser's `data.py`
changed under a dependant before (Project 14 log, week 6). A test compares
the vendored loader with OmniFuser's whenever both are present.

| From | File: objects | Used for |
|---|---|---|
| OmniFuser | `data.py`: `RECORD_ID`, `_read_csv`, `_decode_record_id`, `_nonastreda_span_bounds` and constants, the checks of `load_nonastreda`, `leave_one_tool_out_folds`, `ToolFold`, `ForceNormalizer`, `load_photo`, `ImageBank`, `IMAGE_MEAN`, `IMAGE_STD`, `nonastreda_source_signature` | §5 |
| OmniFuser | `spectrogram.py` | Encoder SP |
| OmniFuser | `__init__.py` `_guard_windows_wmi`; `resources.py`; `tracking.py`; `model_bundle.py`; `cross_validation.py` (fold driver, summaries, pooling, resume signatures); `train.py` (interruption-safe resume) | Engineering |
| OmniFuser | `slurm/env.sh`, `job.sbatch`, `chain.sh`, `status.sh`, `checks.sbatch`; `TRAINING_RUNBOOK_RUCHE.md` §3-§7 | §16 |
| OmniFuser | `scripts/eda.py` (figure style), `scripts/end_to_end_trace.py` | Stage 1, documentation |
| Project 14 | `encoders.py`: `MantisEncoder`, `DinoV2Encoder` (switched to `facebook/dinov2-base`), `hf_revision`; `preprocess.py`: `level_descriptors`, `slow_summary` | §6.2 |
| Project 14 | `features.py`: `FeatureStore`, `extract_images` | §5.8 |
| Project 14 | `metrics.py`: `student_t_nll`, `interval_scale`, `phm2012_score`; `stats.py`: `unit_bootstrap`; `losses.py` (alternatives only) | §9, §10 |
| Project 14 | `rul.py`: `relative_to_start`; `planb.py`: `_schedule`, `_calibrate`; `gate2.py`: `_paired`, `decide` | §5.9, §8, §4.4 |
| Project 14 | `explain.py` (occlusion, deletion curves); `leakage.py`; `resources.py` `claim_resources` | §11, §15 |
| Project 14 | `slurm/prefetch_weights.sh`, `slurm/cpu.sbatch`, offline Hugging Face cache in `env.sh`; `docs/RUCHE_RUNBOOK.md` | §16 |

### 13.2 Lessons learned, and how this plan applies them

| # | Lesson (source) | Applied as |
|---|---|---|
| L1 | Same-pass photos leak the label (OmniFuser F1) | D9, plus a test |
| L2 | The run number is a 71 % shortcut (F2) | No run or block codes; usage explicit and switchable; B1-B3 everywhere |
| L3 | Fz drifts (F3) | All channels input; the generic drift descriptor (change since the recording started); Fz's contribution measured |
| L4 | One validation tool is noisy; replicates differ by up to 0.017 (F4, F14) | Smoothed whole-life selection, seed ensembles, Rule A with 7 of 10 wins |
| L5 | The cross-tool gap (training 0.88, test 0.67), not under-fitting | Frozen encoders, a shared low-rank projection, trajectory augmentation, about 0.12 M parameters; token mode tested in Gate 2b |
| L6 | Frozen ImageNet features read wear poorly; the crop helps; fine-tuning helps most | Edge crop, DINOv2-B features, wear readout; LoRA documented as an alternative |
| L7 | Absolute embeddings plus elapsed time memorize the training units (Project 14) | Trajectory augmentation; s_U diagnostic; token mode tested in Gate 2b (in Stage 2, absolute features identified tools *less* than relative ones) |
| L8 | Longer history did not help with 9 units (Project 14 XJTU-SY) | "Last pass only" control, reported |
| L9 | Spectrograms of the raw force gave OmniFuser's best forecasts | Documented as an alternative encoder; the level descriptors keep the force level that normalized encoders drop |
| L10 | C2F and the image GAT gained nothing over concatenation; late fusion was hard to beat (Project 14) | Finalists F1 and F0; C2F documented as an alternative |
| L11 | Labels are noisy and wear is non-monotone (F12) | Confirmed TTF; isotonic targets; causal smoothing for inputs; TTF sensitivity |
| L12 | Error phase-locked to pass boundaries (OmniFuser report §6.3) | Prediction at every segment; error by position in the pass |
| L13 | Laptop: WMI hangs, other jobs hold the CPU; bare `python` hung | WMI guard; `claim_resources`; the project's own venv |
| L14 | Ruche's 24 h limit and interruptions | Resumable fits, arrays that skip finished fits, `chain.sh` |
| L15 | **New:** recorded passes are not the physical history (unrecorded runs) | D12, with a wear-jump check per gap |
| L16 | **New:** whole-pass preprocessing leaks the pass length into the pass | Causal trim and time features (D14), with tests |
| L17 | **New:** calibrating on the tool that chose the step is optimistic | Out-of-fold calibration; validation coverage as an independent check |
| L18 | **New:** per-input normalized encoders cannot see absolute level or trend (Project 14 `preprocess.py`) | Level descriptors in every channel vector (§6.2) |
| L19 | **New:** PCA bases and machine-specific features tie a model to one dataset | No PCA in the model; generic descriptors; channel descriptors (§6.10) |
| L20 | **New:** passes, runs and named photo types exist only in Nonastreda | RUL in seconds; one generic core plus a small adapter (D18) |

---

## 14. Repository and engineering

### 14.1 Where things live

| What | Where | Why |
|---|---|---|
| Code, configs, docs, tests | `C:\Users\ali_a\OneDrive\Desktop\RUL Model` (git) | Small; synced |
| Raw data copy (force, labels, tool, chip, work), stores, caches, runs | `RULFUSE_DATA_ROOT` = `D:\RULModel` | Gigabytes of memory-mapped files must not sync through OneDrive (locks, slow sync) |
| Python environment | `D:\RULModel\.venv`, Python 3.11 or 3.12 | The encoder libraries (momentfm, mantis-tsfm, chronos-forecasting) lag behind Python 3.14, which OmniFuser's venv uses. The same minor version is used on Ruche |
| Hugging Face weights | `$WORKDIR/hf_cache` on Ruche only (none on the laptop) | Offline loading in jobs |

### 14.2 Layout

```text
RUL Model/
  pyproject.toml            package "rulfuse"; extras [test, eda, dev, encoders]
  configs/                  data/nonastreda.yaml, final.yaml, gate2/, sweep/, gate3/, shapley/, sensitivity/
  rulfuse/
    __init__.py             WMI guard (vendored)
    config.py               dataclasses; YAML I/O; signature()
    core/                   dataset-agnostic (D18): never imports rulfuse.adapters
      schema.py             Unit, Recording, Channel (descriptors), Observation, failure time, settings
      segments.py           S-second segmentation, the segment store
      targets.py            RUL in seconds, cap H, windows, masks
      channels.py           magnitude of 3-axis groups; the 7 level descriptors
      normalization.py      descriptor standardization, causal baseline (Scaler/PCA kept for B4 and analysis)
      folds.py              leave-one-unit-out with the next unit as validation
    adapters/
      nonastreda/           everything Nonastreda-specific
        load.py             MAT/CSV loading, verification, duplicate groups (vendored OmniFuser checks)
        trim.py             causal span rule (and OmniFuser's, for the sensitivity arm)
        timeline.py         gap-aware axis, D8 attachment, photo availability
        failure.py          the 5 TTF rules -> failure time; flank-wear HI; stage labels
        images.py           image types: tool crop, chip, workpiece standardization
      toy/                  synthetic adapter for the transfer-contract test (P22)
    features/
      encoders.py           Mantis-8M and DINOv2-B wrappers (vendored from Project 14)
      cache.py              FeatureStore (vendored), signatures
      probe.py              feature check, cross-fitted HI readouts, leakage probe
    models/
      tokens.py             LayerNorm, low-rank projections, channel-set pooling, descriptor and
                            time embeddings, causal baseline, relative-time bias
      fusion.py             F1 token fusion, F0 experts (usage, signals, images) + gPoE
      heads.py              Student-t RUL (s), quantile, HI nowcast
      memory.py             CausalMemory: GRU/LSTM over block means, causal (D21; built 2026-11-06)
      losses.py             monotonic hinge, slope consistency (D22; built 2026-11-06)
    degradation.py          HI extrapolation used by B5 and R1
    augment.py              trajectory augmentation, modality, channel and token dropout
    losses.py
    train.py                fit one (fold, seed): phases, EMA, selection, resume
    uncertainty.py          mixture, calibration (a, b, c), Kalman filter
    baselines.py            B0-B5, R1
    evaluation.py           §10 metrics
    importance.py           Shapley (retraining, masking), staleness, IG/occlusion, channels
    stats.py                bootstrap, Wilcoxon + Holm (vendored)
    cross_validation.py     driver, aggregation, test-score seal (§4.6)
    tracking.py, resources.py, bundle.py   (vendored)
    cli.py                  rulfuse <command> --dataset nonastreda
  scripts/                  eda_rul.py, end_to_end_trace.py, make_figures.py
  slurm/                    env.sh, job.sbatch, cpu.sbatch, checks.sbatch, features.sh, stage3.sh,
                            chain.sh, status.sh, prefetch_weights.sh (array.sbatch: Stage 4)
  tests/                    §15.2
  docs/                     EDA_RUL.md, END_TO_END.md, RUCHE_RUNBOOK.md, RESULTS.md, archive/
```

The Stage 0-2 modules (`rulfuse/data/*`) move into this layout in Stage 3
(§17.2). Their tests move with them and must stay green.

### 14.3 Commands

```bash
rulfuse verify-data                                             # hashes, counts, groups, gaps
rulfuse build-force-store                                       # the MAT file once, memory-mapped
rulfuse build-store   --dataset nonastreda                      # adapter -> generic segment store
rulfuse extract-features --kind signal --encoder mantis8m       # Ruche J2 (GPU)
rulfuse extract-features --kind image  --encoder dinov2b        # Ruche J2 (GPU)
rulfuse feature-check                                           # Ruche J3: go/no-go, readouts, leakage
rulfuse baselines                                               # B0-B2 in seconds
rulfuse manifest      --study gate2a && sbatch --array=0-59%4 slurm/array.sbatch runs/gate2a/manifest.tsv
rulfuse aggregate     --study gate2a                            # ensembles, calibration, decision.json
rulfuse freeze        --config configs/final.yaml              # D15; records the hash
rulfuse aggregate     --study final --after-freeze
rulfuse report        --runs runs/final runs/baselines runs/shapley runs/sensitivity --out runs/report
rulfuse predict       --model runs/final/fold-T3 --tool T3      # 3-seed ensemble
```

### 14.4 Outputs of every fit

| File | Content |
|---|---|
| `config.yaml`, `signature.json` | Resolved configuration; data and feature signatures |
| `metrics.jsonl` | Per evaluation step |
| `predictions.parquet` | Per point: unit, recording, segment, τ; y, RUL (s), μ, σ, ν, quantiles; F0's experts and weights; masks; `split` (test rows sealed before the freeze) |
| `selection.json`, `diagnostics.json` | §7.2, §7.8 |
| `model.pt`, `DONE` | Weights; completion marker |

### 14.5 Code quality

- pytest, ruff and pyright, as in OmniFuser.
- Deterministic seeds (`torch.use_deterministic_algorithms` where
  available).
- Every config field is part of the signature.
- `docs/END_TO_END.md` is regenerated by `scripts/end_to_end_trace.py` after
  every substantial pipeline change (your standing instruction).

---

## 15. Verification and validation, part by part

Three levels apply to every part.

- **Tests** (synthetic data) run in CI in under a minute.
- **Real-data checks** (`@pytest.mark.data`) skip when the data is absent.
- **Scientific validation** asks whether the part does its job, not only
  whether the code matches the plan.

A part is done only when all three pass.

### 15.1 The matrix

| Part | What can go wrong | Tests (synthetic) | Real-data checks | Scientific validation | Accept when |
|---|---|---|---|---|---|
| **P1 Loading** | Mis-parsed IDs; label mismatch; missing or corrupt photos | ID parsing; MAT-CSV mismatch raises; non-finite force raises; missing image raises | 512 passes; labels agree; chip and workpiece hashes match `MANIFEST.json`; 132 groups, 134 non-latest members; vendored = OmniFuser on shared fields | `verify-data` gives the same manifest hash on the laptop and on Ruche | All equal |
| **P2 Causal trim** | Reference uses the future; real cutting cut away | Idle lead-in, ramp and spike passes give the expected bounds; the start is unchanged when the pass is truncated after 6 s | Start index within 200 samples of OmniFuser's on ≥ 95 % of passes; plots of every disagreement | The 4 flagged passes of `EDA.md` §5 behave as documented | ≥ 95 % agreement; disagreements reviewed |
| **P3 Timeline** | Gaps miscounted; wrong duplicate attachment; wrong durations | Synthetic tool with an interior gap, a leading gap and a duplicate group: positions, `force_present`, durations, τ monotone; the leading gap is not counted | T2 +4 and T4 +16 unrecorded passes; 134 force-missing passes, ≥ 127 of them block 3; photo availability = end of its pass | Wear jump across each gap / per-run wear rate ≈ gap length in runs (T4 5.6, T2 1.6; `EDA_RUL.md`) | Counts exact; jump ratios documented |
| **P4 Segmentation** | Lost or double-counted samples; boundaries that depend on the future | Remainder rule; segments tile the trimmed pass exactly; sub-windows cover every sample; boundaries depend only on samples already seen | 3,747 pre-TTF points; per-tool counts of §2.2 | – | Exact |
| **P5 Labels** | Wrong TTF; RUL arithmetic; caps; a TTF inside a gap | Each rule on hand-made sequences (flips, ties, sequence end); RUL in seconds; caps and masks; isotonic non-decreasing; causal isotonic uses only the past | The TTF table of §2.3 and the windows of §2.2 exactly | Smoothed wear at the TTF in 109-123 µm for every tool | Exact |
| **P6 Folds** | Tool in two roles | Disjoint; cyclic; each tool tested once | – | – | Exact |
| **P7 Normalization** | Statistics from validation or test tools; non-causal baseline | Poison test: extreme values in validation and test tools leave the scalers and PCA unchanged; perturbing later segments leaves b_u(t) unchanged | – | Reported, not a gate: the test tool's shift and the tool identifiability for absolute against relative features (Stage 2: shift 0.31 against 0.35 sd; identification 23 % against 35 %, so the premise was not confirmed beyond static offsets; the token mode moved to Gate 2b) | Poison and causality tests pass |
| **P8 Encoders and caches** | Wrong checkpoint; non-determinism; mixed caches; a lost force level | Shapes; same input → same output; the signature changes with checkpoint or preprocessing; a cache with a foreign signature is refused; descriptors on synthetic signals (known RMS, kurtosis, slope) | Duplicated recordings → identical embeddings; the Mantis scale check (a window and its 2× copy); offline loading on Ruche (J0) | Descriptors carry the level when Mantis does not | All pass |
| **P9 Feature check, readouts and leakage probe** | Probe leakage; in-sample readouts; photo features that encode the stage label | Ridge recovers known coefficients; a cross-fitted readout of a tool never uses that tool's labels; PCA fitted on training only | Readout MAE on held-out tools reported per photo type | Force (Mantis + descriptors) and each kept photo type beat the training-mean VB predictor; shuffling VB within training tools collapses the probe to the mean | Leakage tests pass; H beats the mean; stage-step leakage probe: a stage boundary is the largest run-boundary step in ≤ 3 of 10 tools (`rulfuse.audit`) |
| **P10 Tokens and time** | D_p or the future reaching an input | Usage features at (r, s) do not change when the recording's later samples or its total duration change; image tokens sit at their availability time; the relative bias depends only on Δτ | Spot-check of 20 random points against the timeline table | – | All pass |
| **P11 Models F1, F0** | Look-ahead; mask bugs; expert bugs | **Truncation invariance** (outputs at t unchanged when later tokens are removed or replaced by noise); **no photo leak** (changing pass *p*'s photo changes nothing up to the end of pass *p*); missing-modality masks; gPoE with one expert equals that expert; F0 with all sensing masked equals the U expert; tiny overfit | – | Sanity checks 1, 4, 6 (§7.7) | All pass |
| **P12 Heads and losses** | Range errors; NLL moving μ; unit imbalance | μ within [0, H]; ν ≥ 3; the NLL gradient never reaches μ; consistency loss 0 for slope −1; scaling by H | – | On synthetic Student-t data the head recovers σ within 10 % | All pass |
| **P13 Augmentation** | Corrupted labels; photos detached | Trajectory augmentation keeps the TTF pass, recomputes targets, never touches the baseline passes, keeps photos and wear on their passes; dropout rates match configuration; channel dropout zeroes the right block | Histogram of augmented TTFs (wider than the real 19-47) | Usage sensitivity s_U falls with trajectory augmentation on (Gate 2b) | Tests pass; s_U reported |
| **P14 Training loop** | Selection bugs; non-resumable; non-deterministic | Phase schedule; EMA restart at step 101; selection picks the smoothed minimum; resume after a kill gives identical weights; same seed → same weights on CPU | – | Sanity checks 2, 3, 5, 7; learning curves without the memorization warning | All pass |
| **P15 Ensemble, calibration, filter** | Wrong mixture; leaky calibration; non-causal filter | Mixture moments and quantiles against Monte Carlo; the scale reaches the target coverage on synthetic data; inner folds never contain the outer validation or test tool; conformal coverage on exchangeable synthetic data; the filter on a synthetic linear-decreasing truth lowers the error and is causal | – | Out-of-fold calibration: validation coverage within [0.85, 0.95], checked before any test score | All pass |
| **P16 Baselines** | Formula or protocol errors | B2 on a toy case equals a hand computation; B5 and R1 extrapolation on a synthetic power-law tool recover its crossing | B0-B2 reproduce §2.7 (412 / 513 / 468 s) | R1 beats every deployable model. If it does not, the extrapolation is broken | Exact; R1 best |
| **P17 Metrics and statistics** | Wrong formulas | Every metric against hand-computed values; Student-t CRPS against Monte Carlo; Wilcoxon and Holm against SciPy; bootstrap reproducible by seed | – | – | Exact |
| **P18 Importance** | Shapley errors; unfair v(∅) | A synthetic additive game recovers its values; efficiency Σφ = v(N) − v(∅); in an additive synthetic model, masking ≈ retraining | – | The signs of retraining and masking agree for the top modality; a sensing-only model's s_U is near 0 | All pass |
| **P19 Sensitivity configs** | An arm silently equal to the default | Each arm's config changes exactly the intended field; label arms produce the TTFs of §2.3 | – | – | Exact |
| **P20 Ruche pipeline** | Lost or duplicated fits; offline failures | Manifest generation; `DONE` skip; resume after a simulated kill; `status.sh` parsing on a fake tree | J0 checks; pilot J1; one fold on the laptop and on Ruche: same MAE within 1e-3 relative (CPU equal, GPU tolerance) | – | All pass |
| **P21 Report** | Numbers not reproducible | Golden-file test: a tiny synthetic run's report equals the stored one | Deleting `runs/report` and rerunning gives identical tables (hash) | – | Identical |
| **P22 Transfer contract** | Nonastreda leaking into the core | `rulfuse.core`, `features` and `models` import nothing from `rulfuse.adapters` (an import-graph test); a synthetic toy adapter (2 accelerometer channels at 2 kHz, one image type, H = 600 s, no gaps or duplicates) runs one fold end to end | The Nonastreda adapter's output validates against the schema | – | Passes without any core change |

### 15.2 Test files

| File | Parts |
|---|---|
| `test_loader.py` | P1 |
| `test_trim.py` | P2 |
| `test_timeline.py` | P3 |
| `test_segments.py` | P4 |
| `test_labels.py` | P5 |
| `test_folds.py` | P6 |
| `test_normalization.py` | P7 |
| `test_encoders.py`, `test_cache.py` | P8 |
| `test_probe.py` | P9 |
| `test_tokens.py` | P10 |
| `test_causality.py`, `test_models.py` | P11 |
| `test_heads.py`, `test_losses.py` | P12 |
| `test_augment.py` | P13 |
| `test_training.py`, `test_pipeline_smoke.py` | P14 |
| `test_uncertainty.py` | P15 |
| `test_baselines.py` | P16 |
| `test_metrics.py`, `test_stats.py` | P17 |
| `test_importance.py` | P18 |
| `test_configs.py` | P19 |
| `test_slurm.py` | P20 |
| `test_report.py` | P21 |
| `test_schema.py`, `test_transfer_contract.py` | P22 |

---

## 16. Training on Ruche

### 16.1 What runs where

| Work | Where |
|---|---|
| Development, tests, small checks, the Stage-4 sanity checks on copied caches | **Laptop CPU**: check idle CPU first (`rulfuse resources`) |
| Stores, pretrained encoder features (Mantis-8M, DINOv2-B), the feature check | **Ruche** (your instruction of 2026-11-05): jobs J0, J2, J3. The laptop keeps no model weights |
| Fit arrays (Gates 2-3, sweep, final, Shapley, sensitivity) | Ruche `gpua100`, or a CPU partition if the queue is long (the models are small) |
| Aggregation and report | Laptop or `cpu_short` |

### 16.2 One-time setup

**The full procedure is `docs/RUCHE_RUNBOOK.md`** (written from OmniFuser's and
Project 14's Ruche runbooks). In short:

1. `ssh $Ruche`; `cd $WORKDIR` (500 GB, **not backed up**); the `~/.conda`
   symlink to `$WORKDIR/.conda` (home quota 50 GB).
2. Code: `git clone https://github.com/taanakoum/RUL-Model.git` (or the archive
   from `scripts/pack_for_ruche.ps1 -Code`).
3. Data: `scripts/pack_for_ruche.ps1` tars the raw copy (about 1.2 GB) with its
   sha256; `scp` it; `sha256sum -c`; unpack into `$WORKDIR/rul_data/raw`.
4. Environment:

   ```bash
   module load anaconda3/2023.09-0/none-none
   conda create -n rulfuse python=3.11 -y && source activate rulfuse
   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
   pip install --no-cache-dir -e ".[test,encoders]"
   ```

5. `slurm/env.sh` loads the module and activates `rulfuse`; sets
   `RULFUSE_DATA_ROOT=$WORKDIR/rul_data` and `HF_HOME=$WORKDIR/hf_cache`; sets
   `HF_HUB_OFFLINE=1` inside jobs, and `PYTHONUNBUFFERED=1`.
6. `bash slurm/prefetch_weights.sh` **on the login node**: Mantis-8M and
   DINOv2-B, config and safetensors only (about 380 MB).

### 16.3 SLURM scripts

| Script | Content |
|---|---|
| `job.sbatch` | One command on one GPU: `gpua100`, 8 CPUs, 24 h, `--export=NONE` (OmniFuser's) |
| `cpu.sbatch` | The same on a CPU partition (Project 14's) |
| `array.sbatch` (new) | `--array=0-N%4`: task *i* runs row *i* of a manifest TSV (arm, fold, seeds); finished fits are skipped; `%4` respects the 4-GPU limit per user |
| `chain.sh` | Only for jobs that can exceed 24 h |
| `status.sh` | Jobs; finished, running and missing fits per study; logs with errors |
| `checks.sbatch` | Job J0: data verification, stores, baselines, offline weight loading, all tests |
| `features.sh`, `stage3.sh` | Job J2's body; the Stage-3 submission chain (J0, J0g → J2 → J3) |
| `prefetch_weights.sh` | Login-node download of the two checkpoints |

### 16.4 Job plan

Estimates assume about 1.5 min per fit; the pilot measures the real cost.

| Job | Stage | Content | Tasks | Fits | Estimate |
|---|---|---|---|---|---|
| J0 | 3 | Checks (`cpu_short`): verify-data, force and segment stores, baselines, offline weights, all tests | 1 | – | 10-20 min |
| J0g | 3 | GPU check (`gpu_test`) | 1 | – | 1 min |
| J2 | 3 | Feature caches (`gpua100`): Mantis-8M signals, DINOv2-B photos, scale and determinism checks | 1 | – | 10-30 min |
| J3 | 3 | Feature check (`cpu_short`): go/no-go, readouts, leakage probe | 1 | – | 2-10 min |
| J1 | 4 | Pilot: one fold, F1 on the cached features | 1 | 3 | 30 min |
| J5 | 5 | Gate 2a: 2 families × 3 learning rates × 10 folds | 60 | 180 | 4-5 GPU-h |
| J6 | 5 | Gate 2b: 4 options + 3 controls (last recording; no memory; no hinge) × 10 folds | 70 | 210 | 5-6 GPU-h |
| J7 | 5 | Sweep: 10 variants × 10 folds | 100 | 300 | 7-8 GPU-h |
| J8 | 6 | Gate 3: 2 heads × 10 folds, each with 4 inner fits per seed | 20 | 300 | 7-8 GPU-h |
| J10 | 7 | Final: 10 folds × 3 seeds + 4 inner fits each | 10 | 150 | 3.5 GPU-h |
| J11 | 8 | Shapley: 7 coalitions × 10 folds (15 if the workpiece passes) | 70 (150) | 210 (450) | 5 (11) GPU-h |
| J12 | 8 | Sensitivity tier 1 (5 refit arms) / tier 2 (up to 9 arms + features) | 50 / 90 | 150 / 270 | 4 / 8 GPU-h |
| J13 | 9 | Report | 1 | – | 15 min |
| B6 | after J3 | B6 OmniFuser-RUL, the full run: caches, then 10 folds × 3 seeds (`b6-train`, `slurm/omnifuser_rul.sh`) | 1 | 30 | 1.5-3 GPU-h |
| B6-R | after B6, and from 2026-12-01 | B6's report (`b6-report`, `slurm/omnifuser_results.sh`; test results from the freeze or with `--unseal`) | 1 | – | 5 min |

The total is about 40-50 A100-hours for RUL-Fuse (the Stage-3 feature job adds under one), plus 1.5-3 for baseline B6's single run. Submit arrays in the evening; run
`status.sh` each morning.

### 16.5 Safe practice

- Fits are resumable. Re-running a task skips `DONE` fits and finishes an
  interrupted one from its last checkpoint.
- Manifests are committed with their study.
- Retrieval: `rsync -av --exclude '*.pt' ruche:$WORKDIR/RUL/runs/ $RULFUSE_DATA_ROOT/runs/`.
  Only the final checkpoints are exported (`rulfuse export`).
- `squeue -u $USER`; `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`.

---

## 17. Stages

### 17.1 Overview

```
                  W1     W2     W3     W4     W5     W6
                  Nov 4  Nov 11 Nov 18 Nov 25 Dec 2  Dec 9  Dec 16
S0 Foundations    ███
S1 Data audit      █████
S2 Data pipeline    ███████
S3 Features, check      █████
S4 Training engine        ████████
S5 Fusion, G2                   ████████
S6 Uncertainty, G3                    ████████ ◆ freeze Dec 1
S7 Final runs                                 ███
S8 Analysis                                    ████████
S9 Report                                            ████████
```

**Critical path:** S0 → S2 → S4 → S5 → S6 → S7 → S8 → S9.

- S1 must settle the time axis (D12) before S2's labels are final.
- S3 (the core/adapter refactor on the laptop; features and the feature check on
  Ruche, jobs J0-J3) runs beside S4 and must finish before S5 starts.

**Minimum viable result**, if time runs short: S0-S4, then F1 only in S5, then
the Student-t head with out-of-fold calibration in S6, then S7 and Shapley
(S8).

**Drop order:** tier-2 sensitivity → the sweep → F0 → the quantile head.

### 17.2 Stage by stage

#### Stage 0: Foundations (Wed Nov 4 to Fri Nov 6)

- **Goal:** a working skeleton on the real data.
- **Tasks:**
  - `D:\RULModel` with a Python 3.11 or 3.12 venv;
  - copy the raw data and the chip and workpiece photos there;
  - git init (§19);
  - `pyproject.toml`; the vendored modules with provenance headers; ruff,
    pyright, pytest;
  - `rulfuse verify-data`;
  - check that the DINOv3 and Chronos access and APIs work.
- **Outputs:** the package skeleton, `verify-data` output, and
  `PROJECT_LOG.md` started.
- **Verification:** P1; the vendored loader equals OmniFuser's.
- **Exit:** P1 accepted; CI green; your answers to §19 questions 1-2 (or
  their defaults).

#### Stage 1: Data audit and RUL EDA (Thu Nov 5 to Mon Nov 9)

- **Goal:** settle every data fact the labels depend on.
- **Tasks**, written up in `docs/EDA_RUL.md`, with `scripts/eda_rul.py`
  turning the scratchpad scripts into the project's own:
  - the unrecorded runs and their wear jumps (§2.5);
  - the duplicates and the force-missing pattern;
  - causal trim against OmniFuser's trim (P2);
  - the five TTF rules;
  - per-tool wear curves with TTF and gaps marked;
  - force indicators (relative RMS, band energy) and photo statistics against
    RUL;
  - B0-B2;
  - pass-duration differences (T9, T10).
- **Outputs:** `EDA_RUL.md` and its figures; the §2 numbers regenerated by
  script.
- **Verification:** P2 real-data checks; §2.2, §2.3 and §2.7 reproduced.
- **Exit:** your confirmation of D12, or the recorded axis chosen instead and
  the numbers redone; P2 accepted.

#### Stage 2: Data pipeline (Fri Nov 6 to Thu Nov 12)

- **Goal:** the segment store and labels, correct and causal.
- **Tasks:** `trim.py`, `timeline.py`, `segments.py`, `labels.py`,
  `folds.py`, `images.py`; the segment store; fold-level normalization; the
  causal baseline; `build-store`.
- **Outputs:** `interim/segments/<signature>/`.
- **Verification:** P3-P7.
- **Exit:** P3-P7 accepted; the store signature recorded.

#### Stage 3: Core/adapter refactor, features and the feature check (Tue Nov 10 to Sat Nov 14)

- **Goal:** a dataset-agnostic core behind a Nonastreda adapter; cached Mantis-8M and
  DINOv2-B features, checked and leak-tested, computed on Ruche (your instruction of
  2026-11-05). No encoder comparison.
- **Tasks:**
  - the core/adapter refactor of the Stage 0-2 code (D18): the generic
    schema, `rulfuse/adapters/nonastreda/`, the toy adapter and the
    transfer-contract test (P22). The existing tests move along and stay green;
  - the `encoders` extra (CPU torch, transformers, mantis-tsfm);
  - the vendored Mantis-8M and DINOv2-B wrappers;
  - the level descriptors (Project 14's `level_descriptors` and
    `slow_summary`);
  - the Ruche setup (`docs/RUCHE_RUNBOOK.md`, `slurm/`) and jobs J0, J0g, J2, J3
    (`bash slurm/stage3.sh`): the feature caches of 3,777 × 4 channel vectors and
    1,536 photos, then the feature check;
  - the full handcrafted set H, for baseline B4;
  - the feature check of §6.2: scale check, go/no-go, cross-fitted readouts,
    leakage probe.
- **Outputs:** `interim/features/force-mantis8m-<sig>/`,
  `interim/features/photo-dinov2b-<sig>/`, `runs/feature-check/report.json`.
- **Verification:** P8, P9, P22.
- **Exit:**
  - force is a go (otherwise the MOMENT-1-base fallback is logged and run);
  - the photo types are fixed by the leakage probe (decision 1, §17.3).

#### Stage 4: Training engine and sanity checks (Thu Nov 12 to Thu Nov 19)

- **Goal:** a trustworthy training loop before any real comparison.
- **Tasks:**
  - `tokens.py` (with the relative-time bias), F1 and F0, the heads, losses,
    augmentation, `train.py`;
  - `uncertainty.py` (mixture, calibration (a)-(c), the filter);
  - `baselines.py`, `degradation.py`, `evaluation.py`, `stats.py`;
  - the diagnostics; the seven sanity checks (laptop, on the caches copied
    from Ruche);
  - pilot job J1 (`slurm/array.sbatch` and manifests arrive here).
- **Outputs:** sanity-check report in `PROJECT_LOG.md`; baselines in
  `runs/baselines`.
- **Verification:** P10-P17.
- **Exit:** all seven sanity checks pass; P10-P17 accepted; pilot J1 timed.

#### Stage 5: Fusion and Gate 2 (Wed Nov 18 to Wed Nov 25)

- **Goal:** the fusion family and its options.
- **Tasks:**
  - J5 (Gate 2a: learning rates, then F1 against F0);
  - J6 (Gate 2b: options and the last-pass control);
  - J7 (sweep), each aggregated and decided by its rule.
- **Outputs:** `runs/gate2a`, `runs/gate2b`, `runs/sweep` with
  `decision.json`.
- **Verification:**
  - the diagnostics of every fit are reviewed: no memorization warnings left
    unexplained;
  - s_U reported;
  - the validation MAE of the winner below B2's on validation.
- **Exit:** decisions 2-7 logged.

#### Stage 6: Uncertainty, Gate 3 and design freeze (Tue Nov 24 to Tue Dec 1)

**In plain words.** By Stage 6 the model already predicts how long a tool has
left. Stage 6 makes its "give or take" trustworthy, then locks the design.

1. **Each prediction comes with a range.** The model outputs a most likely value
   and a spread, for example "≈ 640 s, 90 % between 420 and 900 s".
2. **Three copies vote.** The final model is trained 3 times with different random
   starts. Where the copies disagree, the range widens: the model is less sure.
3. **Check the ranges against reality.** A 90 % range must contain the true
   remaining life 90 % of the time, like a weather forecast whose "70 % chance
   of rain" days are rainy 70 % of the time.
   - The check uses tools the model did not train on: the 8 training tools
     through 4 extra inner trainings (each leaves 2 tools out), with the
     validation tool as an independent second check.
   - The test tool is never used.
4. **Stretch or shrink.** If the ranges are too narrow (say 70 % instead of
   90 %), one factor widens all of them until they are right, and likewise if
   they are too wide.
5. **Pick the best way to do it.** Two output types (a Student-t spread, or
   seven quantiles) and three ways of fitting the factor are compared, and the
   most accurate ranges on the validation tools win. An optional smoother then
   enforces what real remaining life does: it falls by exactly the time spent
   cutting.
6. **Freeze.** The chosen design is written to `configs/final.yaml`, its hash is
   recorded, and from then on nothing changes before the test scores are read.

**In detail.**

- **Goal:** calibrated intervals and a frozen design.
- **Tasks:**
  - J8 (heads × calibration methods, with the inner fits);
  - the filter, by Rule A;
  - write `configs/final.yaml`; `rulfuse freeze`.
- **Outputs:** `runs/gate3`; the frozen configuration and its hash.
- **Verification:** the P15 scientific check (validation coverage in band, both
  units); reliability curves on validation.
- **Exit:** decisions 8-10 logged; freeze recorded (D15). **No test score has
  been read up to here.**

#### Stage 7: Final runs and headline (Wed Dec 2 to Fri Dec 4)

- **Goal:** the headline result.
- **Tasks:**
  - J10;
  - `rulfuse aggregate --study final --after-freeze`;
  - S1 and S2 tests; decision curves; per-tool figures.
- **Outputs:** `runs/final`; the headline tables.
- **Verification:** S1 and S2 evaluated as pre-registered; R1 above every
  deployable model (P16).
- **Exit:** the headline tables are final.

#### Stage 8: Importance and sensitivity (Thu Dec 3 to Thu Dec 10)

- **Goal:** what each modality and channel contributes, and how robust the
  result is.
- **Tasks:** J11 (Shapley); the masking, time-resolved, staleness,
  sensing-only and channel studies; J12 (tier 1, then tier 2).
- **Outputs:** `runs/shapley`, `runs/sensitivity`.
- **Verification:** P18, P19.
- **Exit:** S3 evaluated; tier-1 sensitivity complete.

#### Stage 9: Report (Wed Dec 9 to Wed Dec 16)

- **Goal:** a reproducible write-up.
- **Tasks:**
  - `docs/RESULTS.md` (methods, results, importance, sensitivity,
    limitations, the decision log);
  - `docs/END_TO_END.md` regenerated;
  - figures; optional slides.
- **Outputs:** the report.
- **Verification:** P21 (S5).
- **Exit:** every reported number regenerates via `rulfuse report`.

### 17.3 Decision log

Each decision is appended to `PROJECT_LOG.md` with its numbers.

| # | Decision | Gate | Rule |
|---|---|---|---|
| 1 | Photo types that pass the stage-step leakage probe | Stage-3 feature check | A stage boundary is the largest run-boundary step in ≤ 3 of 10 tools |
| 2 | Learning rate of F1; of F0 | 2a | B |
| 3 | Fusion family (F1 or F0) | 2a | B |
| 4 | Trajectory augmentation | 2b | A |
| 5 | Phase-A warm-up | 2b | A |
| 6 | Token mode (relative, absolute, both) | 2b | B (peers, after the Stage-2 evidence) |
| 7 | Sweep adoptions (each factor, including the projection rank) | Sweep | A |
| – | (Control) last pass only | 2b | Reported, not decided |
| 8 | Head (Student-t or quantile) | 3 | B |
| 9 | Calibration method (a, b, c) | 3 | B |
| 10 | Causal RUL filter | 3 | A |

No encoder, derived-channel or LoRA decision remains (D16, D17).

---

## 18. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 10 tools give wide intervals; gaps between models are not significant | High | Paired tests; effect sizes with intervals; few pre-registered comparisons; both summaries |
| The model leans on usage and adds little over B2 | Medium-high | Trajectory augmentation; s_U diagnostic; B2 and B3 everywhere; sensing-only study |
| The force carries little RUL information (Project 14: flat force pathway) | Medium | Level descriptors with the Mantis embedding; both token halves; the channel study; the MOMENT fallback |
| Photo features stay weak | Medium | Crop, DINOv2-B features, wear readout; the chip photo as a second visual cue; LoRA documented as an alternative |
| The gap-aware axis is wrong (the runs were not cut) | Low (you confirmed it; the wear jumps agree) | The recorded-axis arm (tier 1) |
| T4's thin window (20 points) and T5's short life | High | Whole-life selection; both summaries; per-tool reporting |
| Label noise near the TTF (T3, T5) | Medium | Confirmed rule; TTF sensitivity arms |
| Out-of-fold calibration misses tool-specific spread | Medium | Validation coverage checked before the freeze; per-tool coverage reported |
| The 6-week scope slips | Medium | Minimum viable result and drop order (§17.1) |
| Ruche queue | Low-medium | CPU partitions; features need no Ruche; no gated checkpoint is used |
| A causality bug | Low after tests | P10, P11 tests; oracle and permutation checks |
| OneDrive locking or syncing data files | Medium if ignored | Data root on `D:` (§14.1) |

---

## 19. Open questions for you

Each has a default that the plan assumes until you answer.

1. **Unrecorded runs (D12).** *Answered 2026-11-04: yes, T2's run 10 and T4's
   runs 6-9 were cut but not recorded.* The gap-aware axis is final. T1's
   leading run is still not counted: it changes no RUL target, and it stays a
   tier-1 sensitivity arm.
1b. **Workpiece photos (Stage 1).** Their brightness and texture change in a
   step exactly when `tool_label` changes stage, in every tool
   (`docs/EDA_RUL.md` §6). *Answered 2026-11-04: the paper does not describe a
   change per stage.* The cause stays unknown, so the default stands:
   per-image standardization plus the Stage-3 leakage probe, and exclusion if
   the probe fails.
2. **Commits.** Should I commit to a new private GitHub repository, or will
   you commit personally, as in Project 14? *Default: changes left
   uncommitted, with proposed commit groups.*
3. **Ruche.** Is your account active, with the same `$WORKDIR`? Any GPU-hour
   limit? *Default: OmniFuser's setup; 65-80 A100-hours.*
4. **Cutting conditions.** T9's passes last about 81 s and T10's about 122 s.
   Does the paper give feed or speed per tool? *Default: not used.*
5. **Usage in the headline model.** Elapsed cutting time is an input (a
   factory knows it); the sensing-only result is reported separately.
   *Default: yes.*
6. **LoRA**: withdrawn (2026-11-04). With the fixed encoders (D16) it became
   a documented alternative.
7. **Deliverables:** a report only, or slides too? *Default: report and
   figures; slides optional.*

---

## Appendix A: default settings

| Setting | Default | Section |
|---|---|---|
| TTF rule | Confirmed first dulled (dulled, then ≥ 2 dulled among the next 3 recorded) | D2 |
| Time axis | Gap-aware; interior gaps counted at the tool's median pass duration; leading gaps not | D12 |
| Horizon (cap and window) | H = 1,500 s of cutting (a dataset setting) | D3, §3.2 |
| Selection and gate metric | MAE in seconds (whole life with the loss weights for selection; window for gates) | §4.3 |
| Segment | 10 s from the causally trimmed start, remainder < 5 s merged | D7, §5.2 |
| Trim reference | Median envelope over the first 5 s | §5.2 |
| Channels | Fx, Fy, Fz + \|F\| (magnitude of each 3-axis group), as a channel set with descriptor embeddings | §5.4, §6.3 |
| Duplicates | Attached to the latest member; the others force-missing | D8 |
| Photo crop (tool) | Rows 0.30-0.85, full width, 224 × 672 | §5.7 |
| Encoders | Mantis-8M (force, 512-sample sub-windows, mean + sd pooled) and DINOv2-B/14 (photos, CLS + 2 × 4 grid), frozen; 7 level descriptors per channel | §6.2 |
| Projection | LayerNorm + shared low-rank projection (rank 16) per modality; no PCA | §6.3, §6.10 |
| Tool baseline and token mode | First B = 4 recordings, causal running mean; token mode "both" (absolute + relative halves); readouts absolute | §5.9 |
| Fusion | F1 or F0 (Gate 2a) | §6.4 |
| Memory | GRU, hidden 64, over the force tokens (B6: 10-step blocks of 1 s steps) | D21 |
| F1 backbone | Causal transformer: 2 layers, d = 64, 4 heads, FFN 128, dropout 0.2, relative-time bias | §6.4 |
| Head | Student-t RUL in seconds: μ = H·sigmoid; σ floor 0.005·H; ν = 3 + softplus | §6.6 |
| Loss weights | MSE 1; NLL 0.5 (mean detached); consistency 0.5; monotonic hinge 1.0; HI 0.3 | §6.7, D22 |
| Point weights | Equal total per tool; w_far = 0.3 | §7.3 |
| Optimizer | AdamW, lr 1e-3 (Gate 2a), weight decay 0.05, gradient clip 1.0 | §7.3 |
| Schedule | Phase A 1-100 (warm-up 20); phase B 101-500, cosine; NLL ramp 101-150; EMA 0.99 restarted at 101; patience 10 evaluations | §7.4 |
| Batch | 8 training tools × 4 augmented copies | §7.3 |
| Augmentation | Trajectory (recordings) ρ ~ U[0, 0.2]; modality dropout 0.3 each (all together 0.05); channel 0.1; token 0.1; feature noise 0.1 | §6.8 |
| Seeds | 3 (gates and final) | §4.5, D19 |
| Calibration | Out-of-fold scale, 4 inner folds (Gate 3 decides) | §9 |
| Gate rules | A: ≥ 10 s (H/150) lower validation MAE and ≥ 7 of 10 wins; B: best mean | §4.4 |

## Appendix B: how the numbers were measured

All measurements were taken on 2026-11-04 with OmniFuser's virtual
environment (`../OmniFuser/.venv/Scripts/python.exe`). The scripts are in
this session's scratchpad and move to `scripts/eda_rul.py` in Stage 1.

- **Labels and durations.** Per-tool label sequences come from
  `Data/labels.csv`. Trimmed durations are `trim_end − trim_start` from
  OmniFuser's `experiment_results/eda/records.csv`, which uses OmniFuser's
  trim; the causal trim is checked in Stage 1.
- **Wear.** Isotonic wear by `scipy.optimize.isotonic_regression`.
- **Points.** Segment counts are round(D_p / 10 s), identical to D7's
  remainder rule.
- **Gaps.** Missing run numbers between a tool's first and last recorded run.
  Wear jumps are run means across the gap, against the median between
  consecutive recorded runs.
- **Duplicates.** Groups are collisions of (tool, run, samples, force sum);
  they match OmniFuser's 132 bit-identity groups. Photo hashes come from
  `D:\Project14\data\raw\nonastreda\MANIFEST.json`.
- **Baselines.** Computed at segment ends, on both axes, with the fold
  protocol D1 and the TTF of D2. B2 has the cap inside the expectation.

## Appendix C: references

- OmniFuser: arXiv:2511.01320; the reproduction in `../OmniFuser`
  (`FINAL_REPORT_REPRODUCTION.md`, `FIXES_AND_VALIDATION.md`,
  `FUTURE_WORK_AND_ANALYSIS.md`, `EDA.md`).
- Project 14: `../Multimodal AI for Predictive Maintenance` (`PROJECT_LOG.md`,
  `docs/PLAN.md`, `docs/RUCHE_RUNBOOK.md`, `pdmfm/`).
- Nonastreda: H. Truchan, Z. Ahmadi, *Data in Brief* (2025),
  doi:10.17632/m892d2wtzh.1.
- Prognostic metrics: Saxena et al., "Metrics for evaluating performance of
  prognostic techniques", PHM 2008; the C-MAPSS scoring function (2008).
- Deep ensembles: Lakshminarayanan et al., NeurIPS 2017. β-NLL: Seitzer et
  al., ICLR 2022. Split and cross-conformal prediction: Lei et al., JASA
  2018; Vovk 2015. Generalized product of experts: Cao & Fleet, 2014. Kalman
  filtering for prognostics: e.g. Saha & Goebel, 2008.
- Flamingo gated cross-attention: Alayrac et al., NeurIPS 2022. MulT: Tsai et
  al., ACL 2019. Shapley values: Lundberg & Lee, NeurIPS 2017.
- Encoders: MOMENT (Goswami et al., ICML 2024), Mantis (2025), Chronos (Ansari
  et al., 2024), AST (Gong et al., 2021), DINOv2 (Oquab et al., 2024), SigLIP 2
  (2025), DINOv3 (2025).
