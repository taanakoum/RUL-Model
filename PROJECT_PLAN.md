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
| D3 | Horizon | **k = 15 passes**; in seconds, 1,500 s of cutting (D6). Headline metrics use the points with RUL ≤ k. Training uses every pre-TTF point, with its target capped at k | You (seconds window derived) |
| D4 | Modalities | Force with **all three channels** Fx, Fy, Fz at 1 kHz, plus two derived channels (§5.4); tool photo; chip photo; workpiece photo. Workpiece photos are standardized per image and every photo type must pass the Gate-1 stage-step leakage probe (§6.2; `docs/EDA_RUL.md` §6) | You; leakage probe from Stage 1 |
| D5 | Timeline | 6 weeks, to 2026-12-16. Design freeze on 2026-12-01 (D15) | You |
| D6 | Output units | **Both, at every segment:** RUL in seconds of cutting and in passes, each with its own interval, from two heads on one shared backbone. Cutting time is spindle-engaged time. The idle time between passes is not recorded, so wall-clock time to failure cannot be predicted | You |
| D7 | Segment | 10 s of raw 1 kHz force, from the (causally trimmed) pass start. A remainder shorter than 5 s joins the previous segment | Default; §12 tests 5 s and 20 s |
| D8 | Duplicated force | A duplicated recording (§2.4) is attached to the **latest** pass of its group; the earlier members have missing force. This affects 134 of the 512 passes (26 %), almost always block 3 | Default (causally safe) |
| D9 | Photo availability | A photo of pass *p* is input only to predictions made after pass *p* ends | Default; removes OmniFuser's lag-0 leakage |
| D10 | Loss | MSE on the predicted mean (your request), plus a likelihood term for the spread with the mean detached, plus auxiliary wear and consistency terms (§6.7) | Default |
| D11 | Sign | RUL = t_TTF − t ≥ 0 | Convention |
| D12 | Time axis | **Gap-aware.** Positions follow (run, block). Runs missing *between* a tool's first and last recorded run count as unrecorded cutting, at the median duration of the tool's recorded passes before the gap: T2 run 10 (4 passes), T4 runs 6-9 (16 passes). T1's missing *leading* run 1 is not counted | Review (§2.5); supported by the Stage-1 wear-jump test (`docs/EDA_RUL.md` §2); **confirmed by you (2026-11-04)** |
| D13 | Fusion | **Two candidates: F1 token fusion and F0 late fusion (generalized product of experts).** The others are documented as alternatives (§6.5) | You |
| D14 | Causal inputs | Every input at a prediction point is computable from data recorded up to that point. This covers trimming, segmentation and the time features | Review |
| D15 | Test-score policy | No test-tool score is read before the design freeze (end of Stage 6, 2026-12-01). After the freeze nothing about the model changes; the headline, importance and sensitivity results are read from then on | Review |

---

## 1. Goal and success criteria

**Goal.** At the end of every 10 s force segment, predict how long the tool
has left before it is (confirmed) dulled. The prediction comes in **seconds of
cutting** and in **passes**. The model reads everything recorded so far:
every past force segment and every photo already taken. It outputs a
calibrated interval as well as a point value, in each unit. The project also
measures how much each modality and each force channel contributes.

**Success criteria.** These are fixed in advance and measured on the test
tools with the 5-seed ensemble of the frozen design.

| # | Criterion | Threshold |
|---|---|---|
| S1 | Accuracy | k-window MAE below the best *deployable* baseline B2 in both units: **3.82 passes and 468 s** (§2.7). One-sided paired Wilcoxon test over the 10 test tools, Holm-adjusted over the 2 units, adjusted p < 0.05. With 10 tools this needs about 8 or more tools improved, with small losses on the rest. The window-aware constant B0 (4.13 passes, 412 s) is reported alongside |
| S2 | Intervals | Pooled coverage of the nominal 90 % interval within [0.85, 0.95] in both units; per-tool coverage reported. Winkler interval score below B2's empirical intervals (§8) |
| S3 | Modality importance | Shapley values for force, tool, chip and workpiece, with 95 % bootstrap intervals over tools. The sign of the most important modality is resolved |
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
| Fz drifts | Sawtooth that resets every 1-3 passes; its level correlates with wear at −0.04 (F3) | All channels input (D4); a causal, offset-removed Fz_rel neutralizes the drift (§5.4); the channel study (§11.5) measures what Fz adds |
| Tool-to-tool force offset | T1's Fx sits at +1.15 sd of the training distribution (`EDA.md` §8) | Tokens carry the absolute level and the change from each tool's healthy start (§5.9). Relative-only removes static offsets such as T1's but not the cross-tool shift (Stage 2) |
| Frozen ImageNet features read wear poorly | Photo probe 0.61-0.67 accuracy against 0.95 possible; the crop helps; resolution alone does not | Edge crop, DINOv2 grid, wear-regression readout; LoRA as a stretch goal |
| Absolute embeddings plus elapsed time memorize units | Project 14 RUL GRU peaked after 1 epoch (`pdmfm/rul.py`) | Trajectory augmentation, small models, PCA; the token mode is decided in Gate 2b |
| Wear is not monotone | Flank wear decreases at 189 of 502 steps (F12) | Isotonic-smoothed wear targets; causal smoothing wherever wear is an *input* (R1) |
| Workpiece photos encode the stage label (found in Stage 1) | Mean brightness steps at the first run of `tool_label` stage 2 in 10 of 10 tools (96 % of its range); texture also at stage 3 (`docs/EDA_RUL.md` §6) | Per-image standardization; Gate-1 stage-step leakage probe for every photo type (§6.2); exclusion if it fails |
| Whole-pass preprocessing leaks within the pass | OmniFuser's trim uses 5 % of the pass length | Causal trim (§5.2) |
| Unrecorded runs | §2.5 | D12 |

### 2.7 The bar to beat

Settings: segment ends; gap-aware axis; fold protocol D1; the 8 training
tools of each fold.

| Predictor | Uses | MAE (passes) | RMSE (passes) | MAE (s) | RMSE (s) |
|---|---|---|---|---|---|
| B0: constant at half the window (7.5 passes; 750 s) | That the point lies in the window, which is the truth itself | 4.13 | 4.64 | **412** | 463 |
| B1: mean training TTF − elapsed time, capped | Elapsed time | 5.17 | 5.65 | 513 | 562 |
| B2: E[min(TTF − t, k) \| TTF > t] over the training tools | Elapsed time | **3.82** | 4.31 | 468 | 526 |

- **Same points as the model.** These baselines are computed at the segment
  ends where the model predicts, with the cap inside the expectation (the
  MSE-optimal usage-only predictor of the capped target).
- **In seconds, B0 beats B2.** The window-aware constant (412 s) beats the
  usage rule (468 s).
- **B2's empirical 90 % intervals under-cover:** 0.42 in passes and 0.50 in seconds
  (Winkler 49.4 passes, 5,388 s), the S2 reference.
- **Recorded axis, for reference:** B2 scores 4.01 passes and 480 s.

---

## 3. Task definition

### 3.1 Notation

For tool *u*, positions *p* = 1…N_u follow (run, block) on the gap-aware
axis. *p* = 4 (run − r₀) + block, where r₀ is the tool's first recorded run.

- D_p: the causally trimmed cutting duration of pass *p* (§5.2). An
  unrecorded pass gets the tool's median. τ_p = Σ_{q≤p} D_q is the cumulative
  cutting time at its end.
- Pass *p* is cut into segments *s* = 1…n_p (D7). Segment (p, s) ends at
  elapsed time e_{p,s} inside the pass.
- **Label time** in pass units: t_{p,s} = (p − 1) + e_{p,s} / D_p. **Time in
  seconds:** τ_{p,s} = τ_{p−1} + e_{p,s}.
- t_{p,s} contains D_p, which is unknown until the pass ends. It defines
  labels only, never inputs (§3.3).
- P*_u: the confirmed TTF pass. Failure is at the **end** of pass P*, when the
  dulled state is first observed: t_TTF = P*, τ_TTF = τ_{P*}.

### 3.2 Targets

$$
\mathrm{RUL}_{p,s} = P^\ast - t_{p,s}\ \text{(passes)},\qquad
\mathrm{RUL}^{\mathrm{sec}}_{p,s} = \tau_{P^\ast} - \tau_{p,s}\ \text{(s of cutting)},
$$
$$
y_{p,s} = \min(\mathrm{RUL}_{p,s},\,k),\ k=15;\qquad
y^{\mathrm{sec}}_{p,s} = \min(\mathrm{RUL}^{\mathrm{sec}}_{p,s},\,k^{\mathrm{sec}}),\ k^{\mathrm{sec}}=1{,}500\ \text{s}.
$$

**Both targets are predicted at every segment.**

- Seconds answer "how long can I keep cutting"; passes answer "how many more
  passes".
- The two differ across tools because pass length does.
- Inside a pass, the seconds target falls by exactly the elapsed cutting
  time.

**Points, windows and training.**

- **Prediction points:** every segment end of the recorded passes 1…P*:
  3,747 points. Points after the TTF carry no RUL target.
- **Evaluation windows:** passes: RUL ≤ 15 (1,338 points); seconds: RUL^sec
  ≤ 1,500 s (1,361 points). Inside its window each target equals the true
  RUL.
- **Training:** every pre-TTF point, with the capped (piecewise-linear)
  targets.
- **Auxiliary targets** (training tools only):
  - VB̃_p, the isotonic-smoothed flank wear of the current pass, over all
    recorded passes including those after the TTF;
  - optionally the ordinal wear class.

**Worked examples.**

- **T5** (P* = 19). At the end of pass 4, RUL = 15 and the window opens. At
  the end of segment 5 of pass 10 (record T5R3B2, 96.05 s long), t = 9 + 50/96.05
  = 9.52 and RUL = 9.48 passes. τ = 909.7 s and T5 fails at 1,827.0 s, so
  RUL^sec = 917.3 s, about 15 min of cutting. At the end of pass 19 both are 0.
  `scripts/end_to_end_trace.py` reproduces these numbers.
- **T4** (P* = 38). At the end of R5B4 (pass 20), RUL = 18 passes. The
  recorded axis would say 2.

### 3.3 What the model may read at point (p, s)

| Input | Available | Never |
|---|---|---|
| Force | Segments of passes < *p*, plus segments 1…*s* of pass *p* | Later segments |
| Photos (tool, chip, workpiece) | Of passes < *p* | Of pass *p* or later |
| Usage | Completed passes *p* − 1 (gap-aware), elapsed cutting seconds τ_{p,s}, elapsed seconds in the current pass e_{p,s} | D_p before the pass ends; t_{p,s} |
| Labels | None | `image_label`, `tool_label`, flank wear, gaps, overhang, run or block codes |

The causality tests of §15 (part P10, P11) enforce this table.

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
| Standardization, PCA, ridge probes (α by inner leave-one-tool-out) | The 8 training tools |
| Wear readouts used as training inputs | Cross-fitted on the training tools (§7.2) |
| Tool baseline (§5.9) | Each tool's own first passes, causally, including the test tool |
| Wear threshold θ and degradation prior (B5, R1) | The training tools |
| Step selection | The validation tool |
| Interval calibration | Out-of-fold predictions on the training tools (default), or the validation tool (§9) |
| Test tool | Prediction only |

### 4.3 Metrics for selection and decisions

**Pass-equivalent MAE.** Both units count equally:

$$
\mathrm{MAE}_{\mathrm{eq}} = \tfrac12\big(\mathrm{MAE}_{\mathrm{pass}} + \mathrm{MAE}_{\mathrm{sec}}/100\ \mathrm{s}\big),
$$

where 100 s = k_sec / k. Reports always give both units separately.

**In-fit step selection.**

- Every 10 steps, compute the validation tool's MAE_eq over its whole pre-TTF
  life, with the training-loss weights (§7.3), using the EMA weights.
- Keep the step with the lowest mean of 3 consecutive evaluations.
- Why the whole life and not only the k-window: T4, the validation tool of
  fold 3, has only 20 window points.

### 4.4 Design decisions: gates with rules fixed in advance

All gate scores come from the 3-seed ensemble, on the validation tools.

- **Rule A: adding complexity or switching an option on.** Adopt only if all
  three hold:
  - the mean validation k-window metric over the 10 folds improves by at
    least the gate's threshold;
  - the option wins on at least 7 of the 10 validation tools (the chance
    probability of 7 or more is 0.17);
  - neither unit worsens by more than 5 %.
- **Rule B: choosing among peers of similar cost.** Take the best mean. Within
  half the gate's threshold, take the cheaper or more interpretable one.

| Gate | Metric | Rule-A threshold |
|---|---|---|
| Gate 1 (encoders, derived channels) | Validation VB MAE of the ridge probe (µm) | 1.0 µm |
| Gate 2 (fusion, options, sweep) | k-window MAE_eq | 0.10 |
| Gate 3 (head, calibration, filter) | Winkler 90 % interval score, pass-equivalent, subject to validation coverage within [0.85, 0.95] | 3 % relative |

**Caveat.** Fold *i*'s validation tool is fold *i+1*'s test tool, so
cross-fold choices touch every tool once. This is the same as in OmniFuser
and Project 14. The plan keeps the decisions few (16, listed in §17.3) and
states the count in the report.

### 4.5 Seeds and ensembles

- Gates use the 3-seed ensemble; the final runs use the 5-seed ensemble.
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

**All three measured channels are always input:** Fx, Fy and Fz. Two derived
channels are added.

| Channel | Definition | Why |
|---|---|---|
| Fx, Fy, Fz | Raw, trimmed | Every measured channel (your instruction) |
| Fz_rel | Fz minus its mean over the pass's first second | Removes the drift's sawtooth level but keeps Fz's dynamics; causal |
| \|F\| | √(Fx² + Fy² + Fz²) | The resultant force, the classic wear indicator |

- **Per channel:**
  1. the encoder embeddings (mean and sd over the sub-windows);
  2. concatenated with that channel's physics features;
  3. standardized and PCA-reduced to 16 components (8 on the absolute values,
     8 on the values relative to the tool's healthy start; §5.9);
  4. the five blocks concatenated into an 80-dimensional force token.
- A channel can therefore be masked cleanly (channel dropout, §6.8; the
  channel study, §11.5).
- **Gate 1** decides only whether the two *derived* channels stay. The raw
  channels are never removed from the main model.

### 5.5 Segment

- 10 s windows from the trimmed pass start; a remainder under 5 s joins the
  previous segment. This gives 8 segments per pass on T9, 12 on T10 and 10
  elsewhere.
- Encoder sub-windows: 512 samples, the last aligned to the segment end so
  every sample is covered.
- Each segment stores: tool, pass, index, start and end sample, e_{p,s},
  t_{p,s} (label time only), τ_{p,s}, `force_present`.

### 5.6 Labels: `rulfuse/data/labels.py`

- The five TTF rules of §2.3, selected by config key `ttf_rule`. Each is
  applied to the recorded label sequence; a TTF can never fall in a gap.
- RUL in passes and in seconds; capped targets; evaluation and pre-TTF masks.
- VB̃ per pass: isotonic regression per tool over the recorded passes
  (`scipy.optimize.isotonic_regression`). It is a target only. The causal
  version (isotonic fit on the passes up to now) is used where measured wear
  is an *input* (oracle R1).

### 5.7 Images: `rulfuse/data/images.py`

Reuse OmniFuser's `load_photo` (crop) and `ImageBank`. All photos are
normalized with the ImageNet statistics.

| Photo | Preparation |
|---|---|
| Tool | Crop rows 0.30-0.85 at full width (the OmniFuser probe's best band). Resize to 224 × 672 (Project 14's setting), or tile into 3 × 224² tiles near native height |
| Chip | Resize to 448², or to 224² (Gate-1 candidates) |
| Workpiece | Resize to 224²; standardize each image's luminance (zero mean, unit variance) to remove the per-stage exposure step (`docs/EDA_RUL.md` §6) |

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
- **Feature caches:** `interim/features/<modality>-<encoder>-<signature>/`,
  holding an `(n_items, dim)` float16 array and an index.
  - The signature names the checkpoint revision and the preprocessing
    (Project 14's `FeatureStore`).
  - Duplicated recordings are encoded once, which saves 26 % of the encoder
    calls.

### 5.9 Normalization

- **Fold level.** Standardize each feature with the training tools' mean and
  sd. Then apply PCA, fitted on the training tools: 32 components per photo
  type and 16 per force channel in total, half on the absolute features and
  half on the tool-relative ones (token mode "both", below).
- **Tool level (causal baseline).**
  - The relative half of each token is taken against the tool's healthy start:
    $$\tilde z = z - b_u(t),\qquad b_u(t) = \text{mean of } z \text{ over the tool's force segments in its first } B=4 \text{ recorded passes, among those seen by } t .$$
  - Photos are taken relative to the tool's first photo.
  - The wear readouts (§6.2) stay **absolute**: a dull edge looks dull on any
    tool.
  - **Token mode (changed in Stage 2): "both" by default.** Every token carries an
    absolute half and a tool-relative half, at the same total size.
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
 │ force seg│...│ force seg│...│ force seg│  raw 1 kHz Fx, Fy, Fz (+ Fz_rel, |F|), 10 s
 └────┬─────┘   └────┬─────┘   └────┬─────┘
      ▼ per channel: frozen TS encoder on 512-sample sub-windows (mean+sd) ⊕ physics features
      ▼ standardize → per channel PCA(8) absolute ⊕ PCA(8) relative to the causal baseline → MLP → d=64
 photos of passes < p (tool | chip | workpiece)
      ▼ frozen image encoder on the crop → PCA(32) → minus the first photo → MLP → d=64
      ▼ + absolute wear readout VB̂ (cross-fitted ridge probe)
 ─────────────────────────────────────────────────────────────────────────────
 tokens ordered by availability time + modality embedding (+ usage features, switchable)
      ▼
 causal fusion model with a relative-time attention bias:
      F1 token fusion (one transformer)   |   F0 late fusion (experts + gPoE)
      ▼ at every force-segment token (= prediction point)
 heads: RUL in seconds and in passes, Student-t (μ, σ, ν) each · wear nowcast VB̂
      ▼
 5-seed ensemble → calibration (out-of-fold) → optional causal RUL filter
      → "≈ 640 s (90 %: 420-900 s) of cutting left; ≈ 6.6 passes (4.3-9.2)"
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

### 6.2 Encoders, screened in Gate 1

**Force** (per channel and 10 s segment):

| ID | Encoder | Dim per window | Notes |
|---|---|---|---|
| H | Physics features: mean, sd, RMS, peak-to-peak, skewness, kurtosis, crest factor; 16 band energies 0-500 Hz; energy at the 66.7 Hz spindle line and harmonics (Project 14 EDA F11); spectral centroid and entropy; period-15 synchronous-average amplitude | about 20 per channel | Always present; the cheap reference |
| MA | Mantis-8M (`paris-noah/Mantis-8M`) | 256 | Project 14 wrapper |
| MO | MOMENT-1 base / large (`AutonLab/MOMENT-1-base`, `-large`) | 768 / 1024 | Project 14 wrapper (base) |
| CH | Chronos encoder embeddings (`amazon/chronos-t5-small`, `pipeline.embed`) | 512 | New wrapper; check the API at install |
| AST | Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) | 768 | Project 14 wrapper |
| SP | Segment spectrograms (VLT-style STFT, FFT and recurrence maps, OmniFuser `spectrogram.py`) into frozen DINOv2-S | 384 | OmniFuser's best forecasting input |
| PT | Own PatchTST, masked-reconstruction pretrained per fold on the training tools | 128 | Project 14 `patchtst.py`; stretch goal |

**Photos** (per pass and photo type):

| ID | Encoder | Notes |
|---|---|---|
| EB0 | EfficientNet-B0, ImageNet (torchvision) | OmniFuser's encoder; the reference |
| D2-S/B | DINOv2 small or base: CLS plus patch grid pooled to 2 × 4, flattened | Project 14 wrapper |
| SG2 | SigLIP 2 base (`google/siglip2-base-patch16-224`) | New wrapper |
| D3 | DINOv3 ViT-S/B (gated; check access and licence in Stage 0) | Optional |
| HC | Handcrafted: HSV colour histograms (chip tint), LBP and GLCM texture (workpiece), edge-band gradient profile (tool) | Cheap reference |

**Gate 1 (CPU, minutes once the features are cached).**

1. For each modality and candidate, pool to one vector per recorded pass
   (force: the mean over the pass's segments). Standardize and apply PCA on
   the training tools.
2. Fit a ridge regression on the 8 training tools (α by inner
   leave-one-tool-out) to VB̃ of the pass. Score it on each fold's validation
   tool.
3. **Choose by validation VB MAE (Rule B).** A per-pass probe cannot
   integrate history, so the wear *state* is what a pass-level feature must
   carry. A snapshot probe of RUL mostly separates healthy from worn (Project
   14).
4. **Guard:** a ridge probe of capped RUL (with usage features) must not be
   worse than H's by more than 0.3 MAE_eq.
5. Concatenations of two encoders are candidates too.
6. The **derived force channels** (Fz_rel, |F|) are decided here by Rule A
   on VB MAE.
7. The chosen photo encoder's ridge probe becomes the **wear readout** VB̂
   (µm) of each photo. It is cross-fitted on the training tools (§7.2).
8. **Leakage probe for every photo type (added in Stage 1).**
   - Run `rulfuse.audit.stage_steps` on the chosen encoder's first 3 PCA
     components and on its wear readout.
   - A photo type passes if a `tool_label` stage boundary is its largest
     run-boundary step in **at most 3 of 10 tools**. The force and the tool
     photo's global statistics score 0-1 of 10; the raw workpiece brightness
     scores 10 of 10.
   - Workpiece photos are luminance-standardized per image before encoding.
   - A failing type leaves the headline model and the Shapley study and is
     reported as a leakage demonstration (`docs/EDA_RUL.md` §6).

### 6.3 Tokens and time

| Token | Content | Position |
|---|---|---|
| Force segment (p, s) | Projected force token (absolute and tool-relative halves); `force_present` flag. A force-missing segment uses a learned mask vector | Its end time τ_{p,s} |
| Photo (p, m), m ∈ {tool, chip, work} | Projected photo features (absolute and relative-to-first-photo halves) + the absolute wear readout VB̂ | Immediately after the last segment of pass *p*, so only from pass *p+1* on (D9) |

- **Usage features (U, switchable):** Fourier features of the completed
  passes *p* − 1 and of τ_{p,s}, plus e_{p,s}. They are added to every force
  token. The absolute time embedding is part of U and switches off with it.
- **Relative time:** each attention head adds a learned bias
  b_h(Δτ) = MLP_h(log(1 + Δτ / 10 s)), where Δτ is the cutting time between
  the query and the key. This gives every token, photos included, its age
  relative to the prediction point, without revealing absolute time.
- **Modality embedding** on every token.
- **Sensing-only control (§11.4):** a causal model over the full history can
  infer elapsed time from the token count, even with U off. The control
  therefore reads only the last 4 recorded passes, with U off.

### 6.4 Fusion: the two candidates (D13)

| ID | Design | Why it is a finalist |
|---|---|---|
| **F1 Token fusion** | One causal transformer over the interleaved sequence of §6.3: 2 layers, d = 64, 4 heads, FFN 128, pre-LN, dropout 0.2, relative-time bias, U features. The prediction is read at every force token | The most flexible. Attention can learn cross-modal timing, for example "the force has risen since the last photo" |
| **F0 Late fusion (generalized product of experts)** | Five experts, each with a dual-unit Student-t head (details below) | Robust to a missing modality. The weights α_m read directly as time-resolved modality importance. Project 14 found late fusion hard to beat on Nonastreda |

**F0's experts.**

- **U:** an MLP on the usage features.
- **F:** a causal transformer over the force tokens.
- **T, C, W:** a causal GRU (d = 32) over each photo type's tokens. Inside a
  pass, a photo expert's mean falls with the cutting time since its last
  photo.
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
| F5 Health indicator + extrapolation | Fused wear nowcast VB̂(t), then a degradation curve with a prior from the training tools; RUL = crossing of θ ≈ 112 µm | Not a fusion strategy. Its machinery is kept and used by baselines B5 and R1 (§8) | F0 and F1 lose to B5; then the fused nowcast + F5 becomes a candidate |
| TFT-style variable selection | Learned per-token feature gating | Overlaps with F0's weights | — |

### 6.6 Heads

| Head | Parameterization | Status |
|---|---|---|
| **Dual-unit Student-t** | Per unit u ∈ {s, pass}: μ_u = k_u·sigmoid(w_{μ,u}ᵀh); σ_u = softplus(w_{σ,u}ᵀh) + 0.005·k_u; ν_u = 3 + softplus(w_{ν,u}ᵀh) (finite variance) | Default |
| Quantile | 7 quantiles 0.05…0.95 per unit, non-crossing by cumulative softplus increments; pinball loss | Gate-3 comparator |
| Single-unit control | One seconds head; passes = μ_s / D̄, with D̄ the tool's mean recorded pass duration so far | Reported control |
| Wear nowcast VB̂ | Linear on h | Auxiliary task |
| Ordinal wear (CORAL) | 2 thresholds | Auxiliary, off |
| NIG (evidential), discrete hazard | Project 14 `losses.nig_*`; hazard over pass bins | Alternatives, not run: the ensemble already gives the epistemic part |

Intervals are clipped to [0, k_u].

### 6.7 Loss

$$
\mathcal L = \sum_{u\in\{\mathrm{s},\,\mathrm{pass}\}}\Big[\underbrace{\textstyle\sum_i w_{u,i}(\mu_{u,i}-y_{u,i})^2/k_u^2}_{\text{MSE (primary)}}
+ \lambda_{\text{nll}}\,\textstyle\sum_i w_{u,i}\,\mathrm{NLL}_t\!\big(y_{u,i} \mid \mathrm{sg}(\mu_{u,i}),\sigma_{u,i},\nu_{u,i}\big)
+ \lambda_{\text{cons}}\!\!\sum_{(i,j)}\!\frac{\big(\mu_{u,i}-\mu_{u,j}-(T_{u,j}-T_{u,i})\big)^2}{k_u^2}\,\mathbb 1[y_{u,i}<k_u]\Big]
+ \lambda_{\text{vb}}\,\mathrm{Huber}\big(\widehat{VB}-\widetilde{VB}\big)/s_{VB}
$$

- **Point weights** w_{u,i}: defined in §7.3.
- **Stop-gradient.** sg stops the gradient, so the mean is driven by MSE as
  you asked, and the spread learns from the likelihood without pulling the
  mean.
- **Consistency term.** T_u is the label time of a point: τ in seconds, t in
  passes. The term rewards the slope −1 that true RUL has between nearby
  points of the same tool (pairs at most 2 passes apart). In seconds it is
  exact even inside a pass. For F0 it applies to the fused output.
- **Defaults:** λ_nll = 0.5, λ_vb = 0.3, λ_cons = 0.1. MSE and MAE are always
  both reported (§10).

### 6.8 Regularization and augmentation (8 trajectories per fold)

1. Frozen encoders, PCA (§5.9), d = 64, dropout 0.2, weight decay 0.05.
2. **Trajectory augmentation (the main guard against the usage shortcut).**
   - Delete passes (probability ρ ~ U[0, ρ_max], ρ_max = 0.2) or duplicate
     them, between pass B+1 and P* − 1. Unrecorded passes may be deleted too.
   - Recompute positions, times, the TTF and every target.
   - With cached tokens this is index manipulation only. The tools then wear
     faster or slower per pass, which breaks the link between elapsed time
     and RUL.
3. **Modality dropout.**
   - F1: each sensing modality is dropped for the whole sequence with
     p = 0.3, independently; all four together with p = 0.05, so the U-only
     state is seen.
   - F0: the same probabilities drop experts.
4. Token dropout 0.1; Gaussian feature noise, sd 0.1 in standardized units.
5. **Channel dropout:** each force channel's block is zeroed with p = 0.1 per
   sequence.
6. Batching, optimizer, curriculum, selection and diagnostics: §7.

### 6.9 Size

| Part | F1 | F0 |
|---|---|---|
| Projections: force 80 → 64; 3 photo types 32 → 64 | about 12 k | about 12 k |
| Backbone(s) | about 70 k | force about 70 k + 3 photo GRUs about 20 k + U MLP about 5 k |
| Heads, gate | about 2 k | about 10 k |
| **Trainable total** | **about 0.1 M** | **about 0.12 M** |

---

## 7. Training strategy

How one model is trained, step by step, and why the result can be trusted
when each fold has only 8 training tools.

### 7.1 The learning problem, and seven principles

| Supervision per fold (8 training tools) | Count | Character |
|---|---|---|
| RUL targets | About 3,000 prediction points, but only **8 failure events** | Points of one tool are strongly correlated; the effective sample size is closer to 8 than to 3,000 |
| Current wear VB̃ | About 410 recorded passes, including those after the TTF | Dense, physical, comparable across tools |

A flexible model can memorize 8 life curves; Project 14's first RUL model did
so within one epoch.

| # | Principle | Implemented by |
|---|---|---|
| P1 | Learn little; borrow most | Frozen encoders, PCA, about 0.1 M trainable parameters |
| P2 | Dense supervision before sparse | Phase A learns wear from about 410 passes before any RUL target (§7.4) |
| P3 | Give the model nothing that identifies a tool | No run or block codes; PCA on the training tools; equal loss weight per tool; the token mode tested in Gate 2b (Stage 2: relative tokens did not make tools harder to identify) |
| P4 | Break the link between elapsed time and RUL | Trajectory augmentation (§6.8) |
| P5 | Judge only on held-out tools | Selection on the validation tool, calibration out-of-fold, test read after the freeze (D15) |
| P6 | Average away training noise | EMA weights, smoothed selection, seed ensembles, paired gate rules |
| P7 | Prove that it learns before trusting it | Sanity checks (§7.7) and per-fit diagnostics (§7.8) |

### 7.2 The lifecycle of one fit (fold *i*, seed *s*)

Each step writes into `runs/<study>/<arm>/fold-T<i>/seed-<s>/`.

1. **Fold setup.** Read the manifest; assert that test, validation and
   training are disjoint; seed Python, NumPy and PyTorch from (fold, seed).
   Writes `config.yaml` and `signature.json`.
2. **Transforms on the training tools:** standardization and PCA; the VB̃
   scale. Writes `transforms.npz`.
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

   All 10 tools fit in GPU memory at once (about 10 × 600 tokens × ≤ 80
   features).
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
2. **Forward** through the causal model. One pass gives μ, σ, ν (both units)
   and VB̂ at every force token: about 300-500 predictions per sequence.
3. **Point weights** for unit u and point i of tool j:

   $$w_{u,i}=\frac{v_{u,i}}{\sum_{i'\in j} v_{u,i'}},\qquad v_{u,i}=\begin{cases}1 & \mathrm{RUL}_{u,i}\le k_u\\ w_{\mathrm{far}}=0.3 & \text{otherwise.}\end{cases}$$

   - Every tool weighs the same whatever its life length: T10 has 526
     pre-TTF points and T5 188.
   - The healthy plateau does not dominate.
   - Points after the TTF carry only the wear loss.
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
| A: wear warm-up | 1-100 | VB 1.0 | Linear warm-up over 20 steps to the base rate | The backbone learns a tool-invariant wear state before it sees the 8 failure events |
| B: joint | 101-500 | Per unit: MSE 1.0, consistency 0.1, NLL ramped 0 → 0.5 over steps 101-150 (mean detached). Plus VB 0.3 | Cosine, base rate → 1e-5 | RUL from the wear state and the history |
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
| Gate 2b control | Last recorded pass only (reported, not a decision: your specification is the full history) | 30 |
| Sweep | One factor at a time on the winner, each by Rule A: d {32, 96}; dropout {0.1, 0.3}; PCA per photo / per channel {16/8, 64/32}; ρ_max {0.1, 0.3}; w_far {0.1, 1.0} | 10 × 30 = 300 |

There is no per-fold tuning: one validation tool per fold is too noisy to
tune on (OmniFuser F4).

### 7.6 Making the result reliable

| Source of unreliability | Countermeasure |
|---|---|
| Seed noise | 3-seed ensembles for decisions, a 5-seed ensemble for the final model |
| Noisy selection of one step | EMA weights; smoothed whole-life selection; patience. A fixed-step variant (no selection; the median selected step) is reported |
| Long lives dominating the loss | Equal weight per tool |
| The healthy plateau dominating | w_far = 0.3 |
| Unstable variance learning | Mean detached in the NLL; NLL ramp; σ floor; ν ≥ 3; calibration after training |
| Tool-to-tool shift | Token mode chosen in Gate 2b; trajectory, modality and channel dropout |
| One validation tool per fold | Decisions from 10 folds × 3 seeds with paired rules (§4.4) |
| Calibrating on the tool used for selection | Out-of-fold calibration on the 8 training tools; the validation tool then checks coverage independently (§9) |
| Stacking leakage of the wear readout | Cross-fitted readouts |

### 7.7 Sanity checks (exit of Stage 4; laptop, physics features)

| Check | Expected result | What it shows |
|---|---|---|
| 1. Overfit one tool, regularization off | Training MAE_eq < 0.1 within 300 steps | Plumbing and capacity |
| 2. TTFs permuted among the training tools | Validation MAE_eq no better than B2 | No leakage path to the labels |
| 3. Usage-only model B3 (no trajectory augmentation) | Within 0.2 passes and 20 s of B2 | The loop reproduces a known optimum |
| 4. Oracle token: causally smoothed measured wear of the last pass as an input | Validation MAE well below B2, toward R1 | The model uses a strong signal when it exists |
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
   - Delete 20 % of the validation tool's passes *before* each evaluation
     point, keeping the baseline run.
   - The true RUL at that point does not change, but the elapsed time drops by
     Δ.
   - s_U = (increase of the predicted RUL) / Δ. A value of 0 means the model
     reads the sensors only; 1 means it counts time only (B1 has s_U = 1).
   - Reported for the model and for B3.
4. **Modality reliance** during training: validation MAE_eq with each
   modality masked, every 50 steps.
5. **Calibration over steps:** validation coverage of the raw intervals.
6. **Head health:** the share of μ saturated at 0 or k_u; the share of σ at
   its floor; gradient norms; F0's weights.
7. **Seed spread:** the sd of validation MAE_eq across seeds per fold. Above
   0.3, the regularization gets a closer look.
8. **Unit consistency:** μ_s / μ_pass against D̄. A large disagreement means
   one head learned a shortcut the other did not.

### 7.9 Failure modes and responses

Responses are gate options decided on validation tools, never reactions to
test scores.

| Symptom | Likely cause | Response |
|---|---|---|
| s_U near 1; predictions track B1 or B2 | Usage shortcut | ρ_max 0.3 (sweep); check the Gate-1 VB probes for force information |
| Validation optimum within the first 30 steps of phase B while training falls | Memorizing 8 lives | Smaller d or PCA, more dropout, phase A on |
| T5 (shortest life) predicted at the cap | Plateau dominates; few fast-wearing training tools | Lower w_far; trajectory deletion |
| σ at its floor; coverage far below nominal | Over-confident spread | Ensemble, out-of-fold calibration |
| Validation coverage fine, one test tool poor | That tool is unlike the others | Report it per tool |
| F0's photo weights near 0 | Photos not informative to the model | Check the readout quality; LoRA (stretch) |
| NaN or exploding loss | Learning rate; the NLL too early | Clipping, the NLL ramp, the Gate-2a learning rate |

### 7.10 Optional components

- **Per-fold PatchTST (encoder PT, stretch).** Masked-patch reconstruction
  (40 % masked) on the training tools' segments only, 20 epochs, then frozen.
- **LoRA wear fine-tuning of the photo encoder (stretch).**
  - DINOv2-B, LoRA rank 8 on q and v; target VB̃ per photo; Huber loss.
  - Augmentation: small shifts along the edge, ±3° rotation, mild colour
    jitter (none for chip photos, whose tint is signal).
  - 30 epochs, early stopping on the validation tool.
  - Training-tool readouts cross-fitted with 4 inner folds.
  - **Adopted only by Rule A on the full model** (the Gate-2 winner with LoRA
    features against frozen features), not by the probe.

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

Every table shows these, in both units.

| ID | Model | Deployable | Purpose |
|---|---|---|---|
| B0 | Window-aware constant k_u/2 | No | How easy the window is (4.13 passes, 412 s) |
| B1 | Mean training TTF − elapsed time, capped | Yes | Pure usage (5.17 passes, 513 s) |
| B2 | E[min(TTF − t, k) \| TTF > t] over the training tools. Intervals from the empirical quantiles of min(TTF_j − t, k) over the training tools still alive at t | Yes | Best usage-only rule (3.82 passes, 468 s); the S1 and S2 reference |
| B3 | Learned usage-only model: the chosen head on U features, no trajectory augmentation | Yes | A learned B2; v(∅) candidate (§11) |
| B4 | Ridge on the physics features (absolute and tool-relative) (mean of the last 3 segments) + usage features → capped RUL | Yes | Classic tool-wear regression |
| B5 | Photo wear readout VB̂ of the tool photos + degradation extrapolation (F5 machinery, §6.5) | Yes | "Read the photo and extrapolate" |
| R1 | Oracle: *measured* wear, causally isotonic-smoothed, + the same extrapolation | No | Upper reference: what a perfect photo reader would give |
| R2 | Oracle: last photo's class + residual-life table from the training tools | No | How much the class labels give |

**Degradation extrapolation (B5, R1).**

- Fit VB(τ) = α + β·τ^γ (γ ≥ 1) to each training tool's smoothed wear against
  cutting time. The prior over (α, log β, log γ) is Gaussian, fitted to the 8
  triples.
- At time τ, importance-weight 5,000 prior draws by the likelihood of the
  tool's wear history (noise sd from the training residuals).
- RUL samples = the first crossing of θ (the training tools' median VB̃ at
  the TTF, about 112 µm) − τ, capped. Passes = seconds / D̄.

---

## 9. Uncertainty and calibration

1. **Predictive distribution.** Aleatoric: the Student-t heads. Epistemic:
   the seed ensemble.
   - The mixture has mean avg μ_m and variance
     avg(σ_m² ν_m / (ν_m − 2)) + var(μ_m).
   - Its quantiles are found numerically (bisection on the mixture CDF), not
     by a Gaussian approximation.
2. **Calibration methods** (Gate 3, Rule B), one scale *c* per unit applied
   to every member's σ:

   | Method | Fitted on | Note |
   |---|---|---|
   | (a) Validation scale | The validation tool's window | Simple, but the same tool chose the step, so it is optimistic |
   | **(b) Out-of-fold scale (default candidate)** | Out-of-fold predictions of the 8 training tools from the 4 inner fits (§7.2 step 10) | 8 tools of calibration data. The validation tool becomes an independent coverage check before any test score |
   | (c) Cross-conformal | The same out-of-fold residuals, normalized \|y − μ\|/σ; the 90 % quantile | Distribution-free under exchangeability between tools. The coverage on the test tools is measured, not assumed |

3. **Causal RUL filter** (Gate 3, Rule A).
   - A Kalman filter per unit and tool. The state is RUL; the dynamics are
     x_{j+1} = x_j − Δ_j + w_j, with Var w_j = q·Δ_j. The measurement is the
     calibrated ensemble's μ_j, with its variance.
   - q is fitted on the out-of-fold predictions.
   - It is fully causal, and smooths the flicker that true RUL never shows
     (slope −1).
   - Predictions at the cap are passed through unfiltered.
4. **Gate 3** compares the head (Student-t, quantile) × the calibration
   method (a, b, c) by the validation Winkler score. The constraint is
   validation coverage within [0.85, 0.95] in both units. The filter is then
   tested by Rule A.

---

## 10. Evaluation

### 10.1 Where and how

- Every model predicts at every point of a tool in one causal pass. The
  truncation-invariance test proves that this equals predicting point by
  point.
- The headline uses the k-windows. Whole-life metrics (capped targets) and
  pass-end-only metrics come second.
- Force-missing passes (D8) and the points right after T4's and T2's gaps are
  also shown separately.

### 10.2 Metrics (each in seconds and in passes)

| Group | Metrics |
|---|---|
| Point | MAE, MSE, RMSE; bias (mean of ŷ − y); MAE_eq |
| Safety | Late-prediction rate: share of points with ŷ − y > 1 pass (> 100 s). Asymmetric PHM-style score s = e^{−d/2.6} − 1 for d < 0, e^{d/2.0} − 1 for d ≥ 0, with d in pass-equivalents (seconds / 100 s) |
| Prognostic (Saxena et al.) | α-λ accuracy: share within ±20 % of the true RUL at λ = 0.25, 0.5, 0.75 of the window. Prognostic horizon: the earliest time after which predictions stay within ±2 passes (±200 s) until failure |
| Uncertainty | Coverage (PICP) and width (MPIW) at 50, 80, 90, 95 %; Winkler score; CRPS; NLL; reliability curve |
| Stability | Monotonicity violation: mean \|Δμ + Δt\| (and \|Δμ_s + Δτ\|) between consecutive points |
| Decision | Replace the tool when the lower 90 % bound ≤ r₀. Over a grid of r₀: missed failures (tools run past the TTF) against wasted life. Compared with fixed-schedule replacement at quantiles of the training TTFs |

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
- **Breakdowns:** error by RUL bin (0-3, …, 12-15 passes) and by position
  inside the pass (OmniFuser's phase-locking lesson).

### 10.4 Figures

1. Per tool: true RUL, predicted mean, 50 % and 90 % bands, in both units;
   the window shaded; photo labels as a strip; gaps marked.
2. Error and coverage against true RUL.
3. Reliability curves.
4. α-λ plot.
5. Decision curve.
6. Shapley bars with intervals.
7. Modality and channel importance over the horizon.

---

## 11. Modality-importance study

**Players:** F (force), T (tool photo), C (chip photo), W (workpiece photo).
**Usage U is in every coalition** (a factory always knows the usage).

### 11.1 Retraining Shapley (primary)

- Train the frozen design on each of the 15 non-empty subsets: 15 × 10 folds
  × 3 seeds = 450 fits.
- **The empty coalition** is the better, on validation, of B2 and B3 (no
  trajectory augmentation). Trajectory augmentation deliberately destroys
  usage information, so a U-only model trained with it would understate
  v(∅) and inflate every sensing modality.
- Value: v(S) = −k-window MAE_eq on the test tools (after the freeze, D15).

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

- **F0:** the weights α_m over the RUL horizon.
- **F1:** integrated gradients and occlusion over history tokens. How far
  back does the model look? Does the force matter more near failure?
- **Photo staleness:** replace the latest photos with ones 1, 2 or 3 passes
  older (Project 14's staleness test).

### 11.4 Sensing without usage

- A recent-window variant (last 4 recorded passes, U off) with F, with T, and
  with all four.
- Compared with B2. This separates what the sensors know from how long the
  tool has been used. Its usage sensitivity s_U should be near 0.

### 11.5 Force channels

All five force channels are input (§5.4).

1. **Test-time channel masking** on the final model (trained with channel
   dropout): the change in MAE in both units when each channel, or each pair,
   is masked.
2. **Retraining check:** the final design without Fz and Fz_rel, under the
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
| 1 | Force channels | Without Fz and Fz_rel (shared with §11.5) |
| 1 | Missing modality at test time | Each modality absent; 50 % of photos missing at random (no refits) |
| 2 | TTF rule | `tool_label` = 3; dulled from here on |
| 2 | Horizon | k = 10; k = 20 (T5's window is then its whole life) |
| 2 | Duplicates | OmniFuser's deduplication |
| 2 | Tokens | Absolute; both (only if not already settled in Gate 2b) |
| 2 | Trimming | OmniFuser's non-causal 5 %-of-pass reference |
| 2 | Segment length | 5 s, 20 s (new feature extraction; GPU) |

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
| Project 14 | `encoders.py`: `MantisEncoder`, `MomentEncoder`, `PatchTSTEncoder`, `ASTEncoder`, `DinoV2Encoder`, `hf_revision` | §6.2 |
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
| L3 | Fz drifts (F3) | All channels input; Fz_rel; Fz's contribution measured |
| L4 | One validation tool is noisy; replicates differ by up to 0.017 (F4, F14) | Smoothed whole-life selection, seed ensembles, Rule A with 7 of 10 wins |
| L5 | The cross-tool gap (training 0.88, test 0.67), not under-fitting | Frozen encoders, PCA, trajectory augmentation, about 0.1 M parameters; token mode tested in Gate 2b |
| L6 | Frozen ImageNet features read wear poorly; the crop helps; fine-tuning helps most | Edge crop, DINOv2 grid, wear readout; LoRA by Rule A on the full model |
| L7 | Absolute embeddings plus elapsed time memorize the training units (Project 14) | Trajectory augmentation; s_U diagnostic; token mode tested in Gate 2b (in Stage 2, absolute features identified tools *less* than relative ones) |
| L8 | Longer history did not help with 9 units (Project 14 XJTU-SY) | "Last pass only" control, reported |
| L9 | Spectrograms of the raw force gave OmniFuser's best forecasts | Encoder SP in Gate 1 |
| L10 | C2F and the image GAT gained nothing over concatenation; late fusion was hard to beat (Project 14) | Finalists F1 and F0; C2F documented as an alternative |
| L11 | Labels are noisy and wear is non-monotone (F12) | Confirmed TTF; isotonic targets; causal smoothing for inputs; TTF sensitivity |
| L12 | Error phase-locked to pass boundaries (OmniFuser report §6.3) | Prediction at every segment; error by position in the pass |
| L13 | Laptop: WMI hangs, other jobs hold the CPU; bare `python` hung | WMI guard; `claim_resources`; the project's own venv |
| L14 | Ruche's 24 h limit and interruptions | Resumable fits, arrays that skip finished fits, `chain.sh` |
| L15 | **New:** recorded passes are not the physical history (unrecorded runs) | D12, with a wear-jump check per gap |
| L16 | **New:** whole-pass preprocessing leaks the pass length into the pass | Causal trim and time features (D14), with tests |
| L17 | **New:** calibrating on the tool that chose the step is optimistic | Out-of-fold calibration; validation coverage as an independent check |

---

## 14. Repository and engineering

### 14.1 Where things live

| What | Where | Why |
|---|---|---|
| Code, configs, docs, tests | `C:\Users\ali_a\OneDrive\Desktop\RUL Model` (git) | Small; synced |
| Raw data copy (force, labels, tool, chip, work), stores, caches, runs | `RULFUSE_DATA_ROOT` = `D:\RULModel` | Gigabytes of memory-mapped files must not sync through OneDrive (locks, slow sync) |
| Python environment | `D:\RULModel\.venv`, Python 3.11 or 3.12 | The encoder libraries (momentfm, mantis-tsfm, chronos-forecasting) lag behind Python 3.14, which OmniFuser's venv uses. The same minor version is used on Ruche |
| Hugging Face weights | `D:\RULModel\hf_cache` (laptop), `$WORKDIR/hf_cache` (Ruche) | Offline loading |

### 14.2 Layout

```text
RUL Model/
  pyproject.toml            package "rulfuse"; extras [test, eda, dev, encoders]
  configs/                  data.yaml, final.yaml, gate1/, gate2/, sweep/, gate3/, shapley/, sensitivity/
  rulfuse/
    __init__.py             WMI guard (vendored)
    config.py               dataclasses; YAML I/O; signature()
    data/
      nonastreda.py         vendored loader; all 512 passes; duplicate groups; chip/work paths
      trim.py               causal span rule (and OmniFuser's, for the sensitivity arm)
      timeline.py           gap-aware axis, D8 attachment, availability times
      segments.py           10 s segmentation, segment store
      labels.py             5 TTF rules, RUL (passes, s), caps, masks, isotonic wear (full, causal)
      images.py             crops, resizing, ImageBank (vendored)
      folds.py              leave_one_tool_out_folds (vendored), manifests
    features/
      physics.py            handcrafted force and image features
      encoders.py           vendored Project 14 wrappers + Chronos, SigLIP 2, DINOv3
      cache.py              FeatureStore (vendored), signatures
      probe.py              Gate-1 ridge probes, cross-fitted wear readouts
    models/
      tokens.py             PCA, causal baseline, projections, embeddings, relative-time bias
      fusion.py             F1 token fusion, F0 experts + gPoE
      heads.py              dual-unit Student-t, quantile, wear, CORAL
    degradation.py          extrapolation used by B5 and R1
    augment.py              trajectory augmentation, modality, channel and token dropout
    losses.py
    train.py                fit one (fold, seed): phases, EMA, selection, resume
    uncertainty.py          mixture, calibration (a, b, c), Kalman filter
    baselines.py            B0-B5, R1, R2
    evaluation.py           §10 metrics
    importance.py           Shapley (retraining, masking), staleness, IG/occlusion, channels
    stats.py                bootstrap, Wilcoxon + Holm (vendored)
    cross_validation.py     driver, aggregation, test-score seal (§4.6)
    tracking.py, resources.py, bundle.py   (vendored)
    cli.py                  rulfuse <command>
  scripts/                  eda_rul.py, end_to_end_trace.py, make_figures.py
  slurm/                    env.sh, job.sbatch, cpu.sbatch, array.sbatch, chain.sh, status.sh,
                            checks.sbatch, prefetch_weights.sh
  tests/                    §15.2
  docs/                     EDA_RUL.md, END_TO_END.md, RUCHE_RUNBOOK.md, RESULTS.md, archive/
```

### 14.3 Commands

```bash
rulfuse verify-data   --raw $RULFUSE_DATA_ROOT/raw              # hashes, counts, groups, gaps
rulfuse build-store   --config configs/data.yaml               # trim, timeline, segments, labels
rulfuse features      --modality force --encoder mantis --device cuda
rulfuse probe         --gate 1 --out runs/gate1                 # encoders, derived channels
rulfuse baselines     --out runs/baselines
rulfuse manifest      --study gate2a && sbatch --array=0-59%4 slurm/array.sbatch runs/gate2a/manifest.tsv
rulfuse aggregate     --study gate2a                            # ensembles, calibration, decision.json
rulfuse freeze        --config configs/final.yaml              # D15; records the hash
rulfuse aggregate     --study final --after-freeze
rulfuse report        --runs runs/final runs/baselines runs/shapley runs/sensitivity --out runs/report
rulfuse predict       --model runs/final/fold-T3 --tool T3      # 5-seed ensemble
```

### 14.4 Outputs of every fit

| File | Content |
|---|---|
| `config.yaml`, `signature.json` | Resolved configuration; data and feature signatures |
| `metrics.jsonl` | Per evaluation step |
| `predictions.parquet` | Per point: tool, pass, segment, τ, label time; per unit y, RUL, μ, σ, ν, quantiles; F0's experts and weights; masks; `split` (test rows sealed before the freeze) |
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
| **P5 Labels** | Wrong TTF; RUL arithmetic; caps; a TTF inside a gap | Each rule on hand-made sequences (flips, ties, sequence end); RUL in both units; caps and masks; isotonic non-decreasing; causal isotonic uses only the past | The TTF table of §2.3 and the windows of §2.2 exactly | Smoothed wear at the TTF in 109-123 µm for every tool | Exact |
| **P6 Folds** | Tool in two roles | Disjoint; cyclic; each tool tested once | – | – | Exact |
| **P7 Normalization** | Statistics from validation or test tools; non-causal baseline | Poison test: extreme values in validation and test tools leave the scalers and PCA unchanged; perturbing later segments leaves b_u(t) unchanged | – | Reported, not a gate: the test tool's shift and the tool identifiability for absolute against relative features (Stage 2: shift 0.31 against 0.35 sd; identification 23 % against 35 %, so the premise was not confirmed beyond static offsets; the token mode moved to Gate 2b) | Poison and causality tests pass |
| **P8 Encoders and caches** | Wrong checkpoint; non-determinism; mixed caches | Shapes; same input → same output; the signature changes with checkpoint or preprocessing; a cache with a foreign signature is refused | Offline loading on Ruche (J0); duplicated recordings → identical embeddings; laptop vs Ruche embeddings of 20 segments agree within 1e-3 relative | – | All pass |
| **P9 Gate-1 probes and readouts** | Probe leakage; in-sample readouts; photo features that encode the stage label | Ridge recovers known coefficients; a cross-fitted readout of a tool never uses that tool's labels; PCA fitted on training only | Readout MAE on held-out tools reported per photo type | The H probe beats the training-mean VB predictor; shuffling VB within training tools collapses the probe to the mean | Leakage tests pass; H beats the mean; stage-step leakage probe: a stage boundary is the largest run-boundary step in ≤ 3 of 10 tools (`rulfuse.audit`) |
| **P10 Tokens and time** | D_p or the future reaching an input | Usage features at (p, s) do not change when D_p or later samples change; photo tokens sit after their pass's last segment; the relative bias depends only on Δτ | Spot-check of 20 random points against the timeline table | – | All pass |
| **P11 Models F1, F0** | Look-ahead; mask bugs; expert bugs | **Truncation invariance** (outputs at t unchanged when later tokens are removed or replaced by noise); **no photo leak** (changing pass *p*'s photo changes nothing up to the end of pass *p*); missing-modality masks; gPoE with one expert equals that expert; F0 with all sensing masked equals the U expert; tiny overfit | – | Sanity checks 1, 4, 6 (§7.7) | All pass |
| **P12 Heads and losses** | Range errors; NLL moving μ; unit imbalance | μ within [0, k_u]; ν ≥ 3; the NLL gradient never reaches μ; consistency loss 0 for slope −1; both heads receive gradient; per-unit scaling by k_u | – | On synthetic Student-t data the head recovers σ within 10 % | All pass |
| **P13 Augmentation** | Corrupted labels; photos detached | Trajectory augmentation keeps the TTF pass, recomputes targets, never touches the baseline passes, keeps photos and wear on their passes; dropout rates match configuration; channel dropout zeroes the right block | Histogram of augmented TTFs (wider than the real 19-47) | Usage sensitivity s_U falls with trajectory augmentation on (Gate 2b) | Tests pass; s_U reported |
| **P14 Training loop** | Selection bugs; non-resumable; non-deterministic | Phase schedule; EMA restart at step 101; selection picks the smoothed minimum; resume after a kill gives identical weights; same seed → same weights on CPU | – | Sanity checks 2, 3, 5, 7; learning curves without the memorization warning | All pass |
| **P15 Ensemble, calibration, filter** | Wrong mixture; leaky calibration; non-causal filter | Mixture moments and quantiles against Monte Carlo; the scale reaches the target coverage on synthetic data; inner folds never contain the outer validation or test tool; conformal coverage on exchangeable synthetic data; the filter on a synthetic linear-decreasing truth lowers the error and is causal | – | Out-of-fold calibration: validation coverage within [0.85, 0.95], checked before any test score | All pass |
| **P16 Baselines** | Formula or protocol errors | B2 on a toy case equals a hand computation; B5 and R1 extrapolation on a synthetic power-law tool recover its crossing | B0-B2 reproduce §2.7 (4.13 / 5.17 / 3.82 passes; 412 / 513 / 468 s) | R1 beats every deployable model. If it does not, the extrapolation is broken | Exact; R1 best |
| **P17 Metrics and statistics** | Wrong formulas | Every metric against hand-computed values; Student-t CRPS against Monte Carlo; Wilcoxon and Holm against SciPy; bootstrap reproducible by seed | – | – | Exact |
| **P18 Importance** | Shapley errors; unfair v(∅) | A synthetic additive game recovers its values; efficiency Σφ = v(N) − v(∅); in an additive synthetic model, masking ≈ retraining | – | The signs of retraining and masking agree for the top modality; a sensing-only model's s_U is near 0 | All pass |
| **P19 Sensitivity configs** | An arm silently equal to the default | Each arm's config changes exactly the intended field; label arms produce the TTFs of §2.3 | – | – | Exact |
| **P20 Ruche pipeline** | Lost or duplicated fits; offline failures | Manifest generation; `DONE` skip; resume after a simulated kill; `status.sh` parsing on a fake tree | J0 checks; pilot J1; one fold on the laptop and on Ruche: same MAE_eq within 1e-3 relative (CPU equal, GPU tolerance) | – | All pass |
| **P21 Report** | Numbers not reproducible | Golden-file test: a tiny synthetic run's report equals the stored one | Deleting `runs/report` and rerunning gives identical tables (hash) | – | Identical |

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

---

## 16. Training on Ruche

### 16.1 What runs where

| Work | Where |
|---|---|
| Development, tests, stores, physics features, Gate 1, baselines, sanity checks | **Laptop CPU**: check idle CPU first (`rulfuse resources`) |
| Pretrained encoder features | Ruche `gpua100` |
| Fit arrays (Gates 2-3, sweep, final, Shapley, sensitivity) | Ruche `gpua100`, or a CPU partition if the queue is long (the models are small) |
| LoRA (stretch) | Ruche `gpua100` |
| Aggregation and report | Laptop or `cpu_short` |

### 16.2 One-time setup

Follows OmniFuser's `TRAINING_RUNBOOK_RUCHE.md` §3-§7.

1. `ssh $Ruche`; `cd $WORKDIR` (500 GB, **not backed up**); the `~/.conda`
   symlink to `$WORKDIR/.conda` (home quota 50 GB).
2. Environment:

   ```bash
   module load anaconda3/2023.09-0/none-none
   conda create -n rulfuse python=3.11 -y && source activate rulfuse
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install --no-cache-dir -e ".[test,encoders]"
   ```

3. `slurm/env.sh`, based on Project 14's:
   - loads the module and activates `rulfuse`;
   - sets `RULFUSE_DATA_ROOT=$WORKDIR/rul_data` and
     `HF_HOME=$WORKDIR/hf_cache`;
   - sets `HF_HUB_OFFLINE=1` inside jobs, and `PYTHONUNBUFFERED=1`.
4. `bash slurm/prefetch_weights.sh` **on the login node**. DINOv3 needs its
   licence accepted first.
5. Data: `tar` the raw copy (about 1.3 GB); `scp` it to `/workdir/<login>/`;
   run `rulfuse verify-data` (P1).
6. Code: a git clone on a `ruche` branch (§19), or a tarball (OmniFuser
   runbook §5).

### 16.3 SLURM scripts

| Script | Content |
|---|---|
| `job.sbatch` | One command on one GPU: `gpua100`, 8 CPUs, 24 h, `--export=NONE` (OmniFuser's) |
| `cpu.sbatch` | The same on a CPU partition (Project 14's) |
| `array.sbatch` (new) | `--array=0-N%4`: task *i* runs row *i* of a manifest TSV (arm, fold, seeds); finished fits are skipped; `%4` respects the 4-GPU limit per user |
| `chain.sh` | Only for jobs that can exceed 24 h |
| `status.sh` | Jobs; finished, running and missing fits per study; logs with errors |
| `checks.sbatch` | Tests; offline weight loading; GPU check |

### 16.4 Job plan

Estimates assume about 1.5 min per fit; the pilot measures the real cost.

| Job | Stage | Content | Tasks | Fits | Estimate |
|---|---|---|---|---|---|
| J0 | 3 | Checks: tests, offline weights, GPU | 2 | – | 15 min |
| J1 | 3 | Pilot: one fold, F1 on physics features | 1 | 3 | 30 min |
| J2 | 3 | Store on Ruche; signature equal to the laptop's | 1 | – | 15 min |
| J3 | 3 | Force features: MA, MO-base, MO-large, CH, AST, SP | 6 | – | 2-4 GPU-h |
| J4 | 3 | Photo features: EB0, D2-S/B, SG2, (D3) × 3 photo types | 5 | – | 1-2 GPU-h |
| J5 | 5 | Gate 2a: 2 families × 3 learning rates × 10 folds | 60 | 180 | 4-5 GPU-h |
| J6 | 5 | Gate 2b: 4 options + 1 control × 10 folds | 50 | 150 | 4 GPU-h |
| J7 | 5 | Sweep: 10 variants × 10 folds | 100 | 300 | 7-8 GPU-h |
| J8 | 6 | Gate 3: 2 heads × 10 folds, each with 4 inner fits per seed | 20 | 300 | 7-8 GPU-h |
| J9 | 6 | LoRA (stretch): 10 folds × 3 photo types × (1 + 4 inner) | 30 | 150 fine-tunes | 10-12 GPU-h |
| J10 | 7 | Final: 10 folds × 5 seeds + 4 inner fits each | 10 | 250 | 6 GPU-h |
| J11 | 8 | Shapley: 15 coalitions × 10 folds | 150 | 450 | 11 GPU-h |
| J12 | 8 | Sensitivity tier 1 (5 refit arms) / tier 2 (up to 9 arms + features) | 50 / 90 | 150 / 270 | 4 / 8 GPU-h |
| J13 | 9 | Report | 1 | – | 15 min |

The total is about 65-80 A100-hours. Submit arrays in the evening; run
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
S3 Features, G1         ████████
S4 Training engine        ████████
S5 Fusion, G2                   ████████
S6 Uncertainty, G3                    ████████ ◆ freeze Dec 1
S7 Final runs                                 ███
S8 Analysis                                    ████████
S9 Report                                            ████████
```

**Critical path:** S0 → S2 → S4 → S5 → S6 → S7 → S8 → S9.

- S1 must settle the time axis (D12) before S2's labels are final.
- S3 (GPU features) runs beside S4 (laptop), and must finish Gate 1 before
  S5 starts.

**Minimum viable result**, if time runs short: S0-S4, then F1 only in S5, then
the Student-t head with out-of-fold calibration in S6, then S7 and Shapley
(S8).

**Drop order:** tier-2 sensitivity → LoRA → the sweep → F0 → the quantile
head.

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
  - B0-B2 in both units;
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

#### Stage 3: Features, Ruche bring-up and Gate 1 (Tue Nov 10 to Tue Nov 17)

- **Goal:** cached features for every candidate encoder, and one encoder per
  modality.
- **Tasks:**
  - physics and handcrafted image features (laptop);
  - the encoder wrappers and the feature cache;
  - Ruche setup and jobs J0-J4;
  - the Gate-1 probes, the derived-channel decision, and the cross-fitted
    readouts (laptop).
- **Outputs:** `interim/features/*`; `runs/gate1/decision.json`.
- **Verification:** P8, P9.
- **Exit:** decisions 1-5 logged (§17.3).

#### Stage 4: Training engine and sanity checks (Thu Nov 12 to Thu Nov 19)

- **Goal:** a trustworthy training loop before any real comparison.
- **Tasks:**
  - `tokens.py` (with the relative-time bias), F1 and F0, the heads, losses,
    augmentation, `train.py`;
  - `uncertainty.py` (mixture, calibration (a)-(c), the filter);
  - `baselines.py`, `degradation.py`, `evaluation.py`, `stats.py`;
  - the diagnostics; the seven sanity checks on physics features (laptop).
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
  - the validation MAE_eq of the winner below B2's on validation.
- **Exit:** decisions 6-12 logged.

#### Stage 6: Uncertainty, Gate 3 and design freeze (Tue Nov 24 to Tue Dec 1)

- **Goal:** calibrated intervals and a frozen design.
- **Tasks:**
  - J8 (heads × calibration methods with inner fits);
  - the filter by Rule A;
  - LoRA (J9) only if Stage 5 ended on time;
  - write `configs/final.yaml`; `rulfuse freeze`.
- **Outputs:** `runs/gate3`; the frozen configuration and its hash.
- **Verification:** P15 scientific check (validation coverage in band, both
  units); reliability curves on validation.
- **Exit:** decisions 13-16 logged; freeze recorded (D15). **No test score has
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
| 1-4 | Encoder for force, tool, chip, workpiece | 1 | B (VB MAE) |
| 5 | Derived force channels (Fz_rel, \|F\|) | 1 | A |
| 6 | Learning rate of F1; of F0 | 2a | B |
| 7 | Fusion family (F1 or F0) | 2a | B |
| 8 | Trajectory augmentation | 2b | A |
| 9 | Phase-A warm-up | 2b | A |
| 10 | Token mode (relative, absolute, both) | 2b | B (peers, after the Stage-2 evidence) |
| 11 | Sweep adoptions (each factor) | Sweep | A |
| 12 | (Control) last pass only | 2b | Reported, not decided |
| 13 | Head (Student-t or quantile) | 3 | B |
| 14 | Calibration method (a, b, c) | 3 | B |
| 15 | Causal RUL filter | 3 | A |
| 16 | LoRA (if run) | 3 | A, on the full model |
| 17 | Photo types that pass the stage-step leakage probe | 1 | Probe: a stage boundary is the largest run-boundary step in ≤ 3 of 10 tools |

---

## 18. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 10 tools give wide intervals; gaps between models are not significant | High | Paired tests; effect sizes with intervals; few pre-registered comparisons; both summaries |
| The model leans on usage and adds little over B2 | Medium-high | Trajectory augmentation; s_U diagnostic; B2 and B3 everywhere; sensing-only study |
| The force carries little RUL information (Project 14: flat force pathway) | Medium | Physics features and spectrograms in Gate 1; both token halves; the channel study |
| Photo features stay weak | Medium | Crop, DINOv2 grid, wear readout; LoRA; the chip photo as a second visual cue |
| The gap-aware axis is wrong (the runs were not cut) | Low (you confirmed it; the wear jumps agree) | The recorded-axis arm (tier 1) |
| T4's thin window (20 points) and T5's short life | High | Whole-life selection; both summaries; per-tool reporting |
| Label noise near the TTF (T3, T5) | Medium | Confirmed rule; TTF sensitivity arms |
| Out-of-fold calibration misses tool-specific spread | Medium | Validation coverage checked before the freeze; per-tool coverage reported |
| The 6-week scope slips | Medium | Minimum viable result and drop order (§17.1) |
| Ruche queue or gated checkpoints | Low-medium | CPU partitions; DINOv3 optional; prefetch in Stage 3 |
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
   per-image standardization plus the Gate-1 leakage probe, and exclusion if
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
6. **LoRA** as a stretch goal in Stage 6. *Default: only if Stage 5 ends on
   time.*
7. **Deliverables:** a report only, or slides too? *Default: report and
   figures; slides optional.*

---

## Appendix A: default settings

| Setting | Default | Section |
|---|---|---|
| TTF rule | Confirmed first dulled (dulled, then ≥ 2 dulled among the next 3 recorded) | D2 |
| Time axis | Gap-aware; interior gaps counted at the tool's median pass duration; leading gaps not | D12 |
| Windows and caps | 15 passes; 1,500 s | D3, §3.2 |
| Selection and gate metric | MAE_eq = ½ (MAE_pass + MAE_sec / 100 s) | §4.3 |
| Segment | 10 s from the causally trimmed start, remainder < 5 s merged | D7, §5.2 |
| Trim reference | Median envelope over the first 5 s | §5.2 |
| Channels | Fx, Fy, Fz (always) + Fz_rel, \|F\| (Gate 1); PCA 8 absolute + 8 relative per channel | §5.4 |
| Duplicates | Attached to the latest member; the others force-missing | D8 |
| Photo crop (tool) | Rows 0.30-0.85, full width, 224 × 672 | §5.7 |
| PCA | 16 + 16 per photo type; 8 + 8 per force channel (absolute + relative); fitted on the training tools | §5.9 |
| Tool baseline and token mode | First B = 4 recorded passes, causal running mean; token mode "both" (absolute + relative halves); readouts absolute | §5.9 |
| Fusion | F1 or F0 (Gate 2a) | §6.4 |
| F1 backbone | Causal transformer: 2 layers, d = 64, 4 heads, FFN 128, dropout 0.2, relative-time bias | §6.4 |
| Heads | Dual-unit Student-t: μ = k_u·sigmoid; σ floor 0.005·k_u; ν = 3 + softplus | §6.6 |
| Loss weights | MSE 1; NLL 0.5 (mean detached); consistency 0.1; VB 0.3 | §6.7 |
| Point weights | Equal total per tool; w_far = 0.3 | §7.3 |
| Optimizer | AdamW, lr 1e-3 (Gate 2a), weight decay 0.05, gradient clip 1.0 | §7.3 |
| Schedule | Phase A 1-100 (warm-up 20); phase B 101-500, cosine; NLL ramp 101-150; EMA 0.99 restarted at 101; patience 10 evaluations | §7.4 |
| Batch | 8 training tools × 4 augmented copies | §7.3 |
| Augmentation | Trajectory ρ ~ U[0, 0.2]; modality dropout 0.3 each (all four 0.05); channel 0.1; token 0.1; feature noise 0.1 | §6.8 |
| Seeds | 3 (gates), 5 (final) | §4.5 |
| Calibration | Out-of-fold scale, 4 inner folds (Gate 3 decides) | §9 |
| Gate rules | A: threshold, ≥ 7 of 10 wins, no unit worse by > 5 %; B: best mean | §4.4 |

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
