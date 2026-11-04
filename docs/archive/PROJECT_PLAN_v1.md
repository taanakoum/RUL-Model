# RUL-Fuse: multimodal remaining-useful-life prediction on Nonastreda

**Project plan.** Drafted 2026-11-04. Target completion 2026-12-16 (6 weeks).

Status: plan only; nothing is implemented yet. Every number below was measured
on `Data/` (identical to OmniFuser's `Nonastreda_Data/`), on OmniFuser's
`experiment_results/eda/records.csv`, and on the chip and workpiece photos in
`D:\Project14\data\raw\nonastreda`. Appendix B lists how.

## Contents

0. [Decisions](#0-decisions)
1. [Goal and success criteria](#1-goal-and-success-criteria)
2. [What the data gives for RUL](#2-what-the-data-gives-for-rul)
3. [Task definition](#3-task-definition)
4. [Cross-validation and model-selection protocol](#4-cross-validation-and-model-selection-protocol)
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
15. [Testing and validation of the code](#15-testing-and-validation-of-the-code)
16. [Training on Ruche](#16-training-on-ruche)
17. [Stages and timeline](#17-stages-and-timeline)
18. [Risks and mitigations](#18-risks-and-mitigations)
19. [Open questions for you](#19-open-questions-for-you)
- [Appendix A: default settings](#appendix-a-default-settings)
- [Appendix B: how the numbers in this plan were measured](#appendix-b-how-the-numbers-in-this-plan-were-measured)
- [Appendix C: references](#appendix-c-references)

---

## 0. Decisions

| # | Decision | Value | Source |
|---|---|---|---|
| D1 | Validation protocol | 10-fold leave-one-tool-out, as in OmniFuser. Fold *i*: test T*i*, validation T*i+1* (cyclic; T1 after T10), training on the other 8 tools | You |
| D2 | Time to failure (TTF) | **Confirmed first dulled**: the first pass labelled `dulled` that is followed by at least 2 more `dulled` passes among its next 3 | You (2026-11-04), chosen from 5 candidates (§2.3) |
| D3 | Horizon | **k = 15 passes.** The headline metrics use the points with RUL ≤ 15. Training uses every pre-TTF segment, with its target capped at 15 | You |
| D4 | Modalities | Force with **all three channels** Fx, Fy, Fz at 1 kHz (plus two derived channels, §5.4); tool photo; chip photo; workpiece photo | You (channels: 2026-11-04) |
| D5 | Timeline | 6 weeks, to 2026-12-16 | You |
| D6 | RUL unit | **Both, at every segment.** RUL in cutting seconds and in passes (fractional inside a pass), each with its own interval, from two heads on one shared backbone (§6.5). Cutting seconds means spindle-engaged time. The idle time between passes is not recorded, so wall-clock time to failure cannot be predicted | You (2026-11-04) |
| D7 | Segment | 10 s of raw 1 kHz force, counted from the start of the pass. A remainder shorter than 5 s joins the previous segment | Default; §12 tests 5 s and 20 s |
| D8 | Time axis | All 512 physical passes. A duplicated force recording (§2.4) is attached to the **latest** pass of its group; the earlier members have missing force | Default (causally safe) |
| D9 | Photo availability | A photo of pass *p* is input only to predictions made after pass *p* ends | Default. This removes OmniFuser's lag-0 label leakage (`EDA.md` §7) |
| D10 | Loss | MSE on the predicted mean (your request), plus a likelihood term for the spread with the mean detached, plus auxiliary wear targets | Default (§6.6) |
| D11 | Sign convention | RUL = t_TTF − t ≥ 0. You wrote "current time step minus TTF"; the sign is flipped so that RUL is positive | Convention |

---

## 1. Goal and success criteria

**Goal.** At the end of every 10 s force segment, predict how long the tool
has left before it is (confirmed) dulled. The prediction comes in two units:
**seconds of cutting** and **passes**. The model reads everything recorded so
far: every past force segment and every photo already taken. It outputs a
calibrated interval as well as a point value, in each unit. The project also
measures how much each modality contributes.

**Success criteria.** These are fixed in advance and measured on the test
tools, as the mean over 10 folds × 5 seeds.

| # | Criterion | Threshold |
|---|---|---|
| S1 | Accuracy | MAE on the k-window below the best *deployable* baseline, **in both units**. That baseline is conditional residual life: 4.22 passes, and 522 s on the 1,500 s window (§2.6). Paired Wilcoxon test over the 10 test tools, p < 0.05. Also below the window-aware constant (4.00 passes, 389 s), which no deployable method can use |
| S2 | Intervals | Pooled coverage of the nominal 90 % interval within [0.85, 0.95]. Per-tool coverage is reported. Interval score better than the baselines' intervals |
| S3 | Modality importance | Shapley values for force, tool, chip and workpiece, with 95 % bootstrap intervals over tools. The sign of the most important modality is resolved |
| S4 | Integrity | Every causality and leakage test passes (§15). No decision reads a test-tool score |
| S5 | Reproducibility | Every reported number regenerates from the saved predictions with one command |

**Non-goals.** There is no published RUL benchmark on Nonastreda to match.
External datasets (MATWI, PHM 2010) are out of scope for 6 weeks (§19).

---

## 2. What the data gives for RUL

### 2.1 Inventory

| Item | Content |
|---|---|
| Tools | 10 (T1-T10), each run from new until well past dulled |
| Passes | 512 (`T#R#B#`), 4 blocks per run, 128 runs. Per tool: 40-60 passes |
| Force | Fx, Fy, Fz at 1 kHz. Trimmed pass length: 97 s median. T9's passes are about 81 s and T10's about 122 s, so the cutting conditions likely differ by tool (§19) |
| Tool photo | 1 per pass, taken **after** the pass. About 1550 × 500 px (T1 and T2: 3100 × 1000). The cutting edge lies between 0.51 and 0.68 of the photo height (OmniFuser photo probe) |
| Chip photo | 1 per pass, 1024 × 1024 JPG, in `D:\Project14\data\raw\nonastreda\chip`. Shows chip shape and heat tint |
| Workpiece photo | 1 per pass, 512 × 512 PNG, in `...\work`. Grey surface texture |
| Publisher spectrograms | `...\spec\{x,y,z}`: 512, 512 and 506 whole-pass images. Not used by default (a whole-pass input is only available after the pass) |
| Labels | `image_label` (sharp, used, dulled; the annotators' visual judgement), `tool_label` (1/2/3, monotone), `flank_wear` (µm), `gaps`, `overhang` (labels_reg.csv) |

### 2.2 Per tool, under the chosen TTF rule (D2)

| Tool | Passes | TTF (pass) | Smoothed wear at TTF (µm) | Cutting time to TTF (s) | Pre-TTF 10 s segments | Points with RUL ≤ 15 |
|---|---|---|---|---|---|---|
| T1 | 40 | 31 | 118 | 3,000 | 308 | 149 |
| T2 | 56 | 40 | 112 | 3,819 | 393 | 150 |
| T3 | 52 | 36 | 123 | 3,464 | 355 | 150 |
| T4 | 40 | 22 | 113 | 2,116 | 216 | 150 |
| T5 | 48 | 19 | 110 | 1,827 | 188 | 150 |
| T6 | 60 | 47 | 109 | 4,474 | 458 | 150 |
| T7 | 56 | 47 | 109 | 4,543 | 465 | 150 |
| T8 | 52 | 46 | 119 | 4,474 | 459 | 150 |
| T9 | 52 | 44 | 111 | 3,841 | 379 | 120 |
| T10 | 56 | 44 | 111 | 5,358 | 526 | 180 |
| **All** | 512 | mean 37.6, sd 9.9 | median 112 | | **3,747** | **1,499** |

The confirmed rule agrees with physics. Smoothed flank wear at the confirmed
TTF is 109-123 µm on every tool, which matches the dataset's 110 µm
"dulled" threshold. This is what makes a wear-threshold model (F5, §6.4)
viable.

### 2.3 TTF definitions compared (pass number of the TTF pass)

| Rule | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | Mean ± sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Confirmed first dulled (D2)** | 31 | 40 | 36 | 22 | 19 | 47 | 47 | 46 | 44 | 44 | 37.6 ± 9.9 |
| Literal first dulled | 31 | 40 | **20** | 22 | **16** | **24** | 47 | **27** | 44 | 42 | 31.3 ± 10.6 |
| Smoothed wear ≥ 110 µm | 31 | 40 | 28 | 22 | 23 | 50 | 49 | 45 | 44 | 44 | 37.6 ± 10.1 |
| `tool_label` = 3 | 29 | 41 | 25 | 21 | 21 | 49 | 49 | 45 | 45 | 45 | 37.0 ± 11.0 |
| Dulled from here on | 31 | 44 | 39 | 22 | 31 | 54 | 49 | 50 | 44 | 46 | 41.0 ± 9.6 |

The literal rule puts T6's failure at 76 µm and T8's at 65 µm: isolated label
flips (`EDA.md` §2). The other four rules agree within a few passes, except on
T3 and T5. All four alternatives are rerun as sensitivity arms (§12). No tool
is censored: every tool reaches the TTF.

### 2.4 Duplicated force recordings are a recording artifact

OmniFuser found 132 groups of bit-identical force arrays within a run (127 of
them B3 + B4). Their tool, chip and workpiece photos are **all distinct**
(0 of 132 groups share a photo hash), and so are their flank-wear values. The
tool therefore did 4 physical passes per run; only the force file was copied.

For RUL the time axis must count every physical pass. Dropping 134 passes, as
OmniFuser's deduplication does, would shorten every life by about a quarter.

- **Default (D8).** Attach the recording to the **latest** pass of its group
  and mark the earlier members force-missing. Their duration is taken from the
  copy. If the recording really belongs to the later pass, attaching it to the
  earlier one would show the model the future by one pass. Attaching it to the
  later one at worst delays real information.
- **Sensitivity arms (§12).** (i) Attach the recording to every member. (ii)
  OmniFuser's deduplication (378 passes, time counted in retained passes).

### 2.5 Known traps, carried over from OmniFuser and Project 14

| Trap | Evidence | Consequence here |
|---|---|---|
| Same-pass photo leaks the label | Copying the lag-0 photo's label scores 0.91-0.95 (`EDA.md` §7) | D9: photos only after their pass |
| Run number is a shortcut | Run number alone gives 71 % accuracy (OmniFuser F2) | No run or block codes. Usage enters only as explicit, switchable time features. Schedule baselines appear in every table |
| Fz drifts | Sawtooth that resets every 1-3 passes; correlation of its level with wear −0.04 (F3) | All three channels are input (D4). The drift is neutralized by a causal, offset-removed copy of Fz (§5.4). The channel study (§11.5) measures what Fz adds |
| Tool-to-tool force offset | T1's Fx sits at +1.15 sd of the training distribution (`EDA.md` §8) | Tokens are taken relative to each tool's own healthy start (causal) |
| Frozen ImageNet features read wear poorly | Photo probe 0.61-0.67 accuracy against 0.95 possible. Cropping the edge band helps; resolution alone does not | Crop rows 0.30-0.85. DINOv2 patch grid. Wear-regression probe. LoRA as a stretch goal |
| Absolute embeddings plus elapsed time memorize units | Project 14 RUL GRU peaked after 1 epoch (`pdmfm/rul.py`) | Relative tokens, trajectory augmentation (§6.7), small models |
| Wear is not monotone | Flank wear decreases at 189 of 502 steps (F12) | Isotonic-smoothed wear for the auxiliary targets |

### 2.6 The bar to beat (pass ends, fold protocol D1)

The passes window is RUL ≤ 15 passes, capped at 15. The seconds window is
RUL ≤ 1,500 s of cutting, capped at 1,500 s; that is 25 min, about 15 passes
of median length.

| Predictor | Uses | MAE (passes) | RMSE (passes) | MAE (s) | RMSE (s) |
|---|---|---|---|---|---|
| Constant at half the window (7.5 passes; 750 s) | The fact that the point lies in the window, which is the truth itself | 4.00 | 4.33 | 389 | 449 |
| Mean training TTF − elapsed time, capped | Elapsed time | 5.71 | 6.46 | 559 | 631 |
| Conditional residual life E[TTF − t \| TTF > t] over the 8 training tools, capped | Elapsed time | **4.22** | 4.89 | **522** | 589 |

The k-window is selected with the true RUL, so a constant k/2 scores about
k/4 without knowing anything about the tool. The model is not told that it is
inside the window. The deployable baselines are the fair comparison. Both are
reported in every table.

---

## 3. Task definition

### 3.1 Notation

For tool *u* with passes *p* = 1…N_u, in (run, block) order:

- D_p: the trimmed cutting duration of pass *p* in seconds. τ_p = Σ_{q≤p} D_q
  is the cumulative cutting time at its end.
- Pass *p* is cut into segments *s* = 1…n_p: 10 s windows from the start of
  the pass (D7). The end of segment (p, s) lies at elapsed time e_{p,s}
  inside the pass.
- Time in pass units: t_{p,s} = (p − 1) + e_{p,s} / D_p. Time in seconds:
  τ_{p,s} = τ_{p−1} + e_{p,s}.
- P*_u: the confirmed TTF pass (D2). The failure time is the **end** of pass
  P*, the moment the dulled state is first observed, so t_TTF = P*.

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

- Seconds answer "how long can I keep cutting".
- Passes answer "how many more passes".
- The two differ across tools because pass length does (81 s on T9, 122 s on
  T10).
- Inside a pass, the seconds target falls by exactly the elapsed cutting time.
  The passes target also depends on the pass's still-unknown length D_p.

**Points, windows and training.**

- **Prediction points:** every segment end of passes 1…P*. This is 3,747
  points over the 10 tools. Points after the TTF carry no RUL target.
- **Evaluation windows:**
  - passes: RUL ≤ 15 (1,499 points);
  - seconds: RUL^sec ≤ 1,500 s (about 12 passes on T10, 18 on T9, about 15
    elsewhere).
  - In its own window, each target equals the true RUL.
- **Training:** every pre-TTF point, with the capped (piecewise-linear)
  targets.
- **Auxiliary targets** (training tools only):
  - VB̃_p, the isotonic-smoothed flank wear of the current pass;
  - optionally the ordinal wear class;
  - optionally `gaps` and `overhang`.

**Worked example.** T5 has P* = 19. At the end of pass 4, RUL = 15 and the
window opens. At the end of segment 5 of pass 10 (passes of about 97 s),
t = 9 + 50/97 = 9.52 and RUL = 9.48 passes. In seconds: T5 reaches its TTF
after 1,827 s of cutting, so at that point (τ ≈ 923 s) RUL^sec ≈ 904 s, about
15 min of cutting left. At the end of pass 19, both are 0.

### 3.3 What the model may read at point (p, s)

| Input | Available | Never |
|---|---|---|
| Force | Segments of passes < *p*, plus segments 1…*s* of pass *p* | Later segments |
| Photos (tool, chip, workpiece) | Of passes < *p* | Of pass *p* or later |
| Usage | Elapsed passes, elapsed cutting seconds, elapsed seconds in the current pass | The pass duration D_p before the pass ends |
| Labels | None | `image_label`, `tool_label`, flank wear, gaps, overhang, run or block codes |

The causality tests in §15 enforce this.

---

## 4. Cross-validation and model-selection protocol

### 4.1 Folds (identical to OmniFuser's `leave_one_tool_out_folds`)

| Fold | Test | Validation | Training |
|---|---|---|---|
| 1 | T1 | T2 | T3-T10 |
| 2 | T2 | T3 | T1, T4-T10 |
| … | … | … | … |
| 10 | T10 | T1 | T2-T9 |

### 4.2 What is fitted where

| Object | Fitted on |
|---|---|
| Feature standardization, PCA | The 8 training tools |
| Tool baseline (§5.9) | Each tool's **own** first passes, causally. This includes the test tool, because it uses only that tool's past |
| Wear threshold θ and degradation priors (F5) | The training tools |
| Epoch selection and interval calibration | The validation tool |
| Test tool | Prediction only |

### 4.3 Model selection inside a fold

- Evaluate the validation tool's k-window MAE every 10 optimizer steps.
- Keep the weights with the lowest mean of 3 consecutive evaluations. This is
  OmniFuser's F4 `smoothed` rule; one validation tool is noisy.

### 4.4 Design decisions across folds: gates with a rule fixed in advance

- An option is adopted only if two conditions hold, with scores averaged over
  3 seeds per fold:
  - it lowers the **validation** k-window MAE (mean over the 10 folds) by at
    least 0.10 passes;
  - it wins on at least 6 of the 10 validation tools.
- Decisions are made on the passes MAE. An option that worsens the seconds
  MAE by more than 5 % is not adopted, even if it passes both conditions.
- Otherwise the simpler option stays.
- Test scores are never read before the final stage.

**Caveat.** Fold *i*'s validation tool is fold *i+1*'s test tool, so
cross-fold design choices touch every tool once. This is the same protocol as
OmniFuser and Project 14. The plan keeps the number of decisions small (about
14, listed in §17) and states it in the report.

### 4.5 Seeds

3 seeds for gate decisions. 5 seeds for the final runs, whose ensemble is also
the epistemic part of the uncertainty (§9).

---

## 5. Preprocessing pipeline

Every step is a pure function with a unit test. The pipeline is cached by a
signature hash: data manifest + settings.

### 5.1 Load: `rulfuse/data/nonastreda.py`, vendored from OmniFuser

- Reuse OmniFuser's checks:
  - ID parsing (`RECORD_ID`, `_decode_record_id`);
  - CSV reading (`_read_csv`);
  - the MAT label matching the CSV label, finite forces, present images, and
    the `(3, samples)` shape.
- Load **all 512 passes** without deduplication.
- Detect duplicate groups exactly as `load_nonastreda` does (bit-identical
  arrays within a run) and return them as metadata. A test asserts 132 groups
  and 134 non-latest members.
- Add the chip and workpiece paths, and verify them against
  `MANIFEST.json`'s sha256.

### 5.2 Trim: reused unchanged

OmniFuser's span rule `_nonastreda_span_bounds` (20-sample envelope, 5 %
reference window, 50 % threshold), with its documented limit on 2 passes
(`EDA.md` §5).

### 5.3 Physical timeline: `rulfuse/data/timeline.py`

- Order the passes by (run, block). Compute D_p and τ_p.
- Apply the D8 attachment rule. A pass with missing force gets the duration
  of its group's copy, its segments marked `force_present = False`.
- Record each photo's availability time: the end of pass *p*.

### 5.4 Channels

**All three measured channels are always input:** Fx, Fy and Fz. Two derived
channels are added:

| Channel | Definition | Why |
|---|---|---|
| Fx, Fy, Fz | Raw, trimmed | Every measured channel; your instruction |
| Fz_rel | Fz minus the mean of the pass's first second | Removes the drift's sawtooth level (OmniFuser F3) but keeps Fz's dynamics. Causal: the first second is known once it has passed |
| \|F\| | √(Fx² + Fy² + Fz²) | The resultant force, the classic wear indicator |

- **Separate encoding per channel.** Each channel is encoded and PCA-reduced
  separately (16 components per channel, fitted on the training tools), and
  the blocks are concatenated.
- This lets one channel be masked cleanly, for channel dropout (§6.7) and the
  channel study (§11.5).
- **Gate 1** decides only whether the two *derived* channels stay. The three
  raw channels are never removed from the main model.

### 5.5 Segment

- 10 s windows from the start of each pass; a remainder under 5 s joins the
  previous segment. This gives 8 segments per pass on T9, 12 on T10, and 10
  elsewhere.
- Each segment stores: raw arrays (float32), tool, pass, index, start and end
  sample, e_{p,s}, t_{p,s}, τ_{p,s}, and `force_present`.

### 5.6 Labels: `rulfuse/data/labels.py`

- The five TTF rules of §2.3, chosen by config key `ttf_rule`.
- RUL in passes and in seconds, the capped target, the evaluation mask, and
  the pre-TTF mask.
- VB̃ per pass: isotonic regression per tool, `scipy.optimize.isotonic_regression`.
- Ordinal class per pass.

### 5.7 Images: `rulfuse/data/images.py`

Reuse OmniFuser's `load_photo` (crop) and `ImageBank`.

| Photo | Preparation |
|---|---|
| Tool | Crop rows 0.30-0.85 at full width (the OmniFuser probe's best band). Resize to 224 × 672 for DINOv2 (Project 14's setting), or tile into 3 × 224² tiles near native height |
| Chip | Resize to 448² (and 224²) |
| Workpiece | Resize to 224² |

All photos are normalized with the ImageNet statistics.

### 5.8 Stores and caches

**Segment store.**

- Location: `data/interim/segments/<signature>/`.
- Content:
  - one memory-mapped `signals.f32` holding every unique force recording
    (about 450 MB);
  - `segments.parquet` (one row per segment);
  - `passes.parquet` (one row per pass, with labels and availability);
  - `meta.json`.

**Feature caches.**

- Location: `data/interim/features/<modality>-<encoder>-<signature>/`.
- Content: an `(n_items, dim)` float16 array plus an index. The pattern is
  Project 14's `FeatureStore`: the signature names the checkpoint revision and
  the preprocessing, so features of different encoders are never mixed.
- Force features are per segment. Photo features are per pass and photo type.

### 5.9 Normalization

- **Fold level.** Standardize each feature with the training tools' mean and
  sd. Then apply PCA, fitted on the training tools: 32 components per photo
  type, and 16 per force channel (5 × 16 = 80 for the force token).
- **Tool level (causal baseline).**
  - Each token is taken relative to the tool's healthy start:
    $$\tilde z = z - b_u(t),\qquad b_u(t) = \text{mean of } z \text{ over the tool's segments in passes } 1\ldots B \text{ seen up to } t,\quad B=4\ (\text{one run}).$$
  - Inside the first B passes, b_u is the running mean of what has been seen.
  - Photos are taken relative to the tool's first photo.
  - Ablation: absolute tokens, and absolute plus relative.

---

## 6. Model architecture

### 6.1 Overview

```
 pass 1 ......... pass p-1 | pass p: segments 1..s                     time →
 ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ force seg│...│ force seg│...│ force seg│  raw 1 kHz Fx, Fy, Fz (+ Fz_rel, |F|), 10 s
 └────┬─────┘   └────┬─────┘   └────┬─────┘
      ▼ frozen TS encoder on 512-sample sub-windows, mean+sd pooled  ⊕ physics features
      ▼ standardize (training tools) → PCA(16) per channel → minus the tool's causal baseline → MLP → d=64
 photos of passes < p (tool | chip | workpiece)
      ▼ frozen image encoder on the crop → PCA(32) → minus the first photo → MLP → d=64
      ▼ (+ wear readout VB̂ from a ridge probe)
 ─────────────────────────────────────────────────────────────────────────────
 token sequence ordered by availability time  + modality embedding + time embedding
      ▼
 causal temporal backbone with fusion  (F0 late PoE | F1 token fusion | F2 gated
                                         cross-attention | F4 pass-level GRU)
      ▼ at every force-segment token (= prediction point)
 heads: RUL in seconds, Student-t (μ, σ, ν) · RUL in passes, Student-t (μ, σ, ν)
        · current-wear nowcast VB̂ · (ordinal wear)
      ▼
 5-seed ensemble → one scale per unit, calibrated on the validation tool
      → "≈ 640 s (90 %: 420-900 s) of cutting left; ≈ 6.6 passes (4.3-9.2)"
```

Why a hierarchy:

- "All previous data" at 1 kHz is up to 47 passes × 100 s × 1000 × 3,
  about 14 M values: too much for one model.
- Frozen encoders turn each 10 s segment into one token. A tool's full
  history then becomes about 470 force tokens plus about 140 photo tokens.
  That is small for a causal transformer and needs no truncation.
- Only 8 trajectories train each fold. The learnable part must stay small
  (about 0.1-0.3 M parameters) and sit on top of fixed, general
  representations.

### 6.2 Encoders, screened in Gate 1

**Force** (per 10 s segment; sub-windows of 512 samples per channel, about 19
per segment; embeddings mean- and sd-pooled over the sub-windows, channels
concatenated):

| ID | Encoder | Dim per window | Notes |
|---|---|---|---|
| H | Physics features: mean, sd, RMS, peak-to-peak, skewness, kurtosis, crest factor per channel; 16 band energies 0-500 Hz; energy at the 66.7 Hz spindle line and its harmonics (Project 14 EDA F11); spectral centroid and entropy; amplitude of the period-15 synchronous average | about 100 | Always present; also the cheap reference |
| MA | Mantis-8M (`paris-noah/Mantis-8M`) | 256 | Project 14 wrapper |
| MO | MOMENT-1 base / large (`AutonLab/MOMENT-1-base`, `-large`) | 768 / 1024 | Project 14 wrapper for base |
| CH | Chronos encoder embeddings (`amazon/chronos-t5-small` via `pipeline.embed`) | 512 | New wrapper; check the API on install |
| AST | Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) on the segment | 768 | Project 14 wrapper |
| SP | Segment spectrograms (VLT-style STFT, FFT and recurrence maps, OmniFuser `spectrogram.py`) into frozen DINOv2-S | 384 | OmniFuser's best forecasting input |
| PT | Own PatchTST, masked-reconstruction pretrained per fold on the training tools | 128 | Project 14 `patchtst.py`; stretch goal |

**Photos** (per pass and photo type):

| ID | Encoder | Notes |
|---|---|---|
| EB0 | EfficientNet-B0, ImageNet (torchvision) | OmniFuser's encoder; the reference |
| D2-S/B | DINOv2 small or base (`facebook/dinov2-small`, `-base`): CLS plus patch grid pooled to 2 × 4 | Project 14 wrapper |
| SG2 | SigLIP 2 base (`google/siglip2-base-patch16-224`) | New wrapper |
| D3 | DINOv3 ViT-S/B (gated on Hugging Face; check access and licence in Stage 0) | Optional |
| HC | Handcrafted: HSV colour histograms (chip heat tint), LBP and GLCM texture (workpiece), edge-band gradient profile (tool) | Cheap reference |

**Gate 1 (CPU, minutes once the features are cached).**

- For each modality and candidate:
  1. pool to one vector per pass (the mean of its segments, for force);
  2. standardize, apply PCA (§5.9) and make the vector tool-relative;
  3. fit a ridge regression on the 8 training tools to two targets:
     (a) the capped RUL at the end of the pass, (b) VB̃.
- Score each candidate on the validation tool of every fold.
- Choose by validation RUL MAE, with VB MAE as the tie-break, under the §4.4
  rule against H (handcrafted).
- More than one encoder may be kept per modality: the concatenation is also a
  candidate.
- The ridge VB probe of the chosen photo encoder becomes the **wear readout**
  token VB̂ (µm) of each photo.

### 6.3 Tokens and time

| Token | Content | Position in the sequence |
|---|---|---|
| Force segment (p, s) | Projected tool-relative features; `force_present` flag. A missing-force segment uses a learned mask vector | At its end time t_{p,s} |
| Photo (p, m), m ∈ {tool, chip, work} | Projected tool-relative features plus the wear readout VB̂ | Immediately after the last segment of pass *p*, so only from pass *p+1* on (D9) |
| Usage (U) | Fourier features of t (passes / 50) and τ (s / 5000), and elapsed seconds in the current pass | Added to every force token; switchable |

- Each token gets: modality embedding + continuous-time Fourier embedding of
  its availability time.
- A photo token also gets the age embedding: passes since the photo, relative
  to the query.

**A causal model over the full history can infer elapsed time from the token
count**, even when U is off. The clean sensing-only control (§11.4) therefore
uses a recent-window variant: the last 4 passes only, no absolute time
features, and the baseline b_u supplied as a vector.

### 6.4 Fusion candidates, compared in Gate 2

| ID | Design | Why |
|---|---|---|
| **F0** Late fusion, product of experts | One causal model per modality, each with its own Student-t head. Combined by precision-weighted product (Project 14 `nig_fuse` pattern) or a learned per-point gate. A photo expert updates once per pass; inside a pass its mean moves down with elapsed time | Robust to a missing modality; the experts' weights read as time-resolved modality importance; Project 14's pilot found late fusion hard to beat |
| **F1** Token fusion (early-intermediate) | One causal transformer over the interleaved sequence of §6.3. 2 layers, d = 64, 4 heads, FFN 128, pre-LN, dropout 0.2. The prediction is read at every force token | The most flexible; attention learns cross-modal timing (for example "force has risen since the last photo") |
| **F2** Gated cross-attention | A force-stream causal transformer. After each layer: h ← h + tanh(α_m)·CrossAttn(h, M_m,<t) for each photo type *m*, with α_m initialized to 0 (Flamingo-style) | Starts as a force-only model and opens each photo channel only if that helps; the gates α_m are interpretable |
| **F4** Pass-level GRU | Per completed pass: [mean and sd of its force tokens, its photo tokens] → GRU. For the current pass: attention-pool its segments so far, concatenated with the GRU state of pass *p−1* → MLP head | A simple intermediate-fusion baseline; few parameters |
| **F5** Health indicator + extrapolation (two-stage) | See below | Physical, data-efficient, natural intervals |
| (F3) | OmniFuser's C2F or a bottleneck (MBT) fusion | Optional, only if week 3 has slack. OmniFuser showed C2F did not beat concatenation |

**F5 in detail.**

1. **Wear nowcaster.** VB̂(t) comes from the fused model's auxiliary head,
   trained on VB̃ of every training-tool pass. That is dense supervision: about
   400 labelled passes, against 8 failure events.
2. **Degradation prior.** Fit VB(n) = α + β·n^γ (γ ≥ 1) to each training
   tool's smoothed curve. The prior over (α, β, γ) is the 8 fitted triples
   (Gaussian on log β and log γ).
3. **Update.** At time t, importance-weight 5,000 prior draws by the
   likelihood of the tool's VB̂ history, with the noise sd taken from the
   nowcaster's training residuals.
4. **Predict.** RUL samples = first crossing of θ − t, where θ is the training
   tools' median VB̃ at the TTF (about 112 µm), fitted per fold. Cap at 15.
   The samples give the mean and the quantiles directly.

**Controls inside the chosen family.**

- History = last pass only (Project 14's lesson L8: more history did not help
  with 9 units).
- Absolute tokens instead of relative tokens.

### 6.5 Heads

| Head | Parameterization | Use |
|---|---|---|
| **Dual-unit Student-t (default)** | Two heads on the same backbone state h, one per unit u ∈ {s, pass}: μ_u = k_u·sigmoid(w_{μ,u}ᵀh); σ_u = softplus(w_{σ,u}ᵀh) + 0.005·k_u; ν_u = 2 + softplus(w_{ν,u}ᵀh), with k_s = 1,500 s and k_pass = 15 | RUL in seconds and in passes, each with its point value and intervals, at every segment |
| Single-unit control | One seconds head; passes derived as μ_s / D̄, where D̄ is the tool's mean pass duration so far (causal) | Tests whether the second head adds anything |
| Quantile | 7 quantiles 0.05…0.95, made non-crossing by cumulative softplus increments; pinball loss | Distribution-free comparison |
| NIG (evidential) | Project 14 `losses.nig_*` | Aleatoric and epistemic split from a single model |
| Discrete hazard | Hazard over pass bins 0…15+; NLL | The full RUL distribution; optional |
| Wear nowcast VB̂ | Linear | Auxiliary task and the F5 input |
| Ordinal wear (CORAL) | 2 thresholds | Auxiliary task; off by default |

Every interval is clipped to [0, 15].

### 6.6 Loss

$$
\mathcal L = \sum_{u\in\{\mathrm{s},\,\mathrm{pass}\}}\Big[\underbrace{\tfrac1N\sum (\mu_u-y_u)^2/k_u^2}_{\text{MSE (primary)}}
+ \lambda_{\text{nll}}\,\mathrm{NLL}_t\!\big(y_u \mid \mathrm{sg}(\mu_u),\sigma_u,\nu_u\big)
+ \lambda_{\text{cons}}\!\!\sum_{(i,j)}\!\big(\mu_{u,i}-\mu_{u,j}-(T_{u,j}-T_{u,i})\big)^2\!/k_u^2\,\mathbb 1[y_{u,i}<k_u]\Big]
+ \lambda_{\text{vb}}\,\mathrm{Huber}\big(\widehat{VB}-\widetilde{VB}\big)/s_{VB}
$$

- T_u is the time of a point in unit u: τ in seconds, t in passes. Dividing by
  k_u² puts both units on the same scale, so neither dominates.
- sg is stop-gradient. The mean is driven by MSE, as you asked; the spread
  learns from the likelihood without pulling the mean.
- The consistency term rewards the slope −1 that true RUL has between nearby
  points of the same tool (pairs at most 2 passes apart). In seconds it is
  exact even inside a pass: 10 s more cutting means 10 s less RUL.
- Defaults: λ_nll = 0.5, λ_vb = 0.3, λ_cons = 0.1.
- Gate-3 arms: MAE (Laplace) as the point loss, β-NLL, no consistency term.
- MSE and MAE are both always reported (§10).

### 6.7 Regularization and augmentation (8 trajectories per fold)

1. Frozen encoders, PCA (§5.9), d = 64, dropout 0.2, weight decay
   0.05.
2. **Trajectory augmentation (new; the main guard against the usage
   shortcut).**
   - Synthesize new run-to-failure trajectories by randomly deleting
     (probability ρ ~ U[0, 0.2]) or duplicating passes between the baseline
     run and pass P* − 1.
   - Recompute the pass index, the TTF and every RUL target.
   - With cached tokens this is only index manipulation.
   - The resulting tools wear faster or slower per pass, which decorrelates
     elapsed time from RUL and forces the model to read the sensors.
   - Gate-2 ablation.
3. Modality dropout: each photo stream and the force stream drop per sequence
   with p = 0.3. This is required for the masking study (§11.2) and for
   robustness.
4. Token dropout (p = 0.1) and Gaussian feature noise (sd 0.1 in standardized
   units).
5. Channel dropout: each force channel's block is zeroed with p = 0.1 per
   sequence. This keeps the model from depending on one channel, and it is
   needed for the channel-masking study (§11.5).
6. Batching, optimizer, curriculum, selection and diagnostics: §7.

### 6.8 Size

| Part | Parameters |
|---|---|
| Projections: force (80 → 64) + 3 photo types × (32 → 64) | about 12 k |
| F1 backbone | about 70 k |
| Heads | about 1 k |
| **Trainable total** | about 0.1 M |

One fit on cached features takes seconds to a few minutes; to be measured in
the pilot.

---

## 7. Training strategy

This section says how one model is trained, step by step, and why the result
can be trusted when each fold has only 8 training tools.

### 7.1 The learning problem, and seven principles

Per fold, the model learns from three kinds of supervision.

| Supervision | Count per fold (8 training tools) | Character |
|---|---|---|
| RUL targets | About 3,000 prediction points, but only **8 failure events** | Points of one tool are strongly correlated, so the effective sample size is closer to 8 than to 3,000 |
| Current wear VB̃ (one per pass) | About 410 labelled passes, including those after the TTF | Dense, physical, comparable across tools |
| Ordinal wear class | About 410 passes | Noisy near the thresholds (`EDA.md` §2) |

A flexible model can memorize 8 life curves; Project 14's first RUL model did
so within one epoch. Seven principles prevent it.

| # | Principle | Implemented by |
|---|---|---|
| P1 | Learn little; borrow most | Frozen pretrained encoders, PCA (§5.9), about 0.1 M trainable parameters (§6.8) |
| P2 | Dense supervision before sparse | Phase A learns current wear from about 410 passes before the model sees a single RUL target (§7.4) |
| P3 | Give the model nothing that identifies a tool | Tool-relative tokens; no run or block codes; PCA fitted on the training tools; equal loss weight per tool |
| P4 | Break the link between elapsed time and RUL | Trajectory augmentation (§6.7) |
| P5 | Judge only on held-out tools | Selection and calibration on the validation tool; the test tool is read once, in the final stage |
| P6 | Average away training noise | EMA weights, smoothed selection, 5-seed ensembles, paired decisions under a rule fixed in advance |
| P7 | Prove that it learns before trusting it | Sanity checks (§7.7) and diagnostics written by every fit (§7.8) |

### 7.2 The lifecycle of one fit (fold *i*, seed *s*)

Each step writes a file into `runs/<study>/<arm>/fold-T<i>/seed-<s>/`.

1. **Fold setup.**
   - Read the fold manifest: test T*i*, validation T*i+1*, 8 training tools.
   - Assert that the three sets are disjoint.
   - Seed Python, NumPy and PyTorch from (fold, seed).
   - Writes `config.yaml` and `signature.json`.
2. **Fit the transforms, on the training tools only.**
   - Per-modality standardization and PCA; the VB̃ scale.
   - For F5: the threshold θ and the degradation prior.
   - Writes `transforms.npz`.
3. **Fit the wear readout probes, with cross-fitting.**
   - Validation and test tools: one ridge probe per photo type, fitted on all
     8 training tools.
   - Each training tool: a probe fitted on the *other 7* training tools (inner
     leave-one-tool-out).
   - Why cross-fitting: an in-sample readout of a training tool is nearly
     perfect. The network would learn to trust it, then fail on the test tool,
     whose readout is noisier (stacking leakage). Cross-fitted readouts carry
     the same kind of error as held-out ones.
   - Writes `readouts.parquet`.
4. **Build the tensors.**
   - One sequence per tool: token features, modality ids, availability times,
     the masks (`force_present`, photo present), the targets (y, VB̃, class)
     and the point weights (§7.3).
   - All 10 tools fit in GPU memory at once (about 10 × 600 tokens × 32
     features).
5. **Phase A, wear warm-up:** steps 1-100 (§7.4).
6. **Phase B, joint RUL training:** steps 101-500 at most.
   - Every 10 steps, evaluate the validation tool's k-window MAE with the EMA
     weights.
   - Stop when the smoothed score (mean of 3 evaluations) has not improved for
     10 evaluations.
7. **Restore the selected EMA weights.** Writes `model.pt` and
   `selection.json` (selected step and the validation curve).
8. **Predict** every point of the validation, test and training tools. The
   training predictions serve only the fit diagnostics. Writes
   `predictions.parquet`.
9. **Diagnostics** (§7.8). Writes `diagnostics.json`, then the `DONE` marker.

When every seed of a fold has finished, `rulfuse aggregate` does three more
steps:

10. Mixes the seeds' predictive distributions into the ensemble (§9).
11. Fits the calibration scale on the **ensemble's** validation predictions
    (not each member's) and applies it to the test tool.
12. Scores the fold and writes `fold-summary.json`.

### 7.3 One optimizer step

1. **Assemble the batch:** 8 training tools × 4 augmented copies = 32
   sequences.
   - Trajectory augmentation (§6.7):
     - delete or duplicate passes between pass B+1 and P*−1, with
       ρ ~ U[0, 0.2];
     - recompute pass indices, times, the TTF and all targets;
     - photos and wear labels stay with their own passes.
   - Modality dropout: each of F, T, C and W is dropped for the whole sequence
     with p = 0.3; at least one sensing modality is kept.
   - Token dropout 0.1 (force segments and photos); feature noise, sd 0.1.
2. **Forward** through the causal model.
   - One pass gives μ, σ, ν and VB̂ at every force token: about 300-500
     predictions per sequence at once.
   - Causal masking guarantees that each prediction sees only its past;
     `test_causality` proves it.
3. **Weight the points.**
   - Every tool contributes the same total weight, whatever its life length:
     w ∝ 1 / n_tool. T10 has 526 pre-TTF points and T5 only 188, so without
     this the long lives would dominate.
   - Within a tool, points inside a unit's window weigh 1 in that unit's loss
     (RUL ≤ 15 passes; RUL ≤ 1,500 s). Capped points weigh w_far = 0.3, so
     the long healthy plateau does not dominate.
   - Points after the TTF carry only the wear loss.
4. **Loss:** §6.6, with the phase weights of §7.4.
5. **Update.**
   - Backward; clip the gradient norm at 1.0; AdamW step.
   - Update the EMA of the weights (decay 0.99).
   - Log every loss term, the gradient norm, the learning rate and, for F2,
     the gate values.

Cost: about 0.05-0.3 s per step on a CPU, so a 500-step fit takes 1-3 min.
The pilot (J1) measures the real figure.

### 7.4 Curriculum and schedules

| Phase | Steps | Losses (weights) | Learning rate | Purpose |
|---|---|---|---|---|
| A: wear warm-up | 1-100 | VB 1.0 (+ ordinal 0.2 if on) | Linear warm-up to 1e-3 over 20 steps | The backbone learns a tool-invariant wear state from about 410 labels before it sees the 8 failure events |
| B: joint | 101-500 | Per unit (seconds and passes): MSE 1.0, consistency 0.1, NLL ramped 0 → 0.5 over steps 101-150 with the mean detached. Plus VB 0.3 | Cosine, 1e-3 → 1e-5 | RUL as a function of the learned wear state and the history |
| C: calibration | After training | None (no gradient) | – | One spread scale on the validation tool (§9) |

- Selection happens only in phase B. Phase A has a fixed length, so it is not
  a selection on validation.
- Ablation in Gate 2: no phase A (joint training from step 1), decided under
  the §4.4 rule.

**Family-specific rules.**

- **F0 (late fusion).** The experts train jointly with deep supervision:
  loss = fused loss + 0.5 × the mean of the experts' own losses. Each expert
  must stand alone, both for the product-of-experts weights and for a missing
  modality.
- **F2 (gated cross-attention).** The gates α_m start at 0 and get no weight
  decay.
- **F5 (health indicator).** The only trained part is the wear nowcaster: the
  fused model's VB head, or a ridge regression on the tokens. The prior and
  its update are closed-form or sampling.

### 7.5 Hyperparameters: how they are set

- **Defaults** are fixed in Appendix A before any run.
- **One small sweep** runs in week 3, on the chosen backbone only, one factor
  at a time around the defaults:
  - learning rate {3e-4, 1e-3, 3e-3};
  - d {32, 64, 96};
  - dropout {0.1, 0.2, 0.3};
  - PCA {16, 32, 64};
  - ρ_max {0.1, 0.2, 0.3};
  - w_far {0.1, 0.3, 1.0}.

  This is 12 variants × 10 folds × 3 seeds = 360 small fits (job J5b).
- A change is adopted only under the §4.4 rule on the validation tools. The
  other families reuse the result.
- **No per-fold tuning.** One validation tool per fold is too noisy to tune on
  (OmniFuser F4).

### 7.6 Making the result reliable

| Source of unreliability | Countermeasure |
|---|---|
| Seed noise | 3 seeds per gate decision, 5 for the final runs. The reported model is the 5-seed ensemble |
| Noisy selection of one step | EMA weights; smoothed selection; patience. A fixed-step variant (no selection; the median selected step over the 10 folds) is reported to show how much selection matters |
| Long lives dominating the loss | Equal loss weight per tool |
| The healthy plateau dominating | Capped points weigh 0.3 |
| Unstable variance learning | Mean detached in the NLL; NLL ramp; σ floor; calibration after training |
| Tool-to-tool shift | Tool-relative tokens; trajectory augmentation; modality dropout |
| One validation tool per fold | Decisions only from 10 folds × 3 seeds, with a paired rule |
| Stacking leakage of the wear readout | Cross-fitted readouts (§7.2 step 3) |

### 7.7 Sanity checks before any real study (exit criterion of S4)

| Check | Expected result | What it shows |
|---|---|---|
| 1. Overfit one tool, regularization off | Training MAE < 0.1 pass within 300 steps | The plumbing works and the capacity suffices |
| 2. TTFs permuted among the training tools | Validation MAE no better than B2 | No leakage path to the labels |
| 3. Usage-only model (B3) | Within 0.2 passes of B2 | The training loop reproduces a known optimum |
| 4. Oracle token: the true VB̃ of the last pass added as an input | Validation MAE well below B2, toward R1 | The architecture uses a strong signal when one exists |
| 5. Synthetic trajectories: a simulated wear signal with known RUL and noise | MAE near the noise floor; 90 % intervals cover 0.88-0.92 | Learning and uncertainty work end to end |
| 6. Truncation invariance (`test_causality`) | Identical outputs | Causality |
| 7. The same seed twice, on a CPU | Identical weights | Determinism |

### 7.8 Per-fit diagnostics: learning or memorizing?

Each fit writes these to `diagnostics.json`; `rulfuse report --diagnostics`
plots them.

1. **Learning curves** per loss term, training against validation, with the
   selected step. A warning fires when the validation optimum lies in the
   first evaluations while the training loss keeps falling (memorization,
   Project 14's failure).
2. **Validation MAE against B2** on the same tool. Folds where the model does
   not beat B2 are flagged.
3. **Usage reliance.**
   - The correlation of μ with B1's prediction, against its correlation with
     the true RUL.
   - The **time-warp test.** Delete 20 % of the validation tool's passes (a
     faster-wearing version of the same tool). A model that reads the sensors
     shortens its predicted RUL accordingly; a usage-only model does not.
   - Reported as the ratio of predicted shift to true shift: 1 means it reads
     the sensors, 0 means it only counts time.
4. **Modality reliance during training:** validation MAE with each modality
   masked, every 50 steps.
5. **Calibration over steps:** validation coverage of the raw intervals.
6. **Health of the heads:**
   - the share of μ saturated at 0 or 15;
   - the share of σ at its floor;
   - gradient norms, and the F2 gates.
7. **Seed spread:** the standard deviation of validation MAE across seeds,
   per fold. Above 0.3 passes, regularization gets a closer look.
8. **Unit consistency:** μ_s / μ_pass against the tool's mean pass duration so
   far. A large disagreement means one head has learned a tool shortcut the
   other has not.

### 7.9 Failure modes and responses

A response is always a gate option decided on validation tools, never a
reaction to test scores.

| Symptom in the diagnostics | Likely cause | Response |
|---|---|---|
| Predictions track B1 or B2; time-warp ratio near 0 | Usage shortcut | Stronger trajectory augmentation (ρ_max 0.3); check in the Gate-1 probes whether the force tokens carry wear |
| Validation optimum in the first 20-30 steps while training falls | Memorizing the 8 lives | Smaller d or PCA, more dropout, phase A on, absolute tokens off |
| Short-life tools (T4, T5) predicted at the cap of 15 | The plateau dominates; no training tool wears that fast | Lower w_far; more trajectory deletion; F5 |
| σ at its floor; coverage far below nominal | Over-confident spread | Ensemble; a higher floor; conformal calibration |
| Validation coverage fine, but poor on one test tool | That tool is unlike its validation tool | Report it; compare conformal against scale calibration |
| F2 photo gates stay near 0 | The photos are not informative to the model | Check the quality of the wear readout; LoRA (stretch) |
| NaN or exploding loss | Learning rate too high, or the NLL too early | Clipping, the NLL ramp, a lower learning rate |

### 7.10 Training the optional components

- **Per-fold PatchTST (stretch; encoder PT).**
  - Masked-patch reconstruction (40 % of patches masked) on the training
    tools' segments only, for 20 epochs; then frozen.
  - It never sees the validation or test tools.
- **LoRA wear fine-tuning of the photo encoder (stretch; job J10).**
  - DINOv2-B with LoRA rank 8 on the attention q and v projections.
  - Target: VB̃ per photo; Huber loss.
  - Augmentation: small shifts along the edge, ±3° rotation, mild colour
    jitter (not for chip photos, whose tint is signal).
  - 30 epochs, early stopping on the validation tool.
  - Training-tool readouts are cross-fitted with a 4-fold split of the 8
    training tools (4 extra fits per fold), for the reason in §7.2 step 3.
  - Adopted only under the Gate-1 rule against the frozen encoder.

### 7.11 How a study runs, end to end

1. Write the study's configs (`configs/gate2/*.yaml`). Each inherits the
   decisions taken so far.
2. `rulfuse manifest --study gate2` writes the TSV of (arm, fold, seeds).
3. Submit: `sbatch --array=0-49%4 slurm/array.sbatch runs/gate2/manifest.tsv`.
   Each task runs §7.2 for its seeds and skips fits that are already done.
4. `bash slurm/status.sh` until every fit has its `DONE` marker.
5. `rulfuse aggregate --study gate2` produces the per-fold ensembles, the
   calibration, the scores, the paired comparison with the reference arm, and
   `decision.json` from the §4.4 rule.
6. The decision and its numbers go into `PROJECT_LOG.md`. The next study's
   configs inherit it.

Local runs follow the same steps with `--local`: fits run sequentially, with
threads sized by `rulfuse resources`.

---

## 8. Baselines and reference models

Every table shows these.

| ID | Model | Deployable? | Purpose |
|---|---|---|---|
| B0 | Window-aware constant k/2 | No: it uses the window | Shows how easy the window is (MAE 4.00) |
| B1 | Mean training TTF − t, capped | Yes | Pure usage (MAE 5.71) |
| B2 | Conditional residual life given survival to t, capped | Yes | The best usage-only rule (MAE 4.22) |
| B3 | Learned usage-only model: the same head, only U features | Yes | v(∅) of the Shapley study; a learned B2 |
| B4 | Physics force features (H) + ridge, pass level | Yes | Classic tool-wear regression |
| B5 | Photo wear readout + ridge | Yes | The "read the last photo" baseline |
| R1 | **Oracle wear**: the *measured* VB̃ of the last pass + F5 extrapolation | No: measured labels as input | Upper reference: what a perfect photo reader could give |
| R2 | Oracle last-photo class + residual-life table | No | How much the class labels themselves give |

---

## 9. Uncertainty and calibration

1. **Aleatoric:** the Student-t head. **Epistemic:** a 5-seed deep ensemble.
   The predictive distribution is the mixture: mean = avg μ_m; variance =
   avg(σ_m² ν_m/(ν_m − 2)) + var(μ_m).
2. **Calibration on the validation tool.**
   - One scale factor *c* per output unit (seconds and passes) multiplies the
     spread. It is chosen so that the validation tool's 90 % interval has 90 %
     coverage inside that unit's window (Project 14 `metrics.interval_scale`).
   - Alternative: choose *c* by NLL.
3. **Conformal alternative.**
   - Split conformal with normalized scores |y − μ| / σ, using the validation
     tool's k-window as the calibration set.
   - The guarantee assumes exchangeability between tools; the plan states
     this assumption and *measures* the coverage on the test tools rather than
     trusting it.
4. **F5** gives intervals from its posterior samples. **Quantile head**
   intervals come directly from the head.
5. **Gate 3** chooses the head and calibration by validation interval score
   (Winkler, 90 %), then NLL, under the §4.4 rule with interval score in place
   of MAE.

---

## 10. Evaluation

### 10.1 Where and how

- Every model predicts at every prediction point of the test tool in one
  causal pass. The truncation-invariance test proves this equals predicting
  point by point.
- Headline metrics use the k-window. Whole-life metrics (capped targets) and
  pass-end-only metrics come second.
- Points of force-missing passes are scored and also shown separately.

### 10.2 Metrics, by group

| Group | Metrics |
|---|---|
| Point | MAE, MSE, RMSE in passes and in seconds; bias (mean of ŷ − y) |
| Safety | **Late-prediction rate:** share of points with ŷ − y > 1 pass, or > 100 s in seconds (overestimated life). Asymmetric PHM-style score: s = e^{−d/2.6} − 1 for d < 0, e^{d/2.0} − 1 for d ≥ 0, with d = ŷ − y in passes; in seconds, d is converted at 100 s per pass. The constants are fixed here |
| Prognostic (Saxena et al.) | α-λ accuracy (share within ±20 % of the true RUL, at λ = 0.25, 0.5, 0.75 of the window); prognostic horizon: the earliest time after which predictions stay within ±2 passes until failure |
| Uncertainty | Coverage (PICP) and mean width (MPIW) at 50, 80, 90 and 95 %; Winkler interval score; CRPS; NLL; coverage-vs-nominal reliability curve |
| Stability | Monotonicity violation: mean \|Δμ + Δt\| between consecutive points |
| Decision | Replace the tool when the lower 90 % bound ≤ r₀. Over a grid of r₀: missed failures (tools run past the TTF) against wasted life (passes left). Compared with fixed-schedule replacement at quantiles of the training tools' TTF |

### 10.3 Aggregation and statistics

- Per tool, then:
  - mean ± sd over the 10 folds (seeds averaged first);
  - pooled over all 1,499 window points.
- Every summary says which of the two it is.
- 95 % intervals by bootstrap over tools (Project 14 `stats.unit_bootstrap`).
- Model comparisons: paired Wilcoxon signed-rank over the 10 test tools, Holm
  correction across the comparisons of one table.
- Error by RUL bin (0-3, 3-6, …, 12-15 passes) and by position inside the
  pass (OmniFuser's phase-locking lesson, `FINAL_REPORT_REPRODUCTION.md` §6.3).

### 10.4 Figures

1. Per tool: true RUL, predicted mean, 50 % and 90 % bands; the window shaded;
   the photo labels as a strip.
2. Error and coverage against true RUL.
3. Reliability curve.
4. α-λ plot.
5. Decision curve.
6. Shapley bars with intervals.
7. Modality importance over the horizon.

---

## 11. Modality-importance study

Players: **F** (force), **T** (tool photo), **C** (chip photo), **W**
(workpiece photo). Usage U is present in every coalition (a factory always
knows the usage), so the empty coalition is the learned usage-only model B3.

### 11.1 Retraining Shapley (primary)

- Train the chosen architecture on each of the 2⁴ = 16 subsets: 16 × 10 folds
  × 3 seeds = 480 small fits.
- Value: v(S) = −MAE on the test k-window. Importance is analysis, not
  selection, so it reads test scores after the design is frozen.

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\,\big[v(S\cup\{i\})-v(S)\big].
$$

- Report:
  - φ_i per fold, the mean with tool-bootstrap intervals, and the share of
    the total gain over B3;
  - pairwise Shapley interactions (for example F × T);
  - the same for coverage and interval score.

### 11.2 Test-time masking (the model's reliance)

- Use the final model, trained with modality dropout. Mask each subset at
  test time.
- Compare with §11.1: a modality can be used when present yet be
  replaceable.

### 11.3 Time-resolved importance

- **F0:** the precision weights of the experts over the RUL horizon.
- **F2:** the gate values α_m.
- **F1:** integrated gradients and occlusion over history tokens. "How far
  back does the model look", and does the force matter more near failure?
- **Photo staleness:** replace the latest photos with ones 1, 2 or 3 passes
  older (Project 14's staleness test).

### 11.4 Sensing without usage

- Recent-window variant (last 4 passes, no absolute time) with F, with T, and
  with all four.
- This separates "what the sensors know" from "how long the tool has been
  used". It is compared with B2.

### 11.5 Force channels

All five force channels (Fx, Fy, Fz, Fz_rel, |F|) are input (§5.4). Their
contributions are measured inside the force modality in two ways.

1. **Test-time channel masking** on the final model, which was trained with
   channel dropout: the change in test MAE (seconds and passes) when each
   channel, or each pair, is masked.
2. **Retraining check** for Fz (§12): the model without Fz and Fz_rel, against
   the full model, under the same paired test. This answers whether Fz adds
   wear information or only drift.

Reported: per channel, the mean change with tool-bootstrap intervals, and its
dependence on the RUL horizon.

---

## 12. Sensitivity and robustness studies

Each runs the final configuration, 10 folds × 3 seeds.

| Study | Arms |
|---|---|
| TTF rule | Literal first dulled, smoothed wear ≥ 110 µm, `tool_label` = 3, dulled from here on (§2.3) |
| Horizon | k = 10, k = 20 (where every tool allows it), whole life with cap 15 |
| Segment length | 5 s, 20 s |
| Duplicate handling | Attach to every member; OmniFuser's deduplication (§2.4) |
| Force channels | All 5 (default: Fx, Fy, Fz, Fz_rel, \|F\|); the 3 raw channels only; without Fz and Fz_rel (to measure Fz's contribution by retraining) |
| Tokens | Relative (default), absolute, both |
| Trajectory augmentation | On (default if Gate 2 keeps it), off |
| Missing modality at test time | Each modality absent; 50 % of photos missing at random |

---

## 13. Reuse from OmniFuser and Project 14, and lessons learned

### 13.1 What to reuse

Vendor the code, with a provenance header naming the file and commit
(OmniFuser `e3b9985`, branch `Ruche-OmniFuser-Training`; Project 14 `master`).
Do not import it as an editable dependency: OmniFuser's `data.py` has changed
under a dependant before (Project 14 log, week 6), and a single package is
simpler to install on Ruche. A test compares the vendored loader's output with
OmniFuser's on the real data whenever both are present.

| From | File: objects | Used for |
|---|---|---|
| OmniFuser | `data.py`: `RECORD_ID`, `_read_csv`, `_decode_record_id`, `_nonastreda_span_bounds` and its constants, the checks of `load_nonastreda`, `leave_one_tool_out_folds`, `ToolFold`, `ForceNormalizer` (Welford merge), `load_photo`, `ImageBank`, `IMAGE_MEAN`, `IMAGE_STD`, `nonastreda_source_signature` | §5 |
| OmniFuser | `spectrogram.py` (VLT maps) | Force encoder SP |
| OmniFuser | `__init__.py`: `_guard_windows_wmi`; `resources.py`; `tracking.py`; `model_bundle.py` (save and predict); `cross_validation.py` (fold driver, summaries, out-of-fold pooling, resume signatures); `train.py` (interruption-safe resume) | Engineering |
| OmniFuser | `slurm/env.sh`, `job.sbatch`, `chain.sh`, `status.sh`, `checks.sbatch`; `TRAINING_RUNBOOK_RUCHE.md` §3-§7 (SSH, `$WORKDIR`, `.conda` symlink, conda install) | §16 |
| OmniFuser | `scripts/eda.py` (figure style), `scripts/end_to_end_trace.py` (the end-to-end document's numbers) | Stage 1, documentation |
| Project 14 | `encoders.py`: `MantisEncoder`, `MomentEncoder`, `PatchTSTEncoder`, `ASTEncoder`, `DinoV2Encoder`, `hf_revision` | §6.2 |
| Project 14 | `features.py`: `FeatureStore`, `extract_images` (views) | §5.8 |
| Project 14 | `losses.py`: `nig_*`, `nig_fuse`; `metrics.py`: `student_t_nll`, `interval_scale`, `phm2012_score`, `ece`; `stats.py`: `unit_bootstrap` | §6.5, §9, §10 |
| Project 14 | `rul.py`: `relative_to_start`, `SnapshotGRU`; `planb.py`: `_schedule` (schedule baseline), `_calibrate`; `gate2.py`: `_paired`, `decide` (rule machinery) | §5.9, §8, §4.4 |
| Project 14 | `explain.py`: occlusion and deletion curves; `leakage.py`: context checks; `resources.py`: `claim_resources` | §11, §15 |
| Project 14 | `slurm/prefetch_weights.sh`, `slurm/cpu.sbatch`, offline Hugging Face cache in `env.sh`; `docs/RUCHE_RUNBOOK.md` (partition table) | §16 |

### 13.2 Lessons learned, and how this plan applies them

| # | Lesson (source) | Applied as |
|---|---|---|
| L1 | Same-pass photos leak the label (OmniFuser F1) | D9, plus a test |
| L2 | The run number is a 71 % shortcut (F2) | No run or block codes; usage explicit and switchable; schedule baselines B1-B3 |
| L3 | Fz drifts (F3) | All three channels input; the drift is neutralized by the offset-removed Fz_rel; Fz's contribution is measured (§11.5, §12) |
| L4 | One validation tool is noisy; replicates differ by up to 0.017 (F4, F14) | Smoothed selection, 3 or 5 seeds, rule fixed in advance, few decisions |
| L5 | The cross-tool gap (training 0.88, test 0.67), not under-fitting | Frozen encoders, PCA, relative tokens, trajectory augmentation, about 0.1 M trainable parameters |
| L6 | Frozen ImageNet features read wear poorly; a crop helps; fine-tuning helps most | Edge crop, DINOv2 grid, wear probe; LoRA fine-tuning as a stretch goal (§17) |
| L7 | Absolute embeddings plus elapsed time memorize the training units (Project 14 `rul.py`) | Relative tokens by default; trajectory augmentation |
| L8 | Longer history did not help with 9 units (Project 14 XJTU-SY) | "Last pass only" control inside the chosen family |
| L9 | Spectrograms of the raw force gave OmniFuser's best forecasts | Encoder SP in Gate 1 |
| L10 | C2F and the image GAT gained nothing over concatenation | Not a default; F3 optional |
| L11 | Labels are noisy and wear is non-monotone (F12) | Confirmed TTF; isotonic wear; TTF sensitivity |
| L12 | Forecast error phase-locked to pass boundaries (OmniFuser report §6.3) | Prediction at every segment; error by position in the pass |
| L13 | Laptop: WMI hangs, other jobs hold the CPU (memory notes) | WMI guard; `claim_resources` before heavy local work; bare `python` hung in this session, so use the venv interpreter |
| L14 | Ruche's 24 h limit and interruptions | Resumable fits, arrays that skip finished tasks, `chain.sh` |

---

## 14. Repository and engineering

### 14.1 Layout

```text
RUL Model/
  pyproject.toml            package "rulfuse"; extras [test, eda, dev, encoders]
  configs/                  YAML: data.yaml, main.yaml, gate1/*.yaml, gate2/*.yaml, ...
  rulfuse/
    __init__.py             WMI guard (vendored)
    config.py               dataclasses DataConfig, ModelConfig, TrainConfig; YAML I/O; signature()
    data/
      nonastreda.py         vendored loader, all 512 passes, duplicate groups, chip/work paths
      timeline.py           physical timeline, D8 attachment, availability times
      segments.py           10 s segmentation, segment store (memmap + parquet)
      labels.py             5 TTF rules, RUL (passes, s), caps, masks, isotonic wear
      images.py             crops, resizing, ImageBank (vendored)
      folds.py              leave_one_tool_out_folds (vendored), fold manifests
    features/
      physics.py            handcrafted force features; colour and texture image features
      encoders.py           vendored Project 14 wrappers + Chronos, SigLIP 2, DINOv3
      cache.py              FeatureStore (vendored), signatures
      probe.py              Gate-1 ridge probes, wear readout
    models/
      tokens.py             PCA, causal baseline, projections, modality and time embeddings
      backbones.py          causal transformer, GRU, recent-window variant
      fusion.py             F0 late PoE, F1 tokens, F2 gated cross-attention, F4 pass-level GRU
      heads.py              Student-t, quantile, NIG, hazard, wear, CORAL
      hi.py                 F5 nowcaster + degradation prior + importance-sampling update
    augment.py              trajectory augmentation, modality and token dropout
    losses.py
    train.py                fit one (fold, seed): steps, smoothed selection, resume
    uncertainty.py          ensemble mixture, scale calibration, split conformal
    baselines.py            B0-B5, R1, R2
    evaluation.py           §10 metrics
    importance.py           Shapley (retraining and masking), staleness, IG/occlusion
    stats.py                bootstrap, Wilcoxon + Holm (vendored)
    cross_validation.py     10-fold driver, per-fold outputs, out-of-fold pooling
    tracking.py, resources.py, bundle.py   (vendored)
    cli.py                  rulfuse <command>
  scripts/                  eda_rul.py, end_to_end_trace.py, make_figures.py
  slurm/                    env.sh, job.sbatch, cpu.sbatch, array.sbatch, chain.sh, status.sh,
                            checks.sbatch, prefetch_weights.sh
  tests/                    §15
  docs/                     EDA_RUL.md, END_TO_END.md, RUCHE_RUNBOOK.md, RESULTS.md
  Data/                     forces_xyz_raw.mat, labels*.csv, tool/, chip/, work/ (not in git)
  runs/                     outputs (git-ignored)
```

### 14.2 Commands

```bash
rulfuse verify-data   --data-dir Data                          # hashes, counts, duplicate groups
rulfuse build-store   --config configs/data.yaml               # timeline, segments, labels
rulfuse features      --modality force --encoder mantis --device cuda
rulfuse probe         --modality tool --encoders eb0 dinov2-s siglip2 hc   # Gate 1
rulfuse cv            --config configs/gate2/f1.yaml --seeds 0 1 2 --out runs/gate2/f1
rulfuse baselines     --out runs/baselines
rulfuse importance    --config configs/final.yaml --subsets all --seeds 0 1 2 --out runs/shapley
rulfuse report        --runs runs/final runs/baselines --out runs/report
rulfuse predict       --model runs/final/fold-T3/seed-0/model.pt --tool T3
```

### 14.3 Outputs of every fit

Folder: `runs/<study>/<arm>/fold-T<i>/seed-<s>/`.

| File | Content |
|---|---|
| `config.yaml` | The full resolved configuration |
| `signature.json` | Data and feature signatures |
| `metrics.jsonl` | Per evaluation step |
| `predictions.parquet` | One row per prediction point: tool, pass, segment, t, τ; for each unit (seconds and passes): y, RUL, μ, σ, ν, quantiles; the experts' μ and σ; masks |
| `model.pt` | The weights |
| `DONE` | Marker; arrays skip a fit that has it |

`rulfuse report` regenerates every table and figure from the prediction files
(S5).

### 14.4 Code quality

- pytest, ruff and pyright, as in OmniFuser.
- Deterministic seeds (`torch.use_deterministic_algorithms` where available).
- Every config field is part of the signature.
- `docs/END_TO_END.md` is regenerated by `scripts/end_to_end_trace.py` after
  every substantial pipeline change. This follows your standing OmniFuser
  instruction.

---

## 15. Testing and validation of the code

Synthetic tests run in CI in under a minute. Real-data tests are marked
`@pytest.mark.data` and skip when `Data/` is absent.

| Test file | What it asserts |
|---|---|
| `test_loader.py` | 512 passes; MAT and CSV labels agree; 132 duplicate groups and 134 non-latest members; photo hashes match `MANIFEST.json`; the vendored loader equals OmniFuser's on shared fields (data) |
| `test_timeline.py` | D8: the recording goes to the latest member, the others are force-missing with the copy's duration; τ is monotone; photo availability = end of its pass |
| `test_segments.py` | 10 s segments, remainder rule, sample bounds cover the trimmed pass exactly; segment count 3,747 before the TTF (data) |
| `test_labels.py` | Each TTF rule on hand-made label sequences (flips, runs, ties); the per-tool TTF table of §2.3 (data); RUL arithmetic in passes and seconds; caps and masks; isotonic smoothing is non-decreasing |
| `test_folds.py` | 10 folds; test ≠ validation ≠ training; every tool is tested once |
| `test_normalization.py` | Scalers and PCA see only the training tools; the causal baseline at t uses only segments up to t |
| `test_causality.py` | **Truncation invariance:** for every model, the outputs at t are unchanged when all tokens after t are removed or replaced by noise. **No-photo-leak:** changing pass *p*'s photo changes no prediction inside pass *p* or earlier |
| `test_leakage.py` | Shuffling the test tool's future changes nothing; training on permuted RUL targets scores no better than B2 (sanity); no label column reaches the model inputs |
| `test_augment.py` | After trajectory augmentation, the TTF and RUL are consistent with the new pass order; photos stay attached to their passes |
| `test_models.py` | Shapes, masking of missing modalities, F2 gates start at 0, the head's output range; a tiny overfit test (one tool, loss drops below ε) |
| `test_uncertainty.py` | Mixture moments; the calibration scale reaches the target coverage on a synthetic Gaussian; conformal coverage on exchangeable synthetic data |
| `test_metrics.py` | MAE, RMSE, PHM score, α-λ, prognostic horizon, PICP, MPIW, Winkler and CRPS against hand-computed values; Student-t CRPS against Monte Carlo |
| `test_importance.py` | Shapley on a synthetic additive game recovers the known values; efficiency: Σφ = v(N) − v(∅) |
| `test_baselines.py` | B1 and B2 reproduce §2.6 on the real labels (data): MAE 5.71 and 4.22 passes; 559 and 522 s |
| `test_training.py` | Per-tool point weights sum to the same value for every tool; capped points get w_far; phase A uses no RUL loss; the NLL ramp; EMA update; cross-fitted readouts of a training tool never use that tool's labels; both unit heads receive gradient and their losses are scaled by k_u |
| `test_pipeline_smoke.py` | One fold, one seed, the stats encoder, 20 steps: end to end on CPU in under a minute; resume after a simulated interruption gives identical weights |

**Validation of the science.** The tests prove the code does what the plan
says. Three further checks probe whether the plan's claims hold:

1. Baselines B0-B2 reproduce §2.6.
2. Oracle R1 beats every deployable model. If it does not, the extrapolation
   step is broken.
3. Masking the force on a force-only model returns it to B3.

---

## 16. Training on Ruche

### 16.1 What runs where

| Work | Where |
|---|---|
| Development, tests, data store, physics features, Gate-1 probes, baselines, F5 | **Laptop CPU**: check idle CPU first with `rulfuse resources`; the stores take minutes |
| Pretrained encoder features (MOMENT-large, AST, Chronos, DINOv2-B, SigLIP 2, DINOv3) | Ruche `gpua100` |
| Fit arrays (Gates 2-3, Shapley, sensitivity, final) | Ruche `gpua100`, or a CPU partition if the queue is long (the models are small) |
| LoRA photo fine-tuning (stretch) | Ruche `gpua100` |
| Aggregation and report | Laptop or `cpu_short` |

### 16.2 One-time setup

Follows `TRAINING_RUNBOOK_RUCHE.md` §3-§7; differences are noted.

1. `ssh $Ruche`. Then `cd $WORKDIR` (500 GB, **not backed up**). The
   `~/.conda` symlink goes to `$WORKDIR/.conda` (home quota 50 GB).
2. Environment:

   ```bash
   module load anaconda3/2023.09-0/none-none
   conda create -n rulfuse python=3.11 -y && source activate rulfuse
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install --no-cache-dir -e ".[test,encoders]"
   ```

   The `encoders` extra holds `transformers`, `peft`, `momentfm`,
   `mantis-tsfm` and `chronos-forecasting`.
3. `slurm/env.sh`, based on Project 14's:
   - loads the module and activates `rulfuse`;
   - sets `RULFUSE_DATA_ROOT=$WORKDIR/rul_data` and
     `HF_HOME=$WORKDIR/hf_cache`;
   - sets `HF_HUB_OFFLINE=1` inside jobs, plus `PYTHONUNBUFFERED=1`.
4. Weights: run `bash slurm/prefetch_weights.sh` **on the login node**. It
   downloads every checkpoint of §6.2 into `$HF_HOME`; DINOv3 needs its
   licence accepted first.
5. Data: on the laptop, `tar -czf rul-data.tar.gz Data` (force, labels, tool,
   chip and workpiece photos; about 1.3 GB). Then `scp` it to
   `/workdir/<login>/`, and on Ruche run `rulfuse verify-data` (hashes and
   counts).
6. Code: git clone of the private repo, on a `ruche` branch (§19). Otherwise a
   tarball, as in the OmniFuser runbook §5.

### 16.3 SLURM scripts

| Script | Content |
|---|---|
| `job.sbatch` | One command on one GPU: `gpua100`, 8 CPUs, 24 h, `--export=NONE`, threads = `SLURM_CPUS_PER_TASK` (OmniFuser's, unchanged) |
| `cpu.sbatch` | The same on a CPU partition (Project 14's) |
| `array.sbatch` (new) | `#SBATCH --array=0-N%4`. Task *i* reads row *i* of a manifest TSV (arm, fold, seeds, config, output) and runs `rulfuse cv-one ...`. Rows with a `DONE` marker exit at once. `%4` respects the 4-GPU limit per user on `gpua100` |
| `chain.sh` | Only for jobs that can exceed 24 h |
| `status.sh` | Jobs, plus finished, running and missing fits per study, plus logs with errors |
| `checks.sbatch` | Tests; offline weight loading; GPU check |

### 16.4 Job plan

GPU-hours are estimates; the pilot (J1) measures them.

| Job | Content | Partition | Tasks | Estimate |
|---|---|---|---|---|
| J0 | `checks.sbatch`: tests, offline weights, GPU matmul | `gpu_test`, `cpu_short` | 2 | 15 min |
| J1 | Pilot: one fold, one seed, F1 on cached features; time per fit | `gpu_test` | 1 | 30 min |
| J2 | Build the store on Ruche; compare its signature with the laptop's | `cpu_short` | 1 | 15 min |
| J3 | Force features: one task per encoder (MA, MO-base, MO-large, CH, AST, SP) | `gpua100` | 6 | 0.2-1 h each |
| J4 | Photo features: EB0, D2-S/B, SG2, (D3) × 3 photo types | `gpua100` | 5 | < 0.5 h each |
| J5 | Gate 2: 5 arms × 10 folds (3 seeds sequential per task) | `gpua100` | 50 | about 10 GPU-h |
| J5b | Hyperparameter sweep (§7.5): 12 variants × 10 folds (3 seeds per task) | `gpua100` or CPU | 120 | about 8 GPU-h |
| J6 | Gate 3: 4 heads × 2 calibrations × 10 folds | `gpua100` | 80 | about 10 GPU-h |
| J7 | Shapley: 16 subsets × 10 folds × 3 seeds | `gpua100` or CPU | 160 | about 15 GPU-h |
| J8 | Sensitivity (§12): about 14 arms × 10 folds × 3 seeds | `gpua100` or CPU | 140 | about 15 GPU-h |
| J9 | Final: 10 folds × 5 seeds + masking study + staleness | `gpua100` | 10 | about 3 GPU-h |
| J10 | LoRA wear fine-tuning of the photo encoder (stretch): 10 folds × 3 photo types | `gpua100` | 30 | about 8 GPU-h |
| J11 | Report: aggregation and figures | `cpu_short` | 1 | 15 min |

The total is about 60-80 A100-hours: modest, but it needs the queue across
weeks 2-6. Submit arrays in the evening; `status.sh` each morning.

### 16.5 Safe practice

- Every fit is resumable. A task re-run skips fits with a `DONE` marker and
  finishes an interrupted fit from its last checkpoint (OmniFuser pattern).
- Every array manifest is committed with its study, so a rerun is exactly
  reproducible.
- Retrieval: `rsync -av --exclude '*.pt' ruche:$WORKDIR/RUL/runs/ runs/` (no
  checkpoints). Checkpoints of the final runs only: `rulfuse export`.
- `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`;
  OmniFuser runbook §9 lists the common pending reasons.

---

## 17. Stages and timeline

Each stage ends with its exit criteria met and its documents updated.

| Week | Dates | Stage | Main work | Exit criteria |
|---|---|---|---|---|
| 1 | Nov 4-10 | **S0 Setup** | git repository, package skeleton, vendored modules with provenance, CI (pytest, ruff, pyright), copy the chip and workpiece photos into `Data/`, `verify-data` | `test_loader` green; vendored loader equals OmniFuser's |
| 1 | | **S1 RUL EDA** (`docs/EDA_RUL.md`) | TTF rules (§2.3); per-tool wear curves (raw and isotonic) with TTF marks; force indicators (relative RMS, band energy) against RUL per tool; photo wear-probe preview; usage baselines; correlation of each modality's pass-level features with RUL; pass-duration differences T9 / T10 | §2 numbers reproduced by script |
| 1-2 | Nov 8-12 | **S2 Data pipeline** | Timeline, segments, labels, folds, stores, causal baseline; every §15 data test | `test_timeline`, `test_segments`, `test_labels`, `test_causality` (data part) green |
| 2 | Nov 11-17 | **S3 Baselines and features** | B0-B5, R1, R2 (laptop); physics features (laptop); Ruche setup and J0-J4 | §2.6 reproduced; every feature cache present with signatures |
| 2 | | **Gate 1** | Encoder probes per modality (laptop) | One encoder (or concatenation) per modality; decisions 1-4 logged |
| 3 | Nov 18-24 | **S4 Models + Gate 2** | Sanity checks §7.7 first; then F0, F1, F2, F4, F5; trajectory augmentation and phase-A ablations; "last pass only" and absolute-token controls (J5); the hyperparameter sweep on the chosen family (J5b) | All seven sanity checks pass; fusion family chosen (decision 5); augmentation, derived-channel, token, phase-A and sweep options decided (decisions 6-8, 13-14) |
| 4 | Nov 25 - Dec 1 | **S5 Uncertainty + Gate 3** | Heads, ensemble, scale calibration, conformal (J6). Stretch: LoRA wear fine-tuning (J10), adopted only if it passes Gate 1's rule | Head and calibration chosen (decisions 9-10); validation coverage within [0.85, 0.95] |
| 5 | Dec 2-8 | **S6 Importance + S7 Sensitivity** | Freeze the design. Shapley (J7), masking, staleness, time-resolved importance; §12 arms (J8) | All arms finished; importance tables drafted |
| 6 | Dec 9-14 | **S8 Final runs** | 10 folds × 5 seeds (J9); first and only read of the test scores for the headline; decision curve | S1-S5 assessed and written up |
| 6 | Dec 14-16 | **S9 Report** | `docs/RESULTS.md` (methods, results, importance, limitations), `docs/END_TO_END.md`, figures, optional slides | Every number regenerates via `rulfuse report` |

**Buffer.** About 2 days are absorbed by running stages S3 and S4 in parallel
(laptop and Ruche). If week 3 slips, F3 and the hazard head are dropped first,
then LoRA, then the 20 s segment arm.

**Decision log** (about 14; each is appended to `PROJECT_LOG.md` with its
numbers):

1-4. Encoder per modality
5. Fusion family
6. Trajectory augmentation
7. Derived force channels (Fz_rel, |F|): keep or drop
8. Token mode
9. Head
10. Calibration
11. LoRA (if run)
12. History length
13. Phase-A wear warm-up
14. Hyperparameter sweep adoptions

---

## 18. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 10 tools give wide intervals; the gaps between models are not significant | High | Paired tests over tools; report effect sizes with intervals; few, pre-registered comparisons; pooled and per-tool results both |
| The model leans on elapsed time (usage) and adds little over B2 | Medium-high | Trajectory augmentation; B2 and B3 in every table; the sensing-without-usage study (§11.4) reports it honestly |
| The force carries little RUL information (Project 14: the force pathway was flat) | Medium | Physics features and spectrograms in Gate 1; tool-relative tokens; F5 uses force only through the wear nowcast |
| Photo features stay weak | Medium | Crop, DINOv2 grid, wear probe; LoRA stretch goal; the chip photo as a second visual cue |
| Short-life tools (T4, T5) are mispredicted (all usage baselines fail there) | High | Report them; trajectory augmentation creates short lives; F5 adapts to the observed wear rate |
| Label noise near the TTF (T3, T5) | Medium | The confirmed rule; TTF sensitivity arms; per-tool reporting |
| One validation tool miscalibrates the intervals | Medium | Conformal and NLL scaling compared; coverage on test tools reported per tool; the ensemble widens the intervals |
| The 6-week scope slips | Medium | Stage order puts the minimum viable result first (baselines + F1 + Student-t + Shapley); the drop list in §17 |
| Ruche queue or Hugging Face checkpoints unavailable (gated DINOv3) | Low-medium | Small models also run on CPU partitions; DINOv3 is optional; prefetch in week 2 |
| A causality bug (the most damaging kind) | Low after tests | `test_causality` truncation invariance on every model; the oracle and permutation sanity checks |

---

## 19. Open questions for you

Each has a default the plan assumes until you answer.

1. **Commits.** Should I commit to a new private GitHub repository, or will
   you commit personally, as in Project 14? *Default: I leave the changes
   uncommitted and propose commit groups.*
2. **Ruche.** Is your account still active, with the same `$WORKDIR` and
   enough quota (about 20 GB for this project)? Any GPU-hour allocation limit?
   *Default: the same setup as OmniFuser, 60-80 A100-hours.*
3. **Cutting conditions.** T9's passes last about 81 s and T10's about 122 s,
   against about 97 s elsewhere. Does the Nonastreda paper give a feed or
   speed per tool? If so, it would enter as a context input. *Default: not
   used; the duration differences are absorbed by cutting time and the
   tool-relative tokens.*
4. **Usage in the headline model.** The plan treats elapsed cutting time as a
   legitimate input (a factory knows it) and reports sensing-only separately
   (§11.4). Agree? *Default: yes.*
5. **LoRA fine-tuning of the photo encoder.** It is the biggest image lever
   (OmniFuser), but a stretch goal in 6 weeks. *Default: attempted in week 4
   only if Gates 1-2 are on time.*
6. **Deliverables.** A written report in `docs/RESULTS.md` only, or also
   slides or a paper-style write-up? *Default: report + figures; slides
   optional in week 6.*

---

## Appendix A: default settings

| Setting | Default | Section |
|---|---|---|
| TTF rule | Confirmed first dulled (dulled, then ≥ 2 dulled among the next 3) | D2 |
| k, cap | 15 passes, 15 | D3 |
| Seconds window, cap | 1,500 s of cutting, 1,500 s | D6, §3.2 |
| Output heads | Dual-unit Student-t (seconds and passes), shared backbone | §6.5 |
| Segment | 10 s, remainder < 5 s merged | D7 |
| Channels | Fx, Fy, Fz (always) + Fz_rel, \|F\| (derived); PCA 16 per channel | §5.4 |
| Duplicates | Attached to the latest member; the others force-missing | D8 |
| Photo crop (tool) | Rows 0.30-0.85, full width, 224 × 672 | §5.7 |
| PCA | 32 per photo type; 16 per force channel (80 per force token); fitted on the training tools | §5.9 |
| Baseline | First B = 4 passes, causal running mean | §5.9 |
| Backbone | Causal transformer: 2 layers, d = 64, 4 heads, FFN 128, dropout 0.2 | §6.4 |
| Head parameterization | Per unit: μ = k_u·sigmoid, σ floor 0.005·k_u, ν = 2 + softplus | §6.5 |
| Loss weights | MSE 1, NLL 0.5 (mean detached), VB 0.3, consistency 0.1 | §6.6 |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05, cosine, 20 warm-up steps, ≤ 400 steps | §6.7 |
| Batch | 8 training tools × 4 augmented copies | §6.7 |
| Augmentation | Trajectory ρ ~ U[0, 0.2]; modality dropout 0.3; token dropout 0.1; feature noise 0.1 | §6.7 |
| Selection | Validation k-window MAE, mean of 3 evaluations, every 10 steps | §4.3 |
| Seeds | 3 (gates), 5 (final) | §4.5 |
| Curriculum | Phase A: wear only, steps 1-100; phase B: joint, steps 101-500, NLL ramp over 50 steps | §7.4 |
| Point weights | Equal total per tool; capped points 0.3 | §7.3 |
| Stability | EMA 0.99; gradient-norm clip 1.0; patience 10 evaluations | §7.2, §7.3 |
| Wear readouts | Ridge probes, cross-fitted on the training tools | §7.2 |
| Calibration | One spread scale for 90 % coverage on the validation tool | §9 |
| Gate rule | ≥ 0.10 passes lower validation MAE and ≥ 6 of 10 validation-tool wins | §4.4 |

## Appendix B: how the numbers in this plan were measured

All measurements were taken on 2026-11-04 with OmniFuser's virtual
environment (`../OmniFuser/.venv/Scripts/python.exe`); the scripts are in this
session's scratchpad and will move to `scripts/eda_rul.py` in S1.

- §2.2, §2.3: per-tool label sequences from `Data/labels.csv`; trimmed
  durations from OmniFuser `experiment_results/eda/records.csv`
  (`trim_end − trim_start`); isotonic wear by
  `scipy.optimize.isotonic_regression`; segments as round(D_p / 10 s).
- §2.4: duplicate groups as (tool, run, samples, force sum) collisions,
  identical to OmniFuser's bit-identity groups (132); photo hashes from
  `D:\Project14\data\raw\nonastreda\MANIFEST.json`.
- §2.6: baselines at pass ends, fold protocol D1, the TTF of D2, window
  RUL ≤ 15.

## Appendix C: references

- OmniFuser: arXiv:2511.01320; this group's reproduction in `../OmniFuser`
  (`FINAL_REPORT_REPRODUCTION.md`, `FIXES_AND_VALIDATION.md`,
  `FUTURE_WORK_AND_ANALYSIS.md`, `EDA.md`).
- Project 14: `../Multimodal AI for Predictive Maintenance` (`PROJECT_LOG.md`,
  `docs/PLAN.md`, `docs/RUCHE_RUNBOOK.md`, `pdmfm/`).
- Nonastreda: H. Truchan, Z. Ahmadi, *Data in Brief* (2025),
  doi:10.17632/m892d2wtzh.1.
- Prognostic metrics: Saxena et al., "Metrics for evaluating performance of
  prognostic techniques", PHM 2008 (α-λ, prognostic horizon); Saxena et al.,
  C-MAPSS scoring function (2008).
- Deep ensembles: Lakshminarayanan et al., NeurIPS 2017. Deep evidential
  regression: Amini et al., NeurIPS 2020. β-NLL: Seitzer et al., ICLR 2022.
  Split conformal: Lei et al., JASA 2018.
- Flamingo gated cross-attention: Alayrac et al., NeurIPS 2022. Shapley
  values for features or modalities: Lundberg & Lee, NeurIPS 2017.
- Encoders: MOMENT (Goswami et al., ICML 2024), Mantis (2025), Chronos (Ansari
  et al., 2024), AST (Gong et al., 2021), DINOv2 (Oquab et al., 2024), SigLIP 2
  (2025), DINOv3 (2025).
