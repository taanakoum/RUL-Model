# Review of the project plan (version 1 → version 2)

2026-11-04. This review read every section of plan version 1
(`docs/archive/PROJECT_PLAN_v1.md`). It checked each decision against the data
and against the rest of the plan, and recomputed the key numbers. Version 2
(`PROJECT_PLAN.md`) applies every resolution below.

Severity:

- **Critical:** wrong labels or leakage; the results would be invalid.
- **Major:** biased results, or a rule that cannot be applied as written.
- **Minor:** an inconsistency or unclear wording.

## 1. Summary

- **Three critical problems, all fixed in version 2:**
  - Wrong RUL labels for T2 and T4, because runs are missing from the
    record.
  - Two leaks of the current pass's still-unknown length into the inputs.
  - A diagnostic whose logic was wrong.
- **The headline bar moved.** Recomputed correctly, the deployable bar is
  **3.82 passes and 467 s**, not 4.22 passes and 522 s.
- **Your instructions** are applied: two fusion finalists, the others
  documented as alternatives.
- **Calibration** now uses 8 tools of out-of-fold data instead of the
  validation tool alone. The validation tool becomes an independent check.
- **Test-score policy.** A design freeze on 2026-12-01 replaces the
  contradictory "first read of the test scores" in week 6.
- **Verification.** Every part of the pipeline has tests, real-data checks,
  a scientific validation and an acceptance criterion (plan §15).
- **Stages.** The project is cut into 10 stages, each with dates, outputs,
  verification and an exit gate (plan §17).
- **One input needed from you:** confirm the gap-aware time axis (§3 below).

## 2. Findings and resolutions

| # | Severity | v1 section | Finding | Resolution in v2 |
|---|---|---|---|---|
| R1 | Critical | §2, §3 | **Runs missing from the record.** T4 lacks runs 6-9, T2 run 10, T1 run 1. Wear rises 41.7 µm across T4's gap, 5.6× the median per-run rise, so the tool kept cutting. On the recorded axis every T4 point before the gap gets an RUL 16 passes too short (T2: 4), in 8 of 10 training folds, and T4 looks like the fastest-wearing tool | Gap-aware axis D12 (§2.5): interior gaps count as unrecorded cutting; T4's TTF becomes 38, T2's 44. The recorded axis becomes a tier-1 sensitivity arm. **Needs your confirmation** |
| R2 | Critical | §6.3, §3.3 | **Usage input leaked the pass length.** The input used fractional elapsed passes t = (p − 1) + e/D_p, and D_p is unknown until the pass ends. Time embeddings were defined in pass units | Inputs use completed passes, elapsed seconds and seconds in the pass only. t is a label time (§3.1, §3.3, D14). Test P10 |
| R3 | Critical | §5.2 | **Trimming leaked the pass length.** OmniFuser's trim uses the first 5 % *of the pass length* as its reference | Causal trim: reference over the first 5 s. Agreement with OmniFuser's trim is verified in Stage 1 (P2); the old rule becomes a tier-2 arm |
| R4 | Critical | §7.8 | **The time-warp diagnostic was wrong.** Deleting past passes does not change the true RUL at a point, so the "ratio of predicted to true shift" divided by zero, and the text said the sensing model would shorten its RUL | Usage sensitivity s_U = (increase in predicted RUL) / (elapsed time removed): 0 means it reads the sensors, 1 means it counts time (§7.8) |
| R5 | Major | §6.3 | A photo's "age relative to the query" cannot be a token embedding: a key is shared by all queries | A learned relative-time attention bias b_h(Δτ) per head (§6.3) |
| R6 | Major | §9 | Calibration used the validation tool, which also selected the training step, so the intervals come out too narrow. One tool is also little data | Out-of-fold calibration from 4 inner fits on the 8 training tools (default candidate). The validation tool's coverage becomes an independent check before any test score (§9, §7.2) |
| R7 | Major | §2.6 | Baselines were computed at pass ends while the model predicts at segment ends, and B2 put the cap outside the expectation (not the MSE-optimal usage rule for capped targets) | Recomputed at segment ends with the cap inside, gap-aware: B2 3.82 passes / 467 s; B0 4.13 / 412 s; B1 5.18 / 513 s (§2.7). S1 updated |
| R8 | Major | §11.1 | The Shapley empty coalition was a usage-only model trained with trajectory augmentation. That augmentation exists to destroy usage information, so it would inflate every sensing modality's importance | v(∅) = the better of B2 and B3 trained without trajectory augmentation (§11.1) |
| R9 | Major | §6.7 | Modality dropout always kept one sensing modality, so masking all four at test time (needed by the masking study) was never seen in training | All four are dropped together with p = 0.05 (§6.8) |
| R10 | Major | §17 | Contradiction: Shapley and sensitivity in week 5 read test scores before the week-6 "first and only read" | D15: design freeze on Dec 1. The final runs come first (Stage 7), then the analyses (Stage 8). `aggregate` seals test rows until the freeze (§4.6) |
| R11 | Major | §9 | Gate 3 applied the "0.10 passes" threshold to an interval score, which has other units | Per-gate metrics and thresholds (§4.4 table) |
| R12 | Major | §4.4 | Rule too lenient: 6 of 10 wins happens by chance with probability 0.38, across about 14 decisions | Rule A (adding complexity): threshold + ≥ 7 of 10 wins (chance 0.17) + no unit worse by > 5 %. Rule B (peers): best mean |
| R13 | Major | §4.3 | Selection and decisions used passes only, so the seconds head you asked for was never selected for | MAE_eq = ½ (MAE_pass + MAE_sec / 100 s) for selection and gates; both units always reported |
| R14 | Major | §4.3 | With the gap-aware axis, T4 (fold 3's validation tool) has 20 window points, too few to select a step on | In-fit selection on the validation tool's whole-life weighted MAE_eq; gates still use the k-window |
| R15 | Major | §6.2 | Gate 1 ranked encoders by a per-pass ridge probe of RUL. A snapshot cannot integrate history, so this mostly measures healthy-vs-worn (Project 14) | Rank by the validation VB MAE (the wear state a pass feature must carry), with an RUL guard (§6.2) |
| R16 | Major | §6.4 | F0 would count usage once per expert in the product of experts, which is overconfident | Sensing experts get relative time only; usage is its own expert U; a generalized product of experts with learned weights (§6.4) |
| R17 | Major | §8 | The oracle R1 used full-sequence isotonic smoothing, which reads future measurements | Causal isotonic smoothing for any wear used as an input (§5.6, §8) |
| R18 | Major | §1 S2 | "Better than the baselines' intervals", but no baseline had intervals | B2's intervals are the empirical quantiles of the training tools' residual lives (§8) |
| R19 | Major | §7.10 | LoRA was to be adopted by the ridge probe, after Gates 2-3 were settled with frozen features | Adopted only by Rule A on the full model (§7.10) |
| R20 | Major | §14 | Data, memory-mapped stores and runs were to live in the OneDrive folder (locking, slow sync), and OmniFuser's venv is Python 3.14, which the encoder libraries lag behind | `RULFUSE_DATA_ROOT = D:\RULModel` and a 3.11 or 3.12 venv there (§14.1) |
| R21 | Minor | App. A vs §7 | Step budget: "≤ 400 steps" against phases 1-100 + 101-500 | 100 + up to 400 = 500, everywhere |
| R22 | Minor | §6.5 | Intervals were clipped to [0, 15], which is wrong for seconds | Clipped to [0, k_u] |
| R23 | Minor | §6.5 | ν = 2 + softplus allows an almost infinite variance in the mixture | ν = 3 + softplus |
| R24 | Minor | §10.3, §4.5 | "Seeds averaged first" contradicted "the reported model is the ensemble" | Gates on 3-seed ensembles, final on the 5-seed ensemble; the seed spread is a diagnostic |
| R25 | Minor | §5.4 | How physics features and the per-channel PCA combine was undefined | Per channel: encoder statistics ⊕ physics features → PCA 16 (§5.4) |
| R26 | Minor | §5.4 vs §17 | The derived-channel decision sat in Gate 1 in one place and in Gate 2 in another | Gate 1 (decision 5) |
| R27 | Minor | §17 | "History length" was a decision, though you specified the full history | A reported control, not a decision |
| R28 | Minor | §2.4 | The size of the force-missing problem was not stated | 134 passes (26 %), nearly all block 3; scored separately (§2.4, §10.1) |
| R29 | Minor | §7.3 | The order of the point-weight normalization was unclear | An explicit formula (§7.3) |
| R30 | Minor | §7.4 | The EMA spanned phases A and B | EMA restarted at step 101 |
| R31 | Minor | §7.5 | Fusion families were compared at one shared learning rate, which can favour one | Each family at its best of 3 learning rates (Gate 2a) |
| R32 | Minor | §1 S1 | The Wilcoxon power with n = 10 was not stated | Stated: one-sided, Holm over 2 units; needs about 8 or more tools improved (§1, §10.3) |
| R33 | Minor | §8 | B4 and B5 were vague ("+ ridge") | B4: ridge on recent physics features + usage. B5: the photo wear readout + degradation extrapolation |

## 3. Needs your input (answered 2026-11-04: yes, the runs were cut; D12 is final)

**R1, the gap-aware axis (plan §2.5, D12, open question 1).** The evidence:

| Tool | Missing | Wear across the gap | Median rise between consecutive runs | Ratio |
|---|---|---|---|---|
| T2 | Run 10 | +14.3 µm | 8.7 µm | 1.6 |
| T4 | Runs 6-9 | +41.7 µm | 7.4 µm | 5.6 |
| T1 | Run 1 (leading) | – (first recorded run 46 µm; other tools' first runs 27-35 µm, T4 47 µm) | – | ambiguous |

What changes under the gap-aware axis:

- T4's TTF goes from pass 22 to 38, and T2's from 40 to 44.
- The TTF mean ± sd goes from 37.6 ± 9.9 to 39.6 ± 8.5.
- T4's k-window shrinks to 20 observed points.
- The bar drops from 4.00 to 3.82 passes (B2).

If the runs were *not* cut, the recorded axis is right, and the tier-1 arm
shows how much it matters. The Nonastreda data paper may settle it.

## 4. Optimizations adopted

| # | Optimization | Effect |
|---|---|---|
| O1 | **Out-of-fold calibration** (4 inner fits) | Calibration from 8 tools instead of 1; the validation tool checks coverage independently. Cost about 5-6 GPU-h for each study that uses it (Gate 3 and the final runs) |
| O2 | **Causal RUL filter** (Kalman, slope −1 dynamics), a Gate-3 option | Removes the flicker that true RUL never has; cheap and causal |
| O3 | **Fusion reduced to 2 finalists** (your instruction); heads reduced to 2 (Student-t, quantile) | Gate 2a has 180 fits, against about 450 for five families with the same learning-rate check. The alternatives are documented with the condition under which to revisit them (§6.5) |
| O4 | **Training engine on physics features first** (Stage 4, laptop) | Runs alongside the GPU feature jobs; the sanity checks need no GPU |
| O5 | **Duplicated recordings encoded once** | 26 % fewer encoder calls |
| O6 | **Tiered sensitivity** | Tier 1 (5 refit arms, about 4 GPU-h) must run; tier 2 only as time allows |
| O7 | **Fit packing:** one array task = one (arm, fold) with its 3 seeds (and inner fits) | Fewer queue entries; respects the 4-GPU limit |
| O8 | **Gate 1 on cached features, on the CPU** | Encoder choice in minutes, no GPU queue |
| O9 | **Minimum viable result and drop order** (§17.1) | A defined fallback if the schedule slips |

## 5. Decisions scrutinized and kept

| Decision | Why it stays |
|---|---|
| D2 confirmed TTF | Smoothed wear at the TTF is 109-123 µm on every tool; robust to isolated flips |
| D3 k = 15 passes / 1,500 s | Every tool still has a non-empty window. T4's is thin because of the gap, not because of k |
| D7 10 s segments | About 10 points per pass; tokens fit a small model; 5 s and 20 s are tier-2 arms |
| D8 latest attachment | The only causally safe choice; the alternative is a tier-1 arm |
| D9 photos after their pass | Removes the leakage that dominated OmniFuser's accuracy |
| D10 MSE with the mean detached in the NLL | Keeps your MSE objective for the mean and still learns the spread |
| Tool-relative tokens; trajectory augmentation; phase A | Each answers a measured failure (L5, L7) and each is tested by Rule A |
| 5-seed final ensemble | Also gives the epistemic part of the uncertainty |
| Leave-one-tool-out CV (D1) | Your protocol; its single-validation-tool weakness is handled by whole-life selection, ensembles and out-of-fold calibration |

## 6. Checks that are still open and owned by a stage

| Check | Stage |
|---|---|
| The causal trim agrees with OmniFuser's on ≥ 95 % of passes | 1 (P2) |
| Chronos `pipeline.embed` and DINOv3 access work as assumed | 0 |
| The time per fit (all GPU estimates rest on about 1.5 min) | 3 (J1) |
| The test tools' shift from the training distribution falls after the tool-relative step | 2 (P7) |
