# Results

On unseen tools, the frozen multimodal model (RUL-Fuse, design F0) reaches a window MAE of
**453.8 s**, with **457.4 s** after the causal filter. A constant guess (B0) scores **412.0 s**
and the usage-only baseline (B2) scores **467.8 s**. The model therefore does not beat a
constant guess on tools it has never seen, and it is only about 14 s better than usage alone;
the causal filter does not help on test. On the validation tools the same model scored about
362 s. That gap is not a scoring bug (test and validation predictions come from the same
kept checkpoint). Validation was optimistic by 66 to 116 s,
because checkpoints and design choices were selected on the same tools that were scored. The
test score is the unbiased one, and the design freeze is what makes it trustworthy. The main
cause of the weak result is the data: with 10 tools, frozen image and force features estimate
flank wear only to about 21 µm, and at the observed wear rates that is about 920 s of RUL, more
than half of the 1,500 s horizon. This page reports what was measured, what is hypothesis, and
what I would change. For the architecture, see the [README](../README.md).

## 1. Setup

**Task.** Predict remaining useful life (RUL, seconds of cutting) of milling tools from
3-axis force signals (frozen Mantis-8M embeddings plus 7 level descriptors), tool photos
(frozen DINOv2-B) and usage (operating time). The data is the Nonastreda dataset: 10 tools run
to failure. Targets are capped at a horizon H = 1,500 s. The metric is the **window MAE**: the
mean absolute error over points whose true RUL is at most H, averaged over tools.

**Protocol.** 10-fold leave-one-tool-out. In fold Tk, tool Tk is the test tool and tool T(k+1)
is the validation tool: it selects the checkpoint and scores the decision gates. The other
8 tools train. One seed was run (time budget).

**Frozen design.** F0 late fusion (generalized product of Student-t experts: usage, signal,
image), learning rate 3e-3, trajectory augmentation off, phase A on (learn the wear index
first), absolute tokens, players signal plus tool photos, causal GRU memory, calibration
method b (out-of-fold Student-t scale) at 90 %, causal Kalman filter on. Freeze hash
`16aa562e5180`. The freeze was recorded with an amendment, because no calibration method met
the coverage band [0.85, 0.95].

**Baselines (window MAE).**

| Baseline | Window MAE |
|---|---|
| B0 constant H/2 | 412 s |
| B2 usage only (expected remaining life given survival, from training tools) | 465 s validation, 467.8 s test |
| R1 oracle with measured wear | 307 s (validation) |

## 2. Design selection on validation

All numbers in this section are on validation tools, one seed, per-tool mean window MAE.

| Gate | Comparison | Result | Decision |
|---|---|---|---|
| 2a-fix | F0 vs F1 | F0 368.0 s, F1 389.1 s (pooled points: F0 335.8, F1 325.5; without T4: F0 332.3, F1 310.8) | Rule B adopted F0 |
| 2b-fix | default | 368.0 s | reference |
| 2b-fix | no trajectory augmentation | 358.5 s | dropped augmentation (-9.5 s, CI -37 to +16) |
| 2b-fix | no phase A | 402.1 s | phase A kept (Rule A: +34.2 s, 8/10 wins) |
| 2b-fix | tokens relative / absolute | 367.8 s / 367.3 s | absolute (tie within 0.7 s) |
| 3 | calibration | method b, coverage 0.962 (above band); intervals span the whole horizon on 9 of 10 tools; out-of-fold scale ranged 0.77 to 38 | method b chosen |
| 3 | causal filter | 361.9 s to 333.1 s, 9/10 wins, coverage 0.945 | filter adopted |

Two things to note. First, the F0 vs F1 call depended on the metric: F1 was better on pooled
points, F0 on the per-tool mean, and the choice was made by rule rather than by a clear
margin. Second, on 2026-11-15 an audit found and fixed a future-photo leak in F0's training
batches (about 81 % of training points were affected; validation inputs were not). F0 was
retrained, and about 30 s of the earlier validation accuracy had never been real. The numbers
above are the post-fix ones.

## 3. Final test result

Each tool was scored once as a test tool, with the frozen design.

| Quantity | Test |
|---|---|
| Model window MAE | 453.8 s |
| With causal filter | 457.4 s |
| B0 constant | 412.0 s |
| B2 usage only | 467.8 s |
| Filter coverage / Winkler score | 0.833 / 2594 s |

The test tools are the same 10 tools as the validation tools (B0 is 412 in both, which
confirms the identical tool set). The same model scores about 362 s when a tool is its
validation tool and 454 s when it is its test tool. The filter's validation gain (361.9 to
333.1 s) did not carry over: on test it is slightly worse than the unfiltered model (457.4 vs 453.8 s). Its coverage
of 0.833 is below the 90 % target.

## 4. Why validation and test disagree

The checkpoint of every fit was chosen as the best smoothed value of many noisy evaluations on
that fit's validation tool, and the gates chose among arms on the same tools. Both steps pick
noise that happens to look good. The validation-to-test drop, from the retraining Shapley runs
below, measures how much:

| Coalition | Validation | Test | Optimism |
|---|---|---|---|
| B2 usage only | about 465 s | 467.8 s | not applicable |
| F: force + usage | about 395 s | about 461 s | 66 s |
| T: photos + usage | about 361 s | about 477 s | 116 s |
| F+T frozen model | about 362 s | about 454 s | 92 s |

The optimism of 66 to 116 s is larger than every effect the gates measured (the largest was
34.2 s for phase A). So the gate decisions were made inside noise: they were reasonable
procedures, but the validation scores could not tell the arms apart. The frozen design was
sealed before any test tool was scored, so the 453.8 s figure carries no such selection
effect.

## 5. Modality importance (Shapley)

Retraining Shapley: usage is in every coalition, the empty coalition is B2, and the value is
v = -window MAE. F is force, T is photos.

| | Validation | Test |
|---|---|---|
| phi F | 34.3 s [-8.8, 77.5] | 14.7 s [-20.1, 51.5] |
| phi T | 68.6 s [4.5, 128.5] | -0.6 s [-74.4, 81.7] |
| Interaction | -71.0 s | 16.4 s [-76.6, 103.1] |

On validation, photos looked like the most useful modality, and force and photos looked
redundant (negative interaction). On test, neither modality is distinguishable from zero. The
total gain over B2 is 14.1 s [-75.2, 113.6], and the model beats B2 on 4 of 10 tools. With
10 tools the intervals are wide, so the honest reading is "no demonstrated gain", not
"measured zero gain".

## 6. Root cause: the label is hard to read from frozen features

The diagnostics below use only the store and the frozen feature caches. No model is involved.

**Features alone.** The first table asks whether the features carry tool identity, and whether
a leave-one-tool-out ridge regression to capped RUL beats a constant on the same points.

| Features | Tool identity accuracy (centroids from first third of life classify middle third; chance 0.10) | Ridge to capped RUL, window MAE | Constant on same points |
|---|---|---|---|
| Tool photos (DINOv2-B) | 0.64 | 497 s | 426 s |
| Force embeddings (Mantis-8M) | 0.26 | 504 s | 412 s |
| Force level descriptors | 0.23 | 491 s | 412 s |

Neither modality's features map linearly to RUL across tools better than a constant. A
leave-one-tool-out ridge from features to measured flank wear does better, but not well enough:
R^2 is 0.55 (photos), 0.54 (force embeddings) and 0.50 (descriptors), with MAE of about 21 to
23 µm.

**Physics of the label.** Failure is flank wear reaching about 110 to 120 µm (wear at failure
per tool: 108.5 to 122.6 µm). So RUL here means "time until wear crosses roughly 110 µm".

| Quantity | Value |
|---|---|
| Wear rate in the last 1,500 s before failure | 1.07 (T4) to 4.78 (T5) µm per 100 s |
| Mean wear rate | 2.28 µm per 100 s (coefficient of variation 43 %) |
| Wear at the start of the 1,500 s window | 37.7 µm (T5) to 99.7 µm (T3) |
| Lifetimes | 1,827 s (T5) to 5,367 s (T10) |

**The 920 s argument.** A wear-estimation error of 21 µm, divided by the mean wear rate of
2.28 µm per 100 s, is about 920 s of RUL (21 / 2.28 x 100). This is an order-of-magnitude
argument (the 21 µm is averaged over whole lives, the rate over the last 1,500 s), but it is
more than half of the 1,500 s horizon. Wear rates also differ by 4.5 times between tools (1.07 vs 4.78 µm per 100 s),
so even perfectly measured wear leaves rate uncertainty: the oracle R1, which knows the
measured wear, only reaches 307 s against 412 s for a constant. And with 8 training tools there
are effectively 8 examples of a lifetime to learn from. This is the dominant limit, and I read
it as data-limited rather than a modelling failure.

**Contributing factors (hypotheses, not proven).** The evidence is consistent with the
following, but I did not isolate either:

- Photo features identify the tool (accuracy 0.64 against 0.10 chance). That would let a
  model fit each training tool's appearance instead of its wear. The photo-only model had the
  largest validation-to-test drop (116 s), which fits this picture.
- Several design elements give the model a shortcut toward elapsed time: usage features, the
  usage expert and the GRU memory, with trajectory augmentation dropped on a noisy gate. That
  would make the model behave like the usage baseline, which matches a test score close to B2.

## 7. Fit-level diagnostics

Fit-level checks are pending: per-tool bias against the lifetime gap, gate weights,
train/validation/test MAE, and inner-fit out-of-fold MAE. They are run by
`scripts/diagnose_final.py --fits` and will be added here. Until then, the hypotheses in
section 6 remain untested at the fit level.

<!-- FIT DIAGNOSTICS: to fill -->

## 8. What I would change

This is post-hoc judgment, not a result.

- Select a fixed step count from out-of-fold fits, instead of a per-tool best checkpoint on
  the validation tool. This removes the main source of optimism.
- Run several seeds, so that arm differences can be told from run-to-run noise.
- Keep trajectory augmentation and relative tokens, which were dropped or tied on a noisy gate.
- Fine-tune, or train a wear-specific image model, instead of relying on frozen DINOv2-B
  features.
- Collect more tools. This is the dominant limit.

## 9. Limitations

- Only 10 tools. All intervals on test are wide, and per-tool conclusions rest on single runs.
- One seed. Seed variance is not measured.
- The label is close to visual: failure is defined by flank wear, which the tool photos show
  directly, so part of any photo gain could be definitional rather than predictive.
- Prediction intervals are saturated: on validation they span the whole horizon on 9 of 10
  tools, and on test the filter's coverage is 0.833 with a Winkler score of 2594 s.
- The freeze amendment (no calibration method met the coverage band) means calibration was
  adopted as the least-bad option, not as a verified fit.
