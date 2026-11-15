# Fixes and validation for the 2026-11-15 audit

Response to [AUDIT_2026-11-15.md](AUDIT_2026-11-15.md) (written by Codex). Every finding
was checked against the code before it was fixed. Each section gives the verdict, the fix, how
it was validated, and what remains open. Nothing was run on Ruche and no test result was read
(D15).

## Summary

| Finding | Verdict | Fixed | Validated by |
|---|---|---|---|
| F01 F0 reads future photos in padded batches | Confirmed (same numbers as the audit) | Yes | 9 tests; the audit's reproduction before and after |
| F02 Gate 3 inner fits select on the validation tool | Confirmed | Yes | Inner fit equals the outer fit at its kept step |
| F03 Fit reuse ignores encoders and code | Confirmed | Yes, with limits | 5 tests: reuse, Gate 2a-style old fits, skip of a finished fit, checkpoint resume, mixed versions at aggregation |
| F04 Freeze accepts a failed coverage gate | Confirmed | Yes | Freeze refuses; `--amended` records an override |
| F05 CUDA resume loses the GPU random state | Confirmed by reading the code | Yes, GPU check pending | CPU test; `scripts/check_cuda_resume.py` for Ruche |
| F06 Image-free datasets fail | Confirmed | Yes | Image-free toy through features, fit and prediction |
| F07 B6 ensembles unrequested seeds | Confirmed | Yes | Requested seeds only; unequal members refused |
| F08 D28 summaries missing | Confirmed | Yes for Gates 2 and 3 | Pooled and without-thinnest values on unequal point counts |
| F09 Status undercounts inner fits | Confirmed | Yes | `slurm/status.sh` on a fake run tree |

**Validation run (2026-11-15, laptop, CPU):**

- full suite: **275 passed, 2 skipped**. The 2 skipped tests need the cached Mantis and
  DINOv2 weights.
- the 4 tests added afterwards also pass: the old-style fits, mixed versions, the inner-step
  check and the B6 `arm.json` fallback. With them, `tests/test_audit_fixes.py` and
  `tests/test_gate3.py` give 20 passed.
- ruff and pyright are clean for the whole package, the new tests and the GPU script. The
  audit recorded 1 ruff error and 4 pyright errors.
- `slurm/study.sh gate3`, run with stub `sbatch` and `python`, submits the outer array
  (tasks 0-9), then the inner array (tasks 10-49) waiting for it, then the aggregation waiting
  for both.

**Not yet validated:**
- F05 on a GPU (`scripts/check_cuda_resume.py`);
- the real submission on Ruche.

---

## F01: F0 read photos not yet taken (9/10)

**Verdict: confirmed.** The audit's script gives the same numbers on the old code: the short
unit predicts 60.54 s alone and 64.53 s in a padded batch, and changing its 20 s photo moves
its 5 s prediction by 0.41 s.

**Cause.** `collate` pads image times with 0. F0 counted every image time before the query
time, padding included, and used that count to index the image-GRU states. In a padded unit
the index ran past its last real photo visible at that time, so the model read the state of
its *later* photos. The validity mask stopped padding from updating the GRU but not from
being counted.

**Fix** (`rulfuse/models/fusion.py`). The lookup takes the last slot that is both valid
and earlier than the query time. Padding and dropped photos are never visible.

**Validation** (`tests/test_padded_images.py`, 9 tests, F0 and F1):

- a padded batch gives the same predictions as the unit alone;
- editing a photo taken at 20 s leaves the 5 s and 15 s predictions unchanged;
- a unit with no photo in a batch with photos;
- a dropped photo acts as if it was never taken;
- truncating a sequence leaves earlier predictions unchanged.

**Against the old code**, the three F0 tests that target the bug fail and every F1 test
passes. F1 never had the defect, which the audit had not established.

**The audit's reproduction after the fix:** 60.54 s alone and in the batch; the change is 0.

**How much training it touched** (read from the real store, no model run). Tool photos per
tool range from 40 to 60. In every fold, 5 to 7 of the 8 training tools have fewer photos than
the batch maximum, so they were padded at every step. About 81 % of training points (56-86 %
per fold) belonged to a padded tool. A point of a tool with d fewer photos could read up to
d photos ahead. Single-unit inference (validation, test, diagnostics) was never padded, so
the stored validation scores are honest for those weights. What is in doubt is the
*training*: every F0 fit learned partly from photos it cannot see at prediction time.

## F02: Gate 3 inner fits used a new validation selection (8/10)

**Verdict: confirmed.** Plan §7.2 step 10 says each inner fit trains on 6 tools "for the
outer fit's selected number of steps (no inner selection)". The first Gate 3 version
early-stopped every inner fit on the validation tool.

**Fix.** `fit(heldout=..., fixed_steps=...)` trains for exactly the step its outer fit kept,
on the outer learning-rate schedule. It does no evaluation, keeps no snapshot and never builds
the validation tool, so that tool stays an independent coverage check.

- `run_task` reads the step from the outer fit's `selection.json`, and refuses to start before
  that outer fit is finished.
- `slurm/study.sh` submits the inner array with `--dependency=afterok` on the outer array.
- The aggregation checks that every inner fit trained for its outer fit's step.
- It is also cheaper: no evaluations every 10 steps and no patience tail.

**Validation** (`tests/test_gate3.py`):

- trained on the outer fit's own tools with the same seed and step count, an inner fit has
  exactly the outer fit's weights;
- an inner task run before its outer fit raises;
- end to end on a 6-unit toy, with held-out groups of two tools as on Ruche, the seeds run
  side by side, the inner fits are copied by B1, then aggregation and freeze;
- dry run on the real store with stub decisions: the manifest has 50 tasks (10 outer,
  40 inner), and every fold has 4 splits of 2 held-out and 6 fitted tools. Neither the
  validation nor the test tool is ever inside a split.

## F03: fit reuse did not check what was computed (8/10)

**Verdict: confirmed.** Two specs with different encoders gave the same key, and the old fit
was copied.

**Fix** (`rulfuse/train.py`, `rulfuse/study.py`). Every fit records an identity: the
configuration key, the code version, the test, validation and training units, the seed, and,
for an inner fit, the held-out units and the step count. The identity is checked:

- **on reuse (B1):** the source also needs the same store and encoders;
- **when a finished fit is skipped:** `StaleFitError` instead of a silent skip;
- **on checkpoint resume:** a checkpoint from another computation is discarded and the fit
  starts afresh;
- **at aggregation:** code versions are written to `summary.csv`, `decision.json` and the
  report, with a warning when they are mixed.

**Code versions are per fusion family**: F0 is now version 2 (version 3 after the test review's T01 fix, see TEST_REVIEW_2026-11-15_FIXES.md) and F1 stays at version 1, since
F1 is unchanged (F01 tests). Fits written before the field count as version 1. Old F1 fits
are therefore still copied, and old F0 fits are not. A study can list
`accept_code_versions` to copy older fits knowingly.

**Validation** (`tests/test_audit_fixes.py`):

- reuse is refused for another encoder, configuration or code version, and allowed when the
  version is accepted;
- a finished fit of another computation raises, both in `fit` and in `run_task`;
- a checkpoint from another seed is discarded, and the result equals a fresh fit;
- old-style fits like those on Ruche (no identity, no code version): an F1 fit is copied, an
  F0 fit is refused, and accepted only with `accept_code_versions: [1]`;
- an aggregation over fits of two code versions flags `mixed_code_versions` and the report
  warns;
- a Gate 3 inner fit whose step differs from its outer fit's makes the aggregation raise.

**Limits:**
- encoders are compared by name and the store by signature; a feature cache re-extracted
  under the same name would not be detected;
- the code version is a constant that must be bumped by hand when a change alters what a fit
  computes.

## F04: freeze accepted a failed coverage constraint (8/10)

**Fix.** `freeze` refuses unless at least one candidate's mean validation coverage lies in
[0.85, 0.95]. `rulfuse freeze --amended "TEXT"` records an explicit decision that accepts the
failure. `configs/final.json` stores `constraint_met`, the amendment and the code versions.

**Validation:** 10 synthetic folds with coverage 0.3 give `constraint_met: false`, and the
freeze raises. The end-to-end test covers both branches.

## F05: CUDA checkpoint resume (6/10)

**Fix.**
- The checkpoint now saves the CUDA generator states and restores them on resume.
- It is loaded with `map_location="cpu"`, so the CPU random state stays a CPU tensor (it
  could land on the GPU before);
- `load_state_dict` moves the weights and the optimizer state to the device.

**Validation:**
- On the CPU, the existing kill-and-resume test still gives identical weights.
- A new test checks the checkpoint's keys: the identity, a CPU random state, and no CUDA state
  on a CPU fit.
- **Not yet validated on a GPU.** Run once on Ruche (a few minutes, toy data):

  ```bash
  sbatch --job-name=cuda-resume --time=00:30:00 slurm/job.sbatch python scripts/check_cuda_resume.py
  ```

  It prints PASS when a resumed GPU fit equals an uninterrupted one. It prints INCONCLUSIVE if
  two uninterrupted GPU runs already differ.

## F06: image-free datasets (5/10)

**Fix.**
- The image table is written with its columns even when empty (`IMAGE_SCHEMA`).
- A store written before this, with an empty table, gets empty columns when opened.

**Validation:**
- The toy dataset stripped of its photos builds a store with the image columns.
- A usage-only fold builds.
- Signal features are extracted, then a signal-only fit trains and predicts.
- An old-style empty image table opens.
- The audit's reproduction now gives `fold_build: passed`.

## F07: B6 included unrequested seeds (6/10)

**Fix.**
- `load_predictions` and `score_arm` take the requested seeds. Without them, the loader uses
  the seeds in the arm's `arm.json`.
- The B6 study and report pass `spec.seeds`.
- `ensemble` refuses seed members that predict different points.

**Validation:** with seeds 0 and 1 on disk and seed 0 requested, the ensemble is 100 (not
200). Unequal members raise, and `arm.json` seeds are used by default.

**Note:** the audit script calls the loader without seeds and without `arm.json`, so it still
loads both. Every real caller now passes the seeds.

## F08: D28 summaries (4/10)

**Fix.**
- Gate 2 studies (`decision.json`, `REPORT.md`) now report the pooled-point MAE (each tool
  weighed by its window points) and the per-tool mean without the tool with the fewest
  window points (T4, 20 points), beside the per-tool mean the rules use.
- Gate 3 reports the same for coverage and Winkler.

**Validation:** synthetic rows with one 20-point tool give the expected pooled and
without-thinnest values, and the report names the tool.

**Open:** the final report (Stage 7) is not written yet. Gate 2a's existing `REPORT.md` gets
the columns only if it is re-aggregated with the new code; the decision itself is unchanged.

## F09: status counter (3/10)

**Fix.** `slurm/status.sh` counts `fold-*/inner-*/seed-*/DONE` as well.

**Validation:** on a fake tree with 2 outer and 2 inner finished fits, it prints `4/4 fits`.

---

## What this means for Gate 2a and Gate 2b

- **Gate 2a, F1 arms:** unaffected (F1 tests pass on the old code). They stay valid and will
  be copied by B1.
- **Gate 2a, F0 arms:** trained through the leak. Their validation scores are honest for those
  weights, but the decision "F0 at lr 3e-3 over F1" compares a model trained with leaked
  photos against one trained without. It cannot be trusted until F0 is retrained.
- **Gate 2b, all 5 arms (all F0):** trained through the leak. Its decisions (trajectory
  augmentation, phase A, token mode) compare leaky models.
- **The running Gate 2b array** keeps using the code it checked out, so it stays consistent,
  as long as the repository on Ruche is not pulled before every task has finished. A task that
  starts after a pull would train the fixed F0, and Gate 2b would mix versions. The new
  aggregation would flag that, but the rule would still compare them.
- **After a pull,** resubmitting Gate 2a or Gate 2b raises `StaleFitError` on their F0 fits:
  by design they are not taken as current. Reruns go under new study names.

The options and the recommendation are in PROJECT_LOG.md (2026-11-15) and PROJECT_PLAN.md D30.

## Not addressed here

- The Stage 7 consumer of `configs/final.json` (final scoring and report): planned work, not
  a defect.
- CI and a dependency lockfile.
- Hashing feature-cache contents into the fit identity.
