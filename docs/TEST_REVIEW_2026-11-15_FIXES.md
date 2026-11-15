# Fixes for the 2026-11-15 test review

Response to [TEST_REVIEW_2026-11-15.md](TEST_REVIEW_2026-11-15.md) (written by Codex). The
production code was fixed until Codex's strict tests pass. No test was weakened, marked `xfail`
or skipped to get there. Nothing was run on Ruche and no test result was read (D15).

| ID | Finding | Fix | Where |
|---|---|---|---|
| T01 | F0's first future photo reached the gate | An absent image expert (no photo yet, or photos dropped) gets a zeroed state and fixed "no image" outputs, so nothing of a photo enters the gate. **F0 code version 3**: version-2 fits are neither copied nor resumed. | `models/fusion.py`, `train.py` |
| T02 | Token building trusted foreign or unfinished caches | `build_fold_data` checks that each cache it reads belongs to the store and is complete for the rows used | `tokens.py` |
| T03 | NaN encoder output could be published as complete | Extraction refuses nonfinite embeddings or image tokens before writing | `features/cache.py` |
| T04 | Scoring accepted a foreign tool, duplicate times, NaN predictions | Every member is checked (one expected tool, strictly increasing finite times, finite sigma > 0, nu > 2, finite window targets) before scoring | new `members.py`, used by `study.ensemble_validation` |
| T05 | An incomplete predecessor supplied settings | Inheriting a decision with `complete: false` raises | `study.Study.inherited` |
| T06 | A misspelled fold made an empty study | Unknown fold names raise | `study.study_folds` |
| T07 | Nonphysical inputs accepted at public boundaries | `FitConfig` checks ranges, positivity, `d_model % heads`, phase B steps and unique players. The schema checks finite horizon, sample rates and durations and valid image preparation. `rul_targets`, `mixture_moments` and `kalman_rul` refuse invalid horizon, scale and noise. | `config.py`, `core/schema.py`, `core/targets.py`, `uncertainty.py` |
| T08 | Multi-seed members were aligned by count, not identity | Members must agree on times, tool, truth and window (main and Gate 3 ensembles). B6 refuses a seed predicting a point twice, a point missing a seed, and disagreeing tool or grid flags. | `members.py`, `gate3.stack_seeds`, `omnifuser_rul/aggregate.py` |
| T09 | An empty or unknown sanity-check list reported `all_pass` | `run_checks` requires a non-empty subset of 1-7 | `sanity.py` |

## Checks on real inputs before the strict checks went in

- The real store has no duplicate prediction time and no missing target on any tool, so T04's
  strict time and target checks cannot stop a real aggregation. Prediction points per tool:
  308, 393, 355, 216, 188, 458, 465, 459, 379 and 526.
- Every real study configuration (`pilot`, `gate2a`, `gate2b`, `gate2a-fix`, `gate2b-fix`,
  `gate3`) builds valid `FitConfig`s under the new T07 checks.

## Validation

Full suite (2026-11-15, laptop CPU), Codex's tests and real-data checks included, with
`-m "not weights and not gpu"`:

- **492 passed, 0 failed, 2 skipped, 46 deselected** (20 min 52 s). Before these fixes the
  same tests had 63 failures.
- The 2 skipped tests need the cached Mantis and DINOv2 weights. The 46 deselected are the
  `weights` and `gpu` tests, which cannot run on this laptop.
- Ruff and pyright are clean.

## What this changes for Ruche

- **F0 is now code version 3.** `gate2a-fix` trains F0 with both leak paths closed (F01 and
  T01). No F0 fit exists yet at version 2 or 3, so nothing is lost.
- The order of jobs is unchanged (runbook §15.7). The new checks refuse corrupt inputs
  instead of producing plausible numbers. If an aggregation now stops with a `ValueError`
  about members, times or caches, send me the message: it means something on disk is not
  what the study asked for.
