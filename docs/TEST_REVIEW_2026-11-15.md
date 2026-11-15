# Test review — 15 November 2026

**Status: the expanded suite exposes unresolved correctness defects. F0 is not ready for
causality sign-off.** Current production acceptance uses one model seed, seed 0.

This review follows the earlier code audit and its fixes. The original audit finding
numbers describe an earlier tree; the findings below have their own `T` identifiers.
Production model code and real experiment outputs were not changed by this test review.
New strict failures remain failures: no `xfail` or skip conceals a reproduced defect.

## Inputs reviewed and scope

Read the complete main project plan, its P1–P22 verification matrix and decision amendments,
`EDA.md`, `docs/EDA_RUL.md`, `docs/OMNIFUSER_RUL_PLAN.md`, the audit/fix reports, and current
project-log amendments before designing the tests. Rechecked live study/CLI/scheduler
changes after the one-seed amendment, including `gate2a-fix`, conditional `gate2b-fix`, and
Gate 3. The latter currently requests one seed and four inner splits per outer fold.

Reviewed existing test coverage against those requirements, retained useful regressions,
and added independent arithmetic, randomized invariants, poisoned inputs, corruption
checks, raw-input and complete synthetic study workflows. Existing test paths are
preserved; categories also apply to those files through `tests/catalog.py`.

The primary acceptance path is single-seed. Retained multi-seed API tests are explicitly
marked `compatibility`; their failures below are not blockers caused by requiring three
production seeds. Reproducibility trials and randomized transformation trials are also
different from running a three-seed production ensemble.

## Prioritized findings

Scores are engineering severity, **not CVSS**: 9–10 critical, 7–8.9 high, 4–6.9 medium,
1–3.9 low. They reflect impact on scientific correctness, silent failure, and relevance
to the current one-seed pipeline. Parametrized failures are grouped by root cause.

| ID | Severity | Confirmed behavior and consequence | Strict regression |
|---|---|---|---|
| T01 | **8.5 / High** | F0 reads the first future photo through its gate. The image expert is marked absent, but its predicted scale enters the gate before the expert's output logit is masked. The other experts' weights can therefore depend on that future photo. All three token modes fail; a nonzero gradient to the unavailable photo independently confirms the path. F1 passes the same checks. Experiment-level impact is not measured. | `test_model_causality.py::test_first_future_photo_cannot_change_any_earlier_output`, `test_gradient_cannot_reach_a_photo_before_it_exists` |
| T02 | **8.0 / High** | Token construction accepts a foreign store signature and caches whose completion arrays are all false. Training can consume incompatible or unfinished features instead of failing before a fit. | `test_artifact_integrity.py::test_token_building_refuses_foreign_or_unfinished_features` |
| T03 | **7.5 / High** | A feature encoder returning NaN arrays does not cause extraction to reject publication. Completion can be recorded for invalid feature contents. | `test_artifact_integrity.py::test_nonfinite_encoder_outputs_cannot_be_published_as_a_complete_cache` |
| T04 | **7.5 / High** | The current **single-seed** validation aggregation accepts rows for a foreign tool, duplicate prediction times, and NaN predictions. A `DONE` file does not establish that rows belong to the requested fold or contain valid scores. | `test_single_seed_protocol.py::test_single_seed_scoring_refuses_wrong_or_corrupt_predictions` |
| T05 | **7.0 / High** | A predecessor with `complete: false` can supply adopted settings to the next study. File existence is checked; decision completeness is not. This can start a costly study from unfinished decisions. | `test_artifact_integrity.py::test_an_incomplete_gate_cannot_supply_the_next_studys_configuration` |
| T06 | **6.0 / Medium** | An unknown requested fold name silently produces an empty study manifest rather than rejecting the misspelling. | `test_artifact_integrity.py::test_unknown_fold_names_cannot_silently_create_an_empty_study` |
| T07 | **6.0 / Medium** | Missing input guards accept nonfinite physical settings, invalid image preparations, unusable fit schedules/dimensions/probabilities, invalid horizons, Student-t scales, and Kalman noises. Failures can surface late or produce nonfinite artifacts. These are engineering acceptance requirements at public construction/function boundaries, not claims that valid shipped configurations are currently malformed. | `tests/unit/test_input_validation.py` |
| T08 | **4.5 / Medium; compatibility** | Retained multi-seed aggregation accepts equal-length tables with mismatched unit/truth/window data. B6 accepts duplicate or inconsistent seed identities when per-point counts happen to agree. This is secondary under seed-0 production, but remains a defect in the supported APIs. | Compatibility cases in `test_artifact_integrity.py` and `test_b6_ensemble_contracts.py` |

T01 is a residual path after the earlier padded-image repair, rather than a request to
restore an obsolete model or seed protocol. Correct its inputs to the gate and rerun the
first-photo and existing padded-image regressions before treating F0's retraining as
causally valid. T02–T05 also apply with exactly one model seed.

## Useful passing evidence

- New production-style studies and fits use seed 0. Current study configurations request
  only that seed; Gate 3's manifest contains 10 outer and 40 inner fits.
- Single-seed aggregation preserves the member mean and Student-t variance/intervals,
  without needing a between-seed variance term.
- Scheduler stubs check one-seed packing, outer-before-inner dependencies, aggregation
  dependencies, failed submissions, and the short final packed task. No job is submitted.
- The raw end-to-end test writes a real MATLAB table and CSV fixture, rebuilds the raw
  force store, adapts and segments it, extracts lightweight features, trains, and checks
  finite predictions. A different generic fixture includes non-force channels, a different
  sample rate/horizon, missing force, gaps, two photo kinds, and post-failure history.
- F0/F1 synthetic lifecycles cover CLI manifest/run/aggregate, fixed-step inner fits,
  out-of-fold calibration, frozen configuration hashes, signal-only/no-HI transfer, exact
  CPU interruption/resume, and repeatable report bytes. Synthetic test-prediction poison
  leaves validation reports and decisions unchanged.
- Held-out descriptor/HI poison leaves training transforms unchanged. Cross-fitting
  excludes the tool's own targets, including ridge selection; annotation/stage metadata
  does not enter the model tokens.
- Read-only real-store checks reproduce the approved EDA per-tool point/missing/window
  counts, rounded failure times, and photo availability. T4 retains its 20-point window.

## Execution evidence

Measured counts, source hashes, environment and coverage are recorded in
`TEST_REVIEW_2026-11-15.evidence.json`. Full per-test requirement/outcome and coverage
artifacts are generated under the ignored `runs/test-review/` directory. Commands and the
P1–P22 coverage/gap matrix are in [TESTING.md](TESTING.md).

The test review uses the local Python 3.11 environment, CPU with two Torch threads.
Real-data checks read existing data; synthetic workflows write only temporary fixtures.
Real held-out test predictions were not scored or unsealed. GPU acceptance and actual
Ruche execution need the deployed hardware; a skipped GPU check is not evidence of a pass.

## Limits and next acceptance gates

Coverage is evidence about executed code, not a guarantee about every harmful behavior.
Full raw-file rehashing, real pretrained feature extraction/offline loading, CUDA resume,
and cross-machine training parity are separate gates. The main importance/sensitivity/final
report studies are not fully implemented and cannot be certified by testing B6's retained
helper APIs. Existing scientific recovery tests do not replace real out-of-fold coverage
or a full real-data sanity checklist.

The current code/configuration also retains historical multi-seed defaults in some generic
entry points; the active fix/Gate-3 configurations explicitly override them with `[0]`.
Use those configurations for current runs. The plan's older three-seed passages are not
used as assertions requiring three seeds.
