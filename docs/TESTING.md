# Testing RULfuse

The tests check the implementation against `PROJECT_PLAN.md` section 15, the accepted
decision amendments, `EDA.md`, `docs/EDA_RUL.md`, and the OmniFuser-RUL plan. The current
studies use **one seed, seed 0**. Historical three-seed experiments are not acceptance
requirements for the current protocol.

## Categories

Every collected test must have exactly one level and at least one requirement ID. Unknown
markers and missing categories fail collection. Existing filenames stay usable; their
classification is in `tests/catalog.py`. New tests live in the directories below.

| Selection | Purpose |
|---|---|
| `unit` / `tests/unit` | Arithmetic, preprocessing boundaries, configuration and loss gradients |
| `integration` / `tests/integration` | Causal models, held-out isolation, artifact integrity, one-seed protocol, shell scheduling |
| `e2e` / `tests/e2e` | Raw MAT/CSV to predictions; CLI study, inner calibration, freeze, reports, interrupted fits |
| `regression` | Reproductions of discovered defects |
| `adversarial` | Poisoning, malformed inputs, corrupt caches and prediction artifacts |
| `scientific` | Statistical recovery, learning on known synthetic truth, distribution checks |
| `system` | CLI/subprocess/scheduler interfaces; shell tests use stubs and submit no jobs |
| `compatibility` | Retained multi-seed behavior, separate from current one-seed acceptance |
| `data` / `tests/data` | Real-data acceptance; new EDA checks read existing tables only |
| `weights` | Actual pretrained checkpoints; needs the local model cache |
| `gpu` / `tests/gpu` | Actual CUDA checkpoint/resume acceptance |
| `slow` | Fits or other extended checks; all end-to-end tests carry this marker |

The physical directory and marker selections differ: `-m unit` includes the old unit tests
as well as `tests/unit`. Parametrized cases count separately. Ten random augmentation
seeds test a transformation's invariants; they do not request ten production model fits.

## Run

Use the project's Python environment with `.[test,dev]` installed, plus PyTorch. The full
suite also needs the encoder and OmniFuser extras. This workspace's interpreter is
`D:/RULModel/.venv/Scripts/python.exe`. The following commands assume that environment is
active. On a machine without optional dependencies, module-level skips are reported.

```bash
# Fast feedback across both old and new unit tests.
python -m pytest -m 'unit and not data and not weights and not compatibility'

# Current protocol: synthetic checks, no real data/checkpoints/GPU needed.
python -m pytest -m 'not data and not weights and not gpu and not compatibility' --requirements-report runs/test-current.json

# Broader supported API checks, including retained multi-seed behavior.
python -m pytest -m 'not data and not weights and not gpu' --cov=rulfuse --cov-branch --cov-report=term-missing --cov-report=json:runs/coverage.json --requirements-report runs/test-all.json

# Environment-specific acceptance: a skip must not count as success.
python -m pytest tests/data --fail-on-skip
python -m pytest -m weights --fail-on-skip
python -m pytest -m gpu --fail-on-skip

python -m ruff check .
python -m pyright
```

`--requirements-report` records selected node IDs, level, requirement IDs, outcome, skips
and failure text. Requirement tags are an index, **not proof that every clause of a
requirement passes**. `--fail-on-skip` fails an otherwise successful run if runtime or
module-level collection skips occurred. Point it at the intended test files when optional
unrelated modules are unavailable. A custom `--basetemp` needs an existing parent directory.

The fixtures use two Torch threads by default (`RULFUSE_TEST_THREADS` overrides this).
CUDA tests set the deterministic cuBLAS workspace before creating a device. Existing
training code still controls deterministic-kernel warnings; an exact resume assertion
remains mandatory on GPU. This CPU machine cannot certify that assertion.

## What the expanded checks enforce

- Photographs must not affect predictions at or before their availability, including the
  **first** photograph. Perturbations and gradients check independent failure paths.
- Later signal tokens, adapter labels/metadata, and held-out statistics must not influence
  earlier predictions or train-fitted transforms. A cross-fitted readout must exclude the
  target tool's labels, including hyperparameter selection.
- Segments tile samples; seconds, gaps, caps and masks agree; augmentation preserves the
  baseline, failure recording, photo attachment, and exact recomputed targets.
- HI-only warm-up, equal tool weighting, detached NLL mean and temporal references,
  physical-lag/cap masks and EMA arithmetic follow their stated contracts.
- Foreign/incomplete/nonfinite feature caches and corrupt prediction tables must be
  rejected before they can create plausible scientific results. These strict acceptance
  tests currently expose missing checks; they are not marked `xfail`.
- One-seed aggregation retains Student-t uncertainty; no artificial seed spread is needed.
  Current study configurations request seed 0, Gate 3 has 10 outer + 40 inner tasks, and
  packed shell tails cover precisely the remaining manifest lines.
- Complete synthetic fits exercise F0/F1, with/without HI and photographs, resumption
  across phases/checkpoints, out-of-fold fixed-step calibration, a frozen configuration's
  hash, and regenerated reports. Poisoning synthetic test predictions must not affect
  validation decisions or reports. Real held-out test results stay sealed under D15.

## Verification matrix and remaining gaps

This matrix describes behaviors exercised, not unconditional completion of a project part.
See the dated test review for measured results and known failures.

| Plan part | Automated evidence | Remaining acceptance |
|---|---|---|
| P1 loading | `test_loader`, raw-adapter end-to-end, EDA counts | Fresh full raw/photo hashing and cross-machine manifest comparison |
| P2 trim | `test_trim`, randomized/boundary preprocessing, ramp regression | Re-run all flagged real trim plots after any trimming change |
| P3 timeline | `test_timeline`, gap-preserving augmentation, EDA counts | Scientific gap/wear interpretation is reviewed evidence |
| P4 segmentation | `test_segments`, randomized exact sample tiling, raw pipeline | EDA counts now pinned to approved default store |
| P5 labels | `test_failure`, `test_targets`, flips/end cases, time scaling | Every alternative TTF rule's real EDA table remains a data acceptance check |
| P6 folds | `test_folds`, one-seed outer/inner manifest layout | None beyond passing disjointness/cyclic contracts |
| P7 normalization | `test_normalization`, `test_tokens`, held-out poison | Shift/identification measurements are diagnostics, not a pass gate |
| P8 encoders/cache | `test_cache`, `test_encoders`, analytic channels, corrupt artifacts | Current cache validation failures; real Mantis/DINO/offline checks need weights/Ruche |
| P9 readouts/probe | `test_probe`, cross-fit label poison, feature overrides | Real probe/shuffled-label acceptance and admission artifacts |
| P10 time/tokens | `test_causality`, `test_tokens`, future-signal/photo perturbation | Confirmed F0 first-photo failure |
| P11 fusion | causal/gradient tests, padded batches, modality combinations, channel permutation | F0 first-photo failure; full real scientific sanity checks |
| P12 heads/loss | hand computations, gradient isolation, equal tool weights, cap/lag masks | Some nonphysical inputs accepted; real scale recovery remains scientific validation |
| P13 augmentation | `test_tokens`, ten deterministic randomized trajectory trials | Empirical dropout-rate and augmented-life-distribution acceptance can be expanded |
| P14 training | schedule/selection tests; F0/F1 exact CPU resume; actual inner fits; all seven sanity-runner paths executed on synthetic fixtures | Vacuous sanity-selection failure; CUDA resume and real scientific sanity checklist on deployed environment |
| P15 uncertainty | one-seed moments/intervals, Monte Carlo, conformal/filter causality, calibration lifecycle | Corrupt artifacts accepted; real independent coverage remains a gate |
| P16 baselines | `test_baselines`, `test_degradation`, toy feature/baseline CLI | Full real B0-B2 reproduction and R1 scientific superiority |
| P17 metrics/stats | `test_evaluation`, `test_metrics`, `test_stats`, `test_uncertainty` | Statistical tests have finite Monte Carlo tolerances |
| P18 importance | Existing additive Shapley/helper tests | Full main RUL-Fuse masking/retraining/importance study is not implemented/verified |
| P19 sensitivity | Configuration identity and input guards; alternative-label helpers | Full main sensitivity-arm runner and every intended-field-only arm check are absent |
| P20 pipeline | one-seed manifests, reuse/resume, conditional study, fake scheduler/status | Incomplete predecessor accepted; real offline J0/pilot and CPU/GPU agreement |
| P21 report | Identical regenerated synthetic report/decision bytes and freeze hash | Full final main report pipeline/golden reference and cross-machine reproduction |
| P22 transfer | `test_transfer_contract`, synthetic raw pipeline, no-photo/no-HI fits | Broader real transfer datasets are untested |

No finite suite can uncover every harmful behavior. Code/branch coverage, requirements
mapping, adversarial tests, real-data checks and scientific validation supply different
evidence; none substitutes for the others. A passing synthetic suite cannot certify the
dataset's label quality or a deployment's scientific performance.
