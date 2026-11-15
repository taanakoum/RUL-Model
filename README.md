# RUL-Fuse: multimodal remaining-useful-life prediction for milling tools

RUL-Fuse predicts how many seconds of cutting a milling tool has left. It reads two inputs:
the tool's **force signals** and its **photographs**. It outputs a calibrated interval
together with the point estimate, for example "≈ 640 s, 90 % between 420 and 900 s".

It is built and evaluated on the public **Nonastreda** milling dataset: 10 tools run to
failure, 512 passes, 3-axis force, and tool photos taken between passes.

> **Status (November 2026): in progress.** The pipeline, models and evaluation machinery are
> implemented and tested. The final design is selected on validation tools only. Test-tool
> scores stay sealed until the design freeze on 1 December 2026, so this README reports no
> headline accuracy yet.

## What makes it interesting

- **Strictly causal by construction.** A prediction at time *t* may only use signals recorded
  and photos taken before *t*.
  - Tests check that perturbing any future signal or photo leaves every earlier output
    unchanged, gradients included.
  - The checks cover padded batches, dropped photos, tools with no photo yet, and truncated
    sequences.
- **Two fusion families compared under one protocol:**
  - **F1**, a joint causal transformer over signal and image tokens;
  - **F0**, late fusion as a generalized product of experts (usage, signal and image experts
    with Student-t heads), whose expert weights read directly as time-resolved modality
    importance.
- **Honest evaluation.**
  - Leave-one-tool-out cross-validation, with a separate validation tool in every fold.
  - Every design choice goes through pre-registered gate rules with fixed thresholds.
  - Test predictions are written but never read before the freeze; the code enforces this.
- **Calibrated uncertainty.** Student-t predictive distributions are calibrated on
  out-of-fold predictions from inner fits. A scale method and a cross-conformal method are
  compared by Winkler score under a coverage constraint, and an optional causal Kalman filter
  enforces that remaining life falls one second per second of cutting.
- **Dataset-agnostic core.** Everything specific to Nonastreda lives in one adapter, which
  turns a dataset into generic units, recordings, images and failure times. A synthetic toy
  adapter proves that the core transfers.
- **Reproducible at cluster scale.**
  - Deterministic fits that resume exactly after an interruption, verified on CPU and A100.
  - Each fit records an identity (configuration, encoders, code version), so an identical fit
    is copied instead of recomputed, and a stale one is never reused.
  - SLURM job arrays pack several fits per GPU.

## Method in one picture

```text
force (3 axes) ──► Mantis-8M embeddings + level descriptors ─┐
tool photos ─────► DINOv2-B tokens + cross-fitted wear readout ┼─► causal fusion (F0 / F1)
operating time ──► usage features ───────────────────────────┘        + causal GRU memory
                                                                          │
              Student-t head (μ, σ, ν) ──► seed ensemble ──► calibration ──► optional Kalman filter
```

The encoders are frozen. Targets are capped at a horizon H = 1,500 s, and the headline metric
is the window MAE (s) over points whose true RUL is at most H.

## Repository layout

| Path | Contents |
|---|---|
| `rulfuse/core/` | Generic schema, segment store, targets, folds, normalization |
| `rulfuse/adapters/` | Nonastreda adapter (raw MAT/CSV files to the generic schema) and a toy adapter |
| `rulfuse/features/` | Frozen encoders, feature caches, readout probes, the feature check |
| `rulfuse/models/` | F0 and F1 fusion, Student-t and HI heads, losses, causal memory |
| `rulfuse/train.py`, `study.py` | Training loop, studies, gate decisions, fit reuse |
| `rulfuse/uncertainty.py`, `gate3.py` | Predictive mixture, calibration, Kalman filter, the design freeze |
| `rulfuse/baselines.py`, `degradation.py` | Usage-only and degradation-model baselines |
| `rulfuse/omnifuser_rul/` | Baseline B6, an OmniFuser architecture adapted to RUL |
| `configs/studies/` | The studies (gates), as JSON |
| `slurm/` | Job scripts for the Ruche cluster (Mésocentre Paris-Saclay) |
| `tests/` | Unit, integration, end-to-end, adversarial and real-data tests |
| `PROJECT_PLAN.md` | The full research plan: decisions, gate rules, verification matrix |
| `docs/` | Exploratory analysis, the end-to-end data trace, the B6 plan, testing guide |

## Getting started

Python 3.11. Install PyTorch first, for your platform, then:

```bash
pip install -e ".[test,dev]"            # core, tests, lint
pip install -e ".[encoders]"            # frozen encoders (Mantis-8M, DINOv2-B)

python -m pytest -m "not data and not weights and not gpu"   # synthetic suite, CPU only
python -m ruff check . && python -m pyright
```

The real dataset is not distributed with the code. Point `RULFUSE_DATA_ROOT` at a copy of
Nonastreda, then the command-line interface builds the pipeline step by step:

```bash
python -m rulfuse.cli build-store                 # segment store from the raw files
python -m rulfuse.cli extract-features --kind signal
python -m rulfuse.cli extract-features --kind image --kinds tool
python -m rulfuse.cli manifest --study gate2a-fix # then run its tasks on a GPU
python -m rulfuse.cli aggregate --study gate2a-fix
```

`docs/TESTING.md` describes the test categories and what each one checks.

## Acknowledgements

The Nonastreda dataset is the work of its authors. The B6 baseline builds on an OmniFuser
reproduction (`omnifuser-reproduction`). The encoders are Mantis-8M and DINOv2-B.
