# RUL-Fuse on Ruche: runbook

How to run this project's jobs on Ruche, the Mésocentre Paris-Saclay cluster. It is built
on what OmniFuser and Project 14 learned there (their `TRAINING_RUNBOOK_RUCHE.md` and
`docs/RUCHE_RUNBOOK.md`), adapted to this repository.

**Part A** sets Ruche up once (about 1 hour, mostly waiting for downloads).
**Part B** runs the experiments, starting with Stage 3 (features and the feature check).
**Part C** covers watching, recovering, downloading and troubleshooting.

Every block marked `bash` is typed **on Ruche** (in the SSH session). Every block marked
`powershell` is typed **on your laptop**, in the repository folder.

---

## Overview

### What runs where

| Work | Where | Why |
|---|---|---|
| Code, tests, plan, small analyses | Laptop | No GPU needed |
| Packing data and code for Ruche | Laptop (`scripts/pack_for_ruche.ps1`) | The data lives there |
| Downloading model weights | **Ruche login node** (`slurm/prefetch_weights.sh`) | Compute nodes have no internet; the laptop keeps no weights |
| Stores, encoder features, feature check (Stage 3) | Ruche jobs J0-J3 | Your instruction: encoders run on Ruche |
| Training studies (Stages 4-8) | Ruche job arrays (J1, J5-J13) | Plan §16.4 |
| Baseline B6, OmniFuser-RUL: one full training run and its report | Ruche jobs `b6-train`, `b6-report` (§12.1-§12.2) | D20; after the Stage-3 photo decision |
| Reading reports, writing the log | Laptop | Results are copied back (§13) |

### The path at a glance

```text
laptop: commit + push ─┐   pack data (1.2 GB) ─┐
                       ▼                        ▼
Ruche (once): connect → ~/.conda link → git clone → upload data → conda env → weights
Ruche (Stage 3): bash slurm/stage3.sh
   J0 checks (cpu_short) ┐
   J0g GPU check (gpu_test) ┴→ J2 features (gpua100) → J3 feature check (cpu_short)
laptop: copy back the report (+ the 55 MB of caches) → PROJECT_LOG.md
```

### What OmniFuser and Project 14 taught us, and where it is applied

| Lesson | Applied in |
|---|---|
| The default PyTorch from `pip` has no kernels for the V100s of `gpu_test`; the CUDA 12.6 build runs on V100 and A100 | §7.2 installs from the `cu126` index |
| `conda init` and `conda activate` edit `~/.bashrc`, which Ruche forbids; use `module load` + `source activate` | `slurm/env.sh` |
| A PyTorch environment fills the 50 GB home quota; conda must live in `$WORKDIR` | §4, the `~/.conda` link |
| Jobs submitted with `--export=NONE` start without the `module` command | `slurm/env.sh` sources `/etc/profile` |
| A job whose log folder does not exist ends at once, without a log | `slurm/stage3.sh` and `chain.sh` create `runs/logs`; §10 does too |
| Compute nodes have no internet; a missing checkpoint fails a job hours after submission | Weights are prefetched on the login node; jobs run offline (`HF_HUB_OFFLINE=1`); job J0 proves offline loading before any GPU job starts |
| Thread pools larger than the granted cores slow everything down | Every job sets `OMP_NUM_THREADS` to `--cpus-per-task` |
| Zip files made by Windows Explorer unpack into wrong names on Linux | `tar` only (`scripts/pack_for_ruche.ps1`) |
| CRLF line endings break shell scripts on Linux | `.gitattributes` forces LF for `*.sh` and `*.sbatch` |
| Long chains of dependent jobs are hard to follow | One submit script per stage, printing every job ID; `slurm/status.sh` on one screen |
| Results compared with each other must come from the same GPU type | All feature and training jobs use `gpua100`; only the GPU check uses `gpu_test` |
| Interrupted work must resume, not restart | Feature caches resume from `done.npy`; fits will resume from checkpoints (Stage 4) |
| Neither `$HOME` nor `$WORKDIR` is backed up | §13: copy results home after every stage |

---

# Part A: Set up Ruche (once)

## 1. Concepts you need

- **Login node** (`ruche01`, `ruche02`): the computer you reach with SSH. You install,
  download, edit and submit there. **Never compute there**: heavy processes are killed and
  you get a warning email.
- **Compute node**: where jobs run, with CPUs or GPUs. You never log in to one directly
  (except for short interactive debugging, §16).
- **Job**: a script plus its resource request (CPUs, GPUs, time, partition), handed to
  SLURM with `sbatch`. It waits in a queue, then runs without you. Its output goes to a log
  file.
- **Partition**: a group of nodes with its own limits (§9).
- **Job ID**: the number `sbatch` prints; you use it to follow, inspect or cancel a job.
- **Dependency**: a job can wait for others (`--dependency=afterok:ID`): it starts only if
  they succeeded.
- **Module**: preinstalled software you load with `module load NAME`. This project loads
  one: conda.
- **`$WORKDIR`** (`/workdir/<login>`): your 500 GB work space. **`$HOME`**: 50 GB, for
  small configuration files only.

## 2. Prepare on your laptop

**2.1 The code.** Ruche clones the repository from GitHub
(`https://github.com/taanakoum/RUL-Model.git`, branch `main`). Your work must therefore be
committed and pushed:

```powershell
git status
git push
git status -sb
git log -1 --oneline origin/main
```

- `git status -sb` must show `## main...origin/main` with nothing after it; `[ahead 1]`
  means a commit is not pushed yet.
- The last line is the commit Ruche will get; note it in `PROJECT_LOG.md`.
- The tests must pass on the laptop first:
  `D:\RULModel\.venv\Scripts\python.exe -m pytest -q`.

**2.2 The data.** Pack the raw copy (labels, MAT file, 3 × 512 photos, `MANIFEST.json`) into
one archive with its checksum:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pack_for_ruche.ps1
```

It checks that every file is present (512 photos per type), writes
`D:\RULModel\transfer\nonastreda_raw.tar` (about 1.2 GB) and
`nonastreda_raw.tar.sha256`, and prints the `scp` line of §6. Nothing else is uploaded:
Ruche rebuilds the force store, the segment store and the caches itself (job J0), so its
results do not depend on files made on Windows.

## 3. Connect to Ruche

You need a Ruche account (login and password) from the Mésocentre. In PowerShell, once
per window:

```powershell
$RucheLogin = "your-ruche-login"
$Ruche = "$RucheLogin@ruche.mesocentre.universite-paris-saclay.fr"
ssh $Ruche
```

- Replace `your-ruche-login`. The two variables are reused by `scp` (§6, §13); define them
  again in every new PowerShell window.
- The first connection asks whether to trust the host key: answer `yes`. This laptop
  already knows Ruche's key (it is in `~\.ssh\known_hosts`).
- Type your password; nothing appears while you type.
- `exit` closes the connection; jobs keep running.

**Optional, recommended: an SSH key**, so `ssh` and `scp` stop asking for the password:

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh $Ruche "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
ssh $Ruche
```

The third line must log you in without the password. Never share the file without `.pub`.

## 4. Storage: where things go

| Path | Variable | Limit | Holds |
|---|---|---|---|
| `/home/<login>` | `$HOME` | 50 GB, 200,000 files | Configuration only; `~/.conda` is a link to `$WORKDIR` |
| `/workdir/<login>/RUL-Model` | | | The code; job logs in `runs/logs/` |
| `/workdir/<login>/rul_data` | `$RULFUSE_DATA_ROOT` | | `raw/nonastreda`, `interim/` (force store 452 MB, segment stores, feature caches about 55 MB), `runs/` (reports) |
| `/workdir/<login>/hf_cache` | `$HF_HOME` | | Mantis-8M and DINOv2-B weights, about 380 MB |
| `/workdir/<login>/torch_cache` | `$TORCH_HOME` | | EfficientNet-B0 ImageNet weights for B6, 21 MB |
| `/workdir/<login>/.conda` | | | The `rulfuse` environment, about 6 GB |

**Neither directory is backed up.** Copy results home (§13). Check usage with
`ruche-quota`.

Move conda to `$WORKDIR` **before** creating the environment (skip if you already did it
for OmniFuser; the last line must show `.conda -> /workdir/<login>/.conda`):

```bash
if [ -d "$HOME/.conda" ] && [ ! -L "$HOME/.conda" ] && [ ! -e "$WORKDIR/.conda" ]; then mv "$HOME/.conda" "$WORKDIR/.conda"; fi
mkdir -p "$WORKDIR/.conda"
if [ ! -e "$HOME/.conda" ]; then ln -s "$WORKDIR/.conda" "$HOME/.conda"; fi
ls -ld "$HOME/.conda"
```

## 5. Get the code

```bash
cd "$WORKDIR"
git clone https://github.com/taanakoum/RUL-Model.git RUL-Model
cd RUL-Model
git log -1 --oneline
```

- The last line must show the commit you noted in §2.1.
- If the repository is private, GitHub asks for your username and a **personal access
  token** (GitHub → Settings → Developer settings → Tokens), not your password.
- Later updates: `git pull` inside `RUL-Model/`. **Never pull while a job of the same stage
  is running**: caches check their inputs, not the code.
- **If the clone fails**, send the committed code as an archive instead:

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\pack_for_ruche.ps1 -Code
  scp D:\RULModel\transfer\RUL-Model-code.tar.gz "${Ruche}:/workdir/$RucheLogin/"
  ```

  then on Ruche `cd "$WORKDIR" && tar -xzf RUL-Model-code.tar.gz && cd RUL-Model`.
  Without git, updates repeat this, and `git log` in the job logs shows `unknown`.

## 6. Upload the data

On the laptop (after §2.2):

```powershell
scp D:\RULModel\transfer\nonastreda_raw.tar D:\RULModel\transfer\nonastreda_raw.tar.sha256 "${Ruche}:/workdir/$RucheLogin/"
```

On Ruche:

```bash
cd "$WORKDIR"
sha256sum -c nonastreda_raw.tar.sha256
mkdir -p rul_data/raw
tar -xf nonastreda_raw.tar -C rul_data/raw && rm nonastreda_raw.tar nonastreda_raw.tar.sha256
ls rul_data/raw/nonastreda
ls rul_data/raw/nonastreda/tool | wc -l
```

- `sha256sum -c` must print `nonastreda_raw.tar: OK`; anything else means a broken
  transfer: upload again.
- The first `ls` must list `MANIFEST.json chip forces_xyz_raw.mat labels.csv
  labels_reg.csv tool work`; the count must be `512`.
- Job J0 then checks every file against the publisher's hashes (`rulfuse verify-data`).

## 7. Python environment (once)

**7.1 The conda module.**

```bash
module avail anaconda miniconda
module load anaconda3/2023.09-0/none-none
```

Use the newest `anaconda3` or `miniconda3` entry the first line lists.

**7.2 Create the environment and install the project** (10-20 minutes, a few GB; allowed
on the login node):

```bash
cd "$WORKDIR/RUL-Model"
conda create -n rulfuse python=3.11 -y
source activate rulfuse
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -e ".[test,encoders,omnifuser,eda]"
```

| Command | What it does |
|---|---|
| `conda create -n rulfuse python=3.11 -y` | An environment named `rulfuse` with Python 3.11 (the project needs 3.11 or 3.12). If it fails with `PackagesNotFoundError`, add `-c conda-forge` |
| `source activate rulfuse` | Switches to it. **Not** `conda activate`, **not** `conda init` |
| `pip install torch torchvision --index-url …/cu126` | PyTorch and the matching torchvision for CUDA 12.6: kernels for the V100s (`gpu_test`) and the A100s (`gpua100`). Installed together, pip picks a torchvision built for that torch (a mismatched pair fails with errors such as `operator torchvision::nms does not exist`) |
| `pip install -e ".[test,encoders,omnifuser,eda]"` | numpy, scipy, Pillow, pyarrow, pytest, transformers, mantis-tsfm, OmniFuser v1.1 (from its GitHub tag, for baseline B6), matplotlib (B6's report figures) and this project (the `rulfuse` command); keeps the PyTorch above. `--no-cache-dir` keeps pip's cache out of the home quota |

Check:

```bash
python -c "import torch, torchvision, transformers, mantis, omnifuser, rulfuse; print(torch.__version__, torchvision.__version__, transformers.__version__, omnifuser.__version__)"
which rulfuse
```

The first line must print a torch version ending in `+cu126`, a torchvision ending in
`+cu126` and OmniFuser `1.1.0`.

**An environment made before 2026-11-05** (without B6) is upgraded in place:

```bash
source slurm/env.sh
python -m pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -e ".[test,encoders,omnifuser,eda]"
python -c "import torch, torchvision, omnifuser; print(torch.__version__, torchvision.__version__, omnifuser.__version__)"
```

If the torchvision install wants to replace torch, install the pair again together
(`pip install --force-reinstall torch torchvision --index-url …/cu126`).

**7.3 Tell the job scripts about the environment.** Open `slurm/env.sh` (`nano
slurm/env.sh`; save with Ctrl+O, Enter; quit with Ctrl+X) and set `CONDA_MODULE` to the
module of 7.1 and `CONDA_ENV="rulfuse"`. Every job loads the environment through this file.
It also sets:

- `RULFUSE_DATA_ROOT=$WORKDIR/rul_data`;
- `HF_HOME=$WORKDIR/hf_cache` and `TORCH_HOME=$WORKDIR/torch_cache`;
- inside jobs only, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- `PYTHONUNBUFFERED=1`, so logs show progress while a job runs.

```bash
source slurm/env.sh
echo "$RULFUSE_DATA_ROOT $HF_HOME"
rulfuse --help
```

## 8. Download the weights (login node, once)

```bash
cd "$WORKDIR/RUL-Model"
source slurm/env.sh
bash slurm/prefetch_weights.sh
```

It downloads only what the loaders read (`config.json`, `model.safetensors`) of
`paris-noah/Mantis-8M` (31 MB) and `facebook/dinov2-base` (about 350 MB) into `$HF_HOME`.
DINOv2's duplicate `pytorch_model.bin` is skipped. With torchvision installed (§7.2) it
also fetches B6's ImageNet EfficientNet-B0 (21 MB) into `$TORCH_HOME`. The fallback encoder
MOMENT-1-base is fetched only if the force go/no-go check fails (§11.5). No weights are
kept on the laptop.

## 9. SLURM in 10 minutes

### Partitions this project uses

| Partition | Hardware per node | Limit | Used for |
|---|---|---|---|
| `cpu_short` | CPU only | 1 h, one job at a time | J0 checks, J3 feature check |
| `gpu_test` | 4 × V100 32 GB | 1 h, one job at a time | J0g GPU check |
| `gpua100` | 4 × A100 40 GB, 32 CPUs | 24 h; at most 4 GPUs in use per user | J2 features and every training job |

Memory is 4 GB per CPU core: 8 cores give 32 GB. Ruche asks users to prefer the A100s.
`sinfo -s` lists the current partitions if a name changes.

### Commands

| Command | What it does |
|---|---|
| `sbatch [options] SCRIPT [arguments]` | Submits a job; prints `Submitted batch job 123456` (`--parsable`: only the ID) |
| `squeue -u $USER` | Your waiting (`PD`) and running (`R`) jobs; the last column gives the node or the reason a job waits |
| `scancel ID` | Cancels a job |
| `sacct -j ID --format=JobID,JobName,State,Elapsed,ExitCode` | `COMPLETED`, `FAILED`, `TIMEOUT` or `CANCELLED` |
| `seff ID` | Time, CPU and memory a finished job used |

Reasons in the last column of `squeue`:

| Reason | Meaning |
|---|---|
| `Priority`, `Resources` | Waiting for its turn: normal |
| `Dependency` | Waiting for the jobs it depends on |
| `DependencyNeverSatisfied` | A job it depends on failed or was cancelled: read that job's log, fix, `scancel` the waiting jobs, resubmit |
| `QOSMaxGRESPerUser` | You already use your 4 GPUs; it starts when one frees |

### The scripts in `slurm/`

| File | Role |
|---|---|
| `env.sh` | Loads the module and the environment; sets the data root, the weight cache and offline mode (§7.3) |
| `job.sbatch` | Runs **one command** on one GPU: `gpua100`, 8 CPUs, 24 h, log in `runs/logs/%x-%j.out`. Whatever follows `slurm/job.sbatch` on the `sbatch` line is the command |
| `cpu.sbatch` | The same on `cpu_short` (8 CPUs, 1 h) |
| `checks.sbatch` | Job J0 (§11.2) |
| `features.sh` | The body of job J2 |
| `stage3.sh` | Submits J0, J0g, J2 and J3 with their dependencies |
| `chain.sh` | Repeats a job for work longer than 24 h (Stage 5 onwards, if a study needs it) |
| `status.sh` | Jobs, stores, caches, reports and logs with errors. Reads files only: fine on the login node |
| `prefetch_weights.sh` | Login-node weight download (§8) |
| `omnifuser_rul.sh` | B6: checks Stage 3's outputs and photo decision, then submits the training job and the report job chained after it (§12.1) |
| `omnifuser_results.sh` | B6: the body of the report job; writes `REPORT.md` and its figures and packs the archive to copy home (§12.2) |

Options on the `sbatch` line go **before** the script; everything after it is the command.
For example `sbatch --partition=gpu_test --time=00:30:00 slurm/job.sbatch …` runs the same
job on the test partition for at most 30 minutes.

---

# Part B: The experiments

## 10. Start of every session

```bash
ssh $Ruche            # from PowerShell, after §3's two variable lines
cd "$WORKDIR/RUL-Model"
source slurm/env.sh
mkdir -p runs/logs
git log -1 --oneline
bash slurm/status.sh
```

## 11. Stage 3: features and the feature check

### 11.1 Submit

```bash
cd "$WORKDIR/RUL-Model"
source slurm/env.sh
bash slurm/stage3.sh
```

It prints four job IDs. `cpu_short` and `gpu_test` run one of your jobs at a time each, so
J0 and J0g run in parallel, then J2, then J3.

| Job | Partition | Content | Expected time |
|---|---|---|---|
| J0 checks | `cpu_short` | `verify-data` (P1); force store from the MAT file; segment store; baselines B0-B2; offline loading of both checkpoints; the whole test suite | 10-20 min |
| J0g GPU check | `gpu_test` | A matrix product on the GPU | 1 min |
| J2 features | `gpua100` | Mantis-8M on 3,777 distinct segments × 4 channels × 20 sub-windows (about 300,000 windows), the 7 level descriptors and the 11 spectral features; then the scale and determinism checks; then DINOv2-B on 1,536 photos | 10-30 min |
| J3 feature check | `cpu_short` | The go/no-go probes, readouts and leakage probe of plan §6.2 | 2-10 min |

Each job writes `runs/logs/<name>-<ID>.out` in the repository folder.

### 11.2 What each log must show

**J0 (`runs/logs/J0-checks-*.out`).**

| Section | Must show |
|---|---|
| header | `torch 2.x+cu126`; data root `/workdir/<login>/rul_data`; `offline: 1` |
| `P1 verify the raw copy` | `files expected 1539, missing 0, mismatched 0; passes 512; …; ok=True` |
| `force store` | `"records": 512`, `"distinct_recordings": 378`, `"duplicate_groups": 132` |
| `segment store` | `segment store <sig>`; counts `"segments": 5125`, `"prediction_points": 3747`, `"window_points": 1361`, `"distinct_segments": 3777`, `"images": 1536` |
| `usage baselines` | `B0: mae 411.972`, `B1: mae 512.547`, `B2: mae 467.843, … coverage90 0.503, winkler90 5388.189` (plan §2.7) |
| `offline weights` | Two `ok` lines, each with a non-empty `revision` |
| `offline weights for B6` | `ok efficientnet_b0 …/torch_cache/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth \| omnifuser 1.1.0 \| torch 2.x+cu126 \| torchvision 0.y+cu126`. `B6 not installed: skipped` means §7.2 ran without the `omnifuser` extra |
| `tests` | `165 passed, 2 skipped` (167 tests as of 2026-11-08), no failure. The two skips are the comparisons with OmniFuser's records, whose checkout is not on Ruche. The two `weights` tests and the B6 tests (`test_omnifuser_rul.py`, `test_models.py`) must **pass**: they prove offline loading and B6 on this environment. Without OmniFuser installed the B6 module is skipped instead, which is not an error for Stage 3. The B6 tests take a few minutes of CPU |

The store signature must equal the laptop's (`rulfuse store-info` on both): the same
inputs give the same signature, so caches copied home fit the laptop's store.

**J0g.** One line such as `2.x+cu126 Tesla V100-SXM2-32GB (7, 0) matmul ok …`, with no
warning above it.

**J2 (`runs/logs/J2-features-*.out`).**

- `[features] nonastreda mantis8m: 3777 segments to encode, 0 cached`, then progress lines.
- A JSON block with `scale_check` and `determinism`; `determinism.ok` must be `true`.
- `[features] nonastreda dinov2b chip|tool|work: 512 images to encode`, then progress.
- The final `du` line: about 31 MB for the signal cache and 21 MB for the image cache.

**J3 (`runs/logs/J3-feature-check-*.out`).** `wrote …/runs/feature-check/nonastreda-<sig>.json`
and the `decisions` block, for example:

```json
{"signals_go": true, "image_types_kept": ["chip", "tool"], "image_types_excluded": {"work": "leaks the stage label"}}
```

### 11.3 Where the outputs are

| Output | Path under `$RULFUSE_DATA_ROOT` |
|---|---|
| Data verification | `runs/verify/verify-data.json` |
| Usage baselines | `runs/baselines/usage-nonastreda-<sig>.json` |
| Signal features | `interim/features/nonastreda/signal-mantis8m-<key>/` (`embeddings.npy`, `descriptors.npy`, `spectral.npy`, `done.npy`, `meta.json`, `checks.json`) |
| Image features | `interim/features/nonastreda/image-dinov2b-<key>/` (`tokens.npy`, `done.npy`, `meta.json`) |
| Feature-check report | `runs/feature-check/nonastreda-<sig>.json` |

### 11.4 Reading the report

| Field | Question it answers | Accept when (plan §6.2, P8, P9) |
|---|---|---|
| `scale_check` | Does Mantis see the absolute level? `change_over_spread` compares the change from doubling a window with the distance between different windows | Reported. The level descriptors are kept either way; they carry the level if Mantis does not |
| `determinism` | Does re-encoding reproduce the cache? | `ok: true` (relative difference below 1e-3) |
| `signals.probe["encoder+descriptors"]` | Do the force features predict the flank wear of held-out tools better than the training mean? | `beats_mean: true` → **force go** |
| `signals.probe["encoder"]`, `["descriptors"]` | What each part adds | Reported |
| `…shuffled_control` | With the wear shuffled inside each training tool, does the probe collapse to the mean? | `mae` close to `mae_mean_predictor`; a much lower value means a leak in the probe |
| `signals.level_vs_hi` | Spearman ρ of each channel's log RMS with wear (Stage 1: about 0.7 for \|F\|) | Reported |
| `images.<type>.probe` | Is the photo type informative? | `beats_mean: true` |
| `images.<type>.readout` | Cross-fitted wear readout, MAE per tool | Reported (feeds the model as the HI readout) |
| `images.<type>.leakage` | Is a `tool_label` stage boundary the largest run-to-run step of the type's first 3 PCs or of its readout? | `leaks: false`, that is a stage boundary is the largest step in **at most 3 of 10 tools** |
| `hi_reference` | The same count for the true smoothed wear | Context: how often real wear itself steps most at a stage boundary |
| `decisions` | Force go/no-go; photo types kept and excluded, with the reason | Logged as decision 1 (plan §17.3) |

### 11.5 What to do with the outcome

- **Force go, at least one photo type kept:** Stage 3 is done. Copy the results home (§13)
  and log them; Stage 4 starts.
- **Workpiece excluded for leakage:** expected (Stage 2). It leaves the headline model and
  the Shapley study; the report is the leakage demonstration.
- **A type excluded narrowly, by one tool on one statistic:** compare its counts with
  `hi_reference`, which says how often real wear itself steps most at a boundary. Only you
  can readmit a type, before any RUL result, in `configs/feature_check_overrides.json`, with
  the date, the report and the reason; every model reads that file. Nonastreda's first
  report (2026-11-11) excluded all three photo types:
  - work (10 of 10 tools) and chip (readout 7 of 10) stay out;
  - tool was readmitted (D24): PC2 4 of 10, readout 3 of 10, the same as real wear.
- **Force no-go:** the plan's fallback is MOMENT-1-base. Its wrapper is not implemented
  yet: report the result, and the wrapper and its prefetch
  (`bash slurm/prefetch_weights.sh AutonLab/MOMENT-1-base`) are added before rerunning J2.
- **A job failed:** read its log (`tail -n 40 runs/logs/<name>-<ID>.out`), fix, `scancel`
  the jobs still waiting, then rerun `bash slurm/stage3.sh`. Finished parts are reused:
  the stores are found by signature and the caches resume from `done.npy`.

To rerun one piece alone:

```bash
sbatch slurm/checks.sbatch
sbatch --job-name=J2-features --time=02:00:00 slurm/job.sbatch bash slurm/features.sh
sbatch --job-name=J3-feature-check slurm/cpu.sbatch rulfuse feature-check
```

## 12. Later stages (job plan, plan §16.4)

These jobs need code that Stages 4-9 build (the training loop, `slurm/array.sbatch`,
manifests). This section is completed as each stage lands; the plan's job table is the
reference.

| Job | Stage | Content | Partition |
|---|---|---|---|
| J1 | 4 | Pilot: one fold, F1 on the cached features; measures the cost of a fit | `gpua100` |
| J5 | 5 | Gate 2a: 2 families × 3 learning rates × 10 folds (60 tasks; §15.3) | `gpua100` |
| J6 | 5 | Gate 2b: the default + 4 options + 3 controls × 10 folds (80 tasks; §15.3) | `gpua100` |
| J7 | 5 | Sweep: the default + 4 variants × 10 folds (50 tasks; §15.3) | `gpua100` |
| J8 | 6 | Gate 3: Student-t head × 10 folds, 4 inner fits per seed | `gpua100` |
| J10 | 7 | Final: 10 folds × 3 seeds + 4 inner fits each | `gpua100` |
| J11 | 8 | Shapley: 3 coalitions × 10 folds | `gpua100` |
| J12 | 8 | Sensitivity tier 1 (tier 2 dropped, D26 B5) | `gpua100` |
| J13 | 9 | Report | `cpu_short` |
| `b6-train` | after J3 | Baseline B6, the full run: its caches, then 10 folds × 3 seeds (§12.1) | `gpua100` |
| `b6-report` | after `b6-train` | B6's report and archive (§12.2) | `cpu_short` |

Rules that will hold for all of them: arrays with `%4` (at most 4 GPUs at once), resumable
fits that skip `DONE` ones, manifests committed with their study, and no test score read
before the design freeze (D15).

### 12.1 Baseline B6, OmniFuser-RUL (D20): one full training run

B6 is OmniFuser's architecture on this project's targets and inputs
(`docs/OMNIFUSER_RUL_PLAN.md`). It needs Stage 3's stores (J0) and its decision on which
photo types are admitted (J3), and nothing from Stages 4-9.

**One configuration, trained once, no comparison of settings** (your decision of
2026-11-07):

| Setting | Value |
|---|---|
| Inputs | Force (7 level descriptors × 4 channels, a signal flag) + every photo type J3 admits (frozen EfficientNet-B0 features) + usage (operating time, seconds into the recording) |
| Steps | 1 s segments, OmniFuser's own step; windows of 96 steps; scored on the project's 10 s points |
| Model | OmniFuser v1.1's encoder (d = 32, C2F fusion) + a **GRU** memory of the whole history (hidden 64) that reads OmniFuser's encoded force series and encoded photos of each complete 96-step block + a linear RUL head |
| Loss | Point-weighted MSE **+ 1/15 × monotonic penalty** (a predicted RUL higher than an earlier one of the same tool) **+ 0.05 × slope consistency** (below the cap, RUL falls by exactly the elapsed time). Both terms move only the earlier prediction of a pair, so a wrong early prediction never drags later ones down |
| Inference | Two modes from the same fits: **current** (each prediction as the model makes it; the primary result) and **context** (the tool's earlier predictions, aged by the time since, blended with the new one; time constant 120 s). The report compares them |
| Training | Adam, learning rate 1e-4, 30 epochs of 2,048 pairs, gradients clipped at 1; epoch chosen on each fold's validation tool |
| Folds and seeds | 10 folds (each tool the test tool once) × 3 seeds = 30 fits; predictions are the mean of the seeds |

**Ready when:**
- §7.2 installed the `omnifuser` extra (OmniFuser `v1.1`, GitHub tag `v1.1`, commit
  `eac0fed`) and `eda` (matplotlib, for the figures);
- §8 fetched EfficientNet-B0;
- J0's log shows the `offline weights for B6` line and the B6 tests passing.

**Submit** (the players are `force` and the photo types J3 admitted):

```bash
cd "$WORKDIR/RUL-Model" && source slurm/env.sh
bash slurm/omnifuser_rul.sh force tool
```

It submits two jobs:

| Job | Partition | Content | Expected time |
|---|---|---|---|
| `b6-train` | `gpua100`, 16 h limit | The 1 s store, its descriptor cache (about a minute), the photo features (a few minutes), then the 30 fits | 5-15 min per fit (the memory encodes about 300 past windows per batch), about 2.5-7.5 h in all |
| `b6-report` | `cpu_short` | Starts only if `b6-train` succeeded: writes the report (§12.2) and the archive | a few minutes |

The script refuses to submit when:
- the force store is missing (Stage 3 has not run);
- no feature-check report exists;
- a photo type you give is not admitted: the report's `image_types_kept` plus the types
  readmitted in `configs/feature_check_overrides.json` (B6 uses force plus every admitted
  type, decision O5; on Nonastreda that is `tool`, readmitted by D24;
  `B6_ALLOW_UNCHECKED=1` skips this check).

**While it runs:** `tail -f runs/logs/b6-train-<ID>.out` shows one line per epoch (the numbers below are only an example):

```text
fold 3 seed 1 epoch 12/30: validation MAE 231.4 s; loss mse 0.0412 hinge 0.00031 slope 0.00122
[omnifuser-rul] epoch 17 kept; 214 s
```

`hinge` is the monotonic penalty (the mean rise of rising pairs, on y/H): it should fall
as training removes rising predictions. On the first fits, check that the validation MAE
moves after the first epochs. On the tiny toy, 96-step memory blocks once left it stuck
at its first value for 4 epochs; real tools complete 19-56 blocks each, so it is not
expected here. `bash slurm/status.sh` shows the fits as `done/30`.

**If `b6-train` stops** (16 h limit, node failure), run `bash slurm/omnifuser_rul.sh …`
again: finished fits are skipped, and a new report job is chained. If the log shows more
than 30 minutes per fit, the 30 fits will not fit in 16 hours. Resubmitting continues
the run.

### 12.2 The report

`b6-report` runs `rulfuse omnifuser-report` and writes
`$RULFUSE_DATA_ROOT/runs/omnifuser_rul/nonastreda-<store>/force+tool/report/`:

| File | Content |
|---|---|
| `REPORT.md` | The run (settings, fits, time per fit, selected epochs), the test results, the paired comparison with B2, per-tool table, error by RUL range, **the two inference modes compared** (test and validation), validation results |
| `run_summary.json` | Every number of the report |
| `test_per_unit.csv`, `validation_per_unit.csv` | One row per tool and inference mode (`mode` column): window MAE, RMSE, bias, whole-life MAE, late and early rates, asymmetric score, share within ±20 %, rising steps, each seed's MAE, and B0, B1, B2 on the same points |
| `mode_comparison.csv` | One row per tool and role: window MAE in each mode, the gain (current − context; positive = the context mode is better), the bias and rising steps in each mode |
| `error_by_bin.csv`, `test_trajectories.csv`, `curves.csv` | MAE by true-RUL range (B6, B6 context, B2); true and predicted RUL of each test tool over its life, in both modes; every epoch's validation MAE |
| `figures/` | `rul_trajectories.png` (true RUL, both modes with the seed range, and B2, per test tool), `window_mae_per_tool.png`, `error_by_rul.png`, each with both modes |

**What the test results report.** Each of the 10 tools is the test tool of one fold, so
every tool gets one result. Each is scored on the project's 10 s points with RUL ≤ 1,500 s,
the same points as RUL-Fuse and baselines B0-B2.

| Quantity | Meaning |
|---|---|
| Window MAE, RMSE, bias | The main accuracy, mean ± sd over the 10 tools, with a 95 % bootstrap interval over tools; bias > 0 means too optimistic |
| Late / early predictions | Share of points predicted more than 100 s too optimistic (the dangerous error) or too pessimistic |
| Asymmetric score | Plan §10.2's score, which penalizes late predictions more than early ones |
| Within ±20 % | Share of points whose prediction is within 20 % of the true RUL |
| Rising steps | Share of consecutive 1 s steps where the predicted RUL rises: what the monotonic penalty pushes down |
| Against B2 | Mean MAE gain over the usage rule B2 (467.8 s), its 95 % interval, tools improved, and a one-sided paired Wilcoxon test |
| Seed sd | How much the 3 seeds disagree on each tool |
| Inference modes | Each metric in both modes, context − current with its 95 % interval, the tools where each mode wins, and a two-sided Wilcoxon test. **Verdict rule, fixed in advance:** the context mode is better if it lowers the mean window MAE by at least 10 s and the interval of that gain lies above 0; the current mode under the mirror condition; otherwise neither. The validation comparison is fair (the time constant was fixed in advance); the test comparison follows the seal |

**When the test results appear.** Before the design freeze (2026-12-01, D15) the report
contains the run and the validation results, and says the test results are sealed. From
2026-12-01 the same job includes them. To read them earlier, run the report again with
`--unseal`; the report then opens with a note that the test was read before the freeze:

```bash
sbatch --job-name=b6-report slurm/cpu.sbatch bash slurm/omnifuser_results.sh --unseal force tool
```

The report can also be rebuilt on the laptop from the copied archive (§13), in seconds.

---

# Part C: Operating

## 13. Bring the results home

After Stage 3, on the laptop:

```powershell
$R = "/workdir/$RucheLogin/rul_data"
scp -r "${Ruche}:$R/runs/feature-check" "${Ruche}:$R/runs/baselines" "${Ruche}:$R/runs/verify" D:\RULModel\runs\
scp -r "${Ruche}:$R/interim/features/nonastreda" D:\RULModel\interim\features\
scp "${Ruche}:/workdir/$RucheLogin/RUL-Model/runs/logs/*" D:\RULModel\runs\ruche-logs\
```

- The first line brings the reports (a few MB); the second the two feature caches
  (about 55 MB, features only, no model weights), which Stage 4's laptop sanity checks
  read. Create `D:\RULModel\interim\features` and `D:\RULModel\runs\ruche-logs` first if
  they do not exist.
- On the laptop, `rulfuse store-info` must show the same store signature as the J0 log, so
  the caches are found (`rulfuse.features.cache.find_features` matches on it).
- Record the job IDs, the commit and the decisions in `PROJECT_LOG.md`.

**B6 results** (after §12.2), on the laptop:

```powershell
scp "${Ruche}:/workdir/$RucheLogin/b6-results-*.tar.gz*" D:\RULModel\transfer\
cd D:\RULModel\transfer
$a = (Get-ChildItem .\b6-results-*.tar.gz | Sort-Object Name | Select-Object -Last 1).Name
$a; (Get-FileHash $a -Algorithm SHA256).Hash.ToLower() -eq (Get-Content "$a.sha256").Split(" ")[0]   # True
tar -xzf $a -C D:\RULModel\runs
```

- These lines take the newest archive; each report job writes its own, named by date and
  time (`b6-results-YYYYMMDD-HHMM.tar.gz`), and the newest holds everything the older ones
  do. The second line must print `True`.

- The report is
  `D:\RULModel\runs\omnifuser_rul\nonastreda-<store>\force+tool\report\REPORT.md`,
  with its figures beside it; the job logs are in `omnifuser_rul\logs\`.
- To rebuild it on the laptop (seconds, CPU only; add `--unseal` to include the test
  results before the freeze):
  `D:\RULModel\.venv\Scripts\python.exe -m rulfuse.cli omnifuser-report --players force tool`.
- Record the fits, the test table and the comparison with B2 in `PROJECT_LOG.md`.

## 14. Watch, resume and cancel

| Task | Command |
|---|---|
| Everything on one screen | `bash slurm/status.sh` |
| Your queue | `squeue -u $USER` |
| A finished job's state | `sacct -j ID --format=JobID,JobName,State,Elapsed,ExitCode` |
| Its resources | `seff ID` |
| A log while it runs | `tail -f runs/logs/<name>-<ID>.out` (Ctrl+C stops watching, not the job) |
| Cancel one job / all yours | `scancel ID` / `scancel -u $USER` |

**Resuming.** Every step is safe to rerun. The force store and the segment stores are
reused when they exist (by signature). A feature cache encodes only what `done.npy` lacks.
A cache folder with another signature is refused, never mixed: delete it to rebuild.

**Daily routine.** `bash slurm/status.sh`. Read any log listed with an error. Copy home
whatever finished (§13). Write the log entry.

## 15. Stage 4 and Stage 5: RUL-Fuse training on Ruche

Stage 4 checks that the RUL-Fuse training code behaves (seven sanity checks) and times one
fit (the pilot). Stage 5 runs the three decision studies that choose the model's design:
Gate 2a, Gate 2b and the sweep. All of them read the Stage 3 feature caches and train on
one A100 per task.

### 15.1 Before you start

- Stage 3 is done: J0-J3 succeeded, the caches under `interim/features/nonastreda/` are
  complete, and the feature check is logged (§11). B6 (§12.1) is not needed.
- The code on Ruche is current, and the environment is loaded:

```bash
cd "$WORKDIR/RUL-Model"
git pull
source slurm/env.sh
mkdir -p runs/logs
```

### 15.2 Stage 4: sanity checks (J4) and pilot (J1)

```bash
bash slurm/stage4.sh
```

It submits two jobs that run at the same time:

| Job | Partition | Content | Expected time |
|---|---|---|---|
| J4 sanity | `gpua100`, 6 h limit | `rulfuse sanity --device cuda`: the seven checks below, on fold 1 (the toy data for check 5). Writes `$RULFUSE_DATA_ROOT/runs/sanity/report.json` | up to a few hours |
| J1 pilot | `gpua100`, 4 h limit | `bash slurm/study.sh pilot`: an array of one task (fold T1, F1 at the defaults, 3 seeds side by side) and then an aggregation job | about 1.5 h expected; this run measures the time of a fit |

Each check reads only training and validation tools (D15). The pass rules are fixed:

| # | Check | Pass when |
|---|---|---|
| 1 | Overfit one unit, regularization off | training MAE < 10 s |
| 2 | Failure times permuted among the training units | validation MAE no better than B2 − 10 s |
| 3 | Usage-only model B3, no trajectory augmentation | within 20 s of B2 |
| 4 | Oracle input: the measured causal HI of the previous recording | validation MAE < 0.8 × B2 |
| 5 | Toy data (known RUL) | validation MAE below B2 and 90 % coverage in [0.8, 0.98] |
| 6 | Truncation invariance | outputs up to a cut unchanged when later tokens are removed |
| 7 | The same seed twice on a CPU | identical weights |

Check 5's coverage band is wider than the plan's 0.88-0.92 because one toy validation tool
has few points (logged as an amendment in `PROJECT_LOG.md`).

**What you should see.**

- `runs/logs/J4-sanity-<ID>.out` has one line per check, `check N: pass {...}` or
  `check N: FAIL {...}`, then `all pass: True (…/runs/sanity/report.json)`.
- The pilot's log is `runs/logs/pilot-<ID>_0.out`. When the aggregation job has run,
  `$RULFUSE_DATA_ROOT/runs/pilot/REPORT.md` exists. The pilot is not a decision.
- **Record the time per fit.** Each fit's `selection.json` has a `seconds` field, under
  `$RULFUSE_DATA_ROOT/runs/pilot/F1-default/fold-T1/seed-<s>/`:

```bash
grep '"seconds"' "$RULFUSE_DATA_ROOT"/runs/pilot/F1-default/fold-T1/seed-*/selection.json
```

  Write the three values in `PROJECT_LOG.md`. They set the time limits of J5-J7 (§15.3).
- **If any check prints `FAIL`, do not start Stage 5.** Copy `runs/sanity/report.json` home
  (§15.5) and send it, with the log, before anything else is submitted.

To run only some checks (for example after a fix), give their numbers; `--steps N` shortens
the training phase for a quick trial:

```bash
sbatch --job-name=J4-sanity --partition=gpua100 --time=06:00:00 slurm/job.sbatch \
    python -m rulfuse.cli sanity --device cuda --checks 1 2
```

### 15.3 Stage 5: three studies, one after another

Each study inherits the decision of the previous one (`inherits` in its file under
`configs/studies/`), so they cannot be submitted together. Read each report before the next
submission.

| Job | Command | Arms × folds | Tasks | Fits (3 seeds) | Study file |
|---|---|---|---|---|---|
| J5 Gate 2a | `bash slurm/study.sh gate2a` | 6 × 10 | 60 | 180 | `gate2a.json` |
| J6 Gate 2b | `bash slurm/study.sh gate2b` | 8 × 10 | 80 | 150 run + 30 copied | `gate2b.json` |
| J7 Sweep | `bash slurm/study.sh sweep` | 5 × 10 | 50 | 120 run + 30 copied (or run) | `sweep.json` |

**Less compute, same decisions (D26, 2026-11-12).** From Gate 2b on:

- **A1:** a task's seeds run side by side on its GPU (`--parallel-seeds`, 6 CPUs, 2 threads
  per seed).
- **A2, A3:** batching and the per-fold photo readout are cached. The readout goes in
  `<store>/readouts/`.
- **B1:** a fit identical to one already finished (same store, configuration, fold and seed)
  is **copied** with a `reused_from.txt`, not rerun. The log says
  `identical fit found, copied from …`. Gate 2b's `default` is Gate 2a's winner, so its 30
  fits are copies. The sweep's `default` is a copy when Gate 2b has that exact arm.
- **B2:** Gate 2b's 3 controls run on seed 0 only.
- **B3:** the sweep keeps 4 variants.

These changes leave the results unchanged: the same seed gives the same weights, checked
before and after on the laptop.

**Gate 2a was submitted with the earlier code. Do not `git pull` while its array or
aggregation is pending or running:** every task reads the checkout when it starts, and one
study must not mix code versions. Pull once `runs/gate2a/decision.json` exists, then submit
Gate 2b.

- **Gate 2a**: learning rate of each family (F1 and F0, rates 3e-4, 1e-3, 3e-3), then F1
  against F0 at their best rates.
- **Gate 2b** (inherits Gate 2a): trajectory augmentation and phase A each stay only if they
  win against the arm without them; the token modes (`default`, `tokens-relative`,
  `tokens-absolute`) are peers; 3 of the 8 arms are **reported controls**
  (`control-last-recording`, `control-no-memory`, `control-no-hinge`) and decide nothing.
- **Sweep** (inherits both): one factor at a time against the default. There are 4
  variants (B3): `d32` and `rank8` (capacity: B6 over-fitted), `wfar0.1` (cap saturation)
  and `rho0.3` (the usage shortcut).

`slurm/study.sh STUDY` does three things in order:

1. `python -m rulfuse.cli manifest --study STUDY` writes
   `$RULFUSE_DATA_ROOT/runs/STUDY/manifest.tsv`, one line per arm × fold. It **refuses to
   start** when an inherited study has no `decision.json` yet.
2. It submits the array: `slurm/array.sbatch`, one A100 per task, 6 CPUs, **4 h limit per
   task**, at most **4 tasks at once** (`%4`). Known bad GPU nodes are excluded
   (`slurm/bad_nodes.txt`). Task N runs
   `python -m rulfuse.cli run-task --study STUDY --task N --device cuda --parallel-seeds`.
   It first copies any identical finished fit (B1), then trains the remaining seeds of that
   arm and fold side by side (A1).
3. It submits the aggregation on `cpu_short`, which starts only when the whole array
   succeeded.

**What you should see.** The script prints
`STUDY: array <ID> (N tasks), aggregation <ID> (waits for the array)` and the `scancel`
line. `bash slurm/status.sh` shows the array; array logs are
`runs/logs/STUDY-<ID>_<task>.out`. When the aggregation has run:

| File under `$RULFUSE_DATA_ROOT/runs/STUDY/` | Content |
|---|---|
| `REPORT.md` | The study's validation scores per arm and the decision, in words |
| `decision.json` | The adopted settings; the next study reads it |
| `summary.csv` | One row per arm and fold |

The aggregation log prints each arm's mean validation window MAE and `adopted: …`.

**Time limits.** A task is 3 fits and has 4 h. Gate 2a runs the seeds one after another:
measured, a fit takes about 50 min and a task about 2.5 h. From Gate 2b the seeds run side by
side (A1): a task is expected to take about 1.5 h, to be measured (next step). If a task is
much longer, stop and ask before submitting: the limit is changed in `slurm/array.sbatch`,
never by shrinking a study.

**Check the A1 speed-up (after submitting Gate 2b).** Once 4 trained tasks have finished (not
the copied default, not the seed-0 controls), run:

```bash
for f in runs/gate2b/{no-trajectory-aug,no-phase-a,tokens-relative,tokens-absolute}/fold-*/seed-*/selection.json; do [ -f "$f" ] && python -c "import json,sys;print(sys.argv[1], round(json.load(open(sys.argv[1]))['seconds']/60,1),'min')" "$f"; done
sacct -j <gate2b array job id> --format=JobID,Elapsed,State,NodeList
```

If a trained task's `Elapsed` exceeds 2:00:00, tell Claude: the sweep and every later study
then switch to seed 0 only (D27 fallback).

**Resuming.** If tasks hit the time limit or a node fails, run the same command again:

```bash
bash slurm/study.sh gate2a
```

Fits with a `DONE` file are skipped and an interrupted fit resumes from its checkpoint (one
is written at every evaluation). If the aggregation ran before every fit finished, it
printed `incomplete: N arm/fold cells missing` and wrote no decision. When the fits are
complete, run it alone on a CPU node:

```bash
sbatch --job-name=gate2a-aggregate slurm/cpu.sbatch python -m rulfuse.cli aggregate --study gate2a
```

**What Stage 5 decides on.** Only validation tools. Each fit also writes test rows, but the
aggregation never loads them (D15); the test stays sealed until 2026-12-01.

**Determinism.** Fits set deterministic kernels; the sbatch wrappers export
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### 15.4 Problems specific to these jobs

| You see | Cause | Fix |
|---|---|---|
| `manifest: … aggregate study gate2a first` (`FileNotFoundError` naming `runs/gate2a/decision.json`) | The previous study has no decision yet: it is not finished, or its aggregation did not run | Finish it (resubmit), run its aggregation (§15.3), then submit this one |
| `incomplete: N arm/fold cells missing` | The aggregation ran before every fit was `DONE` | `bash slurm/study.sh STUDY` again, then the aggregation alone (§15.3) |
| `CUDA error: CUDA-capable device(s) is/are busy or unavailable`, or `BAD GPU NODE: <node>` | The node's GPU is still held by another process (exclusive mode). It is a node fault, not a code fault: the same code runs on other nodes | `status.sh` shows each failing log's node in brackets. Cancel the waiting aggregation (`scancel <id>`; it shows `DependencyNeverSatisfied`), then resubmit without those nodes: `bash slurm/study.sh STUDY --exclude=NODE1,NODE2` (Stage 4: `bash slurm/stage4.sh --exclude=…`). Finished fits are skipped. If it persists on many nodes, tell the Ruche support |
| `CUDA out of memory` in an array log | The fit does not fit the GPU | Do not lower `copies` or any other setting in a study arm: it changes the design of a gate. Send the log and ask before changing anything |
| `DUE TO TIME LIMIT` in an array task | A fit is slower than the pilot suggested | Resubmit (§15.3); if it repeats, send the logs |
| A check prints `FAIL` | See the rule of that check (§15.2) | Do not start Stage 5; send `runs/sanity/report.json` and the J4 log |

### 15.5 Bring the results home

After Stage 4 and after each study, on the laptop (PowerShell). The three top files of a
study are enough (a few hundred KB); the fit folders hold checkpoints and stay on Ruche.

```powershell
$R = "/workdir/$RucheLogin/rul_data/runs"
New-Item -ItemType Directory -Force D:\RULModel\runs\sanity, D:\RULModel\runs\pilot | Out-Null
scp "${Ruche}:$R/sanity/report.json" D:\RULModel\runs\sanity\
scp "${Ruche}:$R/pilot/REPORT.md" "${Ruche}:$R/pilot/decision.json" "${Ruche}:$R/pilot/summary.csv" D:\RULModel\runs\pilot\

foreach ($s in "gate2a", "gate2b", "sweep") {
    New-Item -ItemType Directory -Force "D:\RULModel\runs\$s" | Out-Null
    scp "${Ruche}:$R/$s/REPORT.md" "${Ruche}:$R/$s/decision.json" "${Ruche}:$R/$s/summary.csv" "D:\RULModel\runs\$s\"
}
```

Run the `foreach` once a study has finished; copy only the studies that exist. To bring
every fit folder of a small study as well, use `scp -r "${Ruche}:$R/pilot" D:\RULModel\runs\`.

Then record in `PROJECT_LOG.md`: the job IDs, the commit, the time per fit, and the
decision of each study (`decision.json` → `adopted`). You commit the log and the study
results yourself.

## 16. Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| An email that a login-node process was killed | Computing on the login node | Use `sbatch` (or `srun`, below) |
| The job ends at once and no log exists | `runs/logs` did not exist | `mkdir -p runs/logs`, resubmit (§10) |
| `module: command not found`, `Lmod has detected the following error` | Wrong `CONDA_MODULE` in `slurm/env.sh` | `module avail anaconda miniconda`; correct it (§7.3) |
| `activate: No such file`, `No module named 'rulfuse'` | Wrong `CONDA_ENV`, or §7.2 unfinished | `conda env list`; repeat §7.2 |
| `rulfuse: command not found` | The editable install did not finish | `source slurm/env.sh && python -m pip install --no-cache-dir -e ".[test,encoders]"` |
| `LocalEntryNotFoundError`, `couldn't connect to 'https://huggingface.co'`, `offline mode is enabled` | A checkpoint missing from `$HF_HOME` | `source slurm/env.sh && bash slurm/prefetch_weights.sh` on the login node; rerun J0 |
| `no kernel image is available for execution on the device`, or a V100 "compute capability 7.0" warning | PyTorch without V100 kernels | Reinstall from `…/whl/cu126` (§7.2); the version must end in `+cu126` |
| `CUDA was requested but is not available` | A GPU command on a CPU node, or a driver too old for the build | Check the partition; the `nvidia-smi` line of the log shows the driver; reinstall a matching build |
| `verify-data` reports `mismatched` or `missing` files | A broken upload | Repeat §6; `sha256sum -c` first |
| `labels.csv has N id(s) without force records` | Wrong or partial MAT file | Repeat §6 |
| `… holds other features; delete it to rebuild` | A cache folder whose signature differs (other checkpoint revision, other store) | Delete that folder under `interim/features/nonastreda/`, rerun J2 |
| `0 complete signal-mantis8m caches for <sig>` in J3 | J2 did not finish, or the store changed since | Check J2's log; rerun J2 |
| `DependencyNeverSatisfied` | An earlier job of the chain failed | Read its log; `scancel` the waiting jobs; resubmit |
| `oom-kill`, `Exceeded job memory limit` | More than 32 GB of RAM | Add `--mem=64G` before the script name |
| `DUE TO TIME LIMIT`, state `TIMEOUT` | The walltime ran out | Resubmit the same job: it resumes |
| J0: `AssertionError: paris-noah/Mantis-8M: no cached revision`, although `prefetch_weights.sh` ran | Before 2026-11-11, the revision lookup asked for the whole snapshot. huggingface_hub 1.x refuses that (`IncompleteSnapshotError`) when README or image files are absent, and the prefetch caches only `*.json` and `*.safetensors` | `git pull` (the lookup now uses the same file patterns). Check with `HF_HUB_OFFLINE=1 python -c "from rulfuse.features.encoders import hf_revision as r; print(r('paris-noah/Mantis-8M'), r('facebook/dinov2-base'))"`, which should print two hashes. Then `scancel` the waiting J2 and J3, and run `bash slurm/stage3.sh` again |
| `Disk quota exceeded` | Home or workdir full | `ruche-quota`; check the `~/.conda` link (§4); `rm -rf ~/.cache/pip` |
| `git clone` fails | No GitHub access, or a private repository without a token | A token (§5), or the archive route |
| `sha256sum: WARNING: 1 listed file could not be read` | The checksum file has Windows line endings | Make it again with `scripts\pack_for_ruche.ps1` (it writes LF) |
| `No module named 'omnifuser'` or `'torchvision'`, or `B6 not installed: skipped` in J0 | §7.2 ran without the `omnifuser` extra | The upgrade block of §7.2, then `bash slurm/prefetch_weights.sh` |
| `operator torchvision::nms does not exist`, or torchvision import errors | torch and torchvision from different builds | Reinstall the pair together from `…/whl/cu126` (§7.2) |
| `…efficientnet_b0_rwightman-7f5810bc.pth missing`, or a download error in `b6-prepare` | The EfficientNet weights are not in `$TORCH_HOME` | `source slurm/env.sh && bash slurm/prefetch_weights.sh` on the login node |
| `photo types [...] are not admitted by …` | A photo type J3 excluded was given to `omnifuser_rul.sh` | Give only the report's `image_types_kept` (O5) |
| `test results sealed until the design freeze` in the `b6-report` log | Before 2026-12-01 the test is not read (D15) | Expected; rerun the report from that date, or with `--unseal` |
| `b6-report` pending with `DependencyNeverSatisfied` | `b6-train` failed or hit its time limit | Read its log; `scancel` the report job; run `bash slurm/omnifuser_rul.sh …` again (it resumes) |
| `REPORT.md` without figures | matplotlib is not installed on Ruche | `pip install -e ".[eda]"` (§7.2), or rebuild the report on the laptop (§13) |

**Interactive debugging on a GPU node** (at most 30 minutes):

```bash
srun --partition=gpu_test --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
cd "$WORKDIR/RUL-Model" && source slurm/env.sh
nvidia-smi
rulfuse extract-features --kind signal --encoder mantis8m --device cuda --limit 64 --features-root /tmp/smoke
```

`exit` ends the session and frees the GPU. `--limit` encodes only a few segments and
`--features-root` keeps the trial out of the real caches.

## 17. Rules of the cluster

- **Never compute on the login nodes.** Installing, downloading, editing, moving files,
  submitting and `status.sh` are fine; tests, stores, features and training go into jobs.
- **Request what you use**: one GPU and 8 CPUs per GPU job; `cpu_short` for CPU work under
  an hour.
- **Prefer the A100s** (`gpua100`), as the Mésocentre asks for energy reasons.
- **Do not edit `~/.bashrc`, `~/.bash_profile` or `~/.bash_login`**, and do not run
  `conda init`.
- **Back up your results yourself** (§13).
- **Help:** <ruche.support@universite-paris-saclay.fr>, with your login, lab, the job ID,
  the problem, the log and whether support may read your files.

## 18. Cheat sheet

| Step | Command |
|---|---|
| Session start | `cd "$WORKDIR/RUL-Model" && source slurm/env.sh && mkdir -p runs/logs` |
| Status | `bash slurm/status.sh` |
| Weights (login node, once) | `bash slurm/prefetch_weights.sh` |
| Stage 3, all jobs | `bash slurm/stage3.sh` |
| J0 alone | `sbatch slurm/checks.sbatch` |
| J0g alone | `sbatch --job-name=J0g-gpu-check --partition=gpu_test --time=00:10:00 slurm/job.sbatch python -c "import torch; print(torch.cuda.get_device_name(0))"` |
| J2 alone | `sbatch --job-name=J2-features --time=02:00:00 slurm/job.sbatch bash slurm/features.sh` |
| J3 alone | `sbatch --job-name=J3-feature-check slurm/cpu.sbatch rulfuse feature-check` |
| Store details | `rulfuse store-info` |
| A long job in parts | `bash slurm/chain.sh N --job-name=NAME slurm/job.sbatch COMMAND …` |
| B6 full run + report | `bash slurm/omnifuser_rul.sh force tool` (the admitted photo types: tool, readmitted by D24) |
| B6 report again (test included from 2026-12-01) | `sbatch --job-name=b6-report slurm/cpu.sbatch bash slurm/omnifuser_results.sh force tool` |
| B6 report with the test before the freeze | the same with `--unseal` before the players (recorded in the report) |
| Stage 4: sanity checks (J4) + pilot (J1) | `bash slurm/stage4.sh` (§15.2) |
| Pilot alone | `bash slurm/study.sh pilot` |
| Sanity checks alone, some of them | `sbatch --job-name=J4-sanity --partition=gpua100 --time=06:00:00 slurm/job.sbatch python -m rulfuse.cli sanity --device cuda --checks 1 2` |
| Time per fit (sets J5-J7 limits) | `grep '"seconds"' "$RULFUSE_DATA_ROOT"/runs/pilot/F1-default/fold-T1/seed-*/selection.json` |
| Gate 2a (J5), then read `runs/gate2a/REPORT.md` | `bash slurm/study.sh gate2a` |
| Gate 2b (J6), after Gate 2a's decision | `git pull` (only now: Gate 2a ran the earlier code), then `bash slurm/study.sh gate2b` |
| Sweep (J7), after Gate 2b's decision | `bash slurm/study.sh sweep` |
| Finish an interrupted study | the same `bash slurm/study.sh STUDY` (skips `DONE` fits, resumes the rest) |
| Aggregate a study alone | `sbatch slurm/cpu.sbatch python -m rulfuse.cli aggregate --study STUDY` |
| Write a manifest only | `python -m rulfuse.cli manifest --study STUDY` (login node, seconds) |
| One task by hand | `sbatch --array=N slurm/array.sbatch STUDY` |
| Copy home | §13 |
