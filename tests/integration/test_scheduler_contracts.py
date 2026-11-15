"""Exercise job dependencies and packed tails with shell stubs; never submit a real job."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rulfuse import paths

pytestmark = [
    pytest.mark.integration,
    pytest.mark.system,
    pytest.mark.requirements("P14", "P20", "D29", "D31"),
]


@pytest.fixture
def shell_sandbox(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.is_file() else None
    if bash is None:
        pytest.skip("Bash is required for scheduler contract checks")
    root = tmp_path / "repository"
    (root / "slurm").mkdir(parents=True)
    for name in ("study.sh", "array.sbatch"):
        shutil.copyfile(paths.REPO_ROOT / "slurm" / name, root / "slurm" / name)
    (root / "slurm/exclude.sh").write_text("#!/bin/bash\nexit 0\n")
    # All cluster setup is replaced. The scheduler and Python below only log arguments.
    (root / "slurm/env.sh").write_text('export RULFUSE_DATA_ROOT="$TEST_DATA"\n')
    bin_dir = root / "bin"
    bin_dir.mkdir()
    scripts = {
        "python": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$PYTHON_LOG"\nexit "${PYTHON_STATUS:-0}"\n',
        "sbatch": (
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SBATCH_LOG"\n'
            'if [ "${FAIL_SBATCH:-0}" = 1 ]; then exit 9; fi\n'
            'echo "$((100 + $(wc -l < "$SBATCH_LOG")))"\n'
        ),
    }
    for name, source in scripts.items():
        target = bin_dir / name
        target.write_text(source, encoding="utf-8", newline="\n")
        target.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "TEST_DATA": (root / "data").as_posix(),
        "RULFUSE_DATA_ROOT": (root / "data").as_posix(),
        "SBATCH_LOG": (root / "sbatch.log").as_posix(),
        "PYTHON_LOG": (root / "python.log").as_posix(),
    }
    return bash, root, env


def manifest(root, outer, inner, seeds="0"):
    target = root / "data/runs/example/manifest.tsv"
    target.parent.mkdir(parents=True)
    lines = ["task\tarm\tfold_test\tseeds\tinner"]
    lines += [f"{i}\ta\tT{i}\t{seeds}\t-1" for i in range(outer)]
    lines += [f"{i + outer}\ta\tT{i}\t{seeds}\t0" for i in range(inner)]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    "outer,inner,seeds,expected",
    [
        (20, 0, "0", ("0-6%4",)),
        (10, 40, "0", ("0-3%4", "0-13%4")),
        pytest.param(2, 3, "0,1,2", ("0-1%4", "0-2%4"), marks=pytest.mark.compatibility),
    ],
)
def test_submission_packs_all_lines_and_waits_for_outer_then_inner(
    shell_sandbox,
    outer,
    inner,
    seeds,
    expected,
):
    bash, root, env = shell_sandbox
    manifest(root, outer, inner, seeds)
    result = subprocess.run(
        [bash, "slurm/study.sh", "example"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    calls = (root / "sbatch.log").read_text().splitlines()
    assert len(calls) == len(expected) + 1
    for call, array in zip(calls, expected, strict=False):
        assert f"--array={array}" in call
    if inner:
        assert "--dependency=afterok:101" in calls[1]
        assert "--dependency=afterok:101:102" in calls[-1]
    else:
        assert "--dependency=afterok:101" in calls[-1]


def test_failed_submission_cannot_start_aggregation(shell_sandbox):
    bash, root, env = shell_sandbox
    manifest(root, 2, 0)
    result = subprocess.run(
        [bash, "slurm/study.sh", "example"],
        cwd=root,
        env={**env, "FAIL_SBATCH": "1"},
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    calls = (root / "sbatch.log").read_text().splitlines()
    assert len(calls) == 1
    assert "aggregate" not in calls[0]


def test_unneeded_conditional_study_submits_no_jobs(shell_sandbox):
    bash, root, env = shell_sandbox
    result = subprocess.run(
        [bash, "slurm/study.sh", "example"],
        cwd=root,
        env={**env, "PYTHON_STATUS": "3"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "nothing submitted" in result.stdout
    assert not (root / "sbatch.log").exists()


@pytest.mark.parametrize("index,expected", [(0, [10, 11, 12]), (13, [49])])
def test_array_tail_runs_exactly_the_requested_manifest_lines(shell_sandbox, index, expected):
    bash, root, env = shell_sandbox
    job_env = {
        **env,
        "SLURM_JOB_ID": "123",
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": str(index),
        "SLURM_JOB_NAME": "example",
        "SLURM_SUBMIT_DIR": root.as_posix(),
    }
    result = subprocess.run(
        [bash, "slurm/array.sbatch", "example", "3", "10", "49"],
        cwd=root,
        env=job_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    calls = (root / "python.log").read_text().splitlines()
    command = next(line for line in calls if "run-task" in line).split()
    actual = command[command.index("--task") + 1 : command.index("--device")]
    assert actual == list(map(str, expected))
    assert "--parallel-tasks" in command and "--parallel-seeds" in command


@pytest.mark.parametrize(
    "script",
    sorted(p.name for p in (paths.REPO_ROOT / "slurm").iterdir() if p.suffix in {".sh", ".sbatch"}),
)
def test_all_cluster_shell_scripts_parse(shell_sandbox, script):
    bash, _, _ = shell_sandbox
    result = subprocess.run(
        [bash, "-n", (paths.REPO_ROOT / "slurm" / script).as_posix()],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
