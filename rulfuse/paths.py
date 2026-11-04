"""Where data and outputs live (PROJECT_PLAN.md §14.1).

Everything generated sits under ``RULFUSE_DATA_ROOT`` (default ``D:\\RULModel``), outside
the OneDrive-synced repository: the raw copy, the stores, the feature caches and the runs.
Paths are functions, not constants, so a changed environment variable takes effect.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(r"D:\RULModel")
REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("RULFUSE_DATA_ROOT", str(DEFAULT_DATA_ROOT)))


def raw_dir(dataset: str = "nonastreda") -> Path:
    return data_root() / "raw" / dataset


def interim_dir() -> Path:
    return data_root() / "interim"


def runs_dir() -> Path:
    return data_root() / "runs"


def force_store_dir() -> Path:
    return interim_dir() / "forces"
