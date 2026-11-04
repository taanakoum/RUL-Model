"""CPU and memory checks before heavy local work (PROJECT_PLAN.md §16.1, lesson L13).

``idle_cpus`` and ``cpu_threads`` are vendored from OmniFuser (``omnifuser/resources.py``,
commit e3b9985); they measure idle CPUs without WMI. ``available_memory_mb`` is new. The
laptop is shared with other jobs, so heavy steps call ``check_resources`` first.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass


def available_cpus() -> int:
    """CPUs this process may run on."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


def _windows_idle_fraction(seconds: float) -> float | None:
    import ctypes
    from ctypes import wintypes

    def sample() -> tuple[int, int] | None:
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(  # pyright: ignore[reportAttributeAccessIssue]
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None

        def ticks(value: wintypes.FILETIME) -> int:
            return (value.dwHighDateTime << 32) | value.dwLowDateTime

        return ticks(idle), ticks(kernel) + ticks(user)  # kernel time includes idle time

    first = sample()
    time.sleep(seconds)
    second = sample()
    if first is None or second is None or second[1] <= first[1]:
        return None
    return (second[0] - first[0]) / (second[1] - first[1])


def idle_cpus(sample_seconds: float = 0.25) -> float | None:
    """CPUs currently idle across the machine, or None when it cannot be measured."""
    total = os.cpu_count() or 1
    try:
        if sys.platform == "win32":
            fraction = _windows_idle_fraction(sample_seconds)
            return None if fraction is None else max(0.0, min(1.0, fraction)) * total
        load = os.getloadavg()[0]
    except (AttributeError, OSError, ValueError):
        return None
    return max(0.0, total - load)


def cpu_threads(limit: int = 4) -> int:
    """Threads for CPU work now: the smallest of the limit, the CPUs available and idle."""
    requested = os.environ.get("RULFUSE_CPU_THREADS", "")
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    threads = min(limit, available_cpus())
    omp = os.environ.get("OMP_NUM_THREADS", "")
    if omp.isdigit() and int(omp) > 0:
        threads = min(threads, int(omp))
    if "SLURM_JOB_ID" not in os.environ:
        idle = idle_cpus()
        if idle is not None:
            threads = min(threads, math.floor(idle))
    return max(1, threads)


def available_memory_mb() -> float | None:
    """Physical memory currently available, in MB (None when it cannot be measured)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # pyright: ignore[reportAttributeAccessIssue]
                ctypes.byref(status)
            ):
                return None
            return status.ullAvailPhys / 2**20
        with open("/proc/meminfo", encoding="ascii") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, AttributeError):
        return None
    return None


@dataclass(frozen=True)
class ResourceStatus:
    idle_cpus: float | None
    available_memory_mb: float | None
    threads: int

    def describe(self) -> str:
        idle = "unknown" if self.idle_cpus is None else f"{self.idle_cpus:.1f}"
        memory = (
            "unknown" if self.available_memory_mb is None else f"{self.available_memory_mb:,.0f} MB"
        )
        return f"idle CPUs {idle} of {os.cpu_count()}, free memory {memory}, threads {self.threads}"


class ResourcesBusy(RuntimeError):
    """Raised when the machine is too busy for a heavy step."""


def check_resources(
    min_memory_mb: float = 0.0, min_idle_cpus: float = 0.0, limit: int = 4
) -> ResourceStatus:
    """Measure the machine; raise ``ResourcesBusy`` if it is below the given minimums."""
    status = ResourceStatus(idle_cpus(), available_memory_mb(), cpu_threads(limit))
    if status.available_memory_mb is not None and status.available_memory_mb < min_memory_mb:
        raise ResourcesBusy(f"needs {min_memory_mb:,.0f} MB free; {status.describe()}")
    if status.idle_cpus is not None and status.idle_cpus < min_idle_cpus:
        raise ResourcesBusy(f"needs {min_idle_cpus:.1f} idle CPUs; {status.describe()}")
    return status
