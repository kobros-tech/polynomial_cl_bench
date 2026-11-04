# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Resource measurement that is identical for every algorithm, plus environment capture.

What is measured, and what each number means
--------------------------------------------
* ``wall_time_seconds`` / ``cpu_time_seconds``: measured by the *runner* around the adapter's
  ``run_sequence`` call, the same way for every algorithm. This is the only cost figure that
  is comparable across algorithms. It includes the algorithm's in-loop evaluation and any
  feature construction done inside the call. CPU time is the process's user+system time (all
  threads), so it is less sensitive to a busy machine than wall time but is not a substitute
  for it.
* ``peak_memory_mb``: peak resident set size of the process *during* the run. It is only
  reported when the peak counter could be reset immediately before the run (Linux, via
  ``/proc/self/clear_refs``); otherwise it is ``None``. It is an absolute RSS, so it includes
  whatever the process already held (the dataset, the interpreter); ``rss_before_run_mb``
  is recorded next to it so the increase can be judged. Process isolation (one fresh process
  per run) keeps that baseline identical across algorithms.
* ``process_peak_rss_mb``: the whole-process high-water mark (includes dataset loading).
  Descriptive only.

Training and inference time are adapter-reported (``None`` when an adapter cannot separate
them, as for the polynomial core) and are kept apart from the runner's figures.

Unavailable measurements are ``None``, never zero.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

try:  # not available on Windows
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None

_PROC_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _proc_status_kb(key: str) -> float | None:
    try:
        for line in _PROC_STATUS.read_text().splitlines():
            if line.startswith(key + ":"):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def current_rss_mb() -> float | None:
    """Resident set size now, in MiB (Linux); ``None`` where it cannot be read."""
    kb = _proc_status_kb("VmRSS")
    return None if kb is None else kb / 1024.0


def process_peak_rss_mb() -> float | None:
    """Whole-process peak RSS in MiB since start or since the last successful reset."""
    kb = _proc_status_kb("VmHWM")
    if kb is not None:
        return kb / 1024.0
    if _resource is None:  # pragma: no cover - platform dependent
        return None
    peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def reset_peak_rss() -> bool:
    """Reset the peak-RSS counter to the current RSS. ``False`` if the platform cannot."""
    try:
        _CLEAR_REFS.write_text("5")
    except OSError:
        return False
    return True


@dataclass
class RunMeter:
    """Measure one run: wall time, CPU time, and peak RSS during the run.

    ``with RunMeter() as meter: ...; meter.report()``. Use ``fresh process`` isolation for a
    clean memory baseline; see the module docstring.
    """

    peak_reset: bool = False
    rss_before_run_mb: float | None = None
    wall_time_seconds: float | None = None
    cpu_time_seconds: float | None = None
    peak_memory_mb: float | None = None
    process_peak_rss_mb: float | None = None
    _wall0: float = field(default=0.0, repr=False)
    _cpu0: float = field(default=0.0, repr=False)

    def __enter__(self) -> Self:
        self.peak_reset = reset_peak_rss()
        self.rss_before_run_mb = current_rss_mb()
        self._wall0, self._cpu0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc) -> bool:
        self.wall_time_seconds = time.perf_counter() - self._wall0
        self.cpu_time_seconds = time.process_time() - self._cpu0
        peak = process_peak_rss_mb()
        self.process_peak_rss_mb = peak
        self.peak_memory_mb = peak if self.peak_reset else None
        return False  # never swallow the run's exception

    def report(self) -> dict:
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_memory_scope": "process_rss_during_run"
            if self.peak_memory_mb is not None
            else None,
            "rss_before_run_mb": self.rss_before_run_mb,
            "process_peak_rss_mb": self.process_peak_rss_mb,
        }


def environment_info() -> dict:
    """Execution environment recorded with every result (no commits: see provenance.py)."""
    import numpy

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "numpy_version": numpy.__version__,
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }
