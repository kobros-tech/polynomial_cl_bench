# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Break the Phase 3 modules in specific ways and check that the Phase 3 tests notice.

    python3 tools/mutation_check_phase3.py

Each mutation is applied to a temporary copy of the repository; the Phase 3 test modules are
run against it. A mutation is 'caught' when those tests fail. Needs no core (the integration
module is skipped when the core is absent, so mutations that only it could catch are reported
as missed in that case).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests.test_runner",
    "tests.test_results_and_aggregation",
    "tests.test_runner_integration",
]

MUTATIONS = [
    (
        "aggregate.py",
        "std = np.nanstd(stack, axis=0, ddof=1)",
        "std = np.nanstd(stack, axis=0, ddof=0)",
        "elementwise std ddof",
    ),
    (
        "aggregate.py",
        "std = float(v.std(ddof=1))",
        "std = float(v.std(ddof=0))",
        "summary std ddof",
    ),
    (
        "aggregate.py",
        'if s in by_seed and by_seed[s]["status"] != "completed"',
        "if False",
        "failures dropped from aggregation",
    ),
    (
        "aggregate.py",
        "missing = [s for s in expected if s not in by_seed]",
        "missing = []",
        "missing runs not counted",
    ),
    (
        "aggregate.py",
        "diffs = [a_ok[s][",
        "diffs = [b_ok[s][",
        "paired diff sign/ordering",
    ),
    (
        "aggregate.py",
        "return _T95[df - 1]",
        "return _T95[df - 2]",
        "t critical off by one",
    ),
    (
        "results.py",
        'if expected_key != record["run_key"]:',
        "if False:",
        "run_key not verified",
    ),
    (
        "results.py",
        "elif any(u > b for u, b in zip(used, budget)):",
        "elif False:",
        "memory budget not enforced",
    ),
    (
        "results.py",
        "if not _close(m[key], recomputed[key]):",
        "if False:",
        "metrics not re-derived",
    ),
    (
        "runner.py",
        'if existing.get("fingerprint") != spec.fingerprint():',
        "if False:",
        "resume mixes experiments",
    ),
    (
        "runner.py",
        'if existing["run_key"] != expected_key(spec, entry, seed):',
        "if False:",
        "stale run key reused",
    ),
    (
        "runner.py",
        'if _provenance_signature(existing["environment"]) != here:',
        "if False:",
        "commits mixed on resume",
    ),
    ("runner.py", "if not retry_failed:", "if False:", "failed rerun without request"),
    (
        "runner.py",
        "path.rename(path.with_name(",
        "path.with_name(",
        "failed attempt history lost",
    ),
    (
        "runner.py",
        'if plan_fingerprint is not None and config.get("plan_fingerprint") != plan_fingerprint:',
        "if False:",
        "plan use not verified",
    ),
    (
        "runner.py",
        "enforce_run_mode(spec.mode, core_provenance(), benchmark_provenance())  # before any work",
        "pass",
        "official mode not enforced",
    ),
    (
        "runner.py",
        "if done.returncode == 3:",
        "if False:",
        "worker provenance refusal swallowed",
    ),
    (
        "runner.py",
        "except subprocess.TimeoutExpired as error:",
        "except ZeroDivisionError as error:",
        "timeout not handled",
    ),
    (
        "validation.py",
        "if not _single(plans.values()):",
        "if False:",
        "plan equality across algorithms unchecked",
    ),
    ("validation.py", "if len(commits) > 1:", "if False:", "mixed commits unchecked"),
    (
        "validation.py",
        'if cross is not None and not cross["agrees"]:',
        "if False:",
        "core cross-check ignored",
    ),
    (
        "summarize.py",
        "except results.RecordError as error:\n            invalid.append((path.name, str(error)))",
        "except results.RecordError as error:\n            pass",
        "invalid records silently skipped",
    ),
    (
        "summarize.py",
        'if ".attempt" in path.name:',
        "if False:",
        "archived attempts counted as runs",
    ),
]


def main() -> int:
    missed = []
    for filename, old, new, name in MUTATIONS:
        if name is None:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "repo"
            shutil.copytree(
                ROOT,
                copy,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "results", "data"),
            )
            target = copy / "src" / "polynomial_cl_bench" / filename
            text = target.read_text()
            if old not in text:
                print(f"STALE   {name}: pattern not found in {filename}")
                missed.append(name)
                continue
            target.write_text(text.replace(old, new, 1))
            env = {
                **os.environ,
                "PYTHONPATH": f"{copy / 'src'}:{copy / 'tests'}"
                + (":" + os.environ["STUBCORE"] if os.environ.get("STUBCORE") else ""),
            }
            run = subprocess.run(
                [sys.executable, "-m", "unittest", *TESTS],
                cwd=copy,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            caught = run.returncode != 0
            print(("caught  " if caught else "MISSED  ") + name)
            if not caught:
                missed.append(name)
    print(f"\n{len(MUTATIONS) - 1 - len(missed)} caught, {len(missed)} missed")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
