# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""tools/build_benchmark_results.py: numbers come from the files, nothing is invented."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import build_benchmark_results as tool  # noqa: E402


def fake_results(root: Path, accuracy=0.8):
    cmp_dir = root / "cmp"
    (cmp_dir / "runs").mkdir(parents=True)
    metrics = {
        "final_accuracy": {"mean": accuracy, "std": 0.01, "ci95": [0.7, 0.9]},
        "mean_forgetting": {"mean": 0.1, "std": 0.01, "ci95": [0.05, 0.15]},
    }
    summary = {
        "protocol": {"name": "p", "version": "1.0", "evaluation_split": "test"},
        "seeds": [0, 1],
        "aggregate": {
            "algorithms": {"a": {"metrics": metrics}},
            "paired_comparisons": [],
        },
        "reproducibility": [],
        "validation": {"publishable": False, "publication_blockers": ["dev mode"]},
    }
    env = {
        "run_mode": "development",
        "python_version": "3",
        "numpy_version": "2",
        "cpu_count": 1,
        "thread_env": {"OMP_NUM_THREADS": "1"},
    }
    (cmp_dir / "summary.json").write_text(json.dumps(summary))
    (cmp_dir / "manifest.json").write_text(json.dumps({"environment": env}))
    for seed in (0, 1):
        record = {
            "status": "completed",
            "algorithm": {"label": "a"},
            "resources": {
                "wall_time_seconds": 2.0,
                "peak_memory_mb": 10.0,
                "n_parameters": 5,
            },
        }
        (cmp_dir / "runs" / f"a__seed{seed}.json").write_text(json.dumps(record))
    return cmp_dir


class BenchmarkResultsTool(unittest.TestCase):
    def test_missing_input_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.rst"
            code = tool.main(
                ["--results", d, "--comparison", d + "/none", "--out", str(out)]
            )
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())

    def test_formatting_helpers_never_turn_null_into_zero(self):
        self.assertEqual(tool.pct(None), "n/a")
        self.assertEqual(tool.cell(None, None), "n/a")
        self.assertEqual(tool.mean_std([None, None]), (None, None))
        self.assertEqual(tool.mean_std([None, 2.0]), (2.0, None))

    def test_grid_edge_detection(self):
        tuning = {"selected_config": {"lr": 1.0}, "grid": [{"lr": 0.1}, {"lr": 1.0}]}
        self.assertIn("largest", tool.grid_edges({}, tuning)[0])
        inner = copy.deepcopy(tuning)
        inner["selected_config"] = {"lr": 0.5}
        inner["grid"] = [{"lr": 0.1}, {"lr": 0.5}, {"lr": 1.0}]
        self.assertEqual(tool.grid_edges({}, inner), [])

    def test_completed_runs_skips_failures(self):
        with tempfile.TemporaryDirectory() as d:
            cmp_dir = fake_results(Path(d))
            bad = json.loads((cmp_dir / "runs" / "a__seed1.json").read_text())
            bad["status"] = "failed"
            (cmp_dir / "runs" / "a__seed1.json").write_text(json.dumps(bad))
            self.assertEqual(len(tool.completed_runs(cmp_dir)["a"]), 1)

    def test_sha256_matches_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x"
            f.write_bytes(b"abc")
            self.assertEqual(tool.sha256(f), hashlib.sha256(b"abc").hexdigest())


if __name__ == "__main__":
    unittest.main()
