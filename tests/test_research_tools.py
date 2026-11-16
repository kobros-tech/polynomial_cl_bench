# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The two exploratory study tools: ``run_research_questions`` and ``run_degree_control_study``.

They run end to end on tiny archives of the right format (the accuracies are meaningless; what
is checked is the plumbing): output files, resumability, refusal to overwrite, selection-seed
separation, the no-feasible-candidate path and the clear refusal on a core without degree
control. The degree-control tool is only imported where the installed core provides the feature.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from helpers import MNIST_FILES, shipped_protocol, write_idx_images, write_idx_labels

from polynomial_cl_bench.algorithms.polynomial import CORE_HAS_DEGREE_CONTROL

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
needs_control_core = unittest.skipUnless(
    CORE_HAS_DEGREE_CONTROL, "installed core has no degree control"
)


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = module.main(argv)
    return code, out.getvalue()


class Workspace:
    """Tiny archives, a protocol file and a two-entry grid, in a temporary directory."""

    def __init__(self, **protocol_changes):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.protocol = self._protocol(**protocol_changes)
        self.protocol_file = self.dir / "protocol.json"
        self.protocol_file.write_text(json.dumps(self.protocol.to_dict()))
        grid = json.loads((ROOT / "configs" / "polynomial_grid.json").read_text())
        grid["grid"] = grid["grid"][:2]
        grid["selection"]["expected_runs"] = 2 * len(grid["selection"]["seeds"])
        self.grid_file = self.dir / "grid.json"
        self.grid_file.write_text(json.dumps(grid))

    def _protocol(self, **changes):
        """Tiny archives of the right format, large enough for a 10% validation hold-out."""
        rng = np.random.default_rng(0)
        write_idx_images(self.dir / MNIST_FILES[0], rng.integers(0, 256, (1500, 784)))
        write_idx_labels(self.dir / MNIST_FILES[1], np.arange(1500) % 10)
        write_idx_images(self.dir / MNIST_FILES[2], rng.integers(0, 256, (300, 784)))
        write_idx_labels(self.dir / MNIST_FILES[3], np.arange(300) % 10)
        hashes = {
            n: hashlib.sha256((self.dir / n).read_bytes()).hexdigest()
            for n in MNIST_FILES
        }
        base = shipped_protocol()
        return replace(
            base,
            dataset={**base.dataset, "files": hashes},
            **{
                "epochs": 1,
                "batch_size": 32,
                "memory_per_class": 5,
                "seeds": (0, 1),
                **changes,
            },
        )

    def cleanup(self):
        self._tmp.cleanup()


class ResearchQuestionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_tool("run_research_questions")

    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)
        # Two rows instead of five keeps the test fast; the logic is the same.
        patcher = mock.patch.object(
            self.module,
            "ROWS",
            (("expanding", "ordinary"), ("fixed_capacity", "ordinary")),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def argv(self, name, *extra):
        return [
            "--data-dir",
            str(self.ws.dir),
            "--protocol",
            str(self.ws.protocol_file),
            "--grid",
            str(self.ws.grid_file),
            "--out",
            str(self.ws.dir / name),
            *extra,
        ]

    def test_runs_end_to_end_and_writes_both_reports(self):
        code, text = run_main(self.module, self.argv("rq.json"))
        self.assertEqual(code, 0)
        result = json.loads((self.ws.dir / "rq.json").read_text())
        self.assertTrue(result["exploratory"])
        self.assertEqual(result["run_mode"], "development")
        # 2 rows x 2 sampling policies; each tuned on the same grid and run on every test seed.
        self.assertEqual(len(result["cells"]), 4)
        for cell in result["cells"].values():
            self.assertEqual(cell["validation_runs"], 4)
            self.assertEqual(len(cell["test_records"]), len(self.ws.protocol.seeds))
        self.assertTrue((self.ws.dir / "rq.md").exists())
        self.assertIn("RQ2-RQ4", text)

    def test_sampling_is_the_only_protocol_difference_between_policies(self):
        run_main(self.module, self.argv("rq.json"))
        result = json.loads((self.ws.dir / "rq.json").read_text())
        names = set(result["cells"])
        self.assertIn("expanding / ordinary / class_balanced", names)
        self.assertIn("expanding / ordinary / shuffled", names)
        pairs = {(c["a"], c["b"]) for c in result["paired"]}
        self.assertIn(
            (
                "expanding / ordinary / shuffled",
                "expanding / ordinary / class_balanced",
            ),
            pairs,
        )

    def test_refuses_to_overwrite(self):
        (self.ws.dir / "rq.json").write_text("{}")
        code, _ = run_main(self.module, self.argv("rq.json"))
        self.assertEqual(code, 2)
        self.assertEqual((self.ws.dir / "rq.json").read_text(), "{}")

    def test_selection_seeds_must_not_overlap_evaluation_seeds(self):
        ws = Workspace(seeds=(100, 5))
        self.addCleanup(ws.cleanup)
        argv = [
            "--data-dir",
            str(ws.dir),
            "--protocol",
            str(ws.protocol_file),
            "--grid",
            str(ws.grid_file),
            "--out",
            str(ws.dir / "x.json"),
        ]
        with (
            self.assertRaises(SystemExit) as raised,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.module.main(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_a_stopped_run_resumes_from_the_cache_with_identical_results(self):
        cache = self.ws.dir / "cache"
        # A budget that is already spent stops before the first cell and writes no result.
        code, _ = run_main(
            self.module,
            self.argv("a.json", "--cache-dir", str(cache), "--max-seconds", "1e-9"),
        )
        self.assertEqual(code, 3)
        self.assertFalse((self.ws.dir / "a.json").exists())
        # Finish in two stages: one cell cached, then the rest.
        with mock.patch.object(self.module.time, "perf_counter") as clock:
            ticks = iter([0.0, 0.0, 10.0] + [10.0] * 1000)
            clock.side_effect = lambda: next(ticks)
            code, _ = run_main(
                self.module,
                self.argv("a.json", "--cache-dir", str(cache), "--max-seconds", "5"),
            )
        self.assertEqual(code, 3)
        self.assertEqual(len(list(cache.glob("*.json"))), 1)
        code, text = run_main(
            self.module, self.argv("a.json", "--cache-dir", str(cache))
        )
        self.assertEqual(code, 0)
        self.assertIn("cached:", text)
        code, _ = run_main(self.module, self.argv("b.json"))  # uncached reference run
        self.assertEqual(code, 0)
        a = json.loads((self.ws.dir / "a.json").read_text())
        b = json.loads((self.ws.dir / "b.json").read_text())
        for name in a["cells"]:
            self.assertEqual(
                a["cells"][name]["test_records"], b["cells"][name]["test_records"]
            )
            self.assertEqual(
                a["cells"][name]["selected_config"], b["cells"][name]["selected_config"]
            )


class NoDegreeControlCoreTests(unittest.TestCase):
    @unittest.skipIf(CORE_HAS_DEGREE_CONTROL, "installed core has degree control")
    def test_the_degree_control_tool_refuses_cleanly_without_core_support(self):
        done = subprocess.run(
            [sys.executable, str(TOOLS / "run_degree_control_study.py"), "--out", "x"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no degree control", done.stderr)


@needs_control_core
class DegreeControlStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_tool("run_degree_control_study")

    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)

    def argv(self, name, *extra):
        return [
            "--data-dir",
            str(self.ws.dir),
            "--protocol",
            str(self.ws.protocol_file),
            "--out",
            str(self.ws.dir / name),
            "--strategies",
            "ordinary",
            *extra,
        ]

    def test_selection_metrics_need_every_selection_seed(self):
        ok = {
            "status": "completed",
            "arrival_accuracy": [0.9, 0.6],
            "metrics": {
                "final_accuracy": 0.7,
                "mean_forgetting": 0.1,
                "newest_accuracy": 0.6,
            },
        }
        other = {**ok, "arrival_accuracy": [0.8, 0.4]}
        m = self.module.selection_metrics([ok, other])
        self.assertAlmostEqual(
            m["min_mean_immediate_accuracy"], 0.5
        )  # mean of 0.6 and 0.4
        for got, want in zip(m["immediate_accuracy_by_experience"], [0.85, 0.5]):
            self.assertAlmostEqual(got, want)
        self.assertIsNone(
            self.module.selection_metrics([ok, {"status": "failed", "seed": 1}])
        )
        self.assertIsNone(self.module.selection_metrics([]))

    def test_wrong_tuning_result_is_refused(self):
        path = self.ws.dir / "t.json"
        path.write_text(json.dumps({"algorithm": "er_mlp", "selected_config": {}}))
        with self.assertRaises(SystemExit):
            self.module.selected_hyperparameters(path, "polynomial_cl")
        self.assertIsNone(self.module.selected_hyperparameters(None, "er_mlp"))

    def test_with_an_unreachable_floor_nothing_is_selected_and_the_baseline_still_runs(
        self,
    ):
        code, _ = run_main(self.module, self.argv("dc.json", "--floor", "1.0"))
        self.assertEqual(code, 0)
        result = json.loads((self.ws.dir / "dc.json").read_text())
        sel = result["selection"]["ordinary"]
        self.assertIsNone(sel["selected"]["best_overall"])
        self.assertIn("best_overall", sel["no_feasible_candidate"])
        self.assertEqual(list(result["test"]), ["ordinary: baseline"])  # no control run

    def test_with_a_floor_of_zero_a_control_is_selected_and_compared_paired(self):
        code, _ = run_main(self.module, self.argv("dc.json", "--floor", "0.0"))
        self.assertEqual(code, 0)
        result = json.loads((self.ws.dir / "dc.json").read_text())
        self.assertEqual(
            list(result["test"]), ["ordinary: baseline", "ordinary: degree control"]
        )
        comparison = result["paired"][0]
        self.assertEqual(comparison["a"], "ordinary: degree control")
        self.assertEqual(
            comparison["metrics"]["final_accuracy"]["paired_seeds"],
            list(self.ws.protocol.seeds),
        )
        self.assertTrue(result["exploratory"])
        self.assertEqual(result["run_mode"], "development")
        # Selection never saw the evaluation seeds.
        self.assertFalse(set(result["selection_seeds"]) & set(result["test_seeds"]))

    def test_refuses_to_overwrite_and_overlapping_seeds(self):
        (self.ws.dir / "dc.json").write_text("{}")
        code, _ = run_main(self.module, self.argv("dc.json"))
        self.assertEqual(code, 2)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.module.main(self.argv("other.json", "--select-seeds", "0", "1"))


if __name__ == "__main__":
    unittest.main()
