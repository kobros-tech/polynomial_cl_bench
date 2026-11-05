# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Approved grids, the approval gate, the tuning runner and the MNIST file verifier."""

import hashlib
import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from helpers import small_protocol, synthetic_dataset

from polynomial_cl_bench import tune, verify_data
from polynomial_cl_bench.algorithms.base import AdapterError
from polynomial_cl_bench.algorithms.polynomial import PolynomialAdapter
from polynomial_cl_bench.protocol import Protocol
from polynomial_cl_bench.provenance import ProvenanceError

ROOT = Path(__file__).resolve().parents[1]
ER_GRID = ROOT / "configs" / "er_mlp_grid.json"
POLY_GRID = ROOT / "configs" / "polynomial_grid.json"
PROTOCOL = ROOT / "configs" / "split_mnist_cil_v1.json"
REFERENCE = ROOT / "configs" / "polynomial_expanding_ordinary.json"


def approved_copy(directory, source, status="approved"):
    raw = json.loads(Path(source).read_text())
    raw["approval_status"] = status
    path = Path(directory) / Path(source).name
    path.write_text(json.dumps(raw))
    return path


class ApprovedER_MLPGrid(unittest.TestCase):
    def test_grid_is_exactly_the_approved_design(self):
        spec = tune.load_grid(ER_GRID)
        self.assertEqual(spec.approval_status, "approved")
        self.assertEqual(spec.algorithm, "er_mlp")
        self.assertEqual(len(spec.entries), 15)
        self.assertEqual(
            [(e["hidden"], e["learning_rate"]) for e in spec.entries],
            [
                ([h], lr)
                for h, lr in itertools.product(
                    (64, 128, 256), (0.01, 0.03, 0.1, 0.3, 1.0)
                )
            ],
        )
        self.assertEqual(
            spec.fixed,
            {"replay": True, "optimizer": "sgd", "momentum": 0, "weight_decay": 0},
        )
        self.assertEqual(
            (spec.seeds, spec.expected_runs, spec.margin), ((100, 101), 30, 0.005)
        )

    def test_shipped_protocol_satisfies_the_approved_training_setup(self):
        protocol = Protocol.from_file(PROTOCOL)
        self.assertEqual(
            (protocol.memory_per_class, protocol.sampling), (200, "class_balanced")
        )
        self.assertEqual((protocol.epochs, protocol.batch_size), (3, 128))
        self.assertEqual(protocol.validation["holdout_fraction"], 0.1)
        self.assertTrue({100, 101}.isdisjoint(protocol.seeds))

    def test_fixed_settings_cannot_drift_from_the_approved_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads(ER_GRID.read_text())
            raw["fixed"]["momentum"] = 0.9
            path = Path(tmp) / "g.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(tune.GridError):
                tune.load_grid(path)

    def test_malformed_grids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = json.loads(ER_GRID.read_text())
            cases = {
                "extra key": lambda r: r["grid"][0].update(momentum=0.9),
                "duplicate": lambda r: r["grid"].append(dict(r["grid"][0])),
                "empty": lambda r: r.update(grid=[]),
                "unknown algorithm": lambda r: r.update(algorithm="ewc"),
            }
            for name, mutate in cases.items():
                with self.subTest(name):
                    raw = json.loads(json.dumps(base))
                    mutate(raw)
                    path = Path(tmp) / "g.json"
                    path.write_text(json.dumps(raw))
                    with self.assertRaises(tune.GridError):
                        tune.load_grid(path)


class ApprovedPolynomialGrid(unittest.TestCase):
    def test_grid_is_five_rates_by_three_feature_powers_with_the_same_budget(self):
        spec = tune.load_grid(POLY_GRID)
        self.assertEqual(spec.algorithm, "polynomial_cl")
        self.assertEqual(
            [(e["learning_rate"], e["feature_power"]) for e in spec.entries],
            [(lr, p) for p in (0.0, 0.5, 1.0) for lr in (0.01, 0.03, 0.1, 0.3, 1.0)],
        )
        self.assertEqual(spec.fixed, {"condition": "expanding", "strategy": "ordinary"})
        er = tune.load_grid(ER_GRID)
        self.assertEqual(
            (spec.seeds, spec.expected_runs, spec.margin),
            (er.seeds, er.expected_runs, er.margin),
        )
        self.assertEqual(len(spec.entries) * len(spec.seeds), spec.expected_runs)

    def test_grid_contains_the_untuned_reference_configuration(self):
        ref = json.loads(REFERENCE.read_text())
        spec = tune.load_grid(POLY_GRID)
        self.assertIn(
            {
                "learning_rate": ref["learning_rate"],
                "feature_power": ref["feature_power"],
            },
            spec.entries,
        )

    def test_grid_is_marked_approved(self):
        self.assertEqual(tune.load_grid(POLY_GRID).approval_status, "approved")

    def test_a_proposed_grid_is_refused_by_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = tune.load_grid(approved_copy(tmp, POLY_GRID, status="proposed"))
        self.assertEqual(spec.approval_status, "proposed")
        with self.assertRaisesRegex(tune.GridError, "only an approved grid"):
            tune.run_tuning(
                spec, small_protocol(memory_per_class=200), synthetic_dataset()
            )

    def test_command_line_refuses_a_proposed_grid_before_loading_any_data(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                tune,
                "load_split_mnist",
                side_effect=AssertionError("data must not be loaded"),
            ),
        ):
            grid = approved_copy(tmp, POLY_GRID, status="proposed")
            out = Path(tmp) / "r.json"
            self.assertEqual(tune.main(["--grid", str(grid), "--out", str(out)]), 2)
            self.assertFalse(out.exists())


class RunnerOnSyntheticData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset(train_per_class=30)
        cls.protocol = small_protocol(
            epochs=2, batch_size=32, memory_per_class=200, sampling="class_balanced"
        )

    def test_full_er_mlp_grid_runs_30_validation_runs_and_selects_one(self):
        result = tune.run_tuning(tune.load_grid(ER_GRID), self.protocol, self.dataset)
        self.assertEqual(
            (len(result.grid), len(result.runs), result.budget_runs), (15, 30, 30)
        )
        self.assertEqual(result.selection_seeds, (100, 101))
        self.assertEqual({r["evaluation_split"] for r in result.runs}, {"validation"})
        self.assertEqual(sum(c["selected"] for c in result.configs), 1)
        context = result.context
        self.assertEqual(context["run_mode"], "development")
        self.assertEqual(context["grid_approval_status"], "approved")
        self.assertIn("official_problems", context["core"])
        self.assertIn("commit", context["benchmark"])
        self.assertFalse(
            context["official_eligible"]
        )  # no verified remote in this environment
        json.dumps(result.to_dict(), allow_nan=False)

    def test_approved_polynomial_grid_runs_through_the_same_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = tune.load_grid(approved_copy(tmp, POLY_GRID))
        result = tune.run_tuning(spec, self.protocol, self.dataset)
        self.assertEqual((len(result.runs), result.budget_runs), (30, 30))
        self.assertEqual(result.algorithm, "polynomial_cl")
        self.assertEqual({r["evaluation_split"] for r in result.runs}, {"validation"})
        # per-feature step sizes can diverge at the top of the grid: that must be recorded, not hidden
        self.assertTrue(
            all(r["status"] in ("completed", "diverged", "failed") for r in result.runs)
        )
        for c in result.configs:
            self.assertEqual(
                c["eligible"], all(s == "completed" for s in c["statuses"])
            )

    def test_protocol_must_match_the_grids_requirements(self):
        spec = tune.load_grid(ER_GRID)
        with self.assertRaisesRegex(tune.GridError, "memory_per_class"):
            tune.run_tuning(
                spec, replace(self.protocol, memory_per_class=10), self.dataset
            )
        with self.assertRaisesRegex(tune.GridError, "sampling"):
            tune.run_tuning(
                spec, replace(self.protocol, sampling="shuffled"), self.dataset
            )

    def test_official_mode_is_refused_before_any_training_here(self):
        spec = tune.load_grid(ER_GRID)
        with (
            mock.patch.object(
                tune, "build_factory", side_effect=AssertionError("trained")
            ),
            self.assertRaises(ProvenanceError),
        ):
            tune.run_tuning(spec, self.protocol, self.dataset, mode="official")

    def test_official_mode_passes_the_gate_only_with_verified_provenance(self):
        ok = {"official_problems": []}
        spec = tune.load_grid(ER_GRID)
        with (
            mock.patch.object(tune, "core_provenance", lambda: ok),
            mock.patch.object(tune, "benchmark_provenance", lambda: ok),
            mock.patch.object(tune, "tune", lambda *a, **k: k["context"]),
        ):
            context = tune.run_tuning(
                spec, self.protocol, self.dataset, mode="official"
            )
        self.assertTrue(context["official_eligible"])
        self.assertEqual(context["run_mode"], "official")


class CommandLine(unittest.TestCase):
    def test_writes_json_and_markdown_and_never_overwrites(self):
        dataset = synthetic_dataset(train_per_class=30)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                tune, "load_split_mnist", lambda protocol, data_dir: dataset
            ),
        ):
            out = Path(tmp) / "res" / "tuning_er_mlp.json"
            argv = [
                "--grid",
                str(ER_GRID),
                "--protocol",
                str(PROTOCOL),
                "--out",
                str(out),
            ]
            self.assertEqual(tune.main(argv), 0)
            saved = json.loads(out.read_text())
            self.assertEqual(len(saved["runs"]), 30)
            self.assertEqual(saved["context"]["run_mode"], "development")
            table = out.with_suffix(".md").read_text()
            self.assertEqual(table.count("**selected**"), 1)
            self.assertEqual(
                sum(
                    1
                    for line in table.splitlines()
                    if line.startswith("| ") and line.split("|")[1].strip().isdigit()
                ),
                15,
            )
            before = out.read_text()
            self.assertEqual(
                tune.main(argv), 2
            )  # refuses to overwrite a complete result
            self.assertEqual(out.read_text(), before)

    def test_official_mode_writes_nothing_when_provenance_fails(self):
        dataset = synthetic_dataset(train_per_class=30)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                tune, "load_split_mnist", lambda protocol, data_dir: dataset
            ),
        ):
            out = Path(tmp) / "r.json"
            with self.assertRaises(ProvenanceError):
                tune.main(
                    ["--grid", str(ER_GRID), "--out", str(out), "--mode", "official"]
                )
            self.assertFalse(out.exists())


class ValidationTable(unittest.TestCase):
    def test_table_lists_every_configuration_and_every_failure(self):
        from test_plan_and_tuning import FakeAdapter

        from polynomial_cl_bench import tuning

        dataset, protocol = synthetic_dataset(train_per_class=40), small_protocol()
        specs = [FakeAdapter(0.7, 0.1), FakeAdapter(0.9, error=RuntimeError("boom"))]
        grid = [
            {"hidden": [64], "learning_rate": 0.1, "i": 0},
            {"hidden": [128], "learning_rate": 0.3, "i": 1},
        ]
        result = tuning.tune("er_mlp", lambda c: specs[c["i"]], grid, dataset, protocol)
        table = tune.validation_table(result)
        self.assertIn("H=64, lr=0.1", table)
        self.assertIn("H=128, lr=0.3", table)
        self.assertIn("**selected**", table)
        self.assertIn("RuntimeError: boom", table)
        self.assertIn("Failed or divergent runs:", table)
        self.assertEqual(
            table.count("config 1 (H=128, lr=0.3), seed"), 2
        )  # one line per seed


class FeaturePowerValidation(unittest.TestCase):
    def test_invalid_feature_power_is_refused_by_the_adapter(self):
        for bad in (-0.5, float("nan"), float("inf"), [0.0, 1.0]):
            with self.subTest(bad), self.assertRaises(AdapterError):
                PolynomialAdapter(feature_power=bad)
        PolynomialAdapter(feature_power=0.0)
        PolynomialAdapter(feature_power=0.5)


class VerifyData(unittest.TestCase):
    def protocol_for(self, directory, contents):
        files = {}
        for name, data in contents.items():
            (Path(directory) / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
        base = Protocol.from_file(PROTOCOL)
        return replace(base, dataset={**base.dataset, "files": files})

    def test_ok_missing_and_mismatch_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = self.protocol_for(
                tmp, {"a.gz": b"aaa", "b.gz": b"bbb", "c.gz": b"ccc"}
            )
            (Path(tmp) / "b.gz").write_bytes(b"corrupted")
            (Path(tmp) / "c.gz").unlink()
            status = {name: s for name, _, _, s in verify_data.check(protocol, tmp)}
        self.assertEqual(
            status, {"a.gz": "ok", "b.gz": "HASH MISMATCH", "c.gz": "missing"}
        )

    def test_shipped_protocol_names_the_four_standard_archives(self):
        files = Protocol.from_file(PROTOCOL).dataset["files"]
        self.assertEqual(
            sorted(files),
            [
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
            ],
        )
        self.assertTrue(all(len(h) == 64 for h in files.values()))

    def test_exit_code_is_nonzero_until_all_four_archives_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(verify_data.main(["--data-dir", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
