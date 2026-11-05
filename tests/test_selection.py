# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Model selection: four objectives, accuracy floors, missing values, validation-only."""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from polynomial_cl_bench import selection
from polynomial_cl_bench.selection import (
    Candidate,
    SelectionError,
    candidates_from_tuning,
    report_from_tuning,
    select,
    select_all,
)


def C(i, acc, forg, time=None, **kw):
    return Candidate(i, acc, forg, time, **kw)


# ---- the original logic of tools/analyze_per_degree_study.py, copied verbatim in behaviour
# before that script was refactored onto this module. Kept as the regression reference.
def _ref_normalise(values, lower_is_better):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    scaled = [(v - lo) / span for v in values]
    return [s if lower_is_better else 1.0 - s for s in scaled]


def _ref_pick(pool, criterion, weights=(1.0, 1.0, 1.0)):
    # pool: list of (acc, forg, time)
    if criterion == "max_accuracy":
        return min(
            range(len(pool)), key=lambda i: (-pool[i][0], pool[i][1], pool[i][2])
        )
    if criterion == "min_forgetting":
        return min(
            range(len(pool)), key=lambda i: (pool[i][1], -pool[i][0], pool[i][2])
        )
    if criterion == "min_time":
        return min(
            range(len(pool)), key=lambda i: (pool[i][2], -pool[i][0], pool[i][1])
        )
    acc, forg, tm = zip(*pool)
    bad = (
        _ref_normalise(acc, False),
        _ref_normalise(forg, True),
        _ref_normalise(tm, True),
    )
    scores = [
        math.sqrt(sum(w * bad[k][i] ** 2 for k, w in enumerate(weights)))
        for i in range(len(pool))
    ]
    return min(range(len(pool)), key=lambda i: (scores[i], -pool[i][0]))


def _ref_dominates(a, b):
    return (a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]) and (
        a[0] > b[0] or a[1] < b[1] or a[2] < b[2]
    )


class EquivalenceWithOriginalScript(unittest.TestCase):
    def test_random_pools_agree_for_every_objective_weighting_and_floor(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 12)
            # coarse values so exact ties occur
            rows = [
                (
                    round(rng.uniform(0.5, 0.9), 2),
                    round(rng.uniform(0.0, 0.3), 2),
                    round(rng.uniform(1, 20), 1),
                )
                for _ in range(n)
            ]
            cands = [C(i, *r) for i, r in enumerate(rows)]
            for crit in selection.OBJECTIVES:
                for w in selection.SENSITIVITY_WEIGHTS.values():
                    if crit != "best_tradeoff" and w != (1.0, 1.0, 1.0):
                        continue
                    self.assertEqual(
                        select(cands, crit, weights=w).candidate_id,
                        _ref_pick(rows, crit, w),
                        (rows, crit, w),
                    )
                if crit == "max_accuracy":
                    continue
                best = max(r[0] for r in rows)
                keep = [i for i, r in enumerate(rows) if r[0] >= best - 0.02]
                want = keep[_ref_pick([rows[i] for i in keep], crit)]
                self.assertEqual(
                    select(cands, crit, accuracy_floor=0.02).candidate_id, want
                )

    def test_pareto_front_agrees(self):
        rng = random.Random(3)
        for _ in range(100):
            rows = [
                (round(rng.random(), 1), round(rng.random(), 1), round(rng.random(), 1))
                for _ in range(rng.randint(1, 10))
            ]
            cands = [C(i, *r) for i, r in enumerate(rows)]
            want = [
                i
                for i, r in enumerate(rows)
                if not any(_ref_dominates(o, r) for o in rows if o is not r)
            ]
            self.assertEqual([c.id for c in selection.pareto_front(cands)], want)


class Objectives(unittest.TestCase):
    def setUp(self):
        self.cands = [
            C("a", 0.80, 0.10, 10.0),
            C("b", 0.78, 0.02, 12.0),  # least forgetting, within floor
            C("c", 0.70, 0.01, 1.0),  # fastest and most stable, but inaccurate
            C("d", 0.79, 0.05, 2.0),
        ]

    def test_unconstrained_winners(self):
        pick = lambda o, **k: select(self.cands, o, **k).candidate_id
        self.assertEqual(pick("max_accuracy"), "a")
        self.assertEqual(pick("min_forgetting"), "c")
        self.assertEqual(pick("min_time"), "c")

    def test_floor_removes_the_degenerate_winners(self):
        pick = lambda o: select(self.cands, o, accuracy_floor=0.02).candidate_id
        self.assertEqual(pick("min_forgetting"), "b")
        self.assertEqual(pick("min_time"), "d")
        s = select(self.cands, "min_time", accuracy_floor=0.02)
        self.assertAlmostEqual(s.floor_value, 0.78)
        self.assertEqual((s.n_before_floor, s.n_pool), (4, 3))

    def test_floor_boundary_is_inclusive_despite_float_rounding(self):
        cands = [C("top", 0.9, 0.2, 5.0), C("edge", 0.9 - 0.02, 0.0, 9.0)]
        self.assertEqual(
            select(cands, "min_forgetting", accuracy_floor=0.02).candidate_id, "edge"
        )

    def test_floor_not_allowed_for_max_accuracy(self):
        with self.assertRaises(SelectionError):
            select(self.cands, "max_accuracy", accuracy_floor=0.02)

    def test_invalid_requests(self):
        for kwargs in (
            {"accuracy_floor": -0.1},
            {"accuracy_floor": float("nan")},
            {"weights": (1, 1)},
            {"weights": (0, 0, 0)},
            {"weights": (1, -1, 1)},
        ):
            with self.assertRaises(SelectionError, msg=kwargs):
                select(self.cands, "best_tradeoff", **kwargs)
        with self.assertRaises(SelectionError):
            select(self.cands, "nonsense")
        with self.assertRaises(SelectionError):
            select([C("x", 0.5, 0.1, 1.0), C("x", 0.6, 0.1, 1.0)], "max_accuracy")
        with self.assertRaises(SelectionError):
            select([], "max_accuracy")

    def test_exact_ties_follow_the_documented_order(self):
        cands = [C(0, 0.8, 0.1, 5.0), C(1, 0.8, 0.1, 5.0), C(2, 0.8, 0.1, 4.0)]
        self.assertEqual(select(cands, "max_accuracy").candidate_id, 2)  # then time
        self.assertEqual(
            select(cands[:2], "max_accuracy").candidate_id, 0
        )  # then order

    def test_result_does_not_depend_on_candidate_order_when_unique(self):
        rng = random.Random(1)
        shuffled = list(self.cands)
        for objective in selection.OBJECTIVES:
            want = select(self.cands, objective).candidate_id
            for _ in range(10):
                rng.shuffle(shuffled)
                self.assertEqual(select(shuffled, objective).candidate_id, want)

    def test_tradeoff_weights_change_the_answer_and_are_reported(self):
        equal = select(self.cands, "best_tradeoff", weights=(1, 1, 1))
        acc_heavy = select(self.cands, "best_tradeoff", weights=(100, 0.01, 0.01))
        self.assertEqual(
            equal.candidate_id, "d"
        )  # balanced: accurate, quick, low forgetting
        self.assertEqual(acc_heavy.candidate_id, "a")  # accuracy dominates
        self.assertEqual(acc_heavy.weights, (100.0, 0.01, 0.01))


class MissingValues(unittest.TestCase):
    def test_failed_candidates_are_excluded_with_a_reason_never_ranked_as_zero(self):
        cands = [C("ok", 0.7, 0.1, 5.0), C("bad", None, None, None)]
        s = select(cands, "min_forgetting")
        self.assertEqual(s.candidate_id, "ok")
        self.assertEqual([i for i, _ in s.excluded], ["bad"])

    def test_nan_counts_as_missing(self):
        cands = [C("ok", 0.7, 0.1, 5.0), C("nan", float("nan"), 0.0, 1.0)]
        self.assertEqual(select(cands, "min_forgetting").candidate_id, "ok")

    def test_missing_time_only_excludes_from_time_objectives(self):
        cands = [C("fast", 0.7, 0.2, 1.0), C("notime", 0.9, 0.0, None)]
        self.assertEqual(select(cands, "max_accuracy").candidate_id, "notime")
        self.assertEqual(select(cands, "min_forgetting").candidate_id, "notime")
        s = select(cands, "min_time")
        self.assertEqual(s.candidate_id, "fast")
        self.assertEqual(s.excluded[0][0], "notime")

    def test_tradeoff_with_time_ignored_does_not_need_time(self):
        cands = [C("a", 0.9, 0.2, None), C("b", 0.8, 0.0, None)]
        s = select(cands, "best_tradeoff", weights=(1, 1, 0))
        self.assertIn(s.candidate_id, ("a", "b"))
        with self.assertRaises(SelectionError):
            select(cands, "best_tradeoff", weights=(1, 1, 1))

    def test_select_all_reports_an_unavailable_objective_instead_of_raising(self):
        out = select_all([C(0, 0.8, 0.1, None), C(1, 0.7, 0.0, None)])
        self.assertIn("error", out["winners"]["min_time"])
        self.assertEqual(out["winners"]["max_accuracy"]["candidate_id"], 0)
        self.assertEqual(out["pareto_front_ids"], [])  # needs complete triples
        self.assertEqual(
            out["tradeoff_sensitivity"]["time_ignored"][
                "weights_accuracy_forgetting_time"
            ],
            [1.0, 1.0, 0.0],
        )


class SelectAll(unittest.TestCase):
    def test_structure_and_fingerprint(self):
        cands = [C(0, 0.8, 0.1, 5.0), C(1, 0.79, 0.0, 3.0), C(2, 0.6, 0.0, 1.0)]
        out = select_all(cands, accuracy_floor=0.02)
        self.assertEqual(set(out["winners"]), set(selection.OBJECTIVES))
        self.assertEqual(set(out["floor_winners"]), set(selection.FLOOR_OBJECTIVES))
        self.assertEqual(out["floor_winners"]["min_time"]["candidate_id"], 1)
        self.assertEqual(sorted(out["pareto_front_ids"]), [0, 1, 2])
        self.assertNotEqual(
            out["rule_fingerprint"],
            select_all(cands, accuracy_floor=0.05)["rule_fingerprint"],
        )
        self.assertEqual(
            out["rule_fingerprint"],
            select_all(cands, accuracy_floor=0.02)["rule_fingerprint"],
        )
        json.dumps(out, allow_nan=False)

    def test_no_floor_means_no_floor_winners(self):
        out = select_all([C(0, 0.8, 0.1, 5.0)], accuracy_floor=None)
        self.assertEqual(out["floor_winners"], {})


def tuning_dict(split="validation", with_time=True, failed=()):
    configs, runs = [], []
    table = {0: (0.80, 0.10, 10.0), 1: (0.78, 0.02, 12.0), 2: (0.70, 0.01, 1.0)}
    for idx, (acc, forg, wall) in table.items():
        ok = idx not in failed
        statuses = ["completed" if ok else "failed"] * 2
        configs.append(
            {
                "config_index": idx,
                "config": {"learning_rate": 10**-idx},
                "statuses": statuses,
                "eligible": ok,
                "mean_final_accuracy": acc if ok else None,
                "mean_forgetting": forg if ok else None,
            }
        )
        for seed in (100, 101):
            run = {"config_index": idx, "seed": seed}
            if with_time and ok:
                run["wall_time_seconds"] = wall
                run["cpu_time_seconds"] = wall / 2
            runs.append(run)
    return {
        "algorithm": "toy",
        "protocol_fingerprint": "fp",
        "selection_seeds": [100, 101],
        "validation_dataset": {"evaluation_split": split},
        "configs": configs,
        "runs": runs,
    }


class FromTuning(unittest.TestCase):
    def test_candidates_use_configuration_means_and_the_chosen_time_basis(self):
        wall = {c.id: c.time_seconds for c in candidates_from_tuning(tuning_dict())}
        cpu = {
            c.id: c.time_seconds for c in candidates_from_tuning(tuning_dict(), "cpu")
        }
        self.assertEqual(wall, {0: 10.0, 1: 12.0, 2: 1.0})
        self.assertEqual(cpu, {0: 5.0, 1: 6.0, 2: 0.5})
        with self.assertRaises(SelectionError):
            candidates_from_tuning(tuning_dict(), "gpu")

    def test_failed_configuration_is_excluded_and_reported(self):
        out = report_from_tuning(tuning_dict(failed=(1,)))
        for objective in ("min_forgetting", "min_time"):
            self.assertEqual(
                [e["candidate_id"] for e in out["winners"][objective]["excluded"]], [1]
            )
        self.assertNotIn(1, [w["candidate_id"] for w in out["winners"].values()])

    def test_old_tuning_file_without_time_degrades_gracefully(self):
        out = report_from_tuning(tuning_dict(with_time=False))
        self.assertIn("error", out["winners"]["min_time"])
        self.assertEqual(out["winners"]["max_accuracy"]["candidate_id"], 0)

    def test_test_split_results_are_refused(self):
        for split in ("test", None):
            with self.assertRaises(SelectionError):
                report_from_tuning(tuning_dict(split=split))

    def test_report_names_everything_needed_to_trace_it(self):
        out = report_from_tuning(tuning_dict(), accuracy_floor=0.03, time_basis="cpu")
        for key, value in {
            "algorithm": "toy",
            "ranking_basis": "validation",
            "time_basis": "cpu",
            "accuracy_floor": 0.03,
            "protocol_fingerprint": "fp",
        }.items():
            self.assertEqual(out[key], value)
        self.assertIn("rule_fingerprint", out)
        md = selection.render_markdown(out)
        self.assertIn("never test", md)
        self.assertIn("min_time (floor)", md)


class Cli(unittest.TestCase):
    def test_writes_report_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "t.json").write_text(json.dumps(tuning_dict()))
            args = ["--tuning", str(d / "t.json"), "--out-json", str(d / "s.json")]
            self.assertEqual(selection.main(args + ["--out-md", str(d / "s.md")]), 0)
            report = json.loads((d / "s.json").read_text())
            self.assertEqual(report["source_tuning_file"], "t.json")
            self.assertEqual(len(report["source_tuning_sha256"]), 64)
            self.assertEqual(selection.main(args), 2)  # exists

    def test_refuses_test_split(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "t.json").write_text(json.dumps(tuning_dict(split="test")))
            self.assertEqual(
                selection.main(
                    ["--tuning", str(d / "t.json"), "--out-json", str(d / "s.json")]
                ),
                2,
            )
            self.assertFalse((d / "s.json").exists())


class AnalysisToolRegression(unittest.TestCase):
    """tools/analyze_per_degree_study.py now delegates to this module. These winners were
    recorded from the script *before* that refactor, on a fixed synthetic study."""

    @staticmethod
    def _study():
        rng = random.Random(11)

        def block():
            return {
                "accuracy_mean": round(rng.uniform(0.55, 0.8), 3),
                "forgetting_mean": round(rng.uniform(0, 0.3), 3),
                "time_seconds_mean": round(rng.uniform(1, 30), 1),
                "current_accuracy_mean": round(rng.uniform(0.5, 1), 3),
            }

        out = []
        for i in range(25):
            v = block() if i % 7 else dict(block(), time_seconds_mean=None)
            out.append(
                {
                    "id": f"c{i}", "strategy": "ordinary", "sampling": "class_balanced",
                    "vector_label": f"v{i}", "learning_rates_by_degree": [0.1, 0.2, 0.3, 0.4],
                    "epochs": 3, "batch_size": 128, "memory_per_class": 200,
                    "n_ref_per_class": 1, "degree_schedule": [1, 2, 3, 4],
                    "n_features_final": 100, "validation": v,
                    "test": dict(block(), runs=3), "selected_by_core": i == 2,
                }
            )  # fmt: skip
        return out

    def test_winners_floor_winners_and_front_are_unchanged(self):
        import importlib.util

        path = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "analyze_per_degree_study.py"
        )
        spec = importlib.util.spec_from_file_location("analyze_tool", path)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        problems = []
        got = tool.analyse(self._study(), problems, 0.02)
        want = {
            "validation": (
                {"best_tradeoff": "c17", "max_accuracy": "c2", "min_forgetting": "c19", "min_time": "c6"},
                {"best_tradeoff": "c6", "min_forgetting": "c17", "min_time": "c6"},
                ["c10", "c17", "c18", "c19", "c2", "c6"],
            ),
            "test": (
                {"best_tradeoff": "c11", "max_accuracy": "c4", "min_forgetting": "c2", "min_time": "c19"},
                {"best_tradeoff": "c18", "min_forgetting": "c18", "min_time": "c18"},
                ["c0", "c11", "c13", "c14", "c17", "c18", "c19", "c2", "c4", "c6", "c8"],
            ),
        }  # fmt: skip
        for basis, (winners, floor, front) in want.items():
            block = got[basis]
            self.assertEqual(
                {k: v["candidate_id"] for k, v in block["winners"].items()}, winners
            )
            self.assertEqual(
                {
                    k: v["candidate_id"]
                    for k, v in block["accuracy_floor"]["winners"].items()
                },
                floor,
            )
            self.assertEqual(block["pareto_front_ids"], front)
        # only the validation blocks lack time (4 of 25); they are reported, not ranked
        self.assertEqual(len(problems), 1)
        self.assertIn("validation basis: 4 candidates have no time", problems[0])


class TuningRecordsTimeAndFeedsSelection(unittest.TestCase):
    """Real ``tuning.tune`` on a synthetic dataset with prescribed adapters."""

    def test_tune_records_time_and_selection_can_use_it(self):
        import numpy as np
        from helpers import small_protocol, synthetic_dataset

        from polynomial_cl_bench import tuning
        from polynomial_cl_bench.algorithms.base import SequenceResult

        class Adapter:
            def __init__(self, final, sleep=0.0, fail=False):
                self.final, self.sleep, self.fail = final, sleep, fail

            def run_sequence(self, dataset, protocol, seed, plan=None):
                import time

                time.sleep(self.sleep)
                if self.fail:
                    raise RuntimeError("boom")
                m = np.full((5, 5), np.nan)
                for i in range(5):
                    m[i, : i + 1] = self.final
                return SequenceResult(
                    m, {}, {}, "class_balanced",
                    {"dataset": {"evaluation_split": dataset.evaluation_split},
                     "plan_fingerprint": plan.fingerprint()},
                    [0] * 5,
                )  # fmt: skip

        grid = [
            {"final": 0.8, "sleep": 0.03},
            {"final": 0.79, "sleep": 0.0},
            {"final": 0.9, "fail": True},
        ]
        result = tuning.tune(
            "toy",
            lambda e: Adapter(**e),
            grid,
            synthetic_dataset(train_per_class=40),
            small_protocol(epochs=1, batch_size=24, memory_per_class=8),
        ).to_dict()
        json.dumps(result, allow_nan=False)
        for run in result["runs"]:
            if run["status"] == "completed":
                self.assertGreaterEqual(run["wall_time_seconds"], 0.0)
                self.assertGreaterEqual(run["cpu_time_seconds"], 0.0)
            else:  # unavailable, never zero
                self.assertIsNone(run["wall_time_seconds"])
                self.assertIsNone(run["cpu_time_seconds"])
        self.assertGreaterEqual(
            min(
                r["wall_time_seconds"] for r in result["runs"] if r["config_index"] == 0
            ),
            0.03,
        )
        # the existing accuracy-first selection is unchanged by recording time
        self.assertEqual(result["selected_index"], 0)

        report = report_from_tuning(result, accuracy_floor=0.02)
        self.assertEqual(report["winners"]["max_accuracy"]["candidate_id"], 0)
        self.assertEqual(report["winners"]["min_time"]["candidate_id"], 1)
        failed = report["winners"]["min_time"]["excluded"]
        self.assertEqual([e["candidate_id"] for e in failed], [2])  # kept, with reason


if __name__ == "__main__":
    unittest.main()
