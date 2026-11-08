# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The GitHub jobs that produce the paper's results are planned by code that is tested here.

Nothing runs a study: these tests check that the plan lists exactly the cells the analysis tool
reads, that it follows the installed core's strategies and sampling policies, that the pinned
core is read from pyproject.toml, and that the workflow files are well formed and only call
tools that exist.
"""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
WORKFLOWS = ROOT / ".github" / "workflows"
sys.path.insert(0, str(TOOLS))

import core_study_plan as plan  # noqa: E402
import pinned_core  # noqa: E402


class CoreStudyPlanTests(unittest.TestCase):
    def test_strategies_and_samplings_are_the_installed_cores(self):
        from polynomial_cl.continual.experiment import STRATEGIES
        from polynomial_cl.continual.sampling import SAMPLING_POLICIES

        self.assertEqual(set(plan.STRATEGIES), set(STRATEGIES))
        self.assertEqual(set(plan.SAMPLINGS), set(SAMPLING_POLICIES))

    def test_study_has_the_sampling_comparison_and_every_cell_once(self):
        names = [e["name"] for e in plan.study_entries()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "sampling")
        cells = {f"cell_{s}_{p}" for s in plan.STRATEGIES for p in plan.SAMPLINGS}
        self.assertEqual(set(names[1:]), cells)
        self.assertEqual(len(cells), 12)

    def test_epochs_stage_has_one_rerun_per_strategy(self):
        names = [e["name"] for e in plan.epochs_entries()]
        self.assertEqual(names, [f"epochs10_{s}" for s in plan.STRATEGIES])

    def test_output_and_log_names_are_what_the_analysis_reads(self):
        for entry in plan.study_entries() + plan.epochs_entries():
            if entry["name"] == "sampling":  # its file name carries epochs and rate
                self.assertEqual(entry["output"], "sampling_e3_lr0.1.json")
                continue
            self.assertEqual(entry["output"], f"{entry['name']}.json")
            self.assertRegex(entry["output"], r"^(cell|epochs10)_.+\.json$")
        # analyze_per_degree_study.py: epochs10_*.log are 10-epoch runs, all other logs 3
        source = (TOOLS / "analyze_per_degree_study.py").read_text()
        self.assertIn('startswith("epochs10_")', source)
        self.assertIn('glob("cell_*.json")', source)
        self.assertIn('glob("epochs10_*.json")', source)

    def test_every_run_names_the_dataset_and_one_experiment(self):
        for entry in plan.study_entries() + plan.epochs_entries():
            args = entry["args"].split()
            self.assertEqual(args[:2], ["--dataset", "split_mnist"])
            self.assertEqual(args.count("--experiment"), 1)
        stages = {
            e["args"].split("--experiment ")[1].split()[0]
            for e in plan.study_entries() + plan.epochs_entries()
        }
        self.assertEqual(stages, {"sampling", "per-degree", "epochs"})

    def test_cells_select_one_strategy_and_one_sampling(self):
        for entry in plan.study_entries()[1:]:
            args = entry["args"].split()
            self.assertEqual(args.count("--strategies"), 1)
            self.assertEqual(args.count("--samplings"), 1)
            strategy = args[args.index("--strategies") + 1]
            sampling = args[args.index("--samplings") + 1]
            self.assertEqual(entry["name"], f"cell_{strategy}_{sampling}")

    def test_only_the_defaults_of_the_script_are_overridden(self):
        # seeds, epochs of the selection stage and rates are the script's own defaults
        joined = " ".join(e["args"] for e in plan.study_entries()[1:])
        for flag in ("--select-seeds", "--test-seeds", "--epochs", "--lr"):
            self.assertNotIn(flag, joined)
        self.assertEqual(plan.EPOCHS_ROBUSTNESS, 10)

    def test_matrix_is_single_line_json_for_github_output(self):
        for stage in plan.STAGES:
            command = [sys.executable, str(TOOLS / "core_study_plan.py")]
            out = subprocess.run(
                [*command, "matrix", "--stage", stage],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertEqual(out.count("\n"), 1)
            self.assertIn("include", json.loads(out))

    def test_unknown_stage_is_an_error(self):
        with self.assertRaises(ValueError):
            plan.matrix("nonsense")


class PinnedCoreTests(unittest.TestCase):
    def write(self, dependency):
        tmp = Path(tempfile.mkdtemp()) / "pyproject.toml"
        dependencies = f'["numpy>=1", "{dependency}"]'
        tmp.write_text(f'[project]\nname = "x"\ndependencies = {dependencies}\n')
        return tmp

    SHA = "0123456789abcdef0123456789abcdef01234567"

    def test_reads_owner_name_and_sha(self):
        path = self.write(
            f"polynomial-cl @ git+https://github.com/a-b/c_d.git@{self.SHA}"
        )
        self.assertEqual(pinned_core.pinned_core(path), ("a-b/c_d", self.SHA))

    def test_the_repositorys_own_pin_is_readable(self):
        repo, sha = pinned_core.pinned_core(ROOT / "pyproject.toml")
        self.assertRegex(sha, r"^[0-9a-f]{40}$")
        self.assertIn("/", repo)

    def test_a_branch_or_short_sha_is_refused(self):
        for bad in (
            "polynomial-cl @ git+https://github.com/a/b.git@main",
            "polynomial-cl @ git+https://github.com/a/b.git@abc1234",
            "polynomial-cl>=0.1",
        ):
            with self.assertRaises(ValueError):
                pinned_core.pinned_core(self.write(bad))

    def test_command_line_prints_github_output_lines(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "pinned_core.py")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertTrue(out[0].startswith("core_repo="))
        self.assertTrue(re.fullmatch(r"core_sha=[0-9a-f]{40}", out[1]))


class WorkflowFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml
        except ImportError:
            raise unittest.SkipTest("PyYAML is not installed") from None
        cls.workflows = {
            p.name: yaml.safe_load(p.read_text()) for p in WORKFLOWS.glob("*.yml")
        }

    def test_the_results_workflows_exist_and_are_manual_only(self):
        for name in ("paper-results.yml", "benchmark-results.yml"):
            triggers = self.workflows[name][True]  # YAML reads the key 'on' as True
            self.assertEqual(list(triggers), ["workflow_dispatch"])

    def test_every_needed_job_exists(self):
        for name, workflow in self.workflows.items():
            jobs = workflow["jobs"]
            for job, body in jobs.items():
                needs = body.get("needs", [])
                for needed in [needs] if isinstance(needs, str) else needs:
                    self.assertIn(needed, jobs, (name, job))

    def test_called_tools_and_modules_exist(self):
        for path in WORKFLOWS.glob("*.yml"):
            text = path.read_text()
            for tool in re.findall(r"python (?:-u )?(tools/\w+\.py)", text):
                self.assertTrue((ROOT / tool).is_file(), (path.name, tool))
            for module in re.findall(r"python -m (polynomial_cl_bench\.\w+)", text):
                self.assertTrue(
                    (ROOT / "src" / (module.replace(".", "/") + ".py")).is_file(),
                    (path.name, module),
                )

    def test_every_study_cell_reaches_the_analysis_job(self):
        jobs = self.workflows["paper-results.yml"]["jobs"]
        self.assertEqual(sorted(jobs["analysis"]["needs"]), ["core-study", "epochs"])
        self.assertIn("core-study", jobs["epochs"]["needs"])  # needs the finished cells

    def test_the_study_runs_the_core_script_at_the_pinned_commit(self):
        text = (WORKFLOWS / "paper-results.yml").read_text()
        self.assertIn("needs.plan.outputs.core_sha", text)
        self.assertIn("experiments/per_degree_lr.py", text)
        self.assertNotIn(".sh", text.replace("**/*.sh", ""))

    def test_every_flag_a_workflow_passes_exists_in_the_tools_help(self):
        pattern = r"python (?:-u )?(tools/\w+\.py|-m polynomial_cl_bench\.\w+)([^\n]*)"
        for path in WORKFLOWS.glob("*-results.yml"):
            flat = path.read_text().replace("\\\n", " ")
            for target, rest in re.findall(pattern, flat):
                command = [sys.executable, *target.split()]
                help_text = subprocess.run(
                    [*command, "--help"], capture_output=True, text=True
                ).stdout
                self.assertTrue(help_text, (path.name, target))
                for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest):
                    self.assertIn(flag, help_text, (path.name, target))

    def test_mnist_is_loaded_with_a_path_not_a_string(self):
        # load_mnist(root: Path): a str fails with AttributeError on the first job
        for path in WORKFLOWS.glob("*-results.yml"):
            text = path.read_text()
            self.assertIn("load_mnist(Path('data/mnist'))", text)
            self.assertNotRegex(text, r"load_mnist\(['\"]")

    def test_the_epochs_job_runs_where_the_script_finds_the_cells(self):
        # per_degree_lr.py reads cell_*.json from ./results/per_degree by default
        steps = self.workflows["paper-results.yml"]["jobs"]["epochs"]["steps"]
        downloads = [
            s["with"] for s in steps if "download-artifact" in s.get("uses", "")
        ]
        self.assertIn("core/results/per_degree", [d["path"] for d in downloads])
        runs = [s for s in steps if "per_degree_lr.py" in s.get("run", "")]
        self.assertEqual([s["working-directory"] for s in runs], ["core"])

    def test_the_benchmark_workflow_follows_the_documented_pipeline(self):
        text = (WORKFLOWS / "benchmark-results.yml").read_text()
        for needle in (
            "polynomial_cl_bench.tune",
            "configs/er_mlp_grid.json",
            "configs/polynomial_grid.json",
            "configs/experiment_tuned_comparison.json",
            "--reproduce-check",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
