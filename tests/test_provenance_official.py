# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Official-run provenance rules (Phase 2 review item 6).

Runs without ``polynomial_cl``: git repositories are real but local (the remote URL is a
string that is never contacted; the remote-tracking ref is created with ``git update-ref``).
"""

import json
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from polynomial_cl_bench import provenance as prov
from polynomial_cl_bench.provenance import ProvenanceError, enforce_run_mode

PIN_URL = "https://github.com/kobros-tech/polynomial_cl_demo.git"
OTHER_URL = "https://github.com/someone-else/polynomial_cl_demo.git"
PIN = "9" * 40


def git(directory, *args):
    return subprocess.run(
        [
            "git",
            "-C",
            str(directory),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_repo(directory, remote_url=None, track_remote=False, filename="f.txt"):
    """Committed repo; optionally an ``origin`` URL and a remote-tracking ref at HEAD."""
    directory = Path(directory)
    git(directory, "init", "-q")
    (directory / filename).parent.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text("a")
    git(directory, "add", filename)
    git(directory, "commit", "-q", "-m", "c")
    head = git(directory, "rev-parse", "HEAD")
    if remote_url:
        git(directory, "remote", "add", "origin", remote_url)
        if track_remote:
            git(directory, "update-ref", "refs/remotes/origin/main", head)
    return head


class FakeDistribution:
    version = "0.1.0"

    def __init__(self, direct_url):
        self._text = None if direct_url is None else json.dumps(direct_url)

    def read_text(self, name):
        return self._text if name == "direct_url.json" else None


def patched(core_direct_url=None, bench_direct_url=None, pin=None):
    """Patch metadata lookups so no installed package is consulted."""
    pin = pin if pin is not None else {"url": PIN_URL, "commit": PIN}

    def distribution(name):
        direct = {
            "polynomial-cl": core_direct_url,
            "polynomial-cl-bench": bench_direct_url,
        }[name]
        return FakeDistribution(direct)

    stack = mock.patch.object(prov.metadata, "distribution", distribution)
    pin_patch = mock.patch.object(prov, "pinned_core_requirement", lambda: pin)
    return stack, pin_patch


def core_info(direct_url, pin=None):
    a, b = patched(core_direct_url=direct_url, pin=pin)
    with a, b:
        return prov.core_provenance()


def bench_info(direct_url, file_override=None):
    a, b = patched(bench_direct_url=direct_url)
    with a, b:
        if file_override:
            with mock.patch.object(prov, "__file__", str(file_override)):
                return prov.benchmark_provenance()
        return prov.benchmark_provenance()


def pip_url(url=PIN_URL, commit=PIN):
    return {"url": url, "vcs_info": {"vcs": "git", "commit_id": commit}}


def local_url(directory):
    return {"url": f"file://{directory}", "dir_info": {"editable": True}}


class UrlNormalisation(unittest.TestCase):
    def test_equivalent_forms_compare_equal(self):
        forms = [
            "https://github.com/kobros-tech/polynomial_cl_demo.git",
            "https://github.com/kobros-tech/polynomial_cl_demo",
            "git+https://github.com/kobros-tech/polynomial_cl_demo.git",
            "https://GitHub.com/kobros-tech/polynomial_cl_demo/",
            "git@github.com:kobros-tech/polynomial_cl_demo.git",
            "ssh://git@github.com/kobros-tech/polynomial_cl_demo.git",
        ]
        self.assertEqual(
            {prov.normalize_remote_url(f) for f in forms},
            {"github.com/kobros-tech/polynomial_cl_demo"},
        )

    def test_different_repositories_do_not_compare_equal(self):
        self.assertNotEqual(
            prov.normalize_remote_url(PIN_URL), prov.normalize_remote_url(OTHER_URL)
        )
        self.assertIsNone(prov.normalize_remote_url(None))
        self.assertIsNone(prov.normalize_remote_url(""))


class CorePipVcs(unittest.TestCase):
    def test_pinned_commit_from_pinned_remote_is_verified(self):
        info = core_info(pip_url())
        self.assertEqual(
            (info["commit_source"], info["remote_verified"], info["matches_pin"]),
            ("pip_vcs_install", True, True),
        )
        self.assertEqual(info["official_problems"], [])
        self.assertEqual(info["pin_remote_status"], "confirmed")

    def test_equivalent_url_spelling_is_accepted(self):
        info = core_info(
            pip_url(url="https://github.com/kobros-tech/polynomial_cl_demo")
        )
        self.assertEqual(info["official_problems"], [])

    def test_other_remote_is_rejected_even_at_the_right_commit(self):
        info = core_info(pip_url(url=OTHER_URL))
        self.assertFalse(info["remote_verified"])
        self.assertTrue(any("remote identity" in p for p in info["official_problems"]))
        self.assertEqual(info["pin_remote_status"], "unverified")

    def test_wrong_commit_is_rejected(self):
        info = core_info(pip_url(commit="1" * 40))
        self.assertFalse(info["matches_pin"])
        self.assertTrue(
            any(
                "differs from the pinned commit" in p for p in info["official_problems"]
            )
        )


class CoreLocalCheckout(unittest.TestCase):
    def check(self, **repo):
        with tempfile.TemporaryDirectory() as tmp:
            head = make_repo(tmp, **repo)
            return (
                core_info(local_url(tmp), pin={"url": PIN_URL, "commit": head}),
                tmp,
                head,
            )

    def test_clean_checkout_with_matching_remote_and_tracking_ref_is_verified(self):
        info, _, head = self.check(remote_url=PIN_URL, track_remote=True)
        self.assertEqual((info["commit_source"], info["commit"]), ("local_git", head))
        self.assertTrue(info["remote_verified"])
        self.assertEqual(info["remote_url"], PIN_URL)
        self.assertEqual(info["official_problems"], [])

    def test_commit_missing_from_remote_tracking_refs_is_unverified(self):
        info, _, _ = self.check(remote_url=PIN_URL, track_remote=False)
        self.assertFalse(info["remote_verified"])
        self.assertTrue(any("remote identity" in p for p in info["official_problems"]))

    def test_checkout_without_the_pinned_remote_is_unverified(self):
        for remote in (None, OTHER_URL):
            with self.subTest(remote=remote):
                info, _, _ = self.check(remote_url=remote, track_remote=True)
                self.assertFalse(info["remote_verified"])
                self.assertIsNone(info["remote_url"])
                self.assertTrue(info["official_problems"])

    def test_modified_tracked_file_is_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = make_repo(tmp, remote_url=PIN_URL, track_remote=True)
            (Path(tmp) / "f.txt").write_text("b")
            info = core_info(local_url(tmp), pin={"url": PIN_URL, "commit": head})
        self.assertEqual(info["commit_source"], "local_git_dirty")
        self.assertTrue(any("uncommitted" in p for p in info["official_problems"]))

    def test_untracked_source_file_is_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = make_repo(tmp, remote_url=PIN_URL, track_remote=True)
            (Path(tmp) / "extra.py").write_text("x = 1")
            info = core_info(local_url(tmp), pin={"url": PIN_URL, "commit": head})
        self.assertEqual(info["commit_source"], "local_git_dirty")

    def test_ignored_files_do_not_make_a_checkout_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            (Path(tmp) / ".gitignore").write_text("*.pyc\n")
            git(tmp, "add", ".gitignore")
            git(tmp, "commit", "-q", "-m", "ignore")
            head = git(tmp, "rev-parse", "HEAD")
            git(tmp, "remote", "add", "origin", PIN_URL)
            git(tmp, "update-ref", "refs/remotes/origin/main", head)
            (Path(tmp) / "a.pyc").write_text("x")
            self.assertEqual(git(tmp, "status", "--porcelain"), "")  # really ignored
            info = core_info(local_url(tmp), pin={"url": PIN_URL, "commit": head})
        self.assertEqual(info["commit_source"], "local_git")

    def test_commit_other_than_the_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp, remote_url=PIN_URL, track_remote=True)
            info = core_info(local_url(tmp), pin={"url": PIN_URL, "commit": "2" * 40})
        self.assertFalse(info["matches_pin"])
        self.assertTrue(
            any(
                "differs from the pinned commit" in p for p in info["official_problems"]
            )
        )

    def test_unknown_source_has_no_commit_and_is_rejected(self):
        info = core_info({"url": "https://example.invalid/pkg.whl", "archive_info": {}})
        self.assertEqual(
            (info["commit"], info["commit_source"], info["remote_verified"]),
            (None, "unknown", False),
        )
        self.assertIsNone(info["matches_pin"])
        self.assertTrue(info["official_problems"])

    def test_missing_pin_is_rejected(self):
        a, _b = patched(core_direct_url=pip_url())
        with a, mock.patch.object(prov, "pinned_core_requirement", lambda: None):
            info = prov.core_provenance()
        self.assertTrue(
            any("no pinned core commit" in p for p in info["official_problems"])
        )


class BenchmarkProvenance(unittest.TestCase):
    def test_clean_local_checkout_records_the_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = make_repo(tmp, remote_url=PIN_URL, filename="src/pkg/provenance.py")
            info = bench_info(
                local_url(tmp), file_override=Path(tmp) / "src/pkg/provenance.py"
            )
        self.assertEqual((info["commit"], info["commit_source"]), (head, "local_git"))
        self.assertEqual(info["official_problems"], [])

    def test_dirty_local_checkout_is_rejected(self):
        for change in ("tracked", "untracked"):
            with self.subTest(change), tempfile.TemporaryDirectory() as tmp:
                make_repo(tmp, filename="src/pkg/provenance.py")
                target = "src/pkg/provenance.py" if change == "tracked" else "new.py"
                (Path(tmp) / target).write_text("changed")
                info = bench_info(
                    local_url(tmp), file_override=Path(tmp) / "src/pkg/provenance.py"
                )
                self.assertEqual(info["commit_source"], "local_git_dirty")
                self.assertTrue(
                    any("uncommitted" in p for p in info["official_problems"])
                )

    def test_pip_vcs_install_is_accepted(self):
        info = bench_info(
            pip_url(
                url="https://github.com/kobros-tech/polynomial_cl_bench.git",
                commit="a" * 40,
            )
        )
        self.assertEqual(
            (info["commit"], info["commit_source"]), ("a" * 40, "pip_vcs_install")
        )
        self.assertEqual(info["official_problems"], [])

    def test_unrelated_enclosing_repository_is_not_mistaken_for_the_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)  # a repository that does not track this package's files
            info = bench_info(local_url(tmp))
        self.assertEqual((info["commit"], info["commit_source"]), (None, "unknown"))
        self.assertTrue(info["official_problems"])

    def test_no_git_and_no_vcs_metadata_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "provenance.py").write_text("")
            info = bench_info(None, file_override=Path(tmp) / "provenance.py")
        self.assertEqual(info["commit_source"], "unknown")
        self.assertIn(
            "benchmark commit could not be determined", info["official_problems"]
        )


GOOD_CORE = {"official_problems": []}
GOOD_BENCH = {"official_problems": []}


class Enforcement(unittest.TestCase):
    def test_official_accepts_a_verified_core_and_clean_benchmark(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning-only path exists any more
            enforce_run_mode("official", GOOD_CORE, GOOD_BENCH)

    def test_official_rejects_each_failure_alone(self):
        cases = {
            "core": (
                {"official_problems": ["core remote identity not verified: x"]},
                GOOD_BENCH,
            ),
            "benchmark": (
                GOOD_CORE,
                {"official_problems": ["benchmark checkout has uncommitted"]},
            ),
            "no benchmark provenance": (GOOD_CORE, None),
        }
        for name, (core, bench) in cases.items():
            with self.subTest(name), self.assertRaises(ProvenanceError):
                enforce_run_mode("official", core, bench)

    def test_refusal_lists_every_problem(self):
        with self.assertRaises(ProvenanceError) as caught:
            enforce_run_mode(
                "official",
                {"official_problems": ["c1", "c2"]},
                {"official_problems": ["b1"]},
            )
        text = str(caught.exception)
        for fragment in ("core: c1", "core: c2", "benchmark: b1"):
            self.assertIn(fragment, text)

    def test_unverified_remote_is_an_error_not_a_warning(self):
        info = core_info(pip_url(url=OTHER_URL))
        with self.assertRaises(ProvenanceError):
            enforce_run_mode("official", info, GOOD_BENCH)

    def test_development_never_rejects(self):
        bad = {"official_problems": ["anything"]}
        enforce_run_mode("development", bad, bad)
        enforce_run_mode("development", bad)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            enforce_run_mode("beta", GOOD_CORE, GOOD_BENCH)


if __name__ == "__main__":
    unittest.main()
