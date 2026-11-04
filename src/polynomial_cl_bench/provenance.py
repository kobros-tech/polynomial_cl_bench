# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Provenance of the upstream core and of this benchmark, and enforcement for official runs.

Two run modes exist:

* ``development``: any installed core may be used; the provenance (including every reason it
  would not qualify as official) is recorded with the result.
* ``official``: refused with ``ProvenanceError`` before any run starts unless **all** of the
  following hold (every failed condition is listed in ``official_problems``):

  - the core is exactly the pinned commit, **and** its remote identity is verified:
    either a pip VCS install whose recorded URL is the pinned URL (pip fetched that commit
    from that remote), or a *clean* local checkout whose remote URL is the pinned URL and
    whose remote-tracking refs contain the pinned commit;
  - the benchmark itself has a determinable commit (a clean git checkout or a pip VCS
    install). A dirty benchmark checkout is rejected.

An unverified remote is an error, not a warning. Remote verification is evidence-based and
offline: it reads what pip or git already recorded and never touches the network.
Limitation: a remote-tracking ref proves the commit was on the remote at the last fetch or
push; it cannot detect a later force-push or deletion (that needs ``git ls-remote``).
"""

from __future__ import annotations

import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

RUN_MODES = ("official", "development")

_PIN_PATTERN = re.compile(r"polynomial-cl\s*@\s*git\+(\S+?)@([0-9a-f]{40})\b")


class ProvenanceError(RuntimeError):
    """The installed core or benchmark is not acceptable for an official run."""


# --- URL and git helpers ------------------------------------------------------------------


def normalize_remote_url(url: str | None) -> str | None:
    """Canonical form for comparing remote identities (https/ssh/scp forms, ``.git``, case)."""
    if not url:
        return None
    text = url.strip()
    text = text.removeprefix("git+")
    scp = re.match(
        r"^[\w.-]+@([\w.-]+):(?!//)(.+)$", text
    )  # git@github.com:owner/repo.git
    if scp:
        host, path = scp.group(1), scp.group(2)
    else:
        parsed = urlparse(text)
        host, path = parsed.hostname or "", parsed.path
    path = path.strip("/")
    path = path.removesuffix(".git")
    return f"{host.lower()}/{path}" if host else None


def _git(directory, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def commit_from_direct_url(direct_url: dict | None) -> str | None:
    """Commit recorded by pip for a VCS install (PEP 610), else None."""
    if not direct_url:
        return None
    return (direct_url.get("vcs_info") or {}).get("commit_id")


def _direct_url(distribution_name: str):
    """(distribution, parsed direct_url.json or None); distribution None if not installed."""
    try:
        dist = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None, None
    text = dist.read_text("direct_url.json")
    return dist, (json.loads(text) if text else None)


def _local_directory(direct_url: dict | None):
    url = (direct_url or {}).get("url", "")
    if url.startswith("file://"):
        return Path(unquote(urlparse(url).path))
    return None


def pinned_core_requirement() -> dict | None:
    """``{"url": ..., "commit": ...}`` from this package's own metadata pin, if parseable."""
    try:
        requirements = metadata.requires("polynomial-cl-bench") or []
    except metadata.PackageNotFoundError:
        return None
    for requirement in requirements:
        match = _PIN_PATTERN.match(requirement)
        if match:
            return {"url": match.group(1), "commit": match.group(2)}
    return None


def pinned_core_commit() -> str | None:
    """The commit this benchmark's own metadata pins the core to, if parseable."""
    pin = pinned_core_requirement()
    return pin["commit"] if pin else None


def _checkout_state(directory: Path) -> dict:
    """HEAD, dirtiness and remote URLs of a local git checkout (raises on git failure).

    Dirty means any change to a tracked file **or** an untracked, non-ignored file: an extra
    source file in the checkout can change behaviour just as an edit can.
    """
    commit = _git(directory, "rev-parse", "HEAD")
    dirty = bool(_git(directory, "status", "--porcelain"))
    remotes = {}
    try:
        listing = _git(directory, "config", "--get-regexp", r"^remote\..*\.url$")
    except subprocess.CalledProcessError:  # git exits 1 when no remote is configured
        listing = ""
    for line in listing.splitlines():
        key, _, value = line.partition(" ")
        remotes[key[len("remote.") : -len(".url")]] = value
    return {"commit": commit, "dirty": dirty, "remotes": remotes}


def _checkout_state_or_none(directory: Path | None) -> dict | None:
    if directory is None:
        return None
    try:
        return _checkout_state(directory)
    except (OSError, subprocess.SubprocessError):
        return None


def _remote_containing(directory: Path, commit: str, remote_names) -> str | None:
    """Name of the first remote with a remote-tracking ref that contains ``commit``."""
    for name in remote_names:
        try:
            refs = _git(
                directory,
                "for-each-ref",
                "--contains",
                commit,
                "--format=%(refname)",
                f"refs/remotes/{name}",
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if refs:
            return name
    return None


# --- core ---------------------------------------------------------------------------------


def core_provenance() -> dict:
    """Which core was imported, and how much of that claim is verified.

    ``commit_source``:

    * ``pip_vcs_install``: pip installed it from ``remote_url`` and recorded the resolved
      commit (PEP 610). ``remote_verified`` requires that URL to be the pinned URL.
    * ``local_git`` / ``local_git_dirty``: read with git from a local checkout (editable or
      path install). ``remote_verified`` requires a remote whose URL is the pinned URL **and**
      a remote-tracking ref of that remote containing the commit. ``_dirty`` means the
      checkout has modified or untracked non-ignored files.
    * ``unknown``: no commit could be determined.

    ``official_problems`` lists every reason this installation fails the official-run rule.
    """
    dist, direct_url = _direct_url("polynomial-cl")
    version = dist.version if dist is not None else None
    pin = pinned_core_requirement()
    pinned_commit = pin["commit"] if pin else None
    pinned_url = pin["url"] if pin else None
    commit = commit_from_direct_url(direct_url)
    source, remote_url, verification = "unknown", None, "no source of the commit found"
    remote_verified, dirty = False, False
    if commit is not None:
        source, remote_url = "pip_vcs_install", (direct_url or {}).get("url")
        if pinned_url and normalize_remote_url(remote_url) == normalize_remote_url(
            pinned_url
        ):
            remote_verified = True
            verification = "pip fetched the commit from the pinned remote"
        else:
            verification = f"pip installed from {remote_url!r}, not the pinned remote"
    else:
        directory = _local_directory(direct_url)
        state = _checkout_state_or_none(directory)
        if state is not None:
            commit, dirty = state["commit"], state["dirty"]
            source = "local_git_dirty" if dirty else "local_git"
            matching = [
                name
                for name, url in state["remotes"].items()
                if pinned_url
                and normalize_remote_url(url) == normalize_remote_url(pinned_url)
            ]
            if not matching:
                verification = "checkout has no remote whose URL is the pinned remote"
            elif (name := _remote_containing(directory, commit, matching)) is None:
                verification = f"no remote-tracking ref of {matching} contains {commit}; fetch the remote"
            else:
                remote_url, remote_verified = state["remotes"][name], True
                verification = f"remote-tracking ref of {name!r} contains the commit"
    matches = (
        None if commit is None or pinned_commit is None else commit == pinned_commit
    )
    problems = []
    if pinned_commit is None:
        problems.append("no pinned core commit found in this package's metadata")
    if commit is None:
        problems.append("core commit could not be determined")
    if dirty:
        problems.append("core checkout has uncommitted or untracked changes")
    if matches is False:
        problems.append(
            f"core commit {commit} differs from the pinned commit {pinned_commit}"
        )
    if commit is not None and not remote_verified:
        problems.append(f"core remote identity not verified: {verification}")
    return {
        "version": version,
        "commit": commit,
        "commit_source": source,
        "remote_url": remote_url,
        "remote_verified": remote_verified,
        "remote_verification": verification,
        "pinned_commit": pinned_commit,
        "pinned_remote_url": pinned_url,
        "matches_pin": matches,
        "pin_remote_status": "confirmed"
        if remote_verified and matches
        else "unverified",
        "official_problems": problems,
    }


# --- benchmark ----------------------------------------------------------------------------


def benchmark_provenance() -> dict:
    """Commit of this benchmark, recorded with every result; dirty/unknown fails official runs.

    ``commit_source`` is ``pip_vcs_install`` (PEP 610 commit), ``local_git`` /
    ``local_git_dirty`` (this very file is tracked in the checkout at ``HEAD``) or
    ``unknown`` (e.g. an sdist/wheel install or a copy without git).
    """
    dist, direct_url = _direct_url("polynomial-cl-bench")
    version = dist.version if dist is not None else None
    commit = commit_from_direct_url(direct_url)
    source, remote_url, dirty = "unknown", None, False
    if commit is not None:
        source, remote_url = "pip_vcs_install", (direct_url or {}).get("url")
    else:
        here = Path(__file__).resolve()
        directory = _local_directory(direct_url) or here.parent
        state = _checkout_state_or_none(directory)
        tracked = False
        if state is not None:
            try:  # guard against being nested inside an unrelated repository
                _git(directory, "ls-files", "--error-unmatch", str(here))
                tracked = True
            except (OSError, subprocess.SubprocessError):
                pass
        if state is not None and tracked:
            commit, dirty = state["commit"], state["dirty"]
            source = "local_git_dirty" if dirty else "local_git"
            remote_url = state["remotes"].get("origin")
    problems = []
    if commit is None:
        problems.append("benchmark commit could not be determined")
    if dirty:
        problems.append("benchmark checkout has uncommitted or untracked changes")
    return {
        "version": version,
        "commit": commit,
        "commit_source": source,
        "remote_url": remote_url,
        "official_problems": problems,
    }


# --- enforcement --------------------------------------------------------------------------


def enforce_run_mode(mode: str, core: dict, benchmark: dict | None = None) -> None:
    """Raise ``ProvenanceError`` for an official run on an unacceptable core or benchmark.

    Development runs always pass (their provenance is recorded by the caller). Official runs
    fail closed: missing benchmark provenance, any core problem (including an unverified
    remote) and any benchmark problem each refuse the run. There is no warning-only path.
    """
    if mode not in RUN_MODES:
        raise ValueError(f"mode must be one of {RUN_MODES}")
    if mode == "development":
        return
    problems = [f"core: {p}" for p in core["official_problems"]]
    if benchmark is None:
        problems.append("benchmark: provenance was not supplied")
    else:
        problems += [f"benchmark: {p}" for p in benchmark["official_problems"]]
    if problems:
        raise ProvenanceError("official run refused: " + "; ".join(problems))
