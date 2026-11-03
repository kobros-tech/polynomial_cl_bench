"""Provenance of the upstream core and enforcement of the pin for official runs.

Two run modes exist:

* ``development``: any installed core may be used; the provenance (including every reason it
  would not qualify as official) is recorded with the result.
* ``official``: the core must be exactly the pinned commit and the checkout must be clean.
  Anything else raises ``ProvenanceError`` before a run starts.

Remote verification is a separate, visible matter: ``PIN_REMOTE_STATUS`` stays ``"unverified"``
until someone confirms the pinned commit exists on the GitHub remote and edits this constant.
While it is unverified, every official run warns and every record carries the status.
"""

from __future__ import annotations

import json
import re
import subprocess
import warnings
from importlib import metadata
from pathlib import Path

RUN_MODES = ("official", "development")

# Flip to "confirmed" only after `git ls-remote` / a clone shows the pinned commit on the remote,
# and record the evidence (date, ref) in the commit message and README.
PIN_REMOTE_STATUS = "unverified"


class ProvenanceError(RuntimeError):
    """The installed core is not acceptable for an official run."""


class RemoteUnverifiedWarning(UserWarning):
    """The pinned core commit has not been confirmed to exist on the remote."""


def commit_from_direct_url(direct_url: dict | None) -> str | None:
    """Commit recorded by pip for a VCS install (PEP 610), else None."""
    if not direct_url:
        return None
    return (direct_url.get("vcs_info") or {}).get("commit_id")


def pinned_core_commit() -> str | None:
    """The commit this benchmark's own metadata pins the core to, if parseable."""
    for requirement in metadata.requires("polynomial-cl-bench") or []:
        match = re.match(r"polynomial-cl\s*@\s*git\+\S+@([0-9a-f]{40})\b", requirement)
        if match:
            return match.group(1)
    return None


def core_provenance() -> dict:
    """Which core was imported, and how much of that claim is verified.

    ``commit_source`` says where the commit came from:

    * ``pip_vcs_install``: pip installed it from ``remote_url`` and recorded the resolved commit
      (PEP 610). ``remote_verified`` is True: the commit was fetched from that remote.
    * ``local_git`` / ``local_git_dirty``: read with ``git rev-parse`` from a local checkout
      (editable or path install). The commit exists locally; **nothing here shows it exists on
      any remote**, so ``remote_verified`` is False. ``_dirty`` means tracked files are modified.
    * ``unknown``: no commit could be determined.

    ``matches_pin`` compares the commit with the pin in this package's metadata (None if either
    side is unknown). ``pin_remote_status`` is the project-level status of the pin itself.
    ``official_problems`` lists every reason this installation fails the official-run rule.
    """
    dist = metadata.distribution("polynomial-cl")
    text = dist.read_text("direct_url.json")
    direct_url = json.loads(text) if text else None
    commit = commit_from_direct_url(direct_url)
    source, remote_url = "unknown", None
    if commit is not None:
        source, remote_url = "pip_vcs_install", (direct_url or {}).get("url")
    else:
        url = (direct_url or {}).get("url", "")
        if url.startswith("file://"):
            directory = Path(url[len("file://"):])
            try:
                commit = subprocess.run(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True, timeout=30,
                ).stdout.strip()
                dirty = subprocess.run(
                    ["git", "-C", str(directory), "status", "--porcelain",
                     "--untracked-files=no"],
                    capture_output=True, text=True, check=True, timeout=30,
                ).stdout.strip()
                source = "local_git_dirty" if dirty else "local_git"
            except (OSError, subprocess.SubprocessError):
                commit = None
    pinned = pinned_core_commit()
    matches = None if commit is None or pinned is None else commit == pinned
    problems = []
    if pinned is None:
        problems.append("no pinned core commit found in this package's metadata")
    if commit is None:
        problems.append("core commit could not be determined")
    if source == "local_git_dirty":
        problems.append("core checkout has uncommitted changes to tracked files")
    if matches is False:
        problems.append(f"core commit {commit} differs from the pinned commit {pinned}")
    return {
        "version": dist.version,
        "commit": commit,
        "commit_source": source,
        "remote_url": remote_url,
        "remote_verified": source == "pip_vcs_install",
        "pinned_commit": pinned,
        "matches_pin": matches,
        "pin_remote_status": PIN_REMOTE_STATUS,
        "official_problems": problems,
    }


def enforce_run_mode(mode: str, provenance: dict) -> None:
    """Raise ``ProvenanceError`` for an official run on an unacceptable core.

    Development runs always pass (their provenance is recorded by the caller). Official runs on a
    clean core at the pinned commit pass but warn while the pin is remote-unverified.
    """
    if mode not in RUN_MODES:
        raise ValueError(f"mode must be one of {RUN_MODES}")
    if mode == "development":
        return
    if provenance["official_problems"]:
        raise ProvenanceError(
            "official run refused: " + "; ".join(provenance["official_problems"])
        )
    if provenance["pin_remote_status"] != "confirmed":
        warnings.warn(
            f"pinned core commit {provenance['pinned_commit']} is {provenance['pin_remote_status']} "
            "on the remote; this official run is only as trustworthy as the local checkout "
            f"(commit_source={provenance['commit_source']}, "
            f"remote_verified={provenance['remote_verified']})",
            RemoteUnverifiedWarning,
            stacklevel=3,
        )
