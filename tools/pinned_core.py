# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Print the core repository and commit that ``pyproject.toml`` pins.

The GitHub workflows check the core out at exactly the commit the benchmark installs, so a
study can never run against a different core than the benchmark declares:

    python tools/pinned_core.py            # core_repo=<owner>/<name> and core_sha=<40 hex>
    python tools/pinned_core.py --pyproject path/to/pyproject.toml

The output is ``key=value`` lines, ready to append to ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

PIN = re.compile(
    r"^polynomial-cl\s*@\s*git\+https://github\.com/"
    r"(?P<repo>[\w.-]+/[\w.-]+?)(?:\.git)?@(?P<sha>[0-9a-f]{40})$"
)


def pinned_core(pyproject: Path) -> tuple[str, str]:
    """``(owner/name, sha)`` of the core pinned in the project's dependencies."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    pins = [m for d in project["dependencies"] if (m := PIN.match(d.strip()))]
    if len(pins) != 1:
        raise ValueError(
            f"{pyproject} must pin the core exactly once as "
            "'polynomial-cl @ git+https://github.com/<owner>/<name>.git@<40-hex sha>'; "
            f"found {len(pins)}"
        )
    return pins[0]["repo"], pins[0]["sha"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
    )
    args = parser.parse_args(argv)
    try:
        repo, sha = pinned_core(args.pyproject)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"core_repo={repo}")
    print(f"core_sha={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
