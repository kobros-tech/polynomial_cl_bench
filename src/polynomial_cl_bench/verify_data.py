# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Check the MNIST archives against the protocol's file names and sha256 hashes.

    python -m polynomial_cl_bench.verify_data --data-dir data/mnist

Prints every required file with its expected and actual hash; exits 1 unless all four match.
The benchmark never downloads: the archives must be placed in the directory by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets import _sha256
from .protocol import Protocol

DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "configs" / "split_mnist_cil_v1.json"
)


def check(protocol: Protocol, data_dir) -> list:
    """One row per required archive: (name, expected, actual-or-None, status)."""
    rows = []
    for name, expected in protocol.dataset["files"].items():
        path = Path(data_dir) / name
        actual = _sha256(path) if path.is_file() else None
        status = (
            "missing"
            if actual is None
            else ("ok" if actual == expected else "HASH MISMATCH")
        )
        rows.append((name, expected, actual, status))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args(argv)
    rows = check(Protocol.from_file(args.protocol), args.data_dir)
    for name, expected, actual, status in rows:
        print(
            f"{status:14s} {name}\n{'':14s} expected sha256 {expected}\n{'':14s} actual   sha256 {actual}"
        )
    ok = all(r[3] == "ok" for r in rows)
    print(
        "ALL FOUR ARCHIVES VERIFIED"
        if ok
        else f"NOT READY: files must be in {args.data_dir}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
