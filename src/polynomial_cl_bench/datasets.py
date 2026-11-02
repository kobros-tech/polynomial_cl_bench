"""Split MNIST preparation: hash verification, then the core's own loader (never downloads)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import Protocol


class DatasetVerificationError(RuntimeError):
    """A dataset file is missing or does not match the protocol's hash."""


@dataclass(frozen=True)
class SplitMNIST:
    """Projected arrays shared (read-only) by every algorithm and seed."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    fingerprint: str  # sha256 over the four arrays; recorded in result records
    projection_seed: int  # seed the arrays were projected with (checked by adapters)
    source_hashes: dict  # archive name -> sha256 that was verified for these arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mnist_files(protocol: Protocol, data_dir) -> None:
    """Raise unless every archive named in the protocol exists with the expected sha256."""
    root = Path(data_dir)
    problems = []
    for name, expected in protocol.dataset["files"].items():
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif (actual := _sha256(path)) != expected:
            problems.append(f"hash mismatch: {name} (expected {expected}, got {actual})")
    if problems:
        raise DatasetVerificationError(
            f"MNIST files under {root} failed verification:\n  " + "\n  ".join(problems)
        )


def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode() + str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_split_mnist(protocol: Protocol, data_dir) -> SplitMNIST:
    """Verify the archives, then load and project them with the core's loader.

    The core loader is called with ``download=False`` so the benchmark never touches the
    network and never uses an unverified file. The archives are hashed before loading and again
    after it, so a file replaced between verification and reading is detected.
    """
    from polynomial_cl.class_incremental import load_projected_mnist
    from polynomial_cl.class_incremental.data import N_COMPONENTS

    if protocol.n_components != N_COMPONENTS:
        raise DatasetVerificationError(
            f"protocol n_components={protocol.n_components} but the core loader "
            f"projects to {N_COMPONENTS}"
        )
    verify_mnist_files(protocol, data_dir)
    train_x, train_y, test_x, test_y = load_projected_mnist(
        Path(data_dir), projection_seed=protocol.projection_seed, download=False
    )
    verify_mnist_files(protocol, data_dir)  # nothing changed while the core was reading
    for array in (train_x, train_y, test_x, test_y):
        array.setflags(write=False)  # shared by every run: guard against mutation
    return SplitMNIST(
        train_x, train_y, test_x, test_y,
        array_fingerprint(train_x, train_y, test_x, test_y),
        projection_seed=protocol.projection_seed,
        source_hashes=dict(protocol.dataset["files"]),
    )
