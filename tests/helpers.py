# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import gzip
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np

from polynomial_cl_bench.datasets import SplitMNIST, array_fingerprint
from polynomial_cl_bench.protocol import Protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "split_mnist_cil_v1.json"
POLYNOMIAL_CONFIG_PATH = ROOT / "configs" / "polynomial_expanding_ordinary.json"
PER_DEGREE_CONFIG_PATH = (
    ROOT / "configs" / "polynomial_expanding_ordinary_per_degree.json"
)


def unverified_core_provenance() -> dict:
    """Core provenance as recorded, forced to carry an official-run problem.

    Tests of the official-mode gate must state their own premise instead of inheriting the
    machine's: on a clean checkout with the core installed from the pin, the real gate passes.
    """
    from polynomial_cl_bench import provenance

    core = provenance.core_provenance()
    problems = [*core["official_problems"], "forced by test: remote not verified"]
    return {**core, "official_problems": problems}


def shipped_protocol() -> Protocol:
    return Protocol.from_file(PROTOCOL_PATH)


def small_protocol(**changes) -> Protocol:
    """The shipped protocol with a tiny budget, for fast tests."""
    base = {"epochs": 1, "batch_size": 32, "memory_per_class": 10, "seeds": (0, 1)}
    base.update(changes)
    return replace(shipped_protocol(), **base)


def synthetic_dataset(seed=0, train_per_class=30, test_per_class=20) -> SplitMNIST:
    """Ten separable Gaussian classes in the 16 projected dimensions."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 2.0, size=(10, 16))

    def draw(n):
        x = np.concatenate([rng.normal(means[c], 3.0, size=(n, 16)) for c in range(10)])
        return x.astype(np.float32), np.repeat(np.arange(10), n).astype(np.uint8)

    train_x, train_y = draw(train_per_class)
    test_x, test_y = draw(test_per_class)
    for array in (train_x, train_y, test_x, test_y):
        array.setflags(write=False)
    return SplitMNIST(
        train_x,
        train_y,
        test_x,
        test_y,
        array_fingerprint(train_x, train_y, test_x, test_y),
        projection_seed=shipped_protocol().projection_seed,
        source_hashes=dict(shipped_protocol().dataset["files"]),
    )


def write_idx_images(path, images):
    n = len(images)
    with gzip.open(path, "wb") as f:
        f.write(
            struct.pack(">IIII", 2051, n, 28, 28) + images.astype(np.uint8).tobytes()
        )


def write_idx_labels(path, labels):
    with gzip.open(path, "wb") as f:
        f.write(
            struct.pack(">II", 2049, len(labels)) + labels.astype(np.uint8).tobytes()
        )


# --- Phase 3 (runner) helpers: core-free ---------------------------------------------------

import hashlib
from dataclasses import replace as _replace

MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def fake_mnist_protocol(root: Path, **changes) -> Protocol:
    """Tiny valid-format MNIST archives under ``root`` and a protocol that pins their hashes."""
    rng = np.random.default_rng(0)
    write_idx_images(root / MNIST_FILES[0], rng.integers(0, 256, (200, 784)))
    write_idx_labels(root / MNIST_FILES[1], np.arange(200) % 10)
    write_idx_images(root / MNIST_FILES[2], rng.integers(0, 256, (100, 784)))
    write_idx_labels(root / MNIST_FILES[3], np.arange(100) % 10)
    hashes = {
        n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in MNIST_FILES
    }
    base = shipped_protocol()
    return _replace(
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
