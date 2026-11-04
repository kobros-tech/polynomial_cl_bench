# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import hashlib
import tempfile
import unittest
import urllib.request
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from helpers import shipped_protocol, write_idx_images, write_idx_labels

from polynomial_cl_bench.datasets import (
    DatasetVerificationError,
    load_split_mnist,
    verify_mnist_files,
)

FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def make_fake_mnist(root: Path):
    rng = np.random.default_rng(0)
    write_idx_images(root / FILES[0], rng.integers(0, 256, (50, 784)))
    write_idx_labels(root / FILES[1], rng.integers(0, 10, 50))
    write_idx_images(root / FILES[2], rng.integers(0, 256, (20, 784)))
    write_idx_labels(root / FILES[3], rng.integers(0, 10, 20))
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in FILES}
    p = shipped_protocol()
    return replace(p, dataset={**p.dataset, "files": hashes})


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = make_fake_mnist(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_projected_arrays_read_only(self):
        data = load_split_mnist(self.protocol, self.root)
        self.assertEqual(data.train_x.shape, (50, 16))
        self.assertEqual(data.test_x.shape, (20, 16))
        self.assertFalse(data.train_x.flags.writeable)
        self.assertFalse(data.test_y.flags.writeable)

    def test_loading_is_deterministic_and_fingerprinted(self):
        a = load_split_mnist(self.protocol, self.root)
        b = load_split_mnist(self.protocol, self.root)
        self.assertEqual(a.fingerprint, b.fingerprint)
        other = load_split_mnist(replace(self.protocol, projection_seed=1), self.root)
        self.assertNotEqual(a.fingerprint, other.fingerprint)

    def test_hash_mismatch_is_rejected(self):
        with (self.root / FILES[0]).open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(DatasetVerificationError, "hash mismatch"):
            verify_mnist_files(self.protocol, self.root)
        with self.assertRaises(DatasetVerificationError):
            load_split_mnist(self.protocol, self.root)

    def test_missing_file_never_triggers_a_download(self):
        (self.root / FILES[2]).unlink()
        with (
            mock.patch.object(
                urllib.request, "urlopen", side_effect=AssertionError("network used")
            ),
            self.assertRaisesRegex(DatasetVerificationError, "missing"),
        ):
            load_split_mnist(self.protocol, self.root)

    def test_shipped_hash_table_is_complete(self):
        # Structure only; the hashes themselves are checked against real files in test_full_split_mnist.
        files = shipped_protocol().dataset["files"]
        self.assertEqual(set(files), set(FILES))
        self.assertEqual(len(set(files.values())), 4)

    def test_protocol_names_are_exactly_the_files_the_core_loader_reads(self):
        from polynomial_cl.class_incremental import mnist

        self.assertEqual(
            set(shipped_protocol().dataset["files"]), set(mnist.FILES.values())
        )

    def test_every_file_the_core_reads_was_hashed_against_the_protocol(self):
        from polynomial_cl.class_incremental import mnist

        reads = []
        real = mnist.read_idx

        def spy(path, image):
            # Hash the file at the moment the core opens it.
            reads.append(
                (Path(path).name, hashlib.sha256(Path(path).read_bytes()).hexdigest())
            )
            return real(path, image)

        with (
            mock.patch.object(mnist, "read_idx", spy),
            mock.patch.object(
                mnist, "fetch", side_effect=AssertionError("download attempted")
            ),
        ):
            load_split_mnist(self.protocol, self.root)
        self.assertEqual(sorted(n for n, _ in reads), sorted(FILES))
        for name, digest in reads:
            self.assertEqual(digest, self.protocol.dataset["files"][name])

    def test_file_replaced_while_loading_is_detected(self):
        from polynomial_cl.class_incremental import mnist

        real = mnist.read_idx
        state = {"done": False}

        def swap_after_first_read(path, image):
            out = real(path, image)
            if not state["done"]:
                state["done"] = True
                # Modify the archive the core has just finished reading.
                with Path(path).open("ab") as f:
                    f.write(b"tamper")
            return out

        with (
            mock.patch.object(mnist, "read_idx", swap_after_first_read),
            self.assertRaisesRegex(DatasetVerificationError, "hash mismatch"),
        ):
            load_split_mnist(self.protocol, self.root)

    def test_loaded_dataset_records_its_projection_seed_and_hashes(self):
        data = load_split_mnist(self.protocol, self.root)
        self.assertEqual(data.projection_seed, 0)
        self.assertEqual(data.source_hashes, self.protocol.dataset["files"])


if __name__ == "__main__":
    unittest.main()
