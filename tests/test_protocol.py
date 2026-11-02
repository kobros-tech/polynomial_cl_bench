import copy
import json
import re
import unittest
from dataclasses import replace

from helpers import PROTOCOL_PATH, ROOT, shipped_protocol
from polynomial_cl_bench.protocol import Protocol, ProtocolError


def raw():
    return json.loads(PROTOCOL_PATH.read_text())


class ProtocolTests(unittest.TestCase):
    def test_shipped_protocol_values(self):
        p = shipped_protocol()
        self.assertEqual(p.task_order, ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)))
        self.assertEqual(p.n_experiences, 5)
        self.assertEqual(p.projection_seed, 0)
        self.assertEqual(p.sampling, "class_balanced")
        self.assertEqual((p.epochs, p.batch_size, p.memory_per_class), (3, 128, 200))

    def test_round_trip_and_fingerprint(self):
        p = shipped_protocol()
        self.assertEqual(Protocol.from_dict(p.to_dict()), p)
        self.assertEqual(p.fingerprint(), shipped_protocol().fingerprint())
        self.assertNotEqual(p.fingerprint(), replace(p, sampling="shuffled").fingerprint())
        self.assertNotEqual(p.fingerprint(), replace(p, projection_seed=1).fingerprint())

    def test_protocol_has_no_algorithm_settings(self):
        text = json.dumps(raw())
        for word in ("learning_rate", "strategy", "condition", "degree"):
            self.assertNotIn(word, text)

    def test_invalid_configurations_rejected(self):
        def broken(mutate):
            r = copy.deepcopy(raw())
            mutate(r)
            with self.assertRaises(ProtocolError):
                Protocol.from_dict(r)

        broken(lambda r: r.update(task_order=[[0, 1], [1, 2], [3, 4], [5, 6], [7, 8]]))
        broken(lambda r: r.update(task_order=[[0, 1], [2, 3]]))  # classes missing
        broken(lambda r: r.update(task_order=[[0, 1], [], [2, 3, 4, 5, 6, 7, 8, 9]]))
        broken(lambda r: r.update(seeds=[]))
        broken(lambda r: r.update(seeds=[0, 0]))
        broken(lambda r: r["training"].update(sampling="random"))
        broken(lambda r: r["training"].update(epochs=0))
        broken(lambda r: r["memory"].update(examples_per_class=0))
        broken(lambda r: r["evaluation"].update(schedule="never"))
        broken(lambda r: r["dataset"]["files"].update({"x.gz": "abc"}))
        broken(lambda r: r["validation"].update(holdout_fraction=1.0))
        broken(lambda r: r.pop("seeds"))

    def test_core_pin_is_a_full_commit_hash(self):
        text = (ROOT / "pyproject.toml").read_text()
        match = re.search(r"polynomial_cl_demo\.git@(\S+?)\"", text)
        self.assertIsNotNone(match)
        self.assertRegex(match.group(1), r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
