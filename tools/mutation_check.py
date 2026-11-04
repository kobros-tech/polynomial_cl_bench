# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Mutation check of the test suite: break the source in one plausible way at a time and
confirm the relevant tests fail.

    python3 tools/mutation_check.py

Every source file is restored afterwards (and verified by hash). A mutation is **caught** if
the named test modules fail. A mutation marked ``equivalent`` must be shown to leave the
behaviour unchanged (the output of the original and mutated code is compared on many inputs);
it is reported separately and is not counted as caught or missed.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "polynomial_cl_bench"
TESTS = ROOT / "tests"

SAMPLING = "sampling.py"
TUNING = "tuning.py"
ER = "algorithms/er_mlp.py"
PROV = "provenance.py"
T_SAMPLING = ["test_sampling_equivalence"]
T_TUNING = ["test_plan_and_tuning"]
T_ER = ["test_er_mlp", "test_sampling_equivalence"]
T_PROV = ["test_provenance_official"]

# (id, description, file, old, new, test modules, kind)
M = [
    (
        "S1",
        "bonus slots always go to the first classes",
        SAMPLING,
        "bonus[rng.choice(len(classes), extra, replace=False)] = 1",
        "bonus[:extra] = 1",
        T_SAMPLING,
        "real",
    ),
    (
        "S2",
        "bonus classes drawn with replacement",
        SAMPLING,
        "rng.choice(len(classes), extra, replace=False)",
        "rng.choice(len(classes), extra, replace=True)",
        T_SAMPLING,
        "real",
    ),
    (
        "S3",
        "class cycle not reshuffled when exhausted",
        SAMPLING,
        "cycle[c] = pools[c][rng.permutation(len(pools[c]))]\n                    cursor[c] = 0",
        "cursor[c] = 0",
        T_SAMPLING,
        "real",
    ),
    (
        "S4",
        "initial class cycle not shuffled",
        SAMPLING,
        "cycle[c], cursor[c] = pool[rng.permutation(len(pool))], 0",
        "cycle[c], cursor[c] = pool, 0",
        T_SAMPLING,
        "real",
    ),
    (
        "S5",
        "floor instead of ceil updates per epoch",
        SAMPLING,
        "return -(-n_rows // batch_size)",
        "return n_rows // batch_size",
        T_SAMPLING,
        "real",
    ),
    (
        "S6",
        "shuffled policy: no per-epoch reshuffle",
        SAMPLING,
        "order = rng.permutation(n_rows)",
        "order = np.arange(n_rows)",
        T_SAMPLING,
        "real",
    ),
    (
        "S7",
        "short class is not repeated (batch left incomplete)",
        SAMPLING,
        "c, need = int(c), base + int(more)",
        "c, need = int(c), min(base + int(more), len(pools[int(c)]))",
        T_SAMPLING,
        "real",
    ),
    (
        "S8",
        "take at most one example per loop pass (same draws, chunked differently)",
        SAMPLING,
        "take = min(need, len(cycle[c]) - cursor[c])",
        "take = min(need, len(cycle[c]) - cursor[c], 1)",
        T_SAMPLING,
        "equivalent",
    ),
    (
        "T1",
        "tuning evaluates on the real dataset, not the validation view",
        TUNING,
        "run_sequence(view, protocol, seed, plans[seed])",
        "run_sequence(dataset, protocol, seed, plans[seed])",
        T_TUNING,
        "real",
    ),
    (
        "T2",
        "forgetting tie-break reversed (prefers MORE forgetting)",
        TUNING,
        'configs[i]["mean_forgetting"], -configs[i]["mean_final_accuracy"], i))',
        '-configs[i]["mean_forgetting"], -configs[i]["mean_final_accuracy"], i))',
        T_TUNING,
        "real",
    ),
    (
        "T3",
        "no margin: only the exact maximum is a candidate",
        TUNING,
        "best - margin - 1e-12",
        "best - 1e-12",
        T_TUNING,
        "real",
    ),
    (
        "T4",
        "margin doubled to 0.01",
        TUNING,
        "best - margin - 1e-12",
        "best - 2 * margin - 1e-12",
        T_TUNING,
        "real",
    ),
    (
        "T5",
        "margin boundary made exclusive",
        TUNING,
        ">= best - margin - 1e-12",
        "> best - margin",
        T_TUNING,
        "real",
    ),
    (
        "T6",
        "failed runs not recorded",
        TUNING,
        '                runs.append(base | {\n                    "status": _status_for(error),',
        '                continue\n                runs.append(base | {\n                    "status": _status_for(error),',
        T_TUNING,
        "real",
    ),
    (
        "T7",
        "configuration with a failed seed still eligible",
        TUNING,
        'eligible = all(r["status"] == "completed" for r in mine)',
        'eligible = any(r["status"] == "completed" for r in mine)',
        T_TUNING,
        "real",
    ),
    (
        "T8",
        "ProvenanceError swallowed as a failed run",
        TUNING,
        "            except ProvenanceError:\n                raise",
        "            except KeyboardInterrupt:\n                raise",
        T_TUNING,
        "real",
    ),
    (
        "T9",
        "selection seeds may overlap the evaluation seeds",
        TUNING,
        "    if overlap:",
        "    if False:",
        T_TUNING,
        "real",
    ),
    (
        "T10",
        "plan rebuilt per configuration instead of shared",
        TUNING,
        "plans[seed])\n                matrix",
        "make_plan(view.train_y, seed, protocol.memory_per_class))\n                matrix",
        T_TUNING,
        "real",
    ),
    (
        "E1",
        "bias gradient uses mean instead of sum",
        ER,
        "grad_b[layer] = d_z.sum(axis=0)",
        "grad_b[layer] = d_z.mean(axis=0)",
        T_ER,
        "real",
    ),
    (
        "E2",
        "evaluation lets future classes compete",
        ER,
        "task_order[: step + 1] for c in task",
        "task_order[:] for c in task",
        T_ER,
        "real",
    ),
    (
        "E3",
        "replay rows placed before the current rows",
        ER,
        "features = np.concatenate((features, replay_features), axis=0)",
        "features = np.concatenate((replay_features, features), axis=0)",
        T_ER,
        "real",
    ),
    (
        "E4",
        "SGD step is half the learning rate",
        ER,
        "w -= learning_rate * g",
        "w -= 0.5 * learning_rate * g",
        T_ER,
        "real",
    ),
    (
        "E5",
        "per-experience seed ignores the experience index",
        ER,
        "seed=seed + task,",
        "seed=seed,",
        T_ER,
        "real",
    ),
    (
        "E6",
        "ablation flag ignored: replay always used",
        ER,
        "if self.replay else np.empty(0, dtype=np.int64)",
        "if True else np.empty(0, dtype=np.int64)",
        T_ER,
        "real",
    ),
    (
        "P1",
        "untracked files do not make a checkout dirty",
        PROV,
        '_git(directory, "status", "--porcelain")',
        '_git(directory, "status", "--porcelain", "--untracked-files=no")',
        T_PROV,
        "real",
    ),
    (
        "P2",
        "any remote URL counts as the pinned remote",
        PROV,
        "if pinned_url and normalize_remote_url(url) == normalize_remote_url(pinned_url)",
        "if True",
        T_PROV,
        "real",
    ),
    (
        "P3",
        "benchmark problems ignored by the official-run gate",
        PROV,
        'problems += [f"benchmark: {p}" for p in benchmark["official_problems"]]',
        "pass",
        T_PROV,
        "real",
    ),
]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tests(modules):
    result = subprocess.run(
        [sys.executable, "-m", "unittest", *modules],
        cwd=TESTS,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr.strip().splitlines()[
        -1
    ] if result.stderr else ""


def _load(source: str) -> types.ModuleType:
    module = types.ModuleType("sampling_variant")
    exec(compile(source, "sampling_variant", "exec"), module.__dict__)  # noqa: S102
    return module


def prove_equivalent(original: str, mutated: str) -> bool:
    """Original and mutated sampler give identical batches and generator state on many inputs."""
    a, b = _load(original), _load(mutated)
    cases = [
        ([20] * 8 + [300, 310], 128, 25),
        ([3, 200, 200], 128, 40),
        ([2] * 5, 128, 10),
        ([10] * 10, 7, 30),
        ([37], 16, 9),
        ([25, 31], 33, 20),
    ]
    for counts, batch, updates in cases:
        labels = np.repeat(np.arange(len(counts)), counts)
        for seed in range(5):
            ra, rb = np.random.default_rng(seed), np.random.default_rng(seed)
            xa = list(a.class_balanced_batches(labels, updates, batch, ra))
            xb = list(b.class_balanced_batches(labels, updates, batch, rb))
            if len(xa) != len(xb) or not all(
                np.array_equal(p, q) for p, q in zip(xa, xb)
            ):
                return False
            if ra.bit_generator.state != rb.bit_generator.state:
                return False
    return True


def main() -> int:
    caught = missed = equivalent = 0
    originals = {f: _digest(SRC / f) for f in {m[2] for m in M}}
    print(f"{'id':4s} {'verdict':22s} description")
    try:
        for mid, text, file, old, new, modules, kind in M:
            path = SRC / file
            source = path.read_text()
            if old not in source:
                raise SystemExit(f"{mid}: mutation target not found in {file}")
            mutated = source.replace(old, new, 1)
            if kind == "equivalent":
                ok = prove_equivalent(source, mutated)
                verdict = "EQUIVALENT (proved)" if ok else "NOT EQUIVALENT: reclassify"
                equivalent += ok
                missed += not ok
            else:
                path.write_text(mutated)
                try:
                    passed, _ = run_tests(modules)
                finally:
                    path.write_text(source)
                verdict = "MISSED" if passed else "caught"
                caught += not passed
                missed += passed
            print(f"{mid:4s} {verdict:22s} {text}")
    finally:
        for f, digest in originals.items():
            assert _digest(SRC / f) == digest, f"{f} was not restored"
    print(
        f"\n{len(M)} mutations: {caught} caught, {equivalent} proved equivalent, {missed} missed"
    )
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
