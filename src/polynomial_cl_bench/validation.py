# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Experiment-level validation: is this set of records a protocol-compliant comparison?

Two outputs, deliberately separate:

* ``problems``: integrity violations (a record does not match the manifest, algorithms did not
  receive the same plan at some seed, two commits mixed, a core cross-check disagrees, a run is
  missing or unexpected, a file is unreadable). A non-empty list means the results must not be
  used.
* ``publication_blockers``: reasons a clean, integral result set is still not a publishable
  comparison (development mode, untuned configurations, failed runs, test split not used,
  reproducibility not demonstrated, ...). Honest status, not an error.

Nothing here claims an algorithm is better than another.
"""

from __future__ import annotations

from . import results

PROTOCOL_SEEDS_KEY = "seeds"


def _single(values) -> bool:
    return len(set(map(repr, values))) <= 1


def validate_experiment(
    manifest: dict, records: list, invalid: list, reproducibility: list
) -> dict:
    """``records``: validated run records; ``invalid``: ``(filename, problem)`` for the rest."""
    problems, warnings, blockers = [], [], []
    spec = manifest["spec"]
    labels = [a["label"] for a in spec["algorithms"]]
    configs = {a["label"]: a["config"] for a in spec["algorithms"]}
    from .protocol import Protocol

    protocol = Protocol.from_dict(spec["protocol"])
    protocol_fp = protocol.fingerprint()
    split = spec["evaluation_split"]
    identity = results.dataset_identity(protocol, split)

    problems += [f"{name}: {why}" for name, why in invalid]

    expected = {(r["label"], r["seed"]) for r in manifest["expected_runs"]}
    seen: dict = {}
    for record in records:
        key = (record["algorithm"]["label"], record["seed"])
        if key in seen:
            problems.append(f"duplicate record for {key}")
        seen[key] = record
    for key in sorted(expected - set(seen) - {n for n in _invalid_keys(invalid)}):
        problems.append(f"expected run {key[0]} seed {key[1]} has no record")
    for key in sorted(set(seen) - expected):
        problems.append(f"unexpected record {key[0]} seed {key[1]}")

    for key, record in seen.items():
        name = f"{key[0]} seed {key[1]}"
        if key[0] in configs:
            expected_key = results.run_key(
                {"config": configs[key[0]]}, protocol_fp, identity, key[1]
            )
            if record["run_key"] != expected_key:
                problems.append(
                    f"{name}: run_key does not match the manifest's configuration"
                )
            if record["algorithm"]["config"] != configs[key[0]]:
                problems.append(
                    f"{name}: algorithm configuration differs from the manifest"
                )
        if record["protocol"]["fingerprint"] != protocol_fp:
            problems.append(f"{name}: protocol fingerprint differs from the manifest")
        if record["dataset"]["identity"] != identity:
            problems.append(f"{name}: dataset identity differs from the manifest")
        if (record.get("experiment") or {}).get("fingerprint") != manifest[
            "fingerprint"
        ]:
            problems.append(
                f"{name}: record belongs to a different experiment fingerprint"
            )
        budget = record["training_budget"]
        if (budget["epochs"], budget["batch_size"], budget["sampling"]) != (
            protocol.epochs,
            protocol.batch_size,
            protocol.sampling,
        ):
            problems.append(f"{name}: training budget differs from the protocol")
        if record["memory"]["examples_per_class"] != protocol.memory_per_class:
            problems.append(f"{name}: memory budget differs from the protocol")
        cross = record["checks"].get("core_reported_cross_check")
        if cross is not None and not cross["agrees"]:
            problems.append(
                f"{name}: metrics disagree with the algorithm's own report "
                f"(accuracy diff {cross['final_accuracy_abs_diff']:.3g}, "
                f"forgetting diff {cross['mean_forgetting_abs_diff']:.3g})"
            )

    completed = [r for r in seen.values() if r["status"] == "completed"]
    if not _single(r["dataset"]["fingerprint"] for r in completed):
        problems.append("completed runs used different dataset array fingerprints")
    for seed in sorted({r["seed"] for r in completed}):
        plans = {
            r["algorithm"]["label"]: r["plan_fingerprint"]
            for r in completed
            if r["seed"] == seed
        }
        if not _single(plans.values()):
            problems.append(
                f"seed {seed}: algorithms received different experience/replay plans {plans}"
            )
    commits = {
        (
            r["environment"].get("core_commit"),
            r["environment"].get("core_commit_source"),
            r["environment"].get("benchmark_commit"),
            r["environment"].get("benchmark_commit_source"),
        )
        for r in seen.values()
    }
    if len(commits) > 1:
        problems.append(
            f"records were produced by different core/benchmark commits: {sorted(map(str, commits))}"
        )
    manifest_env = manifest.get("environment", {})
    if (
        records
        and commits
        and (
            manifest_env.get("core_commit"),
            manifest_env.get("core_commit_source"),
            manifest_env.get("benchmark_commit"),
            manifest_env.get("benchmark_commit_source"),
        )
        not in commits
    ):
        problems.append("records' commits differ from those recorded in the manifest")

    # --- blockers ----------------------------------------------------------------------
    if (
        any(r["environment"]["run_mode"] != "official" for r in seen.values())
        or spec["mode"] != "official"
    ):
        blockers.append("run mode is not 'official'")
    official_problems = sorted(
        {p for r in seen.values() for p in r["environment"]["official_problems"]}
    )
    if official_problems:
        blockers.append("provenance problems: " + "; ".join(official_problems))
    not_done = sorted(
        f"{k[0]} seed {k[1]}" for k, r in seen.items() if r["status"] != "completed"
    )
    if not_done:
        blockers.append(f"{len(not_done)} run(s) did not complete: {not_done}")
    if split != "test":
        blockers.append(f"evaluation split is {split!r}, not 'test'")
    if tuple(spec["seeds"]) != tuple(spec["protocol"]["seeds"]):
        blockers.append("experiment seeds differ from the protocol's seeds")
    for label in labels:
        provenance = configs[label].get("hyperparameter_provenance") or {}
        if provenance.get("selection") != "validation_tuning":
            blockers.append(
                f"{label}: hyper-parameters were not selected by validation tuning "
                f"(selection={provenance.get('selection')!r}); comparison budgets are not equal"
            )
        elif provenance.get("tuning_official_eligible") is not True:
            blockers.append(f"{label}: its tuning result is not official-eligible")
    if not reproducibility:
        blockers.append("reproducibility check was not run (--reproduce-check)")
    for item in reproducibility:
        if not item["exact_match"]:
            blockers.append(
                f"{item['label']} seed {item['seed']}: rerun did not reproduce the matrix exactly "
                f"(max abs difference {item['max_abs_difference']}); pin thread counts and retry"
            )
    if records and all(
        v is None for v in records[0]["environment"]["thread_env"].values()
    ):
        warnings.append(
            "BLAS/OpenMP thread counts are not pinned; timings and bit-exact reproducibility "
            "may vary between machines"
        )
    return {
        "valid": not problems,
        "problems": problems,
        "warnings": warnings,
        "publication_blockers": blockers,
        "publishable": not problems and not blockers,
        "checks_performed": [
            "every record re-validated (run key, metrics recomputed from the matrix, memory <= budget)",
            "every expected algorithm x seed present, none unexpected",
            "records match the manifest (configuration, protocol, dataset identity, budgets, experiment fingerprint)",
            "identical plan fingerprint for every algorithm at each seed",
            "identical dataset array fingerprint across completed runs",
            "single core/benchmark commit across records, equal to the manifest's",
            "algorithm-reported metrics agree with benchmark-computed metrics where reported",
        ],
    }


def _invalid_keys(invalid) -> list:
    """Run keys recoverable from invalid file names (``<label>__seed<k>.json``)."""
    keys = []
    for name, _ in invalid:
        stem = name.removesuffix(".json")
        label, _, seed = stem.rpartition("__seed")
        if label and seed.isdigit():
            keys.append((label, int(seed)))
    return keys
