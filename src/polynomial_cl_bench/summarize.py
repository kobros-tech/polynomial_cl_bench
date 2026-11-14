# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Validate and aggregate an experiment directory; write ``summary.json`` and ``summary.md``.

    python -m polynomial_cl_bench.summarize --dir results/tuned_comparison

Reads only records (never reruns anything). Unreadable or invalid record files are reported as
problems and counted as missing, never skipped silently. The exit status is non-zero when
integrity problems exist; publication blockers are reported but are not errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import aggregate, results
from .validation import validate_experiment


def load_directory(out_dir) -> tuple:
    """``(manifest, valid records, [(filename, problem)], reproducibility entries)``."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{out_dir} has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    records, invalid = [], []
    for path in sorted((out_dir / "runs").glob("*.json")):
        if ".attempt" in path.name:  # archived earlier attempts, kept for history
            continue
        try:
            records.append(results.read_record(path))
        except results.RecordError as error:
            invalid.append((path.name, str(error)))
    reproducibility = [
        json.loads(p.read_text())
        for p in sorted((out_dir / "reproducibility").glob("*.json"))
    ]
    return manifest, records, invalid, reproducibility


def summarize_directory(out_dir, write: bool = True) -> dict:
    manifest, records, invalid, reproducibility = load_directory(out_dir)
    spec = manifest["spec"]
    labels = [a["label"] for a in spec["algorithms"]]
    validation = validate_experiment(manifest, records, invalid, reproducibility)
    aggregated = aggregate.aggregate_experiment(records, labels, spec["seeds"])
    summary = {
        "schema_version": results.SCHEMA_VERSION,
        "record_type": "experiment_summary",
        "experiment": {
            "name": manifest["name"],
            "fingerprint": manifest["fingerprint"],
        },
        "protocol": {
            "name": spec["protocol"]["name"],
            "version": spec["protocol"]["version"],
            "evaluation_split": spec["evaluation_split"],
        },
        "seeds": spec["seeds"],
        "aggregate": aggregated,
        "validation": validation,
        "reproducibility": reproducibility,
        "problems": validation["problems"],
    }
    summary["markdown"] = to_markdown(summary)
    if write:
        out = Path(out_dir)
        body = {k: v for k, v in summary.items() if k != "markdown"}
        results.write_json_atomic(out / "summary.json", body)
        (out / "summary.md").write_text(summary["markdown"] + "\n")
    return summary


def _fmt(stat, digits=4, scale=1.0) -> str:
    if not stat or stat["mean"] is None:
        return "—"
    text = f"{stat['mean'] * scale:.{digits}f}"
    if stat["std"] is not None:
        text += f" ± {stat['std'] * scale:.{digits}f}"
    return text


def _ci(stat, digits=4) -> str:
    if not stat or not stat.get("ci95"):
        return "—"
    low, high = stat["ci95"]
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def to_markdown(summary: dict) -> str:
    agg = summary["aggregate"]
    exp, proto = summary["experiment"], summary["protocol"]
    lines = [
        f"# {exp['name']}",
        "",
        (
            f"Protocol {proto['name']} v{proto['version']}, evaluated on the **{proto['evaluation_split']}** "
            f"split; seeds {summary['seeds']}. Mean ± sample std over completed runs; the interval is a "
            "95% t-interval for the mean. Descriptive statistics only: seeds do not vary the data split."
        ),
        "",
        "| algorithm | completed / failed / missing | final accuracy | 95% CI | mean forgetting | wall s | train s | infer s | features s | peak MB | parameters |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, a in agg["algorithms"].items():
        m, r = a["metrics"], a["resources"]
        lines.append(
            f"| {label} | {a['n_completed']} / {a['n_failed']} / {a['n_missing']} "
            f"| {_fmt(m['final_accuracy'])} | {_ci(m['final_accuracy'])} | {_fmt(m['mean_forgetting'])} "
            f"| {_fmt(r['wall_time_seconds'], 1)} | {_fmt(r['training_time_seconds'], 1)} "
            f"| {_fmt(r['inference_time_seconds'], 2)} "
            f"| {_fmt(r['feature_construction_seconds'], 2)} | {_fmt(r['peak_memory_mb'], 0)} "
            f"| {_fmt(r['n_parameters'], 0)} |"
        )
    lines += [
        "",
        (
            "`—` means not available (null), never zero. Wall time is measured by the runner around the "
            "whole run for every algorithm and includes in-loop evaluation; train/infer seconds are "
            "adapter-reported and unavailable for algorithms that cannot separate them."
        ),
    ]
    failed = [
        (label, f) for label, a in agg["algorithms"].items() for f in a["failures"]
    ]
    missing = [
        (label, s) for label, a in agg["algorithms"].items() for s in a["missing_seeds"]
    ]
    if failed or missing:
        lines += ["", "## Runs without a result"]
        lines += [
            f"- {l} seed {f['seed']}: **{f['status']}** — {f['error']}"
            for l, f in failed
        ]
        lines += [f"- {l} seed {s}: **missing** (no record)" for l, s in missing]
    if agg["paired_comparisons"]:
        lines += [
            "",
            "## Paired comparisons (a − b, same seed)",
            "",
            "| a | b | metric | pairs | mean diff | 95% CI | a higher / b higher / tie |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in agg["paired_comparisons"]:
            ci = (
                "—"
                if not c["ci95_mean_difference"]
                else f"[{c['ci95_mean_difference'][0]:.4f}, {c['ci95_mean_difference'][1]:.4f}]"
            )
            mean = (
                "—" if c["mean_difference"] is None else f"{c['mean_difference']:+.4f}"
            )
            lines.append(
                f"| {c['a']} | {c['b']} | {c['metric']} | {c['n_pairs']} | {mean} | {ci} "
                f"| {c['a_higher']} / {c['b_higher']} / {c['ties']} |"
            )
        lines += ["", "_" + agg["paired_comparisons"][0]["qualification"] + "_"]
    v = summary["validation"]
    lines += ["", "## Validation", ""]
    lines.append("Integrity: **" + ("passed" if v["valid"] else "FAILED") + "**")
    lines += [f"- PROBLEM: {p}" for p in v["problems"]]
    lines.append("")
    lines.append(
        "Publishable comparison: **" + ("yes" if v["publishable"] else "no") + "**"
    )
    lines += [f"- blocker: {b}" for b in v["publication_blockers"]]
    lines += [f"- warning: {w}" for w in v["warnings"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True)
    args = parser.parse_args(argv)
    summary = summarize_directory(args.dir)
    print(summary["markdown"])
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
