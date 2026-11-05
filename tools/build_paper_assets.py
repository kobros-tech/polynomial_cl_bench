# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Turn validated result files into the paper's tables, figures and number macros (Phase 5).

Consumes only files written by ``analyze_per_degree_study.py`` and ``run_comparison_study.py``
and the core's stored cell results; it never trains anything. Every number in the paper's text,
tables and figures comes from here (``paper/generated/numbers.tex`` holds the macros).

    python tools/build_paper_assets.py --analysis results/analysis/per_degree_analysis.json \\
        --comparison results/comparison --out paper
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
STRATEGY_COLOURS = {
    "ordinary": "#4C72B0",
    "ordinary_no_replay": "#999999",
    "small_old_lr": "#DD8452",
    "frozen_old": "#55A868",
    "strategy_b": "#C44E52",
    "strategy_b_new": "#8172B3",
}
T_CRIT_95 = {
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
}


# ------------------------------------------------------------------ formatting helpers


def esc(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tt(text):
    return r"\texttt{" + esc(text) + "}"


def pm(mean, std, digits=1, scale=100.0):
    if mean is None:
        return "--"
    s = f"{scale * mean:.{digits}f}"
    return s if std is None else rf"{s}{{\scriptsize$\pm${scale * std:.{digits}f}}}"


def seconds(x, digits=2):
    return "--" if x is None else f"{x:.{digits}f}"


def lr_tex(vector):
    if not isinstance(vector, (list, tuple)):
        return f"{vector:g}"
    return "[" + ", ".join(f"{v:g}" for v in vector) + "]"


def macro_name(*parts):
    words = []
    for part in parts:
        words += re.findall(r"[A-Za-z]+", str(part).replace("_", " "))
    return "num" + "".join(w[:1].upper() + w[1:] for w in words)


class Macros:
    def __init__(self):
        self.items = {}

    def add(self, name, value):
        self.items[name] = value

    def tex(self):
        return (
            "\n".join(
                f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(self.items.items())
            )
            + "\n"
        )


# ------------------------------------------------------------------ data access


def load_runs(path: Path):
    runs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            runs.setdefault((r["label"], r["split"], r["epochs"]), []).append(r)
    return runs


def ok_runs(runs, label, split="test", epochs=3):
    return sorted(
        (r for r in runs.get((label, split, epochs), []) if r["status"] == "completed"),
        key=lambda r: r["seed"],
    )


def poly_label(strategy, sampling, lr, condition="expanding"):
    text = (
        "[" + ",".join(f"{v:g}" for v in lr) + "]"
        if isinstance(lr, (list, tuple))
        else f"{lr:g}"
    )
    return f"poly|{condition}|{strategy}|{sampling}|lr{text}"


def summarise(rs):
    f = lambda k: np.array([r[k] for r in rs])
    sd = lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {
        "n": len(rs),
        "acc": float(f("final_accuracy").mean()),
        "acc_sd": sd(f("final_accuracy")),
        "forg": float(f("mean_forgetting").mean()),
        "forg_sd": sd(f("mean_forgetting")),
        "new": float(f("current_accuracy").mean()),
        "time": float(f("wall_seconds").mean()),
        "params": rs[0]["n_parameters"],
        "memory": (rs[0]["replay_examples_by_task"] or [None])[-1],
    }


def paired(a_runs, b_runs):
    by = {r["seed"]: r for r in b_runs}
    pairs = [(a, by[a["seed"]]) for a in a_runs if a["seed"] in by]
    for a, b in pairs:  # identical experiences and replay exemplars by construction
        assert a["plan_fingerprint"] == b["plan_fingerprint"], (
            "plans differ: not a paired comparison"
        )
    out = {"n": len(pairs)}
    for name, key, sign in (
        ("acc", "final_accuracy", 1),
        ("forg", "mean_forgetting", -1),
    ):
        d = np.array([a[key] - b[key] for a, b in pairs])
        half = T_CRIT_95[len(d) - 1] * d.std(ddof=1) / math.sqrt(len(d))
        out[name] = {
            "mean": float(d.mean()),
            "lo": float(d.mean() - half),
            "hi": float(d.mean() + half),
            "wins": int((sign * d > 0).sum()),
        }
    return out


# ------------------------------------------------------------------ main


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    report = json.loads(args.analysis.read_text())
    summary = json.loads((args.comparison / "summary.json").read_text())
    runs = load_runs(args.comparison / "runs.jsonl")
    out_gen, out_fig = args.out / "generated", args.out / "figures"
    out_gen.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    M = Macros()
    cands = report["candidates"]
    cand_by_id = {c["id"]: c for c in cands}
    er_cands = {c["id"]: c for c in summary["er_candidates"]}

    # ---- polynomial rows re-timed here, keyed by (strategy, sampling, lr, epochs)
    poly_here = {}
    for entry in summary["polynomial"]:
        sp = entry["spec"]
        key = (
            sp["strategy"],
            sp["sampling"],
            tuple(sp["learning_rate"]),
            entry["epochs"],
        )
        poly_here[key] = summarise(
            [r for r in entry["test_runs"] if r["status"] == "completed"]
        )

    def poly_row(w):
        key = (
            w["strategy"],
            w["sampling"],
            tuple(w["learning_rates_by_degree"]),
            w["epochs"],
        )
        return poly_here[key]

    # =============================================================== Table: selection criteria
    titles = {
        "max_accuracy": "Highest accuracy",
        "min_forgetting": "Lowest forgetting",
        "min_time": "Smallest time",
        "best_tradeoff": "Best trade-off",
    }
    lines = [
        r"\begin{tabular}{@{}llllrrrrr@{}}",
        r"\toprule",
        r"Criterion & Update rule & Sampling & $\eta$ (degrees 0--3) & Acc.\ (\%) & Forg.\ (\%) & Newest (\%) & $t_{\rm log}$ (s) & $t_{\rm here}$ (s) \\",
        r"\midrule",
    ]
    for basis, heading in (
        (
            "validation",
            r"\multicolumn{9}{@{}l}{\emph{Ranked on validation (selection seeds 0--2); test seeds 3--9 reported}}\\",
        ),
        (
            "test",
            r"\multicolumn{9}{@{}l}{\emph{Ranked on the test seeds themselves (hindsight; optimistic)}}\\",
        ),
    ):
        block = report["analysis"][basis]
        lines += [heading]
        groups = [
            ("", block["winners"], list(titles)),
            (
                " (floor)",
                block["accuracy_floor"]["winners"],
                ["min_forgetting", "min_time", "best_tradeoff"],
            ),
        ]
        for suffix, winners, order in groups:
            for crit in order:
                w = winners[crit]
                t = w["test_result"]
                here = poly_here.get(
                    (
                        w["strategy"],
                        w["sampling"],
                        tuple(w["learning_rates_by_degree"]),
                        w["epochs"],
                    )
                )
                ep = "" if w["epochs"] == 3 else rf" ({w['epochs']} ep.)"
                lines.append(
                    f"{titles[crit]}{suffix}{ep} & {tt(w['strategy'])} & {tt(w['sampling'])} & {lr_tex(w['learning_rates_by_degree'])} & "
                    f"{pm(t['accuracy_mean'], t['accuracy_std'])} & {pm(t['forgetting_mean'], t['forgetting_std'])} & "
                    f"{100 * t['current_accuracy_mean']:.1f} & {seconds(t['time_seconds_mean'])} & {seconds(here['time'] if here else None)} \\\\"
                )
                tag = f"{basis}{'Floor' if suffix else ''}{crit}"
                for k, v in (
                    ("Acc", 100 * t["accuracy_mean"]),
                    ("Forg", 100 * t["forgetting_mean"]),
                    ("New", 100 * t["current_accuracy_mean"]),
                ):
                    M.add(macro_name("Poly", tag, k), f"{v:.1f}")
                M.add(
                    macro_name("Poly", tag, "AccSd"), f"{100 * t['accuracy_std']:.1f}"
                )
                M.add(
                    macro_name("Poly", tag, "ForgSd"),
                    f"{100 * t['forgetting_std']:.1f}",
                )
                M.add(macro_name("Poly", tag, "Rule"), tt(w["strategy"]))
                M.add(
                    macro_name("Poly", tag, "Lr"), lr_tex(w["learning_rates_by_degree"])
                )
        lines.append(r"\midrule" if basis == "validation" else "")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_gen / "table_criteria.tex").write_text("\n".join(lines) + "\n")
    M.add(
        "numCandidatesValidation", str(report["analysis"]["validation"]["n_candidates"])
    )
    M.add("numCandidatesTest", str(report["analysis"]["test"]["n_candidates"]))
    M.add(
        "numParetoValidation",
        str(len(report["analysis"]["validation"]["pareto_front_ids"])),
    )
    M.add(
        "numFloorCandidates",
        str(report["analysis"]["validation"]["accuracy_floor"]["n_candidates"]),
    )
    for basis in ("validation", "test"):
        for name, s in report["analysis"][basis]["tradeoff_sensitivity"].items():
            M.add(
                macro_name("sens", basis, name),
                tt(s["candidate_id"].replace("|", " / ")),
            )

    # =============================================================== head-to-head table
    pv = report["analysis"]["validation"]
    pw, pf = pv["winners"], pv["accuracy_floor"]["winners"]
    er_an = summary["er_analysis"]

    def er_row(candidate_id):
        c = er_cands[candidate_id]
        t = c["test"]
        return {
            "acc": t["accuracy_mean"],
            "acc_sd": t["accuracy_std"],
            "forg": t["forgetting_mean"],
            "forg_sd": t["forgetting_std"],
            "new": t["current_accuracy_mean"],
            "time": t["time_seconds_mean"],
            "params": t["n_parameters"],
            "memory": t["replay_examples_final"],
            "spec": c["spec"],
        }

    def er_desc(candidate_id):
        sp = er_cands[candidate_id]["spec"]
        h = "linear head" if not sp["hidden"] else f"$H={sp['hidden'][0]}$"
        return f"{h}, $\\eta={sp['learning_rate']:g}$"

    rows = []  # (group, label, description, stats)
    rows.append(("Polynomial (expanding degrees $1,2,3,3,3$)", None, None, None))
    for crit_title, w in (
        ("Highest accuracy (val.)", pw["max_accuracy"]),
        ("Best trade-off (val., literal)", pw["best_tradeoff"]),
        ("Lowest forgetting (val., floor)", pf["min_forgetting"]),
    ):
        rows.append(
            (
                "p",
                crit_title,
                f"{tt(w['strategy'])}, $\\eta={lr_tex(w['learning_rates_by_degree'])}$",
                poly_row(w),
            )
        )
    for cond, title in (
        ("fixed_linear", "Degree 1 throughout (val.-tuned $\\eta$)"),
        ("fixed_capacity", "Degree 3 from the start (val.-tuned $\\eta$)"),
    ):
        a = summary["polynomial_ablations"].get(cond)
        if a:
            rows.append(
                (
                    "p",
                    title,
                    f"{tt('ordinary')}, $\\eta={a['spec']['learning_rate']:g}$",
                    summarise(
                        [r for r in a["test_runs"] if r["status"] == "completed"]
                    ),
                )
            )
    # no replay: the core's selected no-replay vector with class-balanced sampling, re-timed here
    nr = next(
        c
        for c in cands
        if c["strategy"] == "ordinary_no_replay"
        and c["sampling"] == "class_balanced"
        and c["selected_by_core"]
        and c["epochs"] == 3
    )
    nr_runs = ok_runs(
        runs,
        poly_label(
            "ordinary_no_replay", "class_balanced", nr["learning_rates_by_degree"]
        ),
    )
    if nr_runs:
        rows.append(
            (
                "p",
                "No replay (core-selected $\\eta$)",
                f"{tt('ordinary_no_replay')}, $\\eta={lr_tex(nr['learning_rates_by_degree'])}$",
                summarise(nr_runs),
            )
        )
    rows.append(("ER-MLP (one hidden ReLU layer, plain SGD, replay)", None, None, None))
    ev = er_an["validation"]
    for crit_title, cid in (
        ("Highest accuracy (val.)", ev["winners"]["max_accuracy"]["candidate_id"]),
        (
            "Best trade-off (val., literal)",
            ev["winners"]["best_tradeoff"]["candidate_id"],
        ),
        (
            "Best trade-off (val., floor)",
            ev["accuracy_floor"]["winners"]["best_tradeoff"]["candidate_id"],
        ),
        (
            "Lowest forgetting (val., floor)",
            ev["accuracy_floor"]["winners"]["min_forgetting"]["candidate_id"],
        ),
        (
            "Highest accuracy (hindsight, test)",
            er_an["test"]["winners"]["max_accuracy"]["candidate_id"],
        ),
    ):
        rows.append(("e", crit_title, er_desc(cid), er_row(cid)))
    lin = max(
        (c for c in er_cands.values() if not c["spec"]["hidden"]),
        key=lambda c: (
            c["validation"]["accuracy_mean"],
            -c["validation"]["forgetting_mean"],
        ),
    )
    rows.append(
        ("e", "Linear head (val.-tuned $\\eta$)", er_desc(lin["id"]), er_row(lin["id"]))
    )
    nr_er = summary["ten_epoch_and_ablation_runs"].get("_er_no_replay_3ep")
    if nr_er:
        s = summarise([r for r in nr_er["test_runs"] if r["status"] == "completed"])
        sp = nr_er["spec"]
        rows.append(
            (
                "e",
                "No replay",
                f"$H={sp['hidden'][0]}$, $\\eta={sp['learning_rate']:g}$",
                s,
            )
        )

    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Setting & Configuration & Params & Memory & Acc.\ (\%) & Forg.\ (\%) & Newest (\%) & Time (s) \\",
        r"\midrule",
    ]
    for kind, label, desc, st in rows:
        if label is None:
            lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{kind}}}}}\\")
            continue
        lines.append(
            f"{label} & {desc} & {st['params']:,} & {st['memory']:,} & {pm(st['acc'], st['acc_sd'])} & {pm(st['forg'], st['forg_sd'])} & {100 * st['new']:.1f} & {seconds(st['time'])} \\\\".replace(
                ",", "{,}"
            )
        )
        key = re.sub(r"[^A-Za-z]", "", label.split("(")[0])
        grp = "Poly" if kind == "p" else "ER"
        M.add(macro_name("h2h", grp, key, "Acc"), f"{100 * st['acc']:.1f}")
        M.add(macro_name("h2h", grp, key, "AccSd"), f"{100 * st['acc_sd']:.1f}")
        M.add(macro_name("h2h", grp, key, "Forg"), f"{100 * st['forg']:.1f}")
        M.add(macro_name("h2h", grp, key, "ForgSd"), f"{100 * st['forg_sd']:.1f}")
        M.add(macro_name("h2h", grp, key, "New"), f"{100 * st['new']:.1f}")
        M.add(macro_name("h2h", grp, key, "Time"), f"{st['time']:.2f}")
        M.add(
            macro_name("h2h", grp, key, "Params"),
            f"{st['params']:,}".replace(",", "{,}"),
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_gen / "table_head2head.tex").write_text("\n".join(lines) + "\n")
    named = {
        ("p", "Highest accuracy (val.)"): "PolyBest",
        ("p", "Best trade-off (val., literal)"): "PolyTrade",
        ("p", "Lowest forgetting (val., floor)"): "PolyLowForg",
        ("p", "Degree 1 throughout (val.-tuned $\\eta$)"): "PolyLinear",
        ("p", "Degree 3 from the start (val.-tuned $\\eta$)"): "PolyFixedCap",
        ("p", "No replay (core-selected $\\eta$)"): "PolyNoReplay",
        ("e", "Highest accuracy (val.)"): "ErBest",
        ("e", "Best trade-off (val., literal)"): "ErTrade",
        ("e", "Best trade-off (val., floor)"): "ErFloorTrade",
        ("e", "Lowest forgetting (val., floor)"): "ErLowForg",
        ("e", "Highest accuracy (hindsight, test)"): "ErHind",
        ("e", "Linear head (val.-tuned $\\eta$)"): "ErLinear",
        ("e", "No replay"): "ErNoReplay",
    }
    for kind, label, desc, st in rows:
        nm = named.get((kind, label)) if label else None
        if nm:
            for k, v in (
                ("Acc", 100 * st["acc"]),
                ("AccSd", 100 * st["acc_sd"]),
                ("Forg", 100 * st["forg"]),
                ("ForgSd", 100 * st["forg_sd"]),
                ("New", 100 * st["new"]),
            ):
                M.add(f"x{nm}{k}".replace("x", "num", 1), f"{v:.1f}")
            M.add(f"num{nm}Time", f"{st['time']:.2f}")
            M.add(f"num{nm}Params", f"{st['params']:,}".replace(",", "{,}"))
    M.add(
        "numFeatureSeconds",
        f"{summary['polynomial_feature_construction_seconds_one_time']:.1f}",
    )

    # =============================================================== paired comparisons
    poly_best_w = pw["max_accuracy"]
    poly_best_label = poly_label(
        poly_best_w["strategy"],
        poly_best_w["sampling"],
        poly_best_w["learning_rates_by_degree"],
    )
    poly_best = ok_runs(runs, poly_best_label)
    er_val_best = ev["winners"]["max_accuracy"]["candidate_id"]
    er_hind_best = er_an["test"]["winners"]["max_accuracy"]["candidate_id"]
    er_floor_tr = ev["accuracy_floor"]["winners"]["best_tradeoff"]["candidate_id"]
    comps = []
    for title, cid in (
        ("ER-MLP, validation-selected", er_val_best),
        ("ER-MLP, hindsight-best on test", er_hind_best),
        ("ER-MLP, best trade-off within accuracy floor", er_floor_tr),
    ):
        comps.append((title, cid, paired(poly_best, ok_runs(runs, cid))))
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Polynomial (validation-selected) $-$ competitor & $\Delta$ acc.\ (pts) & 95\% CI & $\Delta$ forg.\ (pts) & 95\% CI \\",
        r"\midrule",
    ]
    for title, cid, p in comps:
        lines.append(
            f"{title} ({er_desc(cid)}) & {100 * p['acc']['mean']:+.1f} & [{100 * p['acc']['lo']:+.1f}, {100 * p['acc']['hi']:+.1f}] & "
            f"{100 * p['forg']['mean']:+.1f} & [{100 * p['forg']['lo']:+.1f}, {100 * p['forg']['hi']:+.1f}] \\\\"
        )
        key = re.sub(r"[^A-Za-z]", "", title.split(",")[0] + title.split(",")[-1][:12])
        M.add(macro_name("paired", key, "AccDiff"), f"{100 * p['acc']['mean']:+.1f}")
        M.add(macro_name("paired", key, "AccLo"), f"{100 * p['acc']['lo']:+.1f}")
        M.add(macro_name("paired", key, "AccHi"), f"{100 * p['acc']['hi']:+.1f}")
        M.add(macro_name("paired", key, "ForgDiff"), f"{100 * p['forg']['mean']:+.1f}")
        M.add(macro_name("paired", key, "ForgLo"), f"{100 * p['forg']['lo']:+.1f}")
        M.add(macro_name("paired", key, "ForgHi"), f"{100 * p['forg']['hi']:+.1f}")
        M.add(macro_name("paired", key, "AccWins"), str(p["acc"]["wins"]))
        M.add(macro_name("paired", key, "ForgWins"), str(p["forg"]["wins"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_gen / "table_paired.tex").write_text("\n".join(lines) + "\n")
    M.add("numPairs", str(comps[0][2]["n"]))

    # =============================================================== update rule x sampling
    lines = [
        r"\begin{tabular}{@{}llrrrrlrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Shared rate $\eta$} & & \multicolumn{4}{c}{Per-degree vector (validation-selected)} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){7-10}",
        r"Update rule & Sampling & $\eta$ & Acc.\ (\%) & Forg.\ (\%) & & $\eta$ (deg.\ 0--3) & Acc.\ (\%) & Forg.\ (\%) & $t$ (s) \\",
        r"\midrule",
    ]
    cell_rows = []
    for strategy in (
        "ordinary",
        "ordinary_no_replay",
        "small_old_lr",
        "frozen_old",
        "strategy_b",
        "strategy_b_new",
    ):
        for sampling in ("class_balanced", "shuffled"):
            sel = next(
                c
                for c in cands
                if c["strategy"] == strategy
                and c["sampling"] == sampling
                and c["selected_by_core"]
                and c["epochs"] == 3
            )
            sh = next(
                c
                for c in cands
                if c["strategy"] == strategy
                and c["sampling"] == sampling
                and c["vector_label"] == "shared"
                and c["epochs"] == 3
            )
            sel_here = ok_runs(
                runs, poly_label(strategy, sampling, sel["learning_rates_by_degree"])
            )
            t_here = summarise(sel_here)["time"] if sel_here else None
            cell_rows.append((strategy, sampling, sh, sel, t_here))
            lines.append(
                f"{tt(strategy)} & {tt(sampling)} & {sh['learning_rates_by_degree'][0]:g} & {pm(sh['test']['accuracy_mean'], sh['test']['accuracy_std'])} & "
                f"{pm(sh['test']['forgetting_mean'], sh['test']['forgetting_std'])} & & {lr_tex(sel['learning_rates_by_degree'])} & "
                f"{pm(sel['test']['accuracy_mean'], sel['test']['accuracy_std'])} & {pm(sel['test']['forgetting_mean'], sel['test']['forgetting_std'])} & {seconds(t_here)} \\\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_gen / "table_cells.tex").write_text("\n".join(lines) + "\n")
    # sampling effect with the best update rule
    for strategy in ("ordinary", "frozen_old"):
        cb = next(r for r in cell_rows if r[0] == strategy and r[1] == "class_balanced")
        sh_ = next(r for r in cell_rows if r[0] == strategy and r[1] == "shuffled")
        M.add(
            macro_name("cell", strategy, "cb", "Sel", "Acc"),
            f"{100 * cb[3]['test']['accuracy_mean']:.1f}",
        )
        M.add(
            macro_name("cell", strategy, "sh", "Sel", "Acc"),
            f"{100 * sh_[3]['test']['accuracy_mean']:.1f}",
        )
        M.add(
            macro_name("cell", strategy, "cb", "Shared", "Acc"),
            f"{100 * cb[2]['test']['accuracy_mean']:.1f}",
        )
        M.add(
            macro_name("cell", strategy, "sh", "Shared", "Acc"),
            f"{100 * sh_[2]['test']['accuracy_mean']:.1f}",
        )
        M.add(
            macro_name("cell", strategy, "sh", "Shared", "Forg"),
            f"{100 * sh_[2]['test']['forgetting_mean']:.1f}",
        )
    gains = [
        r[3]["test"]["accuracy_mean"] - r[2]["test"]["accuracy_mean"] for r in cell_rows
    ]
    M.add("numPerDegreeGainMin", f"{100 * min(gains):+.1f}")
    M.add("numPerDegreeGainMax", f"{100 * max(gains):+.1f}")
    cb_gain = [
        r[3]["test"]["accuracy_mean"] - r[2]["test"]["accuracy_mean"]
        for r in cell_rows
        if r[1] == "class_balanced" and r[0] != "ordinary_no_replay"
    ]
    M.add("numPerDegreeGainCbMin", f"{100 * min(cb_gain):+.1f}")
    M.add("numPerDegreeGainCbMax", f"{100 * max(cb_gain):+.1f}")

    # =============================================================== epochs 3 vs 10
    ten = summary["ten_epoch_and_ablation_runs"]
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Method & Configuration & Acc.\ 3 ep. & Acc.\ 10 ep. & Forg.\ 3 ep. & Forg.\ 10 ep. & Time 3 ep. & Time 10 ep. \\",
        r"\midrule",
    ]
    for label, spec_desc, r3, r10 in [
        (
            "Polynomial",
            f"{tt(poly_best_w['strategy'])}, $\\eta={lr_tex(poly_best_w['learning_rates_by_degree'])}$",
            summarise(poly_best),
            summarise(ok_runs(runs, poly_best_label, epochs=10)),
        )
    ] + [
        (
            "ER-MLP",
            er_desc(cid),
            summarise(ok_runs(runs, cid)),
            summarise(ok_runs(runs, cid, epochs=10)),
        )
        for cid in dict.fromkeys([er_val_best, er_hind_best, er_floor_tr])
        if ok_runs(runs, cid, epochs=10)
    ]:
        lines.append(
            f"{label} & {spec_desc} & {pm(r3['acc'], r3['acc_sd'])} & {pm(r10['acc'], r10['acc_sd'])} & {pm(r3['forg'], r3['forg_sd'])} & {pm(r10['forg'], r10['forg_sd'])} & {seconds(r3['time'])} & {seconds(r10['time'])} \\\\"
        )
        if label == "Polynomial":
            M.add("numPolyTenAcc", f"{100 * r10['acc']:.1f}")
            M.add("numPolyTenForg", f"{100 * r10['forg']:.1f}")
            M.add("numPolyTenAccSd", f"{100 * r10['acc_sd']:.1f}")
        else:
            pass
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_gen / "table_epochs.tex").write_text("\n".join(lines) + "\n")
    er_ten_best = max(
        (
            summarise(ok_runs(runs, cid, epochs=10))["acc"]
            for cid in er_cands
            if ok_runs(runs, cid, epochs=10)
        ),
        default=None,
    )
    if er_ten_best is not None:
        M.add("numErTenBestAcc", f"{100 * er_ten_best:.1f}")

    # =============================================================== validation -> test transfer
    pool = [c for c in cands if c["validation"] is not None and c["epochs"] == 3]
    v_acc = np.array([c["validation"]["accuracy_mean"] for c in pool])
    t_acc = np.array([c["test"]["accuracy_mean"] for c in pool])
    v_forg = np.array([c["validation"]["forgetting_mean"] for c in pool])
    t_forg = np.array([c["test"]["forgetting_mean"] for c in pool])
    top = [i for i in np.argsort(-v_acc)[:60]]
    M.add("numSpearmanAcc", f"{scipy_stats.spearmanr(v_acc, t_acc)[0]:.2f}")
    M.add(
        "numSpearmanAccTop", f"{scipy_stats.spearmanr(v_acc[top], t_acc[top])[0]:.2f}"
    )
    M.add("numSpearmanForg", f"{scipy_stats.spearmanr(v_forg, t_forg)[0]:.2f}")
    M.add("numMeanAccGap", f"{100 * float((t_acc - v_acc).mean()):+.1f}")
    best_test = max(c["test"]["accuracy_mean"] for c in pool)
    sel_test = pw["max_accuracy"]["test_result"]["accuracy_mean"]
    M.add("numRegret", f"{100 * (best_test - sel_test):.1f}")

    # =============================================================== figure 1: accuracy / forgetting / newest
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    poly_3 = [c for c in cands if c["epochs"] == 3]
    below = 0
    for strategy, colour in STRATEGY_COLOURS.items():
        for sampling, marker in (("class_balanced", "o"), ("shuffled", "^")):
            sub = [
                c
                for c in poly_3
                if c["strategy"] == strategy and c["sampling"] == sampling
            ]
            xs = [100 * c["test"]["forgetting_mean"] for c in sub]
            ys = [100 * c["test"]["accuracy_mean"] for c in sub]
            ns = [100 * c["test"]["current_accuracy_mean"] for c in sub]
            axes[0].scatter(
                xs, ys, s=9, c=colour, marker=marker, alpha=0.65, linewidths=0
            )
            axes[1].scatter(
                xs, ns, s=9, c=colour, marker=marker, alpha=0.65, linewidths=0
            )
            below += sum(1 for y in ys if y < 55)
    ex = [100 * c["test"]["forgetting_mean"] for c in er_cands.values()]
    ey = [100 * c["test"]["accuracy_mean"] for c in er_cands.values()]
    en = [100 * c["test"]["current_accuracy_mean"] for c in er_cands.values()]
    axes[0].scatter(
        ex, ey, s=16, c="black", marker="x", linewidths=0.8, label="ER-MLP grid"
    )
    axes[1].scatter(ex, en, s=16, c="black", marker="x", linewidths=0.8)
    for crit, mk in (
        ("max_accuracy", "*"),
        ("best_tradeoff", "D"),
        ("min_forgetting", "s"),
    ):
        w = pf[crit] if crit != "max_accuracy" else pw[crit]
        t = w["test_result"]
        axes[0].scatter(
            [100 * t["forgetting_mean"]],
            [100 * t["accuracy_mean"]],
            s=70,
            marker=mk,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            zorder=5,
        )
        axes[1].scatter(
            [100 * t["forgetting_mean"]],
            [100 * t["current_accuracy_mean"]],
            s=70,
            marker=mk,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            zorder=5,
        )
    axes[0].set_xlabel("Mean forgetting (%)  [lower is better]")
    axes[0].set_ylabel("Final mean accuracy (%)")
    axes[0].set_xlim(0, 40)
    axes[0].set_ylim(55, 80)
    axes[0].set_title("(a) Accuracy vs forgetting")
    axes[1].set_xlabel("Mean forgetting (%)")
    axes[1].set_ylabel("Accuracy on the newest experience (%)")
    axes[1].set_xlim(0, 40)
    axes[1].set_ylim(20, 100)
    axes[1].set_title("(b) Low forgetting can mean little learning")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=col, label=s.replace("_", " "))
        for s, col in STRATEGY_COLOURS.items()
    ]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="grey", label="class-balanced"),
        plt.Line2D([], [], marker="^", ls="", color="grey", label="shuffled"),
        plt.Line2D([], [], marker="x", ls="", color="black", label="ER-MLP"),
        plt.Line2D(
            [],
            [],
            marker="*",
            ls="",
            mfc="none",
            mec="red",
            label="validation-selected",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    fig.savefig(out_fig / "fig_tradeoffs.pdf", bbox_inches="tight")
    plt.close(fig)
    M.add("numBelowRange", str(below))

    # =============================================================== figure 2: learning curves + per-experience
    def mean_acc_by_step(rs):
        curves = []
        for r in rs:
            m = np.array(
                [
                    [np.nan if v is None else v for v in row]
                    for row in r["accuracy_matrix"]
                ]
            )
            curves.append([np.nanmean(m[i, : i + 1]) for i in range(len(m))])
        return np.array(curves)

    series = [
        ("Polynomial, highest acc. (val.)", poly_best, "#55A868", "-"),
        ("ER-MLP, highest acc. (val.)", ok_runs(runs, er_val_best), "black", "-"),
        ("ER-MLP, floor trade-off", ok_runs(runs, er_floor_tr), "black", "--"),
    ]
    a = summary["polynomial_ablations"].get("fixed_linear")
    if a:
        series.append(
            (
                "Polynomial, degree 1 only",
                [r for r in a["test_runs"] if r["status"] == "completed"],
                "#4C72B0",
                ":",
            )
        )
    if nr_runs:
        series.append(("Polynomial, no replay", nr_runs, "#999999", ":"))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))
    for name, rs, colour, ls in series:
        c = mean_acc_by_step(rs)
        x = np.arange(1, c.shape[1] + 1)
        axes[0].plot(
            x, 100 * c.mean(0), color=colour, ls=ls, marker="o", ms=3, label=name
        )
        axes[0].fill_between(
            x,
            100 * (c.mean(0) - c.std(0, ddof=1)),
            100 * (c.mean(0) + c.std(0, ddof=1)),
            color=colour,
            alpha=0.12,
            lw=0,
        )
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xlabel("Experiences learned")
    axes[0].set_ylabel("Mean accuracy over experiences seen (%)")
    axes[0].set_title("(a) Accuracy as classes arrive")
    axes[0].legend(frameon=False, loc="lower left")
    width = 0.38
    for k, (name, rs, colour) in enumerate(
        (
            ("Polynomial", poly_best, "#55A868"),
            ("ER-MLP", ok_runs(runs, er_val_best), "black"),
        )
    ):
        acc = np.array([r["accuracy_per_experience"] for r in rs])
        axes[1].bar(
            np.arange(5) + (k - 0.5) * width,
            100 * acc.mean(0),
            width,
            yerr=100 * acc.std(0, ddof=1),
            color=colour,
            alpha=0.85,
            label=name,
            capsize=1.5,
            error_kw={"lw": 0.7},
        )
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(["{0,1}", "{2,3}", "{4,5}", "{6,7}", "{8,9}"])
    axes[1].set_xlabel("Experience (classes)")
    axes[1].set_ylabel("Final accuracy per experience (%)")
    axes[1].set_ylim(40, 108)
    axes[1].set_title("(b) What is retained at the end")
    axes[1].legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out_fig / "fig_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    pe_poly = np.array([r["accuracy_per_experience"] for r in poly_best]).mean(0)
    pe_er = np.array(
        [r["accuracy_per_experience"] for r in ok_runs(runs, er_val_best)]
    ).mean(0)
    for i in range(5):
        M.add(macro_name("pe", "poly", chr(65 + i)), f"{100 * pe_poly[i]:.1f}")
        M.add(macro_name("pe", "er", chr(65 + i)), f"{100 * pe_er[i]:.1f}")

    # =============================================================== figure 3: validation -> test transfer
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    colours = [STRATEGY_COLOURS[c["strategy"]] for c in pool]
    axes[0].scatter(100 * v_acc, 100 * t_acc, s=8, c=colours, alpha=0.7, linewidths=0)
    lim = [100 * min(v_acc.min(), t_acc.min()), 100 * max(v_acc.max(), t_acc.max())]
    axes[0].plot(lim, lim, color="grey", lw=0.7, ls="--")
    axes[0].set_xlabel("Validation accuracy (%)")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title(rf"(a) Accuracy, Spearman $\rho$={M.items['numSpearmanAcc']}")
    axes[1].scatter(100 * v_forg, 100 * t_forg, s=8, c=colours, alpha=0.7, linewidths=0)
    lim = [100 * min(v_forg.min(), t_forg.min()), 100 * max(v_forg.max(), t_forg.max())]
    axes[1].plot(lim, lim, color="grey", lw=0.7, ls="--")
    axes[1].set_xlabel("Validation forgetting (%)")
    axes[1].set_ylabel("Test forgetting (%)")
    axes[1].set_title(rf"(b) Forgetting, Spearman $\rho$={M.items['numSpearmanForg']}")
    fig.tight_layout()
    fig.savefig(out_fig / "fig_transfer.pdf", bbox_inches="tight")
    plt.close(fig)

    # =============================================================== misc counts for the text
    M.add("numPolyCandidatesThree", str(len([c for c in cands if c["epochs"] == 3])))
    M.add("numErCandidates", str(len(er_cands)))
    M.add("numTestSeeds", str(len(summary["test_seeds"])))
    checks = summary["reproduction_checks_against_core_results"]
    M.add("numReproChecks", str(len(checks)))
    M.add(
        "numReproMaxDiff",
        f"{max(c['max_abs_difference'] for c in checks if c['max_abs_difference'] is not None):.1e}",
    )
    M.add(
        "numReproExact", str(sum(1 for c in checks if c["max_abs_difference"] == 0.0))
    )
    fl = [c for c in cands if c["epochs"] == 3 and c["test"]["time_seconds_mean"]]
    M.add("numLogTimeMin", f"{min(c['test']['time_seconds_mean'] for c in fl):.1f}")
    M.add("numLogTimeMax", f"{max(c['test']['time_seconds_mean'] for c in fl):.1f}")
    for sp in ("class_balanced", "shuffled"):
        sub = [
            c
            for c in poly_3
            if c["sampling"] == sp and c["strategy"] != "ordinary_no_replay"
        ]
        M.add(
            macro_name("median", sp, "Acc"),
            f"{100 * float(np.median([c['test']['accuracy_mean'] for c in sub])):.1f}",
        )
        M.add(
            macro_name("median", sp, "Forg"),
            f"{100 * float(np.median([c['test']['forgetting_mean'] for c in sub])):.1f}",
        )

    (out_gen / "numbers.tex").write_text(M.tex())
    print(f"wrote tables, figures and {len(M.items)} macros under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
