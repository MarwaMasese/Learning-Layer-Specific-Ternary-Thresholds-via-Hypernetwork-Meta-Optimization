"""Aggregate run directories into summary tables (mean +- std over seeds).

    python scripts/aggregate.py                 # TEST accuracy, markdown
    python scripts/aggregate.py --latex tables  # also write the paper's
                                                # Tabs. 1-5 (paper_table*.tex)
    python scripts/aggregate.py --heldout       # HELD-OUT accuracy, for
                                                # monitoring only

The paper's tables are meant to be regenerated from this script rather
than edited by hand; scripts/latex_tables.py holds the layouts. Runs are
grouped
by (dataset, model, method + tag); the seed count n is always shown, so
incomplete groups are visible. Reported accuracy is the final-epoch model's
test accuracy with exact ternary weights; held-out accuracy is never used to
choose anything (paper Sec. 4.1).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from latex_tables import write_paper_tables

METHOD_ORDER = [
    "fp",
    "twn",
    "ttq",
    "lsq",
    "rtn",
    "fatnn",
    "metaquant",
    "fixed",
    "perlayer",
    "ours_joint",
    "ours",
]
MODEL_ORDER = [
    "resnet18",
    "resnet50",
    "mobilenetv2",
    "convnext_t",
    "vit_s16",
    "swin_t",
]


def load(runs_dir: str) -> dict:
    """{(dataset, model, variant): [results dicts]} from runs/*/."""
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(runs_dir, "*", "results.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        c = r["config"]
        groups[(c["dataset"], c["model"], c["method"] + c["tag"])].append(r)
    return groups


def heldout(r: dict) -> float | None:
    """Final-epoch held-out accuracy (hard ternary for ternary runs)."""
    h = (r.get("history") or [{}])[-1]
    return h.get("heldout_acc_hard", h.get("heldout_acc"))


def value(r: dict, metric: str) -> float | None:
    return heldout(r) if metric == "heldout" else r.get("test_acc")


def stat(runs: list, metric: str):
    v = [value(r, metric) for r in runs]
    v = np.array([x for x in v if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def fmt(s, latex=False) -> str:
    if s is None:
        return "--"
    m, sd, n = s
    pm = "\\pm" if latex else "±"
    txt = f"{m:.2f} {pm} {sd:.2f}" if n > 1 else f"{m:.2f}"
    return f"${txt}$" if latex else f"{txt} (n={n})"


def paired(groups, ds, model, a, b, metric):
    """Per-seed difference a - b over seeds present in both."""
    va = {
        r["config"]["seed"]: value(r, metric)
        for r in groups.get((ds, model, a), [])
    }
    vb = {
        r["config"]["seed"]: value(r, metric)
        for r in groups.get((ds, model, b), [])
    }
    d = [va[s] - vb[s] for s in va if s in vb]
    if not d:
        return None
    return np.mean(d), (np.std(d, ddof=1) if len(d) > 1 else 0.0), len(d)


def order_key(v: str):
    return (METHOD_ORDER.index(v) if v in METHOD_ORDER else 99, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--latex", default=None, help="directory for .tex rows")
    ap.add_argument(
        "--paper_std",
        action="store_true",
        help="paper tables: show mean ± std instead of the mean only",
    )
    ap.add_argument(
        "--heldout",
        action="store_true",
        help="report held-out instead of test accuracy",
    )
    a = ap.parse_args()
    metric = "heldout" if a.heldout else "test"
    groups = load(a.runs)
    if not groups:
        print("no results found")
        return

    for ds in sorted({k[0] for k in groups}):
        models = [
            m for m in MODEL_ORDER if any(k[:2] == (ds, m) for k in groups)
        ]
        variants = sorted({k[2] for k in groups if k[0] == ds}, key=order_key)
        what = "HELD-OUT (monitoring only)" if a.heldout else "TEST"
        print(
            f"\n## {ds}  ({what} top-1 %, hard ternary; "
            "mean ± std over seeds)\n"
        )
        print("| method | " + " | ".join(models) + " |")
        print("|---|" + "---|" * len(models))
        for v in variants:
            cells = [
                fmt(stat(groups.get((ds, m, v), []), metric)) for m in models
            ]
            print(f"| {v} | " + " | ".join(cells) + " |")
        for base in ("lsq", "twn"):
            cells = [
                fmt(paired(groups, ds, m, "ours", base, metric))
                for m in models
            ]
            print(f"| ours − {base} (paired) | " + " | ".join(cells) + " |")

        print(
            "\n| model | ternary layers | params | method params | "
            "s/epoch fixed | s/epoch ours | peak MB fixed | peak MB ours |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for m in models:
            o = groups.get((ds, m, "ours"), [])
            f = groups.get((ds, m, "fixed"), [])
            if not o:
                continue

            def mean(runs, k):
                v = [r[k] for r in runs if r.get(k)]
                return float(np.mean(v)) if v else float("nan")

            print(
                f"| {m} | {o[0].get('num_ternary_layers')} | "
                f"{o[0].get('params'):,} | {o[0].get('method_params'):,} | "
                f"{mean(f, 'mean_epoch_time_s'):.0f} | "
                f"{mean(o, 'mean_epoch_time_s'):.0f} | "
                f"{mean(f, 'peak_mem_mb'):.0f} | "
                f"{mean(o, 'peak_mem_mb'):.0f} |"
            )

    if a.latex and not a.heldout:
        # paper_table{1..5}_*.tex (test accuracy)
        write_paper_tables(a.latex, groups, stat, paired, std=a.paper_std)


if __name__ == "__main__":
    main()
