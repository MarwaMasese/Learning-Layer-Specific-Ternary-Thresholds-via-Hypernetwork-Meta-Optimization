"""Figures 2 and 3 of the paper, read from run directories.

    python scripts/plot_paper_figures.py --runs runs --out plots

    fig2_imagenet_efficiency.pdf  Fig. 2: per-backbone ImageNet top-1 of ours
                                  (left) and hypernetwork memory overhead,
                                  peak memory ours / fixed (right)
    fig3_ablations.pdf            Fig. 3: CIFAR-10 / ResNet-18 ablations for
                                  bilevel structure, statistical conditioning
                                  and quantization design; dashed line = full
                                  method, labels = drop relative to it

Values come from the same run groups as scripts/aggregate.py (test accuracy,
final epoch, mean over seeds); nothing is typed by hand. Needs matplotlib.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from aggregate import load, stat  # noqa: E402
from latex_tables import MODEL_PRETTY, PAPER_ABLATION_GROUPS  # noqa: E402
from plot_thresholds import AXIS, GRID, INK, INK_2, MUTED, SERIES  # noqa: E402

BAR = SERIES[0]  # one series per panel: one hue, identity from the axis
FIG3_GROUPS = (
    "Bilevel structure",
    "Statistical conditioning",
    "Quantization design",
)


def _style(ax) -> None:
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=2)
    ax.tick_params(axis="y", length=0, colors=INK)


def _mean(runs: list, key: str) -> float | None:
    v = [r[key] for r in runs if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def fig2(groups: dict, path: str) -> bool:
    rows = []
    for m, label in MODEL_PRETTY.items():
        o = groups.get(("imagenet", m, "ours"), [])
        f = groups.get(("imagenet", m, "fixed"), [])
        s = stat(o, "test")
        if not o or s is None:
            continue
        acc = s[0]
        po, pf = _mean(o, "peak_mem_mb"), _mean(f, "peak_mem_mb")
        rows.append((label, acc, po / pf if po and pf else np.nan))
    if not rows:
        return False
    rows = rows[::-1]  # first backbone on top
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.0, 0.42 * len(rows) + 1.0), sharey=True
    )
    for ax, vals, fmt, xlab in (
        (a, [r[1] for r in rows], "{:.1f}%", "Top-1 Accuracy (%)"),
        (b, [r[2] for r in rows], "{:.2f}×", "Memory Overhead (×)"),
    ):
        _style(ax)
        ax.barh(y, vals, height=0.6, color=BAR, edgecolor="white", lw=2)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.annotate(
                    fmt.format(v),
                    (v, yi),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=INK_2,
                )
        ax.set_xlabel(xlab, color=INK_2, fontsize=9)
    finite = [r[1] for r in rows]
    a.set_xlim(np.floor(min(finite) / 5) * 5 - 5, max(finite) + 5)
    mem = [r[2] for r in rows if np.isfinite(r[2])]
    if mem:
        b.set_xlim(1.0, max(mem) + 0.03)
    a.set_yticks(y, labels)
    fig.suptitle(
        "ImageNet Accuracy and Hypernetwork Memory Cost",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def fig3(groups: dict, path: str) -> bool:
    key = ("cifar10", "resnet18")
    full = stat(groups.get(key + ("ours",), []), "test")
    if full is None:
        return False
    full = full[0]
    labels, vals, heads = [], [], []
    for title, items in PAPER_ABLATION_GROUPS:
        if title not in FIG3_GROUPS:
            continue
        present = [
            (lab, s)
            for v, lab in items
            if (s := stat(groups.get(key + (v,), []), "test")) is not None
        ]
        if not present:
            continue
        heads.append((len(labels), title))
        for lab, s in present:
            labels.append(lab)
            vals.append(s[0])
    if not vals:
        return False
    n = len(vals)
    y = -np.arange(n, dtype=float)
    # leave a gap above each group for its heading
    for start, _ in heads:
        y[start:] -= 0.8
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * n + 0.8 * len(heads) + 0.8))
    _style(ax)
    ax.barh(y, vals, height=0.6, color=BAR, edgecolor="white", lw=2)
    ax.axvline(full, color=MUTED, linestyle="--", linewidth=1)
    for yi, v in zip(y, vals):
        ax.annotate(
            f"{v - full:+.1f}",
            (v, yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    for start, title in heads:
        ax.text(
            0.0,
            y[start] + 0.8,
            title,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
        )
    lo = np.floor(min(vals + [full])) - 1
    ax.set_xlim(lo, max(vals + [full]) + 1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Top-1 accuracy (%)", color=INK_2, fontsize=9)
    ax.set_title(f"Full method: {full:.1f}%", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()
    groups = load(a.runs)
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (
        ("fig2_imagenet_efficiency.pdf", fig2),
        ("fig3_ablations.pdf", fig3),
    ):
        path = os.path.join(a.out, name)
        print(("wrote " if fn(groups, path) else "no runs for ") + path)


if __name__ == "__main__":
    main()
