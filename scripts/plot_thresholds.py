"""Threshold-evolution and sparsity figure, read from run directories.

    python scripts/plot_thresholds.py \\
        --ours runs/cifar10_resnet18_ours_s0 \\
        --compare "Fixed (tuned)=runs/cifar10_resnet18_fixed_tau0.3_s0" \\
        --compare TWN=runs/cifar10_resnet18_twn_s0 \\
        --compare LSQ=runs/cifar10_resnet18_lsq_s0 \\
        --title ResNet-18 --out plots/threshold_evolution_resnet18.pdf

(a) per-layer thresholds τ+ / τ- of our method over training, coloured by
    layer depth (σ units, from logs/metrics.jsonl);
(b) final fraction of zero weights per layer for our method and up to three
    baselines (from results.json).

Every value comes from the run directories; nothing is typed by hand.
Needs matplotlib (requirements-dev.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Validated palette (categorical slots 1-4, light surface) and one-hue
# ordinal ramp for layer depth; chart chrome inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
DEPTH_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def read_history(run: Path) -> list[dict]:
    with open(run / "logs" / "metrics.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def final_zero_fraction(run: Path) -> np.ndarray:
    with open(run / "results.json", encoding="utf-8") as f:
        r = json.load(f)
    if r.get("ternary_fractions"):
        return np.array([d["0"] for d in r["ternary_fractions"]])
    return np.array(r["history"][-1]["zero_fraction"])


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def direct_labels(ax, ends: list[tuple[float, float, str]], x: float):
    """Label line ends on the right; labels are spread symmetrically so they
    never collide, and a hairline leader ties each label to its line."""
    lo, hi = ax.get_ylim()
    gap = 0.075 * (hi - lo)
    ends = sorted(ends)
    ys = [y for y, _, _ in ends]
    pos = list(ys)
    for _ in range(50):  # relax until every neighbour pair is >= gap apart
        moved = False
        for i in range(1, len(pos)):
            d = pos[i] - pos[i - 1]
            if d < gap:
                shift = (gap - d) / 2
                pos[i - 1] -= shift
                pos[i] += shift
                moved = True
        if not moved:
            break
    for (y, _, label), y_lab in zip(ends, pos):
        ax.annotate(
            label,
            (x, y),
            xytext=(x + 1.2, y_lab),
            textcoords="data",
            va="center",
            fontsize=7,
            color=INK,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument(
        "--compare",
        action="append",
        default=[],
        help="LABEL=RUN_DIR, up to three",
    )
    ap.add_argument("--fixed_tau", type=float, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="plots/threshold_evolution.pdf")
    a = ap.parse_args()
    if len(a.compare) > 3:
        ap.error("at most three --compare runs (four series in total)")

    hist = read_history(a.ours)
    epochs = np.array([h["epoch"] + 1 for h in hist])
    tau = np.array([h["thresholds"] for h in hist])  # (E, L, 2)
    n_layers = tau.shape[1]

    plt.rcParams.update({"font.size": 7, "axes.labelcolor": INK})
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # (a) threshold trajectories, colour = depth
    cmap = LinearSegmentedColormap.from_list("depth", DEPTH_RAMP)
    for layer in range(n_layers):
        c = cmap(layer / max(n_layers - 1, 1))
        ax_a.plot(epochs, tau[:, layer, 0], color=c, linewidth=0.9)
        ax_a.plot(epochs, tau[:, layer, 1], color=c, linewidth=0.9)
    ax_a.axhline(0, color=AXIS, linewidth=0.6)
    if a.fixed_tau is not None:
        for s in (1, -1):
            ax_a.axhline(
                s * a.fixed_tau, color=MUTED, linewidth=0.8, linestyle="--"
            )
        ax_a.text(
            epochs[-1],
            a.fixed_tau,
            r"fixed $\pm\tau$",
            color=INK_2,
            fontsize=6.5,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel(r"Threshold ($\sigma$ units)")
    ax_a.set_title(
        f"(a) Learned $\\tau^+_l$, $\\tau^-_l$ {a.title}".strip(),
        fontsize=7.5,
        color=INK,
    )
    style(ax_a)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(1, max(n_layers, 2))
    )
    cb = fig.colorbar(sm, ax=ax_a, fraction=0.05, pad=0.02)
    cb.set_label("Layer depth", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=2)
    cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # (b) final zero fraction per layer
    runs = [("Ours", a.ours)] + [
        (lab, Path(p)) for lab, p in (c.split("=", 1) for c in a.compare)
    ]
    layers = np.arange(1, n_layers + 1)
    ends = []
    for i, (label, run) in enumerate(runs):
        z = final_zero_fraction(run)
        ax_b.plot(
            layers[: len(z)],
            z,
            color=SERIES[i],
            linewidth=1.2,
            marker=MARKERS[i],
            markersize=3,
            label=label,
            zorder=3 + len(runs) - i,  # ours drawn on top
        )
        ends.append((z[-1], i, label))
    ax_b.set_xlabel("Layer (forward order)")
    ax_b.set_ylabel("Fraction of zero weights")
    ax_b.set_ylim(0, 1)
    ax_b.set_xlim(0.5, n_layers + 0.5)
    ax_b.set_title("(b) Sparsity after training", fontsize=7.5, color=INK)
    style(ax_b)
    ax_b.legend(frameon=False, fontsize=6.5, loc="upper left", labelcolor=INK)
    direct_labels(ax_b, ends, n_layers)

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
