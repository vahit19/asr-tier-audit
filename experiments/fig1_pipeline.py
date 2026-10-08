"""Fig 1: block diagram of the claim-certification pipeline (two rows, page width)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

TOP = [
    ("Published claim\n\"WER ≤ 5%\" for L,\nor \"preferred by p%\"", "#eef3fa"),
    ("Sampling design\nlength-proportional\ndraws", "#eef3fa"),
    ("Collect output\none API call per draw,\ncached", "#eef3fa"),
    ("Normalizer\nWhisper basic,\nlanguage-aware\nor character level", "#fdf0e6"),
]
# The bottom row runs right to left, under top boxes 3, 2 and 1.
BOTTOM = [
    ("Bounded loss\nmin(E/N, 2) / 2,\nor 1 − win", "#eef3fa"),
    ("Two betting tests\ncertify and refute\nwealth", "#eef3fa"),
    ("Decision\ncertified, refuted\nor undecided", "#e8f5ec"),
]
W, H, GAP = 1.7, 1.0, 0.27


def box(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=color, ec="0.35", lw=0.9))
    ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=9, linespacing=1.3)


def arrow(ax, start, end, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=0.9,
                                 color="0.35", **kw))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.set_xlim(0, 4 * W + 3 * GAP + 0.1)
    ax.set_ylim(0, 2 * H + 0.75)
    ax.axis("off")
    y_top, y_bot = H + 0.6, 0.1
    xs = [0.05 + i * (W + GAP) for i in range(4)]
    for i, (text, color) in enumerate(TOP):
        box(ax, xs[i], y_top, text, color)
        if i < len(TOP) - 1:
            arrow(ax, (xs[i] + W, y_top + H / 2), (xs[i + 1], y_top + H / 2))
    for k, (text, color) in enumerate(BOTTOM):
        box(ax, xs[3 - k], y_bot, text, color)
        if k < len(BOTTOM) - 1:
            arrow(ax, (xs[3 - k], y_bot + H / 2), (xs[2 - k] + W, y_bot + H / 2))
    arrow(ax, (xs[3] + W / 2, y_top), (xs[3] + W / 2, y_bot + H))
    arrow(ax, (xs[2] + W * 0.3, y_bot + H), (xs[1] + W * 0.7, y_top), linestyle="--")
    ax.text(xs[1] + W * 0.55, (y_top + y_bot + H) / 2, "no decision yet:\ndraw again",
            fontsize=8.5, color="0.3", ha="right", va="center")
    fig.tight_layout()
    out = ROOT / "figures" / "fig1_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
