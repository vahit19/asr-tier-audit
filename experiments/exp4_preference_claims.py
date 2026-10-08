"""Exp 4: what published blind-preference percentages need before they separate from 50%.

Claims are copied verbatim from vendor posts (fetched 2026-10-08); neither post reports
the number of votes, listeners or lines. For each percentage p this computes
  * the independent votes a one-sided binomial test at alpha = 0.05 needs to reject 50%;
  * the same under clustering (votes from one listener or one line are correlated),
    using the design effect 1 + (m - 1) * rho with m = 20 votes per cluster;
  * the median votes the anytime-valid betting test needs when p is the true rate.

Writes results/tables/exp4_preference_claims.csv and figures/fig6_preference_votes.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from asr_tier_audit import certify

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 9.5})
ALPHA, M_PER_CLUSTER = 0.05, 20

# (source, comparison, published share). Sources:
#   https://cartesia.ai/blog/sonic-3.6  (2026-08-27)
#   https://elevenlabs.io/blog/eleven-v4 (2026-09-28)
CLAIMS = [
    ("Cartesia Sonic-3.6", "en-US vs Eleven v3", 0.92),
    ("Cartesia Sonic-3.6", "es-ES vs Eleven v3", 0.91),
    ("Cartesia Sonic-3.6", "fr-FR vs Eleven v3", 0.86),
    ("Cartesia Sonic-3.6", "en-AU vs Eleven v3", 0.77),
    ("Cartesia Sonic-3.6", "en-GB vs Eleven v3", 0.72),
    ("Cartesia Sonic-3.6", "es-MX vs Eleven v3", 0.70),
    ("Cartesia Sonic-3.6", "pt-BR vs Eleven v3", 0.67),
    ("Cartesia Sonic-3.6", "hi-IN vs Gemini 3.1", 0.55),
    ("ElevenLabs Eleven v4", "lowest of range (65-81%)", 0.65),
    ("ElevenLabs Eleven v4", "headline (~75%)", 0.75),
]


def min_votes_exact(p: float) -> int:
    """Smallest n where observing round(p*n) wins rejects 50% (one-sided exact test)."""
    for n in range(2, 20000):
        k = int(round(p * n))
        if binom.sf(k - 1, n, 0.5) <= ALPHA:
            return n
    return -1


def median_sequential_votes(p: float, rng, sims: int = 400, max_n: int = 20000) -> float:
    stops = []
    for _ in range(sims):
        votes = (rng.random(max_n) < p).astype(float)
        # A preference claim is a tier claim on the loss 1 - win: "loss rate < 0.5".
        t = certify.sequential_rate_test(1 - votes, tau=0.5, alpha=ALPHA, cap=1.0)
        stops.append(t.stop_index if t.decision == "certified" else max_n)
    return float(np.median(stops))


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for source, comparison, p in CLAIMS:
        n_iid = min_votes_exact(p)
        rows.append({
            "source": source, "comparison": comparison, "published_share": p,
            "votes_iid": n_iid,
            "votes_rho_0.05": int(np.ceil(n_iid * (1 + (M_PER_CLUSTER - 1) * 0.05))),
            "votes_rho_0.20": int(np.ceil(n_iid * (1 + (M_PER_CLUSTER - 1) * 0.20))),
            "median_votes_sequential": median_sequential_votes(p, rng),
        })

    out = ROOT / "results" / "tables" / "exp4_preference_claims.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    grid = np.linspace(0.52, 0.95, 200)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    base = np.array([(1.645 * 0.5 / (g - 0.5)) ** 2 for g in grid])
    ax.plot(100 * grid, base, color="#1f5fa8", lw=1.5, label="independent votes")
    ax.plot(100 * grid, base * (1 + 19 * 0.05), color="#e67e22", lw=1.2, ls="--",
            label="clustered, ρ = 0.05 (20 votes per listener)")
    ax.plot(100 * grid, base * (1 + 19 * 0.20), color="#c0392b", lw=1.2, ls=":",
            label="clustered, ρ = 0.20")
    for r in rows:
        mk = "o" if r["source"].startswith("Cartesia") else "s"
        ax.scatter(100 * r["published_share"], r["votes_iid"], marker=mk, color="0.15", s=18, zorder=3)
    ax.scatter([], [], marker="o", color="0.15", s=18, label="published shares, Cartesia (2026)")
    ax.scatter([], [], marker="s", color="0.15", s=18, label="published shares, ElevenLabs (2026)")
    ax.set_yscale("log")
    ax.set_xlabel("Published preference share (%)")
    ax.set_ylabel("Votes needed to reject 50% (log)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig6_preference_votes.png", dpi=200)
    fig.savefig(Path(ROOT / "figures" / "fig6_preference_votes.png").with_suffix(".pdf"))
    print(f"wrote {out} and figures/fig6_preference_votes.png")


if __name__ == "__main__":
    main()
