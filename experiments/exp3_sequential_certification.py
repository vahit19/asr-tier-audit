"""Exp 3: anytime-valid certification of the "WER <= 5%" claim with length-proportional
sampling, simulated by drawing from the transcribed pool as a finite population.

For each language and normalizer:
  * decision rates and median audio needed at tau = 5% (the vendor's claim);
  * error rate at tau = pool WER (the boundary, where any decision is wrong);
  * a fixed-sample comparison: utterances a one-shot z-test would need for 80% power.

Writes results/tables/exp3_sequential.csv, figures/fig4_sequential_trajectories.png and
figures/fig5_audio_needed.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from asr_tier_audit import certify, data, metrics

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 9.5})
TAU, ALPHA, MAX_DRAWS, N_SIM = 0.05, 0.05, 3000, 500


def durations(rows: list[dict]) -> np.ndarray:
    return np.array([r["duration_s"] for r in rows])


def simulate(rates, probs, dur, tau, rng, cap=2.0):
    decisions, n_used, sec_used = [], [], []
    for _ in range(N_SIM):
        idx = rng.choice(len(rates), size=MAX_DRAWS, p=probs)
        t = certify.sequential_rate_test(rates[idx], tau, ALPHA, cap=cap)
        n = t.stop_index or MAX_DRAWS
        decisions.append(t.decision)
        n_used.append(n)
        sec_used.append(dur[idx[:n]].sum())
    return np.array(decisions), np.array(n_used), np.array(sec_used)


def fixed_n(rates, probs, tau) -> float:
    """Draws a one-sided z-test at level alpha needs for 80% power (PPS design)."""
    mu = float(np.sum(probs * rates))
    sd = float(np.sqrt(np.sum(probs * (rates - mu) ** 2)))
    if abs(mu - tau) < 1e-12:
        return float("inf")
    return ((1.645 + 0.842) * sd / abs(mu - tau)) ** 2


def hoeffding_n(pool_wer: float, tau: float, cap: float = 2.0) -> float:
    """Fixed sample size at which a one-sided Hoeffding bound on Y = r/cap in [0, 1]
    separates the true mean from tau/cap at level alpha (best case: estimate = truth)."""
    gap = abs(pool_wer - tau) / cap
    return float(np.log(1 / ALPHA) / (2 * gap ** 2)) if gap > 0 else float("inf")


def main() -> None:
    rng = np.random.default_rng(0)
    out_rows, examples = [], {}
    for cfg in data.available():
        lang, name = data.LANGUAGES[cfg]
        rows = data.load_manifest(cfg)
        dur = durations(rows)
        for scheme in ("whisper_basic", "lang_join"):
            c = data.counts_for(rows, lang, scheme)
            n = c.ref_len.astype(float)
            rates = c.errors / n
            probs = n / n.sum()
            pool_wer = metrics.wer(c)
            dec, n_used, sec = simulate(rates, probs, dur, TAU, rng)
            dec_b, _, _ = simulate(rates, probs, dur, pool_wer, rng)
            dec1, n1, _ = simulate(rates, probs, dur, TAU, rng, cap=1.0)
            dec1_b, _, _ = simulate(rates, probs, dur, pool_wer, rng, cap=1.0)
            decided = dec != "undecided"
            out_rows.append({
                "language": name, "scheme": scheme, "pool_wer": pool_wer,
                "p_certify": np.mean(dec == "certified"), "p_refute": np.mean(dec == "refuted"),
                "p_undecided": np.mean(~decided),
                "median_utts_when_decided": float(np.median(n_used[decided])) if decided.any() else None,
                "median_minutes_when_decided": float(np.median(sec[decided]) / 60) if decided.any() else None,
                "fixed_n_80pct_power": fixed_n(rates, probs, TAU),
                "hoeffding_fixed_n": hoeffding_n(pool_wer, TAU),
                "boundary_error_rate": np.mean(dec_b != "undecided"),
                "rates_above_cap": int(np.sum(rates > 2.0)),
                "cap1_median_utts_when_decided": float(np.median(n1[dec1 != "undecided"])) if (dec1 != "undecided").any() else None,
                "cap1_p_decided_correct": float(np.mean(dec1 == ("certified" if pool_wer < TAU else "refuted"))),
                "cap1_boundary_error_rate": float(np.mean(dec1_b != "undecided")),
                "rates_above_1": int(np.sum(rates > 1.0)),
            })
            examples[(name, scheme)] = (rates, probs)

    out = ROOT / "results" / "tables" / "exp3_sequential.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)

    # Fig 4: one capital trajectory per language under the language-aware normalizer.
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    traj_rng = np.random.default_rng(7)
    for (name, scheme), (rates, probs) in examples.items():
        if scheme != "lang_join":
            continue
        idx = traj_rng.choice(len(rates), size=600, p=probs)
        t = certify.sequential_rate_test(rates[idx], TAU, ALPHA)
        k = np.maximum(t.certify_capital, 1e-3)
        k_r = np.maximum(t.refute_capital, 1e-3)
        line, = ax.plot(np.arange(1, 601), k, lw=1.2, label=name)
        ax.plot(np.arange(1, 601), k_r, lw=1.0, ls=":", color=line.get_color())
    ax.axhline(1 / ALPHA, color="0.3", ls="--", lw=1)
    ax.text(5, 1 / ALPHA * 1.3, "decision threshold 1/α = 20", ha="left", fontsize=8.5, color="0.3")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("Utterances transcribed (length-proportional draws)")
    ax.set_ylabel("Test capital (log scale)")
    ax.plot([], [], color="0.4", lw=1.2, label="solid: certify capital")
    ax.plot([], [], color="0.4", lw=1.0, ls=":", label="dotted: refute capital")
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig4_sequential_trajectories.png", dpi=200)
    fig.savefig(Path(ROOT / "figures" / "fig4_sequential_trajectories.png").with_suffix(".pdf"))
    # Fig 5: utterances to reach a decision. The fixed-sample bar is an oracle: it
    # assumes the true WER is known when the sample size is planned.
    sel = [r for r in out_rows if r["median_utts_when_decided"] is not None]
    labels = [f"{r['language']}, {'Whisper basic' if r['scheme'] == 'whisper_basic' else 'language-aware'}"
              for r in sel]
    y = np.arange(len(sel))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(y - 0.27, [r["median_utts_when_decided"] for r in sel], 0.27, color="#1f5fa8",
            label="Anytime-valid betting test: median utterances to decide")
    ax.barh(y, [min(r["hoeffding_fixed_n"], 1e5) for r in sel], 0.27, color="#e67e22",
            label="Fixed-sample Hoeffding bound, same validity (best case)")
    ax.barh(y + 0.27, [min(r["fixed_n_80pct_power"], 1e5) for r in sel], 0.27, color="#bbbbbb",
            label="Fixed-sample z-test sized with the true WER and variance known (oracle)")
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Utterances (log scale)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.15), ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig5_audio_needed.png", dpi=200)
    fig.savefig(Path(ROOT / "figures" / "fig5_audio_needed.png").with_suffix(".pdf"))
    print(f"wrote {out} and figures 4-5")


if __name__ == "__main__":
    main()
