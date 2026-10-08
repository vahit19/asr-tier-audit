"""Exp 1: corpus WER per language under each normalizer, with clustered and naive CIs.

Writes results/tables/exp1_wer_by_normalizer.csv and figures/fig2_wer_by_normalizer.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from asr_tier_audit import data, metrics

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 9.5})
TAU = 0.05
LABELS = {"whisper_basic": "Whisper basic (word)", "lang_join": "Language-aware, apostrophe joined (word)",
          "lang_split": "Language-aware, apostrophe split (word)",
          "lang_char": "Language-aware, character level"}


def main() -> None:
    rows_out = []
    for cfg in data.available():
        lang, name = data.LANGUAGES[cfg]
        rows = data.load_manifest(cfg)
        for scheme in metrics.SCHEMES + ("lang_char",):
            c = data.counts_for(rows, lang, scheme)
            lo, hi = metrics.bootstrap_ci(c, clustered=True)
            nlo, nhi = metrics.bootstrap_ci(c, clustered=False)
            rows_out.append({
                "language": name, "config": cfg, "scheme": scheme,
                "unit": "char" if lang in metrics.CHAR_LANGS or scheme == "lang_char" else "word",
                "utterances": len(rows), "sentences": int(c.cluster.max() + 1),
                "ref_tokens": int(c.ref_len.sum()), "wer": metrics.wer(c),
                "ci_low": lo, "ci_high": hi, "naive_ci_low": nlo, "naive_ci_high": nhi,
                "width_ratio": (hi - lo) / (nhi - nlo),
                "tier_status": ("above" if lo > TAU else "below" if hi < TAU else "straddles"),
            })

    out = ROOT / "results" / "tables" / "exp1_wer_by_normalizer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)

    langs = list(dict.fromkeys(r["language"] for r in rows_out))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    offsets = {"whisper_basic": -0.27, "lang_join": -0.09, "lang_split": 0.09, "lang_char": 0.27}
    markers = {"whisper_basic": "s", "lang_join": "o", "lang_split": "^", "lang_char": "D"}
    colors = {"whisper_basic": "#c0392b", "lang_join": "#1f5fa8", "lang_split": "#6aa0d8",
              "lang_char": "#27ae60"}
    for scheme in metrics.SCHEMES + ("lang_char",):
        sel = [r for r in rows_out if r["scheme"] == scheme]
        x = np.array([langs.index(r["language"]) for r in sel]) + offsets[scheme]
        y = np.array([100 * r["wer"] for r in sel])
        err = np.array([[100 * (r["wer"] - r["ci_low"]) for r in sel],
                        [100 * (r["ci_high"] - r["wer"]) for r in sel]])
        ax.errorbar(x, y, yerr=err, fmt=markers[scheme], color=colors[scheme], capsize=3,
                    ms=5, lw=1.2, label=LABELS[scheme])
    ax.axhline(100 * TAU, color="0.3", ls="--", lw=1)
    ax.text(len(langs) - 0.55, 100 * TAU + 0.2, "5% tier bound", ha="right",
            va="bottom", fontsize=8.5, color="0.3")
    char_langs = {name for code, name in data.LANGUAGES.values() if code in metrics.CHAR_LANGS}
    names = [f"{l}\n(all character level)" if l in char_langs else l for l in langs]
    ax.set_xticks(range(len(langs)), names)
    ax.set_ylabel("Error rate (%), 95% sentence-cluster CI")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = ROOT / "figures" / "fig2_wer_by_normalizer.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    fig.savefig(Path(fig_path).with_suffix(".pdf"))
    print(f"wrote {out} and {fig_path}")


if __name__ == "__main__":
    main()
