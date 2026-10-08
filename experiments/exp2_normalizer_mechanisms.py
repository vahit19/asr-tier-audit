"""Exp 2: where the normalizer gap comes from, by switching one mechanism at a time.

Starting from the language-aware word-level WER, measure what each choice changes:
  bracket deletion   Whisper's BasicTextNormalizer drops text inside (...) and [...];
                     FLEURS references read that text aloud, so correct words in the
                     hypothesis become insertions.
  dotted capital I   'İ'.lower() gives 'i' + U+0307; Whisper's normalizer turns the mark
                     into a space, so 'İZMİR' becomes 'i zmi r'.
  mark splitting     the same rule splits words at every combining mark, which in
                     Brahmic scripts (Malayalam, Kannada) cuts words into fragments
                     and makes word error behave like a sub-word error.
  numerals           share of the error from utterances whose reference or hypothesis
                     contains a digit (numbers written as digits on one side and as
                     words on the other).

Writes results/tables/exp2_mechanisms.csv and figures/fig3_mechanisms.png
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from whisper_normalizer.basic import BasicTextNormalizer

from asr_tier_audit import data, metrics

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 9.5})
_wb = BasicTextNormalizer()


def wb_variant(text: str, lang: str, keep_brackets: bool, fix_dotted_i: bool) -> str:
    if keep_brackets:
        text = re.sub(r"[\[\]()]", " ", text)
    if fix_dotted_i:
        text = text.replace("İ", "i")
        if lang in metrics.DOTTED_I_LANGS:
            text = text.replace("I", "ı")
    out = " ".join(_wb(text).split())
    if lang in metrics.CHAR_LANGS:
        out = " ".join(out.replace(" ", ""))
    return out


def wer_with(rows, lang, **kw) -> float:
    refs = [wb_variant(r["ref_raw"], lang, **kw) for r in rows]
    hyps = [wb_variant(r["hyp"], lang, **kw) for r in rows]
    return metrics.wer(metrics.error_counts(refs, hyps, [r["sentence_id"] for r in rows]))


def main() -> None:
    out_rows = []
    for cfg in data.available():
        lang, name = data.LANGUAGES[cfg]
        rows = data.load_manifest(cfg)
        wb = wer_with(rows, lang, keep_brackets=False, fix_dotted_i=False)
        wb_keep = wer_with(rows, lang, keep_brackets=True, fix_dotted_i=False)
        wb_keep_i = wer_with(rows, lang, keep_brackets=True, fix_dotted_i=True)
        lj = metrics.wer(data.counts_for(rows, lang, "lang_join"))
        no_digit = [r for r in rows if not re.search(r"\d", r["ref_raw"] + r["hyp"])]
        lj_no_digit = metrics.wer(data.counts_for(no_digit, lang, "lang_join"))
        out_rows.append({
            "language": name,
            "refs_with_brackets": sum(bool(re.search(r"[(\[]", r["ref_raw"])) for r in rows),
            "refs_with_dotted_I": sum("İ" in r["ref_raw"] for r in rows),
            "utts_with_digits": len(rows) - len(no_digit),
            "wer_lang_join": lj, "wer_whisper_basic": wb,
            "bracket_effect": wb - wb_keep,
            "dotted_i_effect": wb_keep - wb_keep_i,
            "other_whisper_effect": wb_keep_i - lj,
            "wer_lang_join_no_digit_utts": lj_no_digit,
        })

    out = ROOT / "results" / "tables" / "exp2_mechanisms.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)

    names = [r["language"] for r in out_rows]
    parts = [("bracket_effect", "Whisper: bracketed words deleted", "#c0392b"),
             ("dotted_i_effect", "Whisper: capital İ split", "#e67e22"),
             ("other_whisper_effect", "Whisper: other (apostrophes, combining-mark splits)", "#8e44ad")]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(names))
    h = 0.22
    for k, (key, label, color) in enumerate(parts):
        ax.barh(y + (k - 1) * h, [100 * r[key] for r in out_rows], h, color=color, label=label)
    ax.scatter([100 * (r["wer_lang_join_no_digit_utts"] - r["wer_lang_join"]) for r in out_rows],
               y + 1.6 * h, marker="D", s=16, color="#1f5fa8", zorder=3,
               label="Language-aware: drop utterances with digits")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Change in error rate vs language-aware word-level score (points)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig3_mechanisms.png", dpi=200)
    fig.savefig(Path(ROOT / "figures" / "fig3_mechanisms.png").with_suffix(".pdf"))
    print(f"wrote {out} and figures/fig3_mechanisms.png")


if __name__ == "__main__":
    main()
