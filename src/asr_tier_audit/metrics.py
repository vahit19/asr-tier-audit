"""Text normalizers, per-utterance error counts, and clustered bootstrap intervals."""

import re
import unicodedata
from dataclasses import dataclass

import jiwer
import numpy as np
from whisper_normalizer.basic import BasicTextNormalizer

# Languages scored by characters rather than words (no whitespace word boundaries).
CHAR_LANGS = {"ja", "zh", "th", "lo", "my", "km"}
# Languages whose lowercasing needs the dotted/dotless i rule.
DOTTED_I_LANGS = {"tr", "az"}

_whisper_basic = BasicTextNormalizer()


def _lower(text: str, lang: str) -> str:
    if lang in DOTTED_I_LANGS:
        text = text.replace("I", "ı").replace("İ", "i")
    return text.lower()


# Malayalam chillu letters have an atomic code point and a legacy sequence
# (consonant + virama [+ ZWJ]); NFC does not unify them, so both sides are mapped
# to the sequence form. Zero-width joiners are dropped everywhere.
_CHILLU = {"ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്",
           "ൽ": "ല്", "ൾ": "ള്", "ൿ": "ക്"}


def _unify_script(text: str) -> str:
    for atomic, seq in _CHILLU.items():
        text = text.replace(atomic, seq)
    return text.replace("‍", "").replace("‌", "")


def _strip_punct(text: str, apostrophe: str) -> str:
    """Drop Unicode punctuation and symbols; apostrophes are joined or split."""
    text = re.sub(r"['’ʼ]", apostrophe, text)
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)


def normalize(text: str, lang: str, scheme: str) -> str:
    """Apply one normalizer from the family studied in the paper.

    whisper_basic     Whisper's BasicTextNormalizer, the common multilingual default.
    lang_join         language-aware lowercasing, NFC plus script-variant unification,
                      punctuation removed, apostrophes joined.
    lang_split        as lang_join, but apostrophes split a token in two.
    lang_char         as lang_join, scored on characters with spaces removed, so word
                      segmentation (compounds, clitics) no longer counts as an error.
    """
    if scheme == "whisper_basic":
        out = _whisper_basic(text)
    elif scheme in ("lang_join", "lang_split", "lang_char"):
        text = _unify_script(unicodedata.normalize("NFC", _lower(text, lang)))
        out = _strip_punct(text, " " if scheme == "lang_split" else "")
    else:
        raise ValueError(f"unknown scheme {scheme}")
    out = " ".join(out.split())
    if lang in CHAR_LANGS or scheme == "lang_char":
        out = " ".join(out.replace(" ", ""))
    return out


SCHEMES = ("whisper_basic", "lang_join", "lang_split")


@dataclass(frozen=True)
class Counts:
    errors: np.ndarray       # substitutions + deletions + insertions, per utterance
    ref_len: np.ndarray      # reference tokens per utterance
    cluster: np.ndarray      # integer sentence id per utterance


def error_counts(refs: list[str], hyps: list[str], sentence_ids: list[str]) -> Counts:
    errs, lens = [], []
    for r, h in zip(refs, hyps):
        if not r.strip():
            errs.append(len(h.split()))
            lens.append(0)
            continue
        o = jiwer.process_words(r, h)
        errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(o.substitutions + o.deletions + o.hits)
    ids = {s: i for i, s in enumerate(dict.fromkeys(sentence_ids))}
    return Counts(np.array(errs), np.array(lens), np.array([ids[s] for s in sentence_ids]))


def wer(c: Counts) -> float:
    return c.errors.sum() / c.ref_len.sum()


def cluster_totals(c: Counts) -> tuple[np.ndarray, np.ndarray]:
    """Sum errors and reference length within each sentence cluster."""
    k = c.cluster.max() + 1
    return (np.bincount(c.cluster, weights=c.errors, minlength=k),
            np.bincount(c.cluster, weights=c.ref_len, minlength=k))


def bootstrap_ci(c: Counts, clustered: bool, n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile CI for corpus WER, resampling sentences (clustered) or utterances."""
    rng = np.random.default_rng(seed)
    e, n = cluster_totals(c) if clustered else (c.errors.astype(float), c.ref_len.astype(float))
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    stats = e[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
