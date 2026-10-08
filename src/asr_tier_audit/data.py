"""Load transcription manifests and turn them into error counts per normalizer."""

import json
from pathlib import Path

from asr_tier_audit import metrics

RESULTS = Path(__file__).resolve().parents[2] / "results"

# FLEURS config -> (language code, display name). All are listed by the vendor as
# "Excellent (<= 5% WER)" for Scribe v2 (checked 2026-10-08).
LANGUAGES = {
    "tr_tr": ("tr", "Turkish"),
    "fi_fi": ("fi", "Finnish"),
    "hu_hu": ("hu", "Hungarian"),
    "ml_in": ("ml", "Malayalam"),
    "kn_in": ("kn", "Kannada"),
    "ja_jp": ("ja", "Japanese"),
}


def load_manifest(config: str, n: int = 200, seed: int = 0) -> list[dict]:
    path = RESULTS / "manifests" / f"scribe_v2_{config}_n{n}_s{seed}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def counts_for(rows: list[dict], lang: str, scheme: str) -> metrics.Counts:
    refs = [metrics.normalize(r["ref_raw"], lang, scheme) for r in rows]
    hyps = [metrics.normalize(r["hyp"], lang, scheme) for r in rows]
    return metrics.error_counts(refs, hyps, [r["sentence_id"] for r in rows])


def available() -> list[str]:
    return [c for c in LANGUAGES
            if (RESULTS / "manifests" / f"scribe_v2_{c}_n200_s0.jsonl").exists()]
