"""Transcribe a seeded FLEURS test sample with Scribe v2 and write a manifest.

Usage: uv run python experiments/run_scribe.py tr_tr tr --n 200
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from asr_tier_audit import fleurs, scribe

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="FLEURS config, e.g. tr_tr")
    ap.add_argument("lang", help="language code sent to the API, e.g. tr")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    utts = fleurs.sample(fleurs.load_test(args.config), args.n, args.seed)
    paths = fleurs.extract_audio(args.config, utts)

    def run(u):
        r = scribe.transcribe(paths[u.utt_id], args.lang, u.utt_id)
        return {"utt_id": u.utt_id, "sentence_id": u.sentence_id, "ref_raw": u.raw,
                "ref_normalized": u.normalized, "hyp": r.get("text", ""),
                "detected_language": r.get("language_code"),
                "duration_s": round(sf.info(str(paths[u.utt_id])).duration, 3)}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, utts))

    out = RESULTS / "manifests" / f"scribe_v2_{args.config}_n{args.n}_s{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"{len(rows)} utterances, {len({r['sentence_id'] for r in rows})} sentences -> {out}")


if __name__ == "__main__":
    main()
