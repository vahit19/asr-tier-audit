"""FLEURS test split: download, load transcripts, draw a seeded sample, extract audio."""

import csv
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fleurs"


@dataclass(frozen=True)
class Utterance:
    utt_id: str          # wav file stem, unique per recording
    sentence_id: str     # shared by every speaker who read the same sentence
    raw: str             # transcription as written, with case and punctuation
    normalized: str      # FLEURS-provided normalized transcription


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def load_test(config: str) -> list[Utterance]:
    """Load the test transcripts for a FLEURS config such as 'tr_tr'."""
    tsv = DATA_DIR / config / "test.tsv"
    _download(f"{BASE_URL}/{config}/test.tsv", tsv)
    utts = []
    with open(tsv, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            utts.append(Utterance(utt_id=Path(row[1]).stem, sentence_id=row[0],
                                  raw=row[2], normalized=row[3]))
    return utts


def sample(utts: list[Utterance], n: int, seed: int = 0) -> list[Utterance]:
    """Seeded uniform sample of n recordings, in a stable order."""
    ordered = sorted(utts, key=lambda u: u.utt_id)
    if n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n), key=lambda u: u.utt_id)


def extract_audio(config: str, utts: list[Utterance]) -> dict[str, Path]:
    """Extract only the needed wav files from the test archive; return id -> path."""
    archive = DATA_DIR / config / "test.tar.gz"
    _download(f"{BASE_URL}/{config}/audio/test.tar.gz", archive)
    out_dir = DATA_DIR / config / "wav"
    wanted = {u.utt_id for u in utts}
    paths = {uid: out_dir / "test" / f"{uid}.wav" for uid in wanted}
    missing = {uid for uid, p in paths.items() if not p.exists()}
    if missing:
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if Path(m.name).stem in missing]
            tar.extractall(out_dir, members=members, filter="data")
    return paths
