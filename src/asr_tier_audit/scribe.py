"""ElevenLabs Scribe v2 batch transcription with an append-only on-disk cache."""

import json
import os
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL_ID = "scribe_v2"
CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "raw" / MODEL_ID


def _api_key() -> str:
    load_dotenv(find_dotenv(usecwd=True))
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    return key


def transcribe(wav_path: Path, language_code: str, utt_id: str) -> dict:
    """Return the API response for one file, reusing the cached copy if present."""
    cache_file = CACHE_DIR / language_code / f"{utt_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    with open(wav_path, "rb") as f:
        resp = requests.post(
            API_URL,
            headers={"xi-api-key": _api_key()},
            data={
                "model_id": MODEL_ID,
                "language_code": language_code,
                "tag_audio_events": "false",
                "timestamps_granularity": "none",
            },
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=120,
        )
    resp.raise_for_status()
    result = resp.json()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
