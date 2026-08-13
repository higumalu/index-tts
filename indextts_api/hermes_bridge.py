#!/usr/bin/env python3
"""Hermes Agent command-type TTS bridge for the IndexTTS API service.

Hermes writes input text to a temp file and runs this script with placeholders:

    indextts-hermes-bridge --text-file {input_path} --out {output_path}

Environment variables:
    INDEXTTS_API_URL          API base URL (default: http://127.0.0.1:8001)
    INDEXTTS_DEFAULT_VOICE_ID Required voice_id registered via POST /v1/voices
    INDEXTTS_USE_EMO_TEXT     Optional, set to 1/true to enable text-based emotion
    INDEXTTS_EMO_ALPHA        Optional emotion strength (default: 0.6)
    INDEXTTS_TEMPERATURE      Optional GPT sampling temperature (default: 0.8)
    INDEXTTS_TOP_P            Optional nucleus sampling (default: 0.7)
    INDEXTTS_TOP_K            Optional top-k sampling (default: 30)
    INDEXTTS_LANG             Optional synthesis language, 2.5 only (default: zh)
    INDEXTTS_DURATION_FACTOR  Optional speed factor 0.5-2.0, 2.5 only (default: 1.0)
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes → IndexTTS API TTS bridge")
    parser.add_argument("--text-file", required=True, help="UTF-8 text file from Hermes")
    parser.add_argument("--out", required=True, help="Output audio path")
    parser.add_argument("--voice-id", default=os.getenv("INDEXTTS_DEFAULT_VOICE_ID", ""))
    parser.add_argument("--api-url", default=os.getenv("INDEXTTS_API_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--use-emo-text", action="store_true", default=_env_bool("INDEXTTS_USE_EMO_TEXT"))
    parser.add_argument("--emo-alpha", type=float, default=_env_float("INDEXTTS_EMO_ALPHA", 0.6))
    parser.add_argument("--temperature", type=float, default=_env_float("INDEXTTS_TEMPERATURE", 0.8))
    parser.add_argument("--top-p", type=float, default=_env_float("INDEXTTS_TOP_P", 0.7))
    parser.add_argument("--top-k", type=int, default=_env_int("INDEXTTS_TOP_K", 30))
    parser.add_argument("--lang", default=os.getenv("INDEXTTS_LANG", "zh"))
    parser.add_argument(
        "--duration-factor", type=float, default=_env_float("INDEXTTS_DURATION_FACTOR", 1.0)
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    if not args.voice_id:
        print("error: --voice-id or INDEXTTS_DEFAULT_VOICE_ID is required", file=sys.stderr)
        return 2

    text_path = os.path.expanduser(args.text_file)
    output_path = os.path.expanduser(args.out)

    try:
        text = open(text_path, encoding="utf-8").read().strip()
    except OSError as err:
        print(f"error: cannot read text file: {err}", file=sys.stderr)
        return 1

    if not text:
        print("error: input text is empty", file=sys.stderr)
        return 1

    api_url = args.api_url.rstrip("/")
    payload = {
        "voice_id": args.voice_id,
        "text": text,
        "use_emo_text": args.use_emo_text,
        "emo_alpha": args.emo_alpha,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "lang": args.lang.lower(),
        "duration_factor": args.duration_factor,
        "response_format": "audio",
    }

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(f"{api_url}/v1/tts", json=payload)
            response.raise_for_status()
            audio = response.content
    except httpx.HTTPStatusError as err:
        detail = err.response.text.strip()
        print(f"error: API returned {err.response.status_code}: {detail}", file=sys.stderr)
        return 1
    except httpx.RequestError as err:
        print(f"error: API request failed: {err}", file=sys.stderr)
        return 1

    if not audio:
        print("error: API returned empty audio", file=sys.stderr)
        return 1

    try:
        with open(output_path, "wb") as handle:
            handle.write(audio)
    except OSError as err:
        print(f"error: cannot write output file: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
