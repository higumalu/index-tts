from __future__ import annotations

from fastapi import Request

from indextts_api.tts_engine import TTSEngine
from indextts_api.voice_store import VoiceStore


def get_voice_store(request: Request) -> VoiceStore:
    return request.app.state.voice_store


def get_tts_engine(request: Request) -> TTSEngine:
    return request.app.state.tts_engine
