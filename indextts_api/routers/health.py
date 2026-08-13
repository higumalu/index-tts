from __future__ import annotations

from fastapi import APIRouter, Depends

from indextts_api.dependencies import get_tts_engine, get_voice_store
from indextts_api.schemas import HealthResponse
from indextts_api.tts_engine import TTSEngine, indextts_importable
from indextts_api.voice_store import VoiceStore

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    store: VoiceStore = Depends(get_voice_store),
    engine: TTSEngine = Depends(get_tts_engine),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=engine.is_ready,
        voices_count=store.count(),
        indextts_available=indextts_importable(),
        model_idle_sec=engine.idle_sec(),
        idle_unload_sec=engine.idle_unload_sec,
    )
