from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, Response, status

from indextts_api.dependencies import get_tts_engine, get_voice_store
from indextts_api.schemas import GpuStatusResponse, HealthResponse
from indextts_api.tts_engine import TTSEngine, indextts_importable, probe_gpu
from indextts_api.voice_store import VoiceStore

router = APIRouter(prefix="/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "GPU 探測失敗"}},
)
def health(
    response: Response,
    store: VoiceStore = Depends(get_voice_store),
    engine: TTSEngine = Depends(get_tts_engine),
) -> HealthResponse:
    gpu = probe_gpu()

    # healthy is None 代表還沒有 CUDA context（模型未載入），那是正常狀態。
    # 只有明確探測失敗才算 degraded：回 503 讓 docker healthcheck（curl -fsS）
    # 翻成 unhealthy，否則 CUDA context 中毒時服務會一直假裝健康。
    degraded = gpu.healthy is False
    if degraded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="degraded" if degraded else "ok",
        model_ready=engine.is_ready,
        voices_count=store.count(),
        indextts_available=indextts_importable(),
        model_idle_sec=engine.idle_sec(),
        idle_unload_sec=engine.idle_unload_sec,
        gpu=GpuStatusResponse(**dataclasses.asdict(gpu)),
    )
