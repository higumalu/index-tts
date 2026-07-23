from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from indextts_api.audio import encode_wav_base64, wav_bytes
from indextts_api.dependencies import get_tts_engine, get_voice_store
from indextts_api.schemas import ResponseFormat, TTSJsonResponse, TTSRequest
from indextts_api.tts_engine import TTSEngine
from indextts_api.voice_store import VoiceNotFoundError, VoiceStore

router = APIRouter(prefix="/v1", tags=["tts"])


@router.post("/tts")
async def synthesize(
    payload: TTSRequest,
    store: VoiceStore = Depends(get_voice_store),
    engine: TTSEngine = Depends(get_tts_engine),
):
    try:
        record = store.get(payload.voice_id)
    except VoiceNotFoundError as err:
        raise HTTPException(status_code=404, detail=f"voice_id 不存在: {payload.voice_id}") from err

    try:
        result = await asyncio.to_thread(
            engine.generate,
            text=payload.text,
            reference_wav_path=record.audio_path,
            use_emo_text=payload.use_emo_text,
            emo_alpha=payload.emo_alpha,
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except FileNotFoundError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"TTS 推論失敗: {err}") from err

    if payload.response_format == ResponseFormat.json:
        return TTSJsonResponse(
            audio_base64=encode_wav_base64(result.audio, result.sample_rate),
            sample_rate=result.sample_rate,
            duration_sec=result.duration_sec,
        )

    audio = wav_bytes(result.audio, result.sample_rate)
    headers = {
        "Content-Disposition": f'attachment; filename="{payload.voice_id}.wav"',
        "X-Sample-Rate": str(result.sample_rate),
        "X-Duration-Sec": f"{result.duration_sec:.6f}",
    }
    return Response(content=audio, media_type="audio/wav", headers=headers)
