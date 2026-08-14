from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from indextts_api.audio import encode_wav_base64, wav_bytes
from indextts_api.dependencies import get_tts_engine, get_voice_store
from indextts_api.schemas import ResponseFormat, TTSJsonResponse, TTSRequest
from indextts_api.tts_engine import TTSEngine
from indextts_api.voice_store import VoiceNotFoundError, VoiceStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["tts"])


@router.post("/tts")
async def synthesize(
    payload: TTSRequest,
    response: Response,
    store: VoiceStore = Depends(get_voice_store),
    engine: TTSEngine = Depends(get_tts_engine),
):
    # 每個請求一個 id：客戶端拿到的 500 訊息和伺服器端的 traceback 用它對得起來。
    request_id = uuid.uuid4().hex[:12]
    id_header = {"X-Request-Id": request_id}
    response.headers.update(id_header)

    try:
        record = store.get(payload.voice_id)
    except VoiceNotFoundError as err:
        logger.warning("[%s] voice_id 不存在: %s", request_id, payload.voice_id)
        raise HTTPException(
            status_code=404,
            detail=f"voice_id 不存在: {payload.voice_id}",
            headers=id_header,
        ) from err

    logger.info(
        "[%s] 開始合成 voice_id=%s text_len=%d lang=%s duration_factor=%s "
        "text_normalization=%s use_emo_text=%s temperature=%s top_p=%s top_k=%s",
        request_id,
        payload.voice_id,
        len(payload.text),
        payload.lang.value,
        payload.duration_factor,
        payload.text_normalization,
        payload.use_emo_text,
        payload.temperature,
        payload.top_p,
        payload.top_k,
    )
    started = time.monotonic()

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
            lang=payload.lang.value,
            duration_factor=payload.duration_factor,
            text_normalization=payload.text_normalization,
        )
    except ValueError as err:
        logger.warning("[%s] 參數錯誤: %s", request_id, err)
        raise HTTPException(status_code=400, detail=str(err), headers=id_header) from err
    except FileNotFoundError as err:
        logger.error("[%s] 參考檔案不存在: %s", request_id, err)
        raise HTTPException(status_code=400, detail=str(err), headers=id_header) from err
    except Exception as err:
        # 這裡曾經只把 str(err) 丟回客戶端、伺服器端一個字都沒留，導致 2026-08-12
        # 那場 CUDA context 中毒事故（連續 16.5 小時全數 500）完全無法從日誌追查。
        logger.exception(
            "[%s] TTS 推論失敗 voice_id=%s text_len=%d 耗時 %.2fs",
            request_id,
            payload.voice_id,
            len(payload.text),
            time.monotonic() - started,
        )
        raise HTTPException(
            status_code=500,
            detail=f"TTS 推論失敗，請提供 request_id 供查閱日誌: {request_id}",
            headers=id_header,
        ) from err

    elapsed = time.monotonic() - started
    logger.info(
        "[%s] 合成完成 audio=%.2fs 耗時 %.2fs RTF=%.2f",
        request_id,
        result.duration_sec,
        elapsed,
        elapsed / result.duration_sec if result.duration_sec else 0.0,
    )

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
        "X-Request-Id": request_id,
    }
    return Response(content=audio, media_type="audio/wav", headers=headers)
