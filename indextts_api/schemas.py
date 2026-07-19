from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResponseFormat(str, Enum):
    audio = "audio"
    json = "json"


class VoiceCreateRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1, description="BASE64 編碼的 WAV 檔內容")
    reference_text: str = Field(
        default="",
        description="參考音檔逐字稿（IndexTTS2 可選，保留供日後擴充）",
    )


class VoiceMetadata(BaseModel):
    voice_id: str
    reference_text: str
    sample_rate: int
    duration_sec: float
    created_at: float


class VoiceListResponse(BaseModel):
    voices: list[VoiceMetadata]


class TTSRequest(BaseModel):
    voice_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    use_emo_text: bool = False
    emo_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    response_format: ResponseFormat = ResponseFormat.audio


class TTSJsonResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_sec: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    voices_count: int
    indextts_available: bool
