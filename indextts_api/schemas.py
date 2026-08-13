from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from indextts_api.config import settings


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
    temperature: float = Field(
        default_factory=lambda: settings.temperature,
        ge=0.1,
        le=2.0,
        description="GPT 採樣溫度；越低越穩，雜訊感通常較少",
    )
    top_p: float = Field(
        default_factory=lambda: settings.top_p,
        ge=0.0,
        le=1.0,
        description="nucleus sampling：累積機率門檻",
    )
    top_k: int = Field(
        default_factory=lambda: settings.top_k,
        ge=0,
        le=100,
        description="每次只從機率最高的 K 個 token 抽樣",
    )
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
    model_idle_sec: float | None = Field(
        default=None,
        description="模型載入／上次推論至今的秒數；模型未載入時為 null",
    )
    idle_unload_sec: float = Field(
        default=0.0,
        description="閒置自動釋放 VRAM 的門檻秒數；0 表示停用",
    )
