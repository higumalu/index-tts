from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


@pytest.fixture(autouse=True)
def isolate_indextts_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉環境裡的 INDEXTTS_*，否則開發機的 .env 會滲進測試。

    例如 shell 有 INDEXTTS_DEFAULT_VOICE_ID 時，hermes_bridge 的 argparse
    預設值就會拿到它，「缺少 voice_id 要報錯」的測試不但會失敗，還會對本機
    真實服務發出一次 TTS 請求。
    """
    for key in [k for k in os.environ if k.startswith("INDEXTTS_")]:
        monkeypatch.delenv(key, raising=False)


def _make_wav_bytes(*, sample_rate: int = 24000, duration_sec: float = 1.0, freq: float = 220.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wav = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return _make_wav_bytes()


@pytest.fixture
def wav_base64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("ascii")
