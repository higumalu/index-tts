from __future__ import annotations

import base64
import binascii
import io

import numpy as np
import soundfile as sf


class AudioDecodeError(ValueError):
    """Raised when an incoming BASE64 audio payload cannot be decoded as WAV."""


def decode_base64_wav(audio_base64: str) -> tuple[bytes, np.ndarray, int]:
    """Decode a BASE64-encoded WAV payload.

    Returns ``(raw_bytes, samples, sample_rate)`` where ``samples`` is a 1D float32 array.
    """
    if not audio_base64:
        raise AudioDecodeError("audio_base64 不可為空")

    try:
        raw = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as err:
        raise AudioDecodeError(f"audio_base64 不是合法 BASE64：{err}") from err

    try:
        data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as err:
        raise AudioDecodeError(f"audio_base64 無法被解析為 WAV：{err}") from err

    if data.ndim == 2:
        data = data.mean(axis=1)
    return raw, data.astype(np.float32), int(sample_rate)


def encode_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    return base64.b64encode(wav_bytes(audio, sample_rate)).decode("ascii")


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
