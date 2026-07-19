from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import soundfile as sf

from indextts_api.audio import decode_base64_wav, encode_wav_base64, wav_bytes


def _make_wav_bytes(sample_rate: int = 24000) -> bytes:
    t = np.linspace(0, 0.5, sample_rate // 2, endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.mark.unit
def test_decode_and_encode_roundtrip() -> None:
    raw = _make_wav_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    decoded_raw, samples, sr = decode_base64_wav(b64)
    assert sr == 24000
    assert len(samples) == 12000
    assert decoded_raw == raw

    reencoded = encode_wav_base64(samples, sr)
    _, samples2, sr2 = decode_base64_wav(reencoded)
    assert sr2 == sr
    assert len(samples2) == len(samples)


@pytest.mark.unit
def test_wav_bytes() -> None:
    audio = np.zeros(1000, dtype=np.float32)
    payload = wav_bytes(audio, 24000)
    data, sr = sf.read(io.BytesIO(payload), dtype="float32")
    assert sr == 24000
    assert len(data) == 1000
