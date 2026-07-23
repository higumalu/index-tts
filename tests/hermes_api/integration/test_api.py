from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from indextts_api.main import create_app
from indextts_api.tts_engine import SynthesisResult, TTSEngine
from indextts_api.voice_store import VoiceStore


class FakeTTSEngine(TTSEngine):
    def __init__(self) -> None:
        super().__init__(model_loader=lambda: object())
        self.calls: list[dict] = []
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def generate(
        self,
        *,
        text: str,
        reference_wav_path,
        use_emo_text: bool = False,
        emo_alpha: float = 0.6,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> SynthesisResult:
        self.calls.append(
            {
                "text": text,
                "reference_wav_path": str(reference_wav_path),
                "use_emo_text": use_emo_text,
                "emo_alpha": emo_alpha,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        self._ready = True
        sample_rate = 24000
        audio = np.zeros(sample_rate // 2, dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=sample_rate, duration_sec=0.5)


@pytest.fixture
def fake_engine() -> FakeTTSEngine:
    return FakeTTSEngine()


@pytest.fixture
def client(tmp_path: Path, fake_engine: FakeTTSEngine) -> Iterator[TestClient]:
    app = create_app()
    app.state.voice_store = VoiceStore(tmp_path / "voices")
    app.state.tts_engine = fake_engine
    yield TestClient(app)


def _decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    return data, sr


@pytest.mark.integration
def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is False
    assert body["voices_count"] == 0
    assert "indextts_available" in body


@pytest.mark.integration
def test_voice_crud_flow(client: TestClient, wav_base64: str) -> None:
    create_resp = client.post(
        "/v1/voices",
        json={"audio_base64": wav_base64},
    )
    assert create_resp.status_code == 201, create_resp.text
    meta = create_resp.json()
    voice_id = meta["voice_id"]
    assert meta["sample_rate"] == 24000
    assert meta["reference_text"] == ""

    list_resp = client.get("/v1/voices")
    assert list_resp.status_code == 200
    voices = list_resp.json()["voices"]
    assert len(voices) == 1
    assert voices[0]["voice_id"] == voice_id

    get_resp = client.get(f"/v1/voices/{voice_id}")
    assert get_resp.status_code == 200

    delete_resp = client.delete(f"/v1/voices/{voice_id}")
    assert delete_resp.status_code == 204

    missing_resp = client.get(f"/v1/voices/{voice_id}")
    assert missing_resp.status_code == 404


@pytest.mark.integration
def test_create_voice_rejects_invalid_base64(client: TestClient) -> None:
    resp = client.post(
        "/v1/voices",
        json={"audio_base64": "not-base64!!"},
    )
    assert resp.status_code == 400


@pytest.mark.integration
def test_tts_audio_response(
    client: TestClient,
    wav_base64: str,
    fake_engine: FakeTTSEngine,
) -> None:
    voice = client.post(
        "/v1/voices",
        json={"audio_base64": wav_base64},
    ).json()

    resp = client.post(
        "/v1/tts",
        json={
            "voice_id": voice["voice_id"],
            "text": "Hello world",
            "use_emo_text": True,
            "emo_alpha": 0.8,
            "temperature": 0.5,
            "top_p": 0.7,
            "top_k": 20,
            "response_format": "audio",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "audio/wav"
    samples, sr = _decode_wav(resp.content)
    assert sr == 24000
    assert len(samples) == 12000

    assert len(fake_engine.calls) == 1
    call = fake_engine.calls[0]
    assert call["text"] == "Hello world"
    assert call["use_emo_text"] is True
    assert call["emo_alpha"] == 0.8
    assert call["temperature"] == 0.5
    assert call["top_p"] == 0.7
    assert call["top_k"] == 20


@pytest.mark.integration
def test_tts_json_response(client: TestClient, wav_base64: str) -> None:
    voice = client.post(
        "/v1/voices",
        json={"audio_base64": wav_base64},
    ).json()

    resp = client.post(
        "/v1/tts",
        json={
            "voice_id": voice["voice_id"],
            "text": "json test",
            "response_format": "json",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sample_rate"] == 24000
    assert body["duration_sec"] == pytest.approx(0.5, abs=0.01)
    decoded = base64.b64decode(body["audio_base64"])
    samples, sr = _decode_wav(decoded)
    assert sr == 24000
    assert len(samples) == 12000


@pytest.mark.integration
def test_tts_unknown_voice_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/tts",
        json={
            "voice_id": "0" * 32,
            "text": "hi",
        },
    )
    assert resp.status_code == 404
