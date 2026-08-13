from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from indextts_api.tts_engine import TTSEngine

SAMPLE_RATE = 22050


class FakeModel:
    """替身 IndexTTS2：寫出真的 wav，並記錄 infer 是否被併發呼叫。"""

    def __init__(self, infer_duration_sec: float = 0.02) -> None:
        self.infer_duration_sec = infer_duration_sec
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self.infer_kwargs: list[dict] = []
        self._counter_lock = threading.Lock()
        # 對應 IndexTTS2 的參考音快取，用來驗證 unload 會清掉
        self.cache_spk_cond = "sentinel"
        self.cache_mel = "sentinel"

    def infer(self, *, output_path: str, **kwargs) -> None:
        with self._counter_lock:
            self._active += 1
            self.calls += 1
            self.infer_kwargs.append(dict(kwargs))
            self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            time.sleep(self.infer_duration_sec)
            sf.write(
                output_path,
                np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
                SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )
        finally:
            with self._counter_lock:
                self._active -= 1


@pytest.fixture
def reference_wav(tmp_path: Path, wav_bytes: bytes) -> Path:
    path = tmp_path / "reference.wav"
    path.write_bytes(wav_bytes)
    return path


def _engine(model: FakeModel, **kwargs) -> tuple[TTSEngine, list[int]]:
    load_count = [0]

    def loader():
        load_count[0] += 1
        return model

    defaults = {"idle_unload_sec": 0.0, "idle_check_interval_sec": 0.01}
    defaults.update(kwargs)
    return TTSEngine(loader, **defaults), load_count


@pytest.mark.unit
def test_lang_and_duration_factor_reach_the_model(reference_wav: Path) -> None:
    """lang 是 IndexTTS 2.5 的必填參數，少了它 infer() 會 TypeError。"""
    model = FakeModel()
    engine, _ = _engine(model)

    engine.generate(
        text="hi", reference_wav_path=reference_wav, lang="JA", duration_factor=1.5
    )

    kwargs = model.infer_kwargs[-1]
    assert kwargs["lang"] == "ja", "lang 必須小寫化後傳給模型"
    assert kwargs["duration_factor"] == 1.5


@pytest.mark.unit
def test_lang_defaults_come_from_settings(reference_wav: Path) -> None:
    model = FakeModel()
    engine, _ = _engine(model)

    engine.generate(text="hi", reference_wav_path=reference_wav)

    kwargs = model.infer_kwargs[-1]
    assert kwargs["lang"] == "zh"
    assert kwargs["duration_factor"] == 1.0


@pytest.mark.unit
def test_model_is_lazy_and_loads_once(reference_wav: Path) -> None:
    model = FakeModel()
    engine, load_count = _engine(model)

    assert engine.is_ready is False
    assert engine.idle_sec() is None
    assert load_count[0] == 0

    engine.generate(text="hi", reference_wav_path=reference_wav)
    engine.generate(text="hi again", reference_wav_path=reference_wav)

    assert engine.is_ready is True
    assert load_count[0] == 1
    assert model.calls == 2


@pytest.mark.unit
def test_concurrent_generate_is_serialized(reference_wav: Path) -> None:
    """24 個併發請求曾在生產環境毀掉 CUDA context；infer 必須永不重疊。"""
    model = FakeModel(infer_duration_sec=0.03)
    engine, _ = _engine(model)

    with ThreadPoolExecutor(max_workers=24) as pool:
        results = list(
            pool.map(
                lambda i: engine.generate(text=f"text-{i}", reference_wav_path=reference_wav),
                range(24),
            )
        )

    assert len(results) == 24
    assert all(r.sample_rate == SAMPLE_RATE for r in results)
    assert model.calls == 24
    assert model.max_concurrent == 1, f"infer 被併發呼叫了 {model.max_concurrent} 次"


@pytest.mark.unit
def test_unload_releases_model_and_clears_caches(reference_wav: Path) -> None:
    model = FakeModel()
    engine, load_count = _engine(model)

    assert engine.unload() is False  # 還沒載入時是 no-op

    engine.generate(text="hi", reference_wav_path=reference_wav)
    assert engine.is_ready is True

    assert engine.unload() is True
    assert engine.is_ready is False
    assert engine.idle_sec() is None
    assert model.cache_spk_cond is None
    assert model.cache_mel is None

    # 釋放後仍能服務，只是要重新載入
    engine.generate(text="again", reference_wav_path=reference_wav)
    assert engine.is_ready is True
    assert load_count[0] == 2


@pytest.mark.unit
def test_unload_if_idle_respects_threshold(reference_wav: Path) -> None:
    model = FakeModel()
    engine, _ = _engine(model, idle_unload_sec=60.0)

    engine.generate(text="hi", reference_wav_path=reference_wav)
    assert engine.unload_if_idle() is False  # 才剛用過
    assert engine.is_ready is True

    engine._idle_unload_sec = 0.001
    time.sleep(0.01)
    assert engine.unload_if_idle() is True
    assert engine.is_ready is False


@pytest.mark.unit
def test_idle_reaper_unloads_without_traffic(reference_wav: Path) -> None:
    model = FakeModel()
    engine, _ = _engine(model, idle_unload_sec=0.05, idle_check_interval_sec=0.01)

    engine.start_idle_reaper()
    try:
        engine.generate(text="hi", reference_wav_path=reference_wav)
        assert engine.is_ready is True

        deadline = time.monotonic() + 5.0
        while engine.is_ready and time.monotonic() < deadline:
            time.sleep(0.01)

        assert engine.is_ready is False, "閒置超過門檻後模型仍未被釋放"
    finally:
        engine.stop_idle_reaper()


@pytest.mark.unit
def test_idle_reaper_disabled_when_threshold_not_positive(reference_wav: Path) -> None:
    model = FakeModel()
    engine, _ = _engine(model, idle_unload_sec=0.0)

    engine.start_idle_reaper()
    try:
        engine.generate(text="hi", reference_wav_path=reference_wav)
        time.sleep(0.05)
        assert engine.is_ready is True  # 停用時永不釋放
        assert engine.unload_if_idle() is False
    finally:
        engine.stop_idle_reaper()


@pytest.mark.unit
def test_health_probes_do_not_reset_idle_timer(reference_wav: Path) -> None:
    """/v1/health 每 30 秒打一次，若它重置閒置計時，模型就永遠不會被釋放。"""
    from indextts_api.tts_engine import indextts_importable

    model = FakeModel()
    engine, _ = _engine(model, idle_unload_sec=60.0)
    engine.generate(text="hi", reference_wav_path=reference_wav)

    last_used = engine._last_used
    for _ in range(3):
        engine.is_ready
        engine.idle_sec()
        indextts_importable()
    assert engine._last_used == last_used


@pytest.mark.unit
def test_reaper_does_not_unload_during_inference(reference_wav: Path) -> None:
    """推論進行中不能被背景釋放，否則會對已釋放的張量做運算。"""
    model = FakeModel(infer_duration_sec=0.3)
    engine, _ = _engine(model, idle_unload_sec=0.001, idle_check_interval_sec=0.01)

    engine.start_idle_reaper()
    try:
        result = engine.generate(text="hi", reference_wav_path=reference_wav)
        assert result.sample_rate == SAMPLE_RATE
        assert model.max_concurrent == 1
    finally:
        engine.stop_idle_reaper()
