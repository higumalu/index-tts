from __future__ import annotations

import pytest

from indextts_api import tts_engine
from indextts_api.tts_engine import GpuStatus, probe_gpu


@pytest.fixture(autouse=True)
def reset_probe_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """探測會記住上次狀態以免洗版，測試之間要清掉。"""
    monkeypatch.setattr(tts_engine, "_last_gpu_healthy", None, raising=False)


@pytest.mark.unit
def test_probe_never_creates_a_cuda_context() -> None:
    """探測若自己建立 context，模型未載入時也會吃掉數百 MB VRAM。"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("此機器沒有 CUDA")
    if torch.cuda.is_initialized():
        pytest.skip("context 已存在，測不出這個性質")

    status = probe_gpu()

    assert torch.cuda.is_initialized() is False, "probe_gpu() 不該建立 CUDA context"
    assert status.available is True
    assert status.initialized is False
    assert status.healthy is None, "沒有 context 時不該回報成功或失敗"


@pytest.mark.unit
def test_probe_reports_failure_when_cuda_ops_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA context 中毒後，任何運算都會立刻爆掉。"""
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def poisoned(*args, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(torch, "zeros", poisoned)

    status = probe_gpu()

    assert status.healthy is False
    assert "device-side assert" in (status.detail or "")


@pytest.mark.unit
def test_probe_failure_is_logged_once_per_transition(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    import logging

    with caplog.at_level(logging.ERROR, logger="indextts_api.tts_engine"):
        for _ in range(5):
            probe_gpu()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, f"每次探測都記錄會洗版，實際記了 {len(errors)} 次"


@pytest.mark.unit
def test_gpu_status_shape() -> None:
    status = GpuStatus(available=False, initialized=False, healthy=None, detail="x")
    assert status.device_name is None
    assert status.memory_used_mb is None
