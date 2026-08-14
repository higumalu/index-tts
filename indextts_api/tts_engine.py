from __future__ import annotations

import gc
import logging
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from indextts_api.config import settings

logger = logging.getLogger(__name__)

# IndexTTS2 內部快取「上一次的參考音條件」的屬性；unload 時要一併清掉，
# 否則這些 GPU 張量可能被 reference cycle 留住，VRAM 還不回去。
_MODEL_CACHE_ATTRS = (
    "cache_spk_cond",
    "cache_s2mel_style",
    "cache_s2mel_prompt",
    "cache_spk_audio_prompt",
    "cache_emo_cond",
    "cache_emo_audio_prompt",
    "cache_mel",
)


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_sec: float


def indextts_importable() -> bool:
    """Return True when the IndexTTS 2.5 inference module can be imported."""
    repo_path = settings.repo_path
    if repo_path.is_dir() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        import indextts.infer_v2_5  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class GpuStatus:
    available: bool  # 這台機器看得到 CUDA 裝置嗎
    initialized: bool  # 本 process 已經建立 CUDA context 了嗎
    healthy: bool | None  # None = 還沒有 context，沒東西可驗
    detail: str | None = None
    device_name: str | None = None
    memory_used_mb: int | None = None


# 只在健康狀態翻轉時記錄，否則每 30 秒一次的 health check 會洗版。
_last_gpu_healthy: bool | None = None


def probe_gpu() -> GpuStatus:
    """實際對 GPU 下一個小運算，確認 CUDA context 還活著。

    device-side assert（例如併發推論踩壞共享狀態）會讓整個 CUDA context 中毒，
    此後所有 CUDA 呼叫立刻失敗，但模型物件還在、`/v1/health` 也照樣回 200。
    2026-08-12 就是這樣連續 16.5 小時全數 500 卻顯示健康，只能靠人工重啟。
    """
    global _last_gpu_healthy

    try:
        import torch
    except ImportError:
        return GpuStatus(available=False, initialized=False, healthy=None, detail="torch 未安裝")

    try:
        if not torch.cuda.is_available():
            return GpuStatus(available=False, initialized=False, healthy=None, detail="無 CUDA 裝置")
        if not torch.cuda.is_initialized():
            # 這裡不主動建立 context：那會讓模型未載入時也吃掉數百 MB VRAM，
            # 破壞 lazy load 與閒置釋放的效果。沒有 context 就沒有中毒的可能。
            return GpuStatus(available=True, initialized=False, healthy=None)
    except Exception as err:
        return GpuStatus(
            available=False, initialized=False, healthy=False, detail=f"{type(err).__name__}: {err}"
        )

    try:
        # .item() 會 synchronize，非同步累積的 CUDA 錯誤會在這裡爆出來。
        probe = torch.zeros(1, device="cuda")
        if float((probe + 1).item()) != 1.0:
            raise RuntimeError("GPU 運算結果不正確")
        status = GpuStatus(
            available=True,
            initialized=True,
            healthy=True,
            device_name=torch.cuda.get_device_name(),
            memory_used_mb=int(torch.cuda.memory_reserved() / 2**20),
        )
    except Exception as err:
        status = GpuStatus(
            available=True,
            initialized=True,
            healthy=False,
            detail=f"{type(err).__name__}: {err}",
        )

    if status.healthy != _last_gpu_healthy:
        if status.healthy:
            if _last_gpu_healthy is False:
                logger.info("GPU 恢復正常")
        else:
            logger.error(
                "GPU 探測失敗，CUDA context 可能已損毀，服務需要重啟：%s", status.detail
            )
        _last_gpu_healthy = status.healthy
    return status


def _cuda_reserved_bytes() -> int | None:
    """Bytes currently held by torch's CUDA caching allocator, or None if no CUDA context."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        return int(torch.cuda.memory_reserved())
    except Exception:  # torch 缺席、或 context 已損毀
        return None


def _release_cuda_cache() -> int | None:
    """Return the caching allocator's free blocks to the driver (what nvidia-smi sees)."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return int(torch.cuda.memory_reserved())
    except Exception:
        logger.exception("釋放 CUDA cache 失敗")
        return None


class TTSEngine:
    """IndexTTS2 wrapper：lazy 載入、序列化推論、閒置自動釋放 VRAM。"""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        *,
        idle_unload_sec: float | None = None,
        idle_check_interval_sec: float | None = None,
    ) -> None:
        self._model: Any | None = None
        # RLock：unload() 可能在已持有鎖的區段內被呼叫。
        self._lock = threading.RLock()
        self._model_loader = model_loader or self._default_model_loader
        self._idle_unload_sec = (
            settings.idle_unload_sec if idle_unload_sec is None else idle_unload_sec
        )
        self._idle_check_interval_sec = (
            settings.idle_check_interval_sec
            if idle_check_interval_sec is None
            else idle_check_interval_sec
        )
        self._last_used: float | None = None
        self._stop_event = threading.Event()
        self._reaper: threading.Thread | None = None

    @staticmethod
    def _default_model_loader() -> Any:
        repo_path = settings.repo_path
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"IndexTTS repo 不存在: {repo_path}。"
                "請設定 INDEXTTS_REPO_PATH 指向 clone 的 index-tts 目錄。"
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        if not settings.cfg_path.is_file():
            raise FileNotFoundError(f"找不到 config: {settings.cfg_path}")
        if not settings.model_dir.is_dir():
            raise FileNotFoundError(f"找不到 model_dir: {settings.model_dir}")

        from indextts.infer_v2_5 import IndexTTS2

        # use_qwen_emo 上游預設 False，但那樣 use_emo_text=True 會直接報錯，
        # 所以交給設定決定（預設開啟）。
        return IndexTTS2(
            cfg_path=str(settings.cfg_path),
            model_dir=str(settings.model_dir),
            use_bf16=settings.use_bf16,
            use_cuda_kernel=settings.use_cuda_kernel,
            use_deepspeed=settings.use_deepspeed,
            use_qwen_emo=settings.use_qwen_emo,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def idle_unload_sec(self) -> float:
        return self._idle_unload_sec

    def idle_sec(self) -> float | None:
        """模型載入／上次使用至今的秒數；模型未載入時為 None。"""
        last_used = self._last_used
        if last_used is None:
            return None
        return time.monotonic() - last_used

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                logger.info("開始載入 IndexTTS2 模型…")
                started = time.monotonic()
                self._model = self._model_loader()
                # 載入即視為「剛使用過」，否則載入後推論失敗的模型永遠不會被回收。
                self._last_used = time.monotonic()
                logger.info(
                    "模型載入完成，耗時 %.1fs（VRAM reserved %s）",
                    time.monotonic() - started,
                    _format_bytes(_cuda_reserved_bytes()),
                )
            return self._model

    def unload(self, *, reason: str = "manual") -> bool:
        """釋放模型與其 VRAM。回傳 True 表示這次確實釋放了一個已載入的模型。"""
        with self._lock:
            if self._model is None:
                return False

            before = _cuda_reserved_bytes()
            model = self._model
            self._model = None
            self._last_used = None

            for attr in _MODEL_CACHE_ATTRS:
                if hasattr(model, attr):
                    setattr(model, attr, None)
            del model
            gc.collect()
            after = _release_cuda_cache()

            logger.info(
                "已釋放模型（原因：%s）VRAM reserved %s → %s",
                reason,
                _format_bytes(before),
                _format_bytes(after),
            )
            return True

    def start_idle_reaper(self) -> None:
        """啟動背景執行緒，閒置超過門檻就自動 unload。"""
        if self._idle_unload_sec <= 0:
            logger.info("閒置自動釋放已停用（INDEXTTS_IDLE_UNLOAD_SEC<=0）")
            return
        if self._reaper is not None:
            return
        self._stop_event.clear()
        self._reaper = threading.Thread(
            target=self._idle_loop,
            name="indextts-idle-unload",
            daemon=True,
        )
        self._reaper.start()
        logger.info(
            "閒置自動釋放已啟動：門檻 %.0fs，檢查間隔 %.0fs",
            self._idle_unload_sec,
            self._idle_check_interval_sec,
        )

    def stop_idle_reaper(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        reaper, self._reaper = self._reaper, None
        if reaper is not None:
            reaper.join(timeout=timeout)

    def _idle_loop(self) -> None:
        while not self._stop_event.wait(self._idle_check_interval_sec):
            try:
                self.unload_if_idle()
            except Exception:
                logger.exception("閒置檢查失敗")

    def unload_if_idle(self) -> bool:
        """閒置時間超過門檻就 unload。回傳是否真的釋放了。"""
        if self._idle_unload_sec <= 0:  # 停用
            return False
        if self._model is None:  # 免鎖的快速路徑
            return False
        with self._lock:
            # 必須在鎖內重新判斷：等鎖期間可能剛好完成一次推論。
            if self._model is None or self._last_used is None:
                return False
            idle = time.monotonic() - self._last_used
            if idle < self._idle_unload_sec:
                return False
            return self.unload(reason=f"閒置 {idle:.0f}s")

    def generate(
        self,
        *,
        text: str,
        reference_wav_path: Path | str,
        use_emo_text: bool = False,
        emo_alpha: float = 0.6,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        lang: str | None = None,
        duration_factor: float | None = None,
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise ValueError("text 不可為空")

        ref_path = Path(reference_wav_path)
        if not ref_path.is_file():
            raise FileNotFoundError(f"參考音檔不存在: {ref_path}")

        infer_kwargs: dict[str, Any] = {
            "spk_audio_prompt": str(ref_path),
            "text": text,
            "verbose": False,
            "temperature": settings.temperature if temperature is None else temperature,
            "top_p": settings.top_p if top_p is None else top_p,
            "top_k": settings.top_k if top_k is None else top_k,
            # lang 是 2.5 的必填參數。
            "lang": (settings.lang if lang is None else lang).lower(),
            "duration_factor": (
                settings.duration_factor if duration_factor is None else duration_factor
            ),
        }
        if use_emo_text:
            infer_kwargs["use_emo_text"] = True
            infer_kwargs["emo_alpha"] = emo_alpha

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        # 整段推論必須序列化。IndexTTS2 會共用 cache_spk_cond / cache_s2mel_style /
        # cache_mel 等狀態並以 check-then-set 更新，併發呼叫 infer() 會互相覆寫，
        # 輕則輸出錯誤音色，重則在 GPU 端觸發 index_select 越界 assert，毀掉整個
        # CUDA context（此後所有請求都失敗，只能重啟 process）。
        # 鎖同時擋住背景 unload，避免推論中途模型被釋放。
        try:
            with self._lock:
                model = self._get_model()
                model.infer(output_path=output_path, **infer_kwargs)
                audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
                self._last_used = time.monotonic()
        finally:
            Path(output_path).unlink(missing_ok=True)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        duration_sec = float(len(audio) / sample_rate) if sample_rate else 0.0
        return SynthesisResult(audio=audio, sample_rate=int(sample_rate), duration_sec=duration_sec)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 2**20:.0f}MiB"
