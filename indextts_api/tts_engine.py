from __future__ import annotations

import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from indextts_api.config import settings


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_sec: float


def indextts_importable() -> bool:
    """Return True when the index-tts package can be imported."""
    repo_path = settings.repo_path
    if repo_path.is_dir() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        import indextts.infer_v2  # noqa: F401
    except ImportError:
        return False
    return True


class TTSEngine:
    """Thin wrapper around IndexTTS2 with lazy model loading."""

    def __init__(self, model_loader: Callable[[], Any] | None = None) -> None:
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._model_loader = model_loader or self._default_model_loader

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

        from indextts.infer_v2 import IndexTTS2

        return IndexTTS2(
            cfg_path=str(settings.cfg_path),
            model_dir=str(settings.model_dir),
            use_fp16=settings.use_fp16,
            use_cuda_kernel=settings.use_cuda_kernel,
            use_deepspeed=settings.use_deepspeed,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._model_loader()
        return self._model

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
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise ValueError("text 不可為空")

        ref_path = Path(reference_wav_path)
        if not ref_path.is_file():
            raise FileNotFoundError(f"參考音檔不存在: {ref_path}")

        model = self._get_model()
        infer_kwargs: dict[str, Any] = {
            "spk_audio_prompt": str(ref_path),
            "text": text,
            "verbose": False,
            "temperature": settings.temperature if temperature is None else temperature,
            "top_p": settings.top_p if top_p is None else top_p,
            "top_k": settings.top_k if top_k is None else top_k,
        }
        if use_emo_text:
            infer_kwargs["use_emo_text"] = True
            infer_kwargs["emo_alpha"] = emo_alpha

        with self._lock, tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            model.infer(output_path=output_path, **infer_kwargs)
            audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
        finally:
            Path(output_path).unlink(missing_ok=True)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        duration_sec = float(len(audio) / sample_rate) if sample_rate else 0.0
        return SynthesisResult(audio=audio, sample_rate=int(sample_rate), duration_sec=duration_sec)
