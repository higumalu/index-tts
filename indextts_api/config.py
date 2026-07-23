from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    # Path to the index-tts repository (added to sys.path for imports).
    # Defaults to this repo root when hermes_api lives inside index-tts.
    repo_path: Path = Path(
        os.getenv("INDEXTTS_REPO_PATH", ".")
    ).expanduser().resolve()
    model_dir: Path = Path(
        os.getenv("INDEXTTS_MODEL_DIR", "./checkpoints")
    ).expanduser().resolve()
    cfg_path: Path = Path(
        os.getenv("INDEXTTS_CFG_PATH", "./checkpoints/config.yaml")
    ).expanduser().resolve()
    use_fp16: bool = _bool_env("INDEXTTS_USE_FP16", True)
    use_cuda_kernel: bool = _bool_env("INDEXTTS_USE_CUDA_KERNEL", False)
    use_deepspeed: bool = _bool_env("INDEXTTS_USE_DEEPSPEED", False)
    voices_dir: Path = Path(os.getenv("INDEXTTS_VOICES_DIR", "./voices")).expanduser().resolve()
    api_base_url: str = os.getenv("INDEXTTS_API_URL", "http://127.0.0.1:8001")
    default_voice_id: str = os.getenv("INDEXTTS_DEFAULT_VOICE_ID", "")
    # Sampling defaults aligned with scripts/synthesize_speech.py
    temperature: float = _float_env("INDEXTTS_TEMPERATURE", 0.8)
    top_p: float = _float_env("INDEXTTS_TOP_P", 0.7)
    top_k: int = _int_env("INDEXTTS_TOP_K", 30)


settings = Settings()
