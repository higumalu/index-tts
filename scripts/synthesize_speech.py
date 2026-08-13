#!/usr/bin/env python3
"""最簡單的 IndexTTS 2.5 腳本：直接執行就輸出 wav。

用法：
1. 只改下面「可調整參數區」的常數
2. 執行：uv run python scripts/synthesize_speech.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ========= 可調整參數區（直接改這裡） =========
# 此腳本位於 scripts/，repo root 為上一層（與 indextts/、checkpoints_2.5/ 同層）
_BASE_DIR = Path(__file__).resolve().parent.parent
INDEXTTS_REPO_PATH = str(_BASE_DIR)
MODEL_DIR = str(_BASE_DIR / "checkpoints_2.5")
CFG_PATH = str(_BASE_DIR / "checkpoints_2.5" / "config.yaml")

USE_BF16 = True
USE_CUDA_KERNEL = False
USE_DEEPSPEED = False
VERBOSE = False

# 聲線參考音（不存在時會自動補到 <INDEXTTS_REPO_PATH>/examples/ 下找）
SPK_AUDIO = str(_BASE_DIR / "reference_voice" / "temp_voice_1.wav")

# 合成語言：zh / en / ja / ar / es
LANG = "zh"
# 語速／時長係數：0.5 快 ~ 2.0 慢
DURATION_FACTOR = 1.0

# 要合成的文字
TEXT = "你好，這是最簡單的 Index TTS 2 測試。請問你今天過得怎麼樣？"

# 輸出檔名（留空會自動用時間戳）
OUTPUT_WAV = ""

# 採樣參數（預設 temperature=0.8；調低可減少雜訊感／空氣聲）
TEMPERATURE = 0.8
TOP_P = 0.7
TOP_K = 30

# 情緒模式：只能擇一
# 1) 不用情緒：全部留空/False
# 2) 情緒參考音：填 EMO_AUDIO
# 3) 情緒向量：填 EMO_VECTOR（8 維）
# 4) 文字情緒：USE_EMO_TEXT=True（可搭配 EMO_TEXT）
EMO_AUDIO = ""  # 例如 "emo_sad.wav"
EMO_VECTOR: list[float] | None = None  # 例如 [0, 0, 0.8, 0, 0, 0, 0, 0]
USE_EMO_TEXT = False
EMO_TEXT = ""
EMO_ALPHA = 0.6
USE_RANDOM = False
# ============================================

_BOOTSTRAP_ENV_FLAG = "INDEXTTS_API_BOOTSTRAPPED"


def _resolve_audio(path_str: str, repo_path: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_file():
        return p.resolve()
    candidate = repo_path / "examples" / p.name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"找不到音檔: {path_str}")


def _load_model(repo_path: Path, model_dir: Path, cfg_path: Path) -> Any:
    if not repo_path.is_dir():
        raise FileNotFoundError(f"找不到 IndexTTS repo: {repo_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"找不到模型目錄: {model_dir}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到設定檔: {cfg_path}")

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        from indextts.infer_v2_5 import IndexTTS2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        # 你可能在 indextts_api 的 venv 裡執行，但 IndexTTS2 的依賴在 index-tts 的 venv 才有。
        # 自動切換到 index-tts/.venv/python 再跑一次，確保「直接跑腳本就能出 wav」。
        if os.getenv(_BOOTSTRAP_ENV_FLAG):
            raise
        venv_python = repo_path / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            raise RuntimeError(
                f"IndexTTS2 依賴未安裝且找不到 venv python: {venv_python}\n"
                "請先在 repo root 執行：\n"
                "  uv sync --all-extras\n"
                "然後再直接執行本腳本。"
            )

        env = dict(os.environ)
        env[_BOOTSTRAP_ENV_FLAG] = "1"
        proc = subprocess.run([str(venv_python), str(Path(__file__).resolve())], env=env)
        raise SystemExit(proc.returncode)

    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        use_bf16=USE_BF16,
        use_cuda_kernel=USE_CUDA_KERNEL,
        use_deepspeed=USE_DEEPSPEED,
    )


def _build_kwargs(spk_audio: Path, output_path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "spk_audio_prompt": str(spk_audio),
        "text": TEXT,
        "output_path": str(output_path),
        "lang": LANG,
        "duration_factor": DURATION_FACTOR,
        "verbose": VERBOSE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }

    if EMO_AUDIO:
        kwargs["emo_audio_prompt"] = str(_resolve_audio(EMO_AUDIO, Path(INDEXTTS_REPO_PATH)))
        kwargs["emo_alpha"] = EMO_ALPHA
    elif EMO_VECTOR is not None:
        if len(EMO_VECTOR) != 8:
            raise ValueError("EMO_VECTOR 必須是 8 維")
        kwargs["emo_vector"] = EMO_VECTOR
        kwargs["use_random"] = USE_RANDOM
    elif USE_EMO_TEXT:
        kwargs["use_emo_text"] = True
        kwargs["emo_alpha"] = EMO_ALPHA
        kwargs["use_random"] = USE_RANDOM
        if EMO_TEXT.strip():
            kwargs["emo_text"] = EMO_TEXT.strip()

    return kwargs


def main() -> int:
    repo_path = Path(INDEXTTS_REPO_PATH).expanduser().resolve()
    model_dir = Path(MODEL_DIR).expanduser().resolve()
    cfg_path = Path(CFG_PATH).expanduser().resolve()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(OUTPUT_WAV).expanduser().resolve()
        if OUTPUT_WAV.strip()
        else output_dir / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )

    spk_audio = _resolve_audio(SPK_AUDIO, repo_path)
    model = _load_model(repo_path, model_dir, cfg_path)
    kwargs = _build_kwargs(spk_audio, output_path)

    print(f"開始合成 -> {output_path}")
    model.infer(**kwargs)

    if not output_path.is_file():
        raise RuntimeError(f"合成失敗，找不到輸出檔: {output_path}")
    print(f"完成: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
