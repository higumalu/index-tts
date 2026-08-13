#!/usr/bin/env python3
"""Download IndexTTS-2 model weights into ``./checkpoints``.

Uses the project's HuggingFace / ModelScope auto-switch
(``indextts.utils.model_download``).

Examples:
  uv run python scripts/download_checkpoints.py
  uv run python scripts/download_checkpoints.py --model-dir checkpoints
  uv run python scripts/download_checkpoints.py --version 2      # 舊版 IndexTTS-2
  uv run python scripts/download_checkpoints.py --skip-aux
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 每個版本的檔案組成不同：2.5 改用 tiktoken 詞表並多了 codec.pth，不再有 bpe.model。
REQUIRED_FILES_BY_VERSION = {
    "2": (
        "bpe.model",
        "gpt.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "config.yaml",
    ),
    "2.5": (
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "codec.pth",
        "gpt.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "config.yaml",
    ),
}

DEFAULT_VERSION = "2.5"
REPO_ID = "IndexTeam/IndexTTS-2.5"


def _missing(model_dir: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (model_dir / name).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IndexTTS-2 checkpoints")
    parser.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Target directory (default: checkpoints)",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Model version to fetch (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the model repo id (default: derived from --version)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when required files already exist",
    )
    parser.add_argument(
        "--skip-aux",
        action="store_true",
        help="Skip auxiliary models (w2v-bert / CAMPPlus / BigVGAN / MaskGCT)",
    )
    args = parser.parse_args()

    from indextts.utils.model_download import _VERSION_TO_REPO

    if args.repo_id is None:
        if args.version not in _VERSION_TO_REPO:
            supported = ", ".join(sorted(_VERSION_TO_REPO))
            print(
                f"Unsupported --version {args.version!r}. Supported: {supported}",
                file=sys.stderr,
            )
            return 1
        args.repo_id = _VERSION_TO_REPO[args.version]

    required = REQUIRED_FILES_BY_VERSION.get(args.version)
    if required is None:
        print(
            f"Unknown --version {args.version!r}. Known: "
            f"{', '.join(sorted(REQUIRED_FILES_BY_VERSION))}",
            file=sys.stderr,
        )
        return 1

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    missing = _missing(model_dir, required)
    if missing and not args.force:
        print(f"Missing required files in {model_dir}: {', '.join(missing)}")
        print(f"Downloading {args.repo_id} ...")
        from indextts.utils.model_download import snapshot_download

        try:
            snapshot_download(args.repo_id, local_dir=str(model_dir))
        except Exception as exc:
            print(f"Failed to download {args.repo_id}: {exc}", file=sys.stderr)
            return 1
    elif args.force:
        print(f"Force re-downloading {args.repo_id} into {model_dir} ...")
        from indextts.utils.model_download import snapshot_download

        try:
            snapshot_download(args.repo_id, local_dir=str(model_dir), force_download=True)
        except Exception as exc:
            print(f"Failed to download {args.repo_id}: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"Required files already present in {model_dir}")

    missing = _missing(model_dir, required)
    if missing:
        print(
            f"Download incomplete, still missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    from indextts.utils.model_download import ensure_config_available

    try:
        ensure_config_available(str(model_dir), version=args.version)
    except Exception as exc:
        print(f"Failed to ensure config.yaml: {exc}", file=sys.stderr)
        return 1

    if not args.skip_aux:
        from indextts.utils.model_download import ensure_models_available

        print("Ensuring auxiliary models under hf_cache/ ...")
        try:
            paths = ensure_models_available(str(model_dir))
        except Exception as exc:
            print(f"Failed to download auxiliary models: {exc}", file=sys.stderr)
            return 1
        for key, path in paths.items():
            print(f"  {key}: {path}")

    print("Checkpoints ready.")
    for name in required:
        size = (model_dir / name).stat().st_size
        print(f"  {name}: {size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
