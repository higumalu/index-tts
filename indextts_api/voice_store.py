from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from indextts_api.audio import decode_base64_wav
from indextts_api.schemas import VoiceMetadata

_VOICE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_REFERENCE_FILENAME = "reference.wav"
_METADATA_FILENAME = "metadata.json"


class VoiceNotFoundError(KeyError):
    """Raised when a voice_id does not exist in the store."""


@dataclass
class VoiceRecord:
    metadata: VoiceMetadata
    audio_path: Path


class VoiceStore:
    def __init__(self, voices_dir: Path) -> None:
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)

    @property
    def voices_dir(self) -> Path:
        return self._voices_dir

    def _voice_dir(self, voice_id: str) -> Path:
        if not _VOICE_ID_RE.match(voice_id):
            raise VoiceNotFoundError(voice_id)
        return self._voices_dir / voice_id

    def create(self, audio_base64: str, reference_text: str = "") -> VoiceMetadata:
        raw, samples, sample_rate = decode_base64_wav(audio_base64)
        duration_sec = float(len(samples) / sample_rate) if sample_rate else 0.0

        voice_id = uuid.uuid4().hex
        target_dir = self._voices_dir / voice_id
        target_dir.mkdir(parents=True, exist_ok=False)

        audio_path = target_dir / _REFERENCE_FILENAME
        audio_path.write_bytes(raw)

        metadata = VoiceMetadata(
            voice_id=voice_id,
            reference_text=reference_text,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            created_at=time.time(),
        )
        (target_dir / _METADATA_FILENAME).write_text(
            json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata

    def get(self, voice_id: str) -> VoiceRecord:
        directory = self._voice_dir(voice_id)
        meta_path = directory / _METADATA_FILENAME
        audio_path = directory / _REFERENCE_FILENAME
        if not meta_path.exists() or not audio_path.exists():
            raise VoiceNotFoundError(voice_id)
        metadata = VoiceMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
        return VoiceRecord(metadata=metadata, audio_path=audio_path)

    def list(self) -> list[VoiceMetadata]:
        results: list[VoiceMetadata] = []
        for entry in sorted(self._voices_dir.iterdir()):
            if not entry.is_dir() or not _VOICE_ID_RE.match(entry.name):
                continue
            meta_path = entry / _METADATA_FILENAME
            if not meta_path.exists():
                continue
            try:
                results.append(VoiceMetadata(**json.loads(meta_path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        results.sort(key=lambda m: m.created_at)
        return results

    def delete(self, voice_id: str) -> None:
        directory = self._voice_dir(voice_id)
        if not directory.exists():
            raise VoiceNotFoundError(voice_id)
        shutil.rmtree(directory)

    def count(self) -> int:
        return sum(
            1
            for entry in self._voices_dir.iterdir()
            if entry.is_dir()
            and _VOICE_ID_RE.match(entry.name)
            and (entry / _METADATA_FILENAME).exists()
        )
