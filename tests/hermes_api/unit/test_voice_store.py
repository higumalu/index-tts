from __future__ import annotations

import base64

import pytest

from indextts_api.voice_store import VoiceNotFoundError, VoiceStore


@pytest.mark.unit
def test_voice_store_crud(tmp_path, wav_base64: str) -> None:
    store = VoiceStore(tmp_path / "voices")
    meta = store.create(wav_base64, reference_text="optional")
    assert len(meta.voice_id) == 32

    record = store.get(meta.voice_id)
    assert record.metadata.voice_id == meta.voice_id
    assert record.audio_path.is_file()

    listed = store.list()
    assert len(listed) == 1
    assert store.count() == 1

    store.delete(meta.voice_id)
    with pytest.raises(VoiceNotFoundError):
        store.get(meta.voice_id)


@pytest.mark.unit
def test_voice_store_rejects_invalid_id(tmp_path) -> None:
    store = VoiceStore(tmp_path / "voices")
    with pytest.raises(VoiceNotFoundError):
        store.get("not-a-valid-id")
