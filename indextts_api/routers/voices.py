from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from indextts_api.audio import AudioDecodeError
from indextts_api.dependencies import get_voice_store
from indextts_api.schemas import VoiceCreateRequest, VoiceListResponse, VoiceMetadata
from indextts_api.voice_store import VoiceNotFoundError, VoiceStore

router = APIRouter(prefix="/v1/voices", tags=["voices"])


@router.post("", response_model=VoiceMetadata, status_code=status.HTTP_201_CREATED)
def create_voice(
    payload: VoiceCreateRequest,
    store: VoiceStore = Depends(get_voice_store),
) -> VoiceMetadata:
    try:
        return store.create(payload.audio_base64, payload.reference_text)
    except AudioDecodeError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@router.get("", response_model=VoiceListResponse)
def list_voices(store: VoiceStore = Depends(get_voice_store)) -> VoiceListResponse:
    return VoiceListResponse(voices=store.list())


@router.get("/{voice_id}", response_model=VoiceMetadata)
def get_voice(
    voice_id: str,
    store: VoiceStore = Depends(get_voice_store),
) -> VoiceMetadata:
    try:
        return store.get(voice_id).metadata
    except VoiceNotFoundError as err:
        raise HTTPException(status_code=404, detail=f"voice_id 不存在: {voice_id}") from err


@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_voice(
    voice_id: str,
    store: VoiceStore = Depends(get_voice_store),
) -> None:
    try:
        store.delete(voice_id)
    except VoiceNotFoundError as err:
        raise HTTPException(status_code=404, detail=f"voice_id 不存在: {voice_id}") from err
