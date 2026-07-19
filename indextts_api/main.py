from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from indextts_api import __version__
from indextts_api.config import settings
from indextts_api.routers import health, tts, voices
from indextts_api.tts_engine import TTSEngine
from indextts_api.voice_store import VoiceStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.voice_store = VoiceStore(settings.voices_dir)
    app.state.tts_engine = TTSEngine()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="IndexTTS Voice API",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(voices.router)
    app.include_router(tts.router)
    return app


app = create_app()
