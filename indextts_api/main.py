from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from indextts_api import __version__
from indextts_api.config import settings
from indextts_api.routers import health, tts, voices
from indextts_api.tts_engine import TTSEngine
from indextts_api.voice_store import VoiceStore


def _configure_logging() -> None:
    """uvicorn 只設定自己的 logger，root 沒有 handler，我們的 log 會被丟掉。"""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    app.state.voice_store = VoiceStore(settings.voices_dir)
    engine = TTSEngine()
    app.state.tts_engine = engine
    engine.start_idle_reaper()
    try:
        yield
    finally:
        engine.stop_idle_reaper()


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
