FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.11 \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md .python-version ./
COPY LICENSE LICENSE_ZH.txt DISCLAIMER ./
RUN uv python install 3.11 \
    && uv sync --frozen --no-dev --extra hermes_api --no-install-project

COPY indextts ./indextts
COPY indextts_api ./indextts_api
RUN uv sync --frozen --no-dev --extra hermes_api

ENV INDEXTTS_REPO_PATH=/app \
    INDEXTTS_MODEL_DIR=/app/checkpoints \
    INDEXTTS_CFG_PATH=/app/checkpoints/config.yaml \
    INDEXTTS_VOICES_DIR=/app/voices \
    INDEXTTS_USE_FP16=true \
    INDEXTTS_IDLE_UNLOAD_SEC=600 \
    INDEXTTS_IDLE_CHECK_INTERVAL_SEC=30

RUN mkdir -p /app/voices /app/checkpoints

EXPOSE 8001

CMD ["uv", "run", "--no-dev", "uvicorn", "indextts_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
