# Voicebox SaaS — inference worker image.
#
# Build:
#   docker build --platform linux/amd64 -t voicebox-worker:latest .
#
# TIER arg controls which engine modules are eager-loaded at cold start.

ARG TIER=fast

FROM python:3.12-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_CACHE=/models \
    VOICEBOX_TIER=${TIER}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Worker code ------------------------------------------------------------
COPY handler.py io_layer.py ./
COPY engines/ ./engines/

# ---- Model pre-bake (disabled by default — models downloaded at runtime) ----
ARG PREBAKE_MODELS=false
RUN if [ "$PREBAKE_MODELS" = "true" ]; then \
        python -c "from engines import EngineRegistry; r = EngineRegistry(); r.tts('${TIER}'); r.stt('${TIER}');" ; \
    fi

# ---- Non-root user ----------------------------------------------------------
RUN useradd --create-home --shell /bin/bash worker && \
    mkdir -p /models && chown -R worker:worker /app /models
USER worker

CMD ["python", "-u", "handler.py"]
