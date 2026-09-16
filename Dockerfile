# syntax=docker/dockerfile:1
# One container = API + HUD on one origin. Built by Cloud Build (`make deploy`)
# or locally (`make docker`). Model version and baselines come from data/artifacts;
# `scripts/deploy_ignore.py` keeps only the registered model in the build context.

# ---- 1. HUD: static export -------------------------------------------------
FROM node:20-alpine AS hud
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ .
# same origin as the API -> empty base URL
ENV NEXT_PUBLIC_API_BASE_URL=""
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- 2. API ----------------------------------------------------------------
FROM python:3.11-slim AS api
# libgomp: LightGBM's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY backend/ /app/backend/
COPY data/artifacts/baselines/ /app/data/artifacts/baselines/
COPY data/artifacts/models/ /app/data/artifacts/models/
COPY --from=hud /fe/out /app/frontend/out

ENV PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data \
    MODEL_REGISTRY_DIR=/app/data/artifacts \
    STATIC_DIR=/app/frontend/out \
    API_CORS_ORIGINS="*" \
    PORT=8080
EXPOSE 8080
# Cloud Run injects $PORT; one worker is plenty (15 ms per simulate, engine is in-process)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
