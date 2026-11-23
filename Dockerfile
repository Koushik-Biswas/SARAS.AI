# syntax=docker/dockerfile:1
# SARAS AI — Smart Assessment and Recognition Automation System
# One image: FastAPI API + in-process job worker + the built React SPA.
#
#   docker build -t saras-ai:1.0.0 .
#   docker run -p 8000:8000 -e SARAS_JWT_SECRET=<long random string> saras-ai:1.0.0
#
# Extended profile (PostgreSQL, resource limits): see docker-compose.yml.
# Optional ML extras (TrOCR, SBERT, NLI; needs >= 8 GB RAM):
#   docker build --build-arg INSTALL_ML=1 -t saras-ai:1.0.0-ml .

# ------------------------------------------------------------------ 1. frontend build
FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------ 2. runtime
FROM python:3.12-slim AS runtime

ARG INSTALL_ML=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SARAS_ENV=prod \
    SARAS_DATABASE_URL=sqlite:////data/saras.db \
    SARAS_STORAGE_DIR=/data/storage \
    SARAS_FRONTEND_DIST=/app/frontend/dist \
    HF_HOME=/data/hf-cache \
    PORT=8000 \
    FORWARDED_ALLOW_IPS=127.0.0.1
# PORT: hosting platforms (Render, Railway, Azure App Service, ...) set their own.
# FORWARDED_ALLOW_IPS: proxies whose X-Forwarded-For / -Proto are trusted (real client IP
# for rate limiting, https for HSTS). Behind a platform proxy or Caddy set it to that
# proxy's address, or "*" when the container is reachable ONLY through that proxy.

WORKDIR /app/backend

# Dependencies first so code changes do not invalidate the pip layer.
COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY backend/app ./app
COPY --from=frontend /build/frontend/dist /app/frontend/dist

# Non-root user; /data holds the SQLite file (standalone runs), uploads, page images and
# the Hugging Face model cache. Mount a volume there to keep data across containers.
RUN groupadd --system --gid 10001 saras \
    && useradd --system --uid 10001 --gid saras --home-dir /app --no-create-home saras \
    && mkdir -p /data/storage /data/hf-cache \
    && chown -R saras:saras /data
USER saras
VOLUME ["/data"]

EXPOSE 8000

# Liveness only (no database); /ready also checks the database and the worker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4).status == 200 else 1)"]

# One uvicorn process: the job worker runs as a thread inside it (see docker-compose.yml
# for how to add worker capacity).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --no-server-header"]
