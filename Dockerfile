# syntax=docker/dockerfile:1

# ---- stage 1: build the Tailwind CSS ----
FROM node:20-alpine AS css
WORKDIR /ui
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npx tailwindcss -c tailwind.config.js -i src/input.css -o /ui/out/tailwind.css --minify

# ---- stage 2: the app (Gemini API only, no local models) ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    PORT=8001 DB_DIR=/var/lib/copilot/chroma_db SQLITE_PATH=/var/lib/copilot/records.db \
    KNOWLEDGE_DIR=/app/knowledge STATIC_DIR=/app/static FRONTEND_DIR=/app/frontend
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY frontend/index.html frontend/index.html
COPY knowledge/ knowledge/
COPY --from=css /ui/out/tailwind.css static/tailwind.css

RUN useradd -m -u 1000 app && mkdir -p /var/lib/copilot && chown -R app:app /var/lib/copilot /app
USER app
EXPOSE 8001
# No local models to load now, so the app starts quickly. Embedding the knowledge folders on the first start is the slow part.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '8001'), timeout=4)"
# One worker on purpose: the local Chroma store is not multi-process safe.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001} --proxy-headers --forwarded-allow-ips '*'"]