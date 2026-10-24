# syntax=docker/dockerfile:1

# ---- stage 1: build the Tailwind CSS ----
FROM node:20-alpine AS css
WORKDIR /ui
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npx tailwindcss -c tailwind.config.js -i src/input.css -o /ui/out/tailwind.css --minify

# ---- stage 2: the app ----
FROM python:3.11-slim
ARG WHISPER_MODEL=base
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf TOKENIZERS_PARALLELISM=false ANONYMIZED_TELEMETRY=False \
    WHISPER_MODEL=${WHISPER_MODEL} EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    PORT=7860 DB_DIR=/var/lib/copilot/chroma_db SQLITE_PATH=/var/lib/copilot/records.db \
    KNOWLEDGE_DIR=/app/knowledge STATIC_DIR=/app/static FRONTEND_DIR=/app/frontend
WORKDIR /app

COPY requirements.txt .
# CPU-only PyTorch first: the default wheel drags in several GB of CUDA libraries the app never uses.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Download the speech and embedding models now, so the first request is not slow and nothing is fetched at runtime.
RUN python -c "import os; from sentence_transformers import SentenceTransformer as S; S(os.environ['EMBEDDING_MODEL'])" \
 && python -c "import os; from faster_whisper import WhisperModel as W; W(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8')"

COPY app/ app/
COPY frontend/index.html frontend/index.html
COPY knowledge/ knowledge/
COPY --from=css /ui/out/tailwind.css static/tailwind.css

RUN useradd -m -u 1000 app && mkdir -p /var/lib/copilot && chown -R app:app /var/lib/copilot /opt/hf /app
USER app
EXPOSE 7860
# Start-up loads the models and indexes the knowledge folders, so allow time before the first check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=5 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '7860'), timeout=4)"
# One worker on purpose: the local Chroma store is not multi-process safe. Port 7860 is what Hugging Face Spaces expects.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
