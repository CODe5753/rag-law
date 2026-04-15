# Stage 1: 의존성 설치
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir \
    -r requirements-docker.txt


# Stage 2: 런타임
FROM python:3.11-slim

WORKDIR /app

# 의존성 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 코드
COPY src/ ./src/

# 데이터 (빌드 전 precompute.py + chunks.json + qdrant_data 준비 필요)
COPY data/processed/chunks.json ./data/processed/chunks.json
COPY data/processed/bm25_cache.pkl ./data/processed/bm25_cache.pkl
COPY data/demo_cache/ ./data/demo_cache/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=exaone3.5:7.8b \
    EMBEDDING_MODEL=BAAI/bge-m3

EXPOSE 8000

# HuggingFace 모델 캐시 경로 (PersistentVolume 마운트 권장)
VOLUME /root/.cache/huggingface

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
