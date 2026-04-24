"""
FastAPI 데모 앱 진입점.

실행:
  cd <프로젝트 루트>
  .venv/bin/uvicorn src.api.main:app --reload --port 8000
"""

import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# 프로젝트 루트에서 실행 가정 — src/ 를 경로에 추가
_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src"
for p in (str(_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from api import cache as cache_module
from api.routes import router
from api import session_store


_BM25_DISABLED = os.getenv("BM25_DISABLED", "false").lower() == "true"


def _warm_bm25() -> None:
    """BM25 인덱스를 백그라운드에서 미리 빌드 (파드 재시작 후 첫 요청 지연 방지)."""
    if _BM25_DISABLED:
        logger.info("[BM25] BM25_DISABLED=true — 벡터 전용 모드로 실행")
        return
    try:
        from qdrant_rag import build_bm25_from_qdrant
        build_bm25_from_qdrant()
        logger.info("[BM25] 프리워밍 완료")
    except Exception as e:
        logger.warning("[BM25] 프리워밍 실패: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_store.init_db()
    cache_module.load_cache()
    threading.Thread(target=_warm_bm25, daemon=True, name="bm25-warmup").start()
    yield


app = FastAPI(
    title="Korean Law RAG Demo",
    description="한국 법령 질의응답 RAG 시스템 — CRAG vs Hybrid Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.include_router(router)
