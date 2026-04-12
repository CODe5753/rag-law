"""
FastAPI 데모 앱 진입점.

실행:
  cd <프로젝트 루트>
  .venv/bin/uvicorn src.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 캐시 로드
    cache_module.load_cache()
    yield


app = FastAPI(
    title="Korean Law RAG Demo",
    description="한국 법령 질의응답 RAG 시스템 — CRAG vs Hybrid Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.include_router(router)
