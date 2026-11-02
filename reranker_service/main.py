"""
BGE Reranker v2-m3 macOS 네이티브 서비스.

Ollama와 동일한 패턴으로 CrossEncoder를 macOS에서 서빙.
Gateway만 인증된 loopback 요청으로 이 서비스를 호출.

실행: python main.py
launchd로 자동 시작: ~/Library/LaunchAgents/com.raglaw.reranker.plist
"""

import os
import time
from pathlib import Path
from gateway_auth import load_gateway_token, authorized
from starlette.responses import JSONResponse
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
DEFAULT_GATEWAY_TOKEN = str(Path.home() / ".config/mac-dashboard/model-runtime/reranker.token")
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
PORT = int(os.getenv("RERANKER_PORT", "11435"))

_reranker = None
_gateway_token = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _reranker, _gateway_token
    _gateway_token = load_gateway_token(DEFAULT_GATEWAY_TOKEN)
    print(f"[reranker] 모델 로드: {RERANKER_MODEL}")
    t0 = time.time()
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(RERANKER_MODEL, local_files_only=True)
    print(f"[reranker] 로드 완료: {round(time.time()-t0, 1)}s")
    yield


app = FastAPI(title="BGE Reranker Service", lifespan=lifespan)


@app.middleware("http")
async def gateway_auth(request, call_next):
    if request.method == "POST" and request.url.path == "/rerank" and not authorized(request.headers, _gateway_token):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


@app.get("/health")
def health():
    return {"status": "ok", "model": RERANKER_MODEL, "loaded": _reranker is not None}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if _reranker is None:
        raise HTTPException(status_code=503, detail="Reranker not loaded")
    pairs = [[req.query, doc] for doc in req.documents]
    scores = _reranker.predict(pairs)
    return RerankResponse(scores=[round(float(s), 4) for s in scores])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
