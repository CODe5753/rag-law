from __future__ import annotations
import os
import requests as _requests

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.schemas import QueryRequest, QueryResponse
from api import cache as cache_module
from api.pipeline_runner import run_pipeline

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")

router = APIRouter()
templates = Jinja2Templates(directory="src/templates")


# ── 페이지 ────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    manifest = cache_module.get_manifest()
    return templates.TemplateResponse(request, "index.html", {
        "categories": manifest.get("categories", []),
        "ollama_model": OLLAMA_MODEL,
    })


# ── HTMX 쿼리 엔드포인트 (HTML partial 반환) ─────────────────

@router.post("/query", response_class=HTMLResponse)
async def query_htmx(
    request: Request,
    question: str = Form(...),
    pipeline: str = Form("crag"),
):
    if not question.strip():
        return HTMLResponse("<p class='text-red-500'>질문을 입력해주세요.</p>")

    try:
        result = run_pipeline(question.strip(), pipeline)
    except Exception as e:
        return templates.TemplateResponse(request, "partials/error.html", {
            "message": str(e),
        }, status_code=500)

    return templates.TemplateResponse(request, "partials/result.html", {
        "result": result,
    })


@router.get("/samples/{qid}", response_class=HTMLResponse)
async def sample_htmx(request: Request, qid: str, pipeline: str = "crag"):
    result = cache_module.get_cached(qid, pipeline)
    if result is None:
        return HTMLResponse(
            f"<p class='text-yellow-600'>캐시 없음: {qid} ({pipeline}). "
            "precompute.py 를 먼저 실행하세요.</p>"
        )
    return templates.TemplateResponse(request, "partials/result.html", {
        "result": result,
    })


# ── 순수 JSON API ─────────────────────────────────────────────

@router.post("/api/query", response_model=QueryResponse)
async def query_api(body: QueryRequest):
    return run_pipeline(body.question, body.pipeline)


@router.get("/api/samples")
async def samples_api():
    return cache_module.get_manifest()


@router.get("/api/health")
async def health():
    ollama_ok = False
    try:
        r = _requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "ollama_connected": ollama_ok,
        "cached_samples": cache_module.cache_count(),
        "model": OLLAMA_MODEL,
    }
