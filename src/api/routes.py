from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import requests as _requests

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Request, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from api.schemas import QueryRequest, QueryResponse, RetrievedDoc, PipelineTrace, GradingSummary
from api import cache as cache_module
from api.pipeline_runner import run_pipeline, rewrite_query, intake_classify, run_qdrant_with_history
from api import session_store

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")

router = APIRouter()
templates = Jinja2Templates(directory="src/templates")

# 동시 쿼리 최대 2개 — 이벤트 루프 보호 + OOM 방지
_query_semaphore = asyncio.Semaphore(2)


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
        async with _query_semaphore:
            result = await run_in_threadpool(run_pipeline, question.strip(), pipeline)
    except Exception as e:
        logger.exception("query pipeline failed: %s", e)
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


# ── SSE 스트리밍 엔드포인트 ────────────────────────────────────

def _build_crag_response(question: str, final_state: dict, latency: float) -> QueryResponse:
    """CRAG .stream() 최종 state에서 QueryResponse 재구성 (pipeline_runner.run_crag와 동일한 로직)."""
    decision = "fallback" if final_state.get("fallback_used") else "generate"

    grading_log = final_state.get("grading_log", [])
    total = len(grading_log)
    relevant_count = sum(1 for g in grading_log if g["is_relevant"])
    methods = list({g["graded_by"] for g in grading_log}) if grading_log else ["reranker"]
    graded_by = methods[0] if len(methods) == 1 else "mixed"

    grading_summary = GradingSummary(
        total_docs=total,
        relevant_docs=relevant_count,
        threshold=0.5,
        graded_by=graded_by,
    )

    all_docs = final_state.get("documents", [])
    log_by_id = {g["chunk_id"]: g for g in grading_log}

    retrieved_docs = []
    for doc in all_docs:
        log = log_by_id.get(doc.get("chunk_id", ""), {})
        retrieved_docs.append(RetrievedDoc(
            law_name=doc.get("law_name", ""),
            article_num=doc.get("article_num", ""),
            text=doc.get("text", ""),
            reranker_score=doc.get("reranker_score"),
            is_relevant=log.get("is_relevant"),
        ))

    answer = final_state.get("generation", "")
    if not answer.strip():
        raise RuntimeError("Ollama 서버 응답 생성 실패")

    return QueryResponse(
        question=question,
        answer=answer,
        pipeline="crag",
        latency_sec=latency,
        cached=False,
        retrieved_docs=retrieved_docs,
        pipeline_trace=PipelineTrace(
            nodes_executed=["retrieve", "grade_documents", "generate" if decision == "generate" else "fallback"],
            decision=decision,
            grading_summary=grading_summary,
        ),
    )


@router.get("/query/stream")
async def query_stream(request: Request, question: str, pipeline: str = "crag"):
    q = question.strip()
    if not q:
        return HTMLResponse("<p class='text-red-500'>질문을 입력해주세요.</p>", status_code=400)

    async def sse_pack(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_gen():
        async with _query_semaphore:
            try:
                if pipeline == "crag":
                    loop = asyncio.get_event_loop()
                    queue: asyncio.Queue = asyncio.Queue()
                    final_state_holder = {"state": None}

                    def run_in_thread():
                        try:
                            from crag_rag import build_crag_graph
                            app = build_crag_graph()
                            initial_state = {
                                "question": q,
                                "documents": [],
                                "relevant_docs": [],
                                "generation": "",
                                "fallback_used": False,
                                "grading_log": [],
                            }
                            merged: dict = dict(initial_state)
                            for event in app.stream(initial_state):
                                for node_name, partial in event.items():
                                    if isinstance(partial, dict):
                                        merged.update(partial)
                                    asyncio.run_coroutine_threadsafe(
                                        queue.put({"type": "node", "node": node_name}), loop
                                    ).result()
                            final_state_holder["state"] = merged
                        except Exception as e:
                            asyncio.run_coroutine_threadsafe(
                                queue.put({"type": "error", "message": str(e)}), loop
                            ).result()
                        finally:
                            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

                    t0 = time.time()
                    loop.run_in_executor(None, run_in_thread)

                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        yield await sse_pack(item)
                        if item.get("type") == "error":
                            return

                    latency = round(time.time() - t0, 1)
                    final_state = final_state_holder["state"]
                    if final_state is None:
                        yield await sse_pack({"type": "error", "message": "스트리밍 결과 없음"})
                        return

                    result = _build_crag_response(q, final_state, latency)
                    html = templates.TemplateResponse(
                        request, "partials/result.html", {"result": result}
                    ).body.decode()
                    yield await sse_pack({"type": "done", "html": html})

                else:
                    # Qdrant: 단일 블로킹 호출 — 시작/완료 이벤트만 전송
                    yield await sse_pack({"type": "node", "node": "retrieve"})
                    result = await run_in_threadpool(run_pipeline, q, pipeline)
                    # retrieve + generate가 내부적으로 완료된 상태
                    html = templates.TemplateResponse(
                        request, "partials/result.html", {"result": result}
                    ).body.decode()
                    yield await sse_pack({"type": "done", "html": html})

            except Exception as e:
                logger.exception("stream pipeline failed: %s", e)
                yield await sse_pack({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 채팅 엔드포인트 ───────────────────────────────────────────

@router.post("/chat/new")
async def chat_new():
    sid = await run_in_threadpool(session_store.create_session)
    return {"session_id": sid}


@router.get("/chat/stream")
async def chat_stream(request: Request, session_id: str, question: str, pipeline: str = "qdrant"):
    q = question.strip()
    if not q:
        return HTMLResponse("<p class='text-red-500'>질문을 입력해주세요.</p>", status_code=400)

    async def sse_pack(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_gen():
        async with _query_semaphore:
            try:
                # 1. 사용자 메시지 저장
                await run_in_threadpool(session_store.add_message, session_id, "user", q)

                # 2. 이전 히스토리 로드 (방금 저장한 메시지 제외)
                all_messages = await run_in_threadpool(session_store.get_messages, session_id)
                prior_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in all_messages[:-1]  # 방금 저장한 user 메시지 제외
                ]

                # 3. Intake 분류
                classify = await run_in_threadpool(intake_classify, q, prior_history)

                if classify["action"] == "ask":
                    reply = classify["reply"]
                    # AI 메시지 저장 (추가 질문)
                    await run_in_threadpool(session_store.add_message, session_id, "assistant", reply)
                    yield await sse_pack({"type": "ask", "reply": reply})
                    return

                # 4. Search: RAG 실행
                yield await sse_pack({"type": "node", "node": "retrieve"})
                result = await run_in_threadpool(run_qdrant_with_history, q, prior_history)

                # 5. AI 답변 저장
                docs_for_store = [
                    {
                        "law_name": d.law_name,
                        "article_num": d.article_num,
                        "source": d.source,
                    }
                    for d in (result.retrieved_docs or [])
                ]
                await run_in_threadpool(
                    session_store.add_message,
                    session_id,
                    "assistant",
                    result.answer,
                    docs_for_store,
                )

                # 6. 결과 전송
                docs_data = [
                    {
                        "law_name": d.law_name,
                        "article_num": d.article_num,
                        "source": d.source,
                        "reranker_score": d.reranker_score,
                    }
                    for d in (result.retrieved_docs or [])[:5]
                ]
                yield await sse_pack({
                    "type": "done",
                    "answer": result.answer,
                    "docs": docs_data,
                    "latency": result.latency_sec,
                })

            except Exception as e:
                logger.exception("chat stream pipeline failed: %s", e)
                yield await sse_pack({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_session_page(request: Request, session_id: str):
    manifest = cache_module.get_manifest()
    return templates.TemplateResponse(request, "index.html", {
        "categories": manifest.get("categories", []),
        "ollama_model": OLLAMA_MODEL,
        "initial_session_id": session_id,
    })


@router.get("/api/chat/{session_id}/messages")
async def get_chat_messages(session_id: str):
    messages = await run_in_threadpool(session_store.get_messages, session_id)
    return messages


# ── 비교 엔드포인트 ────────────────────────────────────────────

@router.get("/compare/{qid}", response_class=HTMLResponse)
async def compare_htmx(request: Request, qid: str):
    crag_result = cache_module.get_cached(qid, "crag")
    qdrant_result = cache_module.get_cached(qid, "qdrant")
    if crag_result is None or qdrant_result is None:
        return HTMLResponse(
            "<p class='text-yellow-600'>비교를 위해 두 파이프라인 캐시가 모두 필요합니다.</p>"
        )
    return templates.TemplateResponse(request, "partials/compare.html", {
        "crag": crag_result,
        "qdrant": qdrant_result,
    })


# ── 순수 JSON API ─────────────────────────────────────────────

@router.post("/api/query", response_model=QueryResponse)
async def query_api(body: QueryRequest):
    async with _query_semaphore:
        return await run_in_threadpool(run_pipeline, body.question, body.pipeline)


@router.get("/api/samples")
async def samples_api():
    return cache_module.get_manifest()


@router.get("/api/health")
async def health():
    """Liveness probe — 프로세스 생존 여부만 확인. 외부 의존성 체크 없음."""
    return {"status": "ok", "model": OLLAMA_MODEL}


@router.get("/api/ready")
async def ready():
    """Readiness probe — Ollama 연결 확인. 실패 시 트래픽 차단 (pod 재시작 X)."""
    ollama_ok = False
    try:
        r = await run_in_threadpool(_requests.get, f"{OLLAMA_HOST}/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    status_code = 200 if ollama_ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if ollama_ok else "unavailable",
            "ollama_connected": ollama_ok,
            "cached_samples": cache_module.cache_count(),
            "model": OLLAMA_MODEL,
        },
    )
