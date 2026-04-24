"""
기존 RAG 파이프라인을 API 레이어로 래핑.
crag_rag, qdrant_rag 원본 코드는 수정하지 않음.
"""

from __future__ import annotations
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")

# src/ 디렉토리를 경로에 추가 (crag_rag, qdrant_rag import용)
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.schemas import QueryResponse, RetrievedDoc, PipelineTrace, GradingSummary

# ── Qdrant 파이프라인: 요청별 로드 (싱글턴 제거) ─────────────
# BM25 pickle(2MB)이 0.0s 로드라 싱글턴으로 유지할 필요 없음.
# 요청 후 Python GC가 메모리 해제 → 704Mi → ~71Mi (idle)


# ── CRAG 실행 ────────────────────────────────────────────────

def run_crag(question: str) -> QueryResponse:
    from crag_rag import run_query

    result = run_query(question)

    decision = "fallback" if result.get("fallback_used") else "generate"
    nodes = ["retrieve", "grade_documents", decision]

    grading_log = result.get("grading_log", [])
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

    # 전체 문서 목록 (grade 결과 포함)
    all_docs = result.get("documents", [])
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

    answer = result.get("generation", "")
    if not answer.strip():
        raise RuntimeError("Ollama 서버에 연결할 수 없거나 응답 생성에 실패했습니다. ollama serve 상태를 확인하세요.")

    return QueryResponse(
        question=question,
        answer=answer,
        pipeline="crag",
        latency_sec=result.get("latency_sec", 0.0),
        cached=False,
        retrieved_docs=retrieved_docs,
        pipeline_trace=PipelineTrace(
            nodes_executed=nodes,
            decision=decision,
            grading_summary=grading_summary,
        ),
    )


# ── Qdrant 실행 ──────────────────────────────────────────────

def run_qdrant(question: str) -> QueryResponse:
    from qdrant_rag import load_embed_model, build_bm25_from_qdrant, hybrid_retrieve, generate, _bm25_lock, _bm25_data, _BM25_DISABLED

    model = load_embed_model()

    # BM25_DISABLED=true 이면 벡터 전용 모드
    if _BM25_DISABLED:
        bm25, chunk_ids = None, None
    elif _bm25_data is not None:
        bm25, chunk_ids = _bm25_data
    elif _bm25_lock.acquire(blocking=False):
        _bm25_lock.release()
        bm25, chunk_ids = build_bm25_from_qdrant()
    else:
        bm25, chunk_ids = None, None

    t0 = time.time()
    try:
        docs = hybrid_retrieve(
            question,
            model,
            bm25,
            chunk_ids,
        )
        gen = generate(question, docs)
    finally:
        # 요청 완료 후 Kiwi 싱글턴 해제 → GC가 ~150MB 회수
        import qdrant_rag as _qr
        _qr._kiwi = None
    latency = round(time.time() - t0, 1)

    retrieved_docs = [
        RetrievedDoc(
            law_name=d.get("law_name", ""),
            article_num=d.get("article_num", ""),
            text=d.get("text", ""),
            source=d.get("source"),
            reranker_score=d.get("reranker_score"),
            is_relevant=True,   # qdrant는 grade 없음 — 검색된 모든 문서가 relevant
        )
        for d in docs
    ]

    answer = gen.get("answer", "")
    if not answer.strip():
        raise RuntimeError("Ollama 서버에 연결할 수 없거나 응답 생성에 실패했습니다. ollama serve 상태를 확인하세요.")

    return QueryResponse(
        question=question,
        answer=answer,
        pipeline="qdrant",
        latency_sec=latency,
        cached=False,
        retrieved_docs=retrieved_docs,
        pipeline_trace=PipelineTrace(
            nodes_executed=["retrieve", "generate"],
            decision="generate",
            grading_summary=None,
        ),
    )


# ── 쿼리 재작성 ──────────────────────────────────────────────

def rewrite_query(question: str, history: list[dict]) -> str:
    """history가 없으면 question 그대로 반환. 있으면 Ollama로 독립 쿼리 생성."""
    if not history:
        return question

    turns = "\n".join(
        f"{'사용자' if h['role'] == 'user' else 'AI'}: {h['content']}"
        for h in history[-6:]  # last 3 turns = up to 6 messages
    )
    prompt = (
        "다음은 법률 정보 서비스의 대화 내역입니다.\n"
        f"{turns}\n\n"
        f"새 질문: {question}\n\n"
        "위 대화 맥락을 반영해 새 질문을 독립적으로 검색 가능한 한국어 문장으로 재작성하세요. "
        "재작성된 질문만 출력하세요."
    )

    try:
        import requests as _req
        resp = _req.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=10,
        )
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip()
        return rewritten if rewritten else question
    except Exception as e:
        logger.warning("rewrite_query failed, using original: %s", e)
        return question


# ── 디스패처 ─────────────────────────────────────────────────

def run_pipeline(question: str, pipeline: str) -> QueryResponse:
    if pipeline == "crag":
        return run_crag(question)
    elif pipeline == "qdrant":
        return run_qdrant(question)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
