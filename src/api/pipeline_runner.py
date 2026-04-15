"""
기존 RAG 파이프라인을 API 레이어로 래핑.
crag_rag, qdrant_rag 원본 코드는 수정하지 않음.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

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
    from qdrant_rag import load_chunks, load_embed_model, build_bm25, hybrid_retrieve, generate

    # 요청마다 로드 후 함수 종료 시 GC 해제 (pickle 0.0s, 메모리 반환)
    chunks = load_chunks()
    model = load_embed_model()
    bm25, chunk_ids = build_bm25(chunks)

    t0 = time.time()
    try:
        docs = hybrid_retrieve(
            question,
            model,
            chunks,
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


# ── 디스패처 ─────────────────────────────────────────────────

def run_pipeline(question: str, pipeline: str) -> QueryResponse:
    if pipeline == "crag":
        return run_crag(question)
    elif pipeline == "qdrant":
        return run_qdrant(question)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
