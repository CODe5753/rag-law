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


# ── Intake 분류 ──────────────────────────────────────────────

def intake_classify(question: str, history: list[dict]) -> dict:
    """Ollama로 "정보 충분?" 판단. 충분하면 {"action":"search"}, 아니면 {"action":"ask","reply":"질문"}."""
    ai_turns = sum(1 for h in history if h["role"] == "assistant")

    # 2턴 이상 질문했으면 강제 검색
    if ai_turns >= 2:
        return {"action": "search"}

    history_text = ""
    for h in history:
        prefix = "사용자" if h["role"] == "user" else "AI"
        history_text += f"{prefix}: {h['content']}\n"

    # 첫 메시지: 항상 1번 추가 질문 (SEARCH 금지)
    first_message_instruction = ""
    if ai_turns == 0:
        first_message_instruction = """
⚠ 이것은 첫 번째 메시지입니다. 반드시 추가 정보 수집 질문을 1개 해야 합니다.
"SEARCH"를 출력하지 마세요. 무조건 질문으로 응답하세요.
"""

    prompt = f"""당신은 한국 법률 정보 AI입니다. 사용자의 법률 상황을 파악하는 중입니다.
{first_message_instruction}
대화 내역:
{history_text if history_text else "(첫 번째 메시지)"}

사용자: {question}

판단: 법령·판례 검색에 필요한 핵심 정보가 충분한가?

카테고리별 필요 정보:
- 노동(해고/임금): 고용 기간, 사업장 규모(5인 이상?), 해고/미지급 사유, 증거 여부
- 가족/이혼: 법률혼/사실혼 구분, 자녀 유무, 재산 규모, 귀책사유
- 폭행: 가해자와의 관계, 부상 정도, 증거(CCTV/목격자/진단서)
- 부동산: 전세/월세 구분, 보증금, 계약 기간 및 위반 내용
- 채권/사기: 금액, 증거(차용증/이체내역), 상대방 관계

사용자가 정보를 모르거나 제공할 수 없다고 하면 "SEARCH"만 출력.
충분한 정보가 있으면 "SEARCH"만 출력.
정보가 부족하면 가장 중요한 추가 질문 1개만 한국어로 출력 (질문 형태, 짧게).

출력:"""

    try:
        import requests as _req
        resp = _req.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=20,
        )
        result = resp.json().get("response", "").strip()
        # 첫 메시지에서 SEARCH 응답이 와도 무시하고 강제 질문 생성
        if ai_turns == 0 and result.upper().startswith("SEARCH"):
            follow_up = _generate_follow_up(question)
            return {"action": "ask", "reply": follow_up}
        if result.upper().startswith("SEARCH"):
            return {"action": "search"}
        return {"action": "ask", "reply": result}
    except Exception:
        return {"action": "search"}


def _generate_follow_up(question: str) -> str:
    """첫 메시지에서 SEARCH가 나왔을 때 강제로 추가 질문 생성."""
    prompt = f"""다음 법률 상황에서 가장 중요한 추가 정보 1가지를 짧은 질문으로 물어보세요.

상황: {question}

카테고리별 핵심 질문:
- 노동: 사업장 규모가 5인 이상인가요?
- 부동산: 전세 계약인가요, 월세 계약인가요?
- 가족/이혼: 법률혼(혼인신고)인가요, 사실혼인가요?
- 폭행: 피해 정도(진단서 발급 여부)는 어떻게 되나요?
- 채권: 차용증이나 이체 내역 등 증거가 있나요?

질문만 출력 (한 문장, 짧게):"""
    try:
        import requests as _req
        resp = _req.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=15,
        )
        return resp.json().get("response", "").strip() or "조금 더 자세한 상황을 알려주시겠어요?"
    except Exception:
        return "조금 더 자세한 상황을 알려주시겠어요?"


# ── History 포함 Qdrant 실행 ─────────────────────────────────

def run_qdrant_with_history(question: str, history: list[dict]) -> QueryResponse:
    from qdrant_rag import load_embed_model, hybrid_retrieve, generate_with_gemini, _BM25_DISABLED, _bm25_data, _bm25_lock, build_bm25_from_qdrant
    import qdrant_rag as _qr

    model = load_embed_model()

    # BM25
    if _BM25_DISABLED:
        bm25, chunk_ids = None, None
    elif _bm25_data is not None:
        bm25, chunk_ids = _bm25_data
    elif _bm25_lock.acquire(blocking=False):
        _bm25_lock.release()
        bm25, chunk_ids = build_bm25_from_qdrant()
    else:
        bm25, chunk_ids = None, None

    # Rewrite query
    search_query = rewrite_query(question, history)

    t0 = time.time()
    try:
        docs = hybrid_retrieve(search_query, model, bm25, chunk_ids)
        gen = generate_with_gemini(question, docs, history)
    finally:
        _qr._kiwi = None

    latency = round(time.time() - t0, 1)

    retrieved_docs = [
        RetrievedDoc(
            law_name=d.get("law_name", ""),
            article_num=d.get("article_num", ""),
            text=d.get("text", ""),
            source=d.get("source"),
            reranker_score=d.get("reranker_score"),
            is_relevant=True,
        )
        for d in docs
    ]

    answer = gen.get("answer", "")
    if not answer.strip():
        raise RuntimeError("답변 생성 실패")

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
