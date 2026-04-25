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
    """Gemini로 상황 파악 완료 여부 판단.
    충분하면 {"action":"search","summary":"상황요약"}, 아니면 {"action":"ask","reply":"질문(이유포함)"}"""
    ai_turns = sum(1 for h in history if h["role"] == "assistant")

    # 3턴 초과 시 강제 검색 (무한 루프 방지)
    if ai_turns >= 3:
        return {"action": "search", "summary": ""}

    history_text = ""
    for h in history:
        prefix = "사용자" if h["role"] == "user" else "AI"
        history_text += f"{prefix}: {h['content']}\n"

    prompt = f"""당신은 경험 많은 한국 법률 AI 상담사입니다. 사용자의 법률 상황을 파악하여 적절한 법령·판례를 검색하려 합니다.

대화 내역:
{history_text if history_text else "(대화 시작)"}

사용자: {question}

[역할]
실제 법률 상담처럼 사용자의 상황을 파악하세요. 카테고리 체크리스트를 기계적으로 적용하지 마세요.
이 사람의 구체적 상황에서 법적 판단에 실제로 필요한 정보가 무엇인지 판단하세요.

[판단 기준]
1. 지금까지 파악된 정보로 어떤 법령·판례를 검색해야 할지 충분히 알 수 있으면 → SEARCH
2. 검색 방향을 결정하는 데 꼭 필요한 정보가 빠져 있으면 → 추가 질문 1개
   - 질문할 때 반드시 왜 이 정보가 필요한지 1문장 설명 포함
   - 예시: "임대인이 수선을 거부한 경위를 알아야 법적 의무 위반 여부를 판단할 수 있어서요. 거부 의사를 어떻게 전달받으셨나요?"
   - 이 사람 상황과 무관한 정보는 절대 묻지 마세요 (예: 수선의무 분쟁에서 전세/월세 구분, 보증금 액수)
3. 사용자가 "모르겠다", "없다", "잘 모르겠어요" 등으로 제공 불가를 표현하면 → SEARCH
4. AI가 이미 2회 이상 추가 질문을 했으면 → SEARCH

[출력 형식]
- 추가 질문: 이유(1문장) + 질문(1문장), 총 2~3문장 이내로만 출력
- 검색 준비 완료: "SEARCH: " 뒤에 지금까지 파악된 상황을 1~2문장으로 요약

출력:"""

    try:
        from google import genai
        from google.genai import types as gtypes
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("no GEMINI_API_KEY")
        client = genai.Client(
            api_key=api_key,
            http_options=gtypes.HttpOptions(timeout=20000),
        )
        resp = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        result = resp.text.strip()
        if result.upper().startswith("SEARCH"):
            parts = result.split(":", 1)
            summary = parts[1].strip() if len(parts) > 1 else ""
            return {"action": "search", "summary": summary}
        return {"action": "ask", "reply": result}
    except Exception as e:
        logger.warning("intake_classify Gemini failed, fallback to search: %s", e)
        return {"action": "search", "summary": ""}


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


# ── 쿼리 라우팅 ──────────────────────────────────────────────

def route_query(question: str, history: list[dict]) -> str:
    """질문 유형 분류: 'rag' (법령·판례 검색) | 'general' (일반 법률 지식으로 답변 가능)"""
    if not history:
        return "rag"  # 첫 질문은 항상 RAG

    history_text = "\n".join(
        f"{'사용자' if h['role'] == 'user' else 'AI'}: {h['content'][:200]}"
        for h in history[-6:]
    )

    prompt = f"""한국 법률 AI 상담 라우터입니다. 사용자의 새 질문이 법령·판례 검색이 필요한지 판단하세요.

대화 내역:
{history_text}

사용자 새 질문: {question}

[RAG] 법령·판례 검색이 필요한 경우:
- 새로운 법적 쟁점 (적용 법령, 권리·의무 여부, 처벌 수위)
- 처음으로 법적 상황을 설명하는 경우

[GENERAL] 일반 법률 지식으로 충분한 경우:
- 법률 절차·기간·비용 ("얼마나 걸려요?", "비용은 얼마나 드나요?")
- 이미 파악된 상황에서 다음 행동 조언 ("어떻게 해야 하나요?", "증거는 어떻게 수집하나요?")
- 이전 AI 답변에 대한 후속 질문이나 보충 설명 요청

출력: RAG 또는 GENERAL (한 단어만)"""

    try:
        from google import genai
        from google.genai import types as gtypes
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return "rag"
        client = genai.Client(api_key=api_key, http_options=gtypes.HttpOptions(timeout=10000))
        resp = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        result = resp.text.strip().upper()
        if "GENERAL" in result:
            return "general"
        return "rag"
    except Exception as e:
        logger.warning("route_query failed, fallback to rag: %s", e)
        return "rag"


def run_general_answer(question: str, history: list[dict]) -> str:
    """RAG 없이 Gemini 일반 법률 지식으로 답변 (절차·비용·기간·후속 조언)"""
    from google import genai
    from google.genai import types as gtypes

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("run_general_answer: no GEMINI_API_KEY")
        return "현재 답변을 생성할 수 없습니다."

    system_text = """당신은 한국 법률 AI 상담 어시스턴트입니다.
대화 맥락을 바탕으로 절차·비용·기간·다음 단계 등 실무적인 법률 정보를 안내합니다.
법령 원문 검색 없이 일반적인 한국 법률 실무 지식으로 답변 가능한 상황입니다.

[답변 원칙]
- 한국 법률 실무 기준 일반적인 정보를 제공합니다
- 금액·기간은 사건 복잡도에 따라 달라질 수 있음을 명시하세요
- 답변 마지막에 "구체적인 법적 판단과 책임은 변호사와 상담하시기 바랍니다."를 추가하세요"""

    client = genai.Client(api_key=api_key, http_options=gtypes.HttpOptions(timeout=30000))

    contents = []
    for h in (history or [])[-6:]:
        role = "user" if h["role"] == "user" else "model"
        contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=h["content"])]))
    contents.append(gtypes.Content(role="user", parts=[gtypes.Part(text=question)]))

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=gtypes.GenerateContentConfig(system_instruction=system_text),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning("run_general_answer Gemini failed: %s", e)
        return "일시적으로 답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."


# ── 디스패처 ─────────────────────────────────────────────────

def run_pipeline(question: str, pipeline: str) -> QueryResponse:
    if pipeline == "crag":
        return run_crag(question)
    elif pipeline == "qdrant":
        return run_qdrant(question)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
