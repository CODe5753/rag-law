"""
Phase 7: Modular RAG — CRAG (Corrective RAG) + LangGraph

CRAG 논문 (arXiv:2401.15884) 핵심 아이디어:
  검색된 문서를 LLM으로 채점 → 관련성 낮으면 자기교정
  (웹 검색 or 거부) → 관련성 높은 문서만 사용해 답변 생성

워크플로우:
  retrieve → grade_docs → [모두 무관] → fallback_response
                        → [일부 관련] → generate → END

LangGraph로 구현하면:
  - 상태 기반 조건 분기가 명확해짐
  - 각 노드가 독립적으로 테스트 가능
  - 나중에 노드 추가/수정이 쉬움

웹 검색 옵션:
  TAVILY_API_KEY 환경변수 설정 시 웹 검색 fallback 활성화
  미설정 시 "관련 법령 없음" 응답
"""

import json
import os
import re
import time
from pathlib import Path
from typing import TypedDict, Literal

import requests
from langgraph.graph import StateGraph, END
import embedding_backend

# sentence_transformers/torch는 local 모드일 때만 import (메모리 절감)
if not embedding_backend.is_ollama_backend():
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
QDRANT_PATH = Path("./qdrant_data")
CHUNKS_PATH = Path("data/processed/chunks.json")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
OUT_DIR = Path("data/processed")

# CRAG 판단 임계값
RELEVANCE_THRESHOLD = 0.5   # 이 점수 이상이면 관련 문서로 판정
RERANKER_THRESHOLD = 0.5    # BGE Reranker score 기준 (sigmoid 출력 0~1; 0.5 이상이면 관련)
MIN_RELEVANT_DOCS = 1       # 최소 1개 이상이면 generate (2→1로 완화)

SYSTEM_PROMPT = """당신은 한국 법령·판례 기반 법률 정보 검색 AI입니다.
아래 법령 조문, 판례를 참고하여 관련 법률 정보를 제공하세요.

규칙:
- 제공된 자료에 근거한 법률 정보만 안내합니다. 추측하지 마세요.
- 승소 가능성, 형량 예측, 구체적 법적 결론은 제시하지 않습니다.
- 관련 자료가 없으면 '관련 법령·판례를 찾을 수 없습니다'라고 답하세요.
- 답변 마지막에 '구체적인 법적 판단은 변호사와 상담하세요.'를 한 줄 추가하세요."""

EVAL_QA = [
    {
        "question": "개인정보 수집 시 정보주체에게 알려야 할 사항은 무엇인가?",
        "ground_truth": "개인정보보호법 제15조에 따라 수집·이용 목적, 수집 항목, 보유·이용 기간, 동의 거부 권리 및 불이익을 알려야 한다.",
    },
    {
        "question": "근로기준법상 성인 근로자의 1주 법정 근로시간은?",
        "ground_truth": "근로기준법 제50조에 따라 1주간 근로시간은 휴게시간 제외 40시간을 초과할 수 없다. 합의 시 12시간 연장 가능(최대 52시간).",
    },
    {
        "question": "저작자 사망 후 저작권은 몇 년간 보호되는가?",
        "ground_truth": "저작권법 제39조에 따라 저작재산권은 저작자 생존기간과 사망 후 70년간 존속한다.",
    },
    {
        "question": "공정거래법상 시장지배적 사업자로 추정되는 시장점유율 기준은?",
        "ground_truth": "공정거래법 제6조에 따라 1개 사업자 50% 이상, 또는 3개 이하 사업자 합산 75% 이상이면 시장지배적 사업자로 추정된다. 연간 매출액 80억원 미만 제외.",
    },
    {
        "question": "전자금융거래법상 접근매체의 종류는 무엇인가?",
        "ground_truth": "전자금융거래법 제2조에 따라 전자식 카드, 전자서명생성정보, 인증서, 이용자번호, 생체정보, 비밀번호 등이 접근매체이다.",
    },
    {
        "question": "저작권법상 업무상저작물의 보호기간은?",
        "ground_truth": "저작권법 제41조에 따라 업무상저작물은 공표한 때부터 70년간 존속한다. 창작 후 50년 이내 미공표 시 창작 시부터 70년.",
    },
]


# ── 상태 정의 ─────────────────────────────────────────────

class CRAGState(TypedDict):
    question: str
    documents: list[dict]         # 검색된 원본 문서
    relevant_docs: list[dict]     # 관련성 검증 통과 문서
    generation: str               # 최종 답변
    fallback_used: bool           # fallback 사용 여부
    grading_log: list[dict]       # 각 문서 채점 기록 (디버깅용)


# ── 싱글턴 ───────────────────────────────────────────────

_model = None
_chunks = None
_bm25 = None
_chunk_ids = None
_bge_reranker = None


def _get_model():
    global _model
    if embedding_backend.is_ollama_backend():
        return None
    if _model is None:
        print(f"[임베딩] 모델 로드: {EMBED_MODEL_NAME}")
        t0 = time.time()
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"  완료: {round(time.time()-t0, 1)}s")
    return _model


def _get_chunks():
    global _chunks
    if _chunks is None:
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return _chunks


def _get_bm25():
    global _bm25, _chunk_ids
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        chunks = _get_chunks()
        _chunk_ids = [c["chunk_id"] for c in chunks]
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            tokenized = [
                [t.form for t in kiwi.tokenize(c["text"]) if t.tag not in ("SF", "SP")]
                for c in chunks
            ]
        except ImportError:
            tokenized = [c["text"].split() for c in chunks]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _chunk_ids


def _get_bge_reranker():
    global _bge_reranker
    import os
    if os.getenv("RERANKER_BACKEND", "local") == "remote":
        return None  # qdrant_rag.rerank()이 remote 서비스 호출
    if _bge_reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _bge_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        except Exception:
            _bge_reranker = False
    return _bge_reranker if _bge_reranker is not False else None


# ── Ollama 유틸 ───────────────────────────────────────────

def ollama_call(prompt: str, timeout: int = 90) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [WARN] Ollama 실패: {e}")
        return ""


def extract_score(text: str) -> float:
    """응답에서 0~1 점수 추출. 실패 시 0.5 반환."""
    m = re.search(r"(?:점수|score)[:\s]+([01]\.\d+|\d)", text, re.IGNORECASE)
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    nums = re.findall(r"\b([01]\.\d+)\b", text)
    if nums:
        return float(nums[0])
    # 예/아니오 판단
    if re.search(r"관련|relevant|yes|예\b|맞|있음", text, re.IGNORECASE):
        return 0.8
    if re.search(r"무관|irrelevant|no|아니|없음", text, re.IGNORECASE):
        return 0.2
    return 0.5


# ── 검색 노드 ─────────────────────────────────────────────

def retrieve(state: CRAGState) -> CRAGState:
    """Qdrant hybrid search로 상위 문서 검색"""
    from qdrant_rag import hybrid_retrieve

    question = state["question"]
    model = _get_model()
    chunks = _get_chunks()
    bm25, chunk_ids = _get_bm25()

    # CRAG는 더 많은 후보를 채점하기 위해 top_k_candidate를 늘림
    docs = hybrid_retrieve(
        question, model, chunks, bm25, chunk_ids,
        top_k_candidate=20,
        top_k_final=8,   # 채점 후 필터링하므로 더 많이 가져옴
    )
    print(f"  [retrieve] {len(docs)}개 문서 검색됨")
    return {**state, "documents": docs}


# ── 문서 채점 노드 ────────────────────────────────────────

def grade_documents(state: CRAGState) -> CRAGState:
    """각 문서의 관련성 채점.

    BGE Reranker score가 있으면 그것을 우선 사용 (LLM 호출 없음, ~8x 빠름).
    Reranker score가 없으면 LLM으로 채점 (fallback).
    """
    question = state["question"]
    docs = state["documents"]
    relevant = []
    log = []

    for doc in docs:
        if "reranker_score" in doc:
            score = doc["reranker_score"]
            is_relevant = score >= RERANKER_THRESHOLD
            graded_by = "reranker"
        else:
            prompt = f"""다음 질문과 법령 조문을 보고, 이 조문이 질문에 답하는 데 관련이 있는지 평가하세요.

[질문]
{question}

[법령 조문]
{doc['text']}

관련성을 0.0(전혀 무관)에서 1.0(매우 관련) 사이 점수로 "점수: X.X" 형식으로만 출력하세요."""
            response = ollama_call(prompt)
            score = extract_score(response)
            is_relevant = score >= RELEVANCE_THRESHOLD
            graded_by = "llm"

        log.append({
            "chunk_id": doc.get("chunk_id", ""),
            "law": f"{doc.get('law_name', '')} {doc.get('article_num', '')}",
            "relevance_score": score,
            "is_relevant": is_relevant,
            "graded_by": graded_by,
        })

        if is_relevant:
            relevant.append(doc)

    methods = set(l["graded_by"] for l in log)
    print(f"  [grade] {len(relevant)}/{len(docs)}개 관련 문서 ({'/'.join(methods)})")
    return {**state, "relevant_docs": relevant, "grading_log": log}


# ── 조건 분기 ─────────────────────────────────────────────

def decide_next(state: CRAGState) -> Literal["generate", "fallback"]:
    """관련 문서가 충분하면 generate, 부족하면 fallback"""
    n_relevant = len(state.get("relevant_docs", []))
    if n_relevant >= MIN_RELEVANT_DOCS:
        print(f"  [decide] → generate ({n_relevant}개 관련)")
        return "generate"
    else:
        print(f"  [decide] → fallback ({n_relevant}개 < {MIN_RELEVANT_DOCS}개)")
        return "fallback"


# ── 생성 노드 ─────────────────────────────────────────────

def generate(state: CRAGState) -> CRAGState:
    """관련 문서로 답변 생성"""
    question = state["question"]
    docs = state["relevant_docs"]

    ctx = "\n\n".join(
        f"[{d.get('law_name', '')} {d.get('article_num', '')}]\n{d['text']}"
        for d in docs
    )
    prompt = f"""{SYSTEM_PROMPT}

[참고 법령]
{ctx}

[질문]
{question}

[답변]"""

    answer = ollama_call(prompt)
    print(f"  [generate] 답변 생성 완료 ({len(answer)}자)")
    return {**state, "generation": answer, "fallback_used": False}


# ── Fallback 노드 ─────────────────────────────────────────

def fallback(state: CRAGState) -> CRAGState:
    """관련 문서 부족 → 웹 검색 or 거부 응답"""
    question = state["question"]

    # Tavily 웹 검색 시도 (API 키 있을 때만)
    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            results = client.search(question, max_results=3, search_depth="basic")
            web_ctx = "\n\n".join(
                f"[웹 검색 결과]\n{r['content'][:300]}"
                for r in results.get("results", [])[:3]
            )
            prompt = f"""다음 웹 검색 결과를 참고해 질문에 답하세요. 확실하지 않으면 불확실하다고 명시하세요.

{web_ctx}

[질문]
{question}

[답변]"""
            answer = ollama_call(prompt)
            print(f"  [fallback] 웹 검색 활용")
            return {**state, "generation": answer, "fallback_used": True}
        except Exception as e:
            print(f"  [fallback] 웹 검색 실패: {e}")

    # 웹 검색 불가 → 거부 응답
    answer = f"수집된 법령 데이터에서 '{question}'에 대한 충분한 관련 조문을 찾을 수 없습니다. 법제처(law.go.kr) 또는 전문가에게 문의하세요."
    print(f"  [fallback] 관련 조문 없음 응답")
    return {**state, "generation": answer, "fallback_used": True}


# ── 그래프 빌드 ───────────────────────────────────────────

def build_crag_graph():
    """CRAG LangGraph 워크플로우 구성"""
    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_next,
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ── 단일 쿼리 실행 ─────────────────────────────────────────

def run_query(question: str) -> dict:
    app = build_crag_graph()
    print(f"\n[CRAG] 질문: {question}")

    initial_state: CRAGState = {
        "question": question,
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "fallback_used": False,
        "grading_log": [],
    }

    t0 = time.time()
    result = app.invoke(initial_state)
    latency = round(time.time() - t0, 1)

    print(f"\n답변: {result['generation']}")
    print(f"\nfallback={result['fallback_used']} | 지연={latency}s")
    print(f"채점 로그: {result['grading_log']}")
    return {**result, "latency_sec": latency}


# ── 평가 ──────────────────────────────────────────────────

def run_evaluation():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluator import (
        eval_faithfulness, eval_answer_relevancy,
        eval_context_recall, eval_context_precision,
    )

    # 모델 미리 로드
    _get_model()
    _get_chunks()
    _get_bm25()
    _get_bge_reranker()

    app = build_crag_graph()
    print("=== Phase 7: CRAG LangGraph 평가 시작 ===\n")

    all_scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "context_precision": []}
    sample_results = []
    fallback_count = 0
    t_total = time.time()

    for i, qa in enumerate(EVAL_QA):
        q = qa["question"]
        gt = qa["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_QA)}] {q[:50]}...")

        initial: CRAGState = {
            "question": q,
            "documents": [],
            "relevant_docs": [],
            "generation": "",
            "fallback_used": False,
            "grading_log": [],
        }

        t0 = time.time()
        result = app.invoke(initial)
        rag_latency = round(time.time() - t0, 1)

        answer = result["generation"]
        # 평가에는 관련 문서만 사용 (CRAG 핵심)
        contexts = [d["text"] for d in result.get("relevant_docs", result.get("documents", []))]
        if not contexts:
            contexts = [d["text"] for d in result.get("documents", [])]

        faith = eval_faithfulness(answer, contexts) if contexts else None
        relev = eval_answer_relevancy(q, answer)
        recall = eval_context_recall(gt, contexts) if contexts else None
        prec = eval_context_precision(q, contexts) if contexts else None

        print(f"  faith={faith} relev={relev} recall={recall} prec={prec}")
        print(f"  fallback={result['fallback_used']} latency={rag_latency}s")

        if result["fallback_used"]:
            fallback_count += 1

        for key, val in [("faithfulness", faith), ("answer_relevancy", relev), ("context_recall", recall), ("context_precision", prec)]:
            if val is not None:
                all_scores[key].append(val)

        relevant_log = [x for x in result.get("grading_log", []) if x["is_relevant"]]
        sample_results.append({
            "question": q,
            "answer": answer[:300],
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_recall": recall,
            "context_precision": prec,
            "rag_latency_sec": rag_latency,
            "fallback_used": result["fallback_used"],
            "relevant_docs": [x["law"] for x in relevant_log],
            "grading_log": result.get("grading_log", []),
        })

    elapsed = round(time.time() - t_total, 1)

    def mean(lst): return round(sum(lst) / len(lst), 4) if lst else None

    final = {
        "phase": "Phase7_CRAG",
        "mode": "CRAG_LangGraph",
        "eval_time_sec": elapsed,
        "faithfulness": mean(all_scores["faithfulness"]),
        "answer_relevancy": mean(all_scores["answer_relevancy"]),
        "context_recall": mean(all_scores["context_recall"]),
        "context_precision": mean(all_scores["context_precision"]),
        "fallback_count": fallback_count,
        "samples": sample_results,
    }

    print(f"\n=== Phase 7 CRAG 점수 ===")
    print(f"  phase: {final['phase']}")
    print(f"  eval_time_sec: {elapsed}")
    print(f"  faithfulness:      {final['faithfulness']}")
    print(f"  answer_relevancy:  {final['answer_relevancy']}")
    print(f"  context_recall:    {final['context_recall']}")
    print(f"  context_precision: {final['context_precision']}")
    print(f"  fallback 사용: {fallback_count}/{len(EVAL_QA)}회")

    out = OUT_DIR / "ragas_phase7.json"
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Phase 7 평가")
    parser.add_argument("--query", help="단일 쿼리")
    args = parser.parse_args()

    if args.eval:
        run_evaluation()
    elif args.query:
        run_query(args.query)
    else:
        print("사용법: --eval | --query '질문'")
