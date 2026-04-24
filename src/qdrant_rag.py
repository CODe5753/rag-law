"""
Phase 6: Qdrant 기반 RAG 파이프라인 (multi-collection)

컬렉션:
  - legal_statutes      → 법령 (158k+ points)
  - legal_interpretation → 법령해석례 (86k+ points)
  - case_law            → 판례 (44k+ points)
"""

import json
import os
import threading
import time
from pathlib import Path as _Path

# .env 자동 로드 (QDRANT_HOST 등이 환경변수에 없을 때 대비)
try:
    from dotenv import load_dotenv as _load_dotenv
    _env = _Path(__file__).parent.parent / ".env"
    if _env.exists():
        _load_dotenv(_env, override=False)
except ImportError:
    pass
from pathlib import Path
from collections import defaultdict

import requests
from rank_bm25 import BM25Okapi
import embedding_backend

# sentence_transformers/torch는 local 모드일 때만 import (메모리 절감)
if not embedding_backend.is_ollama_backend():
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = None
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
OUT_DIR = Path("data/processed")
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "local")  # "remote" | "local"
RERANKER_HOST = os.getenv("RERANKER_HOST", "http://localhost:11435")

BM25_CACHE_PATH = Path("data/processed/bm25_cache.pkl")

# 스레드 안전 BM25 싱글턴 — 동시 빌드 방지
_bm25_lock = threading.Lock()
_bm25_data: tuple | None = None  # (BM25Okapi, chunk_ids)

COLLECTIONS = {
    "legal_statutes": "법령",
    "legal_interpretation": "법령해석례",
    "case_law": "판례",
}

TOP_K_CANDIDATE = 20
TOP_K_FINAL = 5

SYSTEM_PROMPT = """당신은 한국 법령·판례 기반 법률 정보 검색 AI입니다.
아래 법령 조문, 법령해석례, 판례를 참고하여 관련 법률 정보를 제공하세요.

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


# ── 싱글턴 ──────────────────────────────────────────────

_kiwi = None
_bge_reranker = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi
            print("[Kiwi] 초기화...")
            _kiwi = Kiwi()
            print("  완료")
        except ImportError:
            _kiwi = False
    return _kiwi if _kiwi is not False else None


def _get_bge_reranker():
    global _bge_reranker
    if _bge_reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("[Re-ranker] BAAI/bge-reranker-v2-m3 로드...")
            _bge_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            print("  완료")
        except Exception as e:
            print(f"  [WARN] BGE Reranker 로드 실패: {e}")
            _bge_reranker = False
    return _bge_reranker if _bge_reranker is not False else None


# ── 임베딩 모델 ─────────────────────────────────────────

def load_embed_model():
    if embedding_backend.is_ollama_backend():
        print(f"[임베딩] Ollama 백엔드 사용: {embedding_backend.OLLAMA_EMBEDDING_MODEL}")
        return None
    print(f"[임베딩] 모델 로드: {EMBED_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"  로드 완료: {round(time.time()-t0, 1)}s")
    return model


# ── BM25 인덱스 (Qdrant 스크롤 기반) ───────────────────────

def build_bm25_from_qdrant() -> tuple[BM25Okapi, list[str]]:
    """Qdrant 전체 스크롤로 BM25 인덱스 구축.

    스레드 안전: _bm25_lock으로 동시 빌드 방지.
    메모리 캐시(_bm25_data) 우선 → 디스크 캐시 → Qdrant 스크롤 순으로 시도.
    """
    import pickle
    global _bm25_data

    # 메모리 캐시 (파드 내 재호출 시 즉시 반환)
    if _bm25_data is not None:
        return _bm25_data

    with _bm25_lock:
        # lock 획득 후 재확인 (대기 중 다른 스레드가 완료했을 수 있음)
        if _bm25_data is not None:
            return _bm25_data

        if BM25_CACHE_PATH.exists():
            print(f"[BM25] 캐시 로드: {BM25_CACHE_PATH}")
            with open(BM25_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("version") == 2:
                _bm25_data = (cached["bm25"], cached["chunk_ids"])
                return _bm25_data
            print("  [BM25] 구버전 캐시 → 재빌드")

        print("[BM25] Qdrant에서 텍스트 로드 중...")
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
        all_chunks = []

        for col_name in COLLECTIONS:
            print(f"  {col_name} 스크롤 중...")
            offset = None
            while True:
                result = qdrant.scroll(
                    collection_name=col_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["text"],
                    with_vectors=False,
                )
                for pt in result[0]:
                    text = (pt.payload or {}).get("text", "")
                    if text:
                        all_chunks.append((f"{col_name}:{pt.id}", text))
                offset = result[1]
                if offset is None:
                    break
            print(f"    → {len(all_chunks)} 누적")

        print(f"[BM25] 총 {len(all_chunks)}개 청크 형태소 분석 중...")
        kiwi = _get_kiwi()
        chunk_ids = [c[0] for c in all_chunks]
        texts = [c[1] for c in all_chunks]

        if kiwi:
            tokenized = [
                [t.form for t in kiwi.tokenize(text) if t.tag not in ("SF", "SP")]
                for text in texts
            ]
        else:
            tokenized = [text.split() for text in texts]

        bm25 = BM25Okapi(tokenized)

        BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"version": 2, "bm25": bm25, "chunk_ids": chunk_ids}, f)
        print(f"[BM25] 캐시 저장: {BM25_CACHE_PATH}")

        _bm25_data = (bm25, chunk_ids)
        return _bm25_data


# ── 멀티 컬렉션 Qdrant 검색 ──────────────────────────────

def search_all_collections(query_vec: list[float], top_k: int = TOP_K_CANDIDATE) -> list[dict]:
    """3개 컬렉션을 순차 검색하여 결과 병합."""
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    per_col = max(top_k // len(COLLECTIONS), 5)
    results = []

    for col_name in COLLECTIONS:
        try:
            pts = qdrant.query_points(
                collection_name=col_name,
                query=query_vec,
                limit=per_col,
                with_payload=True,
            ).points
            for r in pts:
                payload = r.payload or {}
                src = payload.get("source", col_name)
                # 출처별 표시 필드 정규화
                if src == "prec":
                    display_name = payload.get("법원명", "")
                    display_article = payload.get("사건번호", "")
                elif src == "expc":
                    display_name = payload.get("안건명", "")
                    display_article = payload.get("해석기관", "법제처")
                else:
                    display_name = payload.get("law_name", "")
                    display_article = payload.get("article_num", "")
                results.append({
                    "chunk_id": f"{col_name}:{r.id}",
                    "text": payload.get("text", ""),
                    "source": src,
                    "collection": col_name,
                    "score": round(r.score, 4),
                    "law_name": display_name,
                    "article_num": display_article,
                })
        except Exception as e:
            print(f"  [WARN] {col_name} 검색 실패: {e}")

    return results


# ── RRF ────────────────────────────────────────────────

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Re-ranking ──────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    if RERANKER_BACKEND == "remote":
        return _rerank_remote(query, candidates, top_k)

    reranker = _get_bge_reranker()
    if reranker:
        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "reranker_score": round(float(s), 4)} for s, c in ranked[:top_k]]
    print("  [WARN] BGE Reranker 미사용 — LLM 채점으로 전환됩니다 (레이턴시 증가)")
    return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


def _rerank_remote(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """macOS Reranker 서비스(host.lima.internal:11435)로 점수 요청."""
    try:
        resp = requests.post(
            f"{RERANKER_HOST}/rerank",
            json={"query": query, "documents": [c["text"] for c in candidates]},
            timeout=30,
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "reranker_score": round(float(s), 4)} for s, c in ranked[:top_k]]
    except Exception as e:
        print(f"  [WARN] Remote reranker 실패: {e} — LLM 채점으로 전환")
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


# ── 컨텍스트 포맷 ────────────────────────────────────────

def _format_chunk(r: dict) -> str:
    src = r.get("source", "")
    text = r["text"]
    if src == "law":
        label = f"[법령: {r.get('law_name', '')}]"
    elif src == "expc":
        label = "[법령해석례]"
    elif src == "prec":
        label = f"[판례: {r.get('사건번호', '')} {r.get('법원명', '')}]"
    else:
        label = f"[{src}]"
    return f"{label}\n{text}"


# ── 하이브리드 검색 ──────────────────────────────────────

def hybrid_retrieve(
    query: str,
    model,
    bm25: BM25Okapi | None,
    chunk_ids: list[str] | None,
    top_k_candidate: int = TOP_K_CANDIDATE,
    top_k_final: int = TOP_K_FINAL,
) -> list[dict]:
    """BM25 + 멀티컬렉션 벡터 검색 → RRF → BGE Reranker.

    bm25=None이면 BM25 미준비 상태 — 벡터 전용으로 fallback.
    """
    # 벡터 검색
    if model is None:
        q_vec = embedding_backend.encode_query(query)
    else:
        q_vec = model.encode([query], normalize_embeddings=True).tolist()[0]
    vector_results = search_all_collections(q_vec, top_k=top_k_candidate)

    if bm25 is None or chunk_ids is None:
        # BM25 빌드 중 — 벡터 전용
        print("[hybrid_retrieve] BM25 미준비, 벡터 전용 검색")
        return rerank(query, vector_results, top_k=top_k_final)

    vector_ranking = [r["chunk_id"] for r in vector_results]

    # BM25 검색
    kiwi = _get_kiwi()
    if kiwi:
        tokens = [t.form for t in kiwi.tokenize(query) if t.tag not in ("SF", "SP")]
    else:
        tokens = query.split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranking = [
        chunk_ids[i]
        for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:top_k_candidate]
    ]

    # RRF 결합
    fused = rrf([vector_ranking, bm25_ranking])
    top_ids = {doc_id for doc_id, _ in fused[:top_k_candidate]}

    # chunk_id → chunk 매핑 (Qdrant 결과 우선; BM25 전용 결과는 스킵)
    id_to_chunk: dict[str, dict] = {r["chunk_id"]: r for r in vector_results}
    candidates = [id_to_chunk[cid] for cid in top_ids if cid in id_to_chunk]

    return rerank(query, candidates, top_k=top_k_final)


# ── 답변 생성 ────────────────────────────────────────────

def generate(question: str, retrieved: list[dict]) -> dict:
    ctx = "\n\n".join(_format_chunk(r) for r in retrieved)
    prompt = f"""{SYSTEM_PROMPT}

[참고 자료]
{ctx}

[질문]
{question}

[답변]"""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
    except Exception as e:
        answer = f"[오류] {e}"
    return {"answer": answer, "latency_sec": round(time.time() - t0, 1)}


# ── 평가 ────────────────────────────────────────────────

def run_evaluation():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluator import (
        eval_faithfulness, eval_answer_relevancy,
        eval_context_recall, eval_context_precision,
    )

    print("=== Phase 6: Qdrant RAG 평가 시작 (multi-collection) ===\n")
    model = load_embed_model()
    bm25, chunk_ids = build_bm25_from_qdrant()
    _get_bge_reranker()

    all_scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "context_precision": []}
    sample_results = []
    t_total = time.time()

    for i, qa in enumerate(EVAL_QA):
        q = qa["question"]
        gt = qa["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_QA)}] {q[:50]}...")

        t0 = time.time()
        retrieved = hybrid_retrieve(q, model, bm25, chunk_ids)
        gen = generate(q, retrieved)
        answer = gen["answer"]
        contexts = [r["text"] for r in retrieved]
        rag_latency = round(time.time() - t0, 1)

        faith = eval_faithfulness(answer, contexts)
        relev = eval_answer_relevancy(q, answer)
        recall = eval_context_recall(gt, contexts)
        prec = eval_context_precision(q, contexts)

        print(f"  faith={faith} relev={relev} recall={recall} prec={prec}")

        for key, val in [("faithfulness", faith), ("answer_relevancy", relev), ("context_recall", recall), ("context_precision", prec)]:
            if val is not None:
                all_scores[key].append(val)

        sample_results.append({
            "question": q,
            "answer": answer[:300],
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_recall": recall,
            "context_precision": prec,
            "rag_latency_sec": rag_latency,
            "top_sources": [
                f"{r.get('source','')} {r.get('law_name','') or r.get('사건번호','')} ({r.get('score','-')})"
                for r in retrieved
            ],
        })

    elapsed = round(time.time() - t_total, 1)

    def mean(lst): return round(sum(lst) / len(lst), 4) if lst else None

    final = {
        "phase": "Phase6_Qdrant_MultiCol",
        "mode": "HybridRerank_MultiCollection",
        "eval_time_sec": elapsed,
        "faithfulness": mean(all_scores["faithfulness"]),
        "answer_relevancy": mean(all_scores["answer_relevancy"]),
        "context_recall": mean(all_scores["context_recall"]),
        "context_precision": mean(all_scores["context_precision"]),
        "samples": sample_results,
    }

    print(f"\n=== Phase 6 Qdrant RAG 점수 ===")
    print(f"  phase: {final['phase']}")
    print(f"  eval_time_sec: {elapsed}")
    print(f"  faithfulness:      {final['faithfulness']}")
    print(f"  answer_relevancy:  {final['answer_relevancy']}")
    print(f"  context_recall:    {final['context_recall']}")
    print(f"  context_precision: {final['context_precision']}")

    out = OUT_DIR / "ragas_phase6.json"
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")
    return final


# ── 단일 쿼리 ────────────────────────────────────────────

def run_query(question: str):
    model = load_embed_model()
    bm25, chunk_ids = build_bm25_from_qdrant()

    print(f"\n[검색] '{question}'")
    retrieved = hybrid_retrieve(question, model, bm25, chunk_ids)

    for r in retrieved:
        src = r.get("source", "")
        label = r.get("law_name") or r.get("사건번호") or r.get("section") or ""
        print(f"  [{src}] {label} score={r.get('score', '-')}")
        print(f"    {r['text'][:100]}...")

    print("\n[답변 생성 중...]")
    gen = generate(question, retrieved)
    print(f"\n{gen['answer']}\n(지연: {gen['latency_sec']}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Phase 6 평가 실행")
    parser.add_argument("--query", help="단일 쿼리 실행")
    args = parser.parse_args()

    if args.eval:
        run_evaluation()
    elif args.query:
        run_query(args.query)
    else:
        print("사용법: --eval | --query '질문'")
