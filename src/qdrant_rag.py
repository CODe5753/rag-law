"""
Phase 6: Qdrant 기반 RAG 파이프라인

ChromaDB 대비 개선:
  - payload 필터: law_name="개인정보보호법"처럼 특정 법령만 검색 가능
  - 대용량 안정성: ChromaDB 필터링 hang(Issue #4089) 없음
  - 검색 API: query_points() (qdrant-client 1.x, search() 제거됨)

이 파일:
  - Qdrant에서 벡터 검색
  - BM25 하이브리드 + BGE Reranker v2-m3 (Phase 5 파이프라인 유지)
  - 평가 실행 (--eval)
  - 필터 데모 (--filter-demo)
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict

import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
QDRANT_PATH = Path("./qdrant_data")
CHUNKS_PATH = Path("data/processed/chunks.json")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "law_chunks_v1"
OUT_DIR = Path("data/processed")

TOP_K_CANDIDATE = 20
TOP_K_FINAL = 5

SYSTEM_PROMPT = """당신은 한국 법령 전문가입니다. 아래 법령 조문을 참고하여 질문에 정확하게 답변하세요.
법령에 없는 내용은 '관련 조문을 찾을 수 없습니다'라고 답하세요. 추측하지 마세요."""

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


# ── 데이터 로드 ─────────────────────────────────────────

def load_chunks() -> list[dict]:
    """BM25용 청크 로드 (Qdrant payload에도 text 있지만 BM25는 별도 인덱스 필요)"""
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def load_embed_model() -> SentenceTransformer:
    print(f"[임베딩] 모델 로드: {EMBED_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"  로드 완료: {round(time.time()-t0, 1)}s")
    return model


# ── BM25 인덱스 ─────────────────────────────────────────

def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """한국어 형태소 분석 기반 BM25 인덱스 구축"""
    print(f"[BM25] 형태소 분석 중 ({len(chunks)}개)...")
    t0 = time.time()
    kiwi = _get_kiwi()
    chunk_ids = [c["chunk_id"] for c in chunks]

    if kiwi:
        tokenized = [
            [t.form for t in kiwi.tokenize(c["text"]) if t.tag not in ("SF", "SP")]
            for c in chunks
        ]
    else:
        tokenized = [c["text"].split() for c in chunks]

    bm25 = BM25Okapi(tokenized)
    print(f"  완료: {round(time.time()-t0, 1)}s")
    return bm25, chunk_ids


# ── Qdrant 검색 ─────────────────────────────────────────

def search_qdrant(
    query_vec: list[float],
    top_k: int = TOP_K_CANDIDATE,
    law_name: str = None,
) -> list[dict]:
    """Qdrant 벡터 검색. law_name 필터 지원 (ChromaDB에 없던 기능)."""
    qdrant = QdrantClient(path=str(QDRANT_PATH))

    query_filter = None
    if law_name:
        query_filter = Filter(
            must=[FieldCondition(key="law_name", match=MatchValue(value=law_name))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [
        {
            "chunk_id": r.payload.get("chunk_id", str(r.id)),
            "text": r.payload["text"],
            "law_name": r.payload.get("law_name", ""),
            "article_num": r.payload.get("article_num", ""),
            "score": round(r.score, 4),
        }
        for r in results
    ]


# ── RRF ────────────────────────────────────────────────

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Re-ranking ──────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    reranker = _get_bge_reranker()
    if reranker:
        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        # reranker_score를 doc에 포함 (CRAG grading에서 재활용)
        return [{**c, "reranker_score": round(float(s), 4)} for s, c in ranked[:top_k]]
    # fallback: BGE Reranker 미사용 — reranker_score 없이 반환 (CRAG는 LLM 채점으로 fallback)
    print("  [WARN] BGE Reranker 미사용 — LLM 채점으로 전환됩니다 (레이턴시 증가)")
    return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


# ── 하이브리드 검색 ──────────────────────────────────────

def hybrid_retrieve(
    query: str,
    model: SentenceTransformer,
    chunks: list[dict],
    bm25: BM25Okapi,
    chunk_ids: list[str],
    top_k_candidate: int = TOP_K_CANDIDATE,
    top_k_final: int = TOP_K_FINAL,
    law_name: str = None,
) -> list[dict]:
    """BM25 + Qdrant 벡터 검색 → RRF → BGE Reranker"""
    # 벡터 검색
    q_vec = model.encode([query], normalize_embeddings=True).tolist()[0]
    vector_results = search_qdrant(q_vec, top_k=top_k_candidate, law_name=law_name)
    vector_ranking = [r["chunk_id"] for r in vector_results]

    # BM25 검색
    kiwi = _get_kiwi()
    if kiwi:
        tokens = [t.form for t in kiwi.tokenize(query) if t.tag not in ("SF", "SP")]
    else:
        tokens = query.split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranking = [chunk_ids[i] for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:top_k_candidate]]

    # RRF 결합
    fused = rrf([vector_ranking, bm25_ranking])
    top_ids = {doc_id for doc_id, _ in fused[:top_k_candidate]}

    # chunk_id → chunk 매핑 (Qdrant payload에서 가져온 결과 우선)
    id_to_chunk: dict[str, dict] = {r["chunk_id"]: r for r in vector_results}
    # BM25 상위 중 Qdrant에 없는 것 보충
    chunk_map = {c["chunk_id"]: c for c in chunks}
    for cid in top_ids:
        if cid not in id_to_chunk and cid in chunk_map:
            id_to_chunk[cid] = chunk_map[cid]

    candidates = [id_to_chunk[cid] for cid in top_ids if cid in id_to_chunk]

    # Re-ranking
    return rerank(query, candidates, top_k=top_k_final)


# ── 답변 생성 ────────────────────────────────────────────

def generate(question: str, retrieved: list[dict]) -> dict:
    ctx = "\n\n".join(
        f"[{r['law_name']} {r['article_num']}]\n{r['text']}" for r in retrieved
    )
    prompt = f"""{SYSTEM_PROMPT}

[참고 법령]
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
    import re
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluator import (
        eval_faithfulness, eval_answer_relevancy,
        eval_context_recall, eval_context_precision,
    )

    print("=== Phase 6: Qdrant RAG 평가 시작 ===\n")
    model = load_embed_model()
    chunks = load_chunks()
    bm25, chunk_ids = build_bm25(chunks)
    # BGE reranker 미리 로드
    _get_bge_reranker()

    all_scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "context_precision": []}
    sample_results = []
    t_total = time.time()

    for i, qa in enumerate(EVAL_QA):
        q = qa["question"]
        gt = qa["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_QA)}] {q[:50]}...")

        t0 = time.time()
        retrieved = hybrid_retrieve(q, model, chunks, bm25, chunk_ids)
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
            "top_sources": [f"{r['law_name']} {r['article_num']} ({r['score']})" for r in retrieved],
        })

    elapsed = round(time.time() - t_total, 1)

    def mean(lst): return round(sum(lst) / len(lst), 4) if lst else None

    final = {
        "phase": "Phase6_Qdrant",
        "mode": "HybridRerank_Qdrant",
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

def run_query(question: str, law_name: str = None):
    model = load_embed_model()
    chunks = load_chunks()
    bm25, chunk_ids = build_bm25(chunks)

    print(f"\n[검색] '{question}'" + (f" (필터: {law_name})" if law_name else ""))
    retrieved = hybrid_retrieve(question, model, chunks, bm25, chunk_ids, law_name=law_name)

    for r in retrieved:
        print(f"  [{r['law_name']} {r['article_num']}] score={r.get('score', '-')}")
        print(f"    {r['text'][:100]}...")

    print("\n[답변 생성 중...]")
    gen = generate(question, retrieved)
    print(f"\n{gen['answer']}\n(지연: {gen['latency_sec']}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Phase 6 평가 실행")
    parser.add_argument("--query", help="단일 쿼리 실행")
    parser.add_argument("--law", help="법령명 필터 (예: 개인정보보호법)")
    args = parser.parse_args()

    if args.eval:
        run_evaluation()
    elif args.query:
        run_query(args.query, law_name=args.law)
    else:
        print("사용법: --eval | --query '질문' [--law '법령명']")
