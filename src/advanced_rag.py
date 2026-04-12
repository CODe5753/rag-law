"""
Phase 5: Advanced RAG

개선 1: Hybrid Search (BM25 + Vector + RRF)
  - Vector: bge-m3 의미 기반 검색 (현재 사용 중)
  - BM25: 키워드 기반 검색 (rank-bm25 + kiwipiepy 한국어 형태소)
  - RRF(Reciprocal Rank Fusion): 두 순위를 결합

개선 2: Re-ranking (FlashRank)
  - Hybrid Search로 상위 20개 후보 수집
  - Cross-encoder로 query-document 관련성 재채점
  - 상위 5개만 LLM에 전달

개선 3: Multi-query
  - 원본 질문을 3개 변형으로 확장
  - 각 변형으로 검색 → 중복 제거 후 합산

목표: Context Recall 0.70 → 0.79+, Context Precision 0.74 → 0.74+
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_data"))
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "law_chunks_v1"
OUT_DIR = Path("data/processed")

TOP_K_CANDIDATE = 20    # Hybrid 후보 수
TOP_K_FINAL = 5         # Re-rank 후 LLM에 전달할 수

SYSTEM_PROMPT = """당신은 한국 법령 전문가입니다. 아래 법령 조문을 참고하여 질문에 정확하게 답변하세요.
법령에 없는 내용은 '관련 조문을 찾을 수 없습니다'라고 답하세요. 추측하지 마세요."""


# ── 형태소 분석 ──────────────────────────────────────

# Kiwi는 모델 로딩 비용이 크므로 모듈 레벨에서 싱글턴으로 초기화
_kiwi = None

def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi
            print("[Kiwi] 형태소 분석기 초기화...")
            _kiwi = Kiwi()
            print("  완료")
        except ImportError:
            _kiwi = False  # 설치 안됨 표시
    return _kiwi if _kiwi is not False else None


def tokenize_ko(text: str) -> list[str]:
    """
    한국어 형태소 분석으로 BM25 토큰화.
    kiwipiepy 없으면 공백 분리로 폴백.
    ⚠️ 트러블슈팅: Kiwi()를 매번 생성하면 모델 로딩 비용 반복 → 7665개에 5분+
                   싱글턴 패턴으로 1회만 초기화.
    """
    kiwi = _get_kiwi()
    if kiwi:
        tokens = [t.form for t in kiwi.tokenize(text)
                  if t.tag not in ("SF", "SP", "SS", "SE", "SO")]
        return tokens if tokens else text.split()
    return text.split()


# ── BM25 인덱스 ──────────────────────────────────────

class BM25Index:
    """chunks.json을 BM25로 인덱싱"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        print(f"[BM25] 형태소 분석 중 ({len(chunks)}개)...")
        t0 = time.time()
        tokenized = [tokenize_ko(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  완료: {time.time() - t0:.1f}s")

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """쿼리로 BM25 검색. (chunk_idx, score) 리스트 반환."""
        tokens = tokenize_ko(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(idx, float(scores[idx])) for idx in top_indices]


# ── Reciprocal Rank Fusion ────────────────────────────

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    여러 순위 목록을 RRF로 결합.
    rankings: [["id1", "id2", ...], ["id2", "id3", ...]]
    k: RRF 상수 (기본 60, 높을수록 높은 순위에 덜 민감)
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Re-ranking (FlashRank) ────────────────────────────

# BGE Reranker 싱글턴 (한국어 지원 다국어 Cross-encoder)
_bge_reranker = None

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


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    Re-ranking 전략 (우선순위):
      1. BGE Reranker v2-m3 (한국어 지원 다국어 Cross-encoder) — 권장
      2. FlashRank ms-marco (영어 전용, 한국어에서 품질 낮음) — 폴백
      3. 원본 순서 유지 — 최후 폴백

    ⚠️ 트러블슈팅 (2026-04-10):
      ms-marco-MiniLM-L-12-v2(영어)로 Re-ranking 시 Context Recall 0.70→0.33 급락.
      원인: 영어 Cross-encoder가 한국어 조문의 관련성을 잘못 채점.
      해결: bge-reranker-v2-m3 (다국어, 한국어 학습 포함)로 교체.
    """
    # 1순위: BGE 다국어 Reranker
    reranker = _get_bge_reranker()
    if reranker is not None:
        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]

    # 2순위: FlashRank (폴백, 영어 전용)
    try:
        from flashrank import Ranker, RerankRequest
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        return [candidates[r["id"]] for r in results[:top_k]]
    except Exception as e:
        print(f"  [WARN] FlashRank 실패: {e} → 원본 순서")
        return candidates[:top_k]


# ── Multi-query 쿼리 확장 ──────────────────────────────

def expand_query(query: str) -> list[str]:
    """원본 쿼리를 3개 변형으로 확장 (EXAONE 사용)"""
    prompt = f"""다음 법령 관련 질문을 검색에 유리한 3가지 다른 표현으로 바꿔주세요.
한국어로, 각 표현은 한 줄씩, 번호 없이 출력하세요.

원본 질문: {query}

3가지 변형:"""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        lines = [l.strip() for l in resp.json()["response"].strip().split("\n") if l.strip()]
        variants = lines[:3]
        return [query] + variants  # 원본 포함
    except Exception as e:
        print(f"  [WARN] 쿼리 확장 실패: {e}")
        return [query]


# ── Hybrid Search 파이프라인 ──────────────────────────

class AdvancedRAG:
    def __init__(self, chunks: list[dict], embed_model: SentenceTransformer, col):
        self.chunks = chunks
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.embed_model = embed_model
        self.col = col
        self.bm25 = BM25Index(chunks)

    def hybrid_retrieve(self, query: str, top_k: int = TOP_K_CANDIDATE) -> list[dict]:
        """BM25 + Vector → RRF → 중복 제거"""

        # Vector 검색
        q_vec = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        vec_results = self.col.query(
            query_embeddings=q_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = vec_results["ids"][0]
        vec_scores = {id_: round(1 - dist, 4)
                      for id_, dist in zip(vec_results["ids"][0], vec_results["distances"][0])}

        # BM25 검색
        bm25_results = self.bm25.search(query, top_k)
        bm25_ids = [self.chunks[idx]["chunk_id"] for idx, _ in bm25_results]

        # RRF 결합
        fused = rrf([vec_ids, bm25_ids])
        top_ids = [id_ for id_, _ in fused[:top_k]]

        # 결과 조합
        retrieved = []
        for chunk_id in top_ids:
            idx = self.chunk_id_to_idx.get(chunk_id)
            if idx is None:
                continue
            c = self.chunks[idx]
            retrieved.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "law_name": c["law_name"],
                "article_num": c["article_num"],
                "vec_score": vec_scores.get(chunk_id, 0.0),
            })
        return retrieved

    def multi_query_retrieve(self, query: str) -> list[dict]:
        """쿼리 확장 후 각 변형으로 Hybrid Search → 중복 제거"""
        queries = expand_query(query)
        print(f"  쿼리 확장: {len(queries)}개")

        seen_ids = set()
        all_candidates = []
        for q in queries:
            results = self.hybrid_retrieve(q, top_k=TOP_K_CANDIDATE // len(queries) + 5)
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    all_candidates.append(r)

        return all_candidates

    def retrieve(self, query: str, use_multiquery: bool = False) -> list[dict]:
        """검색 → Re-ranking → Top-K"""
        if use_multiquery:
            candidates = self.multi_query_retrieve(query)
        else:
            candidates = self.hybrid_retrieve(query)

        # Re-ranking
        final = rerank(query, candidates, top_k=TOP_K_FINAL)
        return final


def generate(query: str, context_chunks: list[dict]) -> dict:
    """컨텍스트 + 쿼리 → EXAONE 답변"""
    context = "\n\n".join(
        f"[{c['law_name']} {c['article_num']}]\n{c['text']}"
        for c in context_chunks
    )
    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"참고 법령:\n{context}\n\n질문: {query}"},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return {"answer": resp.json()["message"]["content"], "latency_sec": round(time.time() - t0, 2)}


def ask(query: str, rag: AdvancedRAG, use_multiquery: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    retrieved = rag.retrieve(query, use_multiquery=use_multiquery)
    gen = generate(query, retrieved)

    if verbose:
        mode = "MultiQuery+Hybrid+Rerank" if use_multiquery else "Hybrid+Rerank"
        print(f"\n질문: {query} [{mode}]")
        print(f"\n[검색된 조문 Top-{TOP_K_FINAL}]")
        for i, c in enumerate(retrieved, 1):
            print(f"  {i}. {c['law_name']} {c['article_num']} (vec:{c.get('vec_score', 0):.3f})")
        print(f"\n[답변]\n{gen['answer']}")
        print(f"\n[지연시간] 총 {round(time.time()-t0,2)}s (LLM {gen['latency_sec']}s)")

    return {"query": query, "answer": gen["answer"], "sources": retrieved,
            "latency_sec": round(time.time() - t0, 2)}


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from naive_rag import load_embed_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="개인정보 수집 시 동의를 받아야 하는가?")
    parser.add_argument("--multiquery", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Phase 5 평가 실행")
    args = parser.parse_args()

    chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))
    embed_model = load_embed_model()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(COLLECTION_NAME)

    rag = AdvancedRAG(chunks, embed_model, col)

    if args.eval:
        from evaluator import EVAL_QA, eval_faithfulness, eval_answer_relevancy, eval_context_recall, eval_context_precision

        scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "context_precision": []}
        t_total = time.time()

        for i, qa in enumerate(EVAL_QA):
            q = qa["question"]
            gt = qa["ground_truth"]
            print(f"\n[{i+1}/{len(EVAL_QA)}] {q[:50]}...")

            retrieved = rag.retrieve(q, use_multiquery=args.multiquery)
            contexts = [c["text"] for c in retrieved]
            gen = generate(q, retrieved)
            answer = gen["answer"]

            faith = eval_faithfulness(answer, contexts)
            relev = eval_answer_relevancy(q, answer)
            recall = eval_context_recall(gt, contexts)
            prec = eval_context_precision(q, contexts)

            print(f"  faith={faith} relev={relev} recall={recall} prec={prec}")
            if faith is not None: scores["faithfulness"].append(faith)
            if relev is not None: scores["answer_relevancy"].append(relev)
            if recall is not None: scores["context_recall"].append(recall)
            if prec is not None: scores["context_precision"].append(prec)

        elapsed = round(time.time() - t_total, 1)
        def mean(lst): return round(sum(lst)/len(lst), 4) if lst else None
        result = {
            "phase": "Phase5_AdvancedRAG",
            "mode": "MultiQuery" if args.multiquery else "HybridRerank",
            "eval_time_sec": elapsed,
            "faithfulness": mean(scores["faithfulness"]),
            "answer_relevancy": mean(scores["answer_relevancy"]),
            "context_recall": mean(scores["context_recall"]),
            "context_precision": mean(scores["context_precision"]),
        }
        print(f"\n=== Phase 5 Advanced RAG 점수 ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

        out = OUT_DIR / "ragas_phase5.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {out}")
    else:
        ask(args.query, rag, use_multiquery=args.multiquery)
