"""
Phase 2: Naive RAG — LangChain 없이 직접 구현

목적: 원리 이해 + RAGAS 기준선 측정
스택: bge-m3 임베딩 + ChromaDB + Ollama(EXAONE 3.5 7.8B)

파이프라인:
  1. 청크 로드 (data/processed/chunks.json)
  2. bge-m3로 임베딩 → ChromaDB 저장 (최초 1회, 이후 재사용)
  3. 쿼리 임베딩 → 코사인 유사도 Top-K 검색
  4. 컨텍스트 조합 → EXAONE 3.5로 답변 생성
  5. 시스템 리소스 모니터링 (CPU, RAM, 처리 시간)
"""

import json
import time
import os
import threading
from pathlib import Path

import psutil
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ── 설정 ──────────────────────────────────────────────
CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_data"))
COLLECTION_NAME = "law_chunks_v1"
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
TOP_K = 5
BATCH_SIZE = 64   # bge-m3 임베딩 배치 크기


# ── 시스템 리소스 모니터 ───────────────────────────────

class ResourceMonitor:
    """백그라운드 스레드로 CPU/RAM 최대치 추적"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._peak_cpu = 0.0
        self._peak_ram_mb = 0.0
        self._running = False
        self._thread = None
        self.process = psutil.Process()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "peak_cpu_pct": round(self._peak_cpu, 1),
            "peak_ram_mb": round(self._peak_ram_mb, 1),
        }

    def _monitor(self):
        while self._running:
            try:
                cpu = self.process.cpu_percent(interval=None)
                ram = self.process.memory_info().rss / 1024 / 1024
                self._peak_cpu = max(self._peak_cpu, cpu)
                self._peak_ram_mb = max(self._peak_ram_mb, ram)
            except Exception:
                pass
            time.sleep(self.interval)


# ── 임베딩 ────────────────────────────────────────────

def load_embed_model() -> SentenceTransformer:
    print(f"[임베딩] 모델 로드: {EMBED_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"  로드 완료: {time.time() - t0:.1f}s")
    return model


def build_index(model: SentenceTransformer, force: bool = False) -> chromadb.Collection:
    """청크를 임베딩해서 ChromaDB에 저장. 이미 존재하면 재사용."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # 기존 컬렉션 확인
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force:
        col = client.get_collection(COLLECTION_NAME)
        count = col.count()
        if count > 0:
            print(f"[인덱스] 기존 컬렉션 재사용: {count:,}개 청크")
            return col
        client.delete_collection(COLLECTION_NAME)

    col = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"[인덱스] {len(chunks):,}개 청크 임베딩 시작 (배치={BATCH_SIZE})")

    monitor = ResourceMonitor()
    monitor.start()
    t0 = time.time()

    ids, embeddings, documents, metadatas = [], [], [], []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vecs = model.encode(texts, normalize_embeddings=True).tolist()

        for c, vec in zip(batch, vecs):
            ids.append(c["chunk_id"])
            embeddings.append(vec)
            documents.append(c["text"])
            metadatas.append({
                "law_name": c["law_name"],
                "article_num": c["article_num"],
                "char_len": c["char_len"],
            })

        done = min(i + BATCH_SIZE, len(chunks))
        elapsed = time.time() - t0
        print(f"  {done:,}/{len(chunks):,} ({done/len(chunks)*100:.0f}%) | {elapsed:.1f}s")

    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    elapsed = time.time() - t0
    resources = monitor.stop()

    print(f"\n[인덱스 완료]")
    print(f"  소요 시간: {elapsed:.1f}s ({elapsed/60:.1f}분)")
    print(f"  처리 속도: {len(chunks)/elapsed:.0f} 청크/초")
    print(f"  피크 CPU: {resources['peak_cpu_pct']}%")
    print(f"  피크 RAM: {resources['peak_ram_mb']:.0f} MB")

    # 리소스 기록 저장
    log_path = Path("data/processed/embedding_perf.json")
    log_path.write_text(json.dumps({
        "model": EMBED_MODEL_NAME,
        "total_chunks": len(chunks),
        "batch_size": BATCH_SIZE,
        "elapsed_sec": round(elapsed, 1),
        "chunks_per_sec": round(len(chunks) / elapsed, 1),
        **resources,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  리소스 로그: {log_path}")

    return col


# ── 검색 ──────────────────────────────────────────────

def retrieve(query: str, model: SentenceTransformer, col: chromadb.Collection, top_k: int = TOP_K) -> list[dict]:
    """쿼리 임베딩 → ChromaDB 유사도 검색"""
    q_vec = model.encode([query], normalize_embeddings=True).tolist()
    results = col.query(
        query_embeddings=q_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "law_name": meta["law_name"],
            "article_num": meta["article_num"],
            "score": round(1 - dist, 4),   # cosine distance → similarity
        })
    return chunks


# ── 생성 ──────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 한국 법령 전문가입니다. 아래 법령 조문을 참고하여 질문에 정확하게 답변하세요.
법령에 없는 내용은 "관련 조문을 찾을 수 없습니다"라고 답하세요. 추측하지 마세요."""


def generate(query: str, context_chunks: list[dict]) -> dict:
    """컨텍스트 + 쿼리 → Ollama EXAONE 답변 생성"""
    context = "\n\n".join(
        f"[{c['law_name']} {c['article_num']}]\n{c['text']}"
        for c in context_chunks
    )
    user_msg = f"참고 법령:\n{context}\n\n질문: {query}"

    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data["message"]["content"]
    latency = round(time.time() - t0, 2)
    return {"answer": answer, "latency_sec": latency}


# ── RAG 파이프라인 ─────────────────────────────────────

def ask(query: str, model: SentenceTransformer, col: chromadb.Collection, verbose: bool = True) -> dict:
    """단일 쿼리 처리: 검색 → 생성 → 반환"""
    t0 = time.time()

    # 검색
    retrieved = retrieve(query, model, col)

    # 생성
    result = generate(query, retrieved)

    total = round(time.time() - t0, 2)
    output = {
        "query": query,
        "answer": result["answer"],
        "sources": retrieved,
        "latency": {"total_sec": total, "llm_sec": result["latency_sec"]},
    }

    if verbose:
        print(f"\n질문: {query}")
        print(f"\n[검색된 조문 Top-{TOP_K}]")
        for i, c in enumerate(retrieved, 1):
            print(f"  {i}. {c['law_name']} {c['article_num']} (유사도: {c['score']})")
        print(f"\n[답변]\n{result['answer']}")
        print(f"\n[지연시간] 총 {total}s (LLM {result['latency_sec']}s)")

    return output


# ── 메인 ──────────────────────────────────────────────

SAMPLE_QUERIES = [
    "개인정보 수집 시 반드시 동의를 받아야 하는가?",
    "근로자의 주당 최대 근로시간은 얼마인가?",
    "저작권 보호 기간은 얼마나 되는가?",
    "전자금융거래에서 이용자 인증 방법은?",
    "공정거래법상 시장지배적 사업자의 요건은?",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="단일 쿼리 실행")
    parser.add_argument("--rebuild", action="store_true", help="ChromaDB 인덱스 재구축")
    parser.add_argument("--demo", action="store_true", help="샘플 쿼리 5개 실행")
    args = parser.parse_args()

    model = load_embed_model()
    col = build_index(model, force=args.rebuild)

    if args.query:
        ask(args.query, model, col)
    elif args.demo:
        results = []
        print("\n=== 데모 쿼리 5개 실행 ===\n")
        for q in SAMPLE_QUERIES:
            r = ask(q, model, col, verbose=True)
            results.append(r)
            print("\n" + "─" * 60)

        out = Path("data/processed/demo_results.json")
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장: {out}")
    else:
        # 기본: 인덱스 구축 확인 + 첫 번째 샘플 쿼리
        ask(SAMPLE_QUERIES[0], model, col)
