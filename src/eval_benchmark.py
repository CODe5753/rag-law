"""
Phase 재측정 벤치마크 — 85개 QA 기준 전 Phase RAGAS 비교

대상:
  Phase 2: ChromaDB 벡터 검색 (naive_rag.py)
  Phase 4: LangChain LCEL RAG (langchain_rag.py)
  Phase 5: Hybrid + BGE Reranker, ChromaDB (advanced_rag.py)
  Phase 6: Hybrid + BGE Reranker, Qdrant (qdrant_rag.py)
  Phase 7: CRAG + LangGraph (crag_rag.py)

실행:
  cd <프로젝트 루트>
  .venv/bin/python src/eval_benchmark.py [--phases 5,6,7] [--limit 20]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
QA_PATH = Path("data/autorag/qa_combined.json")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)


# ── QA 로드 ──────────────────────────────────────────────────────────

def load_qa(limit: int = None) -> list[dict]:
    """data/autorag/qa_combined.json에서 QA 로드"""
    qa_list = json.loads(QA_PATH.read_text(encoding="utf-8"))
    if limit:
        qa_list = qa_list[:limit]
    print(f"[QA 로드] {len(qa_list)}개")
    return qa_list


# ── RAGAS 평가 공통 ──────────────────────────────────────────────────

def build_ragas_dataset(samples: list[dict]) -> EvaluationDataset:
    return EvaluationDataset([
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in samples
    ])


def run_ragas(samples: list[dict], phase_name: str) -> dict:
    dataset = build_ragas_dataset(samples)

    llm = LangchainLLMWrapper(ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=OLLAMA_MODEL, base_url=OLLAMA_HOST,
    ))
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
    ]
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=1)

    t0 = time.time()
    result = evaluate(
        dataset=dataset, metrics=metrics,
        run_config=run_config, raise_exceptions=False, show_progress=True,
    )
    elapsed = round(time.time() - t0, 1)

    df = result.to_pandas()

    def safe_mean(col):
        vals = df[col].dropna()
        return round(float(vals.mean()), 4) if len(vals) > 0 else None

    return {
        "phase": phase_name,
        "n_samples": len(samples),
        "faithfulness": safe_mean("faithfulness"),
        "answer_relevancy": safe_mean("answer_relevancy"),
        "context_recall": safe_mean("context_recall"),
        "context_precision": safe_mean("context_precision"),
        "eval_time_sec": elapsed,
        "nan_count": int(df[["faithfulness","answer_relevancy","context_recall","context_precision"]].isna().sum().sum()),
    }


# ── Phase 2: ChromaDB 벡터 검색 ──────────────────────────────────────

def eval_phase2(qa_list: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 2: ChromaDB 벡터 검색 (naive_rag)")
    print("="*60)

    import chromadb
    from naive_rag import load_embed_model, retrieve, generate

    model = load_embed_model()
    client = chromadb.PersistentClient(path=str(Path("./chroma_data")))
    col = client.get_collection("law_chunks_v1")

    samples = []
    for i, qa in enumerate(qa_list):
        print(f"  [{i+1}/{len(qa_list)}] {qa['question'][:50]}...")
        retrieved = retrieve(qa["question"], model, col, top_k=5)
        result = generate(qa["question"], retrieved)
        samples.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in retrieved],
            "ground_truth": qa["answer"],
        })

    return run_ragas(samples, "Phase2_ChromaDB_Vector")


# ── Phase 4: LangChain LCEL ───────────────────────────────────────────

def eval_phase4(qa_list: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 4: LangChain LCEL RAG")
    print("="*60)

    import chromadb
    from langchain_rag import build_chain, ask_lcel
    from naive_rag import load_embed_model

    model = load_embed_model()
    client = chromadb.PersistentClient(path=str(Path("./chroma_data")))
    col = client.get_collection("law_chunks_v1")
    chain, retriever = build_chain()

    samples = []
    for i, qa in enumerate(qa_list):
        print(f"  [{i+1}/{len(qa_list)}] {qa['question'][:50]}...")
        result = ask_lcel(qa["question"], chain, retriever, verbose=False)
        contexts = [doc.page_content for doc in result.get("source_documents", [])]
        samples.append({
            "question": qa["question"],
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "ground_truth": qa["answer"],
        })

    return run_ragas(samples, "Phase4_LangChain_LCEL")


# ── Phase 5: ChromaDB + Hybrid + BGE Reranker ────────────────────────

def eval_phase5(qa_list: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 5: Hybrid + BGE Reranker (ChromaDB)")
    print("="*60)

    import chromadb
    from advanced_rag import AdvancedRAG, generate

    client = chromadb.PersistentClient(path=str(Path("./chroma_data")))
    rag = AdvancedRAG(client)

    samples = []
    for i, qa in enumerate(qa_list):
        print(f"  [{i+1}/{len(qa_list)}] {qa['question'][:50]}...")
        retrieved = rag.retrieve(qa["question"])
        result = generate(qa["question"], retrieved)
        samples.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in retrieved],
            "ground_truth": qa["answer"],
        })

    return run_ragas(samples, "Phase5_ChromaDB_Hybrid_BGE")


# ── Phase 6: Qdrant + Hybrid + BGE Reranker ──────────────────────────

def eval_phase6(qa_list: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 6: Hybrid + BGE Reranker (Qdrant)")
    print("="*60)

    from qdrant_rag import load_chunks, load_embed_model, build_bm25, hybrid_retrieve, generate
    from qdrant_client import QdrantClient

    chunks = load_chunks()
    model = load_embed_model()
    bm25, chunk_ids = build_bm25(chunks)
    qclient = QdrantClient(path=str(Path("./qdrant_data")))

    samples = []
    for i, qa in enumerate(qa_list):
        print(f"  [{i+1}/{len(qa_list)}] {qa['question'][:50]}...")
        retrieved = hybrid_retrieve(qa["question"], model, qclient, bm25, chunk_ids, chunks)
        result = generate(qa["question"], retrieved)
        samples.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in retrieved],
            "ground_truth": qa["answer"],
        })

    return run_ragas(samples, "Phase6_Qdrant_Hybrid_BGE")


# ── Phase 7: CRAG + LangGraph ─────────────────────────────────────────

def eval_phase7(qa_list: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 7: CRAG + LangGraph")
    print("="*60)

    from crag_rag import run_query

    samples = []
    for i, qa in enumerate(qa_list):
        print(f"  [{i+1}/{len(qa_list)}] {qa['question'][:50]}...")
        result = run_query(qa["question"])
        # CRAG state: result["relevant_docs"] or result["documents"]
        docs = result.get("relevant_docs") or result.get("documents", [])
        contexts = [d["text"] for d in docs] if docs else []
        samples.append({
            "question": qa["question"],
            "answer": result.get("generation", ""),
            "contexts": contexts,
            "ground_truth": qa["answer"],
        })

    return run_ragas(samples, "Phase7_CRAG_LangGraph")


# ── 결과 출력 및 저장 ────────────────────────────────────────────────

PHASE_MAP = {
    "2": eval_phase2,
    "4": eval_phase4,
    "5": eval_phase5,
    "6": eval_phase6,
    "7": eval_phase7,
}

PHASE_LABELS = {
    "2": "Phase 2  | ChromaDB 벡터",
    "4": "Phase 4  | LangChain LCEL",
    "5": "Phase 5  | Hybrid+BGE (ChromaDB)",
    "6": "Phase 6  | Hybrid+BGE (Qdrant)",
    "7": "Phase 7  | CRAG+LangGraph",
}

def print_table(all_scores: list[dict]):
    print("\n" + "="*90)
    print(f"{'Phase':<30} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Prec':>7} {'N':>4} {'Time(s)':>8}")
    print("-"*90)
    for s in all_scores:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A "
        print(f"{s['phase']:<30} {fmt(s['faithfulness']):>7} {fmt(s['answer_relevancy']):>7} "
              f"{fmt(s['context_recall']):>7} {fmt(s['context_precision']):>7} "
              f"{s['n_samples']:>4} {s['eval_time_sec']:>8.0f}")
    print("="*90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="2,5,6,7", help="실행할 Phase 번호 (쉼표 구분, 예: 5,6,7)")
    parser.add_argument("--limit", type=int, default=None, help="QA 최대 개수 (테스트용)")
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",")]
    qa_list = load_qa(limit=args.limit)

    all_scores = []
    t_total = time.time()

    for phase_num in phases:
        if phase_num not in PHASE_MAP:
            print(f"[경고] Phase {phase_num}은 지원하지 않습니다. (지원: {list(PHASE_MAP.keys())})")
            continue

        try:
            scores = PHASE_MAP[phase_num](qa_list)
            all_scores.append(scores)
            print(f"\n  ✓ {scores['phase']}: F={scores['faithfulness']}, R={scores['answer_relevancy']}, "
                  f"Recall={scores['context_recall']}, Prec={scores['context_precision']}")

            # 중간 저장
            out_path = OUT_DIR / f"benchmark_{scores['phase']}.json"
            out_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as e:
            print(f"\n  [ERROR] Phase {phase_num} 실패: {e}")
            import traceback; traceback.print_exc()

    total_elapsed = round(time.time() - t_total, 1)

    if all_scores:
        print_table(all_scores)

        summary_path = OUT_DIR / "benchmark_summary.json"
        summary_path.write_text(json.dumps({
            "total_elapsed_sec": total_elapsed,
            "qa_count": len(qa_list),
            "results": all_scores,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장: {summary_path}")


if __name__ == "__main__":
    main()
