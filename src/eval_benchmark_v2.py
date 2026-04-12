"""
eval_qa_v2 기반 재평가 벤치마크

eval_qa_v1(합성 QA)과 달리 eval_qa_v2는:
  - 조문 원문에서 생성 (순환 평가 없음)
  - 멀티 청크 retrieval_gt 지원
  - Hard negative 20% (fallback 성능 측정)

평가 전략:
  - Answerable(80개): RAGAS (Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision)
  - Negative(20개): fallback 정확도 (올바른 응답="관련 조문 없음" 포함 여부)
  - 기본 대상: Phase 7 (CRAG, 최고 성능) — 다른 Phase는 --phases 옵션으로

실행:
  cd <프로젝트 루트>
  .venv/bin/python src/eval_benchmark_v2.py [--phases 7] [--limit 20]
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
QA_V2_PATH = Path("data/processed/eval_qa_v2.json")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)

FALLBACK_KEYWORDS = ["관련 조문", "찾을 수 없", "없습니다", "해당 내용이 없", "관련 법령 없"]


def load_qa_v2(limit: int = None) -> tuple[list[dict], list[dict]]:
    """eval_qa_v2.json 로드 → answerable / negative 분리"""
    data = json.loads(QA_V2_PATH.read_text(encoding="utf-8"))
    samples = data["samples"]

    answerable = [q for q in samples if q["qa_type"] != "negative"]
    negatives = [q for q in samples if q["qa_type"] == "negative"]

    if limit:
        answerable = answerable[:limit]
        negatives = negatives[:min(limit // 4, len(negatives))]

    print(f"[QA v2 로드] answerable={len(answerable)}, negative={len(negatives)}")
    return answerable, negatives


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
        "faithfulness": safe_mean("faithfulness"),
        "answer_relevancy": safe_mean("answer_relevancy"),
        "context_recall": safe_mean("context_recall"),
        "context_precision": safe_mean("context_precision"),
        "eval_time_sec": elapsed,
        "nan_count": int(df[["faithfulness","answer_relevancy","context_recall","context_precision"]].isna().sum().sum()),
        "n_samples": len(samples),
    }


def eval_negative_accuracy(negatives: list[dict], rag_fn) -> dict:
    """Hard Negative 평가: fallback 정확도 측정"""
    correct = 0
    results = []

    for qa in negatives:
        result = rag_fn(qa["question"])
        answer = result.get("generation") or result.get("answer", "")
        is_correct = any(kw in answer for kw in FALLBACK_KEYWORDS)
        if is_correct:
            correct += 1
        results.append({
            "qid": qa["qid"],
            "question": qa["question"],
            "answer": answer[:200],
            "fallback_correct": is_correct,
            "negative_reason": qa.get("negative_reason", ""),
        })

    accuracy = round(correct / len(negatives), 4) if negatives else 0
    return {"fallback_accuracy": accuracy, "correct": correct, "total": len(negatives), "details": results}


# ── Phase 7: CRAG + LangGraph ──────────────────────────────────────────

def eval_phase7_v2(answerable: list[dict], negatives: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 7 v2: CRAG + LangGraph (eval_qa_v2 기준)")
    print("="*60)

    from crag_rag import run_query

    # Answerable 평가
    samples = []
    for i, qa in enumerate(answerable):
        print(f"  [answerable {i+1}/{len(answerable)}] {qa['question'][:50]}...")
        t0 = time.time()
        result = run_query(qa["question"])
        latency = round(time.time() - t0, 2)
        docs = result.get("relevant_docs") or result.get("documents", [])
        contexts = [d["text"] for d in docs] if docs else []
        samples.append({
            "question": qa["question"],
            "answer": result.get("generation", ""),
            "contexts": contexts,
            "ground_truth": qa["answer"],
            "qa_type": qa["qa_type"],
            "latency_sec": latency,
        })

    ragas_scores = run_ragas(samples, "Phase7_CRAG_v2")

    # QA 타입별 분석
    type_groups = {}
    for s in samples:
        qt = s["qa_type"]
        type_groups.setdefault(qt, []).append(s)

    # Negative 평가
    print(f"\n  [negative 평가] {len(negatives)}개...")
    neg_result = eval_negative_accuracy(negatives, run_query)
    print(f"  fallback 정확도: {neg_result['fallback_accuracy']:.1%} ({neg_result['correct']}/{neg_result['total']})")

    return {
        "phase": "Phase7_CRAG_v2",
        "eval_dataset": "eval_qa_v2",
        "qa_type_counts": {qt: len(qs) for qt, qs in type_groups.items()},
        **ragas_scores,
        "negative_eval": {
            "fallback_accuracy": neg_result["fallback_accuracy"],
            "correct": neg_result["correct"],
            "total": neg_result["total"],
        },
        "samples": [
            {
                "question": s["question"],
                "answer": s["answer"],
                "qa_type": s["qa_type"],
                "faithfulness": None,
                "context_recall": None,
                "latency_sec": s["latency_sec"],
            }
            for s in samples
        ],
    }


# ── Phase 6: Qdrant + Hybrid + BGE ──────────────────────────────────────

def eval_phase6_v2(answerable: list[dict], negatives: list[dict]) -> dict:
    print("\n" + "="*60)
    print("Phase 6 v2: Hybrid + BGE Reranker (Qdrant, eval_qa_v2 기준)")
    print("="*60)

    from qdrant_rag import load_chunks, load_embed_model, build_bm25, hybrid_retrieve, generate
    from qdrant_client import QdrantClient

    chunks = load_chunks()
    model = load_embed_model()
    bm25, chunk_ids = build_bm25(chunks)
    qclient = QdrantClient(path=str(Path("./qdrant_data")))

    def rag_fn(question):
        retrieved = hybrid_retrieve(question, model, qclient, bm25, chunk_ids, chunks)
        result = generate(question, retrieved)
        result["relevant_docs"] = retrieved
        return result

    samples = []
    for i, qa in enumerate(answerable):
        print(f"  [answerable {i+1}/{len(answerable)}] {qa['question'][:50]}...")
        t0 = time.time()
        retrieved = hybrid_retrieve(qa["question"], model, qclient, bm25, chunk_ids, chunks)
        result = generate(qa["question"], retrieved)
        latency = round(time.time() - t0, 2)
        samples.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in retrieved],
            "ground_truth": qa["answer"],
            "qa_type": qa["qa_type"],
            "latency_sec": latency,
        })

    ragas_scores = run_ragas(samples, "Phase6_Qdrant_v2")

    print(f"\n  [negative 평가] {len(negatives)}개...")
    neg_result = eval_negative_accuracy(negatives, rag_fn)

    return {
        "phase": "Phase6_Qdrant_v2",
        "eval_dataset": "eval_qa_v2",
        **ragas_scores,
        "negative_eval": {
            "fallback_accuracy": neg_result["fallback_accuracy"],
            "correct": neg_result["correct"],
            "total": neg_result["total"],
        },
    }


PHASE_MAP = {
    "6": eval_phase6_v2,
    "7": eval_phase7_v2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="7", help="실행할 Phase 번호 (쉼표 구분, 예: 6,7)")
    parser.add_argument("--limit", type=int, default=None, help="answerable QA 최대 개수 (테스트용)")
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",")]
    answerable, negatives = load_qa_v2(limit=args.limit)

    all_scores = []
    t_total = time.time()

    for phase_num in phases:
        if phase_num not in PHASE_MAP:
            print(f"[경고] Phase {phase_num}은 지원하지 않습니다. (지원: {list(PHASE_MAP.keys())})")
            continue

        try:
            scores = PHASE_MAP[phase_num](answerable, negatives)
            all_scores.append(scores)

            # 중간 저장
            out_path = OUT_DIR / f"ragas_v2_{scores['phase']}.json"
            out_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n  ✓ 저장: {out_path}")
            print(f"    Faith={scores['faithfulness']} Recall={scores['context_recall']} "
                  f"Prec={scores['context_precision']} "
                  f"fallback={scores['negative_eval']['fallback_accuracy']:.1%}")

        except Exception as e:
            print(f"\n  [ERROR] Phase {phase_num} 실패: {e}")
            import traceback; traceback.print_exc()

    total_elapsed = round(time.time() - t_total, 1)

    if all_scores:
        print("\n" + "="*80)
        print(f"{'Phase':<30} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Prec':>7} {'Fallback':>9}")
        print("-"*80)
        for s in all_scores:
            def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
            fb = s.get("negative_eval", {}).get("fallback_accuracy")
            print(f"{s['phase']:<30} {fmt(s['faithfulness']):>7} {fmt(s['answer_relevancy']):>7} "
                  f"{fmt(s['context_recall']):>7} {fmt(s['context_precision']):>7} "
                  f"{f'{fb:.1%}' if fb is not None else 'N/A':>9}")
        print("="*80)
        print(f"\n총 소요: {total_elapsed:.0f}초")


if __name__ == "__main__":
    main()
