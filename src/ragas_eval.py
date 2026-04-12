"""
Phase 3: RAGAS 평가 — Naive RAG 기준선 측정

메트릭:
  - Faithfulness: 답변이 검색된 컨텍스트에만 근거하는가 (환각 감지)
  - Answer Relevancy: 답변이 질문에 얼마나 관련 있는가
  - Context Recall: 정답에 필요한 정보가 검색되었는가
  - Context Precision: 검색된 컨텍스트 중 유용한 비율

⚠️ 트러블슈팅 히스토리:
  - RAGAS 0.4.x는 0.1.x와 API 완전 다름. SingleTurnSample + EvaluationDataset 사용.
  - LLM-as-Judge로 Ollama EXAONE 사용 → 한국어 법령 평가에 적합.
  - OllamaEmbeddings는 Answer Relevancy 계산에만 사용.
"""

import json
import time
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

import chromadb
from sentence_transformers import SentenceTransformer
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_data"))
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "law_chunks_v1"
OUT_DIR = Path("data/processed")

# ── 평가 데이터셋 (수동 작성, ground_truth 포함) ─────────────────
# 실제 법령 조문에서 확인된 정답만 사용
EVAL_QA = [
    {
        "question": "개인정보 수집 시 정보주체에게 알려야 할 사항은 무엇인가?",
        "ground_truth": (
            "개인정보보호법 제15조에 따라 개인정보를 수집할 때는 "
            "① 수집·이용 목적, ② 수집하려는 개인정보 항목, "
            "③ 보유·이용 기간, ④ 동의 거부 권리 및 불이익을 알려야 한다."
        ),
    },
    {
        "question": "근로기준법상 성인 근로자의 1주 법정 근로시간은?",
        "ground_truth": (
            "근로기준법 제50조에 따라 1주간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다. "
            "당사자 합의 시 1주 12시간 한도로 연장 가능하여 최대 52시간이다."
        ),
    },
    {
        "question": "저작자 사망 후 저작권은 몇 년간 보호되는가?",
        "ground_truth": (
            "저작권법 제39조에 따라 저작재산권은 저작자가 생존하는 동안과 "
            "사망한 후 70년간 존속한다."
        ),
    },
    {
        "question": "공정거래법상 시장지배적 사업자로 추정되는 시장점유율 기준은?",
        "ground_truth": (
            "공정거래법 제6조에 따라 1개 사업자의 시장점유율이 50% 이상이거나, "
            "3개 이하 사업자의 합산 점유율이 75% 이상인 경우 시장지배적 사업자로 추정된다. "
            "단, 연간 매출액 또는 구매액이 80억원 미만인 사업자는 제외된다."
        ),
    },
    {
        "question": "전자금융거래법상 접근매체의 종류는 무엇인가?",
        "ground_truth": (
            "전자금융거래법 제2조에 따라 접근매체는 전자식 카드, 전자서명생성정보, "
            "인증서, 이용자번호, 생체정보, 비밀번호 등 전자금융거래에 사용되는 수단을 말한다."
        ),
    },
    {
        "question": "신용정보법상 개인신용정보 제공 시 동의가 불필요한 경우는?",
        "ground_truth": (
            "신용정보법 제32조에 따라 법률에 특별한 규정이 있거나 법령상 의무 준수, "
            "채권추심 등 정당한 이익이 있는 경우, 통계작성·학술연구 목적으로 "
            "특정 개인을 식별할 수 없는 형태로 제공하는 경우 등은 동의 없이 제공 가능하다."
        ),
    },
    {
        "question": "금융소비자보호법상 6대 판매원칙은 무엇인가?",
        "ground_truth": (
            "금융소비자보호법은 적합성 원칙, 적정성 원칙, 설명의무, 불공정영업행위 금지, "
            "부당권유행위 금지, 광고규제를 6대 판매행위 규제로 규정한다."
        ),
    },
    {
        "question": "저작권법상 업무상저작물의 보호기간은?",
        "ground_truth": (
            "저작권법 제41조에 따라 업무상저작물의 저작재산권은 공표한 때부터 70년간 존속한다. "
            "창작한 때부터 50년 이내에 공표되지 않은 경우에는 창작한 때부터 70년간 존속한다."
        ),
    },
]


def load_rag_pipeline():
    """기존 naive_rag 파이프라인 재사용"""
    from naive_rag import load_embed_model, build_index, retrieve, generate
    model = load_embed_model()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(COLLECTION_NAME)
    return model, col, retrieve, generate


def build_ragas_dataset(model, col, retrieve, generate) -> EvaluationDataset:
    """QA 데이터셋 + RAG 결과로 RAGAS EvaluationDataset 구성"""
    samples = []
    print(f"\n[데이터셋 구성] {len(EVAL_QA)}개 쿼리 처리 중...\n")

    for i, qa in enumerate(EVAL_QA):
        q = qa["question"]
        gt = qa["ground_truth"]

        # RAG 실행
        retrieved = retrieve(q, model, col, top_k=5)
        result = generate(q, retrieved)

        contexts = [c["text"] for c in retrieved]
        answer = result["answer"]

        print(f"  [{i+1}/{len(EVAL_QA)}] {q[:40]}...")
        print(f"    유사도: {[c['score'] for c in retrieved]}")

        samples.append(SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts,
            reference=gt,
        ))

    return EvaluationDataset(samples=samples)


def run_ragas(dataset: EvaluationDataset) -> dict:
    """RAGAS 평가 실행 (LLM-as-Judge: EXAONE via Ollama)"""
    print("\n[RAGAS 평가 시작]")
    print(f"  Judge LLM: {OLLAMA_MODEL} (Ollama)")

    llm = LangchainLLMWrapper(ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
    ))

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
    ]

    # ⚠️ 트러블슈팅: Ollama는 단일 스레드 처리. max_workers=1로 순차 실행 필수.
    # 병렬 실행 시 TimeoutError 다발. timeout=300초로 충분한 여유 확보.
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=1)

    t0 = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    elapsed = round(time.time() - t0, 1)

    # RAGAS 0.4.x: result는 EvaluationResult 객체. pandas로 변환 후 평균 계산.
    df = result.to_pandas()
    print("\n[샘플별 점수]")
    print(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].to_string())

    def safe_mean(col):
        """NaN 제외 평균. 전부 NaN이면 None 반환."""
        vals = df[col].dropna()
        return round(float(vals.mean()), 4) if len(vals) > 0 else None

    scores = {
        "faithfulness": safe_mean("faithfulness"),
        "answer_relevancy": safe_mean("answer_relevancy"),
        "context_recall": safe_mean("context_recall"),
        "context_precision": safe_mean("context_precision"),
        "eval_time_sec": elapsed,
        "phase": "Phase2_NaiveRAG",
        "llm": OLLAMA_MODEL,
        "n_samples": len(EVAL_QA),
        "nan_count": int(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].isna().sum().sum()),
    }
    return scores


def print_scores(scores: dict):
    print("\n" + "=" * 50)
    print("RAGAS 기준선 점수 (Phase 2 — Naive RAG)")
    print("=" * 50)
    print(f"  Faithfulness      (환각 억제): {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy  (답변 관련성): {scores['answer_relevancy']:.4f}")
    print(f"  Context Recall    (컨텍스트 완전성): {scores['context_recall']:.4f}")
    print(f"  Context Precision (컨텍스트 정밀도): {scores['context_precision']:.4f}")
    print(f"\n  평가 시간: {scores['eval_time_sec']}s")
    print(f"  샘플 수: {scores['n_samples']}개")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # 1. RAG 파이프라인 로드
    model, col, retrieve, generate = load_rag_pipeline()

    # 2. RAGAS 데이터셋 구성
    dataset = build_ragas_dataset(model, col, retrieve, generate)

    # 3. 평가 실행
    scores = run_ragas(dataset)

    # 4. 출력
    print_scores(scores)

    # 5. 결과 저장
    out_path = OUT_DIR / "ragas_baseline.json"
    out_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")
