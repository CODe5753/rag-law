"""
Phase 3: 직접 구현 LLM-as-Judge 평가기

⚠️ RAGAS 0.4.x + Ollama 조합 실패 (2026-04-10 트러블슈팅):
  증상: 32회 LLM 호출 중 19분 후 stuck → 프로세스 강제 종료
  원인 1: RAGAS 내부 async 실행이 Ollama 단일 스레드와 충돌
  원인 2: RAGAS 프롬프트가 JSON 형식 응답 요구 → EXAONE이 불안정하게 반환
  원인 3: RAGAS 0.4.x에서 LangchainLLMWrapper deprecated, 내부 동작 불안정
  해결: 직접 LLM-as-Judge 구현. 메트릭당 단일 Ollama 호출, 점수 직접 파싱.

평가 메트릭:
  - Faithfulness: 답변이 컨텍스트에만 근거하는가 (환각 감지)
  - Answer Relevancy: 답변이 질문과 관련 있는가
  - Context Recall: 정답 정보가 컨텍스트에 존재하는가
  - Context Precision: 검색된 컨텍스트 중 유용한 비율
"""

import json
import re
import time
import sys
from pathlib import Path

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import os

sys.path.insert(0, str(Path(__file__).parent))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_data"))
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "law_chunks_v1"
OUT_DIR = Path("data/processed")

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


def ollama_call(prompt: str, timeout: int = 60) -> str:
    """단순 Ollama 호출. 실패 시 빈 문자열 반환."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"    [WARN] Ollama 호출 실패: {e}")
        return ""


def extract_score(text: str) -> float | None:
    """응답 텍스트에서 0.0~1.0 점수 추출."""
    # "점수: 0.8" 또는 "Score: 0.75" 형태
    m = re.search(r"(?:점수|score)[:\s]+([01]\.\d+|\d)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return min(max(val, 0.0), 1.0)
    # 숫자만 있는 경우
    nums = re.findall(r"\b([01]\.\d+)\b", text)
    if nums:
        return float(nums[0])
    return None


def eval_faithfulness(answer: str, contexts: list[str]) -> float | None:
    """답변이 컨텍스트에만 근거하는가 (0=완전 환각, 1=완전 근거)"""
    ctx = "\n\n".join(f"[컨텍스트 {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = f"""다음 법령 조문(컨텍스트)과 답변을 보고, 답변이 컨텍스트에만 근거하는지 평가하세요.
컨텍스트에 없는 내용을 답변이 만들어냈다면 점수가 낮습니다.

{ctx}

[답변]
{answer}

0.0(완전 환각)에서 1.0(완전 근거) 사이의 점수를 "점수: X.X" 형식으로만 출력하세요."""
    response = ollama_call(prompt)
    return extract_score(response)


def eval_answer_relevancy(question: str, answer: str) -> float | None:
    """답변이 질문에 얼마나 관련 있는가 (0=무관, 1=완전 관련)"""
    prompt = f"""다음 질문과 답변을 보고, 답변이 질문에 얼마나 잘 대답하는지 평가하세요.

[질문]
{question}

[답변]
{answer}

0.0(완전 무관)에서 1.0(완전히 대답)하는 점수를 "점수: X.X" 형식으로만 출력하세요."""
    response = ollama_call(prompt)
    return extract_score(response)


def eval_context_recall(ground_truth: str, contexts: list[str]) -> float | None:
    """정답에 필요한 정보가 컨텍스트에 있는가 (0=전혀 없음, 1=완전 포함)"""
    ctx = "\n\n".join(f"[컨텍스트 {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = f"""다음 정답과 검색된 컨텍스트를 비교하세요. 정답을 도출하는 데 필요한 정보가 컨텍스트에 얼마나 포함되어 있는지 평가하세요.

[정답]
{ground_truth}

{ctx}

0.0(필요 정보 전혀 없음)에서 1.0(모든 정보 포함) 사이의 점수를 "점수: X.X" 형식으로만 출력하세요."""
    response = ollama_call(prompt)
    return extract_score(response)


def eval_context_precision(question: str, contexts: list[str]) -> float | None:
    """검색된 컨텍스트 중 질문과 실제 관련 있는 비율"""
    relevant = 0
    for ctx in contexts:
        prompt = f"""다음 질문과 컨텍스트를 보고, 이 컨텍스트가 질문 답변에 도움이 되는지 평가하세요.

[질문]
{question}

[컨텍스트]
{ctx}

0.0(전혀 무관)에서 1.0(매우 관련)의 점수를 "점수: X.X" 형식으로만 출력하세요."""
        response = ollama_call(prompt)
        score = extract_score(response)
        if score is not None:
            relevant += score
    return round(relevant / len(contexts), 4) if contexts else None


def run_evaluation():
    from naive_rag import load_embed_model, build_index, retrieve, generate

    print("=== Phase 3: LLM-as-Judge 평가 시작 ===\n")
    model = load_embed_model()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(COLLECTION_NAME)

    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
    }

    sample_results = []
    t_total = time.time()

    for i, qa in enumerate(EVAL_QA):
        q = qa["question"]
        gt = qa["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_QA)}] {q[:50]}...")

        # RAG 실행
        t0 = time.time()
        retrieved = retrieve(q, model, col, top_k=5)
        gen_result = generate(q, retrieved)
        answer = gen_result["answer"]
        contexts = [c["text"] for c in retrieved]
        rag_latency = round(time.time() - t0, 1)

        # 평가
        faith = eval_faithfulness(answer, contexts)
        relev = eval_answer_relevancy(q, answer)
        recall = eval_context_recall(gt, contexts)
        prec = eval_context_precision(q, contexts)

        print(f"  검색 유사도: {[c['score'] for c in retrieved]}")
        print(f"  faithfulness={faith} | relevancy={relev} | recall={recall} | precision={prec}")
        print(f"  RAG 지연: {rag_latency}s")

        if faith is not None: all_scores["faithfulness"].append(faith)
        if relev is not None: all_scores["answer_relevancy"].append(relev)
        if recall is not None: all_scores["context_recall"].append(recall)
        if prec is not None: all_scores["context_precision"].append(prec)

        sample_results.append({
            "question": q,
            "answer": answer[:300],
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_recall": recall,
            "context_precision": prec,
            "rag_latency_sec": rag_latency,
            "top_sources": [f"{c['law_name']} {c['article_num']} ({c['score']})" for c in retrieved],
        })

    elapsed = round(time.time() - t_total, 1)

    def mean(lst): return round(sum(lst) / len(lst), 4) if lst else None

    final_scores = {
        "phase": "Phase2_NaiveRAG",
        "llm_judge": OLLAMA_MODEL,
        "n_samples": len(EVAL_QA),
        "eval_time_sec": elapsed,
        "faithfulness": mean(all_scores["faithfulness"]),
        "answer_relevancy": mean(all_scores["answer_relevancy"]),
        "context_recall": mean(all_scores["context_recall"]),
        "context_precision": mean(all_scores["context_precision"]),
        "samples": sample_results,
    }

    print(f"\n{'='*55}")
    print("RAGAS 기준선 점수 (Phase 2 — Naive RAG, LLM-as-Judge)")
    print(f"{'='*55}")
    print(f"  Faithfulness      (환각 억제): {final_scores['faithfulness']}")
    print(f"  Answer Relevancy  (답변 관련성): {final_scores['answer_relevancy']}")
    print(f"  Context Recall    (컨텍스트 완전성): {final_scores['context_recall']}")
    print(f"  Context Precision (컨텍스트 정밀도): {final_scores['context_precision']}")
    print(f"\n  총 평가 시간: {elapsed}s / 샘플: {len(EVAL_QA)}개")
    print(f"{'='*55}")

    out = OUT_DIR / "ragas_baseline.json"
    out.write_text(json.dumps(final_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")

    return final_scores


if __name__ == "__main__":
    run_evaluation()
