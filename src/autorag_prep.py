"""
AutoRAG 사전 준비: 합성 QA 생성 + 데이터 변환

AutoRAG 요구 형식:
  corpus.parquet  — doc_id, contents, metadata (last_modified_datetime 필수)
  qa.parquet      — qid, query, retrieval_gt (2D list), generation_gt

합성 QA 생성:
  - 청크를 샘플링 (100~200개)
  - EXAONE에게 각 청크에서 질문 + 정답 생성 요청
  - retrieval_gt: 해당 청크의 doc_id를 ground-truth로 설정

AutoRAG는 최소 50~100개 QA 권장. 6개로는 분산이 너무 커서 신뢰 불가.
"""

import json
import os
import re
import random
import time
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
CHUNKS_PATH = Path("data/processed/chunks.json")
OUT_DIR = Path("data/autorag")

# 샘플링 파라미터
SAMPLE_SIZE = 120        # 청크 중 이 개수를 샘플링
MIN_CHUNK_LEN = 80       # 너무 짧은 청크 제외 (질문 생성 어려움)
TARGET_QA = 80           # 목표 QA 쌍 수 (생성 실패 고려해 여유 있게 샘플링)

random.seed(42)


def ollama_call(prompt: str, timeout: int = 60) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return ""


def generate_qa_from_chunk(chunk: dict) -> dict | None:
    """청크에서 질문-정답 쌍 생성"""
    prompt = f"""다음 법령 조문에서 한국어 질문 하나와 해당 조문에 근거한 정답을 생성하세요.

[법령 조문]
{chunk['text']}

다음 형식으로만 출력하세요:
질문: (구체적인 법령 관련 질문)
정답: (조문에 근거한 정확한 답변, 1~3문장)"""

    response = ollama_call(prompt)
    if not response:
        return None

    # 파싱
    q_match = re.search(r"질문[:\s]+(.+?)(?:\n|정답)", response, re.DOTALL)
    a_match = re.search(r"정답[:\s]+(.+?)$", response, re.DOTALL)

    if not q_match or not a_match:
        return None

    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()

    # 품질 필터
    if len(question) < 10 or len(answer) < 10:
        return None
    if question == answer:
        return None

    return {
        "qid": f"qa_{chunk['chunk_id']}",
        "query": question,
        "retrieval_gt": [[chunk["chunk_id"]]],  # 2D list (AutoRAG 형식)
        "generation_gt": [answer],               # list (AutoRAG 형식)
        "source_chunk": chunk["chunk_id"],
        "law_name": chunk.get("law_name", ""),
    }


def prepare_corpus(chunks: list[dict]) -> list[dict]:
    """AutoRAG corpus 형식으로 변환"""
    return [
        {
            "doc_id": c["chunk_id"],
            "contents": c["text"],
            "metadata": {
                "law_name": c.get("law_name", ""),
                "article_num": c.get("article_num", ""),
                "last_modified_datetime": datetime(2026, 1, 1),
            },
        }
        for c in chunks
    ]


def generate_synthetic_qa(chunks: list[dict], sample_size: int = SAMPLE_SIZE) -> list[dict]:
    """청크에서 합성 QA 생성"""
    # 충분히 긴 청크만 샘플링
    candidates = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_LEN]
    sampled = random.sample(candidates, min(sample_size, len(candidates)))

    qa_pairs = []
    failed = 0

    print(f"[QA 생성] {len(sampled)}개 청크에서 합성 QA 생성 중...")
    for i, chunk in enumerate(sampled):
        if i % 10 == 0:
            print(f"  {i}/{len(sampled)} ({len(qa_pairs)}개 생성, {failed}개 실패)")

        qa = generate_qa_from_chunk(chunk)
        if qa:
            qa_pairs.append(qa)
        else:
            failed += 1

        if len(qa_pairs) >= TARGET_QA:
            print(f"  목표 달성: {TARGET_QA}개")
            break

    print(f"\n완료: {len(qa_pairs)}개 QA 생성 ({failed}개 실패)")
    return qa_pairs


def save_parquet(data: list[dict], path: Path):
    """parquet 저장 (pandas 필요)"""
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        print(f"  저장: {path} ({len(df)}행)")
    except ImportError:
        # pandas 없으면 JSON으로 저장
        path_json = path.with_suffix(".json")
        path_json.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        print(f"  저장 (JSON 대체): {path_json} ({len(data)}행)")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"[로드] 청크 {len(chunks)}개")

    # 1. corpus 변환
    print("\n[1] Corpus 변환...")
    corpus = prepare_corpus(chunks)
    save_parquet(corpus, OUT_DIR / "corpus.parquet")

    # 2. 합성 QA 생성
    print("\n[2] 합성 QA 생성 (EXAONE)...")
    t0 = time.time()
    qa_pairs = generate_synthetic_qa(chunks)
    elapsed = round(time.time() - t0, 1)

    # QA JSON 저장 (검토용)
    qa_review = OUT_DIR / "qa_review.json"
    qa_review.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  검토용 저장: {qa_review}")

    # AutoRAG 형식 변환
    qa_autorag = [
        {
            "qid": q["qid"],
            "query": q["query"],
            "retrieval_gt": q["retrieval_gt"],
            "generation_gt": q["generation_gt"],
        }
        for q in qa_pairs
    ]
    save_parquet(qa_autorag, OUT_DIR / "qa.parquet")

    print(f"\n=== 완료 ===")
    print(f"  corpus: {len(corpus)}개 청크")
    print(f"  QA: {len(qa_pairs)}개")
    print(f"  생성 시간: {elapsed}s ({round(elapsed/len(qa_pairs), 1)}s/QA)")
    print(f"\n  다음: python src/autorag_run.py --config src/autorag_config.yaml")

    return {"corpus_count": len(corpus), "qa_count": len(qa_pairs), "elapsed_sec": elapsed}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE, help="샘플링 청크 수")
    parser.add_argument("--target", type=int, default=TARGET_QA, help="목표 QA 수")
    args = parser.parse_args()

    if args.sample != SAMPLE_SIZE:
        SAMPLE_SIZE = args.sample
    if args.target != TARGET_QA:
        TARGET_QA = args.target

    run()
