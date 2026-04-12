"""
AutoRAG 데이터 변환: qa_combined.json + chunks.json → corpus.parquet + qa.parquet
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

CHUNKS_PATH = Path("data/processed/chunks.json")
QA_PATH = Path("data/autorag/qa_combined.json")
OUT_DIR = Path("data/autorag")


def run():
    # 로드
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    qa_raw = json.loads(QA_PATH.read_text(encoding="utf-8"))
    print(f"청크: {len(chunks)}개, QA: {len(qa_raw)}개")

    # corpus.parquet
    corpus = [
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
    corpus_df = pd.DataFrame(corpus)
    corpus_df.to_parquet(OUT_DIR / "corpus.parquet", index=False)
    print(f"corpus.parquet 저장: {len(corpus_df)}행")

    # chunk_id → 존재 확인용 set
    valid_ids = {c["chunk_id"] for c in chunks}

    # qa.parquet
    qa_rows = []
    skipped = 0
    for item in qa_raw:
        chunk_id = item["id"]
        if chunk_id not in valid_ids:
            print(f"  [경고] chunk_id 없음: {chunk_id}")
            skipped += 1
            continue
        qa_rows.append({
            "qid": f"qa_{chunk_id}",
            "query": item["question"],
            "retrieval_gt": [[chunk_id]],       # 2D list
            "generation_gt": [item["answer"]],  # list
        })

    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_parquet(OUT_DIR / "qa.parquet", index=False)
    print(f"qa.parquet 저장: {len(qa_df)}행 (스킵: {skipped}개)")

    # 검증
    print("\n=== 검증 ===")
    c = pd.read_parquet(OUT_DIR / "corpus.parquet")
    q = pd.read_parquet(OUT_DIR / "qa.parquet")
    print(f"corpus 컬럼: {list(c.columns)}")
    print(f"qa 컬럼: {list(q.columns)}")
    print(f"qa 샘플:\n{q.iloc[0]}")


if __name__ == "__main__":
    run()
