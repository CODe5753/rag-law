"""
Phase 6: ChromaDB → Qdrant 마이그레이션

임베딩 재계산 없이 이전:
  1. ChromaDB에서 벡터 + 메타데이터 추출
  2. Qdrant 컬렉션 생성 (cosine, 1024차원 — bge-m3와 동일)
  3. 배치 업서트
  4. 동일 쿼리로 검색 결과 비교 (회귀 검증)

Qdrant 선택 근거:
  - ChromaDB의 필터링 hang (Issue #4089) 문제 없음
  - 법령명/조문 번호 payload 필터 지원 (예: law_name="개인정보보호법"만 검색)
  - 대용량에서 ChromaDB보다 안정적인 ANN 검색 속도

운영 방식: embedded (파일 기반, Docker 불필요)
  QdrantClient(path="./qdrant_data")
"""

import json
import os
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import chromadb

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_data"))
QDRANT_PATH = Path("./qdrant_data")
COLLECTION_NAME = "law_chunks_v1"
VECTOR_DIM = 1024   # bge-m3
BATCH_SIZE = 256


def migrate_chroma_to_qdrant() -> dict:
    """ChromaDB → Qdrant 마이그레이션 (임베딩 재계산 없음)"""

    # 1. ChromaDB에서 전체 데이터 추출
    print("[1] ChromaDB 데이터 추출...")
    t0 = time.time()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = chroma_client.get_collection(COLLECTION_NAME)
    total = col.count()
    print(f"  총 {total:,}개 청크")

    # 전체 데이터 한 번에 읽기
    raw = col.get(
        limit=total,
        include=["embeddings", "documents", "metadatas"],
    )
    elapsed_extract = round(time.time() - t0, 1)
    print(f"  추출 완료: {elapsed_extract}s")

    ids = raw["ids"]
    embeddings = raw["embeddings"]
    documents = raw["documents"]
    metadatas = raw["metadatas"]

    # 2. Qdrant 컬렉션 생성
    print("\n[2] Qdrant 컬렉션 생성...")
    qdrant = QdrantClient(path=str(QDRANT_PATH))

    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  기존 컬렉션 삭제: {COLLECTION_NAME}")
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"  생성 완료: {COLLECTION_NAME} (cosine, dim={VECTOR_DIM})")

    # 3. 배치 업서트
    print(f"\n[3] 업서트 시작 (배치={BATCH_SIZE})...")
    t0 = time.time()

    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i: i + BATCH_SIZE]
        batch_vecs = embeddings[i: i + BATCH_SIZE]
        batch_docs = documents[i: i + BATCH_SIZE]
        batch_metas = metadatas[i: i + BATCH_SIZE]

        points = [
            PointStruct(
                id=j + i,   # Qdrant는 int 또는 uuid ID
                vector=vec,
                payload={
                    "chunk_id": chunk_id,
                    "text": doc,
                    **meta,
                },
            )
            for j, (chunk_id, vec, doc, meta) in enumerate(
                zip(batch_ids, batch_vecs, batch_docs, batch_metas)
            )
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        done = min(i + BATCH_SIZE, len(ids))
        print(f"  {done:,}/{len(ids):,} ({done/len(ids)*100:.0f}%)")

    elapsed_upsert = round(time.time() - t0, 1)
    print(f"  완료: {elapsed_upsert}s")

    # 4. 검증
    count = qdrant.count(COLLECTION_NAME).count
    print(f"\n[4] Qdrant 검증: {count:,}개 포인트 확인")

    return {
        "total_points": count,
        "extract_sec": elapsed_extract,
        "upsert_sec": elapsed_upsert,
    }


def search_qdrant(query_vec: list[float], top_k: int = 5, law_name: str = None) -> list[dict]:
    """
    Qdrant 검색. law_name 필터 지원.
    ChromaDB에 없던 payload 필터링: 특정 법령만 검색 가능.
    """
    qdrant = QdrantClient(path=str(QDRANT_PATH))

    query_filter = None
    if law_name:
        query_filter = Filter(
            must=[FieldCondition(key="law_name", match=MatchValue(value=law_name))]
        )

    # ⚠️ qdrant-client 1.x: search() 제거됨 → query_points() 사용
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [
        {
            "text": r.payload["text"],
            "law_name": r.payload.get("law_name", ""),
            "article_num": r.payload.get("article_num", ""),
            "score": round(r.score, 4),
        }
        for r in results
    ]


def regression_check(model, top_k: int = 5) -> dict:
    """ChromaDB vs Qdrant 검색 결과 일치 여부 확인"""
    from sentence_transformers import SentenceTransformer
    test_queries = [
        "개인정보 수집 동의",
        "근로기준법 근로시간",
        "저작권 보호 기간",
    ]

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    chroma_col = chroma_client.get_collection(COLLECTION_NAME)

    match_count = 0
    total_queries = len(test_queries)

    print("\n[회귀 검증] ChromaDB vs Qdrant 검색 결과 비교")
    for q in test_queries:
        vec = model.encode([q], normalize_embeddings=True).tolist()[0]

        # ChromaDB
        chroma_res = chroma_col.query(query_embeddings=[vec], n_results=top_k)
        chroma_ids = set(chroma_res["ids"][0])

        # Qdrant
        qdrant_res = search_qdrant(vec, top_k=top_k)
        qdrant_ids = {r["text"][:30] for r in qdrant_res}

        # ChromaDB docs로 비교
        chroma_docs = {d[:30] for d in chroma_res["documents"][0]}
        overlap = len(chroma_docs & qdrant_ids)
        match_rate = overlap / top_k

        print(f"  '{q}': 일치율 {overlap}/{top_k} ({match_rate*100:.0f}%)")
        if match_rate >= 0.8:
            match_count += 1

    return {"match_queries": match_count, "total_queries": total_queries}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate", action="store_true", help="ChromaDB → Qdrant 마이그레이션")
    parser.add_argument("--check", action="store_true", help="회귀 검증만 실행")
    parser.add_argument("--filter-demo", help="law_name 필터 데모 쿼리")
    args = parser.parse_args()

    if args.migrate:
        stats = migrate_chroma_to_qdrant()
        print(f"\n=== 마이그레이션 완료 ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # 자동으로 회귀 검증
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        reg = regression_check(model)
        print(f"\n회귀 검증: {reg['match_queries']}/{reg['total_queries']} 쿼리 일치")

    elif args.check:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        regression_check(model)

    elif args.filter_demo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        vec = model.encode([args.filter_demo], normalize_embeddings=True).tolist()[0]

        print(f"\n[전체 검색] '{args.filter_demo}'")
        for r in search_qdrant(vec, top_k=5):
            print(f"  {r['law_name']} {r['article_num']} ({r['score']})")

        print(f"\n[필터: 개인정보보호법만] '{args.filter_demo}'")
        for r in search_qdrant(vec, top_k=5, law_name="개인정보보호법"):
            print(f"  {r['law_name']} {r['article_num']} ({r['score']})")
    else:
        print("사용법: --migrate | --check | --filter-demo '쿼리'")
