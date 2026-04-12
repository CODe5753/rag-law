"""
AutoRAG용 ChromaDB 사전 인제스트

AutoRAG가 7665청크를 한 번에 ChromaDB.add()하면 max_batch_size(5461) 초과 에러.
이 스크립트로 먼저 청크를 배치 단위로 인제스트 → AutoRAG는 기존 id 감지 후 스킵.

ChromaDB 기본 max_batch 설정:
  https://docs.trychroma.com/guides#changing-the-default-maximum-batch-size
  기본 5461 → 이 스크립트는 BATCH=5000으로 안전하게 나눠서 추가
"""

import json
from pathlib import Path
import numpy as np

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = Path("data/autorag/chroma_store")
COLLECTION_NAME = "autorag_law"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_BATCH = 16       # SentenceTransformer 배치 (VRAM/RAM 고려)
CHROMA_BATCH = 5000    # ChromaDB add() 배치 (max 5461)


def run():
    from sentence_transformers import SentenceTransformer
    import chromadb

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"[로드] 청크 {len(chunks)}개")

    # 1. ChromaDB 컬렉션 생성/로드
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col = client.get_collection(COLLECTION_NAME)
        existing_count = col.count()
        print(f"[기존 컬렉션] {COLLECTION_NAME}: {existing_count}개 문서 존재")
        if existing_count >= len(chunks):
            print("  이미 완전히 인제스트됨. 스킵.")
            return
    except Exception:
        col = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        existing_count = 0
        print(f"[새 컬렉션 생성] {COLLECTION_NAME}")

    # 이미 있는 ID 확인
    if existing_count > 0:
        existing_ids = set(col.get(include=[])["ids"])
    else:
        existing_ids = set()

    # 2. 추가할 청크 필터링
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"[인제스트 대상] {len(new_chunks)}개 (기존 {existing_count}개 스킵)")

    if not new_chunks:
        print("추가할 청크 없음.")
        return

    # 3. bge-m3 임베딩
    print(f"\n[임베딩] {EMBED_MODEL} 로드 중...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in new_chunks]
    ids = [c["chunk_id"] for c in new_chunks]

    print(f"[임베딩] {len(texts)}개 청크 인코딩 중 (배치={EMBED_BATCH})...")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  임베딩 완료: shape={embeddings.shape}")

    # 4. ChromaDB 배치 추가
    print(f"\n[ChromaDB 추가] 배치 크기={CHROMA_BATCH}")
    total = len(new_chunks)
    for start in range(0, total, CHROMA_BATCH):
        end = min(start + CHROMA_BATCH, total)
        batch_ids = ids[start:end]
        batch_embeddings = embeddings[start:end].tolist()
        batch_docs = texts[start:end]
        batch_meta = [
            {"law_name": c.get("law_name", ""), "article_num": c.get("article_num", "")}
            for c in new_chunks[start:end]
        ]
        col.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  추가: {end}/{total}")

    final_count = col.count()
    print(f"\n완료! 컬렉션 총 {final_count}개 문서")
    print(f"경로: {CHROMA_PATH}")
    print(f"\n다음: autorag evaluate 실행 (기존 id 감지로 인제스트 스킵됨)")


if __name__ == "__main__":
    run()
