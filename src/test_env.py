"""환경 검증 스크립트 — Phase 0 완료 체크"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_ollama():
    from ai_gateway import generate
    try:
        answer=generate('answer','한 문장으로만 답하세요: 대한민국의 수도는?')
        print(f"[OK] Gateway LLM: {answer[:80]}")
        return True
    except Exception as e:
        print(f"[FAIL] Gateway LLM: {type(e).__name__}")
        return False

def test_embedding():
    from gateway_compat import GatewayEmbeddingModel
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    try:
        model = GatewayEmbeddingModel()
        vec = model.encode("대한민국 법령 테스트 문장입니다.")
        print(f"[OK] Embedding ({model_name}): dim={len(vec)}")
        return True
    except Exception as e:
        print(f"[FAIL] Embedding: {e}")
        return False

def test_chromadb():
    import chromadb
    from gateway_compat import GatewayEmbeddingModel
    try:
        model = GatewayEmbeddingModel()
        client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_data"))
        col = client.get_or_create_collection("test")
        doc = "대한민국 법령 테스트 문서입니다."
        emb = model.encode([doc]).tolist()
        col.add(documents=[doc], embeddings=emb, ids=["test-1"])
        query_emb = model.encode(["법령 테스트"]).tolist()
        result = col.query(query_embeddings=query_emb, n_results=1)
        client.delete_collection("test")
        print(f"[OK] ChromaDB (embedded, bge-m3): {result['documents'][0][0][:40]}")
        return True
    except Exception as e:
        print(f"[FAIL] ChromaDB: {e}")
        return False

if __name__ == "__main__":
    print("=== Phase 0 환경 검증 ===\n")
    results = [test_ollama(), test_embedding(), test_chromadb()]
    print(f"\n{'[PASS] 모든 검증 통과' if all(results) else '[FAIL] 일부 실패'}")
    sys.exit(0 if all(results) else 1)
