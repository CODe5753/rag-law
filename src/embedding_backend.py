"""
임베딩 백엔드 추상화 레이어.

EMBEDDING_BACKEND=ollama  → Ollama /api/embed (bge-m3:latest, F16)
EMBEDDING_BACKEND=local   → SentenceTransformer (기본값, fallback)

환경변수:
  EMBEDDING_BACKEND       : "ollama" | "local" (default: "local")
  OLLAMA_HOST             : Ollama 서버 주소 (default: http://localhost:11434)
  OLLAMA_EMBEDDING_MODEL  : Ollama 임베딩 모델명 (default: "bge-m3")
  EMBEDDING_MODEL         : local 모드에서 사용할 HuggingFace 모델명 (default: "BAAI/bge-m3")
"""

import os
import requests

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

_local_model = None


def encode_query(text: str) -> list[float]:
    """단일 텍스트를 벡터로 변환. 1024차원 정규화 벡터 반환."""
    return encode_batch([text])[0]


def encode_batch(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 벡터 리스트로 변환."""
    if EMBEDDING_BACKEND == "ollama":
        return _encode_ollama(texts)
    return _encode_local(texts)


def _encode_ollama(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": OLLAMA_EMBEDDING_MODEL, "input": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _encode_local(texts: list[str]) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        print(f"[임베딩] 로컬 모델 로드: {model_name}")
        _local_model = SentenceTransformer(model_name)
    return _local_model.encode(texts, normalize_embeddings=True).tolist()


def is_ollama_backend() -> bool:
    return EMBEDDING_BACKEND == "ollama"
