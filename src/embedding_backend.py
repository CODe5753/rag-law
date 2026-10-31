"""
임베딩 백엔드 추상화 레이어.

EMBEDDING_BACKEND=ollama  → Ollama /api/embed (bge-m3:latest, F16)
EMBEDDING_BACKEND=local   → SentenceTransformer (기본값)
EMBEDDING_BACKEND=http    → explicit BGE-M3 HTTP service (no fallback)

환경변수:
  EMBEDDING_BACKEND       : "ollama" | "local" | "http" (default: "local")
  EMBEDDING_BASE_URL      : http 모드의 명시적 HTTP(S) origin
  EMBEDDING_TIMEOUT_SECONDS: http 배치 전체 deadline, 기본 30초/최대 120초
  OLLAMA_HOST             : Ollama 서버 주소 (default: http://localhost:11434)
  OLLAMA_EMBEDDING_MODEL  : Ollama 임베딩 모델명 (default: "bge-m3")
  EMBEDDING_MODEL         : local 모드에서 사용할 HuggingFace 모델명 (default: "BAAI/bge-m3")
"""

import os
import http.client
import json
import math
import time
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
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
    if EMBEDDING_BACKEND == "http":
        return _encode_http(texts)
    if EMBEDDING_BACKEND == "local":
        return _encode_local(texts)
    raise ValueError("Unsupported embedding backend")


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


def is_remote_backend() -> bool:
    """Remote encoders must never instantiate a local model in retrieval callers."""
    return EMBEDDING_BACKEND in {"ollama", "http"}


def _http_json(path, payload=None, timeout=None):
    # Explicit endpoint: no implicit Ollama fallback and no redirect/auth leakage.
    address = urlsplit(os.environ.get("EMBEDDING_BASE_URL", ""))
    if (address.scheme not in {"http", "https"} or not address.hostname
            or address.username or address.password or address.query or address.fragment
            or address.path not in {"", "/"}):
        raise ValueError("EMBEDDING_BASE_URL must be an explicit HTTP(S) origin")
    seconds = float(timeout if timeout is not None else os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("Embedding timeout must be within (0, 120] seconds")
    deadline = time.monotonic() + seconds
    response = None
    connection_type = http.client.HTTPSConnection if address.scheme == "https" else http.client.HTTPConnection
    conn = connection_type(address.hostname, address.port, timeout=seconds)
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Embedding request deadline exceeded")
        if conn.sock:
            conn.sock.settimeout(value)
        elif response is not None and response.fp is not None:
            response.fp.raw._sock.settimeout(value)
        return value
    try:
        body = None if payload is None else json.dumps(payload).encode()
        if body is not None and len(body) > 262144:
            raise ValueError("Embedding input is too large")
        conn.connect()
        remaining()
        conn.request("GET" if payload is None else "POST", path, body=body,
                     headers={"Content-Type": "application/json"})
        remaining()
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Embedding HTTP {response.status}")
        data = bytearray()
        while True:
            remaining()
            chunk = response.read1(min(65536, 262145 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > 262144:
                raise ValueError("Embedding response is too large")
        remaining()
        return json.loads(data)
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError("Embedding transport failed") from None
    finally:
        conn.close()


def _encode_http_inline(texts):
    if not isinstance(texts, list) or len(texts) > 128 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Expected at most 128 nonempty texts")
    seconds = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("Embedding timeout must be within (0, 120] seconds")
    deadline = time.monotonic() + seconds
    result = []
    # One text per bounded request preserves ingestion batches without exceeding
    # the service's total token budget or loading local transformer weights.
    for text in texts:
        data = _http_json("/api/embed", {"model": "bge-m3", "input": [text]}, timeout=deadline - time.monotonic())
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise ValueError("Embedding response count mismatch")
        vector = vectors[0]
        if not isinstance(vector, list) or len(vector) != 1024 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError("Embedding must contain 1024 finite numbers")
        if abs(math.sqrt(sum(v * v for v in vector)) - 1) > 0.01:
            raise ValueError("Embedding must be unit normalized")
        result.append(vector)
    return result


def http_backend_ready():
    try:
        data = _run_isolated("health", timeout=2)
        return isinstance(data, dict) and data.get("status") == "ready" and data.get("dimension") == 1024 and data.get("normalize") is True and data.get("pooling") == "cls"
    except Exception:
        return False


def _run_isolated(operation, texts=None, timeout=None):
    """Bound DNS, TLS, reads, all sequential batch requests and child cleanup."""
    seconds = float(timeout if timeout is not None else os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("Embedding timeout must be within (0, 120] seconds")
    request = json.dumps({"operation": operation, "texts": texts, "timeout": seconds}).encode()
    if len(request) > 8 * 1024 * 1024:
        raise ValueError("Embedding batch is too large")
    try:
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve())],
            input=request, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits: no DNS/HTTP worker survives a timeout.
        raise TimeoutError("Embedding batch deadline exceeded") from None
    except OSError:
        raise RuntimeError("Embedding worker failed") from None
    if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
        raise RuntimeError("Embedding worker failed")
    try:
        response = json.loads(result.stdout)
    except (ValueError, UnicodeError):
        raise RuntimeError("Embedding worker failed") from None
    if response.get("error") == "invalid":
        raise ValueError("Embedding input or response is invalid")
    if response.get("error"):
        raise RuntimeError("Embedding request failed")
    return response["result"]


def _encode_http(texts):
    if not isinstance(texts, list) or len(texts) > 128 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Expected at most 128 nonempty texts")
    if not texts:
        return []
    return _run_isolated("batch", texts)


def _embedding_worker_main():
    try:
        request = json.load(sys.stdin)
        os.environ["EMBEDDING_TIMEOUT_SECONDS"] = str(request["timeout"])
        if request["operation"] == "health":
            result = _http_json("/health", timeout=request["timeout"])
        elif request["operation"] == "batch":
            result = _encode_http_inline(request["texts"])
        else:
            raise ValueError("Invalid worker operation")
        output = {"result": result}
    except ValueError:
        output = {"error": "invalid"}
    except Exception:
        output = {"error": "transport"}
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    _embedding_worker_main()
