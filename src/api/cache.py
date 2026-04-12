"""
Pre-compute 캐시 로드 및 조회.
data/demo_cache/manifest.json 기준으로 샘플 목록과 캐시 결과를 관리.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from api.schemas import QueryResponse

CACHE_DIR = Path("data/demo_cache")
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# 서버 기동 시 1회 로드
_manifest: dict = {}
_cache: dict[str, dict] = {}   # key: "{pipeline}/{qid}"


def load_cache() -> None:
    """서버 startup 시 호출. manifest + 모든 캐시 JSON 로드."""
    global _manifest, _cache

    if not MANIFEST_PATH.exists():
        print("[cache] manifest.json 없음 — precompute.py 를 먼저 실행하세요")
        _manifest = {"categories": []}
        return

    _manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    loaded = 0
    for category in _manifest.get("categories", []):
        for q in category.get("queries", []):
            qid = q["qid"]
            for pipeline in ("crag", "qdrant"):
                path = CACHE_DIR / pipeline / f"{qid}.json"
                if path.exists():
                    key = f"{pipeline}/{qid}"
                    _cache[key] = json.loads(path.read_text(encoding="utf-8"))
                    loaded += 1

    print(f"[cache] {loaded}개 캐시 엔트리 로드 완료")


def get_manifest() -> dict:
    return _manifest


def get_cached(qid: str, pipeline: str) -> Optional[QueryResponse]:
    key = f"{pipeline}/{qid}"
    data = _cache.get(key)
    if data is None:
        return None
    data["cached"] = True
    return QueryResponse(**data)


def cache_count() -> int:
    return len(_cache)
