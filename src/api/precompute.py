"""
샘플 쿼리를 사전 실행하여 demo_cache/ 에 JSON으로 저장.

실행 (프로젝트 루트에서):
  .venv/bin/python src/api/precompute.py

예상 소요: ~10분 (12쿼리 × 2파이프라인 × 25초)
"""

import sys
import json
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src"
for p in (str(_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.pipeline_runner import run_pipeline

CACHE_DIR = _ROOT / "data" / "demo_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# ── 샘플 쿼리 정의 ─────────────────────────────────────────────

SAMPLE_QUERIES = [
    # 일반 질문 (corpus 내 법령)
    {
        "category": "일반 질문",
        "description": "법령 조문에서 직접 답할 수 있는 질문",
        "queries": [
            {"qid": "s01", "question": "개인정보 수집 시 정보주체에게 알려야 할 사항은 무엇인가?", "qa_type": "factual"},
            {"qid": "s02", "question": "근로기준법상 성인 근로자의 1주 법정 근로시간은?", "qa_type": "factual"},
            {"qid": "s03", "question": "저작자 사망 후 저작권은 몇 년간 보호되는가?", "qa_type": "factual"},
            {"qid": "s04", "question": "공정거래법상 시장지배적 사업자로 추정되는 시장점유율 기준은?", "qa_type": "factual"},
            {"qid": "s05", "question": "전자금융거래법상 접근매체의 종류는 무엇인가?", "qa_type": "factual"},
            {"qid": "s06", "question": "저작권법상 업무상저작물의 보호기간은?", "qa_type": "factual"},
        ],
    },
    # Hard Negative (corpus 외 법령 → fallback 시연)
    {
        "category": "Hard Negative",
        "description": "수집된 법령 코퍼스 밖의 질문 — Fallback 처리를 시연합니다",
        "queries": [
            {"qid": "n01", "question": "주택임대차보호법에서 계약갱신청구권의 행사 기간은?", "qa_type": "negative"},
            {"qid": "n02", "question": "상법상 주식회사 설립 시 이사회 구성 요건은?", "qa_type": "negative"},
            {"qid": "n03", "question": "행정소송법상 취소소송의 제소기간은?", "qa_type": "negative"},
        ],
    },
]

PIPELINES = ["crag", "qdrant"]


def run_precompute():
    print("=" * 60)
    print("Pre-compute 시작")
    print(f"캐시 디렉토리: {CACHE_DIR}")
    print("=" * 60)

    # 디렉토리 생성
    for pipeline in PIPELINES:
        (CACHE_DIR / pipeline).mkdir(parents=True, exist_ok=True)

    total = sum(len(cat["queries"]) for cat in SAMPLE_QUERIES) * len(PIPELINES)
    done = 0
    t_start = time.time()

    for pipeline in PIPELINES:
        print(f"\n▶ Pipeline: {pipeline.upper()}")
        for category in SAMPLE_QUERIES:
            for q in category["queries"]:
                qid = q["qid"]
                question = q["question"]
                out_path = CACHE_DIR / pipeline / f"{qid}.json"

                if out_path.exists():
                    print(f"  [{done+1}/{total}] {qid} — 스킵 (캐시 있음)")
                    done += 1
                    continue

                print(f"  [{done+1}/{total}] {qid}: {question[:50]}...")
                t0 = time.time()
                try:
                    result = run_pipeline(question, pipeline)
                    elapsed = round(time.time() - t0, 1)
                    data = result.model_dump()
                    data["cached"] = True
                    out_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    decision = result.pipeline_trace.decision
                    print(f"    → {decision} | {elapsed}s ✓")
                except Exception as e:
                    print(f"    → 실패: {e}")
                done += 1

    # manifest.json 생성
    manifest = {
        "categories": [
            {
                "name": cat["category"],
                "description": cat["description"],
                "queries": [
                    {"qid": q["qid"], "question": q["question"], "qa_type": q["qa_type"]}
                    for q in cat["queries"]
                ],
            }
            for cat in SAMPLE_QUERIES
        ]
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_elapsed = round(time.time() - t_start, 1)
    print(f"\n✓ 완료: {done}개 캐시 생성 | 총 소요 {total_elapsed}s")
    print(f"  manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    run_precompute()
