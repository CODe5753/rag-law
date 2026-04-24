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
    {
        "category": "노동법",
        "description": "부당해고·임금체불·퇴직금",
        "queries": [
            {"qid": "l01", "question": "퇴직금을 못 받았을 때 어떻게 해야 하나요?", "qa_type": "factual"},
            {"qid": "l02", "question": "근로기준법상 성인 근로자의 1주 법정 근로시간은?", "qa_type": "factual"},
            {"qid": "l03", "question": "부당해고를 당했을 때 신청할 수 있는 구제 절차는?", "qa_type": "factual"},
        ],
    },
    {
        "category": "가족/연인",
        "description": "이혼·양육권·재산분할·가정폭력",
        "queries": [
            {"qid": "f01", "question": "협의이혼 절차와 필요한 서류는 무엇인가요?", "qa_type": "factual"},
            {"qid": "f02", "question": "사실혼 관계에서 재산분할을 청구할 수 있나요?", "qa_type": "factual"},
            {"qid": "f03", "question": "가정폭력 피해자가 받을 수 있는 보호 조치는?", "qa_type": "factual"},
        ],
    },
    {
        "category": "폭행",
        "description": "고소·정당방위·쌍방폭행",
        "queries": [
            {"qid": "v01", "question": "폭행 피해를 입었을 때 고소 방법과 절차는?", "qa_type": "factual"},
            {"qid": "v02", "question": "정당방위가 인정되는 요건은 무엇인가요?", "qa_type": "factual"},
            {"qid": "v03", "question": "쌍방폭행 시 처벌은 어떻게 되나요?", "qa_type": "factual"},
        ],
    },
    {
        "category": "부동산/임대차",
        "description": "전세·월세·보증금 분쟁",
        "queries": [
            {"qid": "r01", "question": "전세 보증금을 돌려받지 못할 때 법적 조치는?", "qa_type": "factual"},
            {"qid": "r02", "question": "임대차계약 해지 시 위약금 규정은?", "qa_type": "factual"},
        ],
    },
    {
        "category": "채권/사기",
        "description": "금전대차·계약위반·사기",
        "queries": [
            {"qid": "d01", "question": "빌려준 돈을 돌려받기 위한 법적 방법은?", "qa_type": "factual"},
            {"qid": "d02", "question": "사기죄 성립 요건과 고소 방법은?", "qa_type": "factual"},
        ],
    },
    {
        "category": "교통사고",
        "description": "과실비율·합의·보험",
        "queries": [
            {"qid": "t01", "question": "교통사고 과실비율은 어떻게 결정되나요?", "qa_type": "factual"},
            {"qid": "t02", "question": "교통사고 합의 시 주의해야 할 법적 사항은?", "qa_type": "factual"},
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
