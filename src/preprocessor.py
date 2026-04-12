"""
Phase 1: 법령 데이터 전처리 — 조문 정제 + 200자 청킹

청킹 전략 근거:
  - 청크 크기: 200자 고정 (KCI 2026 논문: KorQuAD 기준 200자 = 68.8% 정확도 최고)
  - 오버랩: 20자 (10% 오버랩, 조문 경계 손실 방지)
  - 조문 단위 선 분리 후 청킹 (조문 제목이 반드시 청크 첫머리에 포함되도록)

출력 스키마 (chunks.json):
  [
    {
      "chunk_id": "개인정보보호법_000",
      "law_name": "개인정보보호법",
      "article_num": "제1조(목적)",
      "text": "...",           # 최대 200자
      "char_len": 182
    },
    ...
  ]
"""

import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw/laws_raw.json")
OUT_DIR = Path("data/processed")
CHUNK_SIZE = 200   # 자
OVERLAP = 20       # 자


def clean_article_text(text: str) -> str:
    """조문 텍스트 정제: 연속 공백/줄바꿈 압축, 불필요한 네비게이션 텍스트 제거"""
    # 법제처 웹 페이지 공통 헤더/푸터 제거
    noise_patterns = [
        r"법령\s*>.*?법령보기",
        r"현재\s*화면\s*에서.*?제공합니다",
        r"홈\s*>\s*법령.*?시행",
        r"조문\s*선택.*?이동",
        r"이\s*페이지.*?드립니다",
        r"자주\s*찾는\s*법령",
        r"관련\s*법령",
        r"판례\s*검색",
        r"법령해석\s*사례",
        r"생활\s*법령",
        r"법령\s*번역",
    ]
    for pat in noise_patterns:
        text = re.sub(pat, " ", text, flags=re.DOTALL | re.IGNORECASE)

    # 연속 공백/탭/줄바꿈 압축
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    고정 크기 슬라이딩 윈도우 청킹.
    단어/문장 경계를 완전히 무시하지 않도록 공백에서 청크 시작.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        # 다음 청크 시작: overlap만큼 뒤로
        start = end - overlap

    return chunks


def process_law(law_data: dict) -> list[dict]:
    """단일 법령 → 청크 리스트 변환"""
    law_name = law_data["law_name"]
    chunks = []
    chunk_idx = 0

    for article in law_data["articles"]:
        article_num = article["article_num"].strip()
        raw_text = article["text"].strip()

        cleaned = clean_article_text(raw_text)
        if len(cleaned) < 10:
            continue

        # 조문 번호가 텍스트에 없으면 앞에 붙임
        if article_num and not cleaned.startswith(article_num):
            cleaned = f"{article_num} {cleaned}"

        # 청킹
        text_chunks = chunk_text(cleaned)
        for chunk in text_chunks:
            chunk = chunk.strip()
            if len(chunk) < 10:
                continue
            chunks.append({
                "chunk_id": f"{law_name}_{chunk_idx:04d}",
                "law_name": law_name,
                "article_num": article_num,
                "text": chunk,
                "char_len": len(chunk),
            })
            chunk_idx += 1

    return chunks


def main():
    print("=== Phase 1: 전처리 시작 ===\n")

    raw_data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    stats = []

    for law in raw_data:
        law_chunks = process_law(law)
        all_chunks.extend(law_chunks)

        char_lens = [c["char_len"] for c in law_chunks]
        avg_len = sum(char_lens) / len(char_lens) if char_lens else 0
        stats.append({
            "law": law["law_name"],
            "articles": law["article_count"],
            "chunks": len(law_chunks),
            "avg_chars": round(avg_len, 1),
        })
        print(f"  {law['law_name']}: {law['article_count']}개 조문 → {len(law_chunks)}개 청크 (평균 {avg_len:.0f}자)")

    out_path = OUT_DIR / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    total_chunks = len(all_chunks)
    all_lens = [c["char_len"] for c in all_chunks]
    overall_avg = sum(all_lens) / len(all_lens) if all_lens else 0

    print(f"\n=== 전처리 완료 ===")
    print(f"총 청크: {total_chunks:,}개")
    print(f"평균 길이: {overall_avg:.1f}자")
    print(f"최소: {min(all_lens)}자 / 최대: {max(all_lens)}자")
    print(f"저장: {out_path}")

    # 통계 저장
    stats_path = OUT_DIR / "stats.json"
    stats_path.write_text(json.dumps({
        "total_chunks": total_chunks,
        "avg_char_len": round(overall_avg, 1),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "laws": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"통계: {stats_path}")


if __name__ == "__main__":
    main()
