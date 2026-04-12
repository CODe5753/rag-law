"""
법제처 웹 스크래퍼 — Playwright 기반 (JS 렌더링 지원)

⚠️ 트러블슈팅 히스토리 (2026-04-10):

#1 — DRF API: OC 없이 호출 불가
  증상: OC="" 또는 임의값으로 DRF API 호출 시 사용자 검증 실패
  원인: law.go.kr DRF API는 회원가입 + 서버 IP 등록 필수 (구 문서의 "OC 빈 문자열" 안내는 현재 동작 안 함)
  해결: DRF 등록 불가 상황에서는 Playwright 웹 스크래퍼 사용

#2 — requests + BeautifulSoup: 조문 텍스트 수집 불가
  증상: curl/requests로 lsInfoP.do 접근 시 HTML 껍데기만 반환, 조문 텍스트 없음
  원인: law.go.kr는 SPA(Single Page Application). 조문은 JS 실행 후 AJAX로 로드됨
  해결: Playwright로 실제 브라우저 구동 + networkidle 대기 후 inner_text() 추출

#3 — lsiSeq 하드코딩 방식 실패
  증상: 임의로 추측한 lsiSeq(예: 253363)가 다른 법령(에너지이용합리화법)을 반환
  원인: lsiSeq는 법령 공포 순서로 부여되며 외부에서 예측 불가. 개정마다 새 번호 발급.
  해결: 검색 페이지 네트워크 요청에서 lsiSeq 동적 추출
        (csmLnkListR.do?lsiSeq=XXXXXX 패턴이 응답 URL에 포함됨)

#4 — 검색 URL의 query 파라미터가 AJAX 결과에 반영 안 됨
  증상: lsSc.do?query=개인정보보호법 접근 시 검색창에 텍스트 입력 됐으나 결과 lsiSeq=0
  원인: 검색 결과가 별도 XHR로 비동기 로드되며 초기 HTML에는 포함 안 됨
  해결: 검색 후 발생하는 네트워크 요청 URL에서 lsiSeq 패턴 캡처 (csmLnkListR.do?lsiSeq=XXXXX)
"""

import asyncio
import json
import re
import time
from pathlib import Path
from playwright.async_api import async_playwright

BASE_URL = "https://www.law.go.kr"
DATA_DIR = Path("data/raw")

# 확인된 주요 법령 lsiSeq (2026-04-10 기준, 개정 시 재조회 필요)
# find_lsi_seq()로 동적 탐색 가능
KNOWN_LSI_SEQS = {
    "개인정보보호법": 270351,   # 확인됨 (2026-04-10)
    "근로기준법": 265959,       # 확인됨 (2026-04-10)
    "소비자기본법": 277233,     # 확인됨 (2026-04-10)
    "저작권법": 270165,         # 확인됨 (2026-04-10)
    "전자금융거래법": 280277,   # 확인됨 (2026-04-10)
    # lsViewWideAll 패턴으로 확인됨 (2026-04-10)
    "전자상거래법": 282793,     # 전자상거래 등에서의 소비자보호에 관한 법률
    "정보통신망법": 277377,     # 정보통신망 이용촉진 및 정보보호 등에 관한 법률 (현행)
    "신용정보법": 260423,       # 신용정보의 이용 및 보호에 관한 법률
    "금융소비자보호법": 277247, # 금융소비자 보호에 관한 법률
    "공정거래법": 277327,       # 독점규제 및 공정거래에 관한 법률
}

TARGET_LAWS = [
    "개인정보보호법",
    "근로기준법",
    "전자상거래법",
    "소비자기본법",
    "저작권법",
    "정보통신망법",
    "신용정보법",
    "금융소비자보호법",
    "전자금융거래법",
    "공정거래법",
]


async def find_lsi_seq(page, law_name: str) -> int | None:
    """
    법제처 검색으로 법령 lsiSeq 동적 탐색.

    전략 1: csmLnkListR.do?lsiSeq=XXXXX 패턴 (URL에서 추출)
    전략 2: lsScListR.do 응답 HTML의 lsViewWideAll('lsiSeq','date',...,'status',...) 패턴
            status='3' = 현행 시행, status='2' = 미시행
            법령 자체(시행령/규칙 제외)의 현행 lsiSeq를 우선 선택
    """
    captured_seqs = []      # 전략 1: URL 패턴
    list_body = []          # 전략 2: 응답 body

    async def capture_response(response):
        url = response.url
        # 전략 1
        m = re.search(r"lsiSeq=(\d{5,})", url)
        if m:
            captured_seqs.append(int(m.group(1)))
        # 전략 2
        if "lsScListR.do" in url:
            try:
                body = await response.text()
                list_body.append(body)
            except Exception:
                pass

    page.on("response", capture_response)
    search_url = f"{BASE_URL}/lsSc.do?query={law_name}&target=law&section=&menuId=1"
    await page.goto(search_url, wait_until="networkidle", timeout=30000)
    await page.wait_for_timeout(3000)
    page.remove_listener("response", capture_response)

    # 전략 1 성공
    if captured_seqs:
        return max(captured_seqs)

    # 전략 2: lsViewWideAll 패턴으로 현행 법령 lsiSeq 추출
    # lsViewWideAll('lsiSeq','date','elemId',$(this),'status','isSeq','isLaw','menuId')
    # status: '3'=현행, '2'=미시행
    for body in list_body:
        # (lsiSeq, date, status) 추출
        matches = re.findall(r"lsViewWideAll\('(\d+)','(\d+)',[^,]+,\$\(this\),'(\d+)'", body)
        if not matches:
            continue
        # 현행(status=3) 우선, 없으면 미시행(status=2)
        active = [(int(seq), date) for seq, date, st in matches if st == "3"]
        pending = [(int(seq), date) for seq, date, st in matches if st == "2"]
        candidates = active if active else pending
        if candidates:
            # 가장 최신 날짜
            return max(candidates, key=lambda x: x[1])[0]

    return None


async def scrape_law_by_seq(page, law_name: str, lsi_seq: int) -> dict:
    """lsiSeq로 법령 전문 텍스트 스크래핑"""
    url = f"{BASE_URL}/lsInfoP.do?lsiSeq={lsi_seq}#0000"
    await page.goto(url, wait_until="networkidle", timeout=30000)
    await page.wait_for_timeout(3000)

    body_text = await page.inner_text("body")

    # 조문 단위로 분리
    article_pattern = re.compile(r"(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\([^)]{1,40}\))?)")
    parts = article_pattern.split(body_text)

    articles = []
    current_num = ""
    current_text = ""

    for part in parts:
        stripped = part.strip()
        if article_pattern.match(stripped):
            if current_num and len(current_text.strip()) > 20:
                full = (current_num + current_text).strip()
                articles.append({
                    "article_num": current_num.strip(),
                    "text": full[:1500],  # 청킹 전 원본 보존 (최대 1500자)
                })
            current_num = stripped
            current_text = ""
        else:
            current_text += part

    if current_num and len(current_text.strip()) > 20:
        articles.append({
            "article_num": current_num.strip(),
            "text": (current_num + current_text).strip()[:1500],
        })

    return {
        "law_name": law_name,
        "lsi_seq": lsi_seq,
        "url": url,
        "articles": articles,
        "article_count": len(articles),
    }


async def collect_all_laws(law_names: list[str] = None) -> list[dict]:
    if law_names is None:
        law_names = TARGET_LAWS

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="ko-KR",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )
        page = await context.new_page()

        for law_name in law_names:
            print(f"\n[{law_name}]")

            # lsiSeq 확인 (캐시 → 동적 탐색)
            lsi_seq = KNOWN_LSI_SEQS.get(law_name)
            if not lsi_seq:
                print(f"  lsiSeq 탐색 중...")
                lsi_seq = await find_lsi_seq(page, law_name)
                if lsi_seq:
                    KNOWN_LSI_SEQS[law_name] = lsi_seq
                    print(f"  lsiSeq={lsi_seq} 발견")
                else:
                    print(f"  lsiSeq 탐색 실패 — 스킵")
                    continue

            # 법령 스크래핑
            data = await scrape_law_by_seq(page, law_name, lsi_seq)
            results.append(data)
            print(f"  → {data['article_count']}개 조문 수집")

            await asyncio.sleep(2)  # 서버 부하 방지

        await browser.close()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--laws", nargs="+", default=TARGET_LAWS)
    parser.add_argument("--output", default="data/raw/laws_raw.json")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(collect_all_laws(args.laws))

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = sum(d["article_count"] for d in results)
    print(f"\n=== 수집 완료 ===")
    print(f"법령: {len(results)}개 / 조문: {total}개")
    print(f"저장: {output}")

    # lsiSeq 캐시 저장 (재실행 시 재탐색 불필요)
    seq_cache = DATA_DIR / "lsi_seq_cache.json"
    with open(seq_cache, "w", encoding="utf-8") as f:
        json.dump(KNOWN_LSI_SEQS, f, ensure_ascii=False, indent=2)
    print(f"lsiSeq 캐시: {seq_cache}")
