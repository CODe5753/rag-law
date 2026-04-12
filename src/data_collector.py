"""
Phase 1: 법령 데이터 수집기

두 가지 모드:
  - Mode A (DRF API): law.go.kr DRF API 사용. 사전에 IP 등록 필요.
  - Mode B (Web Scraper): 인증 없이 공개 웹 페이지에서 법령 텍스트 직접 파싱.

⚠️ 트러블슈팅 #1 — DRF API IP 등록 필요 (2026-04-10 발견)
  증상: OC="" 또는 OC="임의값"으로 호출하면
        {"result": "사용자 정보 검증에 실패하였습니다.",
         "msg": "OPEN API 호출 시 사용자 검증을 위하여 정확한 서버장비의 IP주소 및 도메인주소를 등록해 주세요."}
  원인: law.go.kr DRF API는 사전에 사용자 ID(OC)와 서버 IP를 등록해야만 호출 가능.
         단순히 OC를 빈 문자열로 남기거나 임의 문자열을 넣는 것은 불가.
  해결 방법 A (권장): https://open.law.go.kr 에서 회원가입 후 OC 발급,
                       본인 서버 IP를 등록하면 즉시 사용 가능 (무료).
  해결 방법 B (이 스크립트): 웹 스크래퍼로 공개 법령 페이지에서 직접 파싱.
  트레이드오프: 스크래퍼는 HTML 구조 변경에 취약하고 대용량 수집이 느림.
               API 등록 후에는 Mode A로 전환하는 것을 권장.
"""

import requests
import json
import time
import re
from pathlib import Path
from bs4 import BeautifulSoup

BASE_URL = "https://www.law.go.kr/DRF"
WEB_BASE = "https://www.law.go.kr"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Mode A: DRF API (OC 등록 필요)
# ──────────────────────────────────────────────

def search_laws_api(query: str, oc: str, page: int = 1, display: int = 20) -> dict:
    """DRF API로 법령 목록 검색. OC 등록 필수."""
    params = {
        "OC": oc,
        "target": "law",
        "type": "JSON",
        "query": query,
        "page": page,
        "display": display,
    }
    resp = requests.get(f"{BASE_URL}/lawSearch.do", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_law_content_api(law_id: str, oc: str) -> dict | None:
    """DRF API로 법령 전문 조회. OC 등록 필수."""
    params = {"OC": oc, "target": "law", "ID": law_id, "type": "JSON"}
    resp = requests.get(f"{BASE_URL}/lawService.do", params=params, timeout=15)
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        # ⚠️ 트러블슈팅 #2: type=JSON 지정에도 XML 반환하는 엔드포인트 있음
        # 원인: 일부 고령 법령은 XML 전용 포맷으로만 제공됨
        # 해결: XML 파싱으로 폴백
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        return {"raw_xml": resp.text, "_format": "xml"}


# ──────────────────────────────────────────────
# Mode B: 웹 스크래퍼 (인증 없음, 공개 페이지)
# ──────────────────────────────────────────────

# 주요 법령 목록 (법령명 → 법령 일련번호 lsiSeq)
# law.go.kr URL 패턴: /lsInfoP.do?lsiSeq=<번호>
KNOWN_LAWS = {
    "개인정보보호법": 253363,
    "근로기준법": 254218,
    "전자상거래법": 251603,
    "소비자기본법": 252789,
    "저작권법": 253614,
    "정보통신망법": 253802,
    "전기통신사업법": 254041,
    "금융소비자보호법": 252314,
    "신용정보법": 253941,
    "공정거래법": 252861,
    "약관규제법": 252866,
    "전자금융거래법": 253937,
    "민법": 254265,
    "상법": 254266,
    "행정기본법": 252605,
}


def fetch_law_web(law_name: str, lsi_seq: int) -> dict | None:
    """
    law.go.kr 공개 웹 페이지에서 법령 텍스트 파싱 (인증 불필요).

    ⚠️ 트러블슈팅 #3: lsiSeq는 개정마다 바뀜 (2026-04-10 확인)
    원인: 법령이 개정되면 lsiSeq가 새로 발급되고 구 lsiSeq는 리디렉션됨.
    해결: 정기적으로 lsiSeq를 재확인하거나, DRF API 등록 후 법령ID로 조회.
    """
    url = f"{WEB_BASE}/lsEfInfoP.do?lsiSeq={lsi_seq}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [SKIP] {law_name}: 접속 실패 — {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 조문 텍스트 추출
    articles = []

    # 법령 전문 텍스트 영역 탐색
    # ⚠️ 트러블슈팅 #4: law.go.kr HTML 구조가 JavaScript-rendered SPA라
    #    curl로는 완전한 조문 텍스트가 안 나올 수 있음.
    #    BeautifulSoup으로 파싱 가능한 정적 텍스트만 추출.
    content_div = soup.find("div", class_=re.compile(r"(law|article|content)", re.I))
    if not content_div:
        content_div = soup.find("body")

    if content_div:
        text_blocks = content_div.get_text(separator="\n", strip=True)
        # 조문 패턴으로 분리 (제N조, 제N조의N)
        article_pattern = re.compile(r"(제\d+조(?:의\d+)?(?:\s*\([^)]+\))?)")
        parts = article_pattern.split(text_blocks)

        current_article = ""
        current_num = ""
        for part in parts:
            if article_pattern.match(part):
                if current_num and current_article.strip():
                    articles.append({
                        "article_num": current_num,
                        "text": current_article.strip(),
                    })
                current_num = part
                current_article = part
            else:
                current_article += part

        if current_num and current_article.strip():
            articles.append({
                "article_num": current_num,
                "text": current_article.strip(),
            })

    if not articles:
        # 폴백: 페이지 전체 텍스트를 단일 블록으로 저장
        raw_text = soup.get_text(separator="\n", strip=True)
        if len(raw_text) > 200:
            articles = [{"article_num": "전문", "text": raw_text[:5000]}]

    return {
        "law_name": law_name,
        "lsi_seq": lsi_seq,
        "url": url,
        "articles": articles,
        "article_count": len(articles),
    }


def collect_laws_web(law_dict: dict = None, delay: float = 1.5) -> list[dict]:
    """
    Mode B: 공개 웹 페이지에서 주요 법령 수집.
    delay: 요청 간 대기 시간 (서버 부하 방지)
    """
    if law_dict is None:
        law_dict = KNOWN_LAWS

    results = []
    for law_name, lsi_seq in law_dict.items():
        print(f"  수집 중: {law_name} (lsiSeq={lsi_seq})")
        data = fetch_law_web(law_name, lsi_seq)
        if data:
            results.append(data)
            print(f"  → {data['article_count']}개 조문 파싱")
        time.sleep(delay)

    return results


if __name__ == "__main__":
    print("=== Phase 1: 법령 데이터 수집 (Mode B — 웹 스크래퍼) ===\n")

    laws = collect_laws_web()

    output_path = DATA_DIR / "laws_raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(laws, f, ensure_ascii=False, indent=2)

    total_articles = sum(d["article_count"] for d in laws)
    print(f"\n수집 완료: {len(laws)}개 법령, {total_articles}개 조문")
    print(f"저장 위치: {output_path}")
