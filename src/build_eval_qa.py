"""
수작업 품질 QA 데이터셋 생성

목적:
  합성 QA(청크에서 생성)의 순환 평가 문제를 해결하기 위해
  실제 법령 조문에서 다양한 유형의 QA를 생성한다.

접근:
  - 입력: laws_raw.json (조문 원문, 청킹 아님)
  - 법령당 8문항 × 10개 법령 = 80개 answerable
  - 20개 hard negative (corpus에 없는 내용)
  - retrieval_gt: 조문 번호 → 청크 ID 역매핑 (멀티 청크 지원)

질문 유형 (법령당):
  factual(2): 수치/기한/조건 — 단일 조문으로 답 가능
  procedural(2): 신고→조사→처분 등 절차 흐름
  multi_article(2): 2개 이상 조문이 함께 있어야 답 가능
  conditional(1): 단서 조항 / 예외 조건
  definition(1): 용어 정의
"""

import json
import re
import time
from pathlib import Path
from collections import defaultdict

import requests

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "exaone3.5:7.8b"
RAW_PATH = Path("data/raw/laws_raw.json")
CHUNKS_PATH = Path("data/processed/chunks.json")
OUT_PATH = Path("data/processed/eval_qa_v2.json")

# 법령당 생성할 QA 유형 및 수
QA_PLAN = [
    ("factual",       "단일 조문의 구체적인 수치·기한·요건을 묻는 질문"),
    ("factual",       "단일 조문의 구체적인 수치·기한·요건을 묻는 질문 (다른 조문)"),
    ("procedural",    "신고·신청·처분 등 절차 흐름 전체를 묻는 질문"),
    ("procedural",    "위반 시 처벌·제재 절차를 묻는 질문"),
    ("multi_article", "두 개 이상의 조문을 교차 참조해야 답할 수 있는 질문"),
    ("multi_article", "주된 의무와 예외 규정이 다른 조문에 있어서 함께 봐야 하는 질문"),
    ("conditional",   "단서 조항(단, ·다만) 또는 예외 조건을 묻는 질문"),
    ("definition",    "이 법에서 정의하는 특정 용어의 의미를 묻는 질문"),
]

# Hard negative: corpus에 없는 내용 (20개 수작업)
HARD_NEGATIVES = [
    {
        "qid": "neg_001",
        "question": "주택임대차보호법에서 계약갱신청구권의 행사 기간은 얼마인가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "주택임대차보호법은 corpus에 없음",
    },
    {
        "qid": "neg_002",
        "question": "소득세법에서 근로소득공제율은 어떻게 계산하는가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "소득세법은 corpus에 없음",
    },
    {
        "qid": "neg_003",
        "question": "형법 제250조에서 살인죄의 법정형은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "형법은 corpus에 없음",
    },
    {
        "qid": "neg_004",
        "question": "상가건물 임대차보호법의 권리금 회수 보장 기간은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "상가건물임대차보호법은 corpus에 없음",
    },
    {
        "qid": "neg_005",
        "question": "도로교통법에서 음주운전 혈중 알코올 농도 기준은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "도로교통법은 corpus에 없음",
    },
    {
        "qid": "neg_006",
        "question": "근로기준법에서 퇴직금 지급 기한은 퇴직 후 며칠 이내인가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "근로기준법에 퇴직금 조항 없음 (근로자퇴직급여보장법 소관)",
    },
    {
        "qid": "neg_007",
        "question": "개인정보보호법에서 신용정보 조회 기록의 보존 기간은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "신용정보 조회 기록 보존은 신용정보법 소관이며, 개인정보보호법에 해당 조항 없음",
    },
    {
        "qid": "neg_008",
        "question": "공정거래법에서 담합(카르텔)에 부과하는 과징금의 상한 비율은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "공정거래법 corpus에 과징금 구체 비율 조문이 포함되어 있지 않음",
    },
    {
        "qid": "neg_009",
        "question": "전자금융거래법에서 핀테크 기업의 자본금 요건은 얼마인가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "자본금 요건은 시행령 소관이며 corpus에 없음",
    },
    {
        "qid": "neg_010",
        "question": "저작권법에서 음악 저작물의 저작재산권 보호 기간은 사후 몇 년인가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "보호 기간 조문이 corpus에 포함되어 있지 않음",
    },
    {
        "qid": "neg_011",
        "question": "정보통신망법에서 스팸 문자 발송 시 과태료 금액은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "과태료 금액은 corpus에 없음",
    },
    {
        "qid": "neg_012",
        "question": "소비자기본법에서 제조물책임의 손해배상 청구 시효는?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "제조물책임은 제조물책임법 소관이며 corpus에 없음",
    },
    {
        "qid": "neg_013",
        "question": "금융소비자보호법에서 보험 판매 시 적합성 원칙을 위반했을 때 형사처벌 수위는?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "형사처벌 조항이 corpus에 없음",
    },
    {
        "qid": "neg_014",
        "question": "신용정보법에서 신용평점 산출 알고리즘의 공개 의무가 있는가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "알고리즘 공개 의무 조항은 corpus에 없음",
    },
    {
        "qid": "neg_015",
        "question": "전자상거래법에서 해외 직구 소비자의 환급 청구 절차는?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "해외 직구 특화 조항은 corpus에 없음",
    },
    {
        "qid": "neg_016",
        "question": "공정거래법에서 기업결합 신고 대상의 자산 기준 금액은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "구체적 금액 기준은 시행령 소관이며 corpus에 없음",
    },
    {
        "qid": "neg_017",
        "question": "개인정보보호법과 신용정보법이 동시에 적용될 때 우선 적용 법령은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "두 법령 간 우선순위 조항이 corpus에 명시되어 있지 않음",
    },
    {
        "qid": "neg_018",
        "question": "근로기준법에서 산업재해 발생 시 사용자의 보상 기준은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "산재 보상은 산업재해보상보험법 소관이며 corpus에 없음",
    },
    {
        "qid": "neg_019",
        "question": "저작권법에서 AI가 생성한 콘텐츠의 저작권 귀속 기준은?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "AI 생성물 저작권 귀속 조항은 corpus에 없음",
    },
    {
        "qid": "neg_020",
        "question": "소비자기본법에서 소비자분쟁조정위원회의 위원 수는 몇 명인가?",
        "answer": None,
        "retrieval_gt": [],
        "law_refs": [],
        "article_refs": [],
        "qa_type": "negative",
        "difficulty": "negative",
        "negative_reason": "위원 수 조항이 corpus에 없음",
    },
]


def normalize_article(article_num: str) -> str:
    m = re.match(r"(제\d+조)", article_num)
    return m.group(1) if m else article_num


def build_chunk_index(chunks: list[dict]) -> dict:
    idx = defaultdict(list)
    for c in chunks:
        key = (c["law_name"], normalize_article(c["article_num"]))
        idx[key].append(c["chunk_id"])
    return idx


def ollama_generate(prompt: str, temperature: float = 0.3) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt,
              "stream": False, "options": {"temperature": temperature}},
        timeout=120,
    )
    return resp.json()["response"].strip()


def select_articles(law: dict, n: int = 12) -> list[dict]:
    """실질적 내용이 있는 조문 선택 (너무 짧거나 장 제목만 있는 것 제외)"""
    candidates = []
    for a in law["articles"]:
        text = a["text"]
        # 최소 150자, ① 또는 ②가 있거나 "하여야 한다" 같은 의무 규정 포함
        if len(text) >= 150 and (
            "①" in text or "하여야 한다" in text or "할 수 있다" in text or "아니 된다" in text
        ):
            candidates.append(a)
    # 길이 순 정렬 후 균등하게 샘플링
    candidates.sort(key=lambda a: len(a["text"]), reverse=True)
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


def generate_qa_for_article(
    law_name: str,
    article: dict,
    qa_type: str,
    qa_description: str,
    qid: str,
) -> dict | None:
    """단일 조문에서 QA 생성"""
    article_num = article["article_num"]
    article_text = article["text"][:1200]  # 너무 길면 컨텍스트 초과

    prompt = f"""당신은 한국 법령 RAG 시스템의 품질 평가 전문가입니다.
아래 법령 조문을 읽고, 지정된 유형의 질문 1개와 정답을 생성하세요.

법령명: {law_name}
조문: {article_text}

질문 유형: {qa_description}

요구사항:
- 질문은 조문을 직접 읽지 않은 사람이 실제로 궁금해할 법한 자연스러운 한국어 질문이어야 합니다
- 정답은 위 조문의 내용만을 근거로 해야 합니다 (추측 금지)
- 조문 번호(제X조)를 질문에 직접 언급하지 마세요 (검색 테스트 목적)
- 답변은 1~3문장으로 간결하게

다음 형식으로만 출력하세요:
질문: (질문 내용)
정답: (정답 내용)
참조_조문: (답변 근거가 되는 조문 번호, 예: 제15조, 제17조)"""

    raw = ollama_generate(prompt)

    q_match = re.search(r"질문:\s*(.+?)(?=\n정답:|\Z)", raw, re.DOTALL)
    a_match = re.search(r"정답:\s*(.+?)(?=\n참조_조문:|\Z)", raw, re.DOTALL)
    ref_match = re.search(r"참조_조문:\s*(.+)", raw)

    if not q_match or not a_match:
        return None

    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()

    # 참조 조문 파싱
    article_refs = []
    if ref_match:
        refs_raw = ref_match.group(1).strip()
        article_refs = re.findall(r"제\d+조", refs_raw)
    if not article_refs:
        article_refs = [normalize_article(article_num)]

    return {
        "qid": qid,
        "question": question,
        "answer": answer,
        "retrieval_gt": [],  # 나중에 매핑
        "law_refs": [law_name],
        "article_refs": article_refs,
        "qa_type": qa_type,
        "difficulty": "multi_article" if qa_type == "multi_article" else "single",
        "source_article": article_num,
    }


def generate_multi_article_qa(
    law_name: str,
    articles: list[dict],
    qa_type: str,
    qa_description: str,
    qid: str,
) -> dict | None:
    """두 조문을 함께 사용해 QA 생성"""
    if len(articles) < 2:
        return None

    a1, a2 = articles[0], articles[1]
    combined = f"[조문1] {a1['text'][:600]}\n\n[조문2] {a2['text'][:600]}"

    prompt = f"""당신은 한국 법령 RAG 시스템의 품질 평가 전문가입니다.
아래 두 개의 법령 조문을 읽고, 두 조문 모두를 참조해야 완전히 답할 수 있는 질문과 정답을 생성하세요.

법령명: {law_name}
{combined}

질문 유형: {qa_description}

요구사항:
- 두 조문 모두의 내용이 있어야 완전한 답변이 가능한 질문이어야 합니다
- 조문 번호(제X조)를 질문에 직접 언급하지 마세요
- 자연스러운 한국어 질문
- 답변은 2~4문장

다음 형식으로만 출력하세요:
질문: (질문 내용)
정답: (정답 내용)
참조_조문: {a1['article_num']}, {a2['article_num']}"""

    raw = ollama_generate(prompt)

    q_match = re.search(r"질문:\s*(.+?)(?=\n정답:|\Z)", raw, re.DOTALL)
    a_match = re.search(r"정답:\s*(.+?)(?=\n참조_조문:|\Z)", raw, re.DOTALL)

    if not q_match or not a_match:
        return None

    return {
        "qid": qid,
        "question": q_match.group(1).strip(),
        "answer": a_match.group(1).strip(),
        "retrieval_gt": [],
        "law_refs": [law_name],
        "article_refs": [normalize_article(a1["article_num"]),
                         normalize_article(a2["article_num"])],
        "qa_type": qa_type,
        "difficulty": "multi_article",
        "source_article": f"{a1['article_num']}, {a2['article_num']}",
    }


def map_retrieval_gt(qa: dict, chunk_index: dict) -> dict:
    """article_refs → chunk_id 목록으로 변환"""
    gt = []
    for law in qa["law_refs"]:
        for article in qa["article_refs"]:
            key = (law, article)
            gt.extend(chunk_index.get(key, []))
    qa["retrieval_gt"] = sorted(set(gt))
    return qa


def run():
    print("=== 수작업 품질 QA 생성 시작 ===\n")

    laws = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunk_index = build_chunk_index(chunks)

    all_qa = []
    qid_counter = 1

    for law in laws:
        law_name = law["law_name"]
        print(f"\n[{law_name}] 조문 선택 중...")

        articles = select_articles(law, n=16)
        print(f"  선택된 조문: {len(articles)}개")
        if len(articles) < 4:
            print(f"  ⚠ 조문이 너무 적음, 스킵")
            continue

        law_qa = []
        article_cursor = 0

        for i, (qa_type, qa_desc) in enumerate(QA_PLAN):
            qid = f"qa_{qid_counter:03d}"
            qid_counter += 1

            print(f"  [{i+1}/8] {qa_type}: ", end="", flush=True)

            if qa_type == "multi_article":
                # 두 조문 필요
                pair = articles[article_cursor:article_cursor+2]
                article_cursor = min(article_cursor + 2, len(articles) - 2)
                qa = generate_multi_article_qa(law_name, pair, qa_type, qa_desc, qid)
            else:
                article = articles[article_cursor % len(articles)]
                article_cursor += 1
                qa = generate_qa_for_article(law_name, article, qa_type, qa_desc, qid)

            if qa:
                qa = map_retrieval_gt(qa, chunk_index)
                law_qa.append(qa)
                print(f"✓ (retrieval_gt: {len(qa['retrieval_gt'])}개 청크)")
            else:
                print("✗ 생성 실패")

            time.sleep(0.5)

        print(f"  → {len(law_qa)}개 생성")
        all_qa.extend(law_qa)

    # Hard negative 추가
    all_qa.extend(HARD_NEGATIVES)

    # 저장
    result = {
        "version": "v2_article_based",
        "description": "조문 원문 기반 수작업 품질 QA (청크 순환 평가 해결)",
        "total": len(all_qa),
        "answerable": len([q for q in all_qa if q["qa_type"] != "negative"]),
        "negative": len([q for q in all_qa if q["qa_type"] == "negative"]),
        "qa_type_distribution": {},
        "samples": all_qa,
    }

    # 타입별 분포 계산
    from collections import Counter
    dist = Counter(q["qa_type"] for q in all_qa)
    result["qa_type_distribution"] = dict(dist)

    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 완료 ===")
    print(f"총 {result['total']}개 (answerable: {result['answerable']}, negative: {result['negative']})")
    print(f"타입 분포: {result['qa_type_distribution']}")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    run()
