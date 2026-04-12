# Korean Law RAG

한국 법령을 질문하면 관련 조문을 찾아 답하는 RAG 시스템입니다. Naive RAG에서 출발해 Hybrid 검색, BGE Reranker, CRAG(Corrective RAG)까지 단계적으로 고도화했습니다.

파이프라인을 발전시키면서 평가 데이터셋 자체가 결과를 오염시키고 있다는 걸 발견했습니다. 독립 평가셋을 설계해 다시 측정하니 Faithfulness 0.95가 0.59로 내려갔습니다.

## 평가 데이터셋 문제

처음 CRAG를 평가했을 때 Faithfulness **0.95**, Answer Relevancy **0.97**이 나왔습니다. 이 숫자를 믿지 않았습니다.

QA 데이터셋이 RAG가 검색할 청크에서 직접 생성되었기 때문에, 모델은 항상 정답 문서를 찾을 수 있는 상황이었습니다. 모든 Phase에서 0.95/0.97이 수렴하는 것 자체가 계측 문제를 의심하게 했습니다.

독립 평가셋 **eval\_qa\_v2**를 직접 설계했습니다.

| | 생성 방식 | Hard Negative | n | Faithfulness | Answer Relevancy |
|--|---------|-------------|---|-------------|-----------------|
| v1 (순환) | 청크에서 QA 생성 | 없음 | 6 | 0.95 | 0.97 |
| **v2 (독립)** | 조문 원문에서 QA 생성 | 20% | 80+20 | **0.59** | **0.17** |

v2 설계 기준: 조문 원문 직접 출제 / Hard Negative 20%(코퍼스 외 질문) / Multi-chunk retrieval\_gt / QA 유형 다양화(factual·procedural·multi\_article·conditional·definition)

진짜 숫자는 **0.59**였습니다.

## 시스템 구조

```
Query
  │
  ▼
[Hybrid Retrieve]
  Dense (BGE-M3)  +  Sparse (Kiwi + BM25)
  └──▶ BGE Reranker Cross-Encoder → reranker_score 계산
  │
  ▼
[Grade Documents]  ← reranker_score 직접 재활용 (RERANKER_THRESHOLD = 0.5)
  │
  ├── score ≥ 0.5 → [Generate]
  │
  └── 불충분
       ├── [Query Rewrite] → Retrieve again
       └── [Web Search Fallback]  (Tavily, optional)
```

LangGraph로 파이프라인 상태를 관리합니다. 모델은 exaone3.5:7.8b(Ollama 로컬)를 씁니다.

개발 순서: Naive RAG(ChromaDB) → Hybrid+BGE Reranker(ChromaDB) → Qdrant 마이그레이션 → CRAG. 상세 진화 내역은 [docs/architecture-evolution.md](docs/architecture-evolution.md)를 참고하세요.

## 기술 결정

### 1. Qdrant: Native Hybrid Search

ChromaDB는 Sparse Vector를 지원하지 않아 Dense/BM25 결과를 애플리케이션 레이어에서 수동 병합해야 했습니다. Qdrant의 Named Vector + Sparse Vector 구조로 단일 `query_points()` 호출로 처리합니다. 개발 중 Qdrant가 `search()` API를 제거해 `query_points()`로 마이그레이션한 대응도 포함합니다.

### 2. Kiwi + BM25: 한국어 형태소 기반 희소 검색

영문 토크나이저로는 "개인정보처리자", "분쟁조정위원회" 같은 법령 복합어를 단어 단위로 나누지 못합니다. Kiwi 형태소 분석기로 토크나이즈한 BM25를 Dense 검색과 결합해 법령 용어 매칭 정확도를 높였습니다.

### 3. Reranker 점수 재활용: LLM 호출 제거

| | 방식 | 레이턴시 |
|--|------|--------|
| Before | LLM 채점 8회 직렬 호출 | 30~60초 |
| After | `reranker_score` 필드 전달, Grade 단계 직접 사용 | 15~30초 |

BGE Reranker가 Cross-Encoder로 이미 계산한 관련도 점수가 파이프라인 중간에 버려지고 있었습니다. `reranker_score`를 doc 딕셔너리에 첨부해 CRAG Grade 단계에서 재활용했습니다. LLM 채점 0회.

### 4. 로컬 RAGAS: 외부 API 없는 완전 로컬 평가

RAGAS 기본값은 OpenAI API를 씁니다. Ollama + exaone3.5:7.8b를 LLM Judge로 구성해 API 비용 없이 반복 평가를 가능하게 했습니다. 소규모 모델의 한계로 일부 NaN이 발생하지만(v2: 2/100), 평가 파이프라인 자체를 코드로 소유할 수 있습니다.

## 결과

v2 평가셋 기준. RAGAS 메트릭은 80개 일반 QA, Hard Negative Fallback은 별도 20개로 측정했습니다.  
측정 환경: Mac Mini M2 16GB, Ollama exaone3.5:7.8b 로컬.

| 메트릭 | 값 |
|--------|-----|
| Faithfulness | 0.5858 |
| Answer Relevancy | 0.1655 |
| Context Recall | 0.7000 |
| Context Precision | 0.8335 |
| **Hard Negative Fallback 정확도** | **70% (14/20)** |

**Answer Relevancy 0.17 분석**

원인은 두 계층으로 분리됩니다.

**CRAG 분기 로직 문제 (주요 원인):** `RERANKER_THRESHOLD = 0.5` 단일값에 의존하는 Grade 단계가 answerable 질문에서도 Fallback을 잘못 트리거합니다(False Negative). 특히 multi\_article·conditional 유형에서 정답 조문이 여러 청크에 분산되어 있으면 개별 점수가 임계값 미만으로 떨어집니다. QA 유형별 동적 임계값이나 앙상블 채점으로 해결 가능한 설계 문제입니다.

**7.8B 모델 생성 한계 (부차 원인):** exaone3.5:7.8b는 컨텍스트에 답이 있어도 간결하게 응답하지 못하고 조문 번호를 나열하거나 불필요한 단서를 붙이는 경향이 있습니다. RAGAS의 Answer Relevancy는 답변이 질문에 직접 응답하는 정도를 측정하므로, 이런 생성 패턴이 점수를 끌어내립니다.

Hard Negative Fallback 70%는 Answer Relevancy와 독립된 안전성 지표입니다. 답해서는 안 되는 질문 20개 중 14개를 올바르게 거부했습니다.

False Positive 6건: conditional·multi\_article 유형, `RERANKER_THRESHOLD` 경계(0.4~0.5) 문서 오판.

## 배운 것들

평가 방법론이 모델 선택만큼 중요하다. 0.95는 측정 도구의 결함이었고, 0.59가 진짜였습니다.

이미 계산된 것을 다시 계산하는 구간을 찾아서 제거하는 게 알고리즘 개선보다 빠른 레이턴시 개선으로 이어졌습니다. 시스템 경계를 따라가며 중복 연산을 찾으면 됩니다.

임계값은 한 번에 하나씩 조정해야 합니다. `MIN_RELEVANT_DOCS`와 `RERANKER_THRESHOLD`를 동시에 바꾸면 Hard Negative가 통과합니다. 파라미터 역할을 명확히 분리해야 합니다.

LCEL은 제거했습니다. 프레임워크 추상화가 디버깅을 어렵게 만들었고 성능 차이도 없었습니다. 직접 구현이 더 낫습니다.

## 다음 단계

- **전 Phase v2 재측정:** 현재 Phase 2~6은 v1(6개 샘플), Phase 7만 v2(100개)라 직접 비교가 불가합니다.
- **Grade 단계 정밀화:** QA 유형별 동적 임계값으로 False Negative를 줄여 Answer Relevancy를 개선합니다.
- **데모 앱:** FastAPI + Phase 선택 UI, 파이프라인 시각화.

## 설치 및 실행

```bash
# 1. 의존성
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 환경 변수
cp .env.example .env          # OLLAMA_HOST, OLLAMA_MODEL 설정

# 3. 모델
ollama pull exaone3.5:7.8b

# 4. 실행
.venv/bin/python src/crag_rag.py

# 5. 모니터링
.venv/bin/python src/monitor.py --query "개인정보 수집 시 고지 의무는?" --phase crag
.venv/bin/python src/monitor.py --dashboard

# 6. 벤치마크
.venv/bin/python src/eval_benchmark_v2.py --limit 20
```

첫 쿼리는 모델 로딩으로 40초까지 소요됩니다. 이후 응답은 15~30초 수준입니다.

**Stack:** exaone3.5:7.8b · BGE-M3 · BGE-Reranker-v2-m3 · Kiwi · Qdrant · LangGraph · RAGAS 0.4.3  
**Infra:** Mac Mini M2 16GB · Ollama · k3s + ArgoCD
