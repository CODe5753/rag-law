"""
Phase 4: LangChain LCEL RAG — naive_rag.py의 LangChain 리팩토링

목적:
  - 동일 파이프라인을 LCEL(LangChain Expression Language)로 표현
  - 점수 변화 없어야 정상 (같은 임베딩 + 같은 ChromaDB + 같은 LLM)
  - "LangChain이 무엇을 대신해주는가"를 직접 비교로 확인

비교 포인트 (naive_rag.py → langchain_rag.py):
  retrieve()    → Chroma retriever (as_retriever())
  generate()    → ChatOllama + PromptTemplate + StrOutputParser
  ask()         → LCEL chain: retriever | format_docs | prompt | llm | parser

⚠️ 버전 주의: LangChain 1.2.x (실제 설치된 버전)
  - langchain-ollama: ChatOllama from langchain_ollama (langchain_community deprecated)
  - Chroma: langchain_chroma (별도 패키지)가 없으면 langchain_community.vectorstores 사용
"""

import json
import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_data"))
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "law_chunks_v1"
TOP_K = 5

SYSTEM_PROMPT = """당신은 한국 법령 전문가입니다. 아래 법령 조문을 참고하여 질문에 정확하게 답변하세요.
법령에 없는 내용은 '관련 조문을 찾을 수 없습니다'라고 답하세요. 추측하지 마세요."""


def build_chain():
    """LCEL 체인 구성. 반환값은 invoke()로 호출 가능한 체인."""

    # 1. 임베딩 함수 (LangChain 래퍼로 bge-m3 사용)
    print(f"[임베딩] {EMBED_MODEL_NAME} 로드...")
    t0 = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # MPS/CUDA 가능하면 변경
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"  완료: {time.time() - t0:.1f}s")

    # 2. ChromaDB 벡터스토어 (기존 인덱스 재사용)
    vectorstore = Chroma(
        client=chromadb.PersistentClient(path=str(CHROMA_PATH)),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # 3. LLM
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0,
    )

    # 4. 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "참고 법령:\n{context}\n\n질문: {question}"),
    ])

    # 5. LCEL 체인 조립
    # naive_rag.py의 ask()와 동일한 로직을 선언적으로 표현
    def format_docs(docs) -> str:
        return "\n\n".join(
            f"[{doc.metadata.get('law_name', '')} {doc.metadata.get('article_num', '')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_lcel(query: str, chain, retriever, verbose: bool = True) -> dict:
    """LCEL 체인으로 단일 쿼리 처리"""
    t0 = time.time()

    # 검색 (메타데이터 포함 확인용)
    docs = retriever.invoke(query)
    sources = [
        f"{d.metadata.get('law_name')} {d.metadata.get('article_num')}"
        for d in docs
    ]

    # 생성
    answer = chain.invoke(query)
    elapsed = round(time.time() - t0, 2)

    if verbose:
        print(f"\n질문: {query}")
        print(f"\n[검색된 조문 Top-{TOP_K}]")
        for i, s in enumerate(sources, 1):
            print(f"  {i}. {s}")
        print(f"\n[답변]\n{answer}")
        print(f"\n[지연시간] {elapsed}s")

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "latency_sec": elapsed,
    }


SAMPLE_QUERIES = [
    "개인정보 수집 시 반드시 동의를 받아야 하는가?",
    "근로자의 주당 최대 근로시간은 얼마인가?",
    "저작권 보호 기간은 얼마나 되는가?",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="단일 쿼리")
    parser.add_argument("--eval", action="store_true", help="평가 데이터셋으로 채점")
    args = parser.parse_args()

    chain, retriever = build_chain()

    if args.query:
        ask_lcel(args.query, chain, retriever)
    elif args.eval:
        # naive_rag.py와 동일한 평가 데이터셋으로 점수 비교
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluator import EVAL_QA, eval_faithfulness, eval_answer_relevancy, eval_context_recall, eval_context_precision

        scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "context_precision": []}
        t_total = time.time()

        for i, qa in enumerate(EVAL_QA):
            q = qa["question"]
            gt = qa["ground_truth"]
            print(f"\n[{i+1}/{len(EVAL_QA)}] {q[:50]}...")

            docs = retriever.invoke(q)
            contexts = [d.page_content for d in docs]
            answer = chain.invoke(q)

            faith = eval_faithfulness(answer, contexts)
            relev = eval_answer_relevancy(q, answer)
            recall = eval_context_recall(gt, contexts)
            prec = eval_context_precision(q, contexts)

            print(f"  faith={faith} relev={relev} recall={recall} prec={prec}")
            if faith is not None: scores["faithfulness"].append(faith)
            if relev is not None: scores["answer_relevancy"].append(relev)
            if recall is not None: scores["context_recall"].append(recall)
            if prec is not None: scores["context_precision"].append(prec)

        elapsed = round(time.time() - t_total, 1)

        def mean(lst): return round(sum(lst) / len(lst), 4) if lst else None
        result = {
            "phase": "Phase4_LangChain",
            "eval_time_sec": elapsed,
            "faithfulness": mean(scores["faithfulness"]),
            "answer_relevancy": mean(scores["answer_relevancy"]),
            "context_recall": mean(scores["context_recall"]),
            "context_precision": mean(scores["context_precision"]),
        }
        print(f"\n=== Phase 4 LangChain 점수 ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

        out = Path("data/processed/ragas_phase4.json")
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {out}")
    else:
        # 기본: 샘플 쿼리 1개
        ask_lcel(SAMPLE_QUERIES[0], chain, retriever)
