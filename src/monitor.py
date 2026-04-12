"""
Phase 9: 모니터링 & 지속적 개선

목표:
  - 실서비스 관점의 운영 가시성 확보
  - 쿼리/응답/레이턴시를 JSONL로 누적
  - 통계 대시보드 (rich 기반 CLI)
  - Data Flywheel 기반: 낮은 평가 쿼리를 자동 수집해 eval 셋 보강 후보로 활용

사용법:
  # 단일 쿼리 모니터링
  from monitor import monitored_query
  result = monitored_query("개인정보 수집 시 고지 의무는?", phase="crag")

  # 대시보드 실행
  .venv/bin/python src/monitor.py --dashboard

  # 통계 출력
  .venv/bin/python src/monitor.py --stats
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

LOG_PATH = Path("data/logs/queries.jsonl")

PhaseType = Literal["qdrant", "crag"]  # naive/advanced는 모니터링 래퍼 미지원

# 낮은 신뢰도 임계값 — 이 이하 응답은 review_queue에 쌓임
LOW_CONFIDENCE_KEYWORDS = ["찾을 수 없", "관련 조문", "없습니다", "불확실"]
SLOW_LATENCY_THRESHOLD = 15.0  # 초


# ── 로깅 ─────────────────────────────────────────────────

def log_query(
    question: str,
    answer: str,
    latency_sec: float,
    phase: PhaseType = "crag",
    fallback_used: bool = False,
    relevant_docs_count: int = 0,
    user_feedback: int | None = None,   # 1(좋음) / 0(나쁨) / None(미입력)
) -> dict:
    """쿼리 결과를 JSONL 로그에 기록하고 엔트리를 반환"""
    is_fallback = fallback_used or any(kw in answer for kw in LOW_CONFIDENCE_KEYWORDS)
    is_slow = latency_sec > SLOW_LATENCY_THRESHOLD

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer_preview": answer[:200],
        "latency_sec": round(latency_sec, 2),
        "phase": phase,
        "fallback_used": is_fallback,
        "relevant_docs_count": relevant_docs_count,
        "is_slow": is_slow,
        "user_feedback": user_feedback,
        "review_candidate": is_fallback or is_slow,  # Data Flywheel 후보
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


# ── 모니터링 래퍼 ─────────────────────────────────────────

def monitored_query(question: str, phase: PhaseType = "crag") -> dict:
    """기존 RAG 함수를 감싸서 자동 로깅"""
    if phase == "crag":
        from crag_rag import run_query
        t0 = time.time()
        result = run_query(question)
        latency = round(time.time() - t0, 2)
        answer = result.get("generation", "")
        fallback = result.get("fallback_used", False)
        docs = result.get("relevant_docs", [])

    elif phase == "qdrant":
        # NOTE: 단일 쿼리 데모 전제. 반복 호출 시 초기화를 외부로 분리 권장.
        from qdrant_rag import load_chunks, load_embed_model, build_bm25, hybrid_retrieve, generate
        from qdrant_client import QdrantClient
        chunks = load_chunks()
        model = load_embed_model()
        bm25, chunk_ids = build_bm25(chunks)
        qclient = QdrantClient(path=str(Path("./qdrant_data")))
        t0 = time.time()
        docs = hybrid_retrieve(question, model, chunks, bm25, chunk_ids)
        gen = generate(question, docs)
        latency = round(time.time() - t0, 2)
        answer = gen.get("answer", "")
        fallback = False

    else:
        raise ValueError(f"지원하지 않는 phase: {phase}")

    entry = log_query(
        question=question,
        answer=answer,
        latency_sec=latency,
        phase=phase,
        fallback_used=fallback,
        relevant_docs_count=len(docs) if isinstance(docs, list) else 0,
    )
    return {**entry, "answer": answer}


# ── 통계 집계 ────────────────────────────────────────────

def load_logs() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute_stats(logs: list[dict]) -> dict:
    if not logs:
        return {}

    total = len(logs)
    fallbacks = sum(1 for l in logs if l.get("fallback_used"))
    slow = sum(1 for l in logs if l.get("is_slow"))
    review = sum(1 for l in logs if l.get("review_candidate"))
    avg_latency = sum(l["latency_sec"] for l in logs) / total
    by_phase: dict[str, int] = {}
    for l in logs:
        by_phase[l.get("phase", "unknown")] = by_phase.get(l.get("phase", "unknown"), 0) + 1

    feedbacks = [l["user_feedback"] for l in logs if l.get("user_feedback") is not None]
    satisfaction = sum(feedbacks) / len(feedbacks) if feedbacks else None

    return {
        "total_queries": total,
        "fallback_rate": round(fallbacks / total, 3),
        "slow_rate": round(slow / total, 3),
        "avg_latency_sec": round(avg_latency, 2),
        "review_candidates": review,
        "by_phase": by_phase,
        "satisfaction": round(satisfaction, 3) if satisfaction is not None else None,
    }


# ── CLI 대시보드 ──────────────────────────────────────────

def print_dashboard():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
    except ImportError:
        print("[오류] rich 패키지가 필요합니다: pip install rich")
        return

    console = Console()
    logs = load_logs()
    stats = compute_stats(logs)

    if not logs:
        console.print("[yellow]로그 없음. 쿼리를 먼저 실행하세요.[/yellow]")
        return

    # 통계 패널
    stat_lines = [
        f"총 쿼리: [bold]{stats['total_queries']}[/bold]",
        f"평균 레이턴시: [bold]{stats['avg_latency_sec']}s[/bold]",
        f"Fallback 비율: [bold red]{stats['fallback_rate']:.1%}[/bold red]",
        f"느린 쿼리(>{SLOW_LATENCY_THRESHOLD}s): [yellow]{stats['slow_rate']:.1%}[/yellow]",
        f"리뷰 후보: [bold magenta]{stats['review_candidates']}건[/bold magenta]",
        f"만족도: [bold]{('N/A' if stats['satisfaction'] is None else '{:.1%}'.format(stats['satisfaction']))}[/bold]",
    ]
    by_phase_str = " | ".join(f"{k}: {v}" for k, v in stats["by_phase"].items())
    stat_lines.append(f"Phase별: {by_phase_str}")

    console.print(Panel("\n".join(stat_lines), title="📊 RAG 운영 현황", border_style="blue"))

    # 최근 쿼리 테이블
    table = Table(title="최근 쿼리 (최신 10개)", box=box.ROUNDED, show_lines=True)
    table.add_column("시각", style="dim", width=12)
    table.add_column("질문", width=35)
    table.add_column("Phase", width=8)
    table.add_column("레이턴시", justify="right", width=8)
    table.add_column("Fallback", justify="center", width=8)
    table.add_column("리뷰", justify="center", width=6)

    for entry in logs[-10:][::-1]:
        ts = entry["ts"][11:16]  # HH:MM
        q = entry["question"][:33] + "…" if len(entry["question"]) > 33 else entry["question"]
        latency = f"{entry['latency_sec']}s"
        fallback = "⚠️" if entry.get("fallback_used") else "✅"
        review = "🔍" if entry.get("review_candidate") else ""
        table.add_row(ts, q, entry.get("phase", "-"), latency, fallback, review)

    console.print(table)

    # Data Flywheel 후보
    candidates = [l for l in logs if l.get("review_candidate")]
    if candidates:
        console.print(f"\n[bold magenta]🔄 Data Flywheel 후보 {len(candidates)}건[/bold magenta]")
        console.print("[dim]아래 쿼리는 eval_qa_v2 보강 후보입니다. data/logs/queries.jsonl 참조[/dim]")
        for c in candidates[-3:]:
            console.print(f"  • {c['question'][:60]}")


def print_stats():
    logs = load_logs()
    stats = compute_stats(logs)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


# ── 진입점 ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9: RAG 모니터링")
    parser.add_argument("--dashboard", action="store_true", help="대시보드 출력")
    parser.add_argument("--stats", action="store_true", help="통계 JSON 출력")
    parser.add_argument("--query", help="단일 쿼리 실행 + 로깅")
    parser.add_argument("--phase", default="crag", choices=["qdrant", "crag"], help="사용할 RAG phase")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    if args.dashboard:
        print_dashboard()
    elif args.stats:
        print_stats()
    elif args.query:
        result = monitored_query(args.query, phase=args.phase)
        print(f"\n답변: {result['answer']}")
        print(f"레이턴시: {result['latency_sec']}s | fallback: {result['fallback_used']}")
    else:
        parser.print_help()
