"""
세션 및 메시지 저장소 — SQLite 기반.
"""

from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path("data/sessions.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at INTEGER,
                title TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                retrieved_docs TEXT,
                ts INTEGER
            );
        """)


def create_session() -> str:
    sid = str(uuid.uuid4())
    ts = int(time.time())
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (id, created_at, title) VALUES (?, ?, ?)",
            (sid, ts, None),
        )
    return sid


def add_message(
    session_id: str,
    role: str,
    content: str,
    retrieved_docs: list[dict] | None = None,
) -> None:
    ts = int(time.time())
    docs_json = json.dumps(retrieved_docs, ensure_ascii=False) if retrieved_docs else None
    with _conn() as con:
        con.execute(
            "INSERT INTO messages (session_id, role, content, retrieved_docs, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, docs_json, ts),
        )


def get_history(session_id: str, last_n: int = 3) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, last_n * 2),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_sessions(limit: int = 20) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, created_at, title FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
