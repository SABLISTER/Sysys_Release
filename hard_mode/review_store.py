"""SQLite persistence for human review overlays."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                uid TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                sort_tier INTEGER,
                updated_at REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review (
                uid TEXT PRIMARY KEY,
                reviewer_label TEXT,
                axes_human TEXT,
                integration_verdict TEXT,
                citation_seed_flag INTEGER,
                comment TEXT,
                reviewed_at REAL
            )
            """
        )
        conn.commit()


def sync_from_corpus(db_path: Path, papers: list[dict[str, Any]]) -> None:
    """Upsert paper payloads from scored corpus."""
    init_db(db_path)
    now = time.time()
    with _connect(db_path) as conn:
        for p in papers:
            uid = str(p.get("hard_mode_uid") or "")
            if not uid:
                continue
            tier = int((p.get("derived") or {}).get("priority_tier") or 99)
            conn.execute(
                """
                INSERT INTO papers (uid, payload, sort_tier, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(uid) DO UPDATE SET
                    payload=excluded.payload,
                    sort_tier=excluded.sort_tier,
                    updated_at=excluded.updated_at
                """,
                (uid, json.dumps(p, ensure_ascii=False, default=str), tier, now),
            )
        conn.commit()


def save_review(
    db_path: Path,
    uid: str,
    *,
    reviewer_label: str,
    axes_human: list[str],
    integration_verdict: str,
    citation_seed_flag: bool,
    comment: str,
) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO review (uid, reviewer_label, axes_human, integration_verdict,
                citation_seed_flag, comment, reviewed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                reviewer_label=excluded.reviewer_label,
                axes_human=excluded.axes_human,
                integration_verdict=excluded.integration_verdict,
                citation_seed_flag=excluded.citation_seed_flag,
                comment=excluded.comment,
                reviewed_at=excluded.reviewed_at
            """,
            (
                uid,
                reviewer_label,
                json.dumps(axes_human),
                integration_verdict,
                1 if citation_seed_flag else 0,
                comment,
                time.time(),
            ),
        )
        conn.commit()


def load_queue(db_path: Path, *, max_tier: int | None = None) -> list[dict[str, Any]]:
    """Papers ordered by priority tier then uid."""
    if not db_path.exists():
        return []
    init_db(db_path)
    with _connect(db_path) as conn:
        if max_tier is None:
            rows = conn.execute(
                """
                SELECT p.uid, p.payload, p.sort_tier,
                       r.reviewer_label, r.axes_human, r.integration_verdict,
                       r.citation_seed_flag, r.comment, r.reviewed_at
                FROM papers p
                LEFT JOIN review r ON p.uid = r.uid
                ORDER BY p.sort_tier ASC, p.uid
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT p.uid, p.payload, p.sort_tier,
                       r.reviewer_label, r.axes_human, r.integration_verdict,
                       r.citation_seed_flag, r.comment, r.reviewed_at
                FROM papers p
                LEFT JOIN review r ON p.uid = r.uid
                WHERE p.sort_tier <= ?
                ORDER BY p.sort_tier ASC, p.uid
                """,
                (int(max_tier),),
            ).fetchall()
    out = []
    for row in rows:
        payload = json.loads(row["payload"])
        human = payload.get("human") or {}
        if row["reviewer_label"]:
            human.update(
                {
                    "reviewer_label": row["reviewer_label"],
                    "axes_human": json.loads(row["axes_human"] or "[]"),
                    "integration_verdict": row["integration_verdict"] or "",
                    "citation_seed_flag": bool(row["citation_seed_flag"]),
                    "comment": row["comment"] or "",
                    "reviewed_at": row["reviewed_at"],
                }
            )
        payload["human"] = human
        out.append(payload)
    return out
