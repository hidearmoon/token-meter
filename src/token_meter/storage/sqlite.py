"""SQLite storage backend for TokenMeter.

Uses WAL journal mode for concurrent read/write safety.
All writes are serialised through a threading.Lock so multiple threads
that call the same patched SDK methods don't race on the connection.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import UsageRecord
from .base import BaseStorage

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS usage (
    id           TEXT     PRIMARY KEY,
    timestamp    TEXT     NOT NULL,
    provider     TEXT     NOT NULL,
    model        TEXT     NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    input_cost   REAL     NOT NULL DEFAULT 0,
    output_cost  REAL     NOT NULL DEFAULT 0,
    total_cost   REAL     NOT NULL DEFAULT 0,
    latency_ms   REAL     NOT NULL DEFAULT 0,
    project      TEXT     NOT NULL DEFAULT 'default',
    is_stream    INTEGER  NOT NULL DEFAULT 0,
    metadata     TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON usage (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_provider  ON usage (provider);",
    "CREATE INDEX IF NOT EXISTS idx_model     ON usage (model);",
    "CREATE INDEX IF NOT EXISTS idx_project   ON usage (project);",
    "CREATE INDEX IF NOT EXISTS idx_proj_ts   ON usage (project, timestamp);",
]

_INSERT = """
INSERT INTO usage (
    id, timestamp, provider, model,
    input_tokens, output_tokens, total_tokens,
    input_cost, output_cost, total_cost,
    latency_ms, project, is_stream, metadata
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def _ts(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class SQLiteStorage(BaseStorage):
    """Thread-safe SQLite-backed storage using WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Connection management                                                #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=10,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                cur.execute(idx_sql)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # BaseStorage interface                                                #
    # ------------------------------------------------------------------ #

    def save(self, record: UsageRecord) -> None:
        with self._lock:
            try:
                self._conn.execute(_INSERT, record.to_row())
                self._conn.commit()
            except sqlite3.Error:
                logger.exception("token-meter: failed to save usage record")

    def query(
        self,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        clauses: list[str] = []
        params: list[Any] = []

        if project:
            clauses.append("project = ?")
            params.append(project)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if start:
            clauses.append("timestamp >= ?")
            params.append(_ts(start))
        if end:
            clauses.append("timestamp <= ?")
            params.append(_ts(end))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM usage {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()

        return [UsageRecord.from_row(r) for r in rows]

    def aggregate(
        self,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []

        if project:
            clauses.append("project = ?")
            params.append(project)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if start:
            clauses.append("timestamp >= ?")
            params.append(_ts(start))
        if end:
            clauses.append("timestamp <= ?")
            params.append(_ts(end))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
        SELECT
            COUNT(*)           AS call_count,
            SUM(input_tokens)  AS total_input_tokens,
            SUM(output_tokens) AS total_output_tokens,
            SUM(total_tokens)  AS total_tokens,
            SUM(total_cost)    AS total_cost,
            AVG(latency_ms)    AS avg_latency_ms,
            MIN(timestamp)     AS first_call,
            MAX(timestamp)     AS last_call
        FROM usage {where}
        """

        with self._lock:
            cur = self._conn.execute(sql, params)
            row = cur.fetchone()

        if row is None or row[0] == 0:
            return {
                "call_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
                "first_call": None,
                "last_call": None,
            }

        return {
            "call_count": row[0],
            "total_input_tokens": row[1] or 0,
            "total_output_tokens": row[2] or 0,
            "total_tokens": row[3] or 0,
            "total_cost": round(row[4] or 0.0, 6),
            "avg_latency_ms": round(row[5] or 0.0, 2),
            "first_call": row[6],
            "last_call": row[7],
        }

    def aggregate_by_model(
        self,
        project: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Group costs by provider + model."""
        clauses: list[str] = []
        params: list[Any] = []

        if project:
            clauses.append("project = ?")
            params.append(project)
        if start:
            clauses.append("timestamp >= ?")
            params.append(_ts(start))
        if end:
            clauses.append("timestamp <= ?")
            params.append(_ts(end))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
        SELECT
            provider,
            model,
            COUNT(*)           AS call_count,
            SUM(total_tokens)  AS total_tokens,
            SUM(total_cost)    AS total_cost
        FROM usage {where}
        GROUP BY provider, model
        ORDER BY total_cost DESC
        """

        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "provider": r[0],
                "model": r[1],
                "call_count": r[2],
                "total_tokens": r[3] or 0,
                "total_cost": round(r[4] or 0.0, 6),
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
