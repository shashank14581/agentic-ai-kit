"""
LongTermMemory: SQLite-backed memory that persists across sessions.

Agents can log decisions, read prior entries, and search by keyword.

Usage::

    mem = LongTermMemory("agent_memory.db")
    mem.log("Decided to buy AAPL based on positive earnings outlook.")
    for entry in mem.search("AAPL"):
        print(entry)
"""

from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class LongTermMemory:
    """Persistent key-value + full-text log backed by SQLite.

    Args:
        db_path: Path to the SQLite database file.
                 Use ``":memory:"`` for an in-process store (not persistent).
    """

    def __init__(self, db_path: str | Path = "long_term_memory.db"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                agent     TEXT,
                content   TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, content: str, agent: str = "unknown") -> int:
        """Append an entry and return its row id."""
        ts = datetime.now(tz=timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO memory (agent, content, timestamp) VALUES (?, ?, ?)",
            (agent, content, ts),
        )
        self._conn.commit()
        return cur.lastrowid

    def search(self, keyword: str, agent: str | None = None, limit: int = 20) -> list[dict]:
        """Return entries whose content contains ``keyword``."""
        q = "SELECT id, agent, content, timestamp FROM memory WHERE content LIKE ?"
        args: list = [f"%{keyword}%"]
        if agent:
            q += " AND agent = ?"
            args.append(agent)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(q, args).fetchall()
        return [
            {"id": r[0], "agent": r[1], "content": r[2], "timestamp": r[3]}
            for r in rows
        ]

    def recent(self, n: int = 10, agent: str | None = None) -> list[dict]:
        """Return the most recent ``n`` entries."""
        q = "SELECT id, agent, content, timestamp FROM memory"
        args: list = []
        if agent:
            q += " WHERE agent = ?"
            args.append(agent)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(n)
        rows = self._conn.execute(q, args).fetchall()
        return [
            {"id": r[0], "agent": r[1], "content": r[2], "timestamp": r[3]}
            for r in rows
        ]

    def clear(self, agent: str | None = None) -> int:
        """Delete entries. Returns the number of rows deleted."""
        if agent:
            cur = self._conn.execute("DELETE FROM memory WHERE agent = ?", (agent,))
        else:
            cur = self._conn.execute("DELETE FROM memory")
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        count = self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        return f"<LongTermMemory db={self.db_path!r} entries={count}>"
