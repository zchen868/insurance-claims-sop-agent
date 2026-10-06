"""Durable state in SQLite: conversations (so a restart doesn't drop live chats),
failed verification attempts per policyholder (lockout across chats), and
consent requests (approved or declined from a link outside the chat).

SQLite keeps the demo dependency-free; the same three tables map directly onto
Postgres or Redis for a multi-instance deployment."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, kind TEXT NOT NULL, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS verify_failures (party_id TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS vf_party ON verify_failures (party_id, ts);
CREATE TABLE IF NOT EXISTS consents (token TEXT PRIMARY KEY, session_id TEXT, party_id TEXT, rep_name TEXT,
    status TEXT NOT NULL, created REAL NOT NULL, decided REAL);
"""


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("SOP_DB_PATH", str(ROOT / "data" / "sop.db"))
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def _q(self, sql, args=(), one=False):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.fetchone() if one else cur.fetchall()

    # ---------- conversations ----------
    def save_session(self, sid: str, kind: str, data: dict):
        self._q("INSERT INTO sessions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
                (sid, kind, json.dumps(data, default=str), time.time()))

    def load_session(self, sid: str, kind: str) -> dict | None:
        row = self._q("SELECT data FROM sessions WHERE id=? AND kind=?", (sid, kind), one=True)
        return json.loads(row[0]) if row else None

    def purge_sessions(self, older_than_s: float):
        self._q("DELETE FROM sessions WHERE updated < ?", (time.time() - older_than_s,))

    # ---------- verification lockout ----------
    def record_failure(self, party_id: str):
        self._q("INSERT INTO verify_failures VALUES (?,?)", (party_id, time.time()))

    def recent_failures(self, party_id: str, window_s: float) -> int:
        return self._q("SELECT COUNT(*) FROM verify_failures WHERE party_id=? AND ts>?",
                       (party_id, time.time() - window_s), one=True)[0]

    def clear_failures(self, party_id: str):
        self._q("DELETE FROM verify_failures WHERE party_id=?", (party_id,))

    # ---------- consent requests ----------
    def create_consent(self, token, session_id, party_id, rep_name):
        self._q("INSERT INTO consents VALUES (?,?,?,?,?,?,NULL)", (token, session_id, party_id, rep_name, "pending", time.time()))

    def get_consent(self, token) -> dict | None:
        row = self._q("SELECT token, session_id, party_id, rep_name, status, created, decided FROM consents WHERE token=?",
                      (token,), one=True)
        return dict(zip(("token", "session_id", "party_id", "rep_name", "status", "created", "decided"), row)) if row else None

    def decide_consent(self, token, status) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE consents SET status=?, decided=? WHERE token=? AND status='pending'",
                                   (status, time.time(), token))
            self._db.commit()
            return cur.rowcount == 1


_STORE: Store | None = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE
