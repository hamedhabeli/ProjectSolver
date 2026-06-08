from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class RepoStatus:
    repo_id: str
    title: str
    head: str
    created_at: str


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    repo_id: str
    parent_id: Optional[str]
    message: str
    payload: dict[str, Any]
    created_at: str


class StateStore:
    """
    Git-like state store on SQLite.

    Notes:
    - WAL mode + synchronous=NORMAL are enabled for better concurrency/perf.
    - Foreign keys are deferred so repo_create can insert the root node and repo
      in one transaction without FK breakage.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS repos (
              repo_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              head_node_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(head_node_id) REFERENCES nodes(node_id)
                DEFERRABLE INITIALLY DEFERRED
            );

            CREATE TABLE IF NOT EXISTS nodes (
              node_id TEXT PRIMARY KEY,
              repo_id TEXT NOT NULL,
              parent_id TEXT NULL,
              message TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,

              FOREIGN KEY(repo_id) REFERENCES repos(repo_id)
                ON DELETE CASCADE
                DEFERRABLE INITIALLY DEFERRED,

              FOREIGN KEY(parent_id) REFERENCES nodes(node_id)
                DEFERRABLE INITIALLY DEFERRED
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_repo_created_at
              ON nodes(repo_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_nodes_repo_parent
              ON nodes(repo_id, parent_id);
            """
        )
        self._conn.commit()

    def repo_create(self, title: str, initial_payload: Optional[dict[str, Any]] = None) -> RepoStatus:
        if initial_payload is None:
            initial_payload = {}

        repo_id = _new_id("R")
        node_id = _new_id("N")
        created_at = _utc_now_iso()
        payload_json = json.dumps(initial_payload, ensure_ascii=False, sort_keys=True)

        with self._lock:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO nodes(node_id, repo_id, parent_id, message, payload_json, created_at)
                VALUES (?, ?, NULL, ?, ?, ?)
                """,
                (node_id, repo_id, "root", payload_json, created_at),
            )
            self._conn.execute(
                """
                INSERT INTO repos(repo_id, title, head_node_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (repo_id, title, node_id, created_at),
            )
            self._conn.commit()

        return RepoStatus(repo_id=repo_id, title=title, head=node_id, created_at=created_at)

    def repo_status(self, repo_id: str) -> RepoStatus:
        with self._lock:
            row = self._conn.execute(
                "SELECT repo_id, title, head_node_id, created_at FROM repos WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"repo not found: {repo_id}")
        return RepoStatus(
            repo_id=row["repo_id"],
            title=row["title"],
            head=row["head_node_id"],
            created_at=row["created_at"],
        )

    def commit(self, repo_id: str, parent_id: str, message: str, payload: dict[str, Any]) -> NodeRecord:
        node_id = _new_id("N")
        created_at = _utc_now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        with self._lock:
            _ = self.repo_status(repo_id)

            prow = self._conn.execute(
                "SELECT node_id FROM nodes WHERE node_id = ? AND repo_id = ?",
                (parent_id, repo_id),
            ).fetchone()
            if prow is None:
                raise KeyError(f"parent node not found in repo: {parent_id}")

            self._conn.execute(
                """
                INSERT INTO nodes(node_id, repo_id, parent_id, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (node_id, repo_id, parent_id, message, payload_json, created_at),
            )
            self._conn.execute(
                "UPDATE repos SET head_node_id = ? WHERE repo_id = ?",
                (node_id, repo_id),
            )
            self._conn.commit()

        return self.get_node(repo_id, node_id)

    def checkout(self, repo_id: str, node_id: str) -> NodeRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT node_id FROM nodes WHERE node_id = ? AND repo_id = ?",
                (node_id, repo_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"node not found in repo: {node_id}")

            self._conn.execute(
                "UPDATE repos SET head_node_id = ? WHERE repo_id = ?",
                (node_id, repo_id),
            )
            self._conn.commit()

        return self.get_node(repo_id, node_id)

    def get_node(self, repo_id: str, node_id: str) -> NodeRecord:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT node_id, repo_id, parent_id, message, payload_json, created_at
                FROM nodes
                WHERE repo_id = ? AND node_id = ?
                """,
                (repo_id, node_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"node not found: {node_id}")

        payload = json.loads(row["payload_json"])
        return NodeRecord(
            node_id=row["node_id"],
            repo_id=row["repo_id"],
            parent_id=row["parent_id"],
            message=row["message"],
            payload=payload,
            created_at=row["created_at"],
        )

    def log(self, repo_id: str, from_node_id: Optional[str] = None, limit: int = 50) -> list[NodeRecord]:
        """
        Return commit ancestry using a recursive CTE in SQLite.

        Order: start node -> parent -> parent -> ...
        """
        if limit <= 0:
            return []

        with self._lock:
            if from_node_id is None:
                from_node_id = self.repo_status(repo_id).head

            exists = self._conn.execute(
                "SELECT 1 FROM nodes WHERE repo_id = ? AND node_id = ?",
                (repo_id, from_node_id),
            ).fetchone()
            if exists is None:
                raise KeyError(f"node not found: {from_node_id}")

            rows = self._conn.execute(
                """
                WITH RECURSIVE chain(
                    node_id, repo_id, parent_id, message, payload_json, created_at, depth
                ) AS (
                    SELECT
                        n.node_id, n.repo_id, n.parent_id, n.message, n.payload_json, n.created_at, 0
                    FROM nodes n
                    WHERE n.repo_id = ? AND n.node_id = ?

                    UNION ALL

                    SELECT
                        p.node_id, p.repo_id, p.parent_id, p.message, p.payload_json, p.created_at, chain.depth + 1
                    FROM nodes p
                    JOIN chain ON p.node_id = chain.parent_id
                    WHERE chain.depth + 1 < ?
                )
                SELECT node_id, repo_id, parent_id, message, payload_json, created_at
                FROM chain
                ORDER BY depth ASC
                """,
                (repo_id, from_node_id, limit),
            ).fetchall()

        out: list[NodeRecord] = []
        for row in rows:
            out.append(
                NodeRecord(
                    node_id=row["node_id"],
                    repo_id=row["repo_id"],
                    parent_id=row["parent_id"],
                    message=row["message"],
                    payload=json.loads(row["payload_json"]),
                    created_at=row["created_at"],
                )
            )
        return out
