from __future__ import annotations

import json
import sqlite3

import psai  # noqa: F401


def test_json_dumps_with_lone_surrogate_is_safe() -> None:
    text = "safe\udc81text"
    payload = {"problem": text}

    dumped = json.dumps(payload, ensure_ascii=False)

    assert "\udc81" not in dumped
    assert "\ufffd" in dumped
    dumped.encode("utf-8")


def test_sqlite_accepts_strings_with_lone_surrogates() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE demo(value TEXT NOT NULL)")
    conn.execute("INSERT INTO demo(value) VALUES (?)", ("safe\udc81text",))

    row = conn.execute("SELECT value FROM demo").fetchone()
    assert row is not None
    assert "\udc81" not in row[0]
    assert "\ufffd" in row[0]
