from __future__ import annotations

import json as _json
import sqlite3 as _sqlite3
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_APPLIED_SENTINEL = "_projectsolver_encoding_safety_applied"


def _replace_lone_surrogates(text: str) -> str:
    # Replace any unpaired surrogate code point with the Unicode replacement
    # character so UTF-8 encoding, SQLite bindings, and JSON serialization
    # cannot fail later in the pipeline.
    if not text:
        return text
    changed = False
    out_chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            out_chars.append("\uFFFD")
            changed = True
        else:
            out_chars.append(ch)
    return "".join(out_chars) if changed else text


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, str):
        return _replace_lone_surrogates(obj)
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, Mapping):
        return {
            _sanitize(key) if isinstance(key, str) else key: _sanitize(value)
            for key, value in obj.items()
        }
    if isinstance(obj, tuple):
        return tuple(_sanitize(item) for item in obj)
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    if isinstance(obj, set):
        return {_sanitize(item) for item in obj}
    if isinstance(obj, frozenset):
        return frozenset(_sanitize(item) for item in obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return type(obj)(_sanitize(item) for item in obj)
        except Exception:  # noqa: BLE001
            return [_sanitize(item) for item in obj]
    return obj


def _patch_json() -> None:
    if getattr(_json, _APPLIED_SENTINEL, False):
        return

    original_dumps = _json.dumps
    original_dump = _json.dump

    def dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
        return original_dumps(_sanitize(obj), *args, **kwargs)

    def dump(obj: Any, fp: Any, *args: Any, **kwargs: Any) -> None:
        return original_dump(_sanitize(obj), fp, *args, **kwargs)

    _json.dumps = dumps  # type: ignore[assignment]
    _json.dump = dump  # type: ignore[assignment]
    setattr(_json, _APPLIED_SENTINEL, True)


def _patch_stdout_stderr() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except Exception:  # noqa: BLE001
                pass


class _SafeConnection:
    def __init__(self, conn: _sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._conn, name, value)

    def execute(self, sql: str, parameters: Any = (), /):
        return self._conn.execute(sql, _sanitize(parameters))

    def executemany(self, sql: str, seq_of_parameters: Any, /):
        return self._conn.executemany(
            sql,
            (_sanitize(parameters) for parameters in seq_of_parameters),
        )

    def executescript(self, sql_script: str, /):
        return self._conn.executescript(sql_script)

    def cursor(self, *args: Any, **kwargs: Any):
        cursor = self._conn.cursor(*args, **kwargs)
        return _SafeCursor(cursor)


class _SafeCursor:
    def __init__(self, cursor: _sqlite3.Cursor) -> None:
        object.__setattr__(self, "_cursor", cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._cursor, name, value)

    def execute(self, sql: str, parameters: Any = (), /):
        return self._cursor.execute(sql, _sanitize(parameters))

    def executemany(self, sql: str, seq_of_parameters: Any, /):
        return self._cursor.executemany(
            sql,
            (_sanitize(parameters) for parameters in seq_of_parameters),
        )


def _patch_sqlite() -> None:
    if getattr(_sqlite3, _APPLIED_SENTINEL, False):
        return

    original_connect = _sqlite3.connect

    def connect(*args: Any, **kwargs: Any) -> _SafeConnection:
        return _SafeConnection(original_connect(*args, **kwargs))

    _sqlite3.connect = connect  # type: ignore[assignment]
    setattr(_sqlite3, _APPLIED_SENTINEL, True)


_patch_json()
_patch_stdout_stderr()
_patch_sqlite()
