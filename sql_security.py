"""Deterministic, defence-in-depth SQLite policy for synthetic operations data.

The LLM's SQL is untrusted. Validation is advisory: SQLite query_only plus
an authorizer provide enforcement at the database boundary. This module does
not execute any LLM calls and can be tested offline.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Iterable

import pandas as pd

TABLE_NAME = "operations"
MAX_SQL_CHARS = 2_000
MAX_RESULT_ROWS = 200
SQLITE_PROGRESS_STEP = 1_000
MAX_PROGRESS_CALLS = 2_000  # ~2 million SQLite virtual machine instructions

# A small allowlist appropriate to our demo's counts, grouping and time metrics.
ALLOWED_SQL_FUNCTIONS = frozenset({
    "abs", "avg", "coalesce", "count", "date", "datetime", "ifnull",
    "instr", "julianday", "length", "like", "lower", "ltrim", "max",
    "min", "nullif", "round", "rtrim", "strftime", "substr", "substring",
    "sum", "time", "total", "trim", "upper", "replace", "group_concat",
})


class QueryBlocked(ValueError):
    """An untrusted SQL query violates the execution policy."""


def validate_sql(sql: str) -> str:
    """Allow a single top-level SELECT; reject comments and multi-statements.

    SQLite's authorizer remains the primary security boundary. This stricter
    text validation stops common attempts without invoking the database.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise QueryBlocked("empty_sql")
    statement = sql.strip()
    if len(statement) > MAX_SQL_CHARS:
        raise QueryBlocked("sql_too_long")
    if not re.match(r"(?is)^select\b", statement):
        raise QueryBlocked("non_select")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise QueryBlocked("sql_comments")
    # A single optional terminal ';' is harmless; any other ';' is rejected.
    if ";" in statement.rstrip(";") or statement.count(";") > 1:
        raise QueryBlocked("multiple_statements")
    return statement.rstrip(";").strip()


def enforce_read_only(conn: sqlite3.Connection) -> None:
    """Lock the populated in-memory database before any untrusted SQL executes."""
    conn.enable_load_extension(False)
    conn.execute("PRAGMA trusted_schema=OFF")
    conn.execute("PRAGMA query_only=ON")

    def authorizer(action: int, arg1: str | None, arg2: str | None,
                   db_name: str | None, source: str | None) -> int:
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            # In particular, block sqlite_master/sqlite_schema and temp tables.
            if db_name in ("main", None) and arg1 == TABLE_NAME:
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            name = (arg2 or arg1 or "").lower()
            if name in ALLOWED_SQL_FUNCTIONS:
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorizer)


def execute_read_only(conn: sqlite3.Connection, candidate_sql: str) -> pd.DataFrame:
    """Execute permitted SQL with bounded work and bounded output.

    The caller owns and closes `conn` in a `finally` block. No input values are
    interpolated by this function; it accepts only generated SELECT statements.
    """
    statement = validate_sql(candidate_sql)
    calls = 0

    def progress() -> int:
        nonlocal calls
        calls += 1
        return int(calls > MAX_PROGRESS_CALLS)

    enforce_read_only(conn)
    conn.set_progress_handler(progress, SQLITE_PROGRESS_STEP)
    try:
        try:
            cursor = conn.execute(statement)
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
            if len(rows) > MAX_RESULT_ROWS:
                raise QueryBlocked("result_too_large")
            columns = [d[0] for d in cursor.description or ()]
            return pd.DataFrame.from_records(rows, columns=columns)
        except sqlite3.Error as exc:
            raise QueryBlocked("sqlite_denied_or_invalid") from exc
    finally:
        conn.set_progress_handler(None, 0)
