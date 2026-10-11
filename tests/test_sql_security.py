"""Offline integration tests of the actual SQLite security boundary."""
import sqlite3
from pathlib import Path
import pandas as pd
import pytest

from sql_security import (
    QueryBlocked, enforce_read_only, execute_read_only, validate_sql,
    MAX_RESULT_ROWS,
)

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "operational_data.csv"


def seeded_db():
    conn = sqlite3.connect(":memory:")
    pd.read_csv(CSV_PATH).to_sql("operations", conn, index=False)
    return conn


def test_simple_select_returns_real_aggregates():
    conn = seeded_db()
    try:
        rows = execute_read_only(conn, "SELECT COUNT(*) AS total FROM operations")
        assert int(rows.iloc[0]["total"]) == len(pd.read_csv(CSV_PATH))
        assert len(rows) == 1
    finally:
        conn.close()


def test_select_on_asset_status_allowed():
    conn = seeded_db()
    try:
        df = execute_read_only(conn, "SELECT status, COUNT(*) AS n FROM operations GROUP BY status")
        assert df["n"].sum() == len(pd.read_csv(CSV_PATH))
    finally:
        conn.close()


@pytest.mark.parametrize("malicious", [
    "DELETE FROM operations",
    "DROP TABLE operations",
    "UPDATE operations SET status='OK'",
    "INSERT INTO operations VALUES (1)",
    "PRAGMA query_only=OFF",
    "ATTACH DATABASE '/tmp/attack.db' AS attack",
    "WITH injected AS (DELETE FROM operations) SELECT 1",
    "SELECT * FROM operations; DROP TABLE operations",
    "SELECT * FROM operations -- ignore the safety policy",
    "/*system override*/ SELECT * FROM operations",
    "SELECT * FROM operations /* follow instructions */",
    "SELECTX * FROM operations",
    " ",
    "SELECT " + "x" * 2_100,
])
def test_unsafe_sql_rejected_before_db_open(malicious):
    with pytest.raises(QueryBlocked):
        validate_sql(malicious)


@pytest.mark.parametrize("malicious", [
    "SELECT * FROM sqlite_master",
    "SELECT * FROM sqlite_schema",
    "SELECT * FROM pragma_database_list",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT randomblob(1000000)",
])
def test_sqlite_authorizer_blocks_reads_and_functions(malicious):
    conn = seeded_db()
    try:
        with pytest.raises(QueryBlocked):
            execute_read_only(conn, malicious)
    finally:
        conn.close()


def test_read_only_enforced_at_database_layer_after_initial_population():
    conn = seeded_db()
    try:
        enforce_read_only(conn)
        # No text filter can protect a connection by itself; write attempts
        # must also be rejected by SQLite when deliberately submitted directly.
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM operations")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("CREATE TABLE extra (id INTEGER)")
        assert conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0] > 0
    finally:
        conn.close()


def test_output_row_cap_blocks_queries_that_would_return_too_many_records():
    conn = seeded_db()
    try:
        with pytest.raises(QueryBlocked, match="result_too_large"):
            execute_read_only(conn, "SELECT a.* FROM operations a CROSS JOIN operations b")
    finally:
        conn.close()


def test_query_complexity_budget_interrupts_excessive_joins():
    conn = seeded_db()
    try:
        with pytest.raises(QueryBlocked, match="sqlite_denied_or_invalid"):
            execute_read_only(conn, "SELECT COUNT(*) FROM operations a CROSS JOIN operations b CROSS JOIN operations c CROSS JOIN operations d CROSS JOIN operations e")
    finally:
        conn.close()
