"""NL-to-SQL tool for the synthetic operational dataset.

The LLM output is NOT trusted: SQL must pass deterministic checks and an
SQLite authorizer before it can access the local in-memory demo data.
"""
import sqlite3
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool

from observability import log_event, record_model_call, prompt_version
from sql_security import QueryBlocked, execute_read_only, TABLE_NAME
from prompt_boundaries import sql_messages, SQL_SYSTEM_PROMPT, SQL_ANSWER_SYSTEM_PROMPT

load_dotenv()
client = OpenAI()

CSV_PATH = "data/operational_data.csv"
SQL_PROMPT_VERSION = prompt_version(SQL_SYSTEM_PROMPT, "sql-generate")
SQL_ANSWER_PROMPT_VERSION = prompt_version(SQL_ANSWER_SYSTEM_PROMPT, "sql-answer")



def _get_connection():
    """Populate demo data while writable; the safety layer locks it afterward."""
    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(":memory:")
    df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
    return conn


def _get_schema():
    df = pd.read_csv(CSV_PATH)
    cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    return f"Table '{TABLE_NAME}' with columns: {cols}"




@tool
def query_operational_data(question: str) -> str:
    """Query synthetic operational data for asset status, counts, averages,
    anomalies, dates and other statistics. Read-only SELECT queries only."""
    started = time.perf_counter()
    generate_started = time.perf_counter()
    sql_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=sql_messages(question, _get_schema()), temperature=0,
    )
    record_model_call(model="gpt-4o-mini", operation="sql_generate",
        prompt_version=SQL_PROMPT_VERSION, response=sql_response,
        latency_ms=(time.perf_counter() - generate_started) * 1000)
    sql = sql_response.choices[0].message.content or ""
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    log_event("sql_generated", {"question": question, "sql": sql})

    conn = None
    try:
        # Parse/reject BEFORE creating the DB; enforce again inside execution.
        from sql_security import validate_sql
        validate_sql(sql)
        conn = _get_connection()
        result = execute_read_only(conn, sql)
    except QueryBlocked as exc:
        log_event("sql_blocked", {"sql": sql, "reason": str(exc)})
        return "I can only run safe, read-only data queries."
    except Exception as exc:
        log_event("sql_error", {"sql": sql, "error_type": type(exc).__name__})
        return "I couldn't run that data query safely."
    finally:
        if conn is not None:
            conn.close()

    preview = result.head(10).to_dict(orient="records")
    log_event("sql_result", {"sql": sql, "rows": len(result), "preview": preview})

    answer_prompt = (
        f"Question: {question}\n\nSQL: {sql}\n\n"
        f"Result:\n{result.to_string(index=False)}\n\n"
        "Answer in plain English based on this result. Be concise."
    )
    answer_started = time.perf_counter()
    answer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SQL_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": answer_prompt},
        ],
        temperature=0,
    )
    record_model_call(model="gpt-4o-mini", operation="sql_answer",
        prompt_version=SQL_ANSWER_PROMPT_VERSION, response=answer_response,
        latency_ms=(time.perf_counter() - answer_started) * 1000)
    answer = answer_response.choices[0].message.content
    log_event("sql_answer", {"answer": answer,
          "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
    return answer


if __name__ == "__main__":
    print(query_operational_data.invoke({"question": "How many assets had a FAULT status?"}))
