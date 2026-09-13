"""NL-to-SQL tool for the synthetic operational dataset."""
import sqlite3
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool

from observability import log_event

load_dotenv()
client = OpenAI()

CSV_PATH = "data/operational_data.csv"
TABLE_NAME = "operations"


def _get_connection():
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
    """Query the operational DATA table for counts, averages, statuses, assets,
    power output, anomalies, or other questions answerable from the table.
    Only read-only SELECT queries are permitted."""
    started = time.perf_counter()
    schema = _get_schema()
    sql_prompt = (
        f"You are a SQL expert. Given this table:\n{schema}\n\n"
        f"Write a single SQLite SELECT query that answers:\n{question}\n\n"
        "Return ONLY the SQL query, no explanation, no markdown."
    )
    sql = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0,
    ).choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

    log_event("sql_generated", {"question": question, "sql": sql})

    if not sql.lower().startswith("select"):
        answer = "I can only run read-only data queries."
        log_event("sql_blocked", {"sql": sql, "reason": "non_select"})
        return answer

    try:
        conn = _get_connection()
        result = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception as exc:
        log_event("sql_error", {"sql": sql, "error": str(exc)})
        return f"I couldn't run that query. ({exc})"

    preview = result.head(10).to_dict(orient="records")
    log_event(
        "sql_result",
        {"sql": sql, "rows": len(result), "preview": preview},
    )

    answer_prompt = (
        f"Question: {question}\n\nSQL: {sql}\n\n"
        f"Result:\n{result.to_string(index=False)}\n\n"
        "Answer in plain English based on this result. Be concise."
    )
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
    ).choices[0].message.content
    log_event(
        "sql_answer",
        {"answer": answer, "latency_ms": round((time.perf_counter() - started) * 1000, 1)},
    )
    return answer


if __name__ == "__main__":
    print(query_operational_data.invoke({"question": "How many assets had a FAULT status?"}))
