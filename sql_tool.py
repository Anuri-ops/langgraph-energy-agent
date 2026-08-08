"""
sql_tool.py — the NL-to-SQL engine wrapped as a LangGraph TOOL.
The @tool docstring tells the agent to use this for questions about the
operational DATA (counts, averages, statuses, specific assets, anomalies).
Reads data/operational_data.csv (copy it in from energy-rag).
"""
import os
import sqlite3
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    """Query the operational DATA table for figures about assets: counts, averages,
    statuses (NORMAL/FAULT/CURTAILED), power output, anomalies, or which assets match
    a condition. Use this for 'how many', 'which assets', 'what was the average',
    or any question answerable from a table of readings. Returns a plain-English answer."""
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

    if not sql.lower().startswith("select"):   # SAFETY: read-only
        return "I can only run read-only data queries."
    try:
        conn = _get_connection()
        result = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception as e:
        return f"I couldn't run that query. ({e})"

    answer_prompt = (
        f"Question: {question}\n\nSQL: {sql}\n\n"
        f"Result:\n{result.to_string(index=False)}\n\n"
        "Answer in plain English based on this result. Be concise."
    )
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
    ).choices[0].message.content


if __name__ == "__main__":
    print(query_operational_data.invoke({"question": "How many assets had a FAULT status?"}))
