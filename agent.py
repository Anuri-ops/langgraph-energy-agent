"""
agent.py — a LangGraph agent with TWO tools (documents + data).
The agent decides on its own which tool(s) to call. For a combined question it
can call BOTH in sequence and synthesise the answer — the full agentic pattern.

The run_agent wrapper adds lightweight traceability around observable events.
It does not expose hidden chain-of-thought.
"""
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from rag_tool import search_documents
from sql_tool import query_operational_data
from observability import log_event, new_trace_id, reset_trace_id, set_trace_id

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_documents, query_operational_data]

SYSTEM_PROMPT = """
You are an energy operations assistant with two tools.

Use search_documents for questions about manuals, procedures, thresholds,
guidance, safety rules, maintenance instructions, and technical documentation.

Use query_operational_data for questions requiring values, dates, assets,
statuses, counts, averages, anomalies, or other facts from the operational dataset.

Use both tools only when the user's question explicitly requires both
operational data and document guidance.

If search_documents does not contain the requested information, do not query
operational data unless the question is specifically about operational records.
State clearly that the requested information is not available in the provided
documents. Never invent or infer a missing value.
"""

agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def _extract_requested_tools(messages):
    calls = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", []) or []:
            calls.append({
                "name": call.get("name"),
                "args": call.get("args", {}),
                "id": call.get("id"),
            })
    return calls


def _extract_tools_used(messages):
    names = []
    for msg in messages:
        name = getattr(msg, "name", None)
        if name in {"search_documents", "query_operational_data"}:
            names.append(name)
    return list(dict.fromkeys(names))


def run_agent(question: str, trace_id: str | None = None):
    """Run one agent turn and return answer + observable trace metadata."""
    trace_id = trace_id or new_trace_id()
    token = set_trace_id(trace_id)
    started = time.perf_counter()
    log_event("run_start", {"question": question}, trace_id)
    try:
        result = agent.invoke({"messages": [("user", question)]})
        messages = result["messages"]
        tool_calls = _extract_requested_tools(messages)
        tools_used = _extract_tools_used(messages)
        answer = messages[-1].content
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log_event(
            "run_complete",
            {
                "tools_requested": tool_calls,
                "tools_used": tools_used,
                "answer": answer,
                "latency_ms": elapsed_ms,
            },
            trace_id,
        )
        return {
            "answer": answer,
            "messages": messages,
            "trace_id": trace_id,
            "tools_used": tools_used,
            "tool_calls": tool_calls,
            "latency_ms": elapsed_ms,
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log_event(
            "run_error",
            {"error_type": type(exc).__name__, "error": str(exc), "latency_ms": elapsed_ms},
            trace_id,
        )
        raise
    finally:
        reset_trace_id(token)


def ask(question: str):
    result = run_agent(question)
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    print("\n########## DOCUMENT question ##########")
    ask("At what gearbox oil temperature must a turbine be shut down?")
    print("\n########## DATA question ##########")
    ask("How many assets had a FAULT status?")
    print("\n########## COMBINED question ##########")
    ask("Which assets had a FAULT, and what should I do about a fault?")
