"""Privacy-conscious JSONL tracing for observable LangGraph operations.

No model chain-of-thought or prompt contents are recorded. Missing provider token
usage is represented as unknown, never estimated. This is local demo telemetry,
not a production distributed tracing system.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

TRACE_DIR = Path(os.getenv("AGENT_TRACE_DIR", "traces"))
TRACE_FILE = TRACE_DIR / "agent_traces.jsonl"
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_trace_id", default=None)
_current_run: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_run_metrics", default=None)

# Explicit allowlist: no questions, prompts, model answers, SQL text, tool args,
# database row previews, or exception messages in logs, even if callers pass them.
SAFE_FIELDS = frozenset({
    "prompt_version", "prompt_versions", "model", "operation", "provider",
    "source", "sources", "distances", "chunk_count", "rows", "row_count",
    "tools_used", "tools_requested", "tool", "tool_call_count", "model_call_count",
    "token_usage_complete", "token_usage_available", "input_tokens", "output_tokens",
    "total_tokens", "latency_ms", "error_type", "rating", "status", "reason_code",
    "question_chars", "answer_chars", "usage_source", "token_usage",
})


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: str | None):
    return _current_trace_id.set(trace_id)


def reset_trace_id(token) -> None:
    _current_trace_id.reset(token)


def get_trace_id() -> str | None:
    return _current_trace_id.get()


def prompt_version(prompt: str, label: str) -> str:
    """Content-addressed version: automatically changes when trusted instructions change."""
    return f"{label}-sha256:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, val in payload.items():
        if key not in SAFE_FIELDS:
            continue
        if key == "tools_requested":
            # Older agent code passes dictionaries with untrusted tool args.
            safe[key] = [x.get("name") if isinstance(x, dict) else str(x) for x in val]
        elif key in ("sources", "tools_used", "prompt_versions"):
            if key == "prompt_versions" and isinstance(val, dict):
                safe[key] = {str(k): str(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                safe[key] = [str(v) for v in val]
        elif key == "token_usage" and isinstance(val, dict):
            safe[key] = {k: v for k, v in val.items() if k in {
                "input_tokens", "output_tokens", "total_tokens", "model_call_count",
                "token_usage_available", "token_usage_complete"}}
        elif isinstance(val, (str, int, float, bool)) or val is None:
            safe[key] = val
        elif key == "distances" and isinstance(val, (list, tuple)):
            safe[key] = [float(x) for x in val]
    return safe


def log_event(event_type: str, payload: dict[str, Any], trace_id: str | None = None) -> None:
    """Write safe JSONL event; tracing failures must never break the agent."""
    record = {
        "ts_unix": round(time.time(), 3),
        "trace_id": trace_id or get_trace_id(),
        "event": event_type,
        "payload": _safe_payload(payload),
    }
    try:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, TypeError, ValueError):
        # Metrics should not cause a user-facing failure.
        pass


def begin_run():
    """Scoped accumulator shared with in-run tool calls when context propagates."""
    return _current_run.set({"calls": 0, "known": 0, "input": 0, "output": 0, "total": 0})


def end_run(token) -> None:
    _current_run.reset(token)


def _get_value(obj: Any, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _valid_token(value: Any) -> bool:
    return type(value) is int and value >= 0


def extract_usage(obj: Any, *, embedding: bool = False) -> dict[str, int] | None:
    """OpenAI SDK .usage or LangChain AIMessage usage_metadata/response_metadata.

    Returns None if unavailable or malformed. Does not substitute zeros for
    missing counts. LangChain usage_metadata takes precedence to avoid counting
    both representations of one completion.
    """
    usage = _get_value(obj, "usage_metadata") or _get_value(obj, "usage")
    if usage is None:
        metadata = _get_value(obj, "response_metadata") or {}
        usage = _get_value(metadata, "token_usage")
    if usage is None:
        return None
    input_tokens = _get_value(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _get_value(usage, "prompt_tokens")
    output_tokens = _get_value(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _get_value(usage, "completion_tokens")
    total_tokens = _get_value(usage, "total_tokens")
    if embedding and output_tokens is None:
        output_tokens = 0  # embeddings do not generate text output
    if not (_valid_token(input_tokens) and _valid_token(output_tokens)):
        return None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    if not _valid_token(total_tokens) or total_tokens < input_tokens + output_tokens:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def record_model_call(*, model: str, operation: str, prompt_version: str,
                      response: Any, latency_ms: float | None, embedding: bool = False) -> None:
    """Record a provider-reported token count; no cost or token guesses."""
    usage = extract_usage(response, embedding=embedding)
    state = _current_run.get()
    if state is not None:
        state["calls"] += 1
        if usage is not None:
            state["known"] += 1
            state["input"] += usage["input_tokens"]
            state["output"] += usage["output_tokens"]
            state["total"] += usage["total_tokens"]
    log_event("llm_call", {
        "model": model, "operation": operation, "prompt_version": prompt_version,
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "token_usage_available": usage is not None,
        **(usage or {}),
    })


def usage_summary() -> dict[str, Any]:
    state = _current_run.get()
    if state is None:
        return {"model_call_count": 0, "token_usage_available": False,
                "token_usage_complete": False, "input_tokens": None,
                "output_tokens": None, "total_tokens": None}
    return {
        "model_call_count": state["calls"],
        "token_usage_available": state["known"] > 0,
        "token_usage_complete": state["calls"] > 0 and state["known"] == state["calls"],
        "input_tokens": state["input"] if state["known"] else None,
        "output_tokens": state["output"] if state["known"] else None,
        "total_tokens": state["total"] if state["known"] else None,
    }


def trace_summary(path: Path | None = None) -> dict[str, Any]:
    """Aggregate run-level observability; ignore malformed entries gracefully."""
    path = path or TRACE_FILE
    if not path.exists():
        return {"completed": 0, "errors": 0, "latency_ms_median": None,
                "latency_ms_p95": None, "known_total_tokens": 0, "fully_measured_runs": 0}
    done = []; errors = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get("event") == "run_complete":
                done.append(event.get("payload") or {})
            elif event.get("event") == "run_error":
                errors += 1
    latencies = sorted(x["latency_ms"] for x in done if
                       type(x.get("latency_ms")) in (int, float) and math.isfinite(x["latency_ms"]))
    return {
        "completed": len(done), "errors": errors,
        "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
        "latency_ms_p95": round(latencies[math.ceil(.95 * len(latencies)) - 1], 1) if latencies else None,
        "known_total_tokens": sum(x.get("total_tokens") or 0 for x in done if
                                  type(x.get("total_tokens")) is int),
        "fully_measured_runs": sum(x.get("token_usage_complete") is True for x in done),
    }


if __name__ == "__main__":
    print(json.dumps(trace_summary(), indent=2))
