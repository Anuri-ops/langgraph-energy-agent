"""Lightweight JSONL observability for the demo agent.

Logs only observable execution events (user question, requested tools, retrieved
sources, generated SQL, tool results, final answer, latency, and optional user
feedback). It does not log or expose hidden chain-of-thought.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_DIR = Path(os.getenv("AGENT_TRACE_DIR", "traces"))
TRACE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = TRACE_DIR / "agent_traces.jsonl"

_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trace_id", default=None
)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: Optional[str]):
    return _current_trace_id.set(trace_id)


def reset_trace_id(token) -> None:
    _current_trace_id.reset(token)


def get_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def log_event(event_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None) -> None:
    """Append one structured event to the JSONL trace file."""
    record = {
        "ts_unix": round(time.time(), 3),
        "trace_id": trace_id or get_trace_id(),
        "event": event_type,
        "payload": payload,
    }
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
