"""Authenticated HTTP adapter for the existing LangGraph Energy Agent.

The Streamlit app (app.py) and agent routing (agent.py) remain unchanged.
Start with: uvicorn api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import secrets
from typing import Annotated, Callable

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, StringConstraints

# Local development only: never include .env in a distributed source archive.
load_dotenv()

MAX_QUESTION_CHARS = 500
Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    trace_id: str
    latency_ms: float


app = FastAPI(
    title="LangGraph Energy Agent API",
    description="Authenticated HTTP endpoint over the existing RAG + NL-to-SQL agent.",
    version="0.2.0",
)


def verify_api_key(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
    """Fail closed if no separate application API key has been configured."""
    expected = os.environ.get("ENERGY_AGENT_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": 'ApiKey header="X-API-Key"'},
        )


def get_agent_runner() -> Callable[[str], dict]:
    """Import only after authentication/validation; no model work on unauthorised requests.

    This dependency can be overridden in tests, so tests never spend API tokens.
    """
    from agent import run_agent
    return run_agent


@app.get("/health")
def health() -> dict[str, str]:
    """Process liveness only; does not assert that remote LLMs are available."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    _auth: Annotated[None, Depends(verify_api_key)],
    runner: Annotated[Callable[[str], dict], Depends(get_agent_runner)],
) -> AskResponse:
    """Invoke the *existing* agent and return only safe, public response fields."""
    try:
        result = runner(payload.question)
        # Validate upstream output inside the sanitizing boundary. A malformed
        # agent result must not leak an internal traceback or become a 500.
        return AskResponse(
            answer=result["answer"],
            tools_used=result["tools_used"],
            trace_id=result["trace_id"],
            latency_ms=result["latency_ms"],
        )
    except Exception as exc:
        # Do not put upstream errors, user data or credentials in HTTP responses.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Agent execution failed",
        ) from exc

