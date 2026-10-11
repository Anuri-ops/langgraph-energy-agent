# Upgrade Step 2 — FastAPI adapter (11 October 2026)

## Implemented
- New `api.py`: `POST /ask` authenticated with `X-API-Key` (environment variable `ENERGY_AGENT_API_KEY`), constant-time comparison, fail-closed when unconfigured.
- JSON request validation: 1–500 non-whitespace characters, leading/trailing whitespace stripped, unknown request fields rejected.
- Calls the existing `run_agent()` lazily; preserves LangGraph routing, RAG, NL-to-SQL and Streamlit UI.
- Limits HTTP responses to `answer`, `tools_used`, `trace_id`, `latency_ms`.
- Returns generic `502` on agent failure instead of exposing error internals.
- New `GET /health` provides process liveness only.
- New `.env.example`, API development dependencies, and usage/security instructions in `README.md`.
- New `tests/test_api.py` with 13 mocked tests.

## Local verification
- `python -m pytest tests/test_api.py -q`: **13 passed**.
- `python evaluation/run_eval.py --dry-run`: **Validated 26 evaluation cases** (10 RAG, 8 SQL, 5 combined, 3 refusals).
- `python -m compileall -q api.py tests/test_api.py`: passed.
- No external OpenAI API calls or cloud deployment performed in this environment.
- Existing `app.py`, `agent.py`, `rag_tool.py`, `sql_tool.py` and `evaluation/eval_cases.json` are unchanged.

## Scope and limitations
- Only Step 2 is implemented. SQL database-level read-only enforcement, production rate limiting, Docker, CI/CD, Azure deployment and upgraded end-to-end evaluation are **not** done.
- Streamlit's 5-question session limit does not apply to API callers. Keep the API private and do not expose publicly without HTTPS and rate/usage controls.
- Run `python ingest.py` on first API setup to create the document index (the distributable does not include a generated Chroma index).
- The historical 26/26 claim in README is *not* a new test outcome.
- The original ZIP contained an `.env` file and Git history; neither is included in this clean source distribution. Rotate any previously exposed secret.

## Next step
Step 3: enforce SQLite read-only mode in `sql_tool.py`, validate SQL statement structure and extend prompt-injection tests.
