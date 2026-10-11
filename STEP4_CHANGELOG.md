# Step 4 — Automated tests and regression coverage

Date: 11 October 2026. Built from the Step 3 security-hardening archive, not
from a rebuilt or substituted agent. Streamlit `app.py`, the LangGraph graph
in `agent.py`, and the six-document / synthetic-CSV dataset are unchanged.

## Changes

1. Added mock-only agent orchestration tests (`tests/test_agent_orchestration.py`)
   for registered tools, RAG/SQL/both/neither tool-message reporting,
   requested-versus-executed tools, trace IDs, and exception cleanup.
2. Expanded API tests (`tests/test_api.py`) for invalid request types and
   formats, injection-like strings passed as data, correct HTTP methods, and
   sanitised malformed agent responses. Fixed `api.py` to validate upstream
   responses within the exception-handling boundary (returns HTTP 502).
3. Added fake-client RAG adapter tests (`tests/test_rag_regression.py`) for
   retrieval inputs, source evidence, untrusted prompt placement, and empty
   retrieval. Updated `rag_tool.py` to return an explicit not-found message
   without a model call when no documents were retrieved.
4. Added deterministic benchmark grader tests
   (`tests/test_evaluation_regression.py`) for exact routing, source
   evidence, required/optional facts, refusals, fabrication guards, and
   reports. Strengthened `evaluation/run_eval.py` definition validation
   before live evaluation (no changes to the original 26 case definitions).
5. Added optional `pytest-cov` dependency to `requirements-dev.txt` and
   documented reproducible Windows commands and testing boundaries.

## Results in the build environment

- `python -m pytest -q`: **109 passed** (offline; no OpenAI usage).
- `python evaluation/run_eval.py --dry-run`: **26 case definitions validated**.
- Python module compilation: passed.
- SQL enforcement tests: unchanged from Step 3, passing.
- `app.py` and `agent.py`: source code unchanged from Step 3.

## Not claimed

- No new live 26-case benchmark result.
- No verified production cloud deployment or persistent public API rate limit.
- No universal prompt-injection immunity or end-to-end LLM accuracy claim.
- No revised Aker Solutions CVs yet (Step 8).

## Security

The archive excludes `.env` and other secrets, `.git`, virtual environments,
Chroma indexes, runtime traces, caches, and generated evaluation result files.
Use your private rotated API key locally and do not share screenshots of keys.
