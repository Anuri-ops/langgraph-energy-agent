# Step 3 upgrade — SQL protection & prompt boundaries

Completed after the locally verified Step 2 application, 11 October 2026.

## Changed

- New `sql_security.py`: SELECT-only validation, SQLite `PRAGMA query_only=ON`,
  authorizer restricting reads to `operations` plus safe SQL functions, disabled
  extensions, trusted schema off, complexity budget and output row cap.
- `sql_tool.py`: original CSV dataset and model preserved; generated SQL must
  pass the security policy before execution; safe, non-sensitive error replies;
  no answer-model invocation when the SQL is blocked.
- New `prompt_boundaries.py`: SQL and RAG instructions stay in trusted system
  messages, with questions and retrieved excerpts in user messages.
- `rag_tool.py` and `agent.py`: add explicit instructions to treat document
  excerpts/tool outputs as lower-trust data rather than executable directions.
- `tests/test_sql_security.py`: database integration and negative SQL tests.
- `tests/test_prompt_injection.py`: lower-trust prompt checks and fake-model
  generated malicious SQL tests, without external model calls.
- `README.md`: Step 3 features, limitations and local test instructions.

## Preserved

- Original Streamlit interface and five-question per-session limit.
- FastAPI `/ask` authentication and `/health` endpoint from Step 2.
- Existing LangGraph tool registry and routing architecture, synthetic CSV,
  six source documents and the 26-case evaluation file.

## Verification

- 49 offline automated tests passed (`python -m pytest -q`).
- 26 evaluation definitions validated (`python evaluation/run_eval.py --dry-run`).
- Python compilation succeeded.
- SQLite `query_only` independently tested to reject writes even if the
  authorizer is deliberately removed; tests also exercise the authorizer.

## Not yet verified / out of scope

- A fresh live 26-case evaluation after hardening.
- Windows user's environment after copying Step 3 files.
- Universal prompt-injection resistance (no such guarantee).
- Public API rate limiting, Docker, cloud deployment, formal penetration test.
- CV updates (remain deferred until the applicable improvements are verified).

## Secret hygiene

This archive excludes `.env`, `.git`, virtual environments, local traces,
`.pytest_cache`, evaluation result files and the Chroma index. Keep your own
`.env` private and rotate the previously exposed application API key.
