# Step 5 — LLMOps and Observability

Date: 11 October 2026. Based on the verified Step 4 project.

## Changes

1. Added SHA-256 content-based prompt versions for LangGraph routing, RAG answer, SQL generation and SQL summarization.
2. Added per-provider-call latency, model/operation labels, and OpenAI/LangChain provider-reported token parsing, including embeddings. Unknown token usage stays unknown.
3. Added per-run token aggregates with completeness flags and error-type trace events, preserving the trace-ID mechanism.
4. Replaced verbose raw-question/SQL/answer logs with an allowlisted local JSONL event writer that drops sensitive content and tool arguments.
5. Added `python observability.py` CLI summary: successful runs, failures, median/p95 latency, known token totals and complete-usage run count.
6. Added offline LLMOps tests for token parsing, missing usage, prompt versioning, privacy redaction, nested run isolation, metrics in run completion and summary calculations.
7. Preserved existing FastAPI endpoints, Streamlit UI, LangGraph tool-routing architecture and SQL security.

## Verification

- Mock-only pytest suite: **132 passed** in build environment.
- 26 evaluation case definitions validated via --dry-run.
- Live LLM evaluation **not** rerun for Step 5.
- Local Windows test and real telemetry smoke test requested from user.

## Boundaries

- Router individual provider-call latency is not available from returned message history, so logged as null; run-level duration is measured.
- Telemetry is local demo-scale, not production distributed tracing.
- Some provider adapters may omit usage. Totals then remain incomplete, never fabricated.
- Trace context in tool-worker threads depends on runtime context propagation.
- Previous verbose Step 4 trace files may contain sensitive content; do not migrate/share them.
- No Azure or CI/CD deployment yet; this is scheduled for Step 6.
- Aker CVs remain unchanged until Step 8 and verified local execution.
