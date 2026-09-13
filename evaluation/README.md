# Evaluation and traceability

This folder adds a small, reproducible regression benchmark for the public agent demo.
It is intentionally simple and inspectable rather than a black-box LLM judge.

## What is measured

- **Tool routing** — document search vs operational SQL vs both.
- **Required facts** — deterministic checks for known answers in the synthetic corpus.
- **Grounding evidence** — checks the final answer for a source citation and falls back to the RAG retrieval trace when the outer agent drops the filename while paraphrasing.
- **Unsupported-question refusal** — three out-of-scope cases require an explicit decline and include quantitative fabrication guards so an invented figure cannot pass.
- **Latency** — per end-to-end run.
- **Traceability** — JSONL events for user question, requested tools, retrieved source files, generated SQL, tool results, final answer, failures, and optional user feedback.

These checks are engineering proxies. They are not a complete certification of semantic correctness, hallucination risk, or production safety.

## Benchmark

`eval_cases.json` contains 26 representative cases across:

- document/RAG questions,
- operational-data/SQL questions,
- unsupported questions,
- combined multi-tool questions (with partial-credit-style thresholds where exhaustive wording would create false failures).

## Run

Validate the benchmark without model calls:

```bash
python evaluation/run_eval.py --dry-run
```

Run the full benchmark (requires `OPENAI_API_KEY`):

```bash
python evaluation/run_eval.py
```

Outputs are written to `evaluation/results/` as JSONL, CSV, and Markdown.

Runtime traces are written to `traces/agent_traces.jsonl`.

## Trace boundaries

The logger records observable execution events only. It does **not** expose hidden chain-of-thought.
