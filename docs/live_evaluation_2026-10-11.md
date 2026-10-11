# LangGraph Energy Agent — verified live evaluation

**Run ID:** `20261011T171306Z`  
**Date:** 11 October 2026 (UTC run timestamp)  
**Codebase:** `Anuri-ops/langgraph-energy-agent` — Steps 2–6 merged into `main` via PR #1.  
**Execution:** live agent evaluation performed from the local Docker Compose API container using the existing six synthetic energy-operation documents and operational-data dataset.

## Results at a glance

| Metric | Verified result |
|---|---:|
| End-to-end case pass rate | **26/26 (100.0%)** |
| Tool-routing check | **26/26 (100.0%)** |
| Required-fact check | **26/26 (100.0%)** |
| Applicable source-grounding checks | **15/15 (100.0%)** |
| Unsupported-question refusal checks | **3/3 (100.0%)** |
| Median end-to-end latency | **4,605.6 ms** |
| Approximate p95 latency (as computed by evaluation runner) | **8,345.3 ms** |

### Category breakdown

| Category | Cases passed | Median latency |
|---|---:|---:|
| Document retrieval (RAG) | **10/10** | **4,074.6 ms** |
| Operational data (SQL) | **8/8** | **4,399.4 ms** |
| Combined RAG + SQL | **5/5** | **5,789.8 ms** |
| Unsupported-question refusal | **3/3** | **4,091.2 ms** |

No failing cases were reported in this run.

## Engineering verification beyond the 26-case run

- **165 automated Python tests passed locally** after the Docker/GitHub CI upgrades. These are primarily mocked/offline tests and do not replace the separate 26-case live run.
- **Docker image built and API tested locally.** The authenticated containerized FastAPI `/ask` endpoint returned HTTP 200 and an SQL-tool answer on the synthetic fault-status dataset; API authentication failure was also previously tested.
- **GitHub Actions CI passed** for the upgrade branch and pull request: Python test and Docker image build jobs both succeeded. PR #1 was subsequently merged into `main`.
- **Azure:** CLI sign-in to an enabled personal subscription was verified, but **Azure Container Apps has not been deployed**. No Azure-hosted endpoint is claimed.

## How to reproduce

From the repository root, with the required private environment variables in a local `.env` file and the RAG index prepared:

```cmd
python -m pip install -r requirements-dev.txt
python -m pytest -q
python evaluation/run_eval.py --dry-run
```

To run a *live* evaluation on a local Docker Compose instance after indexing the six demo documents:

```cmd
docker compose build
docker compose run --rm ingest
docker compose up -d api
docker compose exec -T api python evaluation/run_eval.py
```

The runner writes `eval_results_<timestamp>.jsonl`, `eval_summary_<timestamp>.csv`, and `eval_report_<timestamp>.md` into `evaluation/results/`. That directory is gitignored to avoid committing raw model responses and traces by accident. Reported metrics above were independently cross-checked against the 26-row CSV exported from run `20261011T171306Z`.

## Interpretation and limits

The benchmark is a **small, transparent synthetic engineering regression suite**. Automated checks evaluate tool choices, the presence of specified required facts, source-reference signals, refusal phrases/fabrication guards and latency. A source-grounding check may accept a matching source in a retrieval trace even when the final answer omits a filename. These checks **do not establish complete semantic correctness, full hallucination resistance, penetration-test coverage, production security, or readiness for real industrial control decisions**. Latency statistics reflect only 26 observations on one execution environment.

Use CV wording such as **"26/26 live synthetic benchmark cases passed"**, not **"100% reliable AI"** or **"production-safe agent"**.

## Evidence

- Original generated report: `eval_report_20261011T171306Z.md` (local evaluation output, not committed)
- Original case-level summary: `eval_summary_20261011T171306Z.csv` (local evaluation output, not committed)
- Evaluation definition: [`evaluation/eval_cases.json`](../evaluation/eval_cases.json)
- Evaluator: [`evaluation/run_eval.py`](../evaluation/run_eval.py)
- Automated tests: [`tests/`](../tests/)
- GitHub Actions workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
