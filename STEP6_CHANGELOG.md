# Step 6 — Containers, GitHub Actions and Azure deployment preparation

**Base:** Step 5 verified project, not a replacement agent.

## Added

- `Dockerfile`: Python 3.11-slim API container, non-root user, `GET /health` process liveness check, single Uvicorn worker, no credentials baked into image.
- `.dockerignore`: excludes `.env` variants, generated Chroma data, traces, eval output, git history, private keys and caches; `.env.example` retained.
- `compose.yaml`: localhost-only API port, named data and trace volumes, and an explicitly-invoked `ingest` service; normal `up` does not consume OpenAI embedding credits to create the index.
- `scripts/container_entrypoint.py`: optional demo-only startup ingestion behind `AUTO_INGEST_ON_START=1` when the index is empty; off by default.
- `.github/workflows/ci.yml`: Python 3.11 unit tests, 26-case dry-run evaluation validation, Git tracked-file leak detection, Docker image build. No secrets, no deployments.
- `scripts/check_tracked_files.py`: blocks accidentally committed `.env`, vector data, traces, sample secret file types, and evaluation results.
- `docs/azure-container-apps.md`: manual Azure Container Apps staging path and limitations, cloud secret references, non-persistent local Chroma caveat and cost safety.
- `tests/test_deployment.py`: additional infrastructure and entrypoint unit tests, mocked to avoid cloud access and model cost.
- PyYAML only in development dependencies, used to check YAML structure offline.

## Unchanged

- `app.py`, `agent.py`, `api.py`, `sql_tool.py`, `rag_tool.py`, existing prompt security and telemetry behaviour.
- Original synthetic data and 26 evaluation definitions.

## Verification labels

- **Offline tests:** run `python -m pytest -q` in the project folder. No live OpenAI calls.
- **Evaluation structural validation:** `python evaluation/run_eval.py --dry-run`; not a live success rate.
- **Static Docker/Compose validation:** YAML parsed and config properties asserted in pytest.
- **Docker image build & runtime:** NOT verified where Docker engine is not installed; verify on the user's Windows machine and/or through GitHub Actions image build.
- **GitHub Actions CI:** configuration created, but the workflow has not executed on GitHub until uploaded and pushed.
- **Azure:** NOT deployed or cloud tested. Azure creation requires explicit approval and an active subscription.

## Security reminders

- Use new, rotated API credentials. Keep `.env` out of screenshots and Git.
- `GET /health` checks liveness only; it does not validate remote model access or RAG readiness.
- Production internet exposure still needs rate limiting, better auth, budget safeguards and durable data storage.
