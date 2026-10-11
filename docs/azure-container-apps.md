# Azure Container Apps — preparation guide (Step 6)

**Status: deployment preparation only.** This project has not yet been deployed to Azure. Creating Azure resources may incur charges even when the demo is idle. Do not provision resources until you are ready to pay and to restrict access appropriately.

## Architecture and important boundaries

- One Docker container runs `uvicorn api:app` on port 8000 and uses the **existing** LangGraph, RAG and NL-to-SQL tools.
- `/health` is unauthenticated process **liveness** only, not proof of OpenAI access, index readiness or model correctness.
- `/ask` requires `X-API-Key`; Azure external ingress must use HTTPS. This is a demo API-key mechanism, not a complete public-production auth/rate-limiting solution.
- Set `OPENAI_API_KEY` and `ENERGY_AGENT_API_KEY` as **Azure Container Apps secrets**, never in Docker build arguments, source code, the Docker image, CI logs or `git`.
- Python/Chroma writes to `/app/chroma_db`; trace files default to `/app/traces` (or `AGENT_TRACE_DIR`). Container replicas normally have ephemeral writable storage. A fresh instance can lose its local index and metrics. For this *small synthetic* demo you can opt in to rebuilding the index on startup; every reindex makes billable OpenAI embedding requests.
- **Do not assume Azure Files is a reliable long-term SQLite/Chroma database store.** If moving beyond a demo, migrate to a supported managed vector database/storage backend, test durability and concurrency, and review data handling. Azure Files mounts are available, but an SMB-mounted SQLite DB may have locking/performance issues.

## Prerequisites

1. Complete local Step 6 `docker compose` checks, including building the actual image, starting the API and calling `/health` and `/ask`.
2. Azure subscription with sufficient permissions, [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and Container Apps extension.
3. A recent, **rotated** application key distinct from your OpenAI key.
4. Cost expectations and an access plan for public internet exposure. Consider keeping ingress internal rather than external for an early pilot.

## Optional manual Azure staging flow (Windows Command Prompt)

These commands are examples; replace names if already in use. Work from the project folder containing `Dockerfile`. **Do not type API keys into the command line or chat.**

```cmd
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az group create --name rg-energy-agent-demo --location westeurope
az containerapp up --name energy-agent-demo --resource-group rg-energy-agent-demo --location westeurope --source . --ingress external --target-port 8000
```

The `az containerapp up --source .` command can perform a remote build from the included Dockerfile and provision resources, including Azure Container Registry. **The app is not ready to answer agent questions yet**: you must configure its secrets and initialize the RAG index. If the local name or subscription has existing resources, Azure may reuse them; review the CLI output.

### Configure secrets without exposing values in shell history

Use **Azure Portal → Container Apps → energy-agent-demo → Secrets** to create:

- Secret name `openai-api-key`: your private OpenAI API key.
- Secret name `energy-agent-api-key`: your own separate random API authentication key.

Then apply secret references, tracing location and opt-in startup indexing via the CLI (no secret values appear in these commands):

```cmd
az containerapp update --name energy-agent-demo --resource-group rg-energy-agent-demo --set-env-vars OPENAI_API_KEY=secretref:openai-api-key ENERGY_AGENT_API_KEY=secretref:energy-agent-api-key AGENT_TRACE_DIR=/app/traces AUTO_INGEST_ON_START=1 --min-replicas 1 --max-replicas 1
```

`AUTO_INGEST_ON_START=1` is only for this small synthetic demo: the entrypoint checks for a nonempty Chroma collection before calling existing `ingest.py`. It may trigger **new OpenAI embedding costs** whenever the storage is replaced, and initial startup can take time. Configure an appropriate workload size and cost controls. For a single-replica demo, `--min-replicas 1` reduces cold starts but incurs resource charges while idle.

Open the HTTPS app URL output by Azure, visit `/docs`, and verify `/health` returns 200. Then test `/ask` with `X-API-Key` in a trusted client; do not screenshot its key. A non-matching key must return 401. Check Azure revision logs if embedding initialization or app startup fails.

Before exposing the service to untrusted users, add per-key quotas/rate limiting, identity-based authentication or a gateway, network restrictions and budget alerts. A correct `/health` response alone is insufficient.

### Storage and scaling before production

- The local Docker Compose named volume **is not** transferred to Azure.
- For this demo, accept that a replacement Azure replica may re-ingest the small synthetic documents and that local trace history can disappear.
- Azure Container Apps supports [Azure Files storage mounts](https://learn.microsoft.com/azure/container-apps/storage-mounts) for persistent files, but validate Chroma/SQLite suitability first; a safer future architecture uses an external supported vector database, plus durable telemetry.
- Do not run multiple replicas using the same writable local Chroma directory. Set one replica for the demo. Do not represent the setup as highly available or production-hardened.
- GitHub Actions in this Step 6 ZIP perform **test and image build only**. They do not authenticate to Azure, push images or deploy. Promotion/deployment remains a separate approval.

## Cleanup and costs

After confirming no other workloads use this resource group, Azure Portal can delete the demo resource group to stop its associated resources (registry, environment and logs). **Deleting resources is irreversible.** Set a budget alert before deployment.

## Official documentation

- https://learn.microsoft.com/azure/container-apps/containerapp-up
- https://learn.microsoft.com/azure/container-apps/environment-variables
- https://learn.microsoft.com/azure/container-apps/storage-mounts
