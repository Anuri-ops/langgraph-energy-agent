# Step 6: single-process API container (Python 3.11-compatible runtime).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/energyagent

WORKDIR /app

RUN groupadd --system energyagent \
    && useradd --system --gid energyagent --home-dir /home/energyagent --create-home energyagent

# Keep dependency installation cache-friendly: source changes need not reinstall packages.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# .dockerignore removes secrets, cached indexes, traces, and other private files.
COPY --chown=energyagent:energyagent . /app
RUN mkdir -p /app/chroma_db /app/traces \
    && chown -R energyagent:energyagent /app/chroma_db /app/traces

USER energyagent
EXPOSE 8000

# Health measures process liveness, not OpenAI availability or RAG index readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["python", "scripts/container_entrypoint.py"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
