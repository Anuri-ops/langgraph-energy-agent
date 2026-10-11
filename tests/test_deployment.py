"""Step 6 infrastructure checks that never need Docker, Azure or paid model calls."""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose():
    return yaml.safe_load((ROOT / "compose.yaml").read_text())


def ci():
    # BaseLoader avoids the YAML 1.1 implicit boolean conversion of key 'on'.
    return yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)


def test_container_uses_python_311_and_nonroot_account():
    spec = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in spec
    assert "USER energyagent" in spec
    assert "EXPOSE 8000" in spec


def test_container_does_not_embed_keys_at_build_time():
    spec = (ROOT / "Dockerfile").read_text()
    assert "ARG OPENAI_API_KEY" not in spec
    assert "ENV OPENAI_API_KEY" not in spec
    assert "scripts/container_entrypoint.py" in spec


@pytest.mark.parametrize("required", [".env", ".env.*", "chroma_db/", "traces/", ".git/", "__pycache__/"])
def test_private_artifacts_are_excluded_from_docker(required):
    assert required in (ROOT / ".dockerignore").read_text().splitlines()


def test_env_example_is_included_as_a_template():
    assert "!.env.example" in (ROOT / ".dockerignore").read_text().splitlines()


def test_api_port_is_only_local_in_compose():
    assert compose()["services"]["api"]["ports"] == ["127.0.0.1:8000:8000"]


def test_local_compose_requires_external_env_file():
    assert compose()["services"]["api"]["env_file"] == [".env"]
    assert compose()["services"]["ingest"]["env_file"] == [".env"]


def test_ingestion_is_explicit_and_profiled_locally():
    cfg = compose()["services"]
    assert cfg["ingest"]["profiles"] == ["setup"]
    assert cfg["ingest"]["entrypoint"] == ["python", "ingest.py"]
    assert cfg["api"]["environment"]["AUTO_INGEST_ON_START"] == "0"


def test_named_volumes_preserve_index_and_traces_locally():
    cfg = compose()
    volumes = cfg["services"]["api"]["volumes"]
    assert "chroma_data:/app/chroma_db" in volumes
    assert "traces_data:/app/traces" in volumes
    assert {"chroma_data", "traces_data"} <= set(cfg["volumes"])


def test_container_api_has_healthcheck():
    assert "HEALTHCHECK" in (ROOT / "Dockerfile").read_text()
    assert "/health" in (ROOT / "Dockerfile").read_text()


def test_ci_is_triggered_by_pull_requests_and_pushes():
    triggers = ci()["on"]
    assert "pull_request" in triggers and "push" in triggers


def test_ci_uses_readonly_repo_permission():
    assert ci()["permissions"] == {"contents": "read"}


def test_ci_checks_tests_and_evaluation_and_docker():
    jobs = ci()["jobs"]
    tests = " ".join(s.get("run", "") for s in jobs["python-tests"]["steps"])
    docker = " ".join(s.get("run", "") for s in jobs["docker-build"]["steps"])
    assert "python -m pytest -q" in tests
    assert "python evaluation/run_eval.py --dry-run" in tests
    assert "scripts/check_tracked_files.py" in tests
    assert "docker build" in docker


def test_ci_does_not_push_or_call_cloud_deploy():
    config = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "docker push " not in config
    assert "az containerapp up " not in config
    assert "OPENAI_API_KEY:" not in config


@pytest.mark.parametrize("name", [".env", ".env.production", "data/.env", "traces/events.jsonl", "chroma_db/chroma.sqlite3", "evaluation/results/live.json", "abc.pem"])
def test_git_check_rejects_sensitive_paths(name):
    assert load_script("check_tracked_files").is_sensitive_path(name)


@pytest.mark.parametrize("name", [".env.example", "data/docs/maintenance.txt", "api.py", "README.md"])
def test_git_check_allows_normal_sources(name):
    assert not load_script("check_tracked_files").is_sensitive_path(name)


def test_automatic_ingestion_off_by_default(monkeypatch):
    script = load_script("container_entrypoint")
    monkeypatch.delenv("AUTO_INGEST_ON_START", raising=False)
    assert script.prepare_index_if_enabled() is False


def test_automatic_ingestion_skips_existing_index(monkeypatch):
    script = load_script("container_entrypoint")
    monkeypatch.setenv("AUTO_INGEST_ON_START", "1")
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(
        PersistentClient=lambda path: SimpleNamespace(get_or_create_collection=lambda name: SimpleNamespace(count=lambda: 11))
    ))
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **k: pytest.fail("ingestion must be skipped"))
    assert script.prepare_index_if_enabled() is False


def test_automatic_ingestion_runs_when_empty(monkeypatch):
    script = load_script("container_entrypoint")
    monkeypatch.setenv("AUTO_INGEST_ON_START", "1")
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(
        PersistentClient=lambda path: SimpleNamespace(get_or_create_collection=lambda name: SimpleNamespace(count=lambda: 0))
    ))
    calls = []
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    assert script.prepare_index_if_enabled() is True
    assert calls[0][0][0] == [sys.executable, "ingest.py"]
    assert calls[0][1]["check"] is True


def test_entrypoint_execs_application(monkeypatch):
    script = load_script("container_entrypoint")
    called = []
    monkeypatch.setattr(script, "prepare_index_if_enabled", lambda: called.append("prepare"))
    monkeypatch.setattr(script.os, "execvp", lambda *args: called.append(args))
    monkeypatch.setattr(script.sys, "argv", ["entrypoint.py", "uvicorn", "api:app"])
    script.main()
    assert called == ["prepare", ("uvicorn", ["uvicorn", "api:app"])]
