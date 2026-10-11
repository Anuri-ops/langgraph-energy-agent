"""Mocked tests of the actual run_agent wrapper; zero LLM calls or token cost.

No LangChain/Chroma import is needed. These tests verify observable tool
routing metadata, trace lifecycle and error paths, NOT live model routing.
"""
from __future__ import annotations

import contextvars
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_agent(monkeypatch):
    llm_ctor_args = []
    agent_factory_args = []
    events = []
    trace = contextvars.ContextVar("step4_trace", default=None)

    class FakeAgent:
        def __init__(self):
            self.response = {"messages": []}
            self.error = None
            self.requests = []
            self.trace_values = []

        def invoke(self, request):
            self.requests.append(request)
            self.trace_values.append(trace.get())
            if self.error:
                raise self.error
            return self.response

    stub_agent = FakeAgent()
    fake_llm = types.ModuleType("langchain_openai")
    fake_llm.ChatOpenAI = lambda **kwargs: llm_ctor_args.append(kwargs) or "fake-llm"
    fake_agents = types.ModuleType("langchain.agents")
    fake_agents.create_agent = lambda llm, tools, system_prompt: (
        agent_factory_args.append((llm, tools, system_prompt)) or stub_agent
    )
    fake_lc = types.ModuleType("langchain")
    fake_lc.__path__ = []
    fake_rag = types.ModuleType("rag_tool")
    fake_rag.search_documents = object()
    fake_sql = types.ModuleType("sql_tool")
    fake_sql.query_operational_data = object()
    fake_obs = types.ModuleType("observability")
    fake_obs.prompt_version = lambda content, label: f"{label}-test"
    fake_obs.begin_run = lambda: None
    fake_obs.end_run = lambda token: None
    fake_obs.usage_summary = lambda: {"model_call_count": 0, "token_usage_complete": False}
    fake_obs.record_model_call = lambda **kwargs: None
    fake_obs.new_trace_id = lambda: "created-trace"
    fake_obs.set_trace_id = lambda value: trace.set(value)
    fake_obs.reset_trace_id = lambda token: trace.reset(token)
    fake_obs.log_event = lambda kind, payload, trace_id=None: events.append(
        (kind, payload, trace_id or trace.get())
    )
    fake_env = types.ModuleType("dotenv")
    fake_env.load_dotenv = lambda *a, **k: None

    with monkeypatch.context() as mp:
        for name, module in {
            "langchain_openai": fake_llm,
            "langchain": fake_lc,
            "langchain.agents": fake_agents,
            "rag_tool": fake_rag,
            "sql_tool": fake_sql,
            "observability": fake_obs,
            "dotenv": fake_env,
        }.items():
            mp.setitem(sys.modules, name, module)
        spec = importlib.util.spec_from_file_location("_step4_agent", ROOT / "agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return SimpleNamespace(
        module=module, graph=stub_agent, events=events, trace=trace,
        constructors=llm_ctor_args, factory=agent_factory_args,
        rag_tool=fake_rag.search_documents, sql_tool=fake_sql.query_operational_data,
    )


def message(*, name=None, text="", tool_calls=None):
    return SimpleNamespace(name=name, content=text, tool_calls=tool_calls or [])


def test_agent_factory_receives_both_original_tools(isolated_agent):
    fixture = isolated_agent
    assert fixture.constructors == [{"model": "gpt-4o-mini", "temperature": 0}]
    assert len(fixture.factory) == 1
    llm, tools, prompt = fixture.factory[0]
    assert llm == "fake-llm"
    assert tools == [fixture.rag_tool, fixture.sql_tool]
    assert "untrusted data" in prompt


@pytest.mark.parametrize("tool_names", [
    ["search_documents"],
    ["query_operational_data"],
    ["search_documents", "query_operational_data"],
    [],
])
def test_run_agent_reports_real_tool_messages_without_network(isolated_agent, tool_names):
    fixture = isolated_agent
    fixture.graph.response = {"messages": [
        message(text="requesting tools", tool_calls=[
            {"name": name, "args": {"question": "x"}, "id": f"call-{idx}"}
            for idx, name in enumerate(tool_names)
        ]),
        *(message(name=n, text="tool result") for n in tool_names),
        message(text="A grounded answer"),
    ]}
    result = fixture.module.run_agent("What happened?", trace_id="test-trace")
    assert result["answer"] == "A grounded answer"
    assert result["tools_used"] == tool_names
    assert [c["name"] for c in result["tool_calls"]] == tool_names
    assert result["latency_ms"] >= 0
    assert fixture.graph.requests == [{"messages": [("user", "What happened?")]}]
    assert fixture.graph.trace_values == ["test-trace"]
    assert fixture.trace.get() is None
    assert [e[0] for e in fixture.events] == ["run_start", "run_complete"]
    assert all(e[2] == "test-trace" for e in fixture.events)


def test_repeated_tool_messages_are_deduplicated(isolated_agent):
    fixture = isolated_agent
    fixture.graph.response = {"messages": [
        message(name="search_documents"),
        message(name="search_documents"),
        message(name="query_operational_data"),
        message(text="done"),
    ]}
    result = fixture.module.run_agent("faults")
    assert result["trace_id"] == "created-trace"
    assert result["tools_used"] == ["search_documents", "query_operational_data"]


def test_unknown_tool_message_not_reported_as_used(isolated_agent):
    fixture = isolated_agent
    fixture.graph.response = {"messages": [
        message(name="unexpected_plugin"), message(text="done")
    ]}
    result = fixture.module.run_agent("hi")
    assert result["tools_used"] == []


def test_requested_but_not_executed_tool_is_not_reported_as_used(isolated_agent):
    fixture = isolated_agent
    fixture.graph.response = {"messages": [
        message(tool_calls=[{"name": "query_operational_data", "args": {}, "id": "x"}]),
        message(text="I could not complete the tool call."),
    ]}
    result = fixture.module.run_agent("hi")
    assert [c["name"] for c in result["tool_calls"]] == ["query_operational_data"]
    assert result["tools_used"] == []


def test_agent_exception_is_logged_and_trace_context_is_reset(isolated_agent):
    fixture = isolated_agent
    fixture.graph.error = RuntimeError("upstream temporarily unavailable")
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        fixture.module.run_agent("test", trace_id="error-trace")
    assert fixture.trace.get() is None
    assert [e[0] for e in fixture.events] == ["run_start", "run_error"]
    assert fixture.events[-1][1]["error_type"] == "RuntimeError"
    assert fixture.events[-1][2] == "error-trace"


def test_prior_trace_context_restored_even_when_agent_fails(isolated_agent):
    fixture = isolated_agent
    token = fixture.trace.set("outer-trace")
    try:
        fixture.graph.error = ValueError("failure")
        with pytest.raises(ValueError):
            fixture.module.run_agent("question", trace_id="inner-trace")
        assert fixture.trace.get() == "outer-trace"
    finally:
        fixture.trace.reset(token)


def test_agent_does_not_expose_private_reasoning_keys(isolated_agent):
    fixture = isolated_agent
    fixture.graph.response = {"messages": [message(text="public final answer")]}
    result = fixture.module.run_agent("test")
    assert result["answer"] == "public final answer"
    assert "chain_of_thought" not in result
    assert "reasoning" not in result
