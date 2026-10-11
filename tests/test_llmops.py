"""Step 5 LLMOps tests. No OpenAI calls or generated embeddings."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import observability as obs


@pytest.fixture
def trace_file(tmp_path, monkeypatch):
    path = tmp_path / 'traces' / 'agent_traces.jsonl'
    monkeypatch.setattr(obs, 'TRACE_FILE', path)
    return path


def _records(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def test_prompt_version_content_addressed_and_labelled():
    first = obs.prompt_version('trusted prompt', 'agent-router')
    assert first == obs.prompt_version('trusted prompt', 'agent-router')
    assert first.startswith('agent-router-sha256:')
    assert first != obs.prompt_version('trusted prompt.', 'agent-router')
    assert first != obs.prompt_version('trusted prompt', 'sql-generation')


@pytest.mark.parametrize('value,expected', [
    (SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14)),
     {'input_tokens': 11, 'output_tokens': 3, 'total_tokens': 14}),
    (SimpleNamespace(usage_metadata={'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6},
                     response_metadata={'token_usage': {'prompt_tokens': 999, 'completion_tokens': 999}}),
     {'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6}),
    ({'response_metadata': {'token_usage': {'prompt_tokens': 7, 'completion_tokens': 4, 'total_tokens': 11}}},
     {'input_tokens': 7, 'output_tokens': 4, 'total_tokens': 11}),
    ({'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}},
     {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}),
])
def test_extract_usage_supports_provider_and_langchain_formats(value, expected):
    assert obs.extract_usage(value) == expected


@pytest.mark.parametrize('value', [
    {}, {'usage': {}}, {'usage': {'prompt_tokens': 5}},
    {'usage': {'prompt_tokens': -1, 'completion_tokens': 3}},
    {'usage': {'prompt_tokens': 3.0, 'completion_tokens': 1}},
    {'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 4}},
    {'usage': {'prompt_tokens': True, 'completion_tokens': 0}},
])
def test_missing_or_malformed_usage_is_unknown(value):
    assert obs.extract_usage(value) is None


def test_embeddings_count_only_input_tokens_when_provider_reports_usage():
    assert obs.extract_usage({'usage': {'prompt_tokens': 8, 'total_tokens': 8}}, embedding=True) == {
        'input_tokens': 8, 'output_tokens': 0, 'total_tokens': 8
    }
    assert obs.extract_usage({'usage': {'total_tokens': 8}}, embedding=True) is None


def test_log_event_redacts_secrets_questions_sql_and_tool_arguments(trace_file):
    secret = 'sk-sensitive-raw-secret'
    obs.log_event('run_complete', {
        'question': f'key {secret}', 'answer': secret, 'sql': f'SELECT {secret}',
        'preview': [{'private': secret}], 'error': secret,
        'tools_requested': [{'name': 'search_documents', 'args': {'question': secret}}],
        'tools_used': ['search_documents'], 'trace_id': 'not-to-be-overridden',
        'latency_ms': 10.2,
    }, trace_id='safe-trace')
    raw = trace_file.read_text()
    assert secret not in raw
    record = _records(trace_file)[0]
    assert record['trace_id'] == 'safe-trace'
    assert record['payload'] == {
        'tools_requested': ['search_documents'], 'tools_used': ['search_documents'],
        'latency_ms': 10.2,
    }


def test_log_event_handles_missing_log_parent(trace_file):
    obs.log_event('user_feedback', {'rating': 'helpful'}, trace_id='feedback-1')
    assert _records(trace_file)[0]['payload']['rating'] == 'helpful'


def test_run_usage_aggregates_model_calls_and_honestly_reports_partial(trace_file):
    run = obs.begin_run()
    token = obs.set_trace_id('trace-7')
    try:
        obs.record_model_call(model='gpt-4o-mini', operation='sql_generate', prompt_version='v1',
            response={'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}, latency_ms=130)
        obs.record_model_call(model='gpt-4o-mini', operation='sql_answer', prompt_version='v2',
            response={'usage': {'prompt_tokens': 30, 'completion_tokens': 5, 'total_tokens': 35}}, latency_ms=42)
        full = obs.usage_summary()
        assert full == {'model_call_count': 2, 'token_usage_available': True,
                        'token_usage_complete': True, 'input_tokens': 130,
                        'output_tokens': 25, 'total_tokens': 155}
        obs.record_model_call(model='gpt-4o-mini', operation='router', prompt_version='v3',
            response={}, latency_ms=None)
        partial = obs.usage_summary()
        assert partial['model_call_count'] == 3
        assert partial['total_tokens'] == 155
        assert partial['token_usage_complete'] is False
        assert partial['token_usage_available'] is True
    finally:
        obs.end_run(run)
        obs.reset_trace_id(token)
    assert obs.get_trace_id() is None
    assert obs.usage_summary()['total_tokens'] is None
    events = _records(trace_file)
    assert [x['event'] for x in events] == ['llm_call'] * 3
    assert all(x['trace_id'] == 'trace-7' for x in events)
    assert 'input_tokens' not in events[-1]['payload']
    assert events[-1]['payload']['latency_ms'] is None


def test_unknown_usage_is_not_zero(trace_file):
    token = obs.begin_run()
    try:
        obs.record_model_call(model='gpt-4o-mini', operation='agent_router', prompt_version='v1',
            response={}, latency_ms=2)
        summary = obs.usage_summary()
        assert summary['model_call_count'] == 1
        assert summary['total_tokens'] is None
        assert summary['token_usage_available'] is False
        assert summary['token_usage_complete'] is False
    finally:
        obs.end_run(token)


def test_scoped_runs_do_not_leak_metrics_across_requests(trace_file):
    outer = obs.begin_run()
    try:
        obs.record_model_call(model='m', operation='o', prompt_version='v',
            response={'usage': {'prompt_tokens': 1, 'completion_tokens': 2}}, latency_ms=1)
        inner = obs.begin_run()
        try:
            assert obs.usage_summary()['model_call_count'] == 0
        finally:
            obs.end_run(inner)
        assert obs.usage_summary()['total_tokens'] == 3
    finally:
        obs.end_run(outer)


def test_summary_calculates_completed_errors_and_latency_without_raw_data(trace_file):
    obs.log_event('run_complete', {'latency_ms': 20, 'total_tokens': 10, 'token_usage_complete': True})
    obs.log_event('run_complete', {'latency_ms': 30, 'total_tokens': None, 'token_usage_complete': False})
    obs.log_event('run_complete', {'latency_ms': 50, 'total_tokens': 20, 'token_usage_complete': True})
    obs.log_event('run_error', {'error_type': 'TimeoutError'})
    with trace_file.open('a') as f:
        f.write('invalid-json\n')
    assert obs.trace_summary(trace_file) == {
        'completed': 3, 'errors': 1, 'latency_ms_median': 30,
        'latency_ms_p95': 50, 'known_total_tokens': 30, 'fully_measured_runs': 2,
    }


def test_summary_when_no_trace_file(trace_file):
    assert obs.trace_summary(trace_file)['completed'] == 0
    assert obs.trace_summary(trace_file)['latency_ms_median'] is None


def test_logging_disk_failure_never_breaks_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, 'TRACE_FILE', tmp_path)  # directory, not file
    obs.log_event('test', {'status': 'ok'})  # must not raise


def test_embedding_and_prompt_versions_present_in_real_tools():
    from prompt_boundaries import SQL_SYSTEM_PROMPT, SQL_ANSWER_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
    assert obs.prompt_version(SQL_SYSTEM_PROMPT, 'sql-generate')
    assert obs.prompt_version(SQL_ANSWER_SYSTEM_PROMPT, 'sql-answer')
    assert obs.prompt_version(RAG_SYSTEM_PROMPT, 'rag-answer')


def test_run_agent_includes_router_token_usage_in_completed_trace():
    # A fully isolated agent import, with no LangChain / OpenAI network calls.
    import importlib.util
    import sys
    import types
    from types import SimpleNamespace
    from unittest.mock import patch

    events = []

    class FakeGraph:
        def invoke(self, req):
            return {'messages': [
                SimpleNamespace(type='human', name=None, tool_calls=[], content='question'),
                SimpleNamespace(type='ai', name=None, tool_calls=[], content='A',
                                response_metadata={'model_name': 'gpt-4o-mini'},
                                usage_metadata={'input_tokens': 15, 'output_tokens': 4, 'total_tokens': 19}),
            ]}

    fake_agent = types.ModuleType('langchain.agents')
    fake_agent.create_agent = lambda *a, **k: FakeGraph()
    fake_langchain = types.ModuleType('langchain')
    fake_langchain.__path__ = []
    fake_openai = types.ModuleType('langchain_openai')
    fake_openai.ChatOpenAI = lambda **kwargs: object()
    fake_rag = types.ModuleType('rag_tool')
    fake_rag.search_documents = object()
    fake_sql = types.ModuleType('sql_tool')
    fake_sql.query_operational_data = object()
    fake_env = types.ModuleType('dotenv')
    fake_env.load_dotenv = lambda *args, **kwargs: None

    # The real observability implementation is used; patch only the event sink.
    def capture(event, payload, trace_id=None):
        events.append((event, payload, trace_id))

    fake_obs = types.ModuleType('observability')
    for name in ('prompt_version', 'new_trace_id', 'set_trace_id', 'reset_trace_id',
                 'begin_run', 'end_run', 'record_model_call', 'usage_summary'):
        setattr(fake_obs, name, getattr(obs, name))
    fake_obs.log_event = capture

    with patch.dict(sys.modules, {
        'langchain': fake_langchain, 'langchain.agents': fake_agent,
        'langchain_openai': fake_openai, 'rag_tool': fake_rag, 'sql_tool': fake_sql,
        'dotenv': fake_env, 'observability': fake_obs,
    }):
        spec = importlib.util.spec_from_file_location('_llmops_fake_agent',
            Path(__file__).resolve().parents[1] / 'agent.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    result = module.run_agent('Hi!', trace_id='trace-example')
    assert result['answer'] == 'A'
    assert result['trace_id'] == 'trace-example'
    assert [e[0] for e in events] == ['run_start', 'run_complete']
    completed = events[-1][1]
    assert completed['total_tokens'] == 19
    assert completed['input_tokens'] == 15
    assert completed['output_tokens'] == 4
    assert completed['model_call_count'] == 1
    assert completed['token_usage_complete'] is True
    assert completed['prompt_version'].startswith('agent-router-sha256:')
    assert obs.usage_summary()['total_tokens'] is None
