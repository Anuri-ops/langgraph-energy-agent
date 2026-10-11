"""Offline contract tests for the 26-case benchmark and its deterministic grader.

A benchmark dry run validates configuration; these tests exercise the grader.
They do NOT claim the model scored 26/26 on a fresh live run.
"""
from __future__ import annotations

import copy
import sys

import pytest

from evaluation import run_eval


def basic_case(**changes):
    case = {
        "id": "test_case",
        "category": "combined",
        "question": "What faults and guidance?",
        "expected_tools": ["search_documents", "query_operational_data"],
        "required_groups": [["five", "5"], ["check"]],
        "expected_sources": ["04_fault_diagnosis_guide.txt"],
    }
    case.update(changes)
    return case


def grade(case=None, answer="Five assets. Check the manual.", tools=None, trace=None):
    return run_eval.grade_case(
        case or basic_case(), answer,
        ["query_operational_data", "search_documents"] if tools is None else tools,
        [] if trace is None else trace,
    )


def test_the_original_26_cases_are_well_formed_and_have_unique_ids():
    cases = run_eval.load_cases()
    assert run_eval.validate_cases(cases) == 26
    assert {c["category"] for c in cases} == {"rag", "sql", "combined", "refusal"}


@pytest.mark.parametrize("invalid_update", [
    {"category": "unknown"},
    {"question": " "},
    {"expected_tools": ["get_root_shell"]},
    {"expected_tools": ["search_documents", "search_documents"]},
    {"expected_tools": "search_documents"},
    {"required_groups": [[]]},
    {"required_groups": [[""]]},
    {"expected_sources": [123]},
    {"optional_groups": [["safe"]], "min_optional_matches": 2},
    {"optional_groups": [], "min_optional_matches": -1},
    {"forbid_patterns": ["["]},
])
def test_invalid_benchmark_cases_fail_before_live_execution(invalid_update):
    with pytest.raises(ValueError):
        run_eval.validate_cases([basic_case(**invalid_update)])


def test_duplicate_benchmark_case_id_is_rejected():
    test = basic_case()
    with pytest.raises(ValueError, match="Duplicate"):
        run_eval.validate_cases([test, copy.deepcopy(test)])


def test_exact_tool_routing_is_required():
    assert grade(tools=["search_documents"])['routing_ok'] is False
    assert grade(tools=["search_documents", "query_operational_data", "other"])['routing_ok'] is False
    assert grade(tools=["query_operational_data", "search_documents"])['routing_ok'] is True


@pytest.mark.parametrize("answer,expected", [
    ("FIVE assets; CHECK the procedure.", True),
    ("5 assets -- check the procedure.", True),
    ("5 assets but no maintenance advice.", False),
    ("Check the procedure; number unknown.", False),
])
def test_required_facts_are_case_and_format_tolerant(answer, expected):
    assert grade(answer=answer)['facts_ok'] is expected


def test_source_citation_can_be_in_answer():
    result = grade(answer="5 assets. Check 04_FAULT_DIAGNOSIS_GUIDE.TXT.")
    assert result['source_in_answer'] is True
    assert result['source_ok'] is True


def test_source_evidence_can_come_from_retrieval_trace():
    events = [{"event": "rag_retrieval", "payload": {
        "sources": ["04_fault_diagnosis_guide.txt"]
    }}]
    result = grade(trace=events)
    assert result['source_in_answer'] is False
    assert result['source_in_trace'] is True
    assert result['source_ok'] is True


def test_tool_trace_with_wrong_filename_cannot_pass_source_check():
    events = [{"event": "rag_retrieval", "payload": {"sources": ["other.txt"]}}]
    result = grade(trace=events)
    assert result['source_ok'] is False
    assert result['passed'] is False


def test_source_check_does_not_accept_unrelated_trace_event():
    events = [{"event": "sql_result", "payload": {
        "sources": ["04_fault_diagnosis_guide.txt"]
    }}]
    assert grade(trace=events)['source_ok'] is False


def test_optional_group_threshold_passes_with_enough_evidence():
    case = basic_case(expected_sources=[], optional_groups=[
        ["temperature"], ["vibration"], ["lubrication"],
    ], min_optional_matches=2)
    assert grade(case=case, answer="5 faults: check temperature and vibration")['facts_ok']
    assert not grade(case=case, answer="5 faults: check temperature")['facts_ok']


@pytest.mark.parametrize("answer,expected", [
    ("This information is not available in the supplied documents.", True),
    ("I cannot find the requested figure.", True),
    ("It was 7.2 million.", False),
])
def test_refusal_detection_requires_explicit_refusal(answer, expected):
    case = basic_case(expected_sources=[], expected_tools=["search_documents"],
                      required_groups=[], refusal_expected=True)
    outcome = grade(case=case, answer=answer, tools=["search_documents"])
    assert outcome['refusal_ok'] is expected


def test_refusal_fabrication_guard_rejects_made_up_quantities():
    case = basic_case(expected_sources=[], expected_tools=["search_documents"],
                      required_groups=[], refusal_expected=True,
                      forbid_patterns=[r"\b\d+(?:\.\d+)?\s+million\b"])
    result = grade(case=case,
                   answer="Data is not available, but it was 7 million.",
                   tools=["search_documents"])
    assert result['refusal_ok'] is False
    assert result['fabrication_guard_ok'] is False


def test_combined_case_fails_when_only_one_tool_runs():
    case = basic_case(expected_sources=[])
    result = grade(case=case, answer="Five faults, check the manual",
                   tools=["query_operational_data"])
    assert result['routing_ok'] is False
    assert result['passed'] is False


def test_build_report_summarizes_passes_failures_and_latency():
    sample = [{**basic_case(), **grade(trace=[{"event": "rag_answer", "payload": {
        "sources": ["04_fault_diagnosis_guide.txt"]
    }}]), "tools_used": ["search_documents", "query_operational_data"],
        "answer": "5 faults. Check manual.", "latency_ms": 42.0}]
    report = run_eval.build_report(sample, "offline-test")
    assert "Agent Evaluation Report" in report
    assert "1/1 (100.0%)" in report
    assert "42.0 ms" in report
    assert "No failed cases" in report


def test_dry_run_does_not_invoke_agent_or_write_live_results(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--dry-run"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_eval.main()
    output = capsys.readouterr().out
    assert "Validated 26 evaluation cases" in output
    assert "Categories:" in output
