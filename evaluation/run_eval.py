"""Deterministic evaluation harness for the demo agent.

Usage:
    python evaluation/run_eval.py --dry-run
    python evaluation/run_eval.py

The live run requires OPENAI_API_KEY and writes:
- evaluation/results/eval_results_<timestamp>.jsonl
- evaluation/results/eval_summary_<timestamp>.csv
- evaluation/results/eval_report_<timestamp>.md

The checks are intentionally transparent: tool routing, expected fact checks,
source-citation checks, refusal behaviour, end-to-end pass rate, and latency.
They are proxies, not claims of full semantic correctness or safety certification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

CASES_PATH = Path(__file__).with_name("eval_cases.json")
RESULTS_DIR = Path(__file__).with_name("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REFUSAL_MARKERS = [
    "isn't there", "is not there", "not in the", "not provided",
    "cannot find", "can't find", "couldn't find", "could not find",
    "unable to find", "do not have", "don't have", "insufficient",
    "no information", "no specific information", "not available",
    "not found", "no data", "outside the available data",
]


def load_cases():
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def normalize_text(text):
    """Normalize superficial formatting differences for deterministic matching."""
    text = str(text).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def contains_group(answer: str, group):
    a = normalize_text(answer)
    return any(normalize_text(term) in a for term in group)


def _trace_sources(trace_events):
    sources = []
    for event in trace_events or []:
        if event.get("event") in {"rag_retrieval", "rag_answer"}:
            sources.extend(event.get("payload", {}).get("sources", []) or [])
    return list(dict.fromkeys(str(s) for s in sources))


def grade_case(case, answer, tools_used, trace_events=None):
    expected_tools = set(case.get("expected_tools", []))
    actual_tools = set(tools_used)
    routing_ok = actual_tools == expected_tools

    groups = case.get("required_groups", [])
    fact_checks = [contains_group(answer, g) for g in groups]
    required_ok = all(fact_checks) if groups else True

    optional_groups = case.get("optional_groups", [])
    optional_checks = [contains_group(answer, g) for g in optional_groups]
    min_optional = int(case.get("min_optional_matches", 0))
    optional_ok = sum(optional_checks) >= min_optional if optional_groups else True
    facts_ok = required_ok and optional_ok

    expected_sources = case.get("expected_sources", [])
    source_ok = True
    source_in_answer = None
    source_in_trace = None
    trace_sources = _trace_sources(trace_events)
    if expected_sources:
        low = answer.lower()
        source_in_answer = any(src.lower() in low for src in expected_sources)
        source_in_trace = any(
            expected.lower() == actual.lower()
            for expected in expected_sources
            for actual in trace_sources
        )
        # A grounded retrieval should not be scored as a hallucination merely because
        # the outer agent paraphrased the tool answer and dropped the filename.
        source_ok = source_in_answer or source_in_trace

    refusal_ok = True
    fabrication_guard_ok = True
    if case.get("refusal_expected"):
        low = answer.lower()
        refusal_ok = any(marker in low for marker in REFUSAL_MARKERS)
        patterns = case.get("forbid_patterns", [])
        fabrication_guard_ok = not any(re.search(pattern, answer, flags=re.I) for pattern in patterns)
        refusal_ok = refusal_ok and fabrication_guard_ok

    passed = routing_ok and facts_ok and source_ok and refusal_ok
    return {
        "routing_ok": routing_ok,
        "facts_ok": facts_ok,
        "source_ok": source_ok,
        "source_in_answer": source_in_answer,
        "source_in_trace": source_in_trace,
        "trace_sources": trace_sources,
        "refusal_ok": refusal_ok,
        "fabrication_guard_ok": fabrication_guard_ok,
        "passed": passed,
        "fact_checks": fact_checks,
        "optional_checks": optional_checks,
        "optional_matches": sum(optional_checks),
    }


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def build_report(rows, stamp):
    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    routing = sum(r["routing_ok"] for r in rows)
    facts = sum(r["facts_ok"] for r in rows)
    sourced_rows = [r for r in rows if r.get("expected_sources")]
    source_ok = sum(r["source_ok"] for r in sourced_rows)
    refusal_rows = [r for r in rows if r.get("refusal_expected")]
    refusal_ok = sum(r["refusal_ok"] for r in refusal_rows)
    latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]

    by_cat = {}
    for r in rows:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)

    lines = [
        "# Agent Evaluation Report",
        "",
        f"Run: `{stamp}`",
        "",
        "## Summary",
        "",
        f"- End-to-end pass rate: **{passed}/{total} ({pct(passed,total)}%)**",
        f"- Tool-routing accuracy: **{routing}/{total} ({pct(routing,total)}%)**",
        f"- Required-fact check rate: **{facts}/{total} ({pct(facts,total)}%)**",
    ]
    if sourced_rows:
        lines.append(f"- Source-citation check rate: **{source_ok}/{len(sourced_rows)} ({pct(source_ok,len(sourced_rows))}%)**")
    if refusal_rows:
        lines.append(f"- Unsupported-question refusal rate: **{refusal_ok}/{len(refusal_rows)} ({pct(refusal_ok,len(refusal_rows))}%)**")
    if latencies:
        lines.append(f"- Median latency: **{round(statistics.median(latencies),1)} ms**")
        if len(latencies) >= 2:
            ordered = sorted(latencies)
            idx = max(0, min(len(ordered)-1, int(round(0.95*(len(ordered)-1)))))
            lines.append(f"- Approx. p95 latency: **{round(ordered[idx],1)} ms**")

    lines += ["", "## By category", ""]
    for cat, vals in sorted(by_cat.items()):
        n = len(vals); p = sum(v["passed"] for v in vals)
        lines.append(f"- {cat}: **{p}/{n} ({pct(p,n)}%)**")

    failures = [r for r in rows if not r["passed"]]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("No failed cases in this run.")
    else:
        for r in failures:
            lines.append(f"### {r['id']} — {r['question']}")
            lines.append(f"- Expected tools: `{r['expected_tools']}`")
            lines.append(f"- Actual tools: `{r['tools_used']}`")
            lines.append(f"- Routing: {r['routing_ok']} | Facts: {r['facts_ok']} | Source: {r['source_ok']} | Refusal: {r['refusal_ok']}")
            if r.get("expected_sources"):
                lines.append(f"- Source evidence: final answer={r.get('source_in_answer')} | tool trace={r.get('source_in_trace')} | retrieved={r.get('trace_sources', [])}")
            if r.get("optional_groups"):
                lines.append(f"- Optional fact matches: {r.get('optional_matches', 0)}/{len(r.get('optional_groups', []))} (minimum {r.get('min_optional_matches', 0)})")
            lines.append(f"- Answer: {r['answer']}")
            lines.append("")

    lines += [
        "## Interpretation",
        "",
        "These are transparent, deterministic checks over a small synthetic benchmark. "
        "They measure routing, required facts, source citation, refusal behaviour, and latency. "
        "They should be treated as engineering regression checks rather than a complete measure "
        "of semantic correctness, hallucination risk, or production safety.",
    ]
    return "\n".join(lines)


def load_trace_events(trace_id):
    """Read observable events for one run from the JSONL trace file."""
    try:
        from observability import TRACE_FILE
        if not TRACE_FILE.exists():
            return []
        events = []
        with TRACE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("trace_id") == trace_id:
                    events.append(event)
        return events
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate the benchmark without calling the model")
    args = parser.parse_args()
    cases = load_cases()

    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate evaluation case IDs found")
    for c in cases:
        if not c.get("question") or "expected_tools" not in c:
            raise SystemExit(f"Invalid case: {c}")

    print(f"Validated {len(cases)} evaluation cases.")
    if args.dry_run:
        cats = {}
        for c in cases:
            cats[c["category"]] = cats.get(c["category"], 0) + 1
        print("Categories:", cats)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Use --dry-run to validate without API calls.")

    from agent import run_agent

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases)}] {case['id']}: {case['question']}")
        try:
            run = run_agent(case["question"], trace_id=f"eval-{stamp}-{case['id']}")
            answer = str(run["answer"])
            trace_events = load_trace_events(run["trace_id"])
            grade = grade_case(case, answer, run["tools_used"], trace_events)
            row = {
                **case,
                "answer": answer,
                "tools_used": run["tools_used"],
                "latency_ms": run["latency_ms"],
                "trace_id": run["trace_id"],
                **grade,
            }
        except Exception as exc:
            row = {
                **case,
                "answer": "",
                "tools_used": [],
                "latency_ms": None,
                "trace_id": f"eval-{stamp}-{case['id']}",
                "routing_ok": False,
                "facts_ok": False,
                "source_ok": False,
                "source_in_answer": None,
                "source_in_trace": None,
                "trace_sources": [],
                "refusal_ok": False,
                "fabrication_guard_ok": False,
                "optional_checks": [],
                "optional_matches": 0,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print("  PASS" if row["passed"] else "  FAIL")
        time.sleep(0.15)

    jsonl_path = RESULTS_DIR / f"eval_results_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    csv_path = RESULTS_DIR / f"eval_summary_{stamp}.csv"
    fields = ["id","category","passed","routing_ok","facts_ok","source_ok","refusal_ok","latency_ms","trace_id","question","answer"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    report_path = RESULTS_DIR / f"eval_report_{stamp}.md"
    report_path.write_text(build_report(rows, stamp), encoding="utf-8")

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
