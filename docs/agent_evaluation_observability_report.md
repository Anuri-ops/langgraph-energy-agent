# Agent Evaluation & Observability Report
## Agentic RAG/NL2SQL Energy Operations Assistant

**Purpose:** Short technical report for interview/application review  
**Evaluation scope:** 26-case deterministic regression benchmark  
**Final result:** 26/26 cases passed on the unchanged benchmark

---

## 1. Executive summary

I built a small evaluation and observability layer around an agentic energy-operations assistant that can use two tools:

- **Document retrieval** for procedures, maintenance guidance, thresholds, safety rules and technical documentation.
- **Operational-data querying** for asset status, dates, counts, averages, anomalies and other structured records.

The evaluation suite measures:

- tool routing,
- factual correctness,
- source grounding,
- unsupported-query refusal,
- combined multi-tool workflow completion,
- and execution latency.

The first valid benchmark run scored **23/26**. Review of the three failures showed that they were not all agent failures:

1. one was a genuine agent-routing defect,
2. two were evaluator false negatives caused by overly literal grading.

I corrected the evaluator without changing the benchmark cases, added a general routing/refusal policy to the agent, and reran the same 26-case suite. The final run scored **26/26**.

This should be interpreted as a **regression-test result on a small controlled benchmark**, not as a claim of production-wide 100% reliability.

---

## 2. Architecture summary

The assistant follows a simple agentic pattern:

**User question → LLM agent → tool selection → tool execution → grounded response**

The agent has two callable tools:

### `search_documents`
Used for questions about:

- operating procedures,
- maintenance guidance,
- safety rules,
- technical thresholds,
- inspection findings,
- troubleshooting guidance,
- and other document-based knowledge.

The retrieval path uses a persistent vector store built from the project documents. Retrieved context and source information are exposed to the answer-generation step and recorded in the trace.

### `query_operational_data`
Used for questions requiring structured operational facts such as:

- asset status,
- dates,
- power output,
- anomaly flags,
- counts,
- averages,
- and filtered records.

The tool translates the question into a read-only structured query and returns the matching operational records.

### Agent orchestration
The agent decides which tool to call and can use both tools for questions that combine operational data with document guidance.

After the baseline evaluation, a general system policy was added to make the routing behaviour explicit:

- document questions should use document retrieval,
- operational-record questions should use the data tool,
- both tools should be used only when the question requires both,
- an unsuccessful document lookup should not automatically fall through to the operational database,
- missing values should not be invented or inferred.

### Observability
The project records structured JSONL trace events for observable execution behaviour, including:

- run start and completion,
- tool requests and tool usage,
- retrieved document sources,
- generated structured queries,
- result previews,
- errors,
- final answers,
- and latency.

The tracing layer does **not** record hidden chain-of-thought.

---

## 3. Evaluation design

The benchmark contains **26 cases**:

| Category | Cases | Purpose |
|---|---:|---|
| Document / RAG | 10 | Retrieve and answer from technical documentation |
| SQL / structured data | 8 | Query operational records correctly |
| Combined | 5 | Use both document and structured-data tools in one workflow |
| Refusal | 3 | Decline unsupported questions without fabricating values |
| **Total** | **26** | End-to-end regression suite |

The grader checks:

- exact expected tool set,
- required factual content,
- source-grounding evidence where applicable,
- refusal behaviour for unsupported questions,
- anti-fabrication patterns,
- and end-to-end pass/fail.

The benchmark cases were **not changed** between the baseline and final run.

---

## 4. Baseline result

**Baseline: 23/26 passed (88.5%)**

The three failed cases were:

| Case | Baseline diagnosis | What actually happened |
|---|---|---|
| `rag_11` | Refusal detector false negative | The agent correctly said it could not find the requested battery-room humidity limit, but the grader did not recognise the phrase **"couldn't find"** as a valid refusal. |
| `sql_02` | Literal fact-matching false negative | The agent returned the correct assets, **Site2 Wind** and **Site4 Solar**, but the expected values used underscore formatting such as `Site2_Wind`. |
| `refusal_03` | Agent unnecessarily invoked SQL after document miss; refusal phrasing also unrecognised | The agent correctly declined to invent the missing WT-2000 blade-tip-speed value, but unnecessarily called the operational-data tool after document retrieval returned no answer. The refusal detector also failed to recognise the phrase **"couldn't find"**. Both the routing policy and refusal-marker correction contributed to the final pass. |

Baseline tool-routing accuracy was **25/26**.  
Baseline unsupported-question refusal score was **1/3** under the original evaluator.

---

## 5. Changes made after failure analysis

### 5.1 Agent change — behavioural fix

A general routing/refusal system policy was added to the agent.

This addressed the genuine defect exposed by `refusal_03`: after a document-only lookup failed, the agent had unnecessarily called the operational-data tool even though the question was about a technical specification rather than an operational record.

The fix was deliberately general rather than case-specific.

### 5.2 Evaluator change — refusal detection

The refusal phrase list was expanded to recognise common, semantically valid refusal wording such as:

- "couldn't find",
- "could not find",
- "unable to find",
- and "no specific information".

This corrected the measurement problem seen in `rag_11` and also contributed to `refusal_03` being scored correctly.

### 5.3 Evaluator change — text normalization

The fact matcher was normalised so superficial formatting differences such as:

- `Site2_Wind`
- `Site2 Wind`

are treated as equivalent.

This corrected the false negative in `sql_02`.

### 5.4 What was **not** weakened

The benchmark cases remained unchanged.

The anti-fabrication pattern guards were **not changed at any point between the baseline and final run**. Unsupported questions therefore still fail if the answer invents a prohibited value. This is important: the refusal score improved because valid refusal behaviour was recognised and the routing behaviour was corrected, not because the evaluator's anti-fabrication bar was weakened.

---

## 6. Final regression result

After the agent and evaluator fixes, the unchanged benchmark was rerun.

**Final result: 26/26 passed (100% on this regression suite)**

Key final checks:

- **End-to-end:** 26/26
- **Tool routing:** 26/26
- **Unsupported-query refusal:** 3/3
- **Document questions:** all passed
- **Structured-data questions:** all passed
- **Combined workflows:** all passed

The **observed median latency in the final run was 4.1 s**.

That latency value is reported as an observation only. With a small number of API-backed runs and naturally variable response times, it should not be interpreted as proof that the changes caused a latency improvement.

---

## 7. Engineering interpretation

The most useful outcome was not the final 26/26 score by itself. The evaluation process exposed two different classes of problem:

1. **Behavioural defect:** unnecessary tool escalation after a failed document lookup.
2. **Measurement defects:** a refusal detector that missed valid phrasing and a fact matcher that was too formatting-sensitive.

Separating those categories mattered. Treating all three baseline failures as model failures would have led to unnecessary changes to the agent. Treating all three as grader problems would have hidden a real orchestration weakness.

The final result therefore represents:

**build → measure → inspect failures → separate model and measurement errors → apply targeted fixes → rerun regression suite**

That workflow is the main evidence of the system's engineering maturity.

---

## 8. Scope and limitations

This benchmark is intentionally small and controlled. It contains 26 synthetic evaluation cases over a limited project dataset.

Therefore:

- **26/26 does not mean the agent is 100% accurate in production.**
- It does not establish a production reliability rate.
- It does not replace broader adversarial, domain, safety, load or long-horizon testing.
- It should be treated as an engineering regression suite for known behaviours.

A production evaluation programme would expand the benchmark with:

- more paraphrases and ambiguous requests,
- larger and noisier document collections,
- more complex multi-step tool workflows,
- malformed or incomplete operational data,
- adversarial unsupported questions,
- repeated-run variance analysis,
- cost and latency distributions,
- and domain-expert review.

---

## 9. Short CV version

> **Built a 26-case evaluation harness and JSONL tracing layer for an agentic RAG/NL2SQL system, measuring tool routing, factual correctness, source grounding and unsupported-query refusal; diagnosed one agent-routing defect and two evaluator false negatives, then achieved 26/26 on the unchanged regression suite.**

---

## 10. Interview/demo summary

> I added a deterministic evaluation harness and structured tracing to an agent that combines document retrieval with operational-data querying. The first valid run scored 23 out of 26. When I inspected the failures, two were evaluator false negatives and one exposed a genuine routing problem: after failing to find a technical specification in the documents, the agent unnecessarily queried the operational database. I corrected the evaluator, added a general routing and refusal policy to the agent, and reran the same unchanged benchmark. The final regression passed 26 out of 26 cases. I treat that as a regression result, not as a claim of production-wide 100% reliability.
