# ⚡ LangGraph Energy Agent (Agentic AI)

**🔗 Live demo:** https://anurilanggraph-energy-agent.streamlit.app/


An **agentic** AI assistant for energy operations, built with **LangGraph**. Unlike a hand-coded
router, here an LLM agent decides on its own which tools to call — and for a combined question it
calls multiple tools in sequence and synthesises the answer.

This is the agentic counterpart to a hand-built RAG + orchestrator project: same domain, but the
LLM orchestrates its own tool use in a reasoning loop (the ReAct pattern) rather than following
if/else routing I wrote.

## What it does
The agent has two tools and chooses between them per question:
- **search_documents** — RAG over energy-operations documents (procedures, thresholds, guidance), grounded and cited.
- **query_operational_data** — natural-language-to-SQL over an operational data table (counts, statuses, anomalies; read-only queries).

For a combined question ("which assets faulted, and what should I do about it?"), the agent
**decomposes it itself**, calls both tools, and merges the results — no hand-written routing.

## Architecture (LangGraph)
- **State** — the message history that flows through the graph.
- **Agent node** — the LLM decides: call a tool, or answer.
- **Tool node** — runs the chosen tool; the result returns to the agent.
- **Conditional edge** — loops agent → tool → agent until the agent produces a final answer.

## Tech stack
- **Python**, **LangGraph** + **LangChain**, **OpenAI** (gpt-4o-mini + text-embedding-3-small), **Chroma** (vector store)

## Files
- `hello_graph.py` — a minimal graph demonstrating state / node / edge
- `rag_tool.py` — the documents tool (RAG)
- `sql_tool.py` — the data tool (NL-to-SQL)
- `agent.py` — the two-tool agent (the ReAct loop)

## Running locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
# add your key to a .env file:  OPENAI_API_KEY=sk-...
python ingest.py               # build the document index (once)
python agent.py                # run the agent on sample questions
```

## Data
All data is **synthetic** and created for demonstration — no proprietary content.

## Notes
Built independently to demonstrate agentic AI fundamentals — tools, the ReAct loop, and
LLM-driven tool orchestration — using LangGraph.
