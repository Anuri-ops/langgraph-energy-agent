"""
app.py — Streamlit UI for the LangGraph agent.
- Builds the document index on startup if missing (deploy-safe).
- Per-session rate limiting, TIGHTER than the RAG app because an agent LOOPS
  (several LLM calls per question), so each question costs more.
Run locally:  streamlit run app.py
"""
import os
import streamlit as st

# Build the RAG index on first startup if it isn't there (Streamlit's server has no chroma_db).
@st.cache_resource
def ensure_index():
    import chromadb
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="energy_docs")
    if collection.count() > 0:
        return True
    DOCS_DIR = "data/docs"
    def chunk_text(text, size=500, overlap=50):
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start + size]); start += size - overlap
        return chunks
    def embed(text):
        return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    doc_id = 0
    for filename in os.listdir(DOCS_DIR):
        with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text):
            collection.add(ids=[f"doc_{doc_id}"], embeddings=[embed(chunk)],
                           documents=[chunk], metadatas=[{"source": filename}])
            doc_id += 1
    return True

# TIGHTER cap than the RAG app: an agent can make several LLM calls per question.
MAX_QUESTIONS_PER_SESSION = 5

st.set_page_config(page_title="LangGraph Energy Agent", page_icon="⚡")
st.title("⚡ LangGraph Energy Agent")
st.write(
    "An **agentic** assistant: an LLM agent decides on its own which tools to call — "
    "document search (RAG) or the operational data (SQL) — and for a combined question it "
    "calls both and synthesises the answer."
)

with st.spinner("Preparing knowledge base..."):
    ensure_index()

# Import the agent after the index is ready
from agent import agent

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

with st.expander("Example questions"):
    st.markdown(
        "**Document**\n- At what gearbox oil temperature must a turbine be shut down?\n\n"
        "**Data**\n- How many assets had a FAULT status?\n\n"
        "**Combined (agent uses both tools)**\n- Which assets had a FAULT, and what should I do about a fault?"
    )

remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
st.caption(f"Demo limit: {remaining} question(s) remaining this session.")

question = st.text_input("Your question:")

if question:
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("You've reached the demo question limit for this session. "
                   "This protects the demo from overuse. Thanks for trying it!")
    else:
        with st.spinner("Agent reasoning (may call one or more tools)..."):
            result = agent.invoke({"messages": [("user", question)]})
            answer = result["messages"][-1].content
            # show which tools the agent chose, for transparency
            tools_used = [m.name for m in result["messages"] if getattr(m, "name", None)]
        st.session_state.question_count += 1
        st.markdown("### Answer")
        st.write(answer)
        if tools_used:
            st.caption("Tools the agent chose: " + ", ".join(dict.fromkeys(tools_used)))

st.divider()
st.caption(
    "Built with Python, LangGraph, LangChain, OpenAI, and Chroma. The agent orchestrates its own "
    "tools (RAG + NL-to-SQL) in a reasoning loop. Data is synthetic. A demonstration of agentic AI."
)
