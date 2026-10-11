"""RAG tool for the energy-operations document set."""
import os
import time
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool

from observability import log_event, record_model_call, prompt_version
from prompt_boundaries import rag_messages, RAG_SYSTEM_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="energy_docs")
RAG_PROMPT_VERSION = prompt_version(RAG_SYSTEM_PROMPT, "rag-answer")
EMBED_PROMPT_VERSION = "embedding-text-embedding-3-small-v1"


def _embed(text):
    start = time.perf_counter()
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    record_model_call(model="text-embedding-3-small", operation="rag_embedding",
        prompt_version=EMBED_PROMPT_VERSION, response=response, embedding=True,
        latency_ms=(time.perf_counter() - start) * 1000)
    return response.data[0].embedding


@tool
def search_documents(question: str) -> str:
    """Search energy-operations DOCUMENTS for procedures, thresholds, causes, or guidance.
    Use this for 'how', 'why', 'what should I do', or threshold questions.
    Returns a grounded answer with source document(s) cited."""
    started = time.perf_counter()
    q_vec = _embed(question)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = [m["source"] for m in metadatas]
    # A missing document index or empty retrieval is not evidence for an answer.
    # Decline without asking the answer model to improvise from an empty context.
    if not chunks:
        log_event("rag_empty", {"question": question, "chunk_count": 0})
        return "I couldn't find information in the provided documents."
    distances = (results.get("distances") or [[]])[0]
    context = "\n\n".join(f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources))
    log_event(
        "rag_retrieval",
        {
            "question": question,
            "sources": sources,
            "distances": [round(float(d), 6) for d in distances],
            "chunk_count": len(chunks),
        },
    )
    llm_started = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=rag_messages(question, context),
        temperature=0,
    )
    record_model_call(model="gpt-4o-mini", operation="rag_answer",
        prompt_version=RAG_PROMPT_VERSION, response=resp,
        latency_ms=(time.perf_counter() - llm_started) * 1000)
    answer = resp.choices[0].message.content
    log_event(
        "rag_answer",
        {
            "sources": sources,
            "answer": answer,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return answer


if __name__ == "__main__":
    print(search_documents.invoke({"question": "At what gearbox oil temperature must a turbine shut down?"}))
