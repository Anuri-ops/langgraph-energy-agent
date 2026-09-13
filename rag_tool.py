"""RAG tool for the energy-operations document set."""
import os
import time
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool

from observability import log_event

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="energy_docs")


def _embed(text):
    return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding


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
    system_prompt = (
        "Answer using ONLY the context below. If the answer isn't there, say so. "
        "Cite the source file(s)."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
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
