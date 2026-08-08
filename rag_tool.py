"""
rag_tool.py — the RAG engine wrapped as a LangGraph TOOL.
The @tool decorator + docstring is what lets the agent decide to call this.
Assumes you've already run ingest.py (same as energy-rag) to build ./chroma_db.
"""
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="energy_docs")

def _embed(text):
    return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding


@tool
def search_documents(question: str) -> str:
    """Search the energy operations DOCUMENTS (manuals, procedures, safety guides,
    fault-diagnosis guides, curtailment notes) for procedures, thresholds, causes,
    or guidance. Use this for 'how', 'why', 'what should I do', or threshold questions.
    Returns a grounded answer with the source document cited."""
    q_vec = _embed(question)
    results = collection.query(query_embeddings=[q_vec], n_results=3)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources))
    system_prompt = (
        "Answer using ONLY the context below. If the answer isn't there, say so. "
        "Cite the source file(s)."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        temperature=0,
    )
    return resp.choices[0].message.content


# Quick standalone test: call the tool directly (not via an agent yet)
if __name__ == "__main__":
    # .invoke({...}) is how you call a @tool function directly
    print(search_documents.invoke({"question": "At what gearbox oil temperature must a turbine shut down?"}))
