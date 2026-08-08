"""
ingest.py — build the document index (run ONCE).
Same as the energy-rag project: read docs -> chunk -> embed -> store in Chroma.
The agent's search_documents tool reads from the ./chroma_db this creates.
"""
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="energy_docs")

DOCS_DIR = "data/docs"

def chunk_text(text, size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
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
    print(f"Ingested {filename}")
print(f"\nDone. {doc_id} chunks stored.")
