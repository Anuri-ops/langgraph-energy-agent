"""Container startup without leaking keys or generating embeddings at image build.

AUTO_INGEST_ON_START=1 is opt-in for single-replica demos with an initially empty
index. It calls the *existing* ingest.py once only if there are no stored docs.
An empty/ephemeral cloud volume may trigger paid embeddings again after restart.
"""
from __future__ import annotations

import os
import subprocess
import sys


def prepare_index_if_enabled() -> bool:
    if os.getenv("AUTO_INGEST_ON_START", "0") != "1":
        return False
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="energy_docs")
    if collection.count() > 0:
        print("Existing RAG document index found; skipping ingestion.", flush=True)
        return False
    print("Empty RAG index; performing one-time ingestion (OpenAI API usage may be billed).", flush=True)
    subprocess.run([sys.executable, "ingest.py"], check=True)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Container entrypoint requires a command.")
    prepare_index_if_enabled()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
