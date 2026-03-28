from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

# Loaded once at module level
_model = SentenceTransformer("all-MiniLM-L6-v2")

_DEFAULT_DOCUMENTS = [
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "Paris is the capital and largest city of France.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "Jupiter is the largest planet in the solar system.",
    "William Shakespeare wrote Romeo and Juliet.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "The chemical symbol for gold is Au.",
    "World War II ended in 1945.",
    "The mitochondria is the powerhouse of the cell.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
]


_documents = list(_DEFAULT_DOCUMENTS)
_embeddings = _model.encode(_documents)


def load_documents_from_file(filepath: str):
    
    global _documents, _embeddings

    if not os.path.exists(filepath):
        print(f"[rag_engine] File not found: {filepath}. Using defaults.")
        return

    with open(filepath) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if all(isinstance(item, str) for item in raw):
            _documents = raw
        elif all(isinstance(item, dict) for item in raw):
            # Support {"text": "..."} or {"content": "..."} format
            _documents = [item.get("text") or item.get("content", "") for item in raw]
        else:
            print("[rag_engine] Unrecognized format. Using defaults.")
            return

    _embeddings = _model.encode(_documents)
    print(f"[rag_engine] Loaded {len(_documents)} documents from {filepath}")


def add_documents(new_docs: list):
    
    global _documents, _embeddings

    if not new_docs:
        return

    _documents.extend(new_docs)
    new_embeddings = _model.encode(new_docs)
    _embeddings = np.vstack([_embeddings, new_embeddings])


def retrieve_context(query: str, top_k: int = 2) -> str:
   
    query_embedding = _model.encode([query])[0]

    # Cosine similarity via dot product (embeddings are L2-normalised by default)
    similarities = np.dot(_embeddings, query_embedding)

    # Get top_k indices sorted by descending similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved = [_documents[i] for i in top_indices]
    return "\n\n".join(retrieved)