from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Penicillin was discovered by Alexander Fleming.",
    "Paris is the capital of France.",
    "Albert Einstein developed the theory of relativity.",
    "Jupiter is the largest planet in the solar system."
]

doc_embeddings = model.encode(documents)


def retrieve_context(query, top_k=1):
    query_embedding = model.encode([query])[0]

    similarities = np.dot(doc_embeddings, query_embedding)

    top_index = np.argmax(similarities)

    return documents[top_index]