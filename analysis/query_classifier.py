from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = SentenceTransformer("all-MiniLM-L6-v2")

_CATEGORIES = {
    "factual": "questions about facts, people, places, dates, history, science",
    "coding": "questions about programming, code, debugging, algorithms, software",
    "reasoning": "questions asking why, how, explanation, analysis, concepts, logic",
    "general": "casual conversation, opinions, greetings, general questions",
}

_category_embeddings = {
    k: _model.encode([v])[0] for k, v in _CATEGORIES.items()
}


def classify_query(query: str) -> str:
    query_embedding = _model.encode([query])[0]

    scores = {
        category: float(cosine_similarity([query_embedding], [emb])[0][0])
        for category, emb in _category_embeddings.items()
    }

    best_category = max(scores, key=scores.get)
    return best_category


def classify_query_with_scores(query: str) -> dict:
   
    query_embedding = _model.encode([query])[0]

    scores = {
        category: round(float(cosine_similarity([query_embedding], [emb])[0][0]), 4)
        for category, emb in _category_embeddings.items()
    }

    return {
        "category": max(scores, key=scores.get),
        "scores": scores,
    }