from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

categories = {
    "factual": "questions about facts, people, places, history",
    "coding": "questions about programming, code, algorithms",
    "reasoning": "questions asking explanation, why, how, concepts",
    "general": "casual or general questions"
}

category_embeddings = {
    k: model.encode([v])[0] for k, v in categories.items()
}


def classify_query(query):
    query_embedding = model.encode([query])[0]

    scores = {}

    for category, emb in category_embeddings.items():
        score = cosine_similarity([query_embedding], [emb])[0][0]
        scores[category] = score

    best_category = max(scores, key=scores.get)

    return best_category