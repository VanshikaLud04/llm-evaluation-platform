from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def detect_hallucination(answer, ground_truth):
    emb1 = model.encode([answer])
    emb2 = model.encode([ground_truth])

    similarity = cosine_similarity(emb1, emb2)[0][0]

    if similarity > 0.7:
        return False
    else:
        return True