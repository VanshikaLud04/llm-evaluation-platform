from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.5


def detect_hallucination(answer: str, ground_truth: str) -> bool:
    
    if not ground_truth or not ground_truth.strip():
        # No ground truth to compare against — can't judge
        return False

    if not answer or not answer.strip():
        return True

    emb_answer = _model.encode([answer])
    emb_truth = _model.encode([ground_truth])

    similarity = cosine_similarity(emb_answer, emb_truth)[0][0]

    return float(similarity) < SIMILARITY_THRESHOLD


def similarity_score(answer: str, ground_truth: str) -> float:
  
    if not answer or not ground_truth:
        return 0.0

    emb_answer = _model.encode([answer])
    emb_truth = _model.encode([ground_truth])

    return round(float(cosine_similarity(emb_answer, emb_truth)[0][0]), 4)