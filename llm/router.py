from analysis.query_classifier import classify_query
from retrieval.rag_engine import retrieve_context
from llm.generator import generate_response
from config import MODEL_NAME


def route_query(query):
    query_type = classify_query(query)

    if query_type == "factual":
        context = retrieve_context(query)
        return generate_response(MODEL_NAME, query, context)

    elif query_type == "coding":
        return generate_response(MODEL_NAME, query)

    elif query_type == "reasoning":
        prompt = f"Explain step by step: {query}"
        return generate_response(MODEL_NAME, prompt)

    else:
        return generate_response(MODEL_NAME, query)