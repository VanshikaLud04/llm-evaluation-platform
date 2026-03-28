from analysis.query_classifier import classify_query
from retrieval.rag_engine import retrieve_context
from llm.generator import generate_response
from config import MODEL_NAME


def route_query(query: str, model: str = None) -> dict:
    
    model = model or MODEL_NAME

    query_type = classify_query(query)

    if query_type == "factual":
        context = retrieve_context(query)
        result = generate_response(model, query, context=context, mode="rag")

    elif query_type == "coding":
        result = generate_response(model, query, mode="basic")

    elif query_type == "reasoning":
        result = generate_response(model, query, mode="cot")

    else:
        result = generate_response(model, query, mode="basic")

    result["query_type"] = query_type
    return result