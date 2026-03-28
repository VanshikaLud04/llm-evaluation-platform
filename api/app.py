from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os

from llm.router import route_query
from retrieval.rag_engine import retrieve_context
from llm.generator import generate_response

app = FastAPI()

@app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "LLM Evaluation API running"}

@app.get("/ask")
def ask(query: str):
    result = route_query(query)

    return {
        "query": query,
        "answer": result["answer"],
        "latency": result["latency"],
        "mode": result.get("mode", "unknown"),
        "hallucination": result.get("hallucination", None)
    }

@app.get("/compare")
def compare(query: str):
    no_rag = generate_response("phi", query)

    context = retrieve_context(query)
    rag = generate_response("phi", query, context)

    return {
        "no_rag": no_rag,
        "rag": rag
    }

@app.get("/metrics")
def get_metrics():
    folder = "experiment_logs"

    files = sorted([f for f in os.listdir(folder) if "metrics" in f])

    if not files:
        return {"error": "No metrics found"}

    latest_file = files[-1]

    with open(os.path.join(folder, latest_file)) as f:
        return json.load(f)