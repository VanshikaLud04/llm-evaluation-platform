from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import os

from llm.router import route_query
from retrieval.rag_engine import retrieve_context
from llm.generator import generate_response
from analysis.hallucination_detector import detect_hallucination

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("frontend/dist"):
    app.mount("/app", StaticFiles(directory="frontend/dist", html=True), name="frontend")


@app.get("/")
def home():
    return {"message": "LLM Evaluation API is running"}


@app.get("/ask")
def ask(query: str, model: str = "phi"):
  
    result = route_query(query, model=model)
    return {
        "query": query,
        "answer": result["answer"],
        "latency": result["latency"],
        "mode": result.get("mode", "unknown"),
        "model": result.get("model", model),
        "hallucination": result.get("hallucination", None),
    }


@app.get("/compare")
def compare(query: str, model: str = "phi"):
    
    no_rag = generate_response(model, query, mode="basic")

    context = retrieve_context(query)
    rag = generate_response(model, query, context=context, mode="rag")

    return {
        "query": query,
        "model": model,
        "no_rag": {
            "answer": no_rag["answer"],
            "latency": no_rag["latency"],
            "mode": no_rag["mode"],
        },
        "rag": {
            "answer": rag["answer"],
            "latency": rag["latency"],
            "mode": rag["mode"],
            "context_used": context,
        },
    }

@app.get("/metrics")
def get_metrics():
    folder = "experiment_logs"

    if not os.path.exists(folder):
        return {"error": "No experiment logs found. Run experiments/run_experiment.py first."}

    files = sorted([f for f in os.listdir(folder) if "metrics" in f])

    if not files:
        return {"error": "No metrics files found in experiment_logs/"}

    latest_file = files[-1]
    with open(os.path.join(folder, latest_file)) as f:
        data = json.load(f)

    data["_source_file"] = latest_file
    return data


@app.get("/models")
def list_models():
    
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")[1:]  
        models = []
        for line in lines:
            if line.strip():
                name = line.split()[0].split(":")[0]
                models.append(name)
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}