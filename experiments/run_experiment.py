import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import datetime
from tqdm import tqdm

from config import MODELS, MAX_QUERIES
from llm.generator import generate_response
from retrieval.rag_engine import retrieve_context
from analysis.hallucination_detector import detect_hallucination
from metrics.metrics_engine import compute_metrics, compute_model_metrics
from analysis.visualize import plot_rag_vs_no_rag, plot_model_comparison

def run_experiment():
    with open("datasets/factual_qa.json", "r") as f:
        dataset = json.load(f)

    dataset = dataset[:MAX_QUERIES]

    results = []
    rag_results = []
    no_rag_results = []

    for model in MODELS:
        print(f"\nRunning model: {model}")

        for item in tqdm(dataset):
            question = item["question"]
            ground_truth = item["ground_truth"]

            response_no_rag = generate_response(model, question)

            hallucination_no_rag = detect_hallucination(
                response_no_rag["answer"],
                ground_truth
            )

            no_rag_results.append({
                "model": model,
                "question": question,
                "answer": response_no_rag["answer"],
                "latency": response_no_rag["latency"],
                "hallucination": hallucination_no_rag
            })

            
            context = retrieve_context(question)

            response_rag = generate_response(model, question, context)

            hallucination_rag = detect_hallucination(
                response_rag["answer"],
                ground_truth
            )

            rag_results.append({
                "model": model,
                "question": question,
                "answer": response_rag["answer"],
                "latency": response_rag["latency"],
                "hallucination": hallucination_rag
            })

            results.append({
                "model": model,
                "question": question,
                "no_rag_hallucination": hallucination_no_rag,
                "rag_hallucination": hallucination_rag
            })

    return results, rag_results, no_rag_results


if __name__ == "__main__":
    os.makedirs("experiment_logs", exist_ok=True)
    results, rag_results, no_rag_results = run_experiment()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    rag_metrics = compute_metrics(rag_results)
    no_rag_metrics = compute_metrics(no_rag_results)

    model_metrics = compute_model_metrics(rag_results)  

    print("\n--- RAG METRICS ---")
    print(rag_metrics)

    print("\n--- NO RAG METRICS ---")
    print(no_rag_metrics)

    print("\n--- MODEL COMPARISON (RAG) ---")
    for m, data in model_metrics.items():
        print(f"{m}: {data}")

    with open(f"experiment_logs/results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(f"experiment_logs/metrics_{timestamp}.json", "w") as f:
        json.dump({
            "rag_metrics": rag_metrics,
            "no_rag_metrics": no_rag_metrics,
            "model_metrics": model_metrics
        }, f, indent=4)
    
    summary = {
    "rag_metrics": rag_metrics,
    "no_rag_metrics": no_rag_metrics,
    "model_metrics": model_metrics
    }

    with open(f"experiment_logs/summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    plot_rag_vs_no_rag(rag_metrics, no_rag_metrics)
    plot_model_comparison(model_metrics)
  

    
