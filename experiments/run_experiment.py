import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import datetime
from tqdm import tqdm

from config import MODELS, MAX_QUERIES
from llm.generator import generate_response
from retrieval.rag_engine import retrieve_context
from analysis.hallucination_detector import detect_hallucination, similarity_score
from metrics.metrics_engine import compute_metrics, compute_model_metrics
from analysis.visualize import plot_rag_vs_no_rag, plot_model_comparison


def run_experiment(dataset_path: str = "datasets/factual_qa.json"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    dataset = dataset[:MAX_QUERIES]

    rag_results = []
    no_rag_results = []
    comparison = []

    for model in MODELS:
        print(f"\n── Model: {model} ──────────────────────────────")

        for item in tqdm(dataset):
            question = item["question"]
            ground_truth = item["ground_truth"]

            resp_no_rag = generate_response(model, question, mode="basic")
            hall_no_rag = detect_hallucination(resp_no_rag["answer"], ground_truth)
            sim_no_rag = similarity_score(resp_no_rag["answer"], ground_truth)

            no_rag_results.append({
                "model": model,
                "question": question,
                "ground_truth": ground_truth,
                "answer": resp_no_rag["answer"],
                "latency": resp_no_rag["latency"],
                "hallucination": hall_no_rag,
                "similarity": sim_no_rag,
            })

            context = retrieve_context(question)
            resp_rag = generate_response(model, question, context=context, mode="rag")
            hall_rag = detect_hallucination(resp_rag["answer"], ground_truth)
            sim_rag = similarity_score(resp_rag["answer"], ground_truth)

            rag_results.append({
                "model": model,
                "question": question,
                "ground_truth": ground_truth,
                "answer": resp_rag["answer"],
                "latency": resp_rag["latency"],
                "hallucination": hall_rag,
                "similarity": sim_rag,
                "context_used": context,
            })

            comparison.append({
                "model": model,
                "question": question,
                "no_rag_hallucination": hall_no_rag,
                "rag_hallucination": hall_rag,
                "no_rag_similarity": sim_no_rag,
                "rag_similarity": sim_rag,
            })

    return comparison, rag_results, no_rag_results


if __name__ == "__main__":
    os.makedirs("experiment_logs", exist_ok=True)

    comparison, rag_results, no_rag_results = run_experiment()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compute aggregate metrics
    rag_metrics = compute_metrics(rag_results)
    no_rag_metrics = compute_metrics(no_rag_results)
    model_metrics = compute_model_metrics(rag_results)

    print("\n── RAG METRICS ────────────────────────────────")
    print(json.dumps(rag_metrics, indent=2))

    print("\n── NO RAG METRICS ─────────────────────────────")
    print(json.dumps(no_rag_metrics, indent=2))

    print("\n── PER-MODEL METRICS (RAG) ────────────────────")
    for m, data in model_metrics.items():
        print(f"  {m}: hallucination={data['hallucination_rate']:.0%}  latency={data['avg_latency']:.2f}s")

    # Save all outputs
    with open(f"experiment_logs/results_{timestamp}.json", "w") as f:
        json.dump(comparison, f, indent=4)

    with open(f"experiment_logs/metrics_{timestamp}.json", "w") as f:
        json.dump({
            "rag_metrics": rag_metrics,
            "no_rag_metrics": no_rag_metrics,
            "model_metrics": model_metrics,
        }, f, indent=4)

    with open(f"experiment_logs/summary_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": MODELS,
            "total_questions": len(rag_results) // len(MODELS),
            "rag_metrics": rag_metrics,
            "no_rag_metrics": no_rag_metrics,
            "model_metrics": model_metrics,
        }, f, indent=4)

    print(f"\n✓ Logs saved to experiment_logs/ with timestamp {timestamp}")

    # Generate plots (saved)
    plot_rag_vs_no_rag(rag_metrics, no_rag_metrics, save_path=f"experiment_logs/plot_rag_{timestamp}.png")
    plot_model_comparison(model_metrics, save_path=f"experiment_logs/plot_models_{timestamp}.png")
    print("✓ Plots saved")
