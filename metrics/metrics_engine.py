def compute_model_metrics(results):
    models = {}

    for r in results:
        m = r["model"]

        if m not in models:
            models[m] = {"total": 0, "hallucinations": 0, "latency": 0}

        models[m]["total"] += 1
        models[m]["hallucinations"] += int(r["hallucination"])
        models[m]["latency"] += r["latency"]

    for m in models:
        models[m]["hallucination_rate"] = (
            models[m]["hallucinations"] / models[m]["total"]
        )
        models[m]["avg_latency"] = (
            models[m]["latency"] / models[m]["total"]
        )

    return models