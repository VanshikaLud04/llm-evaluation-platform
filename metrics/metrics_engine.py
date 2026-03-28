def compute_metrics(results: list) -> dict:
  
    if not results:
        return {"hallucination_rate": 0.0, "avg_latency": 0.0, "total": 0}

    total = len(results)
    hallucinations = sum(1 for r in results if r.get("hallucination"))
    total_latency = sum(r.get("latency", 0) for r in results)

    return {
        "total": total,
        "hallucinations": hallucinations,
        "hallucination_rate": round(hallucinations / total, 4),
        "avg_latency": round(total_latency / total, 4),
    }


def compute_model_metrics(results: list) -> dict:
   
    models = {}

    for r in results:
        m = r.get("model", "unknown")

        if m not in models:
            models[m] = {"total": 0, "hallucinations": 0, "latency": 0.0}

        models[m]["total"] += 1
        models[m]["hallucinations"] += int(bool(r.get("hallucination")))
        models[m]["latency"] += r.get("latency", 0)

    for m in models:
        t = models[m]["total"]
        models[m]["hallucination_rate"] = round(models[m]["hallucinations"] / t, 4)
        models[m]["avg_latency"] = round(models[m]["latency"] / t, 4)

    return models