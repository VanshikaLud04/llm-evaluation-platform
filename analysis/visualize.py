import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  


def plot_rag_vs_no_rag(
    rag_metrics: dict,
    no_rag_metrics: dict,
    save_path: str = "experiment_logs/plot_rag_vs_no_rag.png",
):
    """Bar chart comparing hallucination rate with and without RAG."""
    labels = ["No RAG", "RAG"]
    values = [
        no_rag_metrics.get("hallucination_rate", 0),
        rag_metrics.get("hallucination_rate", 0),
    ]
    colors = ["#ff4444", "#00cc6e"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.4, edgecolor="black")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("RAG vs No-RAG — Hallucination Rate")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")


def plot_model_comparison(
    model_metrics: dict,
    save_path: str = "experiment_logs/plot_model_comparison.png",
):
    """Side-by-side bar chart of hallucination rate and avg latency per model."""
    models = list(model_metrics.keys())
    hall_rates = [model_metrics[m]["hallucination_rate"] for m in models]
    avg_latencies = [model_metrics[m]["avg_latency"] for m in models]

    x = range(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Hallucination rate
    bars1 = ax1.bar([i - width / 2 for i in x], hall_rates, width, color="#ff4444", label="Hallucination Rate")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Rate")
    ax1.set_title("Hallucination Rate by Model")
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars1, hall_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9)

    # Latency
    bars2 = ax2.bar([i + width / 2 for i in x], avg_latencies, width, color="#4488ff", label="Avg Latency (s)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(models)
    ax2.set_ylabel("Seconds")
    ax2.set_title("Avg Latency by Model")
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars2, avg_latencies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")