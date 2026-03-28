import matplotlib.pyplot as plt


def plot_rag_vs_no_rag(rag_metrics, no_rag_metrics):
    labels = ["No RAG", "RAG"]
    values = [
        no_rag_metrics["hallucination_rate"],
        rag_metrics["hallucination_rate"]
    ]

    plt.bar(labels, values)
    plt.title("RAG vs No-RAG (Hallucination Rate)")
    plt.ylabel("Rate")

    plt.show()


def plot_model_comparison(model_metrics):
    models = list(model_metrics.keys())
    values = [
        model_metrics[m]["hallucination_rate"]
        for m in models
    ]

    plt.bar(models, values)
    plt.title("Model Comparison (Hallucination Rate)")
    plt.ylabel("Rate")

    plt.show()