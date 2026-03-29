# LLM Evaluation Platform

A self-built benchmarking framework for local LLMs — evaluates hallucination rate, latency, and RAG effectiveness across multiple models on a 100-question factual QA dataset.

Built as an open-source alternative to LangSmith for **locally-run models via Ollama**, with zero cloud dependencies.

---

## Demo

### Experiment running — Phi and TinyLlama across 100 questions
![Experiment run](docs/experiment.png)

### Frontend — RAG vs No-RAG comparison on a live query
![Frontend UI](docs/frontend.png)

---

## Key Findings

Benchmarked **Phi** and **TinyLlama** across **100 factual QA pairs** (50 questions × 2 models), evaluated with and without RAG.

### Overall: RAG vs No RAG

| Mode | Hallucination Rate | Avg Latency |
|---|---|---|
| No RAG | **28%** | 0.55s |
| RAG | 55% | 0.46s |

> **Counterintuitive finding:** RAG *increased* hallucination rate by 27 percentage points. A small knowledge base (10 chunks covering ~10% of questions) caused retrieval to return irrelevant context — actively misleading the models rather than grounding them. This shows that **RAG quality depends entirely on knowledge base coverage**, not just retrieval architecture.

### Per-Model Breakdown (RAG mode)

| Model | Hallucination Rate | Avg Latency |
|---|---|---|
| **Phi** | 26% | 0.57s |
| **TinyLlama** | 84% | 0.35s |

### What the numbers show

**1. Model size dominates accuracy — 58 point gap.**
Phi (2.7B) hallucinated on 26% of questions vs TinyLlama's 84% under identical conditions. For factual QA, model capability matters far more than retrieval architecture when the knowledge base is small.

**2. TinyLlama is fast but unusable for factual tasks.**
At 0.35s avg latency, TinyLlama is ~38% faster than Phi — but an 84% hallucination rate makes it unsuitable for any production use case where answers need to be correct.

**3. RAG hurt both models when knowledge base coverage was low.**
The knowledge base covered ~10 of 100 questions. For the other 90, retrieval returned loosely related chunks that added noise. Both models anchored on wrong context instead of using their parametric knowledge — a known RAG failure mode at low coverage.

**4. Query routing eliminated unnecessary retrieval overhead.**
The semantic classifier correctly bypassed RAG for coding and reasoning queries, saving ~0.1s per non-factual query with no accuracy loss.

---

## Architecture

```
llm-evaluation-platform/
│
├── api/
│   └── app.py                     # FastAPI — /ask, /compare, /metrics, /models
│
├── llm/
│   ├── generator.py               # Prompt builder + Ollama query (5 modes: basic, rag, strict, cot, verify)
│   └── router.py                  # Routes query type → generation mode
│
├── retrieval/
│   └── rag_engine.py              # Semantic retrieval via sentence-transformers
│
├── analysis/
│   ├── query_classifier.py        # Classifies: factual / coding / reasoning / general
│   ├── hallucination_detector.py  # Cosine similarity vs ground truth (threshold: 0.5)
│   └── visualize.py               # Saves comparison bar charts as PNG
│
├── metrics/
│   └── metrics_engine.py          # Per-model hallucination rate + latency aggregation
│
├── experiments/
│   └── run_experiment.py          # Batch runner — all models × all questions
│
├── datasets/
│   └── factual_qa.json            # 100 factual QA pairs with ground truth (8 categories)
│
└── frontend/
    └── src/App.jsx                # React UI — ask / compare / metrics
```

---

## Pipeline Flow

```
User Query
    │
    ▼
QueryClassifier              ← sentence-transformers cosine similarity
    │
    ├── factual   ──→  RAGEngine.retrieve()  ──→  Generator (rag mode)
    ├── coding    ──────────────────────────────→  Generator (basic mode)
    ├── reasoning ──────────────────────────────→  Generator (cot mode)
    └── general   ──────────────────────────────→  Generator (basic mode)
                                                        │
                                                        ▼
                                               HallucinationDetector
                                               (cosine similarity vs ground truth)
                                                        │
                                                        ▼
                                               MetricsEngine → experiment_logs/
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- Models pulled:
  ```bash
  ollama pull phi
  ollama pull tinyllama
  ollama pull mistral
  ```

### Install

```bash
git clone https://github.com/VanshikaLud04/llm-evaluation-platform.git
cd llm-evaluation-platform
pip install -r requirements.txt
```

### Run a batch experiment

```bash
python experiments/run_experiment.py
```

Saves timestamped JSON results + PNG plots to `experiment_logs/`.

### Run the API

```bash
# Run from project root
uvicorn api.app:app --reload --port 8000
```

### Run the frontend

```bash
cd frontend
npm install && npm run dev
# → http://localhost:3000
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /ask?query=...&model=phi` | Route query through classifier → generator |
| `GET /compare?query=...&model=phi` | RAG vs no-RAG side by side |
| `GET /metrics` | Latest experiment metrics from logs |
| `GET /models` | Available Ollama models on this machine |

### Example — `/compare`

```json
{
  "query": "What is the capital of France?",
  "model": "mistral",
  "no_rag": {
    "answer": "The capital of France is Paris.",
    "latency": 126.78
  },
  "rag": {
    "answer": "Paris is the capital of France.",
    "latency": 41.66,
    "context_used": "Paris is the capital and largest city of France."
  }
}
```

---

## Hallucination Detection

Uses semantic similarity via `sentence-transformers` (`all-MiniLM-L6-v2`). An answer is flagged as hallucination if cosine similarity to ground truth falls below **0.5**.

Threshold chosen empirically — the naive 0.7 threshold produced false positives on short correct answers like "Paris" or "Au" because sentence embeddings of short strings naturally score lower in magnitude.

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM inference | Ollama (local, no API key) |
| Backend | FastAPI + uvicorn |
| Semantic search + hallucination | sentence-transformers (`all-MiniLM-L6-v2`) |
| Metrics + logging | Python + JSON |
| Visualisation | matplotlib |
| Frontend | React + Vite |

---

## Adding a new model

```bash
ollama pull llama3
```

No code changes needed — it auto-appears in `/models` and the frontend dropdown.

---

## Limitations

- Knowledge base covers ~10% of dataset — RAG results reflect low-coverage retrieval, not RAG at scale
- Hallucination detection uses cosine similarity, not NLI — may miss subtle factual errors that are semantically close but factually wrong
- Only 2 models benchmarked — results may not generalise to larger models
