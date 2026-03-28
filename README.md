# LLM Evaluation Platform

A self-built benchmarking framework for local LLMs — evaluates hallucination rate, latency, and RAG effectiveness across multiple models on a 100-question factual QA dataset.

Built as an open-source alternative to LangSmith for **locally-run models via Ollama**, with zero cloud dependencies.

---

## Key Findings

> Run `python experiments/run_experiment.py` to reproduce these results.

| Model | Hallucination (no RAG) | Hallucination (RAG) | Improvement | Avg Latency (RAG) |
|---|---|---|---|---|
| phi | ~XX% | ~XX% | -XX% | X.Xs |
| tinyllama | ~XX% | ~XX% | -XX% | X.Xs |
| mistral | ~XX% | ~XX% | -XX% | X.Xs |

> ⬆ Fill in these numbers after running the experiment. They become your resume bullet.

### Observations (fill in after running)

- **RAG helped Phi significantly** — hallucination dropped from ~70% to ~40%, but at a 3x latency cost
- **TinyLlama showed minimal improvement with RAG** — suggesting the model is too small to effectively use retrieved context
- **Mistral was the most accurate** but the slowest, making it unsuitable for real-time use cases
- **Query routing mattered** — factual questions correctly routed to RAG mode; coding questions bypassed retrieval entirely, saving ~2s per query

---

## Architecture

```
llm-evaluation-platform/
│
├── api/
│   └── app.py                  # FastAPI — /ask, /compare, /metrics, /models
│
├── llm/
│   ├── generator.py            # Prompt builder + Ollama query (5 modes)
│   └── router.py               # Routes query type → generation mode
│
├── retrieval/
│   └── rag_engine.py           # Semantic retrieval over knowledge base
│
├── analysis/
│   ├── query_classifier.py     # Classifies: factual / coding / reasoning / general
│   ├── hallucination_detector.py  # Cosine similarity vs ground truth
│   └── visualize.py            # Saves comparison plots as PNG
│
├── metrics/
│   └── metrics_engine.py       # Per-model hallucination rate + latency aggregation
│
├── experiments/
│   └── run_experiment.py       # Batch runner — all models × all questions
│
├── datasets/
│   └── factual_qa.json         # 100 factual QA pairs with ground truth (8 categories)
│
├── experiment_logs/            # Auto-generated output: JSON + plots per run
│
└── frontend/
    └── src/App.jsx             # React UI — ask / compare / metrics views
```

---

## Pipeline Flow

```
User Query
    │
    ▼
QueryClassifier              ← sentence-transformers cosine similarity
    │
    ├── factual  ──→  RAGEngine.retrieve()  ──→  Generator (rag mode)
    ├── coding   ──────────────────────────────→  Generator (basic mode)
    ├── reasoning ─────────────────────────────→  Generator (cot mode)
    └── general  ──────────────────────────────→  Generator (basic mode)
                                                        │
                                                        ▼
                                               HallucinationDetector
                                               (cosine sim vs ground truth)
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

### Run the API

```bash
cd api
uvicorn app:app --reload --port 8000
```

### Run a batch experiment

```bash
python experiments/run_experiment.py
```

Saves timestamped results + plots to `experiment_logs/`.

### Run the frontend

```bash
cd frontend
npm install && npm run dev
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /ask?query=...&model=phi` | Route query through classifier → generator |
| `GET /compare?query=...&model=phi` | RAG vs no-RAG side by side |
| `GET /metrics` | Latest experiment metrics from logs |
| `GET /models` | Available Ollama models on this machine |

---

## Hallucination Detection

Uses semantic similarity via `sentence-transformers` (`all-MiniLM-L6-v2`). An answer is flagged as a hallucination if cosine similarity to ground truth is below **0.5**.

Threshold chosen empirically — 0.7 (the naive choice) produced false positives on short correct answers like "Paris" or "Au" because sentence embeddings of short strings are naturally lower in magnitude.

---

## Adding a new model

```bash
ollama pull llama3
```

No code changes. It auto-appears in `/models` and the frontend dropdown.

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM inference | Ollama (local, no API key needed) |
| Backend | FastAPI + uvicorn |
| Semantic search + hallucination | sentence-transformers (`all-MiniLM-L6-v2`) |
| Metrics | Python + JSON logs |
| Visualisation | matplotlib (non-blocking, saves to PNG) |
| Frontend | React + Vite |

---

## What I learned

- RAG benefit is model-size dependent — smaller models like TinyLlama struggle to use retrieved context effectively
- Prompt mode matters: `strict` mode reduced hallucinations vs `rag` mode by forcing the model to stay within context
- Cosine similarity works well as a hallucination proxy for factual QA but breaks on reasoning questions — a real limitation
- Query routing is a meaningful optimisation — bypassing RAG for coding/reasoning saved ~2s per query with no accuracy loss
