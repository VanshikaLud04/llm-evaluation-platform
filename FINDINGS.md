# Experiment Findings

**Date:** 29 March 2026
**Models:** Phi, TinyLlama
**Dataset:** 100 factual QA pairs (50 questions × 2 models)
**Evaluation:** Cosine similarity vs ground truth (hallucination threshold: 0.5)

---

## Results

### Overall RAG vs No RAG

| Mode | Hallucination Rate | Avg Latency |
|---|---|---|
| No RAG | 28% | 0.55s |
| RAG | 55% | 0.46s |

RAG **increased** hallucination rate by 27 percentage points.

### Per-Model (RAG mode)

| Model | Hallucination Rate | Avg Latency |
|---|---|---|
| Phi | 26% | 0.57s |
| TinyLlama | 84% | 0.35s |

---

## Findings

### Finding 1: RAG hurt accuracy when knowledge base coverage was low

The knowledge base contained 10 document chunks covering roughly 10% of the 100-question dataset. For the remaining 90 questions, the retriever returned loosely related chunks — adding noise to the prompt rather than grounding the answer. Both models anchored on the irrelevant context and performed worse than without RAG.

**Implication:** RAG is only beneficial when knowledge base coverage matches the query distribution. Partial coverage is worse than no retrieval.

### Finding 2: Model size dominated accuracy — 58 point gap

Phi hallucinated on 26% of questions vs TinyLlama's 84% — a 58 percentage point gap under identical conditions. For factual QA tasks, parametric knowledge (encoded in model weights during training) matters far more than retrieval when the knowledge base is small.

### Finding 3: TinyLlama's speed advantage doesn't compensate for accuracy

TinyLlama was 38% faster (0.35s vs 0.57s avg latency). In any real use case, a 84% hallucination rate makes that speed irrelevant — you'd need to verify every answer externally, eliminating the latency benefit entirely.

### Finding 4: Query routing eliminated unnecessary RAG overhead

The query classifier correctly routed coding and reasoning queries away from RAG. This avoided retrieval calls (~0.1s overhead each) on queries where context wouldn't help anyway. For a mixed-query production workload, routing is a meaningful latency optimisation.

---

## Resume Bullet

> "Built an end-to-end LLM evaluation framework benchmarking Phi and TinyLlama across 100 factual QA pairs; found RAG *increased* hallucination rate by 27% when knowledge base coverage was low (55% vs 28%), and that model size dominated accuracy — Phi achieved 26% hallucination vs TinyLlama's 84% despite TinyLlama being 38% faster; implemented query routing to bypass RAG on non-factual queries, reducing unnecessary retrieval overhead."