import { useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const MODELS = ["phi", "tinyllama", "mistral"];

function App() {
  const [query, setQuery] = useState("");
  const [model, setModel] = useState("phi");
  const [response, setResponse] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const clearAll = () => {
    setResponse(null);
    setCompareData(null);
    setMetrics(null);
    setError(null);
  };

  const handleAsk = async () => {
    if (!query.trim()) return;
    setLoading(true);
    clearAll();
    try {
      const res = await fetch(
        `${API}/ask?query=${encodeURIComponent(query)}&model=${model}`
      );
      const data = await res.json();
      setResponse(data);
    } catch {
      setError("Could not reach the API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!query.trim()) return;
    setLoading(true);
    clearAll();
    try {
      const res = await fetch(
        `${API}/compare?query=${encodeURIComponent(query)}&model=${model}`
      );
      const data = await res.json();
      setCompareData(data);
    } catch {
      setError("Could not reach the API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleMetrics = async () => {
    clearAll();
    try {
      const res = await fetch(`${API}/metrics`);
      const data = await res.json();
      setMetrics(data);
    } catch {
      setError("Could not load metrics.");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleAsk();
  };

  return (
    <div className="container">
      <h1 className="title">LLM Evaluation System</h1>

      {/* ── Input row ── */}
      <div className="input-box">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="ask something..."
        />
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="model-select"
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button onClick={handleAsk}>ask</button>
        <button onClick={handleCompare}>compare</button>
        <button onClick={handleMetrics}>metrics</button>
      </div>

      {loading && <p className="typewriter">thinking...</p>}
      {error && <p className="error">{error}</p>}

      {/* ── Normal response ── */}
      {response && (
        <div className="response">
          <p className="label">answer</p>
          <p className="typewriter">{response.answer}</p>
          <p className="meta">latency: {response.latency?.toFixed(2)}s</p>
          <p className="meta">mode: {response.mode}</p>
          <p className="meta">query type: {response.query_type}</p>
          <p className="meta">
            hallucination:{" "}
            <span className={response.hallucination ? "bad" : "good"}>
              {response.hallucination === null
                ? "n/a"
                : response.hallucination
                ? "yes ⚠️"
                : "no ✅"}
            </span>
          </p>
        </div>
      )}

      {/* ── Compare view ── */}
      {compareData && (
        <div className="response">
          <p className="label">model: {compareData.model}</p>

          <p className="label" style={{ marginTop: "1rem" }}>without RAG</p>
          <p>{compareData.no_rag.answer}</p>
          <p className="meta">latency: {compareData.no_rag.latency?.toFixed(2)}s</p>

          <p className="label" style={{ marginTop: "1rem" }}>with RAG</p>
          <p>{compareData.rag.answer}</p>
          <p className="meta">latency: {compareData.rag.latency?.toFixed(2)}s</p>

          <p className="label" style={{ marginTop: "1rem" }}>context used</p>
          <p className="context-box">{compareData.rag.context_used}</p>
        </div>
      )}

      {/* ── Metrics view ── */}
      {metrics && (
        <div className="response">
          <p className="label">experiment metrics</p>
          {metrics.error ? (
            <p className="error">{metrics.error}</p>
          ) : (
            <>
              <MetricRow label="RAG hallucination rate" value={metrics.rag_metrics?.hallucination_rate} />
              <MetricRow label="No-RAG hallucination rate" value={metrics.no_rag_metrics?.hallucination_rate} />
              <MetricRow label="RAG avg latency" value={metrics.rag_metrics?.avg_latency} unit="s" />
              <p className="label" style={{ marginTop: "1rem" }}>per model (RAG)</p>
              {metrics.model_metrics &&
                Object.entries(metrics.model_metrics).map(([m, d]) => (
                  <p key={m} className="meta">
                    {m}: {(d.hallucination_rate * 100).toFixed(0)}% hallucination, {d.avg_latency.toFixed(2)}s avg
                  </p>
                ))}
              <p className="meta" style={{ marginTop: "0.5rem", opacity: 0.4 }}>
                source: {metrics._source_file}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value, unit = "" }) {
  if (value === undefined || value === null) return null;
  const display = typeof value === "number"
    ? unit === "s" ? `${value.toFixed(2)}s` : `${(value * 100).toFixed(1)}%`
    : value;
  return (
    <p className="meta">
      {label}: <strong>{display}</strong>
    </p>
  );
}

export default App;