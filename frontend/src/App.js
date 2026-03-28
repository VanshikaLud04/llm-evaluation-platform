import { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!query) return;

    setLoading(true);
    setResponse(null);
    setCompareData(null);

    const res = await fetch(
      `http://127.0.0.1:8000/ask?query=${encodeURIComponent(query)}`
    );
    const data = await res.json();

    setResponse(data);
    setLoading(false);
  };

  const handleCompare = async () => {
    if (!query) return;

    setLoading(true);
    setResponse(null);

    const res = await fetch(
      `http://127.0.0.1:8000/compare?query=${encodeURIComponent(query)}`
    );
    const data = await res.json();

    setCompareData(data);
    setLoading(false);
  };

  const handleMetrics = async () => {
    const res = await fetch("http://127.0.0.1:8000/metrics");
    const data = await res.json();
    setMetrics(data);
  };

  return (
    <div className="container">
      <h1 className="title">LLM Evaluation System</h1>

      <div className="input-box">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="ask something..."
        />
        <button onClick={handleAsk}>ask</button>
        <button onClick={handleCompare}>compare</button>
        <button onClick={handleMetrics}>metrics</button>
      </div>

      {loading && <p className="typewriter">thinking...</p>}

      {/* -------- NORMAL RESPONSE -------- */}
      {response && (
        <div className="response">
          <p className="label">answer</p>
          <p className="typewriter">{response.answer}</p>

          <p className="meta">latency: {response.latency?.toFixed(2)}s</p>
          <p className="meta">mode: {response.mode}</p>
          <p className="meta">
            hallucination:{" "}
            {response.hallucination ? "yes ⚠️" : "no ✅"}
          </p>
        </div>
      )}

      {/* -------- COMPARE -------- */}
      {compareData && (
        <div className="response">
          <p className="label">no rag</p>
          <p>{compareData.no_rag.answer}</p>

          <p className="label">rag</p>
          <p>{compareData.rag.answer}</p>
        </div>
      )}

      {/* -------- METRICS -------- */}
      {metrics && (
        <div className="response">
          <p className="label">system metrics</p>
          <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;