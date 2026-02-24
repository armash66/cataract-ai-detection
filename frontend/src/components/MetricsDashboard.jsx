import { useEffect, useState } from "react";

export default function MetricsDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/metrics/benchmark.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Not found"))))
      .then((json) => { if (!cancelled) setData(json); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, []);

  if (error) return <div className="panel metrics-panel"><div className="panel-header"><h2>Metrics</h2></div><p className="muted">Metrics not loaded. Run build_metrics_dashboard.py and copy artifacts to frontend/public/metrics.</p></div>;
  if (!data?.models?.length) return <div className="panel metrics-panel"><div className="panel-header"><h2>Metrics</h2></div><p className="muted">No model metrics yet.</p></div>;

  return (
    <div className="panel metrics-panel">
      <div className="panel-header">
        <h2>Metrics</h2>
        <span className="panel-sub">From training artifacts</span>
      </div>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>F1 (weighted)</th>
              <th>ROC-AUC (ovr)</th>
            </tr>
          </thead>
          <tbody>
            {data.models.map((m) => (
              <tr key={m.model}>
                <td>{m.model}</td>
                <td>{(Number(m.accuracy) * 100).toFixed(2)}%</td>
                <td>{(Number(m.f1_weighted) * 100).toFixed(2)}%</td>
                <td>{(Number(m.roc_auc_ovr) * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.models.some((m) => m.confusion_matrix_url) && (
        <div className="metrics-images">
          {data.models.slice(0, 2).map((m) => (
            <div key={m.model} className="metrics-image-cell">
              <div className="metrics-image-label">{m.model} confusion</div>
              <img src={m.confusion_matrix_url} alt={`${m.model} confusion matrix`} className="metrics-img" onError={(e) => { e.target.style.display = "none"; }} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
