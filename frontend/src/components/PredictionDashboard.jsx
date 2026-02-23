import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function PredictionDashboard({ result }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Result Summary</h2>
        <span className="panel-sub">Session Output</span>
      </div>

      {!result ? (
        <div className="outcome-empty">Upload an image to see disease probabilities and severity.</div>
      ) : (
        <>
          <div className="badge-row">
            <span className="prediction-badge">{result.topClass}</span>
            <span className="confidence-badge">{Math.round(result.confidence * 100)}% confidence</span>
            <span className="confidence-badge">Severity: {result.severity}</span>
          </div>

          <div className="chart-wrap">
            <ResponsiveContainer>
              <BarChart data={result.probabilities}>
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#38bdf8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
