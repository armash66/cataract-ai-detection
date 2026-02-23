import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function PredictionDashboard({ result }) {
  return (
    <div className="panel">
      <div className="panel-header"><h2>Result Summary</h2><span className="panel-sub">Session Output</span></div>
      {!result ? (
        <div className="outcome-empty">Upload an image to see disease probabilities and severity.</div>
      ) : (
        <>
          <div className="badge-row">
            <span className="prediction-badge">{result.topClass}</span>
            <span className="confidence-badge">{Math.round(result.confidence * 100)}% confidence</span>
            <span className="confidence-badge">Severity: {result.severity}</span>
          </div>
          <div style={{ width: "100%", height: 220 }}>
            <ResponsiveContainer>
              <BarChart data={result.probabilities}>
                <XAxis dataKey="label" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="value" fill="#38bdf8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
