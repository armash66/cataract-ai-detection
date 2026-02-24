import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { jsPDF } from "jspdf";

const CLASS_ORDER = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"];

function canonicalProbabilities(result) {
  const map = new Map((result?.probabilities || []).map((p) => [p.label, Number(p.value) || 0]));
  const raw = CLASS_ORDER.map((label) => ({ label, value: Math.max(0, map.get(label) || 0) }));
  const sum = raw.reduce((acc, item) => acc + item.value, 0);

  if (sum <= 0) {
    const uniform = 1 / CLASS_ORDER.length;
    return CLASS_ORDER.map((label) => ({ label, value: uniform }));
  }

  return raw.map((item) => ({ ...item, value: item.value / sum }));
}

function downloadPdfReport(result) {
  if (!result) return;
  const probs = canonicalProbabilities(result);

  const doc = new jsPDF();
  doc.setFontSize(16);
  doc.text("EyeGPT-AI Screening Report", 14, 18);
  doc.setFontSize(11);
  doc.text(`Primary Class: ${result.topClass}`, 14, 32);
  doc.text(`Confidence: ${Math.round(result.confidence * 100)}%`, 14, 40);
  doc.text(`Severity: ${result.severity}`, 14, 48);
  doc.text("Class Probabilities:", 14, 60);

  let y = 68;
  probs.forEach((p) => {
    doc.text(`- ${p.label}: ${(p.value * 100).toFixed(2)}%`, 16, y);
    y += 7;
  });

  doc.text("Disclaimer: Research use only, not a clinical diagnosis.", 14, y + 8);
  doc.save("eyegpt-report.pdf");
}

export default function PredictionDashboard({ result }) {
  const chartData = result ? canonicalProbabilities(result) : [];

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
            <button className="btn btn-secondary badge-action" type="button" onClick={() => downloadPdfReport(result)}>
              Download PDF
            </button>
          </div>

          {result.normalOverride && (
            <p className="muted" style={{ marginTop: 0, marginBottom: 8 }}>
              Normal-eye stabilization applied to reduce misleading disease percentages.
            </p>
          )}

          <div className="chart-wrap">
            <ResponsiveContainer>
              <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} interval={0} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(2)}%`} />
                <Bar dataKey="value" fill="#38bdf8" radius={[6, 6, 0, 0]} minPointSize={2} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="prob-grid">
            {chartData.map((p) => (
              <div className="prob-item" key={p.label}>
                <span>{p.label}</span>
                <strong>{(p.value * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
