import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { jsPDF } from "jspdf";

function downloadPdfReport(result) {
  if (!result) return;
  const doc = new jsPDF();
  doc.setFontSize(16);
  doc.text("EyeGPT-AI Screening Report", 14, 18);
  doc.setFontSize(11);
  doc.text(`Primary Class: ${result.topClass}`, 14, 32);
  doc.text(`Confidence: ${Math.round(result.confidence * 100)}%`, 14, 40);
  doc.text(`Severity: ${result.severity}`, 14, 48);
  doc.text("Class Probabilities:", 14, 60);

  let y = 68;
  result.probabilities.forEach((p) => {
    doc.text(`- ${p.label}: ${(p.value * 100).toFixed(2)}%`, 16, y);
    y += 7;
  });

  doc.text("Disclaimer: Research use only, not a clinical diagnosis.", 14, y + 8);
  doc.save("eyegpt-report.pdf");
}

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
            <button className="btn btn-secondary" type="button" onClick={() => downloadPdfReport(result)}>Download PDF</button>
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
