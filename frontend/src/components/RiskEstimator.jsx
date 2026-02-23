export default function RiskEstimator({ patient, onChange, risk }) {
  const update = (key, value) => onChange({ ...patient, [key]: value });

  return (
    <div className="panel">
      <div className="panel-header"><h2>Risk Estimation</h2><span className="panel-sub">Rule Engine</span></div>
      <div className="grid2">
        <label>Age<input type="number" value={patient.age} onChange={(e) => update("age", Number(e.target.value))} /></label>
        <label>Symptoms
          <select value={patient.symptoms} onChange={(e) => update("symptoms", e.target.value)}>
            <option value="none">None</option>
            <option value="mild_blur">Mild Blur</option>
            <option value="severe_blur">Severe Blur</option>
            <option value="pain">Pain</option>
          </select>
        </label>
        <label><input type="checkbox" checked={patient.diabetes} onChange={(e) => update("diabetes", e.target.checked)} /> Diabetes</label>
        <label><input type="checkbox" checked={patient.familyHistory} onChange={(e) => update("familyHistory", e.target.checked)} /> Family History</label>
      </div>
      {risk && (
        <div className="risk-box">
          <p><strong>Risk Score:</strong> {risk.score}/100</p>
          <p><strong>Recommendation:</strong> {risk.recommendation}</p>
        </div>
      )}
    </div>
  );
}
