export default function ExplainPanel({ result }) {
  return (
    <div className="panel explain-panel">
      <div className="panel-header"><h2>Explainability</h2><span className="panel-sub">Grad-CAM</span></div>
      {result ? (
        <>
          <p className="explain-text">The highlighted regions indicate areas most responsible for this prediction.</p>
          <ul className="explain-list">
            <li>Primary class: {result.topClass}</li>
            <li>Confidence: {Math.round(result.confidence * 100)}%</li>
            <li>Severity estimate: {result.severity}</li>
            <li>Use heatmap with opacity slider for visual review.</li>
          </ul>
        </>
      ) : (
        <div className="explain-empty">Upload an image to generate explainability insights.</div>
      )}
    </div>
  );
}
