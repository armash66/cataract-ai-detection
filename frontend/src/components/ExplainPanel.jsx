export default function ExplainPanel({ result }) {
  return (
    <div className="panel explain-panel">
      <div className="panel-header">
        <h2>Explainability</h2>
        <span className="panel-sub">Grad-CAM</span>
      </div>

      {result ? (
        <>
          <p className="explain-text">
            Highlighted regions indicate the retinal areas most responsible for the predicted class.
          </p>
          <ul className="explain-list">
            <li>Primary class: {result.topClass}</li>
            <li>Confidence: {Math.round(result.confidence * 100)}%</li>
            <li>Severity estimate: {result.severity}</li>
            <li>Adjust heatmap opacity in Imaging Analysis for visual correlation.</li>
          </ul>
        </>
      ) : (
        <div className="explain-empty">Upload an image to generate explainability insights.</div>
      )}
    </div>
  );
}
