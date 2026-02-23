import { useEffect, useMemo, useState } from "react";
import { ZoomIn, ZoomOut } from "lucide-react";

export default function ImagingPanel({ image, result }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(60);
  const [zoom, setZoom] = useState(1);

  const imageUrl = useMemo(() => (image ? URL.createObjectURL(image) : ""), [image]);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  if (!image) {
    return (
      <div className="panel imaging-panel">
        <div className="panel-header"><h2>Imaging Analysis</h2><span className="panel-sub">Awaiting Input</span></div>
        <div className="imaging-placeholder">Upload an eye image to generate source and attention views.</div>
      </div>
    );
  }

  return (
    <div className="panel imaging-panel">
      <div className="imaging-head">
        <div className="imaging-title">Imaging Analysis</div>
        <div className="imaging-controls">
          <button className="btn btn-secondary" onClick={() => setShowOverlay((v) => !v)}>
            {showOverlay ? "Hide" : "Show"} Heatmap
          </button>
          <input type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          <button className="btn btn-secondary" onClick={() => setZoom((z) => Math.max(1, z - 0.1))}><ZoomOut size={14} /></button>
          <button className="btn btn-secondary" onClick={() => setZoom((z) => Math.min(2, z + 0.1))}><ZoomIn size={14} /></button>
        </div>
      </div>

      <div className="imaging-grid">
        <div className="imaging-cell">
          <img src={imageUrl} alt="source" style={{ transform: `scale(${zoom})` }} />
          <span className="imaging-tag">SOURCE</span>
        </div>
        <div className="imaging-cell">
          <img src={imageUrl} alt="base" className="imaging-base" style={{ transform: `scale(${zoom})` }} />
          {showOverlay && (
            <div
              className="mock-overlay"
              style={{ opacity: opacity / 100 }}
              title={result?.heatmapType || "Grad-CAM"}
            />
          )}
          <span className="imaging-tag">ATTENTION</span>
        </div>
      </div>
    </div>
  );
}
