import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, MoveDiagonal2, ZoomIn, ZoomOut } from "lucide-react";

export default function ImagingPanel({ image, result }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(60);
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [compareImage, setCompareImage] = useState(null);
  const compareRef = useRef(null);

  const imageUrl = useMemo(() => (image ? URL.createObjectURL(image) : ""), [image]);
  const compareUrl = useMemo(() => (compareImage ? URL.createObjectURL(compareImage) : ""), [compareImage]);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      if (compareUrl) URL.revokeObjectURL(compareUrl);
    };
  }, [imageUrl, compareUrl]);

  const transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;

  if (!image) {
    return (
      <div className="panel imaging-panel">
        <div className="panel-header">
          <h2>Imaging Analysis</h2>
          <span className="panel-sub">Awaiting Input</span>
        </div>
        <div className="imaging-placeholder">Upload an eye image to generate source and attention views.</div>
      </div>
    );
  }

  return (
    <div className="panel imaging-panel">
      <div className="imaging-head">
        <div className="imaging-title">Imaging Analysis</div>
        <div className="imaging-controls">
          <button className="btn btn-secondary" type="button" onClick={() => setShowOverlay((v) => !v)}>
            {showOverlay ? "Hide" : "Show"} Heatmap
          </button>
          <input type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          <button className="btn btn-secondary" type="button" onClick={() => setZoom((z) => Math.max(1, z - 0.1))}><ZoomOut size={14} /></button>
          <button className="btn btn-secondary" type="button" onClick={() => setZoom((z) => Math.min(2, z + 0.1))}><ZoomIn size={14} /></button>
          <button className="btn btn-secondary" type="button" onClick={() => setPanX((v) => v - 8)}><ChevronLeft size={14} /></button>
          <button className="btn btn-secondary" type="button" onClick={() => setPanX((v) => v + 8)}><ChevronRight size={14} /></button>
          <button className="btn btn-secondary" type="button" onClick={() => setPanY((v) => v - 8)}><MoveDiagonal2 size={14} /></button>
          <button className="btn btn-secondary" type="button" onClick={() => setPanY((v) => v + 8)}><MoveDiagonal2 size={14} /></button>
          <input ref={compareRef} type="file" className="hidden" accept="image/*" onChange={(e) => setCompareImage(e.target.files?.[0] || null)} />
          <button className="btn btn-secondary" type="button" onClick={() => compareRef.current?.click()}>Compare</button>
        </div>
      </div>

      <div className={`imaging-grid ${compareImage ? "compare-on" : ""}`}>
        <div className="imaging-cell">
          <img src={imageUrl} alt="Source eye" style={{ transform }} />
          <span className="imaging-tag">SOURCE</span>
        </div>

        <div className="imaging-cell">
          <img src={imageUrl} alt="Attention base" style={{ transform }} />
          {showOverlay && result?.heatmapUrl ? (
            <img className="heatmap-overlay" src={result.heatmapUrl} alt="Grad-CAM overlay" style={{ opacity: opacity / 100 }} />
          ) : showOverlay ? (
            <div className="mock-overlay" style={{ opacity: opacity / 100 }} title={result?.heatmapType || "Grad-CAM"} />
          ) : null}
          <span className="imaging-tag">ATTENTION</span>
        </div>

        {compareImage && (
          <div className="imaging-cell">
            <img src={compareUrl} alt="Comparison" style={{ transform }} />
            <span className="imaging-tag">COMPARE</span>
          </div>
        )}
      </div>
    </div>
  );
}
