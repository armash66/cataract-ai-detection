import { useRef, useState } from "react";
import { Upload, ImagePlus, Trash2 } from "lucide-react";
import { runQualityCheck } from "../utils/qualityCheck";

export default function UploadPanel({ onAnalyze, image }) {
  const inputRef = useRef(null);
  const [quality, setQuality] = useState(null);

  const onFile = (file) => {
    if (!file) return;
    const q = runQualityCheck(file);
    setQuality(q);
    onAnalyze(file);
  };

  return (
    <div className="panel upload-panel">
      <div className="panel-header"><h2>New Analysis</h2><span className="panel-sub">Upload Input</span></div>
      <p className="muted">Use fundus or anterior image. Local inference is currently mocked.</p>
      <div className="form-actions">
        <button className="btn btn-primary" onClick={() => inputRef.current?.click()}><ImagePlus size={16} /> Choose</button>
        <button className="btn btn-secondary" onClick={() => image && onAnalyze(image)}><Upload size={16} /> Re-analyze</button>
        <button className="btn btn-secondary" onClick={() => onAnalyze(null)}><Trash2 size={16} /> Remove</button>
      </div>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
      <div className="file-meta">
        <span>{image ? image.name : "No file selected"}</span>
        <span>{quality ? `${quality.score}%` : "--"}</span>
      </div>
      {quality && <div className="quality-pill is-good">Quality: {quality.label}</div>}
    </div>
  );
}
