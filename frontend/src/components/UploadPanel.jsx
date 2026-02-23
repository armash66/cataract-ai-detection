import { useRef, useState } from "react";
import { ImagePlus, Trash2, Upload } from "lucide-react";

import { runQualityCheck } from "../utils/qualityCheck";

export default function UploadPanel({ onAnalyze, image }) {
  const inputRef = useRef(null);
  const [quality, setQuality] = useState(null);

  const onFile = (file) => {
    if (!file) return;
    const qualityResult = runQualityCheck(file);
    setQuality(qualityResult);
    onAnalyze(file);
  };

  return (
    <div className="panel upload-panel">
      <div className="panel-header">
        <h2>New Analysis</h2>
        <span className="panel-sub">Upload Input</span>
      </div>

      <p className="muted">Use fundus or anterior image. Local inference is currently mocked.</p>

      <div className="form-actions">
        <button className="btn btn-primary" type="button" onClick={() => inputRef.current?.click()}>
          <ImagePlus size={16} /> Choose
        </button>

        <button className="btn btn-secondary" type="button" onClick={() => image && onAnalyze(image)}>
          <Upload size={16} /> Re-analyze
        </button>

        <button className="btn btn-secondary" type="button" onClick={() => onAnalyze(null)}>
          <Trash2 size={16} /> Remove
        </button>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0])}
      />

      <div className="file-meta">
        <span>{image ? image.name : "No file selected"}</span>
        <span>{quality ? `${quality.score}%` : "--"}</span>
      </div>

      {quality && <div className="quality-pill">Quality: {quality.label}</div>}
    </div>
  );
}
