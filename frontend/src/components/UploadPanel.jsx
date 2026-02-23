import { useEffect, useRef, useState } from "react";
import { Camera, ImagePlus, Trash2, Upload } from "lucide-react";

import { runQualityCheck } from "../utils/qualityCheck";

export default function UploadPanel({ onAnalyze, image }) {
  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [quality, setQuality] = useState(null);
  const [stream, setStream] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);

  useEffect(() => () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
  }, [stream]);

  const onFile = (file) => {
    if (!file) return;
    const qualityResult = runQualityCheck(file);
    setQuality(qualityResult);
    onAnalyze(file);
  };

  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setStream(s);
      setCameraOn(true);
      if (videoRef.current) videoRef.current.srcObject = s;
    } catch (_) {
      setCameraOn(false);
    }
  };

  const stopCamera = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setStream(null);
    setCameraOn(false);
  };

  const capture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const v = videoRef.current;
    const c = canvasRef.current;
    c.width = v.videoWidth || 640;
    c.height = v.videoHeight || 480;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `camera-${Date.now()}.png`, { type: "image/png" });
      onFile(file);
      stopCamera();
    }, "image/png");
  };

  return (
    <div className="panel upload-panel">
      <div className="panel-header">
        <h2>Input</h2>
        <span className="panel-sub">Upload or Camera</span>
      </div>

      <div className="form-actions">
        <button className="btn btn-primary" type="button" onClick={() => inputRef.current?.click()}>
          <ImagePlus size={16} /> Choose
        </button>
        <button className="btn btn-secondary" type="button" onClick={startCamera}>
          <Camera size={16} /> Camera
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

      {cameraOn && (
        <div className="camera-box">
          <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
          <div className="form-actions">
            <button className="btn btn-primary" type="button" onClick={capture}>Capture</button>
            <button className="btn btn-secondary" type="button" onClick={stopCamera}>Stop</button>
          </div>
          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}

      <div className="file-meta">
        <span>{image ? image.name : "No file selected"}</span>
        <span>{quality ? `${quality.score}%` : "--"}</span>
      </div>

      {quality && <div className="quality-pill">Quality: {quality.label}</div>}
    </div>
  );
}
