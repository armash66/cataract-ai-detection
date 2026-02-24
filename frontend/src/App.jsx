import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Camera, Eye, Sparkles } from "lucide-react";

import UploadPanel from "./components/UploadPanel";
import PredictionDashboard from "./components/PredictionDashboard";
import RiskEstimator from "./components/RiskEstimator";
import ThemeToggle from "./components/ThemeToggle";
import ImagingPanel from "./components/ImagingPanel";
import ExplainPanel from "./components/ExplainPanel";
import ModalitySelector from "./components/ModalitySelector";
import MetricsDashboard from "./components/MetricsDashboard";

import { runInference } from "./utils/inference";
import { computeRisk } from "./utils/riskEngine";

const initialPatient = {
  age: 45,
  diabetes: false,
  familyHistory: false,
  symptoms: "mild_blur"
};

function Landing({ onStart, theme, onToggleTheme }) {
  return (
    <div className={`landing ${theme === "light" ? "theme-light" : ""}`}>
      <header className="landing-topbar">
        <div className="landing-brand"><Eye size={20} /> EyeGPT-AI</div>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </header>

      <main className="landing-main">
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="hero">
          <p className="hero-kicker">Research-grade Ophthalmology AI</p>
          <h1>Multi-disease retinal screening, explainability, and browser inference.</h1>
          <p className="hero-sub">Classify Cataract, Glaucoma, Diabetic Retinopathy, and Normal with model insights and risk estimation.</p>
          <div className="hero-actions">
            <button className="btn btn-primary" type="button" onClick={onStart}><Sparkles size={16} /> Start Analysis</button>
            <button className="btn btn-secondary" type="button" onClick={onStart}><Camera size={16} /> Try Camera Mode</button>
          </div>
        </motion.section>
      </main>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("landing");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [theme, setTheme] = useState("dark");
  const [patient, setPatient] = useState(initialPatient);
  const [modality, setModality] = useState("anterior");

  const risk = useMemo(() => {
    if (!result) return null;
    return computeRisk(result, patient);
  }, [result, patient]);

  const handleAnalyze = async (file, opts = {}) => {
    setImage(file || null);
    if (!file) {
      setResult(null);
      return;
    }
    const output = await runInference(file, {
      modality,
      qualityScore: opts.qualityScore ?? null,
    });
    setResult(output);
  };

  if (page === "landing") {
    return <Landing onStart={() => setPage("analysis")} theme={theme} onToggleTheme={() => setTheme(theme === "light" ? "dark" : "light")} />;
  }

  return (
    <div className={`analysis ${theme === "light" ? "theme-light" : ""}`}>
      <header className="analysis-topbar">
        <div className="analysis-brand"><Eye size={18} /> EyeGPT Lab</div>
        <div className="analysis-actions">
          <button className="btn btn-secondary" type="button" onClick={() => setPage("landing")}>Home</button>
          <ThemeToggle theme={theme} onToggle={() => setTheme(theme === "light" ? "dark" : "light")} />
        </div>
      </header>

      <main className="analysis-main">
        <div className="analysis-grid">
          <div className="left-column">
            <ModalitySelector value={modality} onChange={setModality} />
            <UploadPanel onAnalyze={handleAnalyze} image={image} />
            <PredictionDashboard result={result} />
            <RiskEstimator patient={patient} onChange={setPatient} risk={risk} />
          </div>
          <div className="right-column">
            <ImagingPanel image={image} result={result} />
            <ExplainPanel result={result} />
            <MetricsDashboard />
          </div>
        </div>
      </main>
    </div>
  );
}
