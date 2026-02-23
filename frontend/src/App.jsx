import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Eye, FlaskConical } from "lucide-react";

import UploadPanel from "./components/UploadPanel";
import PredictionDashboard from "./components/PredictionDashboard";
import RiskEstimator from "./components/RiskEstimator";
import ThemeToggle from "./components/ThemeToggle";
import ImagingPanel from "./components/ImagingPanel";
import ExplainPanel from "./components/ExplainPanel";

import { runInference } from "./utils/inference";
import { computeRisk } from "./utils/riskEngine";

const initialPatient = {
  age: 45,
  diabetes: false,
  familyHistory: false,
  symptoms: "mild_blur"
};

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [theme, setTheme] = useState("dark");
  const [patient, setPatient] = useState(initialPatient);

  const risk = useMemo(() => {
    if (!result) return null;
    return computeRisk(result, patient);
  }, [result, patient]);

  const handleAnalyze = async (file) => {
    setImage(file || null);
    if (!file) {
      setResult(null);
      return;
    }
    const output = await runInference(file, { modelUrl: "/models/best_accuracy.onnx" });
    setResult(output);
  };

  return (
    <div className={`app-shell ${theme === "light" ? "theme-light" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-head">
          <div className="brand">EyeGPT</div>
          <div className="brand-sub">Retinal AI Console</div>
        </div>

        <nav className="side-nav" aria-label="Primary Navigation">
          <button className="nav-item active" type="button">New Analysis</button>
          <button className="nav-item" type="button">Model Metrics</button>
          <button className="nav-item" type="button">Explainability</button>
          <button className="nav-item is-disabled" type="button">Clinical Notes</button>
        </nav>

        <div className="sidebar-foot">
          <div className="status-dot" />
          <div>
            <div className="status-title">System: Online</div>
            <div className="status-sub">Mode: Browser Inference</div>
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div>
            <div className="topbar-title">EyeGPT Lab</div>
            <div className="topbar-sub">Multi-disease screening and risk analysis</div>
          </div>

          <div className="topbar-right">
            <span className="topbar-pill">
              <FlaskConical size={12} /> Research
            </span>
            <span className="topbar-pill">
              <Activity size={12} /> Explainable
            </span>
            <ThemeToggle
              theme={theme}
              onToggle={() => setTheme(theme === "light" ? "dark" : "light")}
            />
          </div>
        </header>

        <main className="main-content">
          <div className="content-grid">
            <div className="left-column">
              <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <UploadPanel onAnalyze={handleAnalyze} image={image} />
              </motion.section>

              <motion.section
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.08 }}
              >
                <PredictionDashboard result={result} />
              </motion.section>

              <motion.section
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.16 }}
              >
                <RiskEstimator patient={patient} onChange={setPatient} risk={risk} />
              </motion.section>
            </div>

            <div className="right-column">
              <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <ImagingPanel image={image} result={result} />
              </motion.section>

              <motion.section
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <ExplainPanel result={result} />
              </motion.section>

              <div className="disclaimer">
                <Eye size={14} /> Research use only. Not a diagnostic device.
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
