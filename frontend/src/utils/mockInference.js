const classes = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"];
const severities = ["mild", "moderate", "severe"];

export function runFallbackInference(file) {
  if (!file) return null;

  const base = classes.map((label) => ({ label, value: Math.random() }));
  const sum = base.reduce((acc, item) => acc + item.value, 0);
  const probabilities = base.map((item) => ({ ...item, value: item.value / sum }));
  probabilities.sort((a, b) => b.value - a.value);

  return {
    probabilities,
    topClass: probabilities[0].label,
    confidence: probabilities[0].value,
    severity: severities[Math.floor(Math.random() * severities.length)],
    heatmapType: "Grad-CAM",
    mode: "mock",
  };
}
