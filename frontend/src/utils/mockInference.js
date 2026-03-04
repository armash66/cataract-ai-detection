const classes = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"];
const severities = ["mild", "moderate", "severe"];

/**
 * Generates a stable pseudo-random number based on a string seed.
 * This ensures the same file always produces the same result.
 */
function seededRandom(seed) {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (hash << 5) - hash + seed.charCodeAt(i);
    hash |= 0;
  }
  const x = Math.sin(hash++) * 10000;
  return x - Math.floor(x);
}

export function runFallbackInference(file) {
  if (!file) return null;

  // Use file name and size as a seed for determinism across different environments
  const seed = `${file.name}-${file.size}`;
  
  const base = classes.map((label, idx) => {
    // Generate a unique but stable value for each class
    const val = seededRandom(seed + label + idx);
    return { label, value: val };
  });

  const sum = base.reduce((acc, item) => acc + item.value, 0);
  const probabilities = base.map((item) => ({ ...item, value: item.value / sum }));
  
  // Sort by probability
  probabilities.sort((a, b) => b.value - a.value);

  // Pick a stable severity
  const sevIdx = Math.floor(seededRandom(seed + "sev") * severities.length);

  return {
    probabilities,
    topClass: probabilities[0].label,
    confidence: probabilities[0].value,
    severity: severities[sevIdx],
    heatmapType: "Grad-CAM",
    mode: "mock",
  };
}
