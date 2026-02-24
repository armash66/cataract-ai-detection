import { runFallbackInference } from "./mockInference";
import { getModelUrlForModality } from "../config/modality";

const CLASS_NAMES = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"];
const NORMAL_LABEL = "Normal";

/** Below this confidence we abstain (no definitive disease output). */
const ABSTAIN_CONFIDENCE_THRESHOLD = 0.5;

/** If quality score is below this, we may abstain with reason low_quality. */
const ABSTAIN_QUALITY_THRESHOLD = 60;

async function preprocessImage(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, 224, 224);

  const { data } = ctx.getImageData(0, 0, 224, 224);
  const input = new Float32Array(1 * 3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    input[i] = r;
    input[224 * 224 + i] = g;
    input[2 * 224 * 224 + i] = b;
  }
  return input;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function applyNormalOverride(probabilities) {
  const sorted = [...probabilities].sort((a, b) => b.value - a.value);
  const top = sorted[0];
  const second = sorted[1] || { value: 0 };

  const normal = probabilities.find((p) => p.label === NORMAL_LABEL);
  if (!normal) return { probabilities, normalOverride: false };

  const normalConfident = top.label === NORMAL_LABEL && normal.value >= 0.58 && (normal.value - second.value) >= 0.18;
  if (!normalConfident) return { probabilities, normalOverride: false };

  // Keep some uncertainty, but prevent visually misleading high disease percentages.
  const nonNormalTotal = 0.15;
  const normalTotal = 0.85;

  const nonNormal = probabilities.filter((p) => p.label !== NORMAL_LABEL);
  const nnSum = nonNormal.reduce((acc, p) => acc + p.value, 0) || 1;

  const adjusted = probabilities.map((p) => {
    if (p.label === NORMAL_LABEL) return { ...p, value: normalTotal };
    return { ...p, value: (p.value / nnSum) * nonNormalTotal };
  });

  return { probabilities: adjusted, normalOverride: true };
}

function buildResultFromProbs(probs, mode, heatmapUrl, options = {}) {
  const raw = CLASS_NAMES.map((label, i) => ({ label, value: probs[i] ?? 0 }));
  const { probabilities: adjusted, normalOverride } = applyNormalOverride(raw);
  const sorted = [...adjusted].sort((a, b) => b.value - a.value);
  const confidence = sorted[0]?.value ?? 0;
  const qualityScore = options.qualityScore ?? null;

  // Abstention: never force definitive disease output on bad inputs
  let abstain = false;
  const abstainReasons = [];
  if (confidence < ABSTAIN_CONFIDENCE_THRESHOLD) {
    abstain = true;
    abstainReasons.push("low_confidence");
  }
  if (qualityScore != null && qualityScore < ABSTAIN_QUALITY_THRESHOLD) {
    abstain = true;
    abstainReasons.push("low_quality");
  }

  return {
    probabilities: sorted,
    topClass: sorted[0]?.label ?? null,
    confidence,
    severity: confidence > 0.8 ? "severe" : confidence > 0.6 ? "moderate" : "mild",
    heatmapType: "Grad-CAM",
    heatmapUrl,
    mode,
    normalOverride,
    abstain,
    abstainReasons: abstainReasons.length ? abstainReasons : null,
  };
}

export async function runInference(file, options = {}) {
  if (!file) return null;

  const modality = options.modality;
  const modelUrl = options.modelUrl || (modality ? getModelUrlForModality(modality) : "/models/best_accuracy.onnx");
  const heatmapUrl = options.heatmapUrl || "/heatmaps/latest_gradcam.png";
  const qualityScore = options.qualityScore ?? null;

  try {
    const ort = await import("onnxruntime-web");
    const inputData = await preprocessImage(file);
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const tensor = new ort.Tensor("float32", inputData, [1, 3, 224, 224]);
    const outputs = await session.run({ [inputName]: tensor });
    const logits = Array.from(outputs[outputName].data);
    const probs = softmax(logits);

    return buildResultFromProbs(probs, "onnxruntime-web", heatmapUrl, { qualityScore });
  } catch (_) {
    const fallback = runFallbackInference(file);
    const probs = CLASS_NAMES.map((c) => fallback.probabilities.find((p) => p.label === c)?.value ?? 0);
    return buildResultFromProbs(probs, fallback.mode || "mock", heatmapUrl, { qualityScore });
  }
}
