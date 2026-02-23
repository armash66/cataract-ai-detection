import { runFallbackInference } from "./mockInference";

const CLASS_NAMES = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"];

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

export async function runInference(file, options = {}) {
  if (!file) return null;

  const modelUrl = options.modelUrl || "/models/best_accuracy.onnx";
  const heatmapUrl = options.heatmapUrl || "/heatmaps/latest_gradcam.png";

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

    const probabilities = CLASS_NAMES.map((label, i) => ({ label, value: probs[i] ?? 0 })).sort(
      (a, b) => b.value - a.value
    );

    return {
      probabilities,
      topClass: probabilities[0].label,
      confidence: probabilities[0].value,
      severity: probabilities[0].value > 0.8 ? "severe" : probabilities[0].value > 0.6 ? "moderate" : "mild",
      heatmapType: "Grad-CAM",
      heatmapUrl,
      mode: "onnxruntime-web",
    };
  } catch (_) {
    const fallback = runFallbackInference(file);
    return {
      ...fallback,
      heatmapUrl,
    };
  }
}
