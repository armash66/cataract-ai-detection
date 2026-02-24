/**
 * Modality router config for frontend.
 * Maps imaging modality to the ONNX model URL used for inference.
 * When modality-specific models are not yet deployed, both use the fallback.
 */

export const MODALITIES = Object.freeze(["fundus", "anterior"]);

export const DEFAULT_MODALITY = "anterior";

/** Model URL per modality (under public/models/ or served from backend). */
export const MODALITY_MODEL_URLS = Object.freeze({
  fundus: "/models/fundus_multi_disease.onnx",
  anterior: "/models/anterior_multi_disease.onnx",
});

/** Fallback when modality-specific ONNX is not available (single multi-disease model). */
export const FALLBACK_MODEL_URL = "/models/best_accuracy.onnx";

/**
 * @param {string} modality - 'fundus' | 'anterior'
 * @returns {string} URL to use for runInference (modelUrl)
 */
export function getModelUrlForModality(modality) {
  const key = (modality || "").toLowerCase();
  if (!MODALITIES.includes(key)) return FALLBACK_MODEL_URL;
  return MODALITY_MODEL_URLS[key] || FALLBACK_MODEL_URL;
}
