const diseaseRecommendations = {
  Glaucoma: "Recommend intraocular pressure test and visual field exam.",
  "Diabetic Retinopathy": "Recommend HbA1c panel and retinal specialist follow-up.",
  Cataract: "Recommend ophthalmologist consultation for lens opacity assessment.",
  Normal: "Continue periodic eye screening.",
  Other: "Recommend full ophthalmic exam for differential diagnosis."
};

export function computeRisk(result, patient) {
  const base = Math.round(result.confidence * 60);
  const age = patient.age > 60 ? 15 : patient.age > 45 ? 8 : 3;
  const diabetes = patient.diabetes ? 15 : 0;
  const family = patient.familyHistory ? 10 : 0;
  const symptomMap = { none: 0, mild_blur: 8, severe_blur: 15, pain: 12 };
  const symptom = symptomMap[patient.symptoms] ?? 0;

  const score = Math.min(100, base + age + diabetes + family + symptom);
  return {
    score,
    recommendation: diseaseRecommendations[result.topClass] || diseaseRecommendations.Other
  };
}
