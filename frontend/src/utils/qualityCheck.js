export function runQualityCheck(file) {
  const score = Math.floor(65 + Math.random() * 30);
  let label = "Good";
  if (score < 70) label = "Needs better lighting";
  if (score < 60) label = "Blurry";
  return { score, label, filename: file.name };
}
