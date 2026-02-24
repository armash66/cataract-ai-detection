import { useId } from "react";

const MODALITY_OPTIONS = [
  { value: "fundus", label: "Fundus" },
  { value: "anterior", label: "Anterior" },
];

export default function ModalitySelector({ value, onChange }) {
  const id = useId();
  return (
    <div className="modality-selector">
      <label htmlFor={id} className="modality-label">
        Image type
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="modality-select"
        aria-label="Select imaging modality (fundus or anterior)"
      >
        {MODALITY_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
