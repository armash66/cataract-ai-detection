import { Moon, Sun } from "lucide-react";

export default function ThemeToggle({ theme, onToggle }) {
  const isLight = theme === "light";
  const label = isLight ? "Switch to dark mode" : "Switch to light mode";

  return (
    <button className="theme-toggle" type="button" onClick={onToggle} aria-label={label} title={label}>
      <span className="theme-toggle-track" aria-hidden="true">
        <span className={`theme-toggle-thumb ${isLight ? "is-light" : ""}`}>
          {isLight ? <Moon size={14} /> : <Sun size={14} />}
        </span>
      </span>
      <span className="theme-toggle-text">{isLight ? "Light" : "Dark"}</span>
    </button>
  );
}
