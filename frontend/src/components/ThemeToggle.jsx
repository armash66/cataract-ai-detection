import { Moon, Sun } from "lucide-react";

export default function ThemeToggle({ theme, onToggle }) {
  return (
    <button className="pill-btn" onClick={onToggle}>
      {theme === "light" ? <Moon size={14} /> : <Sun size={14} />} {theme}
    </button>
  );
}
