"use client";
import type { Grade } from "@/lib/types";
import { GradeChip } from "@/components/hud/GradeChip";

export function Slider({ label, grade, value, min = 0, max = 1, step = 0.01, display, onChange, env = false, hint }: {
  label: string; grade: Grade; value: number; min?: number; max?: number; step?: number;
  display: string; onChange: (v: number) => void; env?: boolean; hint?: string;
}) {
  const changed = Math.abs(value - 0.5) > 1e-9 && max === 1;
  return (
    <label className="block">
      <div className="flex justify-between items-center mb-1.5">
        <span className="flex items-center gap-2 text-hud-soft">{label}<GradeChip grade={grade} /></span>
        <span className={`font-mono ${changed ? "text-series-sim" : "text-hud-soft"}`}>
          {display}{hint && <span className="text-hud-muted"> · {hint}</span>}
        </span>
      </div>
      <input type="range" className={`slider ${env ? "env" : ""}`} min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} aria-label={label} />
    </label>
  );
}
