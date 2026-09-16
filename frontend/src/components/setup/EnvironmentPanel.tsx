"use client";
import { useHud } from "@/lib/store";
import type { Compound, LapMeta, Weather } from "@/lib/types";
import { GradeChip, GradeLegend } from "@/components/hud/GradeChip";
import { Slider } from "./Slider";

function Choice<T extends string>({ options, value, onChange, labels }: { options: T[]; value: T; onChange: (v: T) => void; labels?: Record<string, string> }) {
  return (
    <div className="flex gap-0.5">
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)} className={`chip ${o === value ? "chip-on" : "hover:border-hud-muted"}`}>{labels?.[o] ?? o}</button>
      ))}
    </div>
  );
}

export function EnvironmentPanel({ lap }: { lap: LapMeta | undefined }) {
  const { env, setEnv } = useHud();
  const track = env.track_temp_c ?? lap?.track_temp_c ?? 30;
  const air = env.air_temp_c ?? lap?.air_temp_c ?? 22;
  const compound = (env.compound ?? (lap?.compound as Compound | null) ?? "MEDIUM") as Compound;
  const tyre = env.tyre_life ?? lap?.tyre_life ?? 1;
  return (
    <div className="card p-4 flex flex-col gap-3 flex-1">
      <span className="label">Environment</span>
      <Slider env label="Track temp" grade="A" value={track} min={10} max={60} step={1} display={`${track} °C`} hint={env.track_temp_c === null ? "session" : undefined}
        onChange={(v) => setEnv({ track_temp_c: v })} />
      <Slider env label="Air temp" grade="A" value={air} min={5} max={45} step={1} display={`${air} °C`} hint={env.air_temp_c === null ? "session" : undefined}
        onChange={(v) => setEnv({ air_temp_c: v })} />
      <div className="flex justify-between items-center">
        <span className="flex items-center gap-2 text-hud-soft">Weather <GradeChip grade="C" /></span>
        <Choice<Weather> options={["dry", "inter", "wet"]} value={env.weather} onChange={(w) => setEnv({ weather: w })} labels={{ dry: "Dry", inter: "Inter", wet: "Wet" }} />
      </div>
      <div className="flex justify-between items-center">
        <span className="flex items-center gap-2 text-hud-soft">Compound <GradeChip grade="A" /></span>
        <Choice<Compound> options={["SOFT", "MEDIUM", "HARD"]} value={compound} onChange={(c) => setEnv({ compound: c })} labels={{ SOFT: "Soft", MEDIUM: "Med", HARD: "Hard" }} />
      </div>
      <Slider env label="Tyre age" grade="A" value={tyre} min={1} max={45} step={1} display={`${tyre} laps`} hint={env.tyre_life === null ? "baseline" : undefined}
        onChange={(v) => setEnv({ tyre_life: v })} />
      <div className="mt-auto"><GradeLegend /></div>
    </div>
  );
}
