"use client";
import { useHud } from "@/lib/store";
import type { BaselineResponse, Compound, Weather } from "@/lib/types";
import { GradeChip, GradeLegend } from "@/components/hud/GradeChip";
import { Slider } from "./Slider";

function Choice<T extends string>({ options, value, onChange, labels, dim }: { options: T[]; value: T; onChange: (v: T) => void; labels?: Record<string, string>; dim?: (o: T) => boolean }) {
  return (
    <div className="flex gap-0.5">
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)} title={dim?.(o) ? "not raced here this weekend" : undefined}
          className={`chip ${o === value ? "chip-on" : "hover:border-hud-muted"} ${dim?.(o) && o !== value ? "opacity-45" : ""}`}>{labels?.[o] ?? o}</button>
      ))}
    </div>
  );
}

const DRY: Compound[] = ["SOFT", "MEDIUM", "HARD"];

export function EnvironmentPanel({ baseline }: { baseline: BaselineResponse | undefined }) {
  const { env, setEnv } = useHud();
  const lap = baseline?.lap;
  const envelope = baseline?.tyre_envelope ?? {};
  // only the compounds the field actually ran this weekend; all three if the store predates the envelope
  const compounds = DRY.filter((c) => envelope[c]);
  const options = compounds.length ? compounds : DRY;
  const track = env.track_temp_c ?? lap?.track_temp_c ?? 30;
  const air = env.air_temp_c ?? lap?.air_temp_c ?? 22;
  const compound = (env.compound ?? (lap?.compound as Compound | null) ?? options[0]) as Compound;
  const limit = envelope[compound];
  // the slider stops where the data stops: the longest stint anyone ran on this compound this weekend
  const maxLaps = limit ? Math.max(limit.max_laps, lap?.tyre_life ?? 1) : 45;
  const tyre = Math.min(env.tyre_life ?? lap?.tyre_life ?? 1, maxLaps);
  const pickCompound = (c: Compound) => {
    const lim = envelope[c]?.max_laps;
    const cur = env.tyre_life ?? lap?.tyre_life ?? 1;
    setEnv({ compound: c, ...(lim !== undefined && cur > lim ? { tyre_life: lim } : {}) });
  };
  return (
    <div className="card p-3 flex flex-col gap-2 shrink-0">
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
        <Choice<Compound> options={options} value={compound} onChange={pickCompound} labels={{ SOFT: "Soft", MEDIUM: "Med", HARD: "Hard" }}
          dim={(c) => !!envelope[c] && envelope[c].max_by_session.R === undefined} />
      </div>
      <Slider env label="Tyre age" grade="A" value={tyre} min={1} max={maxLaps} step={1} display={`${tyre} laps`} hint={env.tyre_life === null ? "baseline" : undefined}
        onChange={(v) => setEnv({ tyre_life: v })} />
      {limit && (
        <div className="text-[10px] font-mono text-hud-dim -mt-1 leading-snug">
          {limit.max_by_session.R === undefined
            ? <>Nobody raced <span className="text-hud-soft">{compound}</span> here this weekend — only {limit.max_laps} qualifying lap{limit.max_laps === 1 ? "" : "s"} on record, so that is where the slider stops.</>
            : <>{compound} ran up to <span className="text-hud-soft">{limit.max_laps} laps</span> here this weekend · typical stint {limit.median_stint} · {limit.n_stints} stints. The slider stops where the data stops.</>}
        </div>
      )}
      {baseline && baseline.strategies.length > 0 && (
        <div className="text-[10px] font-mono text-hud-dim leading-snug border-t border-hud-line pt-2">
          <span className="label">how the race was run</span>
          {baseline.strategies.slice(0, 3).map((st) => (
            <div key={st.sequence} className="flex justify-between gap-2">
              <span className={st.winner ? "text-hud-soft" : ""}>{st.sequence}{st.winner ? " · winner" : ""}</span>
              <span className="shrink-0">×{st.count}</span>
            </div>
          ))}
        </div>
      )}
      <div className="mt-auto"><GradeLegend /></div>
    </div>
  );
}
