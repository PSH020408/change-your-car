"use client";
import { useHud } from "@/lib/store";
import type { LapMeta } from "@/lib/types";
import { CarSchematic } from "./CarSchematic";
import { Slider } from "./Slider";

export function SetupPanel({ lap }: { lap: LapMeta | undefined }) {
  const { setup, setSetup, reset } = useHud();
  const baseFuel = lap?.session === "Q" ? 8 : 55;
  const fuel = setup.fuel_kg ?? baseFuel;
  return (
    <div className="card p-3 flex flex-col gap-2 shrink-0">
      <div className="flex justify-between items-center">
        <span className="label">Setup</span>
        <button onClick={reset} className="text-[11px] text-hud-muted hover:text-hud-text">reset · 0.50 = this weekend&apos;s car</button>
      </div>
      <CarSchematic setup={setup} />
      <div className="flex flex-col gap-2">
        <Slider label="Front wing" grade="B" value={setup.front_wing} display={setup.front_wing.toFixed(2)} onChange={(v) => setSetup({ front_wing: v })} />
        <Slider label="Rear wing" grade="B" value={setup.rear_wing} display={setup.rear_wing.toFixed(2)} onChange={(v) => setSetup({ rear_wing: v })} />
        <Slider label="Ride height" grade="C" value={setup.ride_height} display={setup.ride_height.toFixed(2)} onChange={(v) => setSetup({ ride_height: v })} />
        <Slider label="Suspension" grade="C" value={setup.suspension} display={setup.suspension.toFixed(2)} onChange={(v) => setSetup({ suspension: v })} />
        <Slider label="Front / rear split" grade="C" value={setup.suspension_split} display={setup.suspension_split.toFixed(2)} onChange={(v) => setSetup({ suspension_split: v })} />
        <Slider label="Fuel" grade="B" value={fuel} min={0} max={110} step={1} display={`${fuel} kg`} hint={setup.fuel_kg === null ? "baseline" : undefined}
          onChange={(v) => setSetup({ fuel_kg: v })} />
      </div>
    </div>
  );
}
