"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { api, kindLabel } from "@/lib/api";
import { useHud } from "@/lib/store";
import type { SegmentInfo, SimulationResponse, TelemetryTrace } from "@/lib/types";
import { TopBar } from "@/components/hud/TopBar";
import { DeltaPanel } from "@/components/hud/DeltaPanel";
import { EngineerLog } from "@/components/hud/EngineerLog";
import { SetupPanel } from "@/components/setup/SetupPanel";
import { EnvironmentPanel } from "@/components/setup/EnvironmentPanel";
import { TrackMap } from "@/components/track/TrackMap";
import { TelemetryChart } from "@/components/telemetry/TelemetryChart";

const DEBOUNCE_MS = 150;

/**
 * First-order speed band for the chart: inside a segment, speed scales
 * roughly with 1 / segment time, so the segment's total_lo/hi (s) become a
 * speed range around the simulated trace. Visual aid only — the numbers that
 * matter are the per-segment time bands in the delta panel.
 */
function speedBand(sim: SimulationResponse, segments: SegmentInfo[], lapLength: number): { lo: number[]; hi: number[] } {
  const bySeg = new Map(sim.segments.map((d) => [d.index, d]));
  const find = (dist: number) => segments.find((s) => (s.end_m >= s.start_m ? dist >= s.start_m && dist < s.end_m : dist >= s.start_m || dist < s.end_m));
  const lo: number[] = [], hi: number[] = [];
  sim.simulated.distance_m.forEach((dist, i) => {
    const v = sim.simulated.speed_kph[i];
    const s = find(((dist % lapLength) + lapLength) % lapLength);
    const d = s ? bySeg.get(s.index) : undefined;
    if (!d) { lo.push(v); hi.push(v); return; }
    const T = Math.max(0.05, d.baseline_time_s + d.total_s);
    hi.push(v * (T / Math.max(0.05, T + (d.total_lo_s - d.total_s))));
    lo.push(v * (T / Math.max(0.05, T + (d.total_hi_s - d.total_s))));
  });
  return { lo, hi };
}

export default function Page() {
  const { ref, setup, env } = useHud();
  const [hover, setHover] = useState<number | null>(null);
  const [sim, setSim] = useState<SimulationResponse | undefined>();
  const [busy, setBusy] = useState(false);
  const [simError, setSimError] = useState<Error | undefined>();
  const abortRef = useRef<AbortController | null>(null);

  const baseline = useSWR(["baseline", ref.season, ref.event, ref.session, ref.driver, ref.lap], () => api.baseline(ref), { keepPreviousData: true, revalidateOnFocus: false });

  // one POST per settled slider position; abort whatever is still in flight
  useEffect(() => {
    if (!baseline.data) return;
    setBusy(true);
    const t = setTimeout(() => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      api.simulate({ baseline: ref, setup, environment: env }, ac.signal)
        .then((r) => { if (!ac.signal.aborted) { setSim(r); setSimError(undefined); setBusy(false); } })
        .catch((e: Error) => { if (e.name !== "AbortError") { setSimError(e); setBusy(false); } });
    }, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [baseline.data, ref, setup, env]);

  // a new baseline invalidates the previous answer
  useEffect(() => { setSim(undefined); }, [ref.season, ref.event, ref.session, ref.driver, ref.lap]);

  const b = baseline.data;
  const band = useMemo(() => (sim && b ? speedBand(sim, b.segments, b.track.lap_length_m) : undefined), [sim, b]);
  const simTrace: TelemetryTrace | undefined = sim?.simulated;
  const hovered = b?.segments.find((s) => s.index === hover);
  const hoveredDelta = sim?.segments.find((s) => s.index === hover);
  const error = (baseline.error as Error | undefined) ?? simError;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <TopBar baseline={b} error={error} busy={busy || baseline.isLoading} />

      {!b && (
        <main className="flex-1 grid place-items-center text-hud-muted font-mono text-[12px]">
          {baseline.error ? (
            <div className="card p-4 max-w-[520px] text-status-loss">
              <div className="label mb-1">baseline not available</div>
              <div className="text-hud-soft whitespace-pre-wrap">{String((baseline.error as Error).message)}</div>
              <div className="text-hud-dim mt-2">Is the API running? <span className="text-hud-soft">make api</span> → http://localhost:8000/health</div>
            </div>
          ) : "loading baseline…"}
        </main>
      )}

      {b && (
        <main className="flex-1 min-h-0 grid grid-cols-[320px_minmax(0,1fr)_368px] gap-3 p-3">
          <aside className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-0.5">
            <SetupPanel lap={b.lap} />
            <EnvironmentPanel lap={b.lap} />
          </aside>

          <section className="flex flex-col gap-3 min-h-0">
            <div className="card flex-1 min-h-0 p-3 flex flex-col">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-baseline gap-2 min-w-0">
                  <span className="text-[13px] font-semibold truncate">{b.lap.circuit ?? b.lap.event_name}</span>
                  <span className="text-[10px] font-mono text-hud-dim">{b.track.lap_length_m.toFixed(0)} m · {b.segments.length} segments · {b.track.published_turns ?? b.track.measured_turns} turns</span>
                </div>
                <div className="text-[10px] font-mono text-hud-muted h-4">
                  {hovered ? `#${hovered.index} ${kindLabel(hovered.kind)} · S${hovered.sector ?? "?"} · ${hovered.length_m.toFixed(0)} m${hovered.min_radius_m ? ` · r ${hovered.min_radius_m.toFixed(0)} m` : ""}${hoveredDelta ? ` · Δ ${hoveredDelta.total_s >= 0 ? "+" : "−"}${Math.abs(hoveredDelta.total_s).toFixed(3)} s` : ""}` : "hover a segment"}
                </div>
              </div>
              <div className="flex-1 min-h-0 mt-2">
                <TrackMap track={b.track} segments={b.segments} deltas={sim?.segments} hover={hover} onHover={setHover} />
              </div>
              <div className="flex gap-3 text-[10px] font-mono text-hud-dim mt-1">
                <span><span className="gain">■</span> faster</span><span><span className="loss">■</span> slower</span><span><span style={{ color: "#3a4350" }}>■</span> unchanged</span>
                <span className="ml-auto">colour intensity ∝ |Δ| · white tick = start/finish</span>
              </div>
            </div>

            <div className="card h-[352px] p-3 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="label">telemetry · real vs simulated</span>
                <div className="flex gap-3 text-[10px] font-mono">
                  <span className="text-series-real">— real {b.lap.driver}</span>
                  <span className="text-series-sim">— simulated</span>
                  <span className="text-hud-dim">▨ interpolated (gap in FastF1 data)</span>
                </div>
              </div>
              <div className="flex-1 min-h-0 mt-1">
                <TelemetryChart real={b.trace} sim={simTrace} segments={b.segments} hover={hover} onHover={setHover} band={band} />
              </div>
            </div>
          </section>

          <aside className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-0.5">
            <DeltaPanel baseline={b} sim={sim} hover={hover} onHover={setHover} />
            <EngineerLog notes={sim?.engineer_log} pending={busy} />
          </aside>
        </main>
      )}
    </div>
  );
}
