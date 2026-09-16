"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
const REPLAY_SPEEDS = [1, 3, 10];

/** Distance (m) reached at lap time t, by linear interpolation on the trace's own time axis. */
function distAt(tr: TelemetryTrace, t: number): number {
  const T = tr.time_s, D = tr.distance_m;
  if (t <= T[0]) return D[0];
  if (t >= T[T.length - 1]) return D[D.length - 1];
  let lo = 0, hi = T.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
  const f = (t - T[lo]) / (T[hi] - T[lo] || 1);
  return D[lo] + f * (D[hi] - D[lo]);
}
/** Lap time (s) at which the trace passes distance x. */
function timeAt(tr: TelemetryTrace, x: number): number {
  const T = tr.time_s, D = tr.distance_m;
  if (x <= D[0]) return T[0];
  if (x >= D[D.length - 1]) return T[T.length - 1];
  let lo = 0, hi = D.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (D[m] <= x) lo = m; else hi = m; }
  const f = (x - D[lo]) / (D[hi] - D[lo] || 1);
  return T[lo] + f * (T[hi] - T[lo]);
}

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
  const [replay, setReplay] = useState({ playing: false, t: 0, speed: 1 });

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

  // replay clock: both cars start together; loops one second after the slower one finishes
  useEffect(() => {
    if (!replay.playing || !b) return;
    const lapEnd = Math.max(b.trace.time_s[b.trace.time_s.length - 1] ?? 0, sim?.simulated.time_s[sim.simulated.time_s.length - 1] ?? 0) + 1;
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      setReplay((r) => ({ ...r, t: (r.t + dt * r.speed) % lapEnd }));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [replay.playing, replay.speed, b, sim]);
  const toggleReplay = useCallback(() => setReplay((r) => ({ ...r, playing: !r.playing })), []);
  const cycleSpeed = useCallback(() => setReplay((r) => ({ ...r, speed: REPLAY_SPEEDS[(REPLAY_SPEEDS.indexOf(r.speed) + 1) % REPLAY_SPEEDS.length] })), []);
  const stopReplay = useCallback(() => setReplay((r) => ({ ...r, playing: false, t: 0 })), []);
  const simTraceForReplay = sim?.simulated ?? b?.trace;
  const replayOn = replay.playing || replay.t > 0;
  const realDist = b && replayOn ? distAt(b.trace, replay.t) : null;
  const simDist = simTraceForReplay && replayOn ? distAt(simTraceForReplay, replay.t) : null;
  const gap = b && simTraceForReplay && realDist !== null ? timeAt(simTraceForReplay, realDist) - replay.t : null;
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

          <section className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-0.5">
            <div className="card shrink-0 h-[calc(100vh-84px)] p-3 flex flex-col relative">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-baseline gap-2 min-w-0">
                  <span className="text-[13px] font-semibold truncate">{b.lap.circuit ?? b.lap.event_name}</span>
                  <span className="text-[10px] font-mono text-hud-dim">{b.track.lap_length_m.toFixed(0)} m · {b.segments.length} segments · {b.track.published_turns ?? b.track.measured_turns} turns</span>
                </div>
                <div className="text-[10px] font-mono text-hud-muted h-4">
                  {hovered ? `#${hovered.index} ${kindLabel(hovered.kind)} · S${hovered.sector ?? "?"} · ${hovered.length_m.toFixed(0)} m${hovered.min_radius_m ? ` · r ${hovered.min_radius_m.toFixed(0)} m` : ""}${hoveredDelta ? ` · Δ ${hoveredDelta.total_s >= 0 ? "+" : "−"}${Math.abs(hoveredDelta.total_s).toFixed(3)} s` : ""}` : "hover a segment"}
                </div>
              </div>
              <div className="flex-1 min-h-0 mt-2 relative">
                {replayOn && gap !== null && (
                  <div className="absolute right-2 top-1 pointer-events-none text-right">
                    <div className="label">sim vs real · same moment</div>
                    <div className={`text-[34px] font-mono leading-none tabular-nums ${gap < -0.0005 ? "gain" : gap > 0.0005 ? "loss" : "text-hud-soft"}`}>
                      {gap >= 0 ? "+" : "−"}{Math.abs(gap).toFixed(3)}<span className="text-[13px] text-hud-muted ml-1">s</span>
                    </div>
                    <div className="text-[11px] font-mono text-hud-muted mt-0.5">{gap < -0.0005 ? "SIM ahead" : gap > 0.0005 ? "SIM behind" : "level"} · t {replay.t.toFixed(1)} s</div>
                  </div>
                )}
                <TrackMap track={b.track} segments={b.segments} deltas={sim?.segments} hover={hover} onHover={setHover}
                  markers={realDist !== null && simDist !== null ? { realDist, simDist } : null} />
              </div>
              <div className="flex items-center gap-2 text-[10px] font-mono text-hud-dim mt-1">
                <button onClick={toggleReplay} className="w-8 h-8 rounded border border-hud-line2 bg-hud-inset hover:border-hud-muted grid place-items-center" title={replay.playing ? "pause" : "play both laps from the line"} aria-label={replay.playing ? "pause" : "play"}>
                  {replay.playing
                    ? <svg width="12" height="12" viewBox="0 0 12 12"><rect x="1.5" y="1" width="3.2" height="10" rx="0.8" fill="#e6e8ec" /><rect x="7.3" y="1" width="3.2" height="10" rx="0.8" fill="#e6e8ec" /></svg>
                    : <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2.5 1.2 L10.8 6 L2.5 10.8 Z" fill="#e6e8ec" /></svg>}
                </button>
                <button onClick={stopReplay} disabled={!replayOn} className="w-8 h-8 rounded border border-hud-line2 bg-hud-inset hover:border-hud-muted disabled:opacity-40 grid place-items-center" title="back to the line" aria-label="stop">
                  <svg width="12" height="12" viewBox="0 0 12 12"><rect x="1.5" y="1.5" width="9" height="9" rx="1" fill="#e6e8ec" /></svg>
                </button>
                <button onClick={cycleSpeed} className="h-8 px-2 rounded border border-hud-line2 bg-hud-inset hover:border-hud-muted text-[11px] font-mono text-hud-text" title="playback speed">{replay.speed}×</button>
                <span className="ml-auto flex gap-3">
                  <span><span className="gain">■</span> sim faster</span><span><span className="loss">■</span> sim slower</span><span><span style={{ color: "#3a4350" }}>■</span> same</span>
                  <span className="text-series-real">● real</span><span className="text-series-sim">● sim</span>
                </span>
              </div>
            </div>

            <div className="card shrink-0 h-[420px] p-3 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="label">telemetry · real vs simulated</span>
                <div className="flex gap-3 text-[10px] font-mono">
                  <span className="text-series-real">— real {b.lap.driver}</span>
                  <span className="text-series-sim">— simulated</span>
                  <span className="text-hud-dim">▨ interpolated · <span className="gain">■</span> sim faster/ahead <span className="loss">■</span> slower/behind</span>
                </div>
              </div>
              <div className="flex-1 min-h-0 mt-1">
                <TelemetryChart real={b.trace} sim={simTrace} segments={b.segments} hover={hover} onHover={setHover} band={band} markerDist={realDist} />
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
