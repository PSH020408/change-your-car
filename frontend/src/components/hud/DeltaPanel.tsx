"use client";
import { useMemo, useState } from "react";
import { fmtDelta, fmtLap, kindLabel } from "@/lib/api";
import type { BaselineResponse, SegmentDelta, SimulationResponse } from "@/lib/types";

const cls = (v: number, eps = 0.0005) => (v < -eps ? "gain" : v > eps ? "loss" : "text-hud-muted");

function Stat({ k, v, sub, tone }: { k: string; v: string; sub?: string; tone?: string }) {
  return (
    <div className="flex flex-col min-w-0">
      <span className="label">{k}</span>
      <span className={`text-[15px] font-mono leading-tight ${tone ?? "text-hud-text"}`}>{v}</span>
      {sub && <span className="text-[10px] font-mono text-hud-dim truncate">{sub}</span>}
    </div>
  );
}

/** Horizontal delta bar centred at zero; lo/hi drawn as a thin range under the bar. */
function DeltaBar({ d, scale, hover, onHover }: { d: SegmentDelta; scale: number; hover: number | null; onHover: (i: number | null) => void }) {
  const W = 120, mid = W / 2;
  const px = (v: number) => mid + (v / scale) * (W / 2 - 2);
  const on = hover === d.index;
  return (
    <div className={`grid grid-cols-[28px_1fr_120px_62px] items-center gap-2 h-6 px-1 rounded cursor-default ${on ? "bg-hud-inset" : ""}`}
      onMouseEnter={() => onHover(d.index)} onMouseLeave={() => onHover(null)}>
      <span className="text-[10px] font-mono text-hud-dim">{String(d.index).padStart(2, "0")}</span>
      <span className="text-[11px] text-hud-soft truncate">{kindLabel(d.kind)} <span className="text-hud-dim">S{d.sector ?? "?"}</span></span>
      <svg viewBox={`0 0 ${W} 12`} width={W} height={12}>
        <line x1={mid} x2={mid} y1={0} y2={12} stroke="#2b333e" />
        <rect x={Math.min(px(0), px(d.total_s))} y={2} width={Math.max(1, Math.abs(px(d.total_s) - px(0)))} height={6}
          fill={d.total_s < 0 ? "#0ca30c" : "#ec835a"} rx={1} />
        <line x1={px(d.total_lo_s)} x2={px(d.total_hi_s)} y1={10.5} y2={10.5} stroke="#5b6472" strokeWidth={1} />
        {Math.abs(d.refused_s) > 0.0005 && <circle cx={px(d.total_s)} cy={5} r={2} fill="#e0a54a" />}
      </svg>
      <span className={`text-[11px] font-mono text-right ${cls(d.total_s)}`}>{fmtDelta(d.total_s)}</span>
    </div>
  );
}

/**
 * Right column, top: the answer. Headline lap delta with its band, how it
 * splits (setup physics / conditions ML / envelope refused), sectors, the
 * segments that move the most, and — expandable — the whole table.
 */
export function DeltaPanel({ baseline, sim, hover, onHover }: {
  baseline: BaselineResponse; sim?: SimulationResponse; hover: number | null; onHover: (i: number | null) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const segs = useMemo(() => sim?.segments ?? [], [sim]);
  const scale = useMemo(() => Math.max(0.05, ...segs.map((s) => Math.max(Math.abs(s.total_lo_s), Math.abs(s.total_hi_s)))), [segs]);
  const top = useMemo(() => [...segs].sort((a, b) => Math.abs(b.total_s) - Math.abs(a.total_s)).slice(0, 7), [segs]);
  const lap = sim?.lap;
  const base = baseline.lap.lap_time_s;

  return (
    <section className="card p-3 flex flex-col gap-3">
      {/* headline */}
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="label">simulated lap</div>
          <div className="text-[30px] font-mono leading-none tabular-nums">{lap ? fmtLap(lap.simulated_lap_time_s) : fmtLap(base)}</div>
          <div className="text-[10px] font-mono text-hud-dim mt-1 whitespace-nowrap">baseline {fmtLap(base)} · {baseline.lap.driver} {baseline.lap.session} {baseline.lap.season}</div>
        </div>
        <div className="text-right">
          <div className="label">delta</div>
          <div className={`text-[30px] font-mono leading-none tabular-nums ${lap ? cls(lap.delta_s) : "text-hud-muted"}`}>{lap ? fmtDelta(lap.delta_s) : "—"}<span className="text-[13px] text-hud-muted ml-1">s</span></div>
          <div className="text-[10px] font-mono text-hud-dim mt-1 whitespace-nowrap">{lap ? `band ${fmtDelta(lap.delta_lo_s)} … ${fmtDelta(lap.delta_hi_s)} (80 %)` : "waiting for simulation"}</div>
        </div>
      </div>

      {/* split */}
      <div className="grid grid-cols-3 gap-2 border-t border-hud-line pt-2">
        <Stat k="setup · physics" v={lap ? fmtDelta(lap.physics_s) : "—"} tone={lap ? cls(lap.physics_s) : undefined} sub="wings · ride height · susp · fuel" />
        <Stat k="conditions · ML" v={lap ? fmtDelta(lap.ml_s) : "—"} tone={lap ? cls(lap.ml_s) : undefined} sub="tyre age · compound · temps" />
        <Stat k="envelope refused" v={lap ? fmtDelta(lap.refused_s) : "—"} tone={lap && Math.abs(lap.refused_s) > 0.0005 ? "text-status-warn" : "text-hud-muted"} sub="g-g limit: not reachable on track" />
      </div>
      {lap && Math.abs(lap.level2_s) > 0.0005 && (
        <div className="text-[10px] font-mono text-hud-dim -mt-1">
          level 2 (session-wide temp shift) {fmtDelta(lap.level2_s)} s — <span className="text-status-warn">reported, not applied</span> (collinear air/track temps, see defect list)
        </div>
      )}

      {/* sectors */}
      <div className="grid grid-cols-3 gap-2 border-t border-hud-line pt-2">
        {[0, 1, 2].map((i) => {
          const v = lap?.sector_deltas_s[i];
          const bt = baseline.lap.sector_times_s[i];
          return (
            <div key={i} className="bg-hud-inset rounded px-2 py-1.5">
              <div className="label">sector {i + 1}</div>
              <div className={`text-[14px] font-mono ${v !== undefined ? cls(v) : "text-hud-muted"}`}>{v !== undefined ? fmtDelta(v) : "—"}</div>
              <div className="text-[10px] font-mono text-hud-dim">{bt != null ? `${bt.toFixed(3)} s` : "—"}</div>
            </div>
          );
        })}
      </div>

      {/* physics state */}
      {sim && (
        <div className="grid grid-cols-4 gap-2 border-t border-hud-line pt-2">
          <Stat k="downforce" v={`${sim.physics.downforce_pct >= 0 ? "+" : ""}${sim.physics.downforce_pct.toFixed(1)} %`} />
          <Stat k="drag" v={`${sim.physics.drag_pct >= 0 ? "+" : ""}${sim.physics.drag_pct.toFixed(1)} %`} />
          <Stat k="mech grip" v={`${sim.physics.mech_grip_pct >= 0 ? "+" : ""}${sim.physics.mech_grip_pct.toFixed(1)} %`} />
          <div className="flex flex-col min-w-0">
            <span className="label">balance</span>
            <svg viewBox="0 0 100 14" className="w-full h-[14px]">
              <line x1={0} x2={100} y1={7} y2={7} stroke="#2b333e" strokeWidth={2} />
              <line x1={50} x2={50} y1={2} y2={12} stroke="#5b6472" />
              <line x1={20} x2={20} y1={4} y2={10} stroke="#3a4350" /><line x1={80} x2={80} y1={4} y2={10} stroke="#3a4350" />
              <circle cx={50 + Math.max(-1, Math.min(1, sim.physics.balance_index)) * 48} cy={7} r={4}
                fill={Math.abs(sim.physics.balance_index) > 0.3 ? "#e0a54a" : "#3987e5"} />
            </svg>
            <span className="text-[10px] font-mono text-hud-dim">{sim.physics.balance_index < -0.05 ? "understeer" : sim.physics.balance_index > 0.05 ? "oversteer" : "neutral"} {sim.physics.balance_index.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* segments */}
      <div className="border-t border-hud-line pt-2">
        <div className="flex items-center justify-between mb-1">
          <span className="label">{showAll ? `all ${segs.length} segments` : "largest movers"}</span>
          <button className="text-[10px] font-mono text-hud-muted hover:text-hud-text" onClick={() => setShowAll((v) => !v)} disabled={!sim}>
            {showAll ? "show top 7 ▴" : `full table (${segs.length}) ▾`}
          </button>
        </div>
        {!sim && <div className="text-[11px] text-hud-dim py-2">move a slider — every segment updates in one call</div>}
        {sim && !showAll && top.map((d) => <DeltaBar key={d.index} d={d} scale={scale} hover={hover} onHover={onHover} />)}
        {sim && showAll && (
          <div className="max-h-[260px] overflow-y-auto -mx-1">
            <table className="w-full text-[10.5px] font-mono">
              <thead className="text-hud-dim sticky top-0 bg-hud-panel">
                <tr className="text-left"><th className="px-1 py-0.5">#</th><th>kind</th><th>S</th><th className="text-right">base</th><th className="text-right">ML</th><th className="text-right">phys</th><th className="text-right">total</th><th className="text-right">band</th><th className="text-right">ref</th></tr>
              </thead>
              <tbody>
                {segs.map((d) => (
                  <tr key={d.index} className={`${hover === d.index ? "bg-hud-inset" : ""} hover:bg-hud-inset`} onMouseEnter={() => onHover(d.index)} onMouseLeave={() => onHover(null)}>
                    <td className="px-1 py-0.5 text-hud-dim">{d.index}</td>
                    <td className="text-hud-soft">{kindLabel(d.kind)}</td>
                    <td className="text-hud-dim">{d.sector ?? "?"}</td>
                    <td className="text-right text-hud-soft">{d.baseline_time_s.toFixed(2)}</td>
                    <td className={`text-right ${cls(d.ml_s)}`}>{fmtDelta(d.ml_s)}</td>
                    <td className={`text-right ${cls(d.physics_s)}`}>{fmtDelta(d.physics_s)}</td>
                    <td className={`text-right ${cls(d.total_s)}`}>{fmtDelta(d.total_s)}</td>
                    <td className="text-right text-hud-dim">{fmtDelta(d.total_lo_s, 2)}…{fmtDelta(d.total_hi_s, 2)}</td>
                    <td className={`text-right ${Math.abs(d.refused_s) > 0.0005 ? "text-status-warn" : "text-hud-dim"}`}>{Math.abs(d.refused_s) > 0.0005 ? fmtDelta(d.refused_s) : "·"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {sim && (
          <div className="text-[10px] font-mono text-hud-dim mt-1.5 flex justify-between">
            <span>bar = total · thin line = 80 % band · <span className="text-status-warn">●</span> envelope refused part</span>
            <span>{sim.computed_ms.toFixed(0)} ms · {sim.model_version}</span>
          </div>
        )}
      </div>
    </section>
  );
}
