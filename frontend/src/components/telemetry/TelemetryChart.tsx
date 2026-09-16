"use client";
import { useMemo, useState } from "react";
import type { SegmentInfo, TelemetryTrace } from "@/lib/types";

const W = 720, PAD_L = 44, PAD_R = 8;
const ROWS = { speed: [16, 150], throttle: [172, 236], brake: [252, 274], drs: [286, 300] } as const;

function xScale(d: number[]) {
  const max = d[d.length - 1] || 1;
  return (v: number) => PAD_L + ((W - PAD_L - PAD_R) * v) / max;
}
function yScale([top, bottom]: readonly [number, number], lo: number, hi: number) {
  return (v: number) => bottom - ((bottom - top) * (v - lo)) / (hi - lo || 1);
}
function line(xs: number[], ys: number[]) {
  return xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
}
function bands(d: number[], on: boolean[], x: (v: number) => number) {
  const out: [number, number][] = [];
  let start: number | null = null;
  on.forEach((b, i) => {
    if (b && start === null) start = d[i];
    if ((!b || i === on.length - 1) && start !== null) { out.push([x(start), x(d[i])]); start = null; }
  });
  return out;
}

/**
 * Real vs simulated, four channels, one distance axis. Hand-drawn SVG:
 * band as a fill, interpolated spans hatched, crosshair + tooltip on hover.
 */
export function TelemetryChart({ real, sim, segments, hover, onHover, band }: {
  real: TelemetryTrace; sim?: TelemetryTrace; segments: SegmentInfo[];
  hover: number | null; onHover: (i: number | null) => void; band?: { lo: number[]; hi: number[] };
}) {
  const [cursor, setCursor] = useState<number | null>(null);
  const d = real.distance_m;
  const x = useMemo(() => xScale(d), [d]);
  const vmax = Math.max(...real.speed_kph, ...(sim?.speed_kph ?? [])) * 1.03;
  const vmin = Math.min(...real.speed_kph, ...(sim?.speed_kph ?? [])) * 0.9;
  const ys = yScale(ROWS.speed, vmin, vmax), yt = yScale(ROWS.throttle, 0, 100);
  const xs = d.map(x);
  const hatch = bands(d, real.interpolated, x);
  const segOf = (dist: number) => segments.find((s) => dist >= s.start_m && dist < s.end_m)?.index ?? null;
  const i = cursor === null ? null : Math.min(d.length - 1, Math.max(0, Math.round(((cursor - PAD_L) / (W - PAD_L - PAD_R)) * (d.length - 1))));
  const hoverSeg = segments.find((s) => s.index === hover);

  return (
    <svg viewBox={`0 0 ${W} 320`} className="w-full h-full select-none" fill="none"
      onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W; setCursor(px); const dist = ((px - PAD_L) / (W - PAD_L - PAD_R)) * d[d.length - 1]; onHover(segOf(dist)); }}
      onMouseLeave={() => { setCursor(null); onHover(null); }}>
      <defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2" height="4" fill="#2b333e" /></pattern></defs>
      {hoverSeg && <rect x={x(hoverSeg.start_m)} y={ROWS.speed[0]} width={Math.max(1, x(hoverSeg.end_m) - x(hoverSeg.start_m))} height={ROWS.drs[1] - ROWS.speed[0]} fill="#3987e5" opacity={0.08} />}
      {hatch.map(([a, b], k) => <rect key={k} x={a} y={ROWS.speed[0]} width={Math.max(1, b - a)} height={ROWS.speed[1] - ROWS.speed[0]} fill="url(#hatch)" opacity={0.6} />)}

      {/* speed */}
      <text x={0} y={12} className="fill-hud-muted" fontSize={10} fontFamily="var(--font-plex-mono)">SPEED km/h</text>
      <path d={`M${PAD_L} ${ROWS.speed[0]} V${ROWS.speed[1]} H${W - PAD_R}`} stroke="#232a33" />
      <text x={PAD_L - 6} y={ROWS.speed[0] + 8} textAnchor="end" fill="#5b6472" fontSize={9} fontFamily="var(--font-plex-mono)">{Math.round(vmax)}</text>
      <text x={PAD_L - 6} y={ROWS.speed[1]} textAnchor="end" fill="#5b6472" fontSize={9} fontFamily="var(--font-plex-mono)">{Math.round(vmin)}</text>
      {sim && band && (
        <path d={`${line(xs, band.hi.map(ys))} ${[...xs].reverse().map((xx, k) => `L${xx.toFixed(1)} ${ys(band.lo[band.lo.length - 1 - k]).toFixed(1)}`).join(" ")} Z`} fill="rgba(57,135,229,0.18)" />
      )}
      <path d={line(xs, real.speed_kph.map(ys))} stroke="#c3c2b7" strokeWidth={1.6} />
      {sim && <path d={line(xs, sim.speed_kph.map(ys))} stroke="#3987e5" strokeWidth={2} />}

      {/* throttle */}
      <text x={0} y={ROWS.throttle[0] - 4} className="fill-hud-muted" fontSize={10} fontFamily="var(--font-plex-mono)">THROTTLE %</text>
      <path d={`M${PAD_L} ${ROWS.throttle[0]} V${ROWS.throttle[1]} H${W - PAD_R}`} stroke="#232a33" />
      <path d={line(xs, real.throttle_pct.map(yt))} stroke="#c3c2b7" strokeWidth={1.4} />
      {sim && <path d={line(xs, sim.throttle_pct.map(yt))} stroke="#3987e5" strokeWidth={1.8} />}

      {/* brake */}
      <text x={0} y={ROWS.brake[0] + 12} className="fill-hud-muted" fontSize={10} fontFamily="var(--font-plex-mono)">BRAKE</text>
      {bands(d, real.brake_on, x).map(([a, b], k) => <rect key={`rb${k}`} x={a} y={ROWS.brake[0]} width={Math.max(1, b - a)} height={9} fill="#c3c2b7" opacity={0.5} />)}
      {sim && bands(d, sim.brake_on, x).map(([a, b], k) => <rect key={`sb${k}`} x={a} y={ROWS.brake[0] + 12} width={Math.max(1, b - a)} height={9} fill="#3987e5" />)}

      {/* DRS */}
      <text x={0} y={ROWS.drs[0] + 10} className="fill-hud-muted" fontSize={10} fontFamily="var(--font-plex-mono)">DRS</text>
      {bands(d, real.drs_open, x).map(([a, b], k) => <rect key={`d${k}`} x={a} y={ROWS.drs[0]} width={Math.max(1, b - a)} height={10} fill="#0ca30c" opacity={0.6} />)}

      {/* axis */}
      {[0, 0.5, 1].map((f) => <text key={f} x={x(f * d[d.length - 1])} y={316} textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"} fill="#5b6472" fontSize={9} fontFamily="var(--font-plex-mono)">{Math.round(f * d[d.length - 1]).toLocaleString()} m</text>)}

      {/* crosshair + tooltip */}
      {i !== null && (
        <g>
          <path d={`M${xs[i]} ${ROWS.speed[0]} V${ROWS.drs[1]}`} stroke="#8b93a3" strokeDasharray="2 3" />
          <circle cx={xs[i]} cy={ys(real.speed_kph[i])} r={3.5} fill="#c3c2b7" stroke="#0f1216" strokeWidth={2} />
          {sim && <circle cx={xs[i]} cy={ys(sim.speed_kph[i])} r={3.5} fill="#3987e5" stroke="#0f1216" strokeWidth={2} />}
          <g transform={`translate(${Math.min(xs[i] + 10, W - 170)}, ${ROWS.speed[0]})`}>
            <rect width={160} height={sim ? 62 : 36} rx={4} fill="#151a20" stroke="#2b333e" />
            <text x={8} y={14} fill="#8b93a3" fontSize={10} fontFamily="var(--font-plex-mono)">{Math.round(d[i]).toLocaleString()} m{hover !== null ? ` · seg ${hover}` : ""}{real.interpolated[i] ? " · interp." : ""}</text>
            <text x={8} y={29} fill="#c3c2b7" fontSize={11} fontFamily="var(--font-plex-mono)">real  {real.speed_kph[i].toFixed(0)} km/h · {real.throttle_pct[i].toFixed(0)}%{real.brake_on[i] ? " · brake" : ""}</text>
            {sim && <text x={8} y={44} fill="#3987e5" fontSize={11} fontFamily="var(--font-plex-mono)">sim   {sim.speed_kph[i].toFixed(0)} km/h · {sim.throttle_pct[i].toFixed(0)}%{sim.brake_on[i] ? " · brake" : ""}</text>}
            {sim && <text x={8} y={57} fill="#8b93a3" fontSize={10} fontFamily="var(--font-plex-mono)">Δv {(sim.speed_kph[i] - real.speed_kph[i]).toFixed(1)} km/h · Δt {(sim.time_s[i] - real.time_s[i]).toFixed(3)} s</text>}
          </g>
        </g>
      )}
    </svg>
  );
}
