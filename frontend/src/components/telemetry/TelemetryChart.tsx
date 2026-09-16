"use client";
import { useMemo, useState } from "react";
import type { SegmentInfo, TelemetryTrace } from "@/lib/types";

const W = 720, PAD_L = 44, PAD_R = 8, H = 392;
const ROWS = { speed: [14, 118], dspeed: [140, 190], dtime: [212, 262], throttle: [284, 324], brake: [338, 352], drs: [362, 374] } as const;
const REAL = "#c3c2b7", SIM = "#3987e5", GAIN = "#0ca30c", LOSS = "#ec835a";
const MONO = "var(--font-plex-mono)";

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
/** Area between the zero line and the series, split by sign so gain/loss get their own colour. */
function signedAreas(xs: number[], v: number[], y: (v: number) => number) {
  const pos = v.map((a) => Math.max(0, a)), neg = v.map((a) => Math.min(0, a));
  const area = (vals: number[]) => `${line(xs, vals.map(y))} L${xs[xs.length - 1].toFixed(1)} ${y(0).toFixed(1)} L${xs[0].toFixed(1)} ${y(0).toFixed(1)} Z`;
  return { pos: area(pos), neg: area(neg) };
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
const Label = ({ y, children }: { y: number; children: React.ReactNode }) => (
  <text x={0} y={y} fill="#8b93a3" fontSize={10} fontFamily={MONO}>{children}</text>
);
const Tick = ({ y, v, anchor = "end" }: { y: number; v: string; anchor?: "end" | "start" }) => (
  <text x={PAD_L - 6} y={y} textAnchor={anchor} fill="#5b6472" fontSize={9} fontFamily={MONO}>{v}</text>
);

/**
 * Real vs simulated on one distance axis. The two DELTA rows are the point:
 * a 0.3 s lap gain is ~0.3 % and invisible on an absolute speed trace, so
 * Δspeed (km/h, sim − real) and the running Δtime (s, sim − real) are drawn
 * as signed areas — green where the simulated car is faster / ahead.
 */
export function TelemetryChart({ real, sim, segments, hover, onHover, band, markerDist }: {
  real: TelemetryTrace; sim?: TelemetryTrace; segments: SegmentInfo[];
  hover: number | null; onHover: (i: number | null) => void; band?: { lo: number[]; hi: number[] };
  markerDist?: number | null;
}) {
  const [cursor, setCursor] = useState<number | null>(null);
  const d = real.distance_m;
  const L = d[d.length - 1];
  const x = useMemo(() => xScale(d), [d]);
  const xs = useMemo(() => d.map(x), [d, x]);
  const vmax = Math.max(...real.speed_kph, ...(sim?.speed_kph ?? [])) * 1.03;
  const vmin = Math.min(...real.speed_kph, ...(sim?.speed_kph ?? [])) * 0.9;
  const ys = yScale(ROWS.speed, vmin, vmax), yt = yScale(ROWS.throttle, 0, 100);

  const dv = useMemo(() => (sim ? sim.speed_kph.map((v, i) => v - real.speed_kph[i]) : []), [sim, real]);
  const dt = useMemo(() => (sim ? sim.time_s.map((t, i) => t - real.time_s[i]) : []), [sim, real]);
  const dvMax = Math.max(2, ...dv.map(Math.abs));
  const dtMax = Math.max(0.05, ...dt.map(Math.abs));
  const ydv = yScale(ROWS.dspeed, -dvMax, dvMax), ydt = yScale(ROWS.dtime, dtMax, -dtMax); // time: ahead (negative) drawn upward
  const dvArea = useMemo(() => (sim ? signedAreas(xs, dv, ydv) : null), [sim, xs, dv, ydv]);
  const dtArea = useMemo(() => (sim ? signedAreas(xs, dt, ydt) : null), [sim, xs, dt, ydt]);

  const hatch = bands(d, real.interpolated, x);
  const segOf = (dist: number) => segments.find((s) => dist >= s.start_m && dist < s.end_m)?.index ?? null;
  const idxAt = (px: number) => Math.min(d.length - 1, Math.max(0, Math.round(((px - PAD_L) / (W - PAD_L - PAD_R)) * (d.length - 1))));
  const i = cursor !== null ? idxAt(cursor) : markerDist != null ? Math.min(d.length - 1, Math.max(0, Math.round((markerDist / L) * (d.length - 1)))) : null;
  const hoverSeg = segments.find((s) => s.index === hover);
  const Y0 = ROWS.speed[0], Y1 = ROWS.drs[1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full select-none" fill="none"
      onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W; setCursor(px); onHover(segOf(((px - PAD_L) / (W - PAD_L - PAD_R)) * L)); }}
      onMouseLeave={() => { setCursor(null); onHover(null); }}>
      <defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2" height="4" fill="#2b333e" /></pattern></defs>
      {hoverSeg && <rect x={x(hoverSeg.start_m)} y={Y0} width={Math.max(1, x(hoverSeg.end_m) - x(hoverSeg.start_m))} height={Y1 - Y0} fill={SIM} opacity={0.08} />}
      {hatch.map(([a, b], k) => <rect key={k} x={a} y={Y0} width={Math.max(1, b - a)} height={ROWS.speed[1] - Y0} fill="url(#hatch)" opacity={0.6} />)}

      {/* speed */}
      <Label y={10}>SPEED km/h</Label>
      <path d={`M${PAD_L} ${ROWS.speed[0]} V${ROWS.speed[1]} H${W - PAD_R}`} stroke="#232a33" />
      <Tick y={ROWS.speed[0] + 8} v={String(Math.round(vmax))} /><Tick y={ROWS.speed[1]} v={String(Math.round(vmin))} />
      {sim && band && (
        <path d={`${line(xs, band.hi.map(ys))} ${[...xs].reverse().map((xx, k) => `L${xx.toFixed(1)} ${ys(band.lo[band.lo.length - 1 - k]).toFixed(1)}`).join(" ")} Z`} fill="rgba(57,135,229,0.16)" />
      )}
      <path d={line(xs, real.speed_kph.map(ys))} stroke={REAL} strokeWidth={2} strokeOpacity={sim ? 0.9 : 1} />
      {sim && <path d={line(xs, sim.speed_kph.map(ys))} stroke={SIM} strokeWidth={1.6} />}

      {/* Δ speed */}
      <Label y={ROWS.dspeed[0] - 4}>Δ SPEED km/h · sim − real</Label>
      <path d={`M${PAD_L} ${ydv(0)} H${W - PAD_R}`} stroke="#2b333e" />
      <Tick y={ROWS.dspeed[0] + 8} v={`+${dvMax.toFixed(0)}`} /><Tick y={ROWS.dspeed[1]} v={`−${dvMax.toFixed(0)}`} />
      {dvArea && <><path d={dvArea.pos} fill={GAIN} opacity={0.55} /><path d={dvArea.neg} fill={LOSS} opacity={0.6} /></>}
      {!sim && <text x={W / 2} y={ydv(0) - 4} textAnchor="middle" fill="#5b6472" fontSize={10} fontFamily={MONO}>no change yet</text>}

      {/* Δ time (running) */}
      <Label y={ROWS.dtime[0] - 4}>Δ TIME s · running, sim − real · up = ahead</Label>
      <path d={`M${PAD_L} ${ydt(0)} H${W - PAD_R}`} stroke="#2b333e" />
      <Tick y={ROWS.dtime[0] + 8} v={`−${dtMax.toFixed(2)}`} /><Tick y={ROWS.dtime[1]} v={`+${dtMax.toFixed(2)}`} />
      {dtArea && <><path d={dtArea.neg} fill={GAIN} opacity={0.45} /><path d={dtArea.pos} fill={LOSS} opacity={0.5} /><path d={line(xs, dt.map(ydt))} stroke="#e6e8ec" strokeWidth={1.2} /></>}

      {/* throttle */}
      <Label y={ROWS.throttle[0] - 4}>THROTTLE %</Label>
      <path d={`M${PAD_L} ${ROWS.throttle[0]} V${ROWS.throttle[1]} H${W - PAD_R}`} stroke="#232a33" />
      <path d={line(xs, real.throttle_pct.map(yt))} stroke={REAL} strokeWidth={1.4} strokeOpacity={0.9} />
      {sim && <path d={line(xs, sim.throttle_pct.map(yt))} stroke={SIM} strokeWidth={1.4} />}

      {/* brake */}
      <Label y={ROWS.brake[0] + 10}>BRAKE</Label>
      {bands(d, real.brake_on, x).map(([a, b], k) => <rect key={`rb${k}`} x={a} y={ROWS.brake[0]} width={Math.max(1, b - a)} height={6} fill={REAL} opacity={0.5} />)}
      {sim && bands(d, sim.brake_on, x).map(([a, b], k) => <rect key={`sb${k}`} x={a} y={ROWS.brake[0] + 8} width={Math.max(1, b - a)} height={6} fill={SIM} />)}

      {/* DRS */}
      <Label y={ROWS.drs[0] + 10}>DRS</Label>
      {bands(d, real.drs_open, x).map(([a, b], k) => <rect key={`d${k}`} x={a} y={ROWS.drs[0]} width={Math.max(1, b - a)} height={10} fill={GAIN} opacity={0.6} />)}

      {/* axis */}
      {[0, 0.5, 1].map((f) => <text key={f} x={x(f * L)} y={H - 4} textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"} fill="#5b6472" fontSize={9} fontFamily={MONO}>{Math.round(f * L).toLocaleString()} m</text>)}

      {/* replay marker */}
      {markerDist != null && cursor === null && <path d={`M${x(markerDist)} ${Y0} V${Y1}`} stroke="#e6e8ec" strokeOpacity={0.7} />}

      {/* crosshair + tooltip */}
      {i !== null && (
        <g>
          {cursor !== null && <path d={`M${xs[i]} ${Y0} V${Y1}`} stroke="#8b93a3" strokeDasharray="2 3" />}
          <circle cx={xs[i]} cy={ys(real.speed_kph[i])} r={3.5} fill={REAL} stroke="#0f1216" strokeWidth={2} />
          {sim && <circle cx={xs[i]} cy={ys(sim.speed_kph[i])} r={3.5} fill={SIM} stroke="#0f1216" strokeWidth={2} />}
          <g transform={`translate(${Math.min(xs[i] + 10, W - 176)}, ${Y0})`}>
            <rect width={166} height={sim ? 64 : 36} rx={4} fill="#151a20" stroke="#2b333e" />
            <text x={8} y={14} fill="#8b93a3" fontSize={10} fontFamily={MONO}>{Math.round(d[i]).toLocaleString()} m{hover !== null ? ` · seg ${hover}` : ""}{real.interpolated[i] ? " · interp." : ""}</text>
            <text x={8} y={29} fill={REAL} fontSize={11} fontFamily={MONO}>real  {real.speed_kph[i].toFixed(0)} km/h · {real.throttle_pct[i].toFixed(0)}%{real.brake_on[i] ? " · brake" : ""}</text>
            {sim && <text x={8} y={44} fill={SIM} fontSize={11} fontFamily={MONO}>sim   {sim.speed_kph[i].toFixed(0)} km/h · {sim.throttle_pct[i].toFixed(0)}%{sim.brake_on[i] ? " · brake" : ""}</text>}
            {sim && <text x={8} y={58} fill={dt[i] < 0 ? GAIN : dt[i] > 0 ? LOSS : "#8b93a3"} fontSize={10} fontFamily={MONO}>Δv {dv[i] >= 0 ? "+" : ""}{dv[i].toFixed(1)} km/h · Δt {dt[i] >= 0 ? "+" : ""}{dt[i].toFixed(3)} s</text>}
          </g>
        </g>
      )}
    </svg>
  );
}
