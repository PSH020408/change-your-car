"use client";
import { useLayoutEffect, useRef } from "react";
import type { SegmentDelta, SegmentInfo, TrackMap as TrackMapT } from "@/lib/types";
import { fmtDelta, kindLabel } from "@/lib/api";

/** A car marker placed at `dist` metres along the circuit path (path length is in metres via pathLength). */
function CarMarker({ pathRef, dist, lapLength, color, label }: {
  pathRef: React.RefObject<SVGPathElement | null>; dist: number; lapLength: number; color: string; label: string;
}) {
  const g = useRef<SVGGElement>(null);
  useLayoutEffect(() => {
    const p = pathRef.current, el = g.current;
    if (!p || !el) return;
    const total = p.getTotalLength();
    const frac = ((dist % lapLength) + lapLength) % lapLength / lapLength;
    const pt = p.getPointAtLength(frac * total);
    el.setAttribute("transform", `translate(${pt.x.toFixed(2)} ${pt.y.toFixed(2)})`);
  }, [pathRef, dist, lapLength]);
  return (
    <g ref={g}>
      <circle r={9} fill={color} fillOpacity={0.25} />
      <circle r={5} fill={color} stroke="#0f1216" strokeWidth={1.5} />
      <title>{label}</title>
    </g>
  );
}

/**
 * Circuit SVG from silver, coloured per segment by its total delta.
 * `pathLength` is set to the lap length in metres, so dash arrays are in
 * metres and each segment is a dash of [length, everything else] offset to
 * its start — no geometry maths on the client. Optional replay markers show
 * where the real and the simulated car are at the same moment.
 */
export function TrackMap({ track, segments, deltas, hover, onHover, markers }: {
  track: TrackMapT; segments: SegmentInfo[]; deltas?: SegmentDelta[];
  hover: number | null; onHover: (i: number | null) => void;
  markers?: { realDist: number; simDist: number } | null;
}) {
  const L = track.lap_length_m;
  const base = useRef<SVGPathElement>(null);
  const byIdx = new Map((deltas ?? []).map((d) => [d.index, d]));
  const maxAbs = Math.max(0.01, ...(deltas ?? []).map((d) => Math.abs(d.total_s)));
  return (
    <svg viewBox={track.view_box} className="w-full h-full" fill="none" strokeLinecap="round" strokeLinejoin="round" aria-label="track map">
      <path ref={base} d={track.path} stroke="#2b333e" strokeWidth={14} />
      {segments.map((s) => {
        const d = byIdx.get(s.index);
        const v = d?.total_s ?? 0;
        const on = Math.abs(v) > 0.002;
        const color = !on ? "#3a4350" : v < 0 ? "#0ca30c" : "#ec835a";
        const opacity = !on ? 1 : 0.45 + 0.55 * Math.min(1, Math.abs(v) / maxAbs);
        const len = s.end_m >= s.start_m ? s.end_m - s.start_m : L - s.start_m + s.end_m;
        return (
          <path key={s.index} d={track.path} pathLength={L} stroke={color} strokeOpacity={hover === null || hover === s.index ? opacity : opacity * 0.35}
            strokeWidth={hover === s.index ? 9 : 6} strokeDasharray={`${len} ${L}`} strokeDashoffset={-s.start_m}
            onMouseEnter={() => onHover(s.index)} onMouseLeave={() => onHover(null)} style={{ cursor: "pointer" }}>
            <title>{`Segment ${s.index} · ${kindLabel(s.kind)} · S${s.sector ?? "?"}${d ? ` · ${fmtDelta(d.total_s)} s` : ""}`}</title>
          </path>
        );
      })}
      {/* start / finish */}
      <path d={track.path} pathLength={L} stroke="#e6e8ec" strokeWidth={16} strokeDasharray={`4 ${L}`} strokeDashoffset={0} />
      {markers && (
        <>
          <CarMarker pathRef={base} dist={markers.realDist} lapLength={L} color="#c3c2b7" label="real lap" />
          <CarMarker pathRef={base} dist={markers.simDist} lapLength={L} color="#3987e5" label="simulated lap" />
        </>
      )}
    </svg>
  );
}
