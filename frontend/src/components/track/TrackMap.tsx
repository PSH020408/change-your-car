"use client";
import type { SegmentDelta, SegmentInfo, TrackMap as TrackMapT } from "@/lib/types";
import { fmtDelta, kindLabel } from "@/lib/api";

/**
 * Circuit SVG from silver, coloured per segment by its total delta.
 * `pathLength` is set to the lap length in metres, so dash arrays are in
 * metres and each segment is a dash of [length, everything else] offset to
 * its start — no geometry maths on the client.
 */
export function TrackMap({ track, segments, deltas, hover, onHover }: {
  track: TrackMapT; segments: SegmentInfo[]; deltas?: SegmentDelta[];
  hover: number | null; onHover: (i: number | null) => void;
}) {
  const L = track.lap_length_m;
  const byIdx = new Map((deltas ?? []).map((d) => [d.index, d]));
  const maxAbs = Math.max(0.01, ...(deltas ?? []).map((d) => Math.abs(d.total_s)));
  return (
    <svg viewBox={track.view_box} className="w-full h-full" fill="none" strokeLinecap="round" strokeLinejoin="round" aria-label="track map">
      <path d={track.path} stroke="#2b333e" strokeWidth={14} />
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
    </svg>
  );
}
