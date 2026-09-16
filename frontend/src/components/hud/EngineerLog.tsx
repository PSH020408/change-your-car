"use client";
import type { EngineerNote } from "@/lib/types";

const TONE = { info: "#8b93a3", warning: "#e0a54a", critical: "#d03b3b" } as const;

function Icon({ severity }: { severity: EngineerNote["severity"] }) {
  const c = TONE[severity];
  return (
    <svg viewBox="0 0 14 14" width={14} height={14} fill="none" stroke={c} strokeWidth={1.4} className="shrink-0 mt-[2px]">
      {severity === "info" && <><circle cx={7} cy={7} r={5.5} /><path d="M7 6.2v4M7 4v.2" strokeLinecap="round" /></>}
      {severity === "warning" && <><path d="M7 1.8 12.6 11.8H1.4Z" strokeLinejoin="round" /><path d="M7 5.5v3M7 10.3v.2" strokeLinecap="round" /></>}
      {severity === "critical" && <><rect x={2} y={2} width={10} height={10} rx={1.5} /><path d="M4.8 4.8l4.4 4.4M9.2 4.8 4.8 9.2" strokeLinecap="round" /></>}
    </svg>
  );
}

/** Rule-based notes from the backend (English by contract). Newest state, not a history. */
export function EngineerLog({ notes, pending }: { notes?: EngineerNote[]; pending: boolean }) {
  return (
    <section className="card p-3 flex flex-col gap-2 min-h-[120px]">
      <div className="flex items-center justify-between">
        <span className="label">engineer log</span>
        <span className="text-[10px] font-mono text-hud-dim">{notes ? `${notes.length} notes` : pending ? "…" : "idle"}</span>
      </div>
      {!notes && <div className="text-[11px] text-hud-dim">Notes appear once a simulation has run: balance warnings, tyre window, aero efficiency, fuel, weather, model confidence.</div>}
      {notes && notes.length === 0 && <div className="text-[11px] text-hud-dim">Nothing to flag — setup within the validated envelope.</div>}
      {notes && notes.map((n, i) => (
        <div key={i} className="flex gap-2 text-[11.5px] leading-snug">
          <Icon severity={n.severity} />
          <div className="min-w-0">
            <span className="text-[9px] font-mono uppercase tracking-wider mr-1.5" style={{ color: TONE[n.severity] }}>{n.channel}</span>
            <span className="text-hud-soft">{n.message}</span>
            {n.suggestion && <div className="text-hud-muted mt-0.5">→ {n.suggestion}</div>}
          </div>
        </div>
      ))}
    </section>
  );
}
