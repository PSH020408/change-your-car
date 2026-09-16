"use client";
import useSWR from "swr";
import { api, fmtLap } from "@/lib/api";
import { useHud } from "@/lib/store";
import type { BaselineResponse } from "@/lib/types";

const SESSION_LABEL: Record<string, string> = { Q: "Qualifying", SQ: "Sprint Quali", S: "Sprint", R: "Race" };

function Select<T extends string | number>({ label, value, options, onChange, disabled }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: string) => void; disabled?: boolean;
}) {
  return (
    <label className="flex flex-col gap-0.5 min-w-0">
      <span className="label">{label}</span>
      <select value={String(value)} disabled={disabled || options.length === 0} onChange={(e) => onChange(e.target.value)}
        className="h-7 bg-hud-inset border border-hud-line2 rounded px-1.5 text-[12px] text-hud-text font-mono focus:outline-none focus:border-series-sim disabled:opacity-50 max-w-[200px] truncate">
        {options.map((o) => <option key={String(o.value)} value={String(o.value)}>{o.label}</option>)}
      </select>
    </label>
  );
}

/**
 * Docking bar: pick season → event → session → driver → lap, and read the
 * baseline lap's identity. Changing the reference resets setup/conditions
 * (store.setRef), because the sliders are relative to the chosen lap.
 */
export function TopBar({ baseline, error, busy }: { baseline?: BaselineResponse; error?: Error; busy: boolean }) {
  const { ref, setRef, reset } = useHud();
  const seasons = useSWR("seasons", api.seasons);
  const events = useSWR(["events", ref.season], () => api.events(ref.season));
  const drivers = useSWR(["drivers", ref.season, ref.event, ref.session], () => api.drivers(ref.season, ref.event, ref.session));

  const ev = events.data?.find((e) => e.event === ref.event);
  const sessions = ev?.sessions ?? [ref.session];
  const lap = baseline?.lap;

  const onSeason = (v: string) => {
    const season = Number(v);
    setRef({ season });
    // event list changes with the season; snap to the first one once it loads
    api.events(season).then((evs) => {
      if (evs.length && !evs.some((e) => e.event === ref.event)) setRef({ event: evs[0].event, session: (evs[0].sessions[0] ?? "Q") as typeof ref.session });
    });
  };
  const onEvent = (v: string) => {
    const e = events.data?.find((x) => x.event === v);
    setRef({ event: v, session: (e?.sessions.includes(ref.session) ? ref.session : (e?.sessions[0] ?? "Q")) as typeof ref.session, lap: "representative" });
  };

  return (
    <header className="h-[60px] px-4 flex items-center gap-4 bg-hud-bar border-b border-hud-line">
      <div className="flex items-baseline gap-2 shrink-0">
        <span className="text-[15px] font-semibold tracking-wide">F1 VIRTUAL SIM</span>
        <span className="text-[10px] font-mono text-hud-dim">2022–2025 · ground-effect era</span>
      </div>

      <div className="flex items-end gap-3 shrink-0">
        <Select label="season" value={ref.season} onChange={onSeason}
          options={(seasons.data ?? [ref.season]).map((s) => ({ value: s, label: String(s) }))} />
        <Select label="event" value={ref.event} onChange={onEvent}
          options={(events.data ?? [{ event: ref.event, event_name: ref.event }]).map((e) => ({ value: e.event, label: e.event_name }))} />
        <Select label="session" value={ref.session} onChange={(v) => setRef({ session: v as typeof ref.session, lap: "representative" })}
          options={sessions.map((s) => ({ value: s, label: SESSION_LABEL[s] ?? s }))} />
        <Select label="driver" value={ref.driver} onChange={(v) => setRef({ driver: v, lap: "representative" })}
          options={(drivers.data ?? [{ driver: ref.driver, team: null, chassis: null, power_unit: null, laps: 0, representative_lap_time_s: null }])
            .map((d) => ({ value: d.driver, label: d.chassis ? `${d.driver} · ${d.chassis}` : d.driver }))} />
        <Select label="lap" value={ref.lap} onChange={(v) => setRef({ lap: v })}
          options={[
            { value: "representative", label: "representative" },
            { value: "fastest", label: "fastest" },
            ...(baseline?.available_laps ?? []).map((l) => ({
              value: l.lap_uid,
              label: `L${l.lap_number ?? "?"} ${l.lap_time_s ? fmtLap(l.lap_time_s) : ""} ${l.compound ?? ""}${l.tyre_life != null ? ` ${l.tyre_life}L` : ""}`.trim(),
            })),
          ]} />
      </div>

      <div className="flex-1 min-w-0 pl-4 border-l border-hud-line">
        <div className="label">baseline lap</div>
        {lap ? (
          <div className="text-[11.5px] font-mono text-hud-soft whitespace-nowrap overflow-hidden text-ellipsis" title={lap.lap_uid}>
            <span className="text-hud-text">{fmtLap(lap.lap_time_s)}</span>
            <span className="text-hud-dim"> · </span>{lap.team ?? "—"}<span className="text-hud-dim"> · </span>{lap.chassis ?? "—"}
            <span className="text-hud-dim"> · </span>{lap.compound ?? "?"} {lap.tyre_life ?? "?"}L
            <span className="text-hud-dim"> · </span>{lap.track_temp_c?.toFixed(0) ?? "?"}/{lap.air_temp_c?.toFixed(0) ?? "?"} °C
            <span className="text-hud-dim"> · </span>L{lap.lap_number ?? "?"} {lap.effort_class ?? ""} {lap.telemetry_quality ?? ""}
          </div>
        ) : <div className="text-[11.5px] font-mono text-hud-dim">—</div>}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <span className={`w-2 h-2 rounded-full ${error ? "bg-status-critical" : busy ? "bg-status-warn animate-pulse" : "bg-status-gain"}`} title={error ? error.message : busy ? "computing" : "live"} />
        <span className="text-[10px] font-mono text-hud-muted w-[64px]">{error ? "API ERROR" : busy ? "COMPUTING" : "LIVE"}</span>
        <button onClick={reset} className="chip hover:border-hud-muted">reset setup</button>
      </div>
    </header>
  );
}
