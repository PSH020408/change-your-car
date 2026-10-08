"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmtDelta, fmtRace } from "@/lib/api";
import { useHud } from "@/lib/store";
import type { BaselineResponse, Compound, Stint, StrategyResponse, StrategyScore } from "@/lib/types";

const COLOR: Record<string, string> = { SOFT: "#ef4444", MEDIUM: "#eab308", HARD: "#e5e7eb" };
const SHORT: Record<string, string> = { SOFT: "S", MEDIUM: "M", HARD: "H" };

function parseSeq(seq: string): Stint[] {
  return seq.split("→").map((p) => p.trim().split(/\s+/)).filter((x) => x.length === 2)
    .map(([c, n]) => ({ compound: c as Compound, laps: Number(n) }));
}
function fitToRace(st: Stint[], laps: number): Stint[] {
  if (!st.length) return st;
  const total = st.reduce((a, s) => a + s.laps, 0);
  const last = st[st.length - 1];
  return [...st.slice(0, -1), { ...last, laps: Math.max(1, last.laps + (laps - total)) }];
}

/** Race strategy, scored from one real race lap: the cards are strategies teams really ran. */
export function StrategyPanel({ baseline }: { baseline: BaselineResponse | undefined }) {
  const { ref } = useHud();
  const isRace = ref.session === "R" && !!baseline?.race_laps && !!baseline?.pit_loss_s;
  const raceLaps = baseline?.race_laps ?? 0;
  const cards = useMemo(() => (baseline?.strategies ?? []).slice(0, 4), [baseline]);
  const [stints, setStints] = useState<Stint[]>([]);
  const [res, setRes] = useState<StrategyResponse | undefined>();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | undefined>();
  const abort = useRef<AbortController | null>(null);

  // default plan: the winner's strategy, else the most common one
  useEffect(() => {
    if (!isRace || !cards.length) { setStints([]); setRes(undefined); return; }
    const pick = cards.find((c) => c.winner) ?? cards[0];
    setStints(fitToRace(parseSeq(pick.sequence), raceLaps));
  }, [isRace, cards, raceLaps, ref.driver]);

  useEffect(() => {
    if (!isRace || !stints.length) return;
    abort.current?.abort();
    const ac = new AbortController(); abort.current = ac;
    setBusy(true); setErr(undefined);
    const t = setTimeout(() => {
      api.strategy({ baseline: ref, stints }, ac.signal)
        .then((r) => { setRes(r); setBusy(false); })
        .catch((e) => { if (e.name !== "AbortError") { setErr(String(e.message ?? e)); setBusy(false); } });
    }, 200);
    return () => { clearTimeout(t); ac.abort(); };
  }, [isRace, stints, ref]);

  if (!isRace) return null;

  const envelope = baseline?.tyre_envelope ?? {};
  const bump = (i: number, d: number) => {
    const next = stints.map((s) => ({ ...s }));
    const last = next.length - 1;
    if (i === last) return;                                  // the last stint absorbs the change
    const cap = envelope[next[i].compound]?.max_laps ?? 99;
    next[i].laps = Math.min(cap, Math.max(1, next[i].laps + d));
    next[last].laps = Math.max(1, raceLaps - next.slice(0, -1).reduce((a, s) => a + s.laps, 0));
    setStints(next);
  };
  const cycle = (i: number) => {
    const order: Compound[] = ["SOFT", "MEDIUM", "HARD"];
    const avail = order.filter((c) => envelope[c]?.max_by_session?.R !== undefined);
    const opts = avail.length ? avail : order;
    const next = stints.map((s) => ({ ...s }));
    next[i].compound = opts[(opts.indexOf(next[i].compound) + 1) % opts.length];
    setStints(next);
  };
  const best = res?.cards.find((c) => !c.refused.length);
  const yours = res?.yours;

  return (
    <section className="card p-3 flex flex-col gap-2 shrink-0">
      <div className="flex justify-between items-baseline">
        <span className="label">Race strategy</span>
        <span className="text-[10px] font-mono text-hud-dim">{raceLaps} laps · pit loss {baseline?.pit_loss_s?.toFixed(1)} s measured{res ? ` (${res.pit_loss_n} stops)` : ""}</span>
      </div>

      {/* your plan: click a compound to change it, +/- to move a stop; the last stint absorbs */}
      <div className="flex items-center gap-1 flex-wrap">
        {stints.map((s, i) => (
          <div key={i} className="flex items-center gap-0.5 bg-hud-inset rounded px-1 py-0.5">
            <button onClick={() => cycle(i)} className="font-mono text-[11px] font-semibold px-1 rounded" style={{ color: COLOR[s.compound] }} title="change compound">{SHORT[s.compound]}</button>
            {i < stints.length - 1 ? (
              <>
                <button onClick={() => bump(i, -1)} className="chip px-1 py-0 text-[10px]">−</button>
                <span className="font-mono text-[11px] w-6 text-center tabular-nums">{s.laps}</span>
                <button onClick={() => bump(i, +1)} className="chip px-1 py-0 text-[10px]">+</button>
              </>
            ) : <span className="font-mono text-[11px] w-6 text-center tabular-nums text-hud-soft">{s.laps}</span>}
            {i < stints.length - 1 && <span className="text-hud-dim text-[10px] ml-0.5">→</span>}
          </div>
        ))}
        <button onClick={() => stints.length < 4 && setStints(fitToRace([...stints, { compound: stints[stints.length - 1].compound, laps: 10 }], raceLaps))}
          className="chip px-1.5 py-0 text-[10px]" title="add a stop">+ stop</button>
        {stints.length > 1 && <button onClick={() => setStints(fitToRace(stints.slice(0, -1), raceLaps))} className="chip px-1.5 py-0 text-[10px]" title="remove the last stint">− stop</button>}
      </div>

      {/* headline */}
      <div className="grid grid-cols-3 gap-2">
        <Stat k="race time" v={yours ? fmtRace(yours.race_s) : "—"} sub={yours ? `${yours.stops} stop${yours.stops === 1 ? "" : "s"} · ${fmtRace(yours.driving_s)} driving` : busy ? "scoring…" : ""} />
        <Stat k="vs best card" v={res?.delta_to_best_s != null ? `${fmtDelta(res.delta_to_best_s, 1)} s` : "—"}
          tone={res?.delta_to_best_s != null ? (res.delta_to_best_s > 0.05 ? "text-status-warn" : "text-status-gain") : undefined}
          sub={best ? best.label : ""} />
        <Stat k="unit lap" v={res ? `L${res.unit_lap.lap_number} ${res.unit_lap.compound[0]}${res.unit_lap.tyre_life}` : "—"}
          sub={res ? `${res.unit_lap.lap_time_s.toFixed(3)} s · fuel ${res.unit_lap.fuel_slope_s_per_kg.toFixed(3)} s/kg (${res.unit_lap.fuel_slope_source})` : ""} />
      </div>
      {yours?.refused.length ? (
        <div className="text-[10px] font-mono text-status-warn leading-snug">outside the data — not scored: {yours.refused.join("; ")}</div>
      ) : null}
      {err && <div className="text-[10px] font-mono text-status-warn">{err}</div>}

      {/* lap chart */}
      {yours && !yours.refused.length && <LapChart yours={yours} best={best} />}

      {/* cards: the strategies teams really ran, scored by the same model */}
      {res && (
        <div className="text-[10px] font-mono leading-snug border-t border-hud-line pt-2 flex flex-col gap-0.5">
          <span className="label">how the race was run · same model</span>
          {res.cards.map((c) => (
            <button key={c.label} onClick={() => setStints(fitToRace(c.stints, raceLaps))}
              className="flex justify-between gap-2 text-left hover:text-hud-text">
              <span className={c.winner ? "text-hud-soft" : "text-hud-dim"}>{c.label}{c.winner ? " · winner" : ""} ×{c.count}</span>
              <span className="shrink-0 tabular-nums">{c.refused.length ? "outside data" : `${fmtRace(c.race_s)}${best && c !== best ? `  ${fmtDelta(c.race_s - best.race_s, 1)}` : ""}`}</span>
            </button>
          ))}
          <span className="text-hud-dim mt-1">ignored: {res.ignored.join(" · ")}. Back-tested on 677 drivers / 40 green-flag races: race total within 0.39 % (median).</span>
        </div>
      )}
    </section>
  );
}

function Stat({ k, v, sub, tone }: { k: string; v: string; sub?: string; tone?: string }) {
  return (
    <div className="bg-hud-inset rounded px-2 py-1.5 min-w-0">
      <div className="label">{k}</div>
      <div className={`text-[14px] font-mono tabular-nums ${tone ?? ""}`}>{v}</div>
      {sub ? <div className="text-[10px] font-mono text-hud-dim truncate" title={sub}>{sub}</div> : null}
    </div>
  );
}

/** Predicted lap time per lap (compound-coloured, pit laps marked) and the running gap to the best card. */
function LapChart({ yours, best }: { yours: StrategyScore; best?: StrategyScore }) {
  const W = 420, H = 120, padL = 34, padR = 6, padT = 6, padB = 16;
  const laps = yours.laps;
  const n = laps.length;
  const ts = laps.map((l) => l.predicted_s);
  const lo = Math.min(...ts), hi = Math.max(...ts);
  const x = (i: number) => padL + (i / Math.max(n - 1, 1)) * (W - padL - padR);
  const y = (t: number) => padT + (1 - (t - lo) / Math.max(hi - lo, 0.1)) * (H - padT - padB);
  // running gap to the best card, pit loss included at the in-lap
  let cum = 0; const gaps: number[] = [];
  const bl = best?.laps ?? [];
  const pl = best ? yours.pit_s / Math.max(yours.stops, 1) : 0;
  for (let i = 0; i < n; i++) {
    const mine = ts[i] + (laps[i].pit_in ? pl : 0);
    const theirs = (bl[i]?.predicted_s ?? ts[i]) + (bl[i]?.pit_in ? pl : 0);
    cum += mine - theirs; gaps.push(cum);
  }
  const gmax = Math.max(1, ...gaps.map((g) => Math.abs(g)));
  const gy = (g: number) => padT + (0.5 - g / (2 * gmax)) * (H - padT - padB);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[120px]">
      <text x={padL - 4} y={y(hi) + 4} textAnchor="end" className="fill-hud-dim" fontSize="8" fontFamily="monospace">{hi.toFixed(1)}</text>
      <text x={padL - 4} y={y(lo) + 4} textAnchor="end" className="fill-hud-dim" fontSize="8" fontFamily="monospace">{lo.toFixed(1)}</text>
      <line x1={padL} x2={W - padR} y1={gy(0)} y2={gy(0)} stroke="currentColor" className="text-hud-line" strokeDasharray="2 3" />
      {best && best !== yours && (
        <path d={gaps.map((g, i) => `${i ? "L" : "M"}${x(i)},${gy(g)}`).join(" ")} fill="none" stroke="#60a5fa" strokeWidth="1" opacity="0.8" />
      )}
      {laps.map((l, i) => i > 0 && (
        <line key={i} x1={x(i - 1)} y1={y(ts[i - 1])} x2={x(i)} y2={y(ts[i])} stroke={COLOR[l.compound] ?? "#999"} strokeWidth="1.5" />
      ))}
      {laps.map((l, i) => l.pit_in && <line key={`p${i}`} x1={x(i)} x2={x(i)} y1={padT} y2={H - padB} stroke="#60a5fa" strokeWidth="1" strokeDasharray="1 2" />)}
      <text x={padL} y={H - 4} className="fill-hud-dim" fontSize="8" fontFamily="monospace">lap 1</text>
      <text x={W - padR} y={H - 4} textAnchor="end" className="fill-hud-dim" fontSize="8" fontFamily="monospace">lap {n} · line = predicted lap time by compound · blue = running gap to best card (±{gmax.toFixed(0)} s)</text>
    </svg>
  );
}
