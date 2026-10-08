import type { BaselineRef, BaselineResponse, DriverInfo, EventInfo, SimulationRequest, SimulationResponse, StrategyRequest, StrategyResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status} ${await res.text()}`);
  return res.json();
}

export const api = {
  seasons: () => get<number[]>("/api/meta/seasons"),
  events: (season: number) => get<EventInfo[]>(`/api/meta/events/${season}`),
  drivers: (season: number, event: string, session: string) =>
    get<DriverInfo[]>(`/api/meta/drivers/${season}/${event}/${session}`),
  baseline: (ref: BaselineRef) =>
    get<BaselineResponse>(`/api/baseline?season=${ref.season}&event=${ref.event}&session=${ref.session}&driver=${ref.driver}&lap=${ref.lap}`),
  simulate: async (body: SimulationRequest, signal?: AbortSignal): Promise<SimulationResponse> => {
    const res = await fetch(`${BASE}/api/simulate`, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal,
    });
    if (!res.ok) throw new Error(`simulate: HTTP ${res.status} ${await res.text()}`);
    return res.json();
  },
  strategy: async (body: StrategyRequest, signal?: AbortSignal): Promise<StrategyResponse> => {
    const res = await fetch(`${BASE}/api/strategy`, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal,
    });
    if (!res.ok) throw new Error(`strategy: HTTP ${res.status} ${await res.text()}`);
    return res.json();
  },
};
export const fmtRace = (s: number) => {
  const h = Math.floor(s / 3600), m = Math.floor((s - 3600 * h) / 60), sec = s - 3600 * h - 60 * m;
  return `${h}:${String(m).padStart(2, "0")}:${sec.toFixed(1).padStart(4, "0")}`;
};

export const fmtDelta = (s: number, digits = 3) => `${s > 0 ? "+" : s < 0 ? "−" : ""}${Math.abs(s).toFixed(digits)}`;
export const fmtLap = (s: number) => {
  const m = Math.floor(s / 60);
  return `${m}:${(s - 60 * m).toFixed(3).padStart(6, "0")}`;
};
export const kindLabel = (k: string) => k.replace("_speed_corner", " corner").replace("_", " ");
