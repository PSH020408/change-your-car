import type { CarSetup, Environment, SimulationResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function simulate(body: {
  season: number;
  event: string;
  session: string;
  driver: string;
  chassis: string;
  setup: CarSetup;
  environment: Environment;
}): Promise<SimulationResponse> {
  const res = await fetch(`${BASE}/api/simulate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`simulate failed: ${res.status}`);
  return res.json();
}
