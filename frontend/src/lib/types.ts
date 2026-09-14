/**
 * Mirrors backend/app/schemas/domain.py — keep the two in lockstep.
 * Phase 4 task: generate this file from the OpenAPI schema instead.
 */
export type Weather = "dry" | "intermediate" | "wet";
export type Compound = "soft" | "medium" | "hard" | "intermediate" | "wet";
export type SegmentKind =
  | "straight"
  | "low_speed_corner"
  | "medium_speed_corner"
  | "high_speed_corner"
  | "braking";

export interface CarSetup {
  front_wing: number;
  rear_wing: number;
  ride_height_front_mm: number;
  ride_height_rear_mm: number;
  suspension_stiffness_front: number;
  suspension_stiffness_rear: number;
  camber_front_deg: number;
  camber_rear_deg: number;
  toe_front_deg: number;
  toe_rear_deg: number;
  brake_bias_pct: number;
  diff_on_throttle: number;
}

export interface Environment {
  track_temp_c: number;
  air_temp_c: number;
  weather: Weather;
  track_evolution: number;
  wind_kph: number;
  compound: Compound;
  fuel_kg: number;
}

export interface TelemetryTrace {
  distance_m: number[];
  speed_kph: number[];
  throttle_pct: number[];
  brake_pct: number[];
  gear: number[];
  drs: number[];
}

export interface SegmentDelta {
  index: number;
  kind: SegmentKind;
  start_m: number;
  end_m: number;
  sector: 1 | 2 | 3;
  baseline_time_s: number;
  simulated_time_s: number;
  delta_s: number;
  confidence: number;
}

export interface PhysicsDelta {
  downforce_delta_pct: number;
  drag_delta_pct: number;
  mechanical_grip_delta_pct: number;
  tyre_thermal_grip_delta_pct: number;
  balance_index: number;
}

export interface EngineerNote {
  severity: "info" | "warning" | "critical";
  channel: "balance" | "tyre" | "aero" | "brakes" | "traction";
  message: string;
  suggestion?: string;
}

export interface SimulationResponse {
  lap_delta_s: number;
  sector_deltas_s: number[];
  segments: SegmentDelta[];
  baseline: TelemetryTrace;
  simulated: TelemetryTrace;
  physics: PhysicsDelta;
  engineer_log: EngineerNote[];
  model_version: string;
  computed_ms: number;
}
