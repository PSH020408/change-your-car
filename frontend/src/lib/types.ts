/**
 * Mirrors backend/app/schemas/domain.py — keep the two in lockstep.
 * Anything the HUD draws is here; anything not here the HUD cannot know.
 */
export type Weather = "dry" | "inter" | "wet";
export type Compound = "SOFT" | "MEDIUM" | "HARD" | "INTERMEDIATE" | "WET";
export type SegmentKind = "straight" | "kink" | "low_speed_corner" | "medium_speed_corner" | "high_speed_corner";
export type Grade = "A" | "B" | "C";

export interface CarSetup {
  front_wing: number;
  rear_wing: number;
  ride_height: number;
  suspension: number;
  suspension_split: number;
  fuel_kg: number | null;
}

export interface Environment {
  track_temp_c: number | null;
  air_temp_c: number | null;
  weather: Weather;
  compound: Compound | null;
  tyre_life: number | null;
}

export interface BaselineRef {
  season: number;
  event: string;
  session: "Q" | "R" | "S" | "SQ";
  driver: string;
  lap: string;
}

export interface SimulationRequest {
  baseline: BaselineRef;
  setup: CarSetup;
  environment: Environment;
}

export interface SegmentInfo {
  index: number; kind: SegmentKind; start_m: number; end_m: number; length_m: number;
  sector: number | null; min_radius_m: number | null; direction: string | null;
}

export interface SegmentDelta {
  index: number; kind: SegmentKind; sector: number | null; baseline_time_s: number;
  ml_s: number; ml_lo_s: number; ml_hi_s: number; level2_s: number;
  physics_s: number; physics_lo_s: number; physics_hi_s: number;
  total_s: number; total_lo_s: number; total_hi_s: number;
  achieved_s: number; refused_s: number;
}

export interface TelemetryTrace {
  distance_m: number[]; time_s: number[]; speed_kph: number[]; throttle_pct: number[];
  brake_on: boolean[]; gear: number[]; drs_open: boolean[]; interpolated: boolean[];
}

export interface LapMeta {
  lap_uid: string; driver: string; team: string | null; chassis: string | null; power_unit: string | null;
  season: number; event: string; event_name: string; session: string; circuit: string | null;
  lap_number: number | null; lap_time_s: number; compound: string | null; tyre_life: number | null;
  fresh_tyre: boolean | null; track_temp_c: number | null; air_temp_c: number | null;
  telemetry_quality: string | null; effort_class: string | null; gap_ahead_s: number | null;
  sector_times_s: (number | null)[];
}

export interface TrackMap {
  view_box: string; path: string; sector_boundaries_m: number[]; lap_length_m: number;
  published_turns: number | null; measured_turns: number;
}

export interface AvailableLap {
  lap_uid: string; lap_number: number | null; lap_time_s: number | null; compound: string | null;
  tyre_life: number | null; effort_class: string | null; telemetry_quality: string | null; gap_ahead_s: number | null;
}

export interface TyreEnvelope { max_laps: number; median_stint: number; n_stints: number; max_by_session: Record<string, number>; }
export interface RaceStrategy { sequence: string; drivers: string[]; count: number; winner: boolean; }

export interface BaselineResponse {
  lap: LapMeta; trace: TelemetryTrace; segments: SegmentInfo[]; track: TrackMap;
  available_laps: AvailableLap[]; integration_note: string;
  tyre_envelope: Record<string, TyreEnvelope>; strategies: RaceStrategy[];
  compounds?: Record<string, string>;
  race_laps?: number | null; pit_loss_s?: number | null;
}

export interface PhysicsState {
  downforce_pct: number; drag_pct: number; mech_grip_pct: number; grip_multiplier: number;
  thermal_grip_pct: number; fuel_delta_kg: number; balance_index: number; warning: string | null;
}

export interface GradeRow { control: string; grade: Grade; note: string; }

export interface EngineerNote {
  severity: "info" | "warning" | "critical";
  channel: "balance" | "tyre" | "aero" | "fuel" | "weather" | "model" | "physics" | "sectors";
  message: string; suggestion: string | null;
}

export interface LapSummary {
  baseline_lap_time_s: number; simulated_lap_time_s: number; delta_s: number; delta_lo_s: number; delta_hi_s: number;
  ml_s: number; level2_s: number; level2_applied: boolean; physics_s: number; refused_s: number; sector_deltas_s: number[];
}

export interface QssFit {
  mu: number; cl_a: number; cd_a: number; speed_rms_kph: number; lap_time_err_s: number;
  grip_multiplier_ml: number; solves: number;
}

export interface SimulationResponse {
  lap: LapSummary; segments: SegmentDelta[]; baseline: TelemetryTrace; simulated: TelemetryTrace;
  physics: PhysicsState; grades: GradeRow[]; engineer_log: EngineerNote[];
  model_version: string; physics_version: string; computed_ms: number;
  engine_mode?: "qss" | "table"; qss_fit?: QssFit | null;
}

export interface EventInfo { event: string; event_name: string; circuit: string | null; sessions: string[]; }
export interface DriverInfo { driver: string; team: string | null; chassis: string | null; power_unit: string | null; laps: number; representative_lap_time_s: number | null; }

export const DEFAULT_SETUP: CarSetup = { front_wing: 0.5, rear_wing: 0.5, ride_height: 0.5, suspension: 0.5, suspension_split: 0.5, fuel_kg: null };
export const DEFAULT_ENV: Environment = { track_temp_c: null, air_temp_c: null, weather: "dry", compound: null, tyre_life: null };

// ---------------------------------------------------------------- P11 strategy
export interface Stint { compound: Compound; laps: number; }
export interface StrategyRequest { baseline: BaselineRef; stints: Stint[]; }
export interface StrategyLap {
  lap: number; compound: string; tyre_life: number; fuel_kg: number; predicted_s: number;
  fuel_delta_s: number; tyre_delta_s: number; pit_in: boolean;
}
export interface StrategyScore {
  label: string; stints: Stint[]; stops: number; driving_s: number; pit_s: number; race_s: number;
  laps: StrategyLap[]; refused: string[]; count?: number | null; winner?: boolean | null; drivers: string[];
}
export interface StrategyUnitLap {
  lap_uid: string; lap_number: number; lap_time_s: number; compound: string; tyre_life: number;
  fuel_kg: number; fuel_slope_s_per_kg: number; fuel_slope_source: "engine" | "table";
}
export interface StrategyResponse {
  race_laps: number; pit_loss_s: number; pit_loss_n: number; green_share: number; unit_lap: StrategyUnitLap;
  yours: StrategyScore; cards: StrategyScore[]; delta_to_best_s: number | null; ignored: string[];
}
