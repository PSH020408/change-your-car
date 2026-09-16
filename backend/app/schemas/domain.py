"""Wire contract shared by backend and frontend — the single source of truth.

The Next.js client mirrors this in `frontend/src/lib/types.ts`. Anything the
HUD draws is here; anything not here the HUD cannot know.

Sliders follow the P3 physics layer: six setup controls + three environment
controls (claude/P3-SCOPE.md). Camber, toe, brake bias and differential were
cut — no public data can verify them.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ----------------------------------------------------------------- inputs
class Weather(str, Enum):
    DRY = "dry"
    INTER = "inter"
    WET = "wet"


class Compound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class CarSetup(BaseModel):
    """Normalised sliders, 0..1, 0.5 = the real car's setup for that weekend
    (unknown to us, embodied in the baseline lap). Every output is relative."""
    front_wing: float = Field(0.5, ge=0, le=1, description="0 = least downforce, 1 = most")
    rear_wing: float = Field(0.5, ge=0, le=1, description="rear wing dominates drag")
    ride_height: float = Field(0.5, ge=0, le=1, description="0 = lowest (bottoming risk), 1 = highest")
    suspension: float = Field(0.5, ge=0, le=1, description="0 = soft, 1 = stiff")
    suspension_split: float = Field(0.5, ge=0, le=1, description=">0.5 = stiffer front (understeer)")
    fuel_kg: float | None = Field(None, ge=0, le=110, description="None = same as the baseline lap")


class Environment(BaseModel):
    """None = keep the baseline lap's value."""
    track_temp_c: float | None = Field(None, ge=5, le=65)
    air_temp_c: float | None = Field(None, ge=0, le=50)
    weather: Weather = Weather.DRY
    compound: Compound | None = None
    tyre_life: int | None = Field(None, ge=1, le=60, description="laps on the tyre")


class BaselineRef(BaseModel):
    season: int = Field(..., ge=2022, le=2025)
    event: str = Field(..., description="event slug, e.g. bahrain_grand_prix")
    session: Literal["Q", "R", "S", "SQ"] = "Q"
    driver: str = Field(..., description="FastF1 3-letter code, e.g. VER")
    lap: str = Field("representative", description="'representative' (median clean push lap, default), "
                                                  "'fastest', or a lap_uid")


class SimulationRequest(BaseModel):
    baseline: BaselineRef
    setup: CarSetup = CarSetup()
    environment: Environment = Environment()


# ---------------------------------------------------------------- outputs
class SegmentKind(str, Enum):
    STRAIGHT = "straight"
    KINK = "kink"
    LOW_SPEED_CORNER = "low_speed_corner"
    MEDIUM_SPEED_CORNER = "medium_speed_corner"
    HIGH_SPEED_CORNER = "high_speed_corner"


class SegmentInfo(BaseModel):
    """Static description of one segment of the circuit (from silver)."""
    index: int
    kind: SegmentKind
    start_m: float
    end_m: float
    length_m: float
    sector: int | None
    min_radius_m: float | None
    direction: str | None


class SegmentDelta(BaseModel):
    """Per segment: where the time comes from, with its uncertainty."""
    index: int
    kind: SegmentKind
    sector: int | None
    baseline_time_s: float
    ml_s: float = Field(..., description="tyre / temperature / conditions, level-1 model, q50 difference")
    ml_lo_s: float
    ml_hi_s: float
    level2_s: float = Field(..., description="session-level temperature / session-type shift")
    physics_s: float = Field(..., description="setup, physics layer, nominal")
    physics_lo_s: float
    physics_hi_s: float
    total_s: float
    total_lo_s: float
    total_hi_s: float
    achieved_s: float = Field(..., description="what the reconstructed trace actually carries")
    refused_s: float = Field(..., description="part of the request the physics envelope refused (0 = none)")


class TelemetryTrace(BaseModel):
    """Distance-indexed channels on the 20 m display grid.

    The grid is measured, not assumed: telemetry arrives every 240 ms, so
    raw spacing is ~20 m at 300 km/h and ~4 m in a hairpin. `interpolated`
    marks display samples farther than 1.5 grid steps from a real one; the
    HUD dims them so it never claims resolution the data does not have.
    Brake is a boolean upstream (no pressure channel exists). DRS is open
    only where the baseline lap really opened it — a track zone.
    """
    distance_m: list[float]
    time_s: list[float]
    speed_kph: list[float]
    throttle_pct: list[float]
    brake_on: list[bool]
    gear: list[int]
    drs_open: list[bool]
    interpolated: list[bool]


class LapMeta(BaseModel):
    lap_uid: str
    driver: str
    team: str | None
    chassis: str | None
    power_unit: str | None
    season: int
    event: str
    event_name: str
    session: str
    circuit: str | None
    lap_number: int | None
    lap_time_s: float
    compound: str | None
    tyre_life: int | None
    fresh_tyre: bool | None
    track_temp_c: float | None
    air_temp_c: float | None
    telemetry_quality: str | None
    effort_class: str | None
    gap_ahead_s: float | None
    sector_times_s: list[float | None]


class TrackMap(BaseModel):
    view_box: str
    path: str
    sector_boundaries_m: list[float]
    lap_length_m: float
    published_turns: int | None
    measured_turns: int


class BaselineResponse(BaseModel):
    lap: LapMeta
    trace: TelemetryTrace
    segments: list[SegmentInfo]
    track: TrackMap
    available_laps: list[dict]
    integration_note: str


class PhysicsState(BaseModel):
    downforce_pct: float
    drag_pct: float
    mech_grip_pct: float
    grip_multiplier: float
    thermal_grip_pct: float
    fuel_delta_kg: float
    balance_index: float = Field(..., description="-1 understeer .. +1 oversteer")
    warning: str | None


class Grade(BaseModel):
    control: str
    grade: Literal["A", "B", "C"]
    note: str


class EngineerNote(BaseModel):
    severity: Literal["info", "warning", "critical"]
    channel: Literal["balance", "tyre", "aero", "fuel", "weather", "model", "physics", "sectors"]
    message: str
    suggestion: str | None = None


class LapSummary(BaseModel):
    baseline_lap_time_s: float
    simulated_lap_time_s: float
    delta_s: float
    delta_lo_s: float
    delta_hi_s: float
    ml_s: float
    level2_s: float
    physics_s: float
    refused_s: float
    sector_deltas_s: list[float]


class SimulationResponse(BaseModel):
    lap: LapSummary
    segments: list[SegmentDelta]
    baseline: TelemetryTrace
    simulated: TelemetryTrace
    physics: PhysicsState
    grades: list[Grade]
    engineer_log: list[EngineerNote]
    model_version: str
    physics_version: str
    computed_ms: float
