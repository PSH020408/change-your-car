"""Wire contract shared by backend and frontend.

This module is the single source of truth for the API shape: the Next.js
client mirrors it in `frontend/src/lib/types.ts`.
"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Setup & environment inputs
# --------------------------------------------------------------------------
class Weather(str, Enum):
    DRY = "dry"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class Compound(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTER = "intermediate"
    WET = "wet"


class CarSetup(BaseModel):
    """Normalised setup sliders. All values are 0-1 unless stated."""

    front_wing: float = Field(0.5, ge=0, le=1, description="0 = min drag, 1 = max downforce")
    rear_wing: float = Field(0.5, ge=0, le=1)
    ride_height_front_mm: float = Field(25.0, ge=15, le=60)
    ride_height_rear_mm: float = Field(60.0, ge=40, le=110)
    suspension_stiffness_front: float = Field(0.5, ge=0, le=1)
    suspension_stiffness_rear: float = Field(0.5, ge=0, le=1)
    camber_front_deg: float = Field(-3.2, ge=-4.5, le=-1.0)
    camber_rear_deg: float = Field(-1.8, ge=-3.0, le=-0.5)
    toe_front_deg: float = Field(0.05, ge=-0.3, le=0.3)
    toe_rear_deg: float = Field(0.15, ge=-0.3, le=0.5)
    brake_bias_pct: float = Field(56.0, ge=50, le=62)
    diff_on_throttle: float = Field(0.5, ge=0, le=1)


class Environment(BaseModel):
    track_temp_c: float = Field(35.0, ge=5, le=60)
    air_temp_c: float = Field(24.0, ge=0, le=45)
    weather: Weather = Weather.DRY
    track_evolution: float = Field(0.7, ge=0, le=1, description="0 = green, 1 = fully rubbered")
    wind_kph: float = Field(8.0, ge=0, le=60)
    compound: Compound = Compound.SOFT
    fuel_kg: float = Field(15.0, ge=5, le=110)


class SimulationRequest(BaseModel):
    season: int = Field(..., ge=2021, le=2025)
    event: str
    session: Literal["FP2", "FP3", "Q", "R"] = "Q"
    driver: str = Field(..., description="FastF1 3-letter code, e.g. VER")
    chassis: str = Field(..., description="e.g. RB20, SF-24, MCL38")
    setup: CarSetup = CarSetup()
    environment: Environment = Environment()


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
class SegmentKind(str, Enum):
    STRAIGHT = "straight"
    LOW_SPEED_CORNER = "low_speed_corner"
    MEDIUM_SPEED_CORNER = "medium_speed_corner"
    HIGH_SPEED_CORNER = "high_speed_corner"
    BRAKING = "braking"


class SegmentDelta(BaseModel):
    index: int
    kind: SegmentKind
    start_m: float
    end_m: float
    sector: Literal[1, 2, 3]
    baseline_time_s: float
    simulated_time_s: float
    delta_s: float
    confidence: float = Field(..., ge=0, le=1)


class TelemetryTrace(BaseModel):
    """Distance-indexed channels. All arrays share the same length."""

    distance_m: list[float]
    speed_kph: list[float]
    throttle_pct: list[float]
    brake_pct: list[float]
    gear: list[int]
    drs: list[int]


class PhysicsDelta(BaseModel):
    downforce_delta_pct: float
    drag_delta_pct: float
    mechanical_grip_delta_pct: float
    tyre_thermal_grip_delta_pct: float
    balance_index: float = Field(..., description="<0 understeer, >0 oversteer")


class EngineerNote(BaseModel):
    severity: Literal["info", "warning", "critical"]
    channel: Literal["balance", "tyre", "aero", "brakes", "traction"]
    message: str
    suggestion: str | None = None


class SimulationResponse(BaseModel):
    lap_delta_s: float
    sector_deltas_s: list[float]
    segments: list[SegmentDelta]
    baseline: TelemetryTrace
    simulated: TelemetryTrace
    physics: PhysicsDelta
    engineer_log: list[EngineerNote]
    model_version: str
    computed_ms: float
