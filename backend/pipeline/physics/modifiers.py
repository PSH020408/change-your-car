"""Inference-time physics layer — UI sliders -> coefficient deltas.

Deliberately analytic (not learned): the ML model predicts how the CAR
responds to a coefficient change, while this module owns the mapping from
"front wing +2 clicks" to "+3.1% downforce, +1.9% drag". Keeping it explicit
makes the simulator interpretable and lets us unit-test monotonicity.

Every function returns a *relative* delta vs. the baseline setup, so the
model never has to learn absolute aero numbers it cannot observe.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AeroDelta:
    downforce_pct: float
    drag_pct: float
    balance_shift: float   # + = front-biased (oversteer), - = rear-biased


def wing_to_aero(front_wing: float, rear_wing: float) -> AeroDelta:
    """Wing angle -> Cl/Cd deltas.

    Rear wing dominates drag; front wing dominates balance. Coefficients are
    calibrated in Phase 3 against trap-speed vs. corner-speed regressions.
    """
    raise NotImplementedError


def ride_height_to_ground_effect(front_mm: float, rear_mm: float) -> AeroDelta:
    """Ground-effect venturi gain rises as ride height falls, then collapses
    (porpoising / plank wear) below a floor. Non-monotonic by design."""
    raise NotImplementedError


def suspension_to_mechanical_grip(stiff_f: float, stiff_r: float) -> float:
    """Stiffer = better aero platform on smooth tracks, worse kerb/bump
    compliance on street circuits. Track-roughness weighted."""
    raise NotImplementedError


def tyre_thermal_grip(track_temp_c: float, compound: str, stint_lap: int) -> float:
    """Compound-specific grip vs. temperature window (bell curve) with a
    degradation term. Outside the window grip drops on both sides."""
    raise NotImplementedError


def weather_grip_multiplier(weather: str, track_evolution: float) -> float:
    raise NotImplementedError


def balance_index(aero: AeroDelta, mech_front: float, mech_rear: float) -> float:
    """Single scalar the Engineering Log turns into over/understeer warnings."""
    raise NotImplementedError
