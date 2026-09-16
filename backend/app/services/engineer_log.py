"""P6 — the AI Engineering Log: rules, not prose generation.

Every note is traceable to a number in the response. The log says what
changed, where the lap gained and lost, what the physics refused, how sure
each layer is (grades), and the two classic warnings a race engineer gives:
balance (under/oversteer) and tyre window.
"""
from __future__ import annotations

from app.schemas import domain as S
from pipeline.physics.modifiers import PhysicsConfig

GRADE_NOTES = {
    "A": "learned from data (66 sessions); interval from the model",
    "B": "physics, coefficient size checked against our own data",
    "C": "physics, literature value only - wide band, cannot be verified with public data",
}


def grades(cfg: PhysicsConfig) -> list[S.Grade]:
    g = cfg.grades()
    rows = [
        ("Front wing", g.get("aero.front_wing.downforce_pct", "B")),
        ("Rear wing", g.get("aero.rear_wing.downforce_pct", "B")),
        ("Ride height", g.get("ground_effect.downforce_pct_full_range", "C")),
        ("Suspension", g.get("suspension.mech_grip_pct_full_range", "C")),
        ("Fuel load", g.get("fuel.s_per_kg_per_lap", "B")),
        ("Tyre compound / age", "A"),
        ("Track / air temperature", "A"),
        ("Weather (inter / wet)", g.get("weather.grip_multiplier.wet", "C")),
    ]
    return [S.Grade(control=c, grade=gr, note=GRADE_NOTES[gr]) for c, gr in rows]


def _kind(k: str) -> str:
    return k.replace("_speed_corner", "-speed corner").replace("_", " ")


def build(req: S.SimulationRequest, b, segs: list[S.SegmentDelta], lap: S.LapSummary, phys: S.PhysicsState,
          predictor, env_changed: bool) -> list[S.EngineerNote]:
    notes: list[S.EngineerNote] = []
    N = S.EngineerNote
    setup, env = req.setup, req.environment

    # --- headline
    if abs(lap.delta_s) < 0.005 and not env_changed and all(
            abs(v - 0.5) < 1e-9 for v in (setup.front_wing, setup.rear_wing, setup.ride_height, setup.suspension, setup.suspension_split)) \
            and setup.fuel_kg is None and env.weather == S.Weather.DRY:
        notes.append(N(severity="info", channel="model", message="Baseline setup and conditions: the simulated lap is the real lap."))
        return notes
    sign = "faster" if lap.delta_s < 0 else "slower"
    notes.append(N(severity="info", channel="sectors",
                   message=f"Lap {abs(lap.delta_s):.3f} s {sign} ({lap.delta_lo_s:+.3f} to {lap.delta_hi_s:+.3f} s band). "
                           f"Setup {lap.physics_s:+.3f} s, conditions {lap.ml_s + lap.level2_s:+.3f} s."))

    # --- where
    if segs:
        gain = min(segs, key=lambda s: s.total_s); loss = max(segs, key=lambda s: s.total_s)
        if gain.total_s < -0.005:
            notes.append(N(severity="info", channel="sectors",
                           message=f"Biggest gain: segment {gain.index} ({_kind(gain.kind.value)}, S{gain.sector}) {gain.total_s:+.3f} s."))
        if loss.total_s > 0.005:
            notes.append(N(severity="info", channel="sectors",
                           message=f"Biggest loss: segment {loss.index} ({_kind(loss.kind.value)}, S{loss.sector}) {loss.total_s:+.3f} s."))

    # --- aero trade
    if abs(phys.downforce_pct) > 0.5 or abs(phys.drag_pct) > 0.5:
        notes.append(N(severity="info", channel="aero",
                       message=f"Aero: downforce {phys.downforce_pct:+.1f} %, drag {phys.drag_pct:+.1f} %. "
                               f"Corners {'gain' if phys.downforce_pct > 0 else 'lose'}, straights {'lose' if phys.drag_pct > 0 else 'gain'}.",
                       suggestion=("Long straights on this circuit: consider trimming the rear wing." if phys.drag_pct > 3
                                   else None)))
    if setup.ride_height < 0.25:
        notes.append(N(severity="warning", channel="aero",
                       message="Ride height in the bottoming zone: floor downforce collapses (porpoising) below ~25 % of the slider.",
                       suggestion="Raise the car until the downforce figure stops falling."))

    # --- balance
    if phys.warning == "understeer":
        notes.append(N(severity="warning", channel="balance",
                       message=f"Understeer bias (index {phys.balance_index:+.2f}): front grip runs out first, slow-corner exits suffer.",
                       suggestion="More front wing, or soften the front / stiffen the rear."))
    elif phys.warning == "oversteer":
        notes.append(N(severity="warning", channel="balance",
                       message=f"Oversteer bias (index {phys.balance_index:+.2f}): rear grip runs out first, traction and tyre wear suffer.",
                       suggestion="More rear wing, or stiffen the front / soften the rear."))

    # --- fuel
    if setup.fuel_kg is not None and abs(phys.fuel_delta_kg) > 1:
        notes.append(N(severity="info", channel="fuel",
                       message=f"Fuel {phys.fuel_delta_kg:+.0f} kg vs the baseline lap: {phys.fuel_delta_kg * 0.03:+.2f} s "
                               f"at 0.030 s/kg (measured 0.029 on our races)."))

    # --- tyres / temperature
    if env.tyre_life is not None and b.lap.get("tyre_life") is not None and env.tyre_life != b.lap["tyre_life"]:
        notes.append(N(severity="info", channel="tyre",
                       message=f"Tyre age {b.lap['tyre_life']} -> {env.tyre_life} laps: {lap.ml_s:+.3f} s from the model "
                               f"(learned from {'66' if predictor else 'no'} sessions)."))
    if env.compound is not None and b.lap.get("compound") and env.compound.value != b.lap["compound"]:
        notes.append(N(severity="info", channel="tyre", message=f"Compound {b.lap['compound']} -> {env.compound.value}: effect from the model."))
    if env.track_temp_c is not None and b.lap.get("track_temp_c") is not None:
        d = env.track_temp_c - b.lap["track_temp_c"]
        if abs(d) >= 3:
            notes.append(N(severity="info", channel="tyre",
                           message=f"Track temperature {d:+.0f} C vs the session: level-2 shift {lap.level2_s:+.3f} s. "
                                   f"With 57 sessions the temperature coefficient is measured, but its band is wide."))
    if env.weather != S.Weather.DRY:
        notes.append(N(severity="warning", channel="weather",
                       message=f"{env.weather.value.capitalize()} conditions: grip x{phys.grip_multiplier:.2f}. The model trained on dry laps only; "
                               f"this is a literature multiplier (grade C) with a wide band."))
    if predictor is None and env_changed:
        notes.append(N(severity="warning", channel="model", message="No registered ML model: condition changes are ignored."))

    # --- physics refusals
    if lap.refused_s > 0.01:
        worst = max(segs, key=lambda s: s.refused_s)
        notes.append(N(severity="critical", channel="physics",
                       message=f"The tyre / power envelope refused {lap.refused_s:.3f} s of the requested gain "
                               f"(most in segment {worst.index}, {_kind(worst.kind.value)}). The trace obeys the envelope; the delta shown is what was achievable."))
    return notes
