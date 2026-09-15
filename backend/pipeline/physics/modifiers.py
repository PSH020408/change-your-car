"""P3 — the physics layer: HUD sliders -> relative physical deltas.

WHY THIS IS ANALYTIC AND NOT LEARNED
FastF1 publishes no setup data — no wing angle, ride height or spring rate
ever appears in a public feed. The ML layer (P4) therefore learns only what
the data varies: tyre, temperature, fuel, driver, chassis. The mapping from
"rear wing +20%" to "+2% downforce, +1.6% drag" has to be written down, and
writing it down has two advantages we exploit: every function is a pure,
unit-testable, monotonic (or deliberately non-monotonic) map, and every
coefficient carries a GRADE and an UNCERTAINTY that the HUD can show.

CONVENTIONS
- Sliders are normalised 0..1; 0.5 is the real car's (unknown) baseline.
- Every output is a RELATIVE change vs. that baseline: 0 in -> 0 out.
- Percent means percent of the baseline quantity (+10 = ten percent more).
- Balance is in "points" of aero balance; + = front-biased -> oversteer.
- `scale` selects the coefficient set: "nominal", "low" (value*(1-u)) or
  "high" (value*(1+u)); the segment layer evaluates all three for a band.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "physics.yaml"
SCALES = ("nominal", "low", "high")


# ------------------------------------------------------------- coefficients
@dataclass(frozen=True)
class Coeff:
    value: float
    u: float = 0.0          # relative uncertainty
    grade: str = "C"

    def at(self, scale: str = "nominal") -> float:
        if scale == "low":
            return self.value * (1.0 - self.u)
        if scale == "high":
            return self.value * (1.0 + self.u)
        if scale != "nominal":
            raise ValueError(f"unknown scale {scale!r}")
        return self.value

    @staticmethod
    def parse(raw: Any) -> "Coeff":
        if isinstance(raw, dict):
            return Coeff(float(raw["value"]), float(raw.get("u", 0.0)), str(raw.get("grade", "C")))
        return Coeff(float(raw), 0.0, "C")


def _c(d: dict, *keys: str) -> Coeff:
    node: Any = d
    for k in keys:
        node = node[k]
    return Coeff.parse(node)


@dataclass
class PhysicsConfig:
    raw: dict
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PhysicsConfig":
        p = Path(path) if path else _DEFAULT_PATH
        with open(p) as fh:
            return cls(yaml.safe_load(fh), p)

    def coeff(self, *keys: str) -> Coeff:
        return _c(self.raw, *keys)

    def grades(self) -> dict[str, str]:
        """Flat {section.key: grade} for the HUD's legend."""
        out: dict[str, str] = {}

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                if "value" in node and "grade" in node:
                    out[prefix] = str(node["grade"])
                    return
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))
        walk(self.raw, "")
        return out


_CFG: PhysicsConfig | None = None


def default_config() -> PhysicsConfig:
    global _CFG
    if _CFG is None:
        _CFG = PhysicsConfig.load()
    return _CFG


# ------------------------------------------------------------------ outputs
@dataclass(frozen=True)
class AeroDelta:
    downforce_pct: float = 0.0
    drag_pct: float = 0.0
    balance_pts: float = 0.0   # + = front-biased (oversteer), - = rear-biased

    def __add__(self, o: "AeroDelta") -> "AeroDelta":
        return AeroDelta(self.downforce_pct + o.downforce_pct,
                         self.drag_pct + o.drag_pct,
                         self.balance_pts + o.balance_pts)


@dataclass(frozen=True)
class MechDelta:
    grip_pct: float = 0.0
    balance_pts: float = 0.0


def _centred(slider: float) -> float:
    """0..1 slider -> -0.5..+0.5 around the baseline, clipped."""
    return min(max(float(slider), 0.0), 1.0) - 0.5


# ----------------------------------------------------------------- B: wings
def wing_to_aero(front_wing: float, rear_wing: float,
                 cfg: PhysicsConfig | None = None, scale: str = "nominal") -> AeroDelta:
    """Wing sliders -> downforce / drag / balance deltas (grade B).

    Linear over the slider range: a wing element's lift is close to linear in
    angle until it stalls, and the HUD range stays inside that. The rear
    wing dominates drag, the front wing dominates balance.
    """
    cfg = cfg or default_config()
    f, r = _centred(front_wing), _centred(rear_wing)
    fw, rw = cfg.raw["aero"]["front_wing"], cfg.raw["aero"]["rear_wing"]
    return AeroDelta(
        downforce_pct=f * _c(fw, "downforce_pct").at(scale) + r * _c(rw, "downforce_pct").at(scale),
        drag_pct=f * _c(fw, "drag_pct").at(scale) + r * _c(rw, "drag_pct").at(scale),
        balance_pts=f * _c(fw, "balance_pts").at(scale) + r * _c(rw, "balance_pts").at(scale),
    )


# ---------------------------------------------------------- C: ride height
def ride_height_to_ground_effect(ride_height: float,
                                 cfg: PhysicsConfig | None = None,
                                 scale: str = "nominal") -> AeroDelta:
    """Ride-height slider -> floor downforce (grade C, non-monotonic).

    Lower = more venturi load, until the floor bottoms and the flow stalls:
    the porpoising that defined 2022. Below `bottoming_threshold` a quadratic
    penalty takes the gain back. 0 = lowest, 1 = highest, 0.5 = baseline.
    """
    cfg = cfg or default_config()
    g = cfg.raw["ground_effect"]
    h = min(max(float(ride_height), 0.0), 1.0)
    gain = -(h - 0.5) * _c(g, "downforce_pct_full_range").at(scale)      # lower -> +
    thr = _c(g, "bottoming_threshold").at(scale)
    if h < thr and thr > 0:
        x = (thr - h) / thr                                               # 0..1 below the threshold
        gain -= x * x * _c(g, "bottoming_penalty_pct").at(scale)
    drag = gain * _c(g, "drag_per_downforce").at(scale)
    bal = -(h - 0.5) * _c(g, "balance_pts_full_range").at(scale)   # lower -> rear load -> negative
    return AeroDelta(downforce_pct=gain, drag_pct=drag, balance_pts=bal)


# ----------------------------------------------------------- C: suspension
def suspension_to_mechanical_grip(stiffness: float, front_rear_split: float = 0.5,
                                  roughness: float | None = None,
                                  cfg: PhysicsConfig | None = None,
                                  scale: str = "nominal") -> MechDelta:
    """Spring stiffness -> mechanical grip and balance (grade C).

    Softer springs let the tyre follow kerbs and bumps (grip on rough
    tracks); stiffer springs hold the floor at its ride height (aero
    platform on smooth ones). `roughness` 0..1 weighs the two; the circuit
    config carries it. `front_rear_split` 0.5 = balanced, >0.5 = stiffer
    front = understeer.
    """
    cfg = cfg or default_config()
    s = cfg.raw["suspension"]
    rough = float(cfg.raw["suspension"].get("default_roughness", 0.5) if roughness is None else roughness)
    rough = min(max(rough, 0.0), 1.0)
    k = _centred(stiffness)
    grip = (-k * _c(s, "mech_grip_pct_full_range").at(scale) * rough
            + k * _c(s, "aero_platform_pct_full_range").at(scale) * (1.0 - rough))
    bal = _centred(front_rear_split) * _c(s, "balance_pts_per_split").at(scale)
    return MechDelta(grip_pct=grip, balance_pts=bal)


# ------------------------------------------------------------------ B: fuel
def baseline_fuel_kg(session: str, lap_number: int | None = None,
                     total_laps: int | None = None, cfg: PhysicsConfig | None = None) -> float:
    """Our best guess of how much fuel the baseline lap carried.

    Qualifying laps run near-empty; a race lap burns down linearly from a
    full tank. We never see the real number — FastF1 has no fuel channel.
    """
    cfg = cfg or default_config()
    full = float(cfg.raw["car"]["fuel_max_kg"])
    if str(session).upper().startswith("Q") or str(session).upper() in ("SQ", "S"):
        return 8.0
    if lap_number and total_laps and total_laps > 0:
        frac = 1.0 - min(max((lap_number - 1) / total_laps, 0.0), 1.0)
        return 3.0 + frac * (full - 3.0)
    return 0.5 * full


def fuel_mass_delta(fuel_kg: float, baseline_kg: float,
                    cfg: PhysicsConfig | None = None) -> tuple[float, float]:
    """(delta_kg, delta_mass_pct) of the whole car vs. the baseline lap."""
    cfg = cfg or default_config()
    dry = float(cfg.raw["car"]["mass_dry_kg"])
    dkg = float(fuel_kg) - float(baseline_kg)
    return dkg, 100.0 * dkg / (dry + float(baseline_kg))


def fuel_lap_penalty_s(delta_kg: float, cfg: PhysicsConfig | None = None,
                       scale: str = "nominal") -> float:
    """Whole-lap time cost of carrying `delta_kg` more fuel (grade B)."""
    cfg = cfg or default_config()
    return delta_kg * cfg.coeff("fuel", "s_per_kg_per_lap").at(scale)


# ------------------------------------------------------------------ A: tyre
def tyre_thermal_grip(track_temp_c: float, compound: str, stint_lap: int = 1,
                      cfg: PhysicsConfig | None = None, scale: str = "nominal") -> float:
    """Grip (%) vs. the compound's optimum at lap 1: 0 at the sweet spot,
    negative on both sides and with age. Grade A — the ML owns the size of
    this in-range; this bell is for extrapolation and explanation."""
    cfg = cfg or default_config()
    t = cfg.raw["tyre"]
    comp = str(compound).upper()
    opt = float(t["window_c"].get(comp, t["window_c"]["MEDIUM"]))
    width = _c(t, "window_width_c").at(scale)
    loss = _c(t, "max_loss_pct").at(scale)
    x = (float(track_temp_c) - opt) / width
    thermal = -loss * min(x * x, 1.0)                      # saturates at the window edge
    deg = -float(t["deg_pct_per_lap"].get(comp, 0.06)) * max(int(stint_lap) - 1, 0)
    return thermal + deg


# --------------------------------------------------------------- C: weather
def weather_grip_multiplier(weather: str, cfg: PhysicsConfig | None = None,
                            scale: str = "nominal") -> float:
    """dry 1.0, inter ~0.8, wet ~0.65 — we train on dry laps only, so this is
    literature (grade C). The uncertainty here is ABSOLUTE (u = +-0.08 means
    0.72..0.88), unlike the relative u elsewhere."""
    cfg = cfg or default_config()
    node = cfg.raw["weather"]["grip_multiplier"].get(str(weather).lower())
    if node is None:
        raise ValueError(f"unknown weather {weather!r}; expected dry / inter / wet")
    c = Coeff.parse(node)
    if scale == "low":
        return c.value - c.u
    if scale == "high":
        return min(c.value + c.u, 1.0)
    return c.value


# --------------------------------------------------------------- balance
def balance_index(aero: AeroDelta, mech: MechDelta, cfg: PhysicsConfig | None = None) -> float:
    """One number in [-1, 1] for the Engineering Log: - understeer, + oversteer."""
    cfg = cfg or default_config()
    pts = aero.balance_pts + mech.balance_pts
    idx = pts * float(cfg.raw["balance"]["pts_to_index"])
    return min(max(idx, -1.0), 1.0)


def balance_warning(idx: float, cfg: PhysicsConfig | None = None) -> str | None:
    cfg = cfg or default_config()
    b = cfg.raw["balance"]
    if idx <= float(b["warn_understeer"]):
        return "understeer"
    if idx >= float(b["warn_oversteer"]):
        return "oversteer"
    return None


# ------------------------------------------------------------ whole setup
@dataclass(frozen=True)
class SetupInput:
    """Everything the HUD's sliders say. Baseline = all 0.5, fuel = baseline."""
    front_wing: float = 0.5
    rear_wing: float = 0.5
    ride_height: float = 0.5
    suspension: float = 0.5
    suspension_split: float = 0.5
    fuel_kg: float | None = None       # None = same as the baseline lap
    weather: str = "dry"
    track_temp_c: float | None = None  # None = same as the baseline lap
    compound: str | None = None
    stint_lap: int = 1


@dataclass(frozen=True)
class PhysicsState:
    """Relative physical state of the modified car vs. the baseline lap."""
    downforce_pct: float
    drag_pct: float
    mech_grip_pct: float
    grip_multiplier: float          # weather, multiplicative on all grip
    thermal_grip_pct: float         # tyre window / age (0 when temp not given)
    mass_pct: float
    fuel_delta_kg: float
    balance_index: float
    warning: str | None
    scale: str = "nominal"
    notes: tuple[str, ...] = field(default_factory=tuple)


def physics_state(setup: SetupInput, *, session: str = "Q", baseline_fuel: float | None = None,
                  baseline_track_temp_c: float | None = None, roughness: float | None = None,
                  cfg: PhysicsConfig | None = None, scale: str = "nominal") -> PhysicsState:
    """Run every modifier once and gather the car's relative state."""
    cfg = cfg or default_config()
    aero = (wing_to_aero(setup.front_wing, setup.rear_wing, cfg, scale)
            + ride_height_to_ground_effect(setup.ride_height, cfg, scale))
    mech = suspension_to_mechanical_grip(setup.suspension, setup.suspension_split, roughness, cfg, scale)

    base_fuel = baseline_fuel if baseline_fuel is not None else baseline_fuel_kg(session, cfg=cfg)
    fuel = base_fuel if setup.fuel_kg is None else float(setup.fuel_kg)
    dkg, mass_pct = fuel_mass_delta(fuel, base_fuel, cfg)

    thermal = 0.0
    notes: list[str] = []
    if setup.track_temp_c is not None and setup.compound:
        # relative to the baseline lap's own temperature, not to the optimum
        ref_t = baseline_track_temp_c if baseline_track_temp_c is not None else setup.track_temp_c
        thermal = (tyre_thermal_grip(setup.track_temp_c, setup.compound, setup.stint_lap, cfg, scale)
                   - tyre_thermal_grip(ref_t, setup.compound, 1, cfg, scale))
        notes.append("tyre thermal term is grade A: the ML's own estimate wins in-range")

    idx = balance_index(aero, mech, cfg)
    return PhysicsState(
        downforce_pct=aero.downforce_pct, drag_pct=aero.drag_pct,
        mech_grip_pct=mech.grip_pct,
        grip_multiplier=weather_grip_multiplier(setup.weather, cfg, scale),
        thermal_grip_pct=thermal, mass_pct=mass_pct, fuel_delta_kg=dkg,
        balance_index=idx, warning=balance_warning(idx, cfg), scale=scale, notes=tuple(notes),
    )
