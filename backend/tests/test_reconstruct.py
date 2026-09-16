"""P5 gate — a delta becomes a trace exactly, legally, and reversibly."""
import numpy as np
import pandas as pd
import pytest

from pipeline.reconstruct import trace as T


def _lap():
    """3 km toy lap on a 10 m grid: straight, hairpin, straight, sweeper, straight."""
    d = np.arange(0.0, 3000.0, 10.0)
    v = np.empty_like(d)
    def ramp(x, x0, x1, va, vb):                      # smooth, continuous speed ramp
        u = np.clip((x - x0) / (x1 - x0), 0, 1)
        return va + (vb - va) * (0.5 - 0.5 * np.cos(np.pi * u))
    for i, x in enumerate(d):
        if x < 900:            v[i] = ramp(x, 0, 900, 240, 300)          # straight: 240 -> 300
        elif x < 1050:         v[i] = ramp(x, 900, 1050, 300, 90)        # hairpin braking 300 -> 90
        elif x < 1200:         v[i] = ramp(x, 1050, 1200, 90, 180)       # hairpin exit 90 -> 180
        elif x < 2100:         v[i] = ramp(x, 1200, 2100, 180, 300)      # straight 180 -> 300
        elif x < 2250:         v[i] = ramp(x, 2100, 2250, 300, 190)      # sweeper 300 -> 190
        elif x < 2400:         v[i] = ramp(x, 2250, 2400, 190, 240)      # sweeper exit
        else:                  v[i] = ramp(x, 2400, 3000, 240, 240)      # straight, holding
    a = T.longitudinal_accel(d, v)
    gear = np.clip((v / 40).astype(int), 1, 8)
    drs = np.where((d > 300) & (d < 850), 12, 8)
    base = pd.DataFrame({"distance_m": d, "speed_kph": v, "brake_on": a < -3.0, "gear": gear, "drs_raw": drs,
                         "throttle_pct": np.where(a > 0.3, 100.0, 0.0)})
    segs = [dict(index=0, kind="straight", start_m=0, end_m=900, peak_curvature_1pm=0.0002, wraps_start_finish=False),
            dict(index=1, kind="low_speed_corner", start_m=900, end_m=1200, peak_curvature_1pm=1 / 34, wraps_start_finish=False),
            dict(index=2, kind="straight", start_m=1200, end_m=2100, peak_curvature_1pm=0.0002, wraps_start_finish=False),
            dict(index=3, kind="high_speed_corner", start_m=2100, end_m=2400, peak_curvature_1pm=1 / 260, wraps_start_finish=False),
            dict(index=4, kind="straight", start_m=2400, end_m=3000, peak_curvature_1pm=0.0002, wraps_start_finish=False)]
    return base, segs


def test_integration_is_distance_over_speed():
    d = np.arange(0, 1001, 10.0)
    assert T.integrate_lap_time(np.full(len(d), 180.0), d) == pytest.approx(1000 / 50.0)


def test_shape_weights_put_the_change_where_the_physics_acts():
    v = np.array([300, 200, 100, 200, 300.0])
    wc, ws = T.shape_weights(v, "low_speed_corner"), T.shape_weights(v, "straight")
    assert wc.argmax() == 2 and ws.argmax() in (0, 4)
    assert wc.min() >= 0.25 and ws.max() == 1.0


def test_zero_delta_returns_the_baseline_untouched():
    base, segs = _lap()
    r = T.reconstruct(base, segs, [0.0] * 5)
    assert np.allclose(r.trace["speed_kph"], base["speed_kph"], atol=0.05)     # bisection tolerance 1e-4 s
    assert r.achieved_delta_s == pytest.approx(0.0, abs=2e-3)
    assert r.clamps == {"lateral_clamped": 0, "traction_clamped": 0, "braking_clamped": 0}
    assert not r.segments["clamped"].any()


def test_each_segment_hits_its_requested_delta_and_only_there():
    base, segs = _lap()
    r = T.reconstruct(base, segs, {0: 0.0, 1: +0.40, 2: -0.15, 3: +0.10, 4: 0.0}, clamp=False)
    seg = r.segments.set_index("segment_index")
    for i, want in {1: 0.40, 2: -0.15, 3: 0.10}.items():
        assert seg.loc[i, "achieved_s"] == pytest.approx(want, abs=0.002)
    assert r.integration_error_s < 0.02
    assert r.achieved_delta_s == pytest.approx(0.35, abs=0.01)     # boundary trapezoids straddle segments: ~1 ms each
    d, v, v0 = r.trace["distance_m"], r.trace["speed_kph"].to_numpy(), base["speed_kph"].to_numpy()
    in_hairpin = (d >= 900) & (d < 1200)
    assert (v[in_hairpin] <= v0[in_hairpin] + 1e-9).all(), "slower corner: no sample faster"
    in_s2 = (d >= 1200) & (d < 2100)
    assert (v[in_s2] >= v0[in_s2] - 1e-9).all(), "faster straight: no sample slower"
    untouched = (d < 900) | (d >= 2400)
    assert np.allclose(v[untouched], v0[untouched], atol=0.05)


def test_slower_corner_slows_the_apex_most():
    base, segs = _lap()
    r = T.reconstruct(base, segs, {1: +0.5}, clamp=False) if False else \
        T.reconstruct(base, segs, [0, 0.5, 0, 0, 0], clamp=False)
    d = r.trace["distance_m"].to_numpy()
    v0 = base["speed_kph"].to_numpy()
    rel_loss = (v0 - r.trace["speed_kph"].to_numpy()) / v0          # share of speed given up
    seg = (d >= 900) & (d < 1200)
    apex = np.argmin(v0[seg])
    assert abs(rel_loss[seg].argmax() - apex) <= 1


def test_impossible_delta_is_clamped_and_reported_not_absorbed():
    """Asking the hairpin to be 3 s faster means far more than ~100 km/h
    through a 34 m radius: the tyres say no. The trace obeys the tyres and
    the residual says how much of the request was refused."""
    base, segs = _lap()
    r = T.reconstruct(base, segs, [0, -3.0, 0, 0, 0])
    seg = r.segments.set_index("segment_index")
    assert seg.loc[1, "clamped"]
    assert seg.loc[1, "achieved_s"] > -2.0, "well short of the request"
    assert r.clamps["lateral_clamped"] > 0
    env = T.Envelope.from_config()
    k = T.effective_curvature(T.segment_curvature(r.trace["distance_m"].to_numpy(), segs),
                              base["speed_kph"].to_numpy(), env)
    v_ms = r.trace["speed_kph"].to_numpy() / 3.6
    assert (v_ms ** 2 * k <= env.grip_accel(v_ms) + 1e-6).all(), "lateral g inside the envelope everywhere"


def test_channels_reproduce_the_drivers_pedals_on_a_zero_delta():
    base, segs = _lap()
    r = T.reconstruct(base, segs, [0.0] * 5)
    t = r.trace
    assert np.mean(t["brake_on"].to_numpy() == base["brake_on"].to_numpy()) >= 0.95
    assert (t["drs_open"].to_numpy() == (base["drs_raw"].to_numpy() == 12)).all()
    full = t["throttle_pct"][base["throttle_pct"] == 100.0]
    assert full.median() > 60, "flat-out zones are recognised as throttle"
    assert (t.loc[t["brake_on"], "throttle_pct"] == 0).all(), "never brake and throttle together"
    order = t.sort_values("speed_kph")["gear"].to_numpy()
    assert (np.diff(order) >= 0).all(), "gear rises with speed"
    assert (np.diff(t["time_s"]) > 0).all()
    assert t["time_s"].iloc[-1] == pytest.approx(r.lap_time_s, rel=0.02)


def test_envelope_speeds_are_physical():
    env = T.Envelope.from_config()
    assert 90 < env.max_corner_speed(np.array([1 / 34]))[0] * 3.6 < 110         # 34 m hairpin: ~100 km/h cap
    assert 150 < env.max_corner_speed(np.array([1 / 60]))[0] * 3.6 < 200        # 60 m: ~165 km/h
    assert env.max_corner_speed(np.array([1 / 2000]))[0] == np.inf              # a gentle bend never limits
    assert 8 < env.traction_accel(np.array([300 / 3.6]))[0] < 14                 # power-limited at v_max
    assert env.braking_accel(np.array([250 / 3.6]))[0] > 40                      # 4 g+ braking from 250


def test_the_envelope_is_anchored_on_the_baseline():
    """A real lap that out-brakes the model's coefficients is evidence, not
    an error: with the anchor a zero delta returns it untouched; without the
    anchor the model would 'correct' reality."""
    base, segs = _lap()
    d, v0 = base["distance_m"].to_numpy(), base["speed_kph"].to_numpy()
    env = T.Envelope.from_config()
    k = T.segment_curvature(d, segs)
    _, free = T.clamp_to_gg_envelope(d, v0, k, env, baseline_kph=None)
    _, anchored = T.clamp_to_gg_envelope(d, v0, k, env, baseline_kph=v0)
    assert sum(free.values()) > 0, "the toy lap does exceed the bare model somewhere"
    assert sum(anchored.values()) == 0


def test_time_axis_is_scaled_to_the_official_lap_time():
    base, segs = _lap()
    t_int = T.integrate_lap_time(base["speed_kph"].to_numpy(), base["distance_m"].to_numpy())
    r = T.reconstruct(base, segs, [0.0] * 5, official_lap_time_s=t_int * 1.005)
    assert r.time_scale == pytest.approx(1.005)
    assert r.trace["time_s"].iloc[-1] == pytest.approx(t_int * 1.005, rel=0.02)
    r2 = T.reconstruct(base, segs, [0, 0.4, 0, 0, 0], official_lap_time_s=t_int * 1.005)
    assert r2.achieved_delta_s == pytest.approx(0.4, abs=0.01), "deltas are not scaled"


def test_anchor_uses_the_same_two_point_acceleration_as_the_passes():
    """A sharp braking onset: central differences under-read it and the
    baseline was clamped on real telemetry (2024 Bahrain Q: 4 samples)."""
    d = np.arange(0, 500, 10.0)
    v = np.where(d < 200, 300.0, 300.0 - 3.0 * (d - 200))        # hard step into braking
    base = pd.DataFrame({"distance_m": d, "speed_kph": v, "brake_on": d >= 200})
    segs = [dict(index=0, kind="straight", start_m=0, end_m=200, peak_curvature_1pm=0.0, wraps_start_finish=False),
            dict(index=1, kind="low_speed_corner", start_m=200, end_m=500, peak_curvature_1pm=1 / 400, wraps_start_finish=False)]
    r = T.reconstruct(base, segs, [0.0, 0.0])
    assert sum(r.clamps.values()) == 0
    assert np.allclose(r.trace["speed_kph"], v, atol=0.05)
