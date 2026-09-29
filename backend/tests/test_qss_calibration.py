"""P9-3 calibration: phase alignment, DRS carry-over and parameter recovery on a synthetic lap."""
import numpy as np
import pandas as pd
import pytest

from pipeline.physics import qss as Q, calibrate_qss as C


def _synthetic(mu=1.7, cl=4.5, cd=1.3, shift_pts=6, noise=1.5, seed=1):
    n = 1060
    d = np.arange(n) * 5.0
    k = np.zeros(n)
    for a, b, r, sg in ((900, 1050, 45, 1), (1800, 2000, 120, -1), (2900, 3300, 300, 1),
                        (4200, 4300, 70, -1), (4800, 4900, 60, 1)):
        k[(d >= a) & (d < b)] = sg / r
    drs = np.zeros(n, bool)
    drs[(d < 800) | ((d > 2100) & (d < 2800))] = True
    car = Q.Car(mass_kg=806, mu=mu, cl_a=cl, cd_a=cd)
    p = Q.solve(d, k, np.zeros(n), drs, car)
    line = pd.DataFrame({"distance_m": d, "curvature_1pm": k, "grade": np.zeros(n),
                         "speed_kph": p.speed_kph[:n]})
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, 340, replace=False))
    trace = {"distance_m": d[idx].tolist(),
             "speed_kph": (np.roll(p.speed_kph[:n], shift_pts)[idx] + rng.normal(0, noise, len(idx))).tolist(),
             "drs_open": np.roll(drs, shift_pts)[idx].tolist()}
    return line, trace, drs, car, p


def test_phase_alignment_and_drs_carry_over():
    line, trace, drs, car, p = _synthetic()
    v, dr, k = C.lap_on_line(trace, line)
    assert k == -6
    assert np.sqrt(np.mean((v - p.speed_kph[:-1]) ** 2)) < 4.0
    assert (dr == drs).mean() > 0.98


def test_fit_recovers_the_parameters_it_was_given():
    pytest.importorskip("scipy")
    line, trace, drs, car, p = _synthetic()
    v, dr, _ = C.lap_on_line(trace, line)
    fit = C.fit_lap(line, v, dr, Q.Car(mass_kg=806))
    assert abs(fit["mu"] - car.mu) < 0.05
    assert abs(fit["cl_a"] - car.cl_a) < 0.3
    assert abs(fit["cd_a"] - car.cd_a) < 0.08
    assert fit["rms_kph"] < 4.0
    assert abs(fit["sim_lap_time_s"] - p.lap_time_s) < 0.15


def test_probes_have_the_right_signs():
    line, trace, drs, car, p = _synthetic()
    segs = [{"kind": "low_speed_corner", "start_m": 900, "end_m": 1050},
            {"kind": "straight", "start_m": 0, "end_m": 900}]
    pr = C.physics_probes(line, drs, car, p, segs)
    assert pr["fuel_s_per_kg"] > 0
    assert pr["wing_vmax_delta_kph"] < 0
    assert pr["grip_corners_faster"] == pr["grip_corners_checked"] == 1
