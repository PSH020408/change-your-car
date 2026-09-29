"""P9-2 QSS solver: invariants that hold on any line, checked on a synthetic one."""
import time

import numpy as np
import pytest

from pipeline.physics import qss as Q


def _line(n=1060, ds=5.0):
    d = np.arange(n) * ds
    k = np.zeros(n)
    for a, b, r, sgn in ((900, 1050, 45, 1), (1800, 2000, 120, -1), (2900, 3300, 300, 1),
                         (4200, 4300, 70, -1), (4800, 4900, 60, 1)):
        k[(d >= a) & (d < b)] = sgn / r
    drs = np.zeros(n, bool)
    drs[(d < 800) | ((d > 2100) & (d < 2800))] = True
    return d, k, np.zeros(n), drs


def test_solves_fast_enough():
    d, k, g, drs = _line()
    t0 = time.perf_counter()
    for _ in range(5):
        Q.solve(d, k, g, drs, Q.Car())
    assert (time.perf_counter() - t0) / 5 < 0.05          # G7 budget is 50 ms per simulation


def test_terminal_speed_matches_power_balance():
    # a corner-free loop is pure drag vs power: v^3 = 2 P / (rho CdA)
    n = 800
    d = np.arange(n) * 5.0
    car = Q.Car(cd_a=1.4, rho=1.2, power_kw=780.0)
    p = Q.solve(d, np.zeros(n), np.zeros(n), None, car)
    v_term = (2 * car.power_kw * 1000 / (car.rho * car.cd_a)) ** (1 / 3)
    assert abs(p.speed_ms.max() - v_term) / v_term < 0.01


def test_monotone_in_mass_grip_drag_and_drs():
    d, k, g, drs = _line()
    car = Q.Car(cd_a=1.5)
    base = Q.solve(d, k, g, drs, car).lap_time_s
    assert Q.solve(d, k, g, drs, car.with_setup(fuel_delta_kg=10)).lap_time_s > base
    assert Q.solve(d, k, g, drs, car.with_setup(mech_grip_pct=5)).lap_time_s < base
    assert Q.solve(d, k, g, drs, car.with_setup(drag_pct=5)).lap_time_s > base
    assert Q.solve(d, k, g, None, car).lap_time_s > base     # no DRS is slower


def test_more_downforce_speeds_up_a_grip_limited_corner():
    d, k, g, drs = _line()
    car = Q.Car(cd_a=1.5)
    p0 = Q.solve(d, k, g, drs, car)
    p1 = Q.solve(d, k, g, drs, car.with_setup(downforce_pct=10, drag_pct=8))   # a rear-wing click
    hairpin = (p0.distance_m >= 900) & (p0.distance_m < 1050)
    assert p1.speed_ms[hairpin].min() > p0.speed_ms[hairpin].min()
    assert p1.speed_ms.max() < p0.speed_ms.max()                # and slower at the end of the straight


def test_uphill_costs_time_downhill_gives_it_back():
    d, k, g, drs = _line()
    car = Q.Car(cd_a=1.5)
    base = Q.solve(d, k, g, drs, car).lap_time_s
    up = g.copy()
    up[(d >= 0) & (d < 800)] = 0.04
    assert Q.solve(d, k, up, drs, car).lap_time_s > base
    down = -up
    assert Q.solve(d, k, down, drs, car).lap_time_s < base


def test_segment_times_partition_the_lap_and_wrap():
    d, k, g, drs = _line()
    p = Q.solve(d, k, g, drs, Q.Car())
    bounds = [(0, 900), (900, 1050), (1050, 2900), (2900, 5300)]
    assert abs(p.segment_times(bounds).sum() - p.lap_time_s) < 1e-9
    wrap = p.segment_times([(5200, 100)])[0]
    split = p.segment_times([(5200, 5300)])[0] + p.segment_times([(0, 100)])[0]
    assert wrap == pytest.approx(split)


def test_speed_never_exceeds_the_corner_limit():
    d, k, g, drs = _line()
    p = Q.solve(d, k, g, drs, Q.Car())
    assert np.all(p.speed_ms <= p.v_limit_ms + 1e-6)
