"""P9-1 racing line: elevation gradient, DRS share and the ensemble's Z carry-through."""
import numpy as np
import pandas as pd

from pipeline.segment import ensemble as E
from pipeline.physics import line as L


def _lap(n=400, lap_len=4000.0, jitter=0.0, seed=0, z_amp=12.0, with_z=True, drs=(500.0, 1200.0)):
    rng = np.random.default_rng(seed)
    d = np.linspace(0, lap_len, n, endpoint=False)
    r = lap_len / (2 * np.pi)
    th = d / r
    x = r * np.cos(th) * 10 + rng.normal(0, jitter, n)         # raw units = 0.1 m
    y = r * np.sin(th) * 10 + rng.normal(0, jitter, n)
    z = (z_amp * np.sin(th) * 10) + rng.normal(0, jitter, n)   # one hill per lap
    speed = 200 + 60 * np.cos(2 * th)
    drs_raw = np.where((d >= drs[0]) & (d <= drs[1]), 12, 0)
    f = pd.DataFrame({"distance_m": d, "pos_x": x, "pos_y": y, "speed_kph": speed, "drs_raw": drs_raw})
    if with_z:
        f["pos_z"] = z
    return f


def test_ensemble_carries_z_and_spread():
    frames = [_lap(jitter=2.0, seed=s) for s in range(5)]
    ens = E.build_ensemble(frames, 4000.0, step_m=5.0)
    assert "pos_z" in ens.frame.columns
    assert ens.z_spread is not None and ens.z_spread.shape == (800,)
    assert 0.5 < float(np.median(ens.z_spread)) < 4.0        # jitter 2 raw units -> spread ~2


def test_ensemble_without_z_stays_flat():
    frames = [_lap(with_z=False, seed=s) for s in range(4)]
    ens = E.build_ensemble(frames, 4000.0, step_m=5.0)
    assert "pos_z" not in ens.frame.columns and ens.z_spread is None


def test_gradient_recovers_a_known_hill():
    lap_len, amp = 4000.0, 12.0
    f = _lap(z_amp=amp)
    d = f["distance_m"].to_numpy()
    z_m = f["pos_z"].to_numpy() * 0.1
    zs, dz = L.elevation_profile(d, z_m, window_m=90.0, order=2, period_m=lap_len)
    r = lap_len / (2 * np.pi)
    expected = amp / r * np.cos(d / r)                          # d/ds of amp*sin(s/r)
    assert np.max(np.abs(zs - z_m)) < 0.2
    assert np.max(np.abs(dz - expected)) < 0.15 * np.max(np.abs(expected))


def test_drs_share_lines_up_with_the_zone():
    frames = [_lap(seed=s) for s in range(4)]
    picks = [("Q", None, f) for f in frames]
    ens = E.build_ensemble(frames, 4000.0, step_m=5.0)
    grid = ens.frame["distance_m"].to_numpy()
    share = L.drs_share(picks, ens, grid)
    inside = (grid > 600) & (grid < 1100)
    outside = (grid > 1500) & (grid < 3800)
    assert share[inside].min() > 0.9 and share[outside].max() < 0.1
