"""P2-6 — driver style bias.

The simulator lets a user pick a driver, so the model has to know what
choosing VER instead of STR actually changes. Style is not a lap time: it is
where a driver brakes, how much speed they carry through the middle, and how
early they get back to power.

The comparison must be SEGMENT-WISE. Comparing a driver's average braking
point across a season against the field's average mixes Monaco's hairpin with
Monza's Parabolica and measures the calendar, not the driver. So every value
is z-scored against the field ON THE SAME SEGMENT of the same session, and
only then averaged per driver.

This also means style is measured relative to the field of that era — which is
what we want, since the reference lap it will be applied to comes from the
same era.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Positive z = later braking, more mid-corner speed, earlier throttle.
STYLE_CHANNELS = {
    "brake_point_frac": +1,     # later in the segment = more aggressive
    "speed_min_kph": +1,        # more speed carried through the apex
    "throttle_on_frac": -1,     # earlier back to power = more aggressive
}

GROUP = ["season", "event", "session", "segment_index"]


def segment_wise_z(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score each style channel against the field on the same segment."""
    df = features.copy()
    for col, sign in STYLE_CHANNELS.items():
        if col not in df:
            continue
        g = df.groupby(GROUP)[col]
        mu, sd = g.transform("mean"), g.transform("std")
        z = (df[col] - mu) / sd.replace(0, np.nan)
        df[f"z_{col}"] = sign * z.clip(-4, 4)
    return df


def driver_bias(features: pd.DataFrame,
                min_samples: int = 20) -> pd.DataFrame:
    """One row per (driver, segment_kind): the driver's style bias there.

    A driver who is aggressive in slow corners is not necessarily aggressive
    in fast ones — that distinction is most of what "driver style" means to a
    race engineer, so the bias is kept per corner type rather than collapsed
    into one number per driver.
    """
    df = segment_wise_z(features)
    zcols = [f"z_{c}" for c in STYLE_CHANNELS if f"z_{c}" in df]
    if not zcols or "driver" not in df:
        return pd.DataFrame()

    grouped = df.groupby(["driver", "segment_kind"])
    out = grouped[zcols].mean().reset_index()
    out["n_samples"] = grouped.size().values

    # A bias measured on a handful of segments is noise wearing a driver's
    # name. Blank it rather than let the model treat it as signal.
    thin = out["n_samples"] < min_samples
    out.loc[thin, zcols] = np.nan

    out["aggression_index"] = out[zcols].mean(axis=1)
    return out.rename(columns={"z_brake_point_frac": "bias_braking_late",
                               "z_speed_min_kph": "bias_apex_speed",
                               "z_throttle_on_frac": "bias_throttle_early"})
