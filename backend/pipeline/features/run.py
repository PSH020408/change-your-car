"""Stage 3 — feature engineering -> data/gold.

Feature families
    segment      : length, curvature stats, apex speed, entry/exit speed,
                   time in throttle, time on brake, DRS availability
    car          : chassis one-hot, PU manufacturer, season regulation era
    driver_style : braking aggression (decel onset vs. apex), throttle
                   aggression (time to 100%), mid-corner minimum-speed bias,
                   computed as a driver's z-score against the field per segment kind
    setup_proxy  : real setups are never published, so we infer them —
                   trap speed  -> drag proxy
                   high-speed corner speed -> downforce proxy
                   trap/apex ratio -> aero balance proxy
                   these proxies are what the physics layer later perturbs
    environment  : track/air temp, compound, tyre age, fuel-corrected pace,
                   track evolution (session-relative rolling best)

Output: gold/features.parquet + feature_spec.json (names, dtypes, ranges).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_segment_features(silver_dir: Path):
    raise NotImplementedError


def build_driver_style_bias(laps):
    raise NotImplementedError


def build_setup_proxies(telemetry, segments):
    raise NotImplementedError


def run(scope_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scope", type=Path, required=True)
    run(p.parse_args().scope)
