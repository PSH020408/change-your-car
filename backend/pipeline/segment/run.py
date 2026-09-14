"""Stage 2 — track segmentation -> data/silver.

Derives track geometry from the fastest clean lap's X/Y trace, then splits
the lap into physically meaningful segments.

    curvature k = |x'y" - y'x"| / (x'^2 + y'^2)^(3/2)

    k > threshold  -> corner, classified by apex speed (low/medium/high)
    k <= threshold -> straight
    braking zones are tagged where Brake > 0 for >= min_segment_len_m

Also emits the SVG path string used by the frontend track map, and the
microsector table (default 28) used for the delta heat overlay.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def compute_centreline(x, y):
    """Smooth the raw GPS trace into a centreline (Savitzky-Golay)."""
    raise NotImplementedError


def compute_curvature(x, y):
    raise NotImplementedError


def segment_track(curvature, speed, cfg: dict):
    """-> list[{index, kind, start_m, end_m, sector, apex_speed_kph}]"""
    raise NotImplementedError


def to_svg_path(x, y, width: int = 1000, height: int = 1000) -> str:
    """Normalise + emit a viewBox-fitted SVG path for the 2D track map."""
    raise NotImplementedError


def run(scope_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scope", type=Path, required=True)
    run(p.parse_args().scope)
