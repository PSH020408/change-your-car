"""P2-4 — the track map the frontend draws.

The HUD colours the circuit by sector delta, so the frontend needs the
outline as a path AND needs to know where along it any distance falls. Both
come from here, in one normalised coordinate system, so the overlay can never
drift out of register with the outline.

Kept deliberately small: a circuit outline is a few hundred points, and the
project's whole premise is a lightweight 2D HUD. Coordinates are rounded to
one decimal in a 1000-unit viewBox — about 0.1% of track width, far finer
than a screen pixel, and it roughly halves the payload versus full floats.
"""
from __future__ import annotations

import numpy as np


def normalise(x: np.ndarray, y: np.ndarray, size: float = 1000.0,
              padding: float = 20.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the circuit into a square viewBox, preserving aspect ratio.

    Preserving aspect is not cosmetic: a circuit stretched to fill the box
    would misrepresent corner geometry, and the same overlay would place a
    braking point in the wrong place on a tall circuit versus a wide one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    span = max(x1 - x0, y1 - y0) or 1.0
    scale = (size - 2 * padding) / span

    nx = (x - x0) * scale + padding + (size - 2 * padding - (x1 - x0) * scale) / 2
    # SVG's y axis points down; flip so the map matches a circuit diagram.
    ny = size - ((y - y0) * scale + padding
                 + (size - 2 * padding - (y1 - y0) * scale) / 2)
    return nx, ny, {"size": size, "scale_px_per_m": scale,
                    "bounds_m": [x0, y0, x1, y1], "padding": padding}


def to_path(x: np.ndarray, y: np.ndarray, close: bool = True,
            precision: int = 1) -> str:
    nx, ny = np.round(x, precision), np.round(y, precision)
    parts = [f"M {nx[0]} {ny[0]}"]
    parts += [f"L {a} {b}" for a, b in zip(nx[1:], ny[1:])]
    if close:
        parts.append("Z")
    return " ".join(parts)


def distance_index(distance_m: np.ndarray, x: np.ndarray, y: np.ndarray,
                   every_m: float = 100.0, precision: int = 1) -> list[dict]:
    """Sparse distance -> screen-position table.

    Lets the frontend place a marker (a sector boundary, a delta hotspot, the
    car) at any lap distance by interpolating between entries, instead of
    shipping the full point list a second time.
    """
    d = np.asarray(distance_m, dtype=float)
    marks = np.arange(0.0, float(d[-1]), every_m)
    return [{"m": round(float(m), 1),
             "x": round(float(np.interp(m, d, x)), precision),
             "y": round(float(np.interp(m, d, y)), precision)} for m in marks]


def build(geo, size: float = 1000.0) -> dict:
    nx, ny, meta = normalise(geo.x_m, geo.y_m, size=size)
    return {
        "view_box": f"0 0 {int(size)} {int(size)}",
        "path": to_path(nx, ny),
        "points": int(len(nx)),
        "distance_index": distance_index(geo.distance_m, nx, ny),
        **meta,
    }
