"""Stage 1 — FastF1 ingestion -> data/bronze.

Responsibilities
    * enable + warm the FastF1 cache (never re-download a session)
    * load laps, car telemetry, weather and session metadata
    * apply lap filters (in/out laps, deleted laps, 107% rule, SC/VSC)
    * resample telemetry onto a uniform distance grid (default 10 m)
    * write partitioned parquet: bronze/{season}/{event}/{session}/*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_session(season: int, event: str, session: str):
    """Return a loaded FastF1 Session. TODO: Phase 1."""
    raise NotImplementedError


def filter_laps(laps, cfg: dict):
    """Drop in/out/deleted/SC laps and laps slower than the 107% gate."""
    raise NotImplementedError


def resample_telemetry(tel, step_m: float):
    """Interpolate channels onto a uniform distance grid — the single most
    important normalisation step: every downstream comparison assumes two
    laps share an x-axis."""
    raise NotImplementedError


def run(scope_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scope", type=Path, required=True)
    run(p.parse_args().scope)
