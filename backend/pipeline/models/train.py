"""Stage 4 — XGBoost/LightGBM delta model.

Target : per-segment time delta vs. the session reference lap
Split  : GroupKFold on (season, event, driver) + a held-out event, so the
         reported error is generalisation, not memorisation
Gates  : configs/model.yaml -> acceptance.*  (a run that misses a gate does
         not get registered)
Checks : monotonicity probes — more rear wing must lower trap speed and
         raise high-speed-corner speed; a model that violates physics is
         rejected even if its MAE is good.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_dataset(gold_dir: Path):
    raise NotImplementedError


def train(config: dict):
    raise NotImplementedError


def evaluate(model, X, y, cfg: dict) -> dict:
    """-> {sector_delta_mae_s, lap_delta_mae_s, unseen_track_mae_s, ...}"""
    raise NotImplementedError


def monotonicity_probe(model, feature_spec: dict) -> dict:
    raise NotImplementedError


def register(model, metrics: dict, registry_dir: Path) -> str:
    """Write model + metrics + feature_spec under a semver dir, return version."""
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    train({"path": p.parse_args().config})
