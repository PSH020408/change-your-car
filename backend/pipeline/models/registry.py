"""P4 — model registry: every run lands in its own versioned directory.

    artifacts/models/
      v2026.09.15-1/   model_q10.joblib model_q50.joblib model_q90.joblib
                       spec.json level2.json metrics.json config.yaml
      latest.json      -> the newest version whose gates all passed

A run that misses a gate is still written (so it can be inspected) but never
becomes `latest`; the API only ever loads `latest`.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

from pipeline.models.dataset import FeatureSpec
from pipeline.models.level2 import Level2Model
from pipeline.models.quantile import QuantileSet


def next_version(root: Path) -> str:
    stamp = f"v{date.today():%Y.%m.%d}"
    n = 1
    while (root / f"{stamp}-{n}").exists():
        n += 1
    return f"{stamp}-{n}"


def register(root: Path, model: QuantileSet, level2: Level2Model, metrics: dict, cfg: dict,
             passed: bool) -> str:
    root.mkdir(parents=True, exist_ok=True)
    version = next_version(root)
    d = root / version
    d.mkdir()
    import joblib
    for q, m in model.models.items():
        joblib.dump(m, d / f"model_q{int(round(q * 100))}.joblib")
    (d / "spec.json").write_text(model.spec.to_json())
    (d / "backend.json").write_text(json.dumps({"backend": model.backend, "quantiles": model.quantiles,
                                                 "params": model.params, "margin": float(model.margin)}))
    (d / "level2.json").write_text(json.dumps(level2.to_dict(), indent=1))
    (d / "metrics.json").write_text(json.dumps({**metrics, "version": version, "gates_passed": passed},
                                               indent=1, default=str))
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if passed:
        (root / "latest.json").write_text(json.dumps({"version": version}))
    return version


def load(root: Path, version: str = "latest") -> tuple[QuantileSet, Level2Model, dict]:
    if version == "latest":
        p = root / "latest.json"
        if not p.exists():
            raise FileNotFoundError(f"no registered model in {root} (no run has passed the gates yet)")
        version = json.loads(p.read_text())["version"]
    d = root / version
    spec = FeatureSpec.from_json((d / "spec.json").read_text())
    b = json.loads((d / "backend.json").read_text())
    qs = QuantileSet(spec=spec, quantiles=list(b["quantiles"]), backend=b["backend"], params=b["params"],
                     margin=float(b.get("margin", 0.0)))
    import joblib
    qs.models = {q: joblib.load(d / f"model_q{int(round(q * 100))}.joblib") for q in qs.quantiles}
    l2 = Level2Model.from_dict(json.loads((d / "level2.json").read_text()))
    metrics = json.loads((d / "metrics.json").read_text())
    return qs, l2, metrics


def accept(root: Path, version: str, reason: str) -> None:
    """Make `version` the model the API loads, gates notwithstanding.

    Used when a gate is missed for a documented reason (the lap gate at
    ~49% of the measured skill ceiling, 2026-09-16). The reason is written
    next to the pointer so nobody mistakes it for a pass.
    """
    d = root / version
    if not d.exists():
        raise FileNotFoundError(d)
    m = json.loads((d / "metrics.json").read_text())
    (root / "latest.json").write_text(json.dumps({
        "version": version, "gates_passed": bool(m.get("gates_passed", False)),
        "accepted_manually": True, "reason": reason}, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="python -m pipeline.models.registry accept <version> --reason ...")
    ap.add_argument("cmd", choices=["accept", "show"])
    ap.add_argument("version", nargs="?")
    ap.add_argument("--reason", default="")
    ap.add_argument("--root", type=Path, default=Path("../data/artifacts/models"))
    a = ap.parse_args()
    if a.cmd == "accept":
        accept(a.root, a.version, a.reason)
        print(f"latest -> {a.version}  (manual accept: {a.reason})")
    else:
        p = a.root / "latest.json"
        print(p.read_text() if p.exists() else "no latest")
