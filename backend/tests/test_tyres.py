"""P10-1: the Pirelli nomination table covers every event in the store and maps every slick label."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / ".." / "data" / "artifacts" / "baselines"


def _table():
    return yaml.safe_load((ROOT / "configs" / "tyres.yaml").read_text())["events"]


def test_every_row_is_three_distinct_ascending_compounds():
    for key, row in _table().items():
        nums = [int(str(row[k]).lstrip("C")) for k in ("hard", "medium", "soft")]
        assert nums == sorted(nums) and len(set(nums)) == 3, key
        assert 1 <= nums[0] and nums[-1] <= 6, key
        assert str(row.get("source", "")).startswith("http"), key


def test_table_covers_the_baseline_store():
    if not STORE.exists():
        return
    events = {f"{p.parts[-3]}/{p.parts[-2]}" for p in STORE.glob("*/*/Q.json")}
    missing = events - set(_table())
    assert not missing, sorted(missing)


def test_engine_labels_compounds():
    from app.services.engine import Engine
    tyres = Engine._load_tyres()
    assert tyres[(2024, "bahrain_grand_prix", "SOFT")] == 3.0
    assert tyres[(2025, "belgian_grand_prix", "MEDIUM")] == 3.0      # the skipped-step weekend
    assert tyres[(2025, "monaco_grand_prix", "SOFT")] == 6.0         # C6
