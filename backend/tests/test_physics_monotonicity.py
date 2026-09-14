"""Phase 3 gate — the physics layer must be directionally correct before the
ML layer is allowed to consume it."""
import pytest

from pipeline.physics import modifiers


@pytest.mark.xfail(reason="Phase 3 — not implemented yet", raises=NotImplementedError)
def test_more_rear_wing_increases_downforce_and_drag():
    low = modifiers.wing_to_aero(0.5, 0.2)
    high = modifiers.wing_to_aero(0.5, 0.8)
    assert high.downforce_pct > low.downforce_pct
    assert high.drag_pct > low.drag_pct


@pytest.mark.xfail(reason="Phase 3 — not implemented yet", raises=NotImplementedError)
def test_tyre_grip_peaks_inside_temperature_window():
    cold = modifiers.tyre_thermal_grip(15.0, "soft", 1)
    ideal = modifiers.tyre_thermal_grip(38.0, "soft", 1)
    hot = modifiers.tyre_thermal_grip(58.0, "soft", 1)
    assert ideal > cold and ideal > hot
