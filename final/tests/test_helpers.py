import pytest

from final.cosmic_agents_final import (
    volume_cm3,
    fits,
    dim_weight,
    choose_box,
    damage_risk_proxy,
    shipping_cost_proxy,
    next_larger_viable_box,
    BOX_CATALOG,
    POLICY,
)

def test_volume_cm3_basic():
    assert volume_cm3((2, 3, 4)) == 24.0


def test_volume_cm3_zero_dimension():
    assert volume_cm3((0, 3, 4)) == 0.0


def test_volume_cm3_decimal_values():
    assert volume_cm3((2.5, 4.0, 2.0)) == 20.0


def test_fits_exact():
    assert fits((10, 20, 30), (10, 20, 30)) is True


def test_fits_with_rotation():
    assert fits((30, 10, 20), (20, 30, 10)) is True


def test_fits_false_when_one_dimension_too_large():
    assert fits((10, 20, 31), (10, 20, 30)) is False


def test_fits_zero_item():
    assert fits((0, 0, 0), (1, 1, 1)) is True


def test_dim_weight_basic():
    assert dim_weight((50, 40, 30)) == 12.0


def test_dim_weight_zero_volume():
    assert dim_weight((0, 10, 10)) == 0.0


def test_dim_weight_custom_divisor():
    assert dim_weight((50, 40, 30), div=6000.0) == 10.0


def test_choose_box_smallest_viable(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.0)
    box = choose_box(0.5, (10, 8, 4))
    assert box["box_id"] == "BX-XS"


def test_choose_box_weight_forces_upgrade(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.0)
    box = choose_box(1.5, (10, 8, 4))
    assert box["box_id"] == "BX-S"


def test_choose_box_flat_box_selected_for_flat_item(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.0)
    box = choose_box(2.5, (28, 20, 4))
    assert box["box_id"] == "BX-F"


def test_choose_box_fallback_to_largest_when_no_viable(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.0)
    box = choose_box(100.0, (100, 100, 100))
    assert box["box_id"] == "BX-XL"


def test_choose_box_respects_weight_safety_buffer(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.10)
    box = choose_box(1.95, (10, 8, 4))
    assert box["box_id"] != "BX-S"
    assert box["box_id"] == "BX-F"


def test_next_larger_viable_box_from_small(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.0)
    chosen = next(b for b in BOX_CATALOG if b["box_id"] == "BX-XS")
    nxt = next_larger_viable_box(chosen, 0.5, (10, 8, 4))
    assert nxt is not None
    assert volume_cm3(nxt["dims_cm"]) > volume_cm3(chosen["dims_cm"])


def test_next_larger_viable_box_none_for_xl(monkeypatch):
    monkeypatch.setitem(POLICY, "weight_safety_buffer_pct", 0.0)
    chosen = next(b for b in BOX_CATALOG if b["box_id"] == "BX-XL")
    nxt = next_larger_viable_box(chosen, 1.0, (10, 10, 10))
    assert nxt is None


def test_damage_risk_proxy_high_fragile_higher_than_low():
    low = damage_risk_proxy("LOW", 3, 1.0)
    high = damage_risk_proxy("HIGH", 1, 1.0)
    assert high > low


def test_damage_risk_proxy_unknown_fragility_uses_default():
    risk = damage_risk_proxy("UNKNOWN", 2, 5.0)
    assert risk > 0


def test_damage_risk_proxy_has_min_floor():
    risk = damage_risk_proxy("LOW", 10, 0.0)
    assert risk == 0.02


def test_shipping_cost_proxy_expedited_more_than_standard():
    standard = shipping_cost_proxy(5.0, "regional", "standard")
    expedited = shipping_cost_proxy(5.0, "regional", "expedited")
    assert expedited > standard


def test_shipping_cost_proxy_national_more_than_local():
    local = shipping_cost_proxy(5.0, "local", "standard")
    national = shipping_cost_proxy(5.0, "national", "standard")
    assert national > local


def test_shipping_cost_proxy_unknown_values_use_defaults():
    a = shipping_cost_proxy(5.0, "weird_zone", "weird_service")
    b = shipping_cost_proxy(5.0, "regional", "standard")
    assert a == b