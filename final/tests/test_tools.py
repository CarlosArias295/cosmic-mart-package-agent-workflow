import json
import pytest

from final.cosmic_agents_final import (
    SegmentOrderTool,
    MakePackPlanTool,
    ValidatePackPlanTool,
    LogOutcomeTool,
)

def test_segment_order_fragile_electronics():
    tool = SegmentOrderTool()
    order = {
        "items": [
            {"sku": "SKU1", "qty": 1, "dims_cm": [10, 10, 10], "weight_kg": 1.0, "fragility": "HIGH", "category": "electronics"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }
    result = json.loads(tool.forward(json.dumps(order)))
    assert result["segment"] == "FRAGILE"

def test_segment_order_softgoods():
    tool = SegmentOrderTool()
    order = {
        "items": [
            {"sku": "SKU1", "qty": 1, "dims_cm": [10, 10, 2], "weight_kg": 0.3, "fragility": "LOW", "category": "apparel"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }
    result = json.loads(tool.forward(json.dumps(order)))
    assert result["segment"] == "SOFTGOODS"

def test_segment_order_restricted():
    tool = SegmentOrderTool()
    order = {
        "items": [
            {"sku": "BAT1", "qty": 1, "dims_cm": [8, 8, 4], "weight_kg": 0.5, "fragility": "LOW", "category": "battery"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }
    result = json.loads(tool.forward(json.dumps(order)))
    assert result["segment"] == "RESTRICTED"

def test_make_pack_plan_fragile_sets_foam():
    segment_tool = SegmentOrderTool()
    plan_tool = MakePackPlanTool()

    order = {
        "items": [
            {"sku": "SKU1", "qty": 1, "dims_cm": [18, 16, 8], "weight_kg": 0.9, "fragility": "HIGH", "category": "electronics"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }

    segment = segment_tool.forward(json.dumps(order))
    plan = json.loads(plan_tool.forward(json.dumps(order), segment))

    assert plan["segment"] == "FRAGILE"
    assert plan["containers"][0]["filler"] == "foam"
    assert "fragile" in plan["containers"][0]["labels"]

def test_make_pack_plan_restricted_forces_split():
    plan_tool = MakePackPlanTool()
    order = {
        "items": [
            {"sku": "BAT1", "qty": 1, "dims_cm": [8, 8, 4], "weight_kg": 0.5, "fragility": "LOW", "category": "battery"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }
    segment = json.dumps({"segment": "RESTRICTED"})
    plan = json.loads(plan_tool.forward(json.dumps(order), segment))

    assert plan["split_shipment"] is True
    assert plan["containers"][0]["filler"] == "air_pillow"

def test_make_pack_plan_empty_items_raises():
    plan_tool = MakePackPlanTool()
    order = {"items": [], "shipping": {"service": "standard", "zone": "regional"}}
    segment = json.dumps({"segment": "GENERAL"})

    with pytest.raises(ValueError):
        plan_tool.forward(json.dumps(order), segment)

def test_validate_plan_approve():
    tool = ValidatePackPlanTool()
    order = {
        "items": [
            {"sku": "SKU1", "qty": 1, "dims_cm": [10, 10, 10], "weight_kg": 1.0, "fragility": "LOW", "category": "general"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }
    plan = {
        "segment": "GENERAL",
        "containers": [{"box_id": "BX-S"}],
        "split_shipment": False
    }

    result = json.loads(tool.forward(json.dumps(order), json.dumps(plan)))
    assert result["decision"] == "APPROVE"

def test_validate_plan_overweight_revise():
    tool = ValidatePackPlanTool()
    order = {
        "items": [
            {"sku": "SKU1", "qty": 1, "dims_cm": [10, 10, 10], "weight_kg": 3.5, "fragility": "LOW", "category": "general"}
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }
    plan = {
        "segment": "GENERAL",
        "containers": [{"box_id": "BX-S"}],
        "split_shipment": False
    }

    result = json.loads(tool.forward(json.dumps(order), json.dumps(plan)))
    assert result["decision"] == "REVISE"
    assert any("exceeds box max" in trigger for trigger in result["triggers"])
