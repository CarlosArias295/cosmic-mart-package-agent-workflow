from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv


from smolagents import CodeAgent, OpenAIServerModel, Tool, ToolCallingAgent
from smolagents.memory import ActionStep

load_dotenv()

openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id = "gpt-4.1-mini",
    api_base = "https://openai.vocareum.com/v1",
    api_key = openai_api_key
)
# -----------------------------
# Tiny in-memory "catalogs"
# -----------------------------
BOX_CATALOG = [
    # id, inner dims (L,W,H) in cm, max_weight_kg, material_cost
    {"box_id": "BX-S", "dims_cm": (20, 15, 10), "max_w_kg": 2.0, "cost": 0.60},
    {"box_id": "BX-M", "dims_cm": (35, 25, 15), "max_w_kg": 6.0, "cost": 0.95},
    {"box_id": "BX-L", "dims_cm": (45, 35, 20), "max_w_kg": 12.0, "cost": 1.40},
    {"box_id": "BX-XL", "dims_cm": (60, 40, 30), "max_w_kg": 20.0, "cost": 2.10},
]

FILLER_CATALOG = [
    # type, cost per "unit", protection score
    {"filler": "paper", "unit_cost": 0.08, "protect": 1},
    {"filler": "air_pillow", "unit_cost": 0.12, "protect": 2},
    {"filler": "foam", "unit_cost": 0.20, "protect": 3},
]


# -----------------------------
# Helpers (deterministic, no LLM)
# -----------------------------
def volume_cm3(dims_cm: Tuple[float, float, float]) -> float:
    l, w, h = dims_cm
    return float(l) * float(w) * float(h)

def fits_in_box(item_dims_cm: Tuple[float, float, float], box_dims_cm: Tuple[float, float, float]) -> bool:
    # allow simple rotation: sort dims
    i = sorted(item_dims_cm)
    b = sorted(box_dims_cm)
    return all(i[k] <= b[k] for k in range(3))

def dim_weight_kg(box_dims_cm: Tuple[float, float, float], dim_divisor: float = 5000.0) -> float:
    # common proxy: (L*W*H)/dim_divisor, cm-based
    return volume_cm3(box_dims_cm) / dim_divisor

def choose_smallest_box(total_item_w_kg: float, largest_item_dims_cm: Tuple[float, float, float]) -> Dict[str, Any]:
    # pick the smallest box that (a) fits largest item and (b) supports weight
    candidates = []
    for b in BOX_CATALOG:
        if b["max_w_kg"] >= total_item_w_kg and fits_in_box(largest_item_dims_cm, b["dims_cm"]):
            candidates.append(b)
    if not candidates:
        # fallback to biggest
        return BOX_CATALOG[-1]
    candidates.sort(key=lambda x: volume_cm3(x["dims_cm"]))
    return candidates[0]

def damage_risk_proxy(fragility: str, filler_protect: int, total_w_kg: float) -> float:
    # crude: fragility drives baseline; heavier items raise risk; better filler reduces risk
    base = {"LOW": 0.10, "MED": 0.25, "HIGH": 0.45}.get(fragility.upper(), 0.25)
    weight_bump = min(0.20, 0.02 * total_w_kg)
    protect_drop = 0.05 * (filler_protect - 1)  # paper 0, air -0.05, foam -0.10
    return max(0.02, base + weight_bump - protect_drop)

def shipping_cost_proxy(billable_w_kg: float, zone: str, service: str) -> float:
    # simple curve: base + per-kg; service multiplies; zone multiplies
    zone_mult = {"local": 1.0, "regional": 1.15, "national": 1.30}.get(zone, 1.15)
    svc_mult = {"standard": 1.0, "expedited": 1.35}.get(service, 1.0)
    return (3.00 + 1.10 * billable_w_kg) * zone_mult * svc_mult


# -----------------------------
# Tools (these are what LLM can call)
# smolagents Tool pattern: name/description/inputs/output_type + forward()
# -----------------------------
class SegmentOrderTool(Tool):
    name = "segment_order"
    description = "Classify an order into a simple product segment based on SKU categories and fragility."
    inputs = {
        "order_json": {
            "type": "string",
            "description": "JSON string with fields: items=[{sku, qty, dims_cm:[L,W,H], weight_kg, fragility, category}], shipping={service, zone}"
        }
    }
    output_type = "string"

    def forward(self, order_json: str) -> str:
        order = json.loads(order_json)
        items = order["items"]

        # quick rule-based segmenting
        categories = [it.get("category", "misc") for it in items for _ in range(int(it.get("qty", 1)))]
        any_high_frag = any(str(it.get("fragility", "MED")).upper() == "HIGH" for it in items)

        if any(c in ("electronics", "glass") for c in categories) or any_high_frag:
            seg = "FRAGILE"
        elif any(c in ("apparel", "softgoods") for c in categories):
            seg = "SOFTGOODS"
        elif any(c in ("books", "media") for c in categories):
            seg = "FLAT"
        elif any(c in ("liquid", "battery") for c in categories):
            seg = "RESTRICTED"
        else:
            seg = "GENERAL"

        return json.dumps({"segment": seg, "categories": sorted(set(categories))})


class MakePackPlanTool(Tool):
    name = "make_pack_plan"
    description = "Generate a simple packaging plan: choose a box, filler, optional split, and estimate costs."
    inputs = {
        "order_json": {"type": "string", "description": "Same JSON string as in segment_order."},
        "segment_json": {"type": "string", "description": "JSON output from segment_order."}
    }
    output_type = "string"

    def forward(self, order_json: str, segment_json: str) -> str:
        order = json.loads(order_json)
        seg = json.loads(segment_json)["segment"]

        items = order["items"]
        zone = order["shipping"].get("zone", "regional")
        service = order["shipping"].get("service", "standard")

        # Expand quantities
        expanded = []
        for it in items:
            for _ in range(int(it.get("qty", 1))):
                expanded.append(it)

        total_w = sum(float(it["weight_kg"]) for it in expanded)
        largest = max(expanded, key=lambda x: volume_cm3(tuple(x["dims_cm"])))
        largest_dims = tuple(largest["dims_cm"])

        # Segment-driven defaults
        if seg == "FRAGILE":
            filler = next(f for f in FILLER_CATALOG if f["filler"] == "foam")
            split = total_w > 10.0  # crude rule
        elif seg == "SOFTGOODS":
            filler = next(f for f in FILLER_CATALOG if f["filler"] == "paper")
            split = False
        elif seg == "FLAT":
            filler = next(f for f in FILLER_CATALOG if f["filler"] == "paper")
            split = False
        elif seg == "RESTRICTED":
            filler = next(f for f in FILLER_CATALOG if f["filler"] == "air_pillow")
            split = True  # play safe
        else:
            filler = next(f for f in FILLER_CATALOG if f["filler"] == "air_pillow")
            split = total_w > 15.0

        # Box choice
        box = choose_smallest_box(total_w, largest_dims)
        # --- Volume efficiency + "optimal" explanation ---
        item_volume_total = sum(volume_cm3(tuple(it["dims_cm"])) for it in expanded)
        chosen_box_volume = volume_cm3(box["dims_cm"])
        void_cm3 = max(0.0, chosen_box_volume - item_volume_total)
        void_pct = (void_cm3 / chosen_box_volume) if chosen_box_volume > 0 else 0.0

        alt_box = next_larger_box(box, total_w, largest_dims)
        savings = None
        rationale = None

        if alt_box is not None:
            print(f"DEBUG: Chosen box {box['box_id']} vs alt {alt_box['box_id']}")
            alt_vol = volume_cm3(alt_box["dims_cm"])
            space_saved_cm3 = alt_vol - chosen_box_volume
            space_saved_pct = (space_saved_cm3 / alt_vol) if alt_vol > 0 else 0.0

            material_saved = float(alt_box["cost"]) - float(box["cost"])

            # shipping proxy difference due to dimensional weight
            chosen_dim_w = dim_weight_kg(box["dims_cm"])
            alt_dim_w = dim_weight_kg(alt_box["dims_cm"])
            chosen_billable = max(total_w, chosen_dim_w)
            alt_billable = max(total_w, alt_dim_w)

            chosen_ship = shipping_cost_proxy(chosen_billable, zone=zone, service=service)
            alt_ship = shipping_cost_proxy(alt_billable, zone=zone, service=service)
            shipping_saved_proxy = alt_ship - chosen_ship

            savings = {
                "compared_to_box_id": alt_box["box_id"],
                "space_saved_cm3": round(space_saved_cm3, 0),
                "space_saved_pct": round(100.0 * space_saved_pct, 1),
                "material_saved_usd": round(material_saved, 2),
                "shipping_saved_proxy_usd": round(shipping_saved_proxy, 2),
                "chosen_void_cm3": round(void_cm3, 0),
                "chosen_void_pct": round(100.0 * void_pct, 1),
                "item_volume_cm3": round(item_volume_total, 0),
                "chosen_box_volume_cm3": round(chosen_box_volume, 0),
            }

            rationale = (
                f"Selected {box['box_id']} as the smallest box that fits the largest item and total weight. "
                f"Compared to the next size up ({alt_box['box_id']}), it reduces box volume by "
                f"{round(space_saved_cm3,0):,.0f} cm³ ({round(100.0*space_saved_pct,1)}% smaller), "
                f"saving about ${round(material_saved,2)} in box material cost and "
                f"${round(shipping_saved_proxy,2)} in shipping proxy due to lower dimensional weight. "
                f"Estimated void space in {box['box_id']}: {round(100.0*void_pct,1)}%."
            )
        else:
            rationale = (
                f"Selected {box['box_id']} as the smallest available box that fits the largest item and total weight. "
                f"No larger comparable box candidate was found for a savings comparison. "
                f"Estimated void space: {round(100.0*void_pct,1)}%."
            )

        # Filler amount heuristic (units)
        # assume filler needed ~ 10% of box volume / 1000 as "units"
        filler_units = max(1, int(0.10 * volume_cm3(box["dims_cm"]) / 1000.0))

        # Costs
        material_cost = float(box["cost"]) + filler_units * float(filler["unit_cost"])
        actual_w = total_w
        dim_w = dim_weight_kg(box["dims_cm"])
        billable_w = max(actual_w, dim_w)
        ship_cost = shipping_cost_proxy(billable_w, zone=zone, service=service)

        # Damage proxy
        worst_frag = "LOW"
        for it in expanded:
            f = str(it.get("fragility", "MED")).upper()
            if f == "HIGH":
                worst_frag = "HIGH"; break
            if f == "MED":
                worst_frag = "MED"
        risk = damage_risk_proxy(worst_frag, filler["protect"], total_w)
        expected_return_cost = 25.0 * risk  # proxy

        plan = {
            "segment": seg,
            "containers": [{
                "box_id": box["box_id"],
                "dims_cm": box["dims_cm"],
                "filler": filler["filler"],
                "filler_units": filler_units,
                "labels": ["fragile"] if seg == "FRAGILE" else []
            }],
            "split_shipment": bool(split),
            "packing_instructions": [
                "Place largest item at bottom; distribute weight evenly.",
                "Fill void space; ensure items do not rattle.",
                "Seal with 2 strips of tape along main seam."
            ] + (["Apply 'FRAGILE' label on 2 sides."] if seg == "FRAGILE" else []),
            "optimality_rationale": rationale,
            "savings_estimate": savings,
            "cost_breakdown": {
                "material_cost": round(material_cost, 2),
                "shipping_cost_proxy": round(ship_cost, 2),
                "expected_damage_return_proxy": round(expected_return_cost, 2),
                "total_proxy": round(material_cost + ship_cost + expected_return_cost, 2),
                "billable_weight_kg": round(billable_w, 2),
                "actual_weight_kg": round(actual_w, 2),
                "dim_weight_kg": round(dim_w, 2)
            }
        }
        return json.dumps(plan)


class ValidatePackPlanTool(Tool):
    name = "validate_plan"
    description = "Validate the packaging plan against simple guardrails. Returns APPROVE/REVISE/ESCALATE."
    inputs = {
        "order_json": {"type": "string", "description": "Order JSON string."},
        "plan_json": {"type": "string", "description": "Plan JSON string from make_pack_plan."},
    }
    output_type = "string"

    def forward(self, order_json: str, plan_json: str) -> str:
        order = json.loads(order_json)
        plan = json.loads(plan_json)

        items = order["items"]
        expanded = []
        for it in items:
            for _ in range(int(it.get("qty", 1))):
                expanded.append(it)

        total_w = sum(float(it["weight_kg"]) for it in expanded)
        seg = plan.get("segment", "GENERAL")

        box_id = plan["containers"][0]["box_id"]
        box = next(b for b in BOX_CATALOG if b["box_id"] == box_id)

        triggers = []
        decision = "APPROVE"

        # Guardrail 1: weight must be <= box max
        if total_w > float(box["max_w_kg"]):
            triggers.append(f"Total weight {total_w:.2f}kg exceeds box max {box['max_w_kg']:.2f}kg.")
            decision = "REVISE"

        # Guardrail 2: fragile + heavy => split or escalate
        any_fragile = any(str(it.get("fragility", "MED")).upper() == "HIGH" for it in expanded)
        if any_fragile and total_w > 12.0 and not plan.get("split_shipment", False):
            triggers.append("Fragile + heavy detected; require split shipment.")
            decision = "REVISE"

        # Guardrail 3: restricted items => escalate if no split
        categories = [it.get("category", "misc") for it in expanded]
        if any(c in ("battery", "liquid") for c in categories) and not plan.get("split_shipment", False):
            triggers.append("Restricted category present (battery/liquid); requires special handling.")
            decision = "ESCALATE"

        report = {
            "decision": decision,
            "triggers": triggers,
            "suggested_fixes": []
        }

        if decision == "REVISE":
            # minimal suggested fixes
            if any("exceeds box max" in t for t in triggers):
                report["suggested_fixes"].append("Choose next larger box or split into 2 containers.")
            if any("require split" in t for t in triggers):
                report["suggested_fixes"].append("Enable split_shipment=true and separate fragile items.")
        if decision == "ESCALATE":
            report["suggested_fixes"].append("Route to human pack lead for restricted-item SOP.")

        return json.dumps(report)


class LogOutcomeTool(Tool):
    name = "log_outcome"
    description = "Log shipment outcome data (stub)."
    inputs = {
        "order_id": {"type": "string", "description": "Order id."},
        "decision": {"type": "string", "description": "Validator decision."},
        "notes": {"type": "string", "description": "Any notes/outcomes."}
    }
    output_type = "string"

    def forward(self, order_id: str, decision: str, notes: str) -> str:
        # For now: just echo back as a record (replace with DB/file later)
        return json.dumps({"logged": True, "order_id": order_id, "decision": decision, "notes": notes})


# -----------------------------
# Agents (smolagents CodeAgent)
# -----------------------------

def build_agents():
    model = OpenAIServerModel(
        model_id="gpt-4.1-mini",
        api_base="https://openai.vocareum.com/v1",
        api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    )

    segmenter = ToolCallingAgent(
        tools=[SegmentOrderTool()],
        model=model,
        instructions=(
            "You MUST call the tool `segment_order` exactly once. "
            "Do NOT answer from your own knowledge. "
            "After the tool returns, output EXACTLY the tool output string and nothing else."
        ),
        return_full_result=True,   # <— important
        add_base_tools=False,
    )

    pack_planner = ToolCallingAgent(
        tools=[MakePackPlanTool()],
        model=model,
        instructions=(
            "You MUST call the tool `make_pack_plan` exactly once. "
            "Use the segment from the previous step to inform your plan. "
            "Do NOT answer from your own knowledge. "
            "After the tool returns, output EXACTLY the tool output string and nothing else."
        ),
        return_full_result=True,   # <— important
        add_base_tools=False,
    )

    validator = ToolCallingAgent(
        tools=[ValidatePackPlanTool()],
        model=model,
        instructions=(
            "You MUST call the tool `validate_plan` exactly once. "
            "Use the pack plan from the previous step to inform your validation. "
            "Do NOT answer from your own knowledge. "
            "After the tool returns, output EXACTLY the tool output string and nothing else."
        ),
        return_full_result=True,   # <— important
        add_base_tools=False,
    )

    learner = ToolCallingAgent(
        tools=[LogOutcomeTool()],
        model=model,
        instructions=(
            "You MUST call the tool `log_outcome` exactly once. "
            "Do NOT answer from your own knowledge. "
            "After the tool returns, output EXACTLY the tool output string and nothing else."
        ),
        return_full_result=True,
        add_base_tools=False,
    )


    return segmenter, pack_planner, validator, learner


# -----------------------------
# Orchestration (simple + deterministic)
# -----------------------------
#HELPERS 

def next_larger_box(chosen_box: Dict[str, Any], total_item_w_kg: float, largest_item_dims_cm: Tuple[float, float, float]) -> Dict[str, Any] | None:
    # find boxes that also fit + support weight, strictly larger by volume
    chosen_vol = volume_cm3(chosen_box["dims_cm"])
    candidates = []
    for b in BOX_CATALOG:
        if b["max_w_kg"] >= total_item_w_kg and fits_in_box(largest_item_dims_cm, b["dims_cm"]):
            if volume_cm3(b["dims_cm"]) > chosen_vol:
                candidates.append(b)
    if not candidates:
        return None
    candidates.sort(key=lambda x: volume_cm3(x["dims_cm"]))
    return candidates[0]

#need this helper to get proper JSON 
def last_json_observation(agent) -> str:
    """
    Return the most recent ActionStep observation that is valid JSON.
    This skips non-JSON observations (e.g., final_answer returning 'FRAGILE').
    """
    for step in reversed(agent.memory.steps):
        if isinstance(step, ActionStep) and step.observations:
            obs = step.observations
            obs_list = obs if isinstance(obs, list) else [obs]

            for candidate in reversed(obs_list):
                if not isinstance(candidate, str):
                    continue
                s = candidate.strip()

                # Skip obvious non-JSON quick
                if not s or s[0] not in "{[":
                    continue

                try:
                    json.loads(s)
                    return s
                except json.JSONDecodeError:
                    continue

    raise RuntimeError("No JSON observation found in agent memory")

def run_pipeline(order: Dict[str, Any]):
    segmenter, pack_planner, validator, learner = build_agents()
    order_json = json.dumps(order)

    # 1) Segment
    segmenter.run(
        f"Call segment_order with order_json={order_json!r}. Return the tool output."
    )
    segment_json = last_json_observation(segmenter)
    segment = json.loads(segment_json)

    # 2) Pack plan
    pack_planner.run(
        f"Call make_pack_plan with order_json={order_json!r} and segment_json={segment_json!r}. Return the tool output."
    )
    plan_json = last_json_observation(pack_planner)
    plan = json.loads(plan_json)

    # 3) Validate
    validator.run(
        f"Call validate_plan with order_json={order_json!r} and plan_json={plan_json!r}. Return the tool output."
    )
    validation_json = last_json_observation(validator)
    validation = json.loads(validation_json)

    # 4) Log
    learner.run(
        f"Call log_outcome with order_id={order.get('order_id','unknown')!r}, "
        f"decision={validation.get('decision','UNKNOWN')!r}, notes={'initial run'!r}. Return the tool output."
    )
    log_json = last_json_observation(learner)
    log = json.loads(log_json)

    return {
        "segment": segment,
        "plan": plan,
        "validation": validation,
        "log": log,
    }

if __name__ == "__main__":
    # Example order
    example_order = {
        "order_id": "ORD-1001",
        "items": [
            {"sku": "SKU-HEADPHONES", "qty": 1, "dims_cm": [18, 16, 8], "weight_kg": 0.9, "fragility": "HIGH", "category": "electronics"},
            {"sku": "SKU-CABLE", "qty": 2, "dims_cm": [10, 6, 2], "weight_kg": 0.1, "fragility": "LOW", "category": "electronics"},
        ],
        "shipping": {"service": "standard", "zone": "regional"}
    }

    out = run_pipeline(example_order)
    print(json.dumps(out, indent=2))