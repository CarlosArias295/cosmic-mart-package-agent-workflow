from __future__ import annotations

import os, json, math
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv

from smolagents import Tool, ToolCallingAgent, OpenAIServerModel
from smolagents.memory import ActionStep

# ============================================================
# ENV + MODEL
# ============================================================

load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4.1-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
)

# ============================================================
# POLICY (PERSISTENT, LEARNED)
# ============================================================

POLICY_PATH = "pack_policy.json"

#The system is adjusted to use a JSON file for policy persistence, allowing it to learn and adapt over time based on feedback. The default policy is defined in code but can be updated by the FeedbackAnalyticsTool and saved back to the JSON file for future runs.
DEFAULT_POLICY = {
    "split_thresholds_kg": {
        "FRAGILE": 10.0,
        "GENERAL": 15.0,
        "RESTRICTED": 0.0,
        "SOFTGOODS": 999.0,
        "FLAT": 999.0,
    },
    "weight_safety_buffer_pct": 0.0, # This buffer is added to the total weight when checking against box max weight, allowing for a safety margin. The FeedbackAnalyticsTool can increase this buffer if it detects too many REVISE decisions due to weight issues, effectively making the agent more conservative in its box choices.
    "filler_by_segment": {
        "FRAGILE": "foam",
        "GENERAL": "air_pillow",
        "RESTRICTED": "air_pillow",
        "SOFTGOODS": "paper",
        "FLAT": "paper",
    },
}

def load_policy():
    if os.path.exists(POLICY_PATH):
        try:
            with open(POLICY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_POLICY))

def save_policy(policy):
    with open(POLICY_PATH, "w") as f:
        json.dump(policy, f, indent=2)

POLICY = load_policy()

# ============================================================
# CATALOGS
# ============================================================

BOX_CATALOG = [
    {"box_id": "BX-XS", "dims_cm": (15, 12,  6), "max_w_kg":  1.0, "cost": 0.45},
    {"box_id": "BX-S",  "dims_cm": (20, 15, 10), "max_w_kg":  2.0, "cost": 0.60},
    {"box_id": "BX-F",  "dims_cm": (30, 22,  6), "max_w_kg":  3.0, "cost": 0.75},
    {"box_id": "BX-M",  "dims_cm": (35, 25, 15), "max_w_kg":  6.0, "cost": 0.95},
    {"box_id": "BX-T",  "dims_cm": (35, 25, 25), "max_w_kg":  8.0, "cost": 1.25},
    {"box_id": "BX-L",  "dims_cm": (45, 35, 20), "max_w_kg": 12.0, "cost": 1.40},
    {"box_id": "BX-XL", "dims_cm": (60, 40, 30), "max_w_kg": 20.0, "cost": 2.10},
]

FILLER_CATALOG = [
    {"filler": "paper",     "unit_cost": 0.08, "protect": 1},
    {"filler": "air_pillow","unit_cost": 0.12, "protect": 2},
    {"filler": "foam",      "unit_cost": 0.20, "protect": 3},
]

# ============================================================
# HELPERS (core refined behavior stays the same)
# ============================================================

def volume_cm3(d: Tuple[float, float, float]) -> float:
    return float(d[0] * d[1] * d[2])

def fits(item_dims: Tuple[float, float, float], box_dims: Tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(sorted(item_dims), sorted(box_dims)))

def dim_weight(dims_cm: Tuple[float, float, float], div: float = 5000.0) -> float:
    return volume_cm3(dims_cm) / div

#MOST IMPORTANT FUNCTION: Selects the smallest viable box that meets constraints
def choose_box(total_w: float, largest_dims: Tuple[float, float, float]) -> Dict[str, Any]:
    # Preserve refined behavior: policy-aware effective weight + smallest viable box
    eff_w = total_w * (1 + POLICY["weight_safety_buffer_pct"])
    viable = [
        b for b in BOX_CATALOG
        if b["max_w_kg"] >= eff_w and fits(largest_dims, b["dims_cm"])
    ]
    return min(viable, key=lambda b: volume_cm3(b["dims_cm"])) if viable else BOX_CATALOG[-1]

# ---- Additive optimizer helpers (do NOT change decisions; only compute extra fields) ----

def next_larger_viable_box(
    chosen_box: Dict[str, Any],
    total_w: float,
    largest_dims: Tuple[float, float, float],
) -> Optional[Dict[str, Any]]:
    """Next larger (by volume) box that is also viable under current POLICY constraints."""
    eff_w = total_w * (1 + POLICY["weight_safety_buffer_pct"])
    chosen_vol = volume_cm3(chosen_box["dims_cm"])
    candidates = [
        b for b in BOX_CATALOG
        if b["max_w_kg"] >= eff_w
        and fits(largest_dims, b["dims_cm"])
        and volume_cm3(b["dims_cm"]) > chosen_vol
    ]
    return min(candidates, key=lambda b: volume_cm3(b["dims_cm"])) if candidates else None
#This function computes a proxy for expected damage risk based on fragility, filler protection level, and total weight.
def damage_risk_proxy(fragility: str, filler_protect: int, total_w_kg: float) -> float:
    base = {"LOW": 0.10, "MED": 0.25, "HIGH": 0.45}.get(fragility.upper(), 0.25)
    weight_bump = min(0.20, 0.02 * float(total_w_kg))
    protect_drop = 0.05 * (int(filler_protect) - 1)  # paper 0, air -0.05, foam -0.10
    return max(0.02, base + weight_bump - protect_drop)
#This function computes a proxy for shipping cost based on billable weight, zone, and service level. It uses a simple formula with multipliers to reflect how these factors typically influence shipping costs.
def shipping_cost_proxy(billable_w_kg: float, zone: str, service: str) -> float:
    zone_mult = {"local": 1.0, "regional": 1.15, "national": 1.30}.get(zone, 1.15)
    svc_mult  = {"standard": 1.0, "expedited": 1.35}.get(service, 1.0)
    return (3.00 + 1.10 * float(billable_w_kg)) * zone_mult * svc_mult

# ============================================================
# TOOLS
# ============================================================

#Segment order tool classifies the order into a segment based on item categories and fragility, which then informs packing decisions. It also returns the categories present for additive enrichment.
class SegmentOrderTool(Tool):
    name = "segment_order"
    description = "Classify an order into a segment such as FRAGILE, FLAT, SOFTGOODS, RESTRICTED, or GENERAL."
    inputs = {
        "order_json": {
            "type": "string",
            "description": "JSON string with fields: items=[{sku, qty, dims_cm:[L,W,H], weight_kg, fragility, category}], shipping={service, zone}",
        }
    }
    output_type = "string"

    def forward(self, order_json: str) -> str:
        order = json.loads(order_json)
        items = [i for it in order["items"] for i in [it] * int(it["qty"])]
        cats  = [i.get("category", "misc") for i in items]
        frag  = any(str(i.get("fragility", "MED")).upper() == "HIGH" for i in items)

        if frag or "electronics" in cats:
            seg = "FRAGILE"
        elif "books" in cats:
            seg = "FLAT"
        elif "battery" in cats or "liquid" in cats:
            seg = "RESTRICTED"
        elif "apparel" in cats:
            seg = "SOFTGOODS"
        else:
            seg = "GENERAL"

        # Additive enrichment (doesn't affect downstream logic)
        return json.dumps({"segment": seg, "categories": sorted(set(cats))})

#MakePackPlanTool creates a packing plan for the order based on the segment and order details. It selects the box, filler, and whether to split, while also computing detailed rationale and cost/savings estimates for additive enrichment.
class MakePackPlanTool(Tool):
    name = "make_pack_plan"
    description = "Create a packing plan for the given order and segment. Specifies box, filler, and whether to split."
    inputs = {
        "order_json": {
            "type": "string",
            "description": "JSON string with fields: items=[{sku, qty, dims_cm:[L,W,H], weight_kg, fragility, category}], shipping={service, zone}",
        },
        "segment_json": {
            "type": "string",
            "description": "JSON string with the segment information",
        },
    }
    output_type = "string"

    # This tool's core decision logic remains focused on selecting the smallest viable box under current policy constraints, but it now also computes a set of additional fields for rationale and cost/savings estimation without changing the fundamental decisions. The rationale explains why the box was chosen and what the trade-offs are compared to the next viable option, while the cost breakdown provides a proxy for material, shipping, and expected damage costs.
    def forward(self, order_json: str, segment_json: str) -> str:
        order = json.loads(order_json)
        seg   = json.loads(segment_json)["segment"]

        items = [i for it in order["items"] for i in [it] * int(it["qty"])]
        total_w = float(sum(i["weight_kg"] for i in items))
        largest = max(items, key=lambda x: volume_cm3(tuple(x["dims_cm"])))
        largest_dims = tuple(largest["dims_cm"])

        # Preserve refined behavior: policy-aware smallest viable box + policy thresholds for split
        box = choose_box(total_w, largest_dims)
        split = (seg == "RESTRICTED") or (total_w > POLICY["split_thresholds_kg"].get(seg, 15.0))

        filler_name = POLICY["filler_by_segment"].get(seg, "air_pillow")
        filler = next(f for f in FILLER_CATALOG if f["filler"] == filler_name)

        # ---- Additive optimizer/cost outputs (like cosmic_agents.py), no decision changes ----
        zone    = order.get("shipping", {}).get("zone", "regional")
        service = order.get("shipping", {}).get("service", "standard")

        item_volume_total = sum(volume_cm3(tuple(i["dims_cm"])) for i in items)
        chosen_box_volume = volume_cm3(box["dims_cm"])
        void_cm3 = max(0.0, chosen_box_volume - item_volume_total)
        void_pct = (void_cm3 / chosen_box_volume) if chosen_box_volume > 0 else 0.0

        alt_box = next_larger_viable_box(box, total_w, largest_dims)
        savings = None

        chosen_dim_w = dim_weight(box["dims_cm"])
        chosen_billable = max(total_w, chosen_dim_w)
        chosen_ship = shipping_cost_proxy(chosen_billable, zone=zone, service=service)

        if alt_box is not None:
            alt_vol = volume_cm3(alt_box["dims_cm"])
            space_saved_cm3 = alt_vol - chosen_box_volume
            space_saved_pct = (space_saved_cm3 / alt_vol) if alt_vol > 0 else 0.0

            material_saved = float(alt_box["cost"]) - float(box["cost"])

            alt_dim_w = dim_weight(alt_box["dims_cm"])
            alt_billable = max(total_w, alt_dim_w)
            alt_ship = shipping_cost_proxy(alt_billable, zone=zone, service=service)
            shipping_saved_proxy = alt_ship - chosen_ship

            #Does not affect box choice, but provides a detailed rationale and savings estimate for the chosen box compared to the next viable option, which can be used for analysis and future learning.
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
                f"Selected {box['box_id']} as the smallest viable box under current policy constraints "
                f"(weight buffer={POLICY['weight_safety_buffer_pct']:.0%}) that fits the largest item and total weight. "
                f"Compared to the next viable size up ({alt_box['box_id']}), it reduces box volume by "
                f"{round(space_saved_cm3, 0):,.0f} cm³ ({round(100.0 * space_saved_pct, 1)}% smaller), "
                f"saving about ${round(material_saved, 2)} in box material cost and "
                f"${round(shipping_saved_proxy, 2)} in shipping proxy due to lower dimensional weight. "
                f"Estimated void space in {box['box_id']}: {round(100.0 * void_pct, 1)}%."
            )
        else:
            rationale = (
                f"Selected {box['box_id']} as the smallest available viable box that fits the largest item and total weight "
                f"under current policy constraints (weight buffer={POLICY['weight_safety_buffer_pct']:.0%}). "
                f"No larger viable box candidate was found for a savings comparison. "
                f"Estimated void space: {round(100.0 * void_pct, 1)}%."
            )

        filler_units = max(1, int(0.10 * chosen_box_volume / 1000.0))
        labels = ["fragile"] if seg == "FRAGILE" else []

        packing_instructions = [
            "Place largest item at bottom; distribute weight evenly.",
            "Fill void space; ensure items do not rattle.",
            "Seal with 2 strips of tape along main seam.",
        ] + (["Apply 'FRAGILE' label on 2 sides."] if seg == "FRAGILE" else [])

        material_cost = float(box["cost"]) + filler_units * float(filler["unit_cost"])
        actual_w = float(total_w)
        billable_w = float(chosen_billable)

        # Expected damage proxy
        worst_frag = "LOW"
        for i in items:
            f = str(i.get("fragility", "MED")).upper()
            if f == "HIGH":
                worst_frag = "HIGH"
                break
            if f == "MED":
                worst_frag = "MED"
        risk = damage_risk_proxy(worst_frag, filler["protect"], actual_w)
        expected_return_cost = 25.0 * risk

        plan = {
            "segment": seg,
            "containers": [{
                "box_id": box["box_id"],
                "dims_cm": box["dims_cm"],
                "filler": filler["filler"],
                "filler_units": filler_units,
                "labels": labels,
            }],
            "split_shipment": bool(split),
            "packing_instructions": packing_instructions,
            "optimality_rationale": rationale,
            "savings_estimate": savings,
            "cost_breakdown": {
                "material_cost": round(material_cost, 2),
                "shipping_cost_proxy": round(chosen_ship, 2),
                "expected_damage_return_proxy": round(expected_return_cost, 2),
                "total_proxy": round(material_cost + chosen_ship + expected_return_cost, 2),
                "billable_weight_kg": round(billable_w, 2),
                "actual_weight_kg": round(actual_w, 2),
                "dim_weight_kg": round(float(chosen_dim_w), 2),
                "zone": zone,
                "service": service,
            },
        }
        return json.dumps(plan)


#Validates the proposed packing plan against order details and simple guardrails. It checks if the total weight exceeds the box max weight (considering the safety buffer) or if fragile+heavy items are packed without splitting, and returns a decision along with triggers and suggested fixes for any issues found.
class ValidatePackPlanTool(Tool):
    name = "validate_plan"
    description = "Validate the proposed packing plan against order details and simple guardrails."
    inputs = {
        "order_json": {
            "type": "string",
            "description": "JSON string with fields: items=[{sku, qty, dims_cm:[L,W,H], weight_kg, fragility, category}], shipping={service, zone}",
        },
        "plan_json": {
            "type": "string",
            "description": "JSON string with the packing plan information",
        },
    }
    output_type = "string"

    def forward(self, order_json: str, plan_json: str) -> str:
        order = json.loads(order_json)
        plan  = json.loads(plan_json)

        items = [i for it in order["items"] for i in [it] * int(it["qty"])]
        total_w = float(sum(i["weight_kg"] for i in items))

        box_id = plan["containers"][0]["box_id"]
        box = next(b for b in BOX_CATALOG if b["box_id"] == box_id)

        triggers: List[str] = []
        decision = "APPROVE"

        if total_w > float(box["max_w_kg"]):
            triggers.append("exceeds box max")
            decision = "REVISE"

        if any(str(i.get("fragility", "MED")).upper() == "HIGH" for i in items) and total_w > 12.0 and not plan.get("split_shipment", False):
            triggers.append("fragile+heavy")
            decision = "REVISE"

        suggested_fixes: List[str] = []
        if decision == "REVISE":
            if "exceeds box max" in triggers:
                suggested_fixes.append("Choose next larger box or split into 2 containers.")
            if "fragile+heavy" in triggers:
                suggested_fixes.append("Enable split_shipment=true and separate fragile items.")

        return json.dumps({"decision": decision, "triggers": triggers, "suggested_fixes": suggested_fixes})

#Logs the outcome of the packing plan validation for future analysis and learning. It takes the order ID, the decision made (APPROVE or REVISE), and any additional notes about the outcome, and returns a JSON string that can be stored and analyzed by the FeedbackAnalyticsTool to identify patterns and suggest policy adjustments.
class LogOutcomeTool(Tool):
    name = "log_outcome"
    description = "Log the outcome of the packing plan validation for future analysis and learning."
    inputs = {
        "order_id": {"type": "string", "description": "The ID of the order"},
        "decision": {"type": "string", "description": "The decision made (APPROVE or REVISE)"},
        "notes": {"type": "string", "description": "Additional notes about the outcome"},
    }
    output_type = "string"

    def forward(self, order_id: str, decision: str, notes: str) -> str:
        return json.dumps({"order_id": order_id, "decision": decision, "notes": notes})

#FeedbackAnalyticsTool analyzes the logs of packing plan validation outcomes to identify patterns (e.g., frequent REVISE decisions due to weight issues) and suggests adjustments to the packing policy (e.g., increasing the weight safety buffer) to improve future decisions. It takes a JSON string of log entries, counts how many times REVISE was triggered, and if it exceeds a certain threshold, it updates the global POLICY and saves it back to the JSON file for persistence.
class FeedbackAnalyticsTool(Tool):
    name = "feedback_analytics"
    description = "Analyze logs to identify patterns and suggest policy adjustments."
    inputs = {
        "logs_json": {
            "type": "string",
            "description": "JSON string list of log objects returned by log_outcome.",
        }
    }
    output_type = "string"

    def forward(self, logs_json: str) -> str:
        global POLICY
        logs = json.loads(logs_json)
        revise_hits = sum(1 for l in logs if l["decision"] == "REVISE")

        # Preserve refined behavior: only adjust when REVISE appears
        if revise_hits:
            POLICY["split_thresholds_kg"]["FRAGILE"] = max(
                4.0, POLICY["split_thresholds_kg"]["FRAGILE"] - 2.0
            )
            POLICY["weight_safety_buffer_pct"] = min(
                0.2, POLICY["weight_safety_buffer_pct"] + 0.05
            )
            save_policy(POLICY)

        return json.dumps({"policy_updated": POLICY})

# ============================================================
# AGENT BUILDER
# ============================================================

def agent(tools):
    return ToolCallingAgent(
        tools=tools,
        model=model,
        return_full_result=True,
        add_base_tools=False,
        instructions="You MUST call the tool exactly once and return ONLY the tool output.",
    )

# ============================================================
# MEMORY JSON EXTRACTION
# ============================================================

def last_json(agent_obj) -> str:
    for step in reversed(agent_obj.memory.steps):
        if isinstance(step, ActionStep):
            obs = step.observations
            obs_list = [obs] if isinstance(obs, str) else obs
            for c in obs_list:
                try:
                    json.loads(c)
                    return c
                except Exception:
                    pass
    raise RuntimeError("No JSON found")

# ============================================================
# PIPELINE
# ============================================================

#Runs the full pipeline for a given order, sequentially invoking the segmentation, packing plan creation, validation, and logging tools. It extracts the relevant JSON outputs at each step for use in the next step and for final analysis. 
def run_pipeline(order: Dict[str, Any]):
    seg_agent = agent([SegmentOrderTool()])
    seg_agent.run(f"segment_order(order_json={json.dumps(order)!r})")
    seg_j = last_json(seg_agent)

    plan_agent = agent([MakePackPlanTool()])
    plan_agent.run(f"make_pack_plan(order_json={json.dumps(order)!r},segment_json={seg_j!r})")
    plan_j = last_json(plan_agent)

    val_agent = agent([ValidatePackPlanTool()])
    val_agent.run(f"validate_plan(order_json={json.dumps(order)!r},plan_json={plan_j!r})")
    val_j = last_json(val_agent)

    v = json.loads(val_j)
    notes = " | ".join(v.get("triggers", [])) if v.get("triggers") else "no triggers"

    log_agent = agent([LogOutcomeTool()])
    log_agent.run(f"log_outcome(order_id={order['order_id']!r},decision={v['decision']!r},notes={notes!r})")
    log_j = last_json(log_agent)

    return json.loads(seg_j), json.loads(plan_j), json.loads(val_j), json.loads(log_j)

# ============================================================
# PRETTY PRINT (clean summaries)
# ============================================================

def _money(x: Any) -> str:
    try:
        return f"${float(x):.2f}"
    except Exception:
        return str(x)

def print_run_summary(order: Dict[str, Any], seg: Dict[str, Any], plan: Dict[str, Any], val: Dict[str, Any], log: Dict[str, Any]) -> None:
    oid = order.get("order_id", "unknown")
    cats = seg.get("categories", [])
    segment = seg.get("segment", "UNKNOWN")

    c0 = plan["containers"][0]
    box_id = c0.get("box_id")
    filler = c0.get("filler")
    filler_units = c0.get("filler_units")
    labels = c0.get("labels", [])
    split = plan.get("split_shipment", False)

    cb = plan.get("cost_breakdown", {})
    total_proxy = cb.get("total_proxy")
    ship_proxy  = cb.get("shipping_cost_proxy")
    mat_cost    = cb.get("material_cost")
    dmg_proxy   = cb.get("expected_damage_return_proxy")

    print(f"\n=== ORDER {oid} ===")
    print(f"Segment: {segment}" + (f" | Categories: {', '.join(cats)}" if cats else ""))
    print(f"Plan: box={box_id} | filler={filler} (units={filler_units}) | split={split} | labels={labels}")

    savings = plan.get("savings_estimate")
    if savings:
        print(
            "Savings (vs {compared_to_box_id}): material {material_saved_usd} | ship-proxy {shipping_saved_proxy_usd} | "
            "size {space_saved_pct}% smaller | void {chosen_void_pct}%".format(
                compared_to_box_id=savings.get("compared_to_box_id"),
                material_saved_usd=_money(savings.get("material_saved_usd")),
                shipping_saved_proxy_usd=_money(savings.get("shipping_saved_proxy_usd")),
                space_saved_pct=savings.get("space_saved_pct"),
                chosen_void_pct=savings.get("chosen_void_pct"),
            )
        )
    else:
        print("Savings: (no larger viable box found for comparison)")

    if total_proxy is not None:
        print(f"Costs (proxy): total={_money(total_proxy)} | material={_money(mat_cost)} | shipping={_money(ship_proxy)} | damage={_money(dmg_proxy)}")

    decision = val.get("decision")
    triggers = val.get("triggers", [])
    fixes = val.get("suggested_fixes", [])
    print(f"Validate: {decision} | triggers={triggers if triggers else '[]'}")
    if fixes:
        print("Suggested fixes:")
        for fx in fixes:
            print(f"  - {fx}")

    # Keep notes output (refined style)
    print(f"Log: {log.get('decision')} | notes='{log.get('notes')}'")

# ============================================================
# TEST HARNESS (keeps refined policy/feedback prints + adds clean per-order summary)
# ============================================================

TEST_ORDERS = [
    # --- existing ---
    {
        "order_id":"T1",
        "items":[{"sku":"A","qty":1,"dims_cm":[10,8,4],"weight_kg":0.2,"fragility":"HIGH","category":"electronics"}],
        "shipping":{}
    },
    {
        "order_id":"T2",
        "items":[{"sku":"B","qty":1,"dims_cm":[55,35,10],"weight_kg":7.5,"fragility":"HIGH","category":"electronics"}],
        "shipping":{}
    },
    {
        "order_id": "TF-01",
        "items": [
            {"sku": "SKU-TV","qty":1,"dims_cm":[55,35,8],"weight_kg":13.5,"fragility":"HIGH","category":"electronics"}
        ],
        "shipping": {}
    },
    {
        "order_id": "TF-02",
        "items": [
            {"sku": "SKU-WEIGHTSET","qty":1,"dims_cm":[35,25,15],"weight_kg":7.5,"fragility":"LOW","category":"sports"}
        ],
        "shipping": {}
    },

    # 1) GENERAL, light, should APPROVE; tests "regional/standard" default behavior
    {
        "order_id": "NEW-GEN-01",
        "items": [
            {"sku": "SKU-MUG", "qty": 2, "dims_cm": [10, 10, 10], "weight_kg": 0.3, "fragility": "MED", "category": "home"}
        ],
        "shipping": {}
    },

    # 2) SOFTGOODS, bulk apparel, should pick paper filler + likely APPROVE
    {
        "order_id": "NEW-SOFT-01",
        "items": [
            {"sku": "SKU-HOODIE", "qty": 3, "dims_cm": [30, 25, 5], "weight_kg": 0.6, "fragility": "LOW", "category": "apparel"}
        ],
        "shipping": {"zone": "local", "service": "standard"}
    },

    # 3) FLAT (books), tests segmenting -> FLAT and paper filler
    {
        "order_id": "NEW-FLAT-01",
        "items": [
            {"sku": "SKU-BOOK", "qty": 4, "dims_cm": [24, 16, 3], "weight_kg": 0.4, "fragility": "LOW", "category": "books"}
        ],
        "shipping": {"zone": "national", "service": "standard"}
    },

    # 4) RESTRICTED (battery), forces split_shipment=True by policy rule (seg == RESTRICTED)
    {
        "order_id": "NEW-REST-01",
        "items": [
            {"sku": "SKU-BATTERY", "qty": 1, "dims_cm": [12, 8, 6], "weight_kg": 0.9, "fragility": "MED", "category": "battery"},
            {"sku": "SKU-CABLE", "qty": 2, "dims_cm": [10, 6, 2], "weight_kg": 0.1, "fragility": "LOW", "category": "electronics"},
        ],
        "shipping": {"zone": "regional", "service": "expedited"}
    },

    # 5) FRAGILE + heavy but BELOW box max; should REVISE if split_shipment=False and total > 12
    # This targets the validator trigger "fragile+heavy".
    # Use dims that fit BX-L/BX-XL; weight around 13 triggers fragile+heavy rule.
    {
        "order_id": "NEW-FRAG-HEAVY-01",
        "items": [
            {"sku": "SKU-GLASS-SET", "qty": 1, "dims_cm": [40, 30, 18], "weight_kg": 13.0, "fragility": "HIGH", "category": "electronics"}
        ],
        "shipping": {"zone": "national", "service": "standard"}
    },

    # 6) Exceeds chosen box max -> REVISE via "exceeds box max"
    # This is designed so it fits physically, but total weight likely exceeds smaller box max.
    {
        "order_id": "NEW-OVERMAX-01",
        "items": [
            {"sku": "SKU-DENSE-BLOCK", "qty": 1, "dims_cm": [34, 24, 14], "weight_kg": 7.0, "fragility": "LOW", "category": "sports"}
        ],
        "shipping": {"zone": "regional", "service": "standard"}
    },

    # 7) Multi-item mixed: electronics present -> FRAGILE segment; tests qty expansion + largest item logic
    {
        "order_id": "NEW-MIX-01",
        "items": [
            {"sku": "SKU-HEADSET", "qty": 1, "dims_cm": [20, 18, 10], "weight_kg": 0.8, "fragility": "MED", "category": "electronics"},
            {"sku": "SKU-SHIRT", "qty": 2, "dims_cm": [28, 22, 3], "weight_kg": 0.3, "fragility": "LOW", "category": "apparel"},
            {"sku": "SKU-NOTEBOOK", "qty": 1, "dims_cm": [21, 14, 2], "weight_kg": 0.2, "fragility": "LOW", "category": "books"},
        ],
        "shipping": {"zone": "local", "service": "expedited"}
    },

    # 8) GENERAL large but light; tests dimensional weight effect (dim_weight dominates actual weight)
    {
        "order_id": "NEW-DIM-01",
        "items": [
            {"sku": "SKU-PILLOW", "qty": 1, "dims_cm": [55, 40, 25], "weight_kg": 1.2, "fragility": "LOW", "category": "home"}
        ],
        "shipping": {"zone": "national", "service": "standard"}
    },
]

if __name__ == "__main__":
    print("POLICY BEFORE:", json.dumps(POLICY, indent=2))

    logs = []
    for o in TEST_ORDERS:
        seg, plan, val, log = run_pipeline(o)
        logs.append(log)

        # old minimal line (kept)
        print(o["order_id"], val["decision"], log["notes"])

        # new clean summary
        print_run_summary(o, seg, plan, val, log)

    fb = agent([FeedbackAnalyticsTool()])
    fb.run(f"feedback_analytics(logs_json={json.dumps(logs)!r})")
    fb_j = last_json(fb)
    print("\nFEEDBACK:", fb_j)

    print("POLICY AFTER:", json.dumps(load_policy(), indent=2))