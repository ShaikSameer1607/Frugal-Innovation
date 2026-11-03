"""
smart_portions.py
=================
Smart Portion Recommendation & Ladle Calibration System for MessMind AI.
Developed for Frugal Innovation course at IIIT Dharwad.

Key Capabilities:
1. Dish-wise plate waste analysis using actual physically measured scale weights from Food Vision AI scans.
2. Identifies dishes with consistently high leftover quantities.
3. Recommends suitable initial serving portions (Smart Ladle Policy) based on consumption patterns.
4. Provides an optional smaller initial serving with an easy second-serving option.
5. Displays dish name, current portion, suggested portion, historical waste, and expected benefit.
6. Enforces minimum historical data requirement (>= 3 records) before generating recommendations.
7. Strictly NEVER uses image surface coverage as food weight.
8. Strictly NEVER reduces kitchen cooking quantities or compromises food availability.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import database as db
from database import SERVING_WEIGHT_KG, DEFAULT_DB_PATH

# -----------------------------------------------------------------------------
# 1. STANDARD PORTION SIZES (Grams per Adult Serving)
# -----------------------------------------------------------------------------
# Based on IIIT Dharwad hostel mess catering standards:
# - Curries / Dals: ~180-200g (standard 200ml kitchen ladle)
# - Rice & Biryani: ~220-250g (standard deep grain scoop)
# - Breads: ~80g (2 Chapatis) / 90g (3 Puris)
# - Breakfast: ~140-160g
# - Snacks: ~140-160g
# - Accompaniments: ~60-100g
STANDARD_PORTIONS: Dict[str, int] = {
    # Main Curries, Dals & Gravies
    "Kadai Paneer": 200, "Paneer Kolhapuri": 200, "Shahi Paneer": 200, "Paneer Butter Masala": 200,
    "Dal Methi": 200, "Dal Tadka": 200, "Dal Makhani": 200, "Dal Fry": 200, "Dal Palak": 200,
    "Dal Pancha Ratan": 200, "Moong Dal": 200, "Masoor Dal": 200, "Channa Dal": 200, "Dal Kolhapuri": 200,
    "Dal": 200, "Rasam": 150, "Brinjal Bhartha": 180, "White Peas Masala": 180, "Jeera Aloo": 180,
    "Chawli Masala": 180, "Brinjal Tomato": 180, "Kadhi Pakoda": 200, "Bhendi Sabji": 170,
    "Veg Kadai": 180, "Ivy Gourd Fry": 160, "Cabbage Baji": 160, "Black Chana Masala": 180,
    "Aloo 65": 160, "Soyabean Sabzi Gravy": 180, "Sorakay Chana": 180, "Veg Kolhapuri": 180,
    "Aloo Mutter Sabzi": 180, "Egg Curry": 220,

    # Rice & Grains
    "Curd Rice": 220, "Veg Pulav": 220, "Onion Rice": 220, "Puliogare": 220, "Jeera Rice": 220,
    "Tomato Rice": 220, "Plain Rice": 220, "Khichadi": 220, "Soyabean Biryani": 240,
    "Veg & Egg Biryani": 250, "Vangi Bath": 220,

    # Breads & Tiffins
    "Chapati": 80, "Puri": 90, "Chole Bhature": 180, "Aloo Paratha": 140, "Masala Dosa": 160,
    "Idli": 140, "Vada": 120, "Upma": 160, "Pongal": 180, "Poha": 150, "Uthappam": 160,
    "Sheera": 140, "Egg Bhurji": 150, "Boiled Egg": 60,

    # Snacks
    "Samosa": 120, "Pani Puri": 150, "Punugulu": 140, "Veg Noodles": 160, "Pasta": 160,
    "Masala Bhel": 140, "Girmit": 140,

    # Accompaniments & Sweets
    "Curd": 100, "Mix Veg Raita": 100, "Papad": 20, "Pickle": 15, "Salad": 60,
    "Boiled Groundnut": 80, "Boiled Sweet Corn": 80, "Matki Sprouts": 60,
    "Green Moong Sprouts": 60, "Black Chana Sprouts": 60, "Chutney": 40, "Coconut Chutney": 40,
    "Motichoor Ladoo": 50, "Gulab Jamun": 60, "Jalebi": 60, "Ice Cream": 80, "Shevai Kheer": 100
}

DEFAULT_STANDARD_PORTION = 180  # grams


def get_standard_portion(dish_name: str) -> int:
    """Retrieve standard baseline initial serving portion in grams for a dish."""
    if not dish_name:
        return DEFAULT_STANDARD_PORTION
    clean_name = dish_name.strip().lower()
    for k, v in STANDARD_PORTIONS.items():
        if k.strip().lower() == clean_name:
            return v
    if "paneer" in clean_name:
        return 200
    if "dal" in clean_name:
        return 200
    if "rice" in clean_name or "biryani" in clean_name or "pulav" in clean_name:
        return 220
    return DEFAULT_STANDARD_PORTION


# -----------------------------------------------------------------------------
# 2. DISH-WISE PLATE WASTE ANALYSIS (Strictly Physical Measurements)
# -----------------------------------------------------------------------------
def analyze_dish_plate_waste(
    dish_name: str,
    min_records: int = 3,
    lookback_limit: int = 50,
    data_source_filter: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Analyze dish-wise plate waste strictly using actual physically measured scale weights.
    Never uses 2D image surface coverage as food weight.
    Strictly separates verified REAL physical weights from simulated DEMO readings.
    Never presents simulated scale readings as actual measured food weights.

    Args:
        dish_name: Name of the dish to analyze.
        min_records: Minimum required records for statistical significance (default 3).
        lookback_limit: Maximum recent records to inspect.
        data_source_filter: 'REAL' (only verified scale logs), 'DEMO' (only simulated demo), or None (prefer REAL).
        db_path: SQLite database file path.

    Returns:
        dict containing record count, average waste in grams, waste ratio, data source, and demo flag.
    """
    std_portion = get_standard_portion(dish_name)
    conn = db.get_connection(db_path)
    real_weights: List[float] = []
    demo_weights: List[float] = []

    try:
        cursor = conn.cursor()
        # 1. Primary Source: Food Vision AI scans with physically weighed scale measurements
        cursor.execute("""
            SELECT measured_weight_grams, leftover_category, scan_timestamp, data_source, is_simulated_weight
            FROM plate_waste_scans
            WHERE dish_name = ? AND measured_weight_grams IS NOT NULL AND measured_weight_grams > 0
            ORDER BY scan_timestamp DESC
            LIMIT ?
        """, (dish_name, lookback_limit))
        scan_rows = cursor.fetchall()

        for r in scan_rows:
            w = float(r["measured_weight_grams"])
            d_src = r["data_source"] if "data_source" in r.keys() else "REAL"
            is_sim = r["is_simulated_weight"] if "is_simulated_weight" in r.keys() else 0
            if d_src == "REAL" and (is_sim == 0 or is_sim is None):
                real_weights.append(w)
            else:
                demo_weights.append(w)

        # 2. Complementary Source: Historical kitchen scale meal logs
        cursor.execute("""
            SELECT quantity_prepared, quantity_served, discarded_quantity, leftover_quantity, data_source
            FROM meal_records
            WHERE dish_name = ? AND quantity_served > 0
            ORDER BY meal_date DESC
            LIMIT ?
        """, (dish_name, lookback_limit))
        meal_rows = cursor.fetchall()

        for m in meal_rows:
            disc_rate = float(m["discarded_quantity"]) / max(1.0, float(m["quantity_served"]))
            calc_g = disc_rate * std_portion
            if calc_g > 0:
                m_src = m["data_source"] if "data_source" in m.keys() else "REAL"
                if m_src == "REAL":
                    real_weights.append(round(calc_g, 1))
                else:
                    demo_weights.append(round(calc_g, 1))
    finally:
        conn.close()

    # Determine which weights to evaluate based on data_source_filter
    if data_source_filter == "REAL":
        selected_weights = real_weights
        is_demo = False
        source = "REAL (Verified Kitchen Scale Measurements)" if real_weights else "No Real Scale Records"
    elif data_source_filter in ("DEMO", "SYNTHETIC"):
        selected_weights = demo_weights
        is_demo = True
        source = "DEMO (Simulated Demonstration Readings — NOT Real Scale Weights)"
    else:
        # Default / ALL: Prefer REAL if sufficient, else fallback to DEMO with explicit disclaimer
        if len(real_weights) >= min_records:
            selected_weights = real_weights
            is_demo = False
            source = "REAL (Verified Kitchen Scale Measurements)"
        elif len(demo_weights) >= min_records:
            selected_weights = demo_weights
            is_demo = True
            source = "DEMO (Simulated Demonstration Readings — NOT Real Scale Weights)"
        else:
            selected_weights = real_weights if real_weights else demo_weights
            is_demo = (len(real_weights) == 0 and len(demo_weights) > 0)
            source = "DEMO (Simulated Demonstration Readings — NOT Real Scale Weights)" if is_demo else "No Data"

    total_records = len(selected_weights)
    is_sufficient = total_records >= min_records

    if not is_sufficient:
        return {
            "dish_name": dish_name,
            "is_sufficient_data": False,
            "is_demo": is_demo,
            "is_simulated_weight": is_demo,
            "record_count": total_records,
            "min_required": min_records,
            "avg_waste_grams": 0.0,
            "min_waste_grams": 0.0,
            "max_waste_grams": 0.0,
            "std_portion_grams": std_portion,
            "waste_ratio": 0.0,
            "data_source": source,
            "demo_notice": "⚠️ DEMO DATA NOTICE: Simulated demo data is active." if is_demo else None,
            "raw_measurements": selected_weights
        }

    avg_waste = float(np.mean(selected_weights))
    min_waste = float(np.min(selected_weights))
    max_waste = float(np.max(selected_weights))
    waste_ratio = avg_waste / max(1.0, float(std_portion))

    demo_warning = (
        "⚠️ DEMO DATA NOTICE: Derived from simulated baseline readings. Never treat as actual measured food weights."
        if is_demo else None
    )

    return {
        "dish_name": dish_name,
        "is_sufficient_data": True,
        "is_demo": is_demo,
        "is_simulated_weight": is_demo,
        "record_count": total_records,
        "min_required": min_records,
        "avg_waste_grams": round(avg_waste, 1),
        "min_waste_grams": round(min_waste, 1),
        "max_waste_grams": round(max_waste, 1),
        "std_portion_grams": std_portion,
        "waste_ratio": round(waste_ratio, 3),
        "data_source": source,
        "demo_notice": demo_warning,
        "raw_measurements": selected_weights
    }


# -----------------------------------------------------------------------------
# 3. SMART PORTION & LADLE CALIBRATION RECOMMENDATION
# -----------------------------------------------------------------------------
def recommend_dish_portion(
    dish_name: str,
    min_records: int = 3,
    max_reduction_pct: float = 25.0,
    data_source_filter: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Generate Smart Portion & Ladle Calibration recommendations for a dish.

    Portion Recommendation Limits:
    - Minimum portion: Standard portion minus max_reduction_pct (default max 25% trim, ensuring students always receive >= 75% standard initial serving).
    - Maximum portion: Standard portion (never reduced if waste is low, never exceeds standard portion on initial scoop).
    - Kitchen cooking targets are strictly NOT reduced.
    """
    waste_analysis = analyze_dish_plate_waste(dish_name, min_records=min_records, data_source_filter=data_source_filter, db_path=db_path)
    std_portion = waste_analysis["std_portion_grams"]
    is_sufficient = waste_analysis["is_sufficient_data"]

    # 1. Handling Insufficient History Scenario (Requirement 6)
    if not is_sufficient:
        return {
            "dish_name": dish_name,
            "is_sufficient_data": False,
            "status": "Insufficient Data",
            "current_portion_grams": std_portion,
            "suggested_portion_grams": std_portion,
            "portion_reduction_pct": 0.0,
            "avg_waste_grams": 0.0,
            "waste_level": "Unknown (Insufficient History)",
            "record_count": waste_analysis["record_count"],
            "min_required": min_records,
            "two_stage_policy": "Standard serving. Free 2nd helpings available.",
            "ladle_specification": f"Use standard ladle (~{std_portion}g portion).",
            "expected_benefit": "Maintain standard portion until at least 3 physical scale records are collected.",
            "recommendation_summary": f"Insufficient historical data ({waste_analysis['record_count']}/{min_records} records). Maintain standard portion.",
            "kitchen_prep_impact": "None. Kitchen preparation targets remain unchanged.",
            "data_source": waste_analysis["data_source"],
            "is_demo": waste_analysis.get("is_demo", False),
            "is_simulated_weight": waste_analysis.get("is_simulated_weight", False),
            "demo_notice": waste_analysis.get("demo_notice")
        }

    # 2. Sufficient History Scenario
    avg_waste = waste_analysis["avg_waste_grams"]
    waste_ratio = waste_analysis["waste_ratio"]

    # Determine Waste Level and Recommended Reduction Percentage
    # High Waste Dish: avg leftover >= 45g or waste_ratio >= 0.20 (20%+)
    # Moderate Waste Dish: 22g <= avg leftover < 45g or 0.10 <= waste_ratio < 0.20
    # Optimal/Low Waste Dish: avg leftover < 22g and waste_ratio < 0.10
    if avg_waste >= 45.0 or waste_ratio >= 0.20:
        waste_level = "High Leftover"
        raw_trim = (avg_waste / std_portion) * 100.0
        reduction_pct = min(max_reduction_pct, max(15.0, round(raw_trim, 0)))
        status = "Ladle Downsizing Recommended"
    elif avg_waste >= 22.0 or waste_ratio >= 0.10:
        waste_level = "Moderate Leftover"
        raw_trim = (avg_waste / std_portion) * 100.0
        reduction_pct = min(15.0, max(8.0, round(raw_trim, 0)))
        status = "Moderate Calibration Recommended"
    else:
        waste_level = "Optimal Consumption"
        reduction_pct = 0.0
        status = "Standard Portion Confirmed"

    # Enforce Safe Portion Bounds (Portion recommendation limits)
    # Never reduce below (1 - max_reduction_pct / 100) * std_portion
    min_allowed_portion = int(round(std_portion * (1.0 - (max_reduction_pct / 100.0))))
    max_allowed_portion = std_portion

    calculated_portion = int(round(std_portion * (1.0 - (reduction_pct / 100.0)) / 5.0) * 5)
    suggested_portion = max(min_allowed_portion, min(max_allowed_portion, calculated_portion))
    actual_reduction_pct = round(((std_portion - suggested_portion) / std_portion) * 100.0, 1)

    # Ladle Specification
    if actual_reduction_pct >= 20.0:
        ladle_spec = f"Switch from standard ladle ({std_portion}g) to calibrated 150ml ladle ({suggested_portion}g). First scoop only."
    elif actual_reduction_pct >= 10.0:
        ladle_spec = f"Serve level ladle scoop ({suggested_portion}g) instead of heaping scoop ({std_portion}g)."
    else:
        ladle_spec = f"Continue standard ladle scoop ({std_portion}g)."

    # Two-Stage Serving Policy (Requirement 4)
    two_stage_policy = (
        f"Smart Two-Stage Serving: Initial ladle = {suggested_portion}g (calibrated). "
        f"Easy second helpings freely available at the refill counter for hungry students."
    )

    # Expected Benefit (Requirement 5)
    saved_grams_per_diner = round(std_portion - suggested_portion, 0)
    dish_cost = 18.0
    monthly_kg_saved = round((saved_grams_per_diner * 400 * 8) / 1000.0, 1)  # ~8 shifts per month
    monthly_cost_saved = round(monthly_kg_saved * (dish_cost / 0.35), 0)

    if actual_reduction_pct > 0:
        benefit_text = (
            f"Prevents ~{saved_grams_per_diner:.0f}g wasted scrapings per diner. "
            f"Estimated campus savings: ~{monthly_kg_saved} kg food (~₹{monthly_cost_saved:,.0f}/month) "
            f"with zero student under-feeding via refill stations."
        )
    else:
        benefit_text = (
            "Current portion perfectly matches student consumption. Negligible plate residue (<10%)."
        )

    # Kitchen Target Independence (Requirement 8)
    kitchen_impact = (
        "Zero reduction to kitchen cooking batches. Full cooking target is maintained to guarantee "
        "abundant food availability for second helpings."
    )

    summary = (
        f"{dish_name}: Current {std_portion}g → Suggested {suggested_portion}g ({actual_reduction_pct:+.0f}%). "
        f"Historical waste: {avg_waste:.1f}g avg. {benefit_text}"
    )

    is_demo = waste_analysis.get("is_demo", False)
    if is_demo:
        status = f"[DEMO SIMULATION] {status}"
        summary = f"[DEMO DATA - SIMULATED ESTIMATE] {summary}"

    return {
        "dish_name": dish_name,
        "is_sufficient_data": True,
        "status": status,
        "current_portion_grams": std_portion,
        "suggested_portion_grams": suggested_portion,
        "portion_reduction_pct": actual_reduction_pct,
        "avg_waste_grams": avg_waste,
        "min_waste_grams": waste_analysis["min_waste_grams"],
        "max_waste_grams": waste_analysis["max_waste_grams"],
        "waste_level": waste_level,
        "record_count": waste_analysis["record_count"],
        "min_required": min_records,
        "two_stage_policy": two_stage_policy,
        "ladle_specification": ladle_spec,
        "expected_benefit": benefit_text,
        "recommendation_summary": summary,
        "kitchen_prep_impact": kitchen_impact,
        "data_source": waste_analysis["data_source"],
        "is_demo": waste_analysis.get("is_demo", False),
        "is_simulated_weight": waste_analysis.get("is_simulated_weight", False),
        "demo_notice": waste_analysis.get("demo_notice")
    }


# -----------------------------------------------------------------------------
# 4. ACTIVE MENU SMART PORTIONS ORCHESTRATOR
# -----------------------------------------------------------------------------
def get_menu_smart_portions(
    day_of_week: str,
    meal_period: str,
    data_source_filter: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve smart portion recommendations for all dishes in a given meal shift."""
    menu_info = db.get_menu_for_meal(day_of_week, meal_period)
    dishes = menu_info.get("dishes", [])
    if not dishes:
        dishes = ["Plain Rice", "Dal Fry"]

    recs = []
    for dish in dishes:
        rec = recommend_dish_portion(dish, data_source_filter=data_source_filter, db_path=db_path)
        recs.append(rec)
    return recs


def get_all_dishes_portion_summary(data_source_filter: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Analyze all known mess menu dishes and compile an operational portion sizing table."""
    all_dishes = list(STANDARD_PORTIONS.keys())
    results = []
    for d in all_dishes:
        rec = recommend_dish_portion(d, data_source_filter=data_source_filter, db_path=db_path)
        if rec["is_sufficient_data"]:
            results.append({
                "Dish Name": rec["dish_name"],
                "Waste Level": rec["waste_level"],
                "Current Portion": f"{rec['current_portion_grams']}g",
                "Suggested Portion": f"{rec['suggested_portion_grams']}g",
                "Portion Trim": f"-{rec['portion_reduction_pct']:.0f}%" if rec['portion_reduction_pct'] > 0 else "0% (Standard)",
                "Avg Leftover": f"{rec['avg_waste_grams']:.1f}g",
                "Data Source": "🟢 REAL" if not rec.get("is_demo") else "🟡 DEMO (Simulated)",
                "Records": f"{rec['record_count']} logs",
                "Two-Stage Policy": rec["two_stage_policy"],
                "Expected Benefit": rec["expected_benefit"]
            })
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    order_map = {"High Leftover": 0, "Moderate Leftover": 1, "Optimal Consumption": 2}
    df["sort_key"] = df["Waste Level"].map(order_map).fillna(3)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])
    return df

