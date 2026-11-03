"""
recommendations.py
==================
Smart Food Demand Prediction & Preparation Recommendation Engine for MessMind AI.
Combines student advance registration counts (Yes, No, Not sure intent breakdown),
strictly chronological historical demand, and the trained Scikit-learn RandomForest pipeline
to recommend optimal preparation quantities for official weekly menu items.
Includes configurable safety margins, fallback estimation, and nontechnical explanations.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from database import (
    DEFAULT_DB_PATH,
    get_connection,
    save_prediction,
    get_menu_for_meal,
    get_attendance_intent_counts,
    get_dish_waste_history,
    record_actual_meal_outcome,
    SERVING_WEIGHT_KG
)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "demand_model.joblib")

# Standard item ingredient cost per serving (INR) across official mess menu
DEFAULT_DISH_COSTS = {
    # Main Curries, Dals & Gravies
    "Kadai Paneer": 34.0, "Paneer Kolhapuri": 34.0, "Shahi Paneer": 34.0, "Paneer Butter Masala": 34.0,
    "Dal Methi": 14.0, "Dal Tadka": 14.0, "Dal Makhani": 16.0, "Dal Fry": 14.0, "Dal Palak": 14.0,
    "Dal Pancha Ratan": 14.0, "Moong Dal": 14.0, "Masoor Dal": 14.0, "Channa Dal": 14.0, "Dal Kolhapuri": 14.0,
    "Dal": 14.0, "Rasam": 10.0, "Brinjal Bhartha": 16.0, "White Peas Masala": 16.0, "Jeera Aloo": 14.0,
    "Chawli Masala": 15.0, "Brinjal Tomato": 15.0, "Kadhi Pakoda": 16.0, "Bhendi Sabji": 18.0,
    "Veg Kadai": 18.0, "Ivy Gourd Fry": 16.0, "Cabbage Baji": 14.0, "Black Chana Masala": 15.0,
    "Aloo 65": 16.0, "Soyabean Sabzi Gravy": 15.0, "Sorakay Chana": 15.0, "Veg Kolhapuri": 18.0,
    "Aloo Mutter Sabzi": 15.0, "Egg Curry": 22.0,
    # Rice & Biryani Varieties
    "Curd Rice": 11.0, "Veg Pulav": 14.0, "Onion Rice": 12.0, "Puliogare": 16.0, "Jeera Rice": 12.0,
    "Tomato Rice": 12.0, "Plain Rice": 9.0, "Khichadi": 12.0, "Soyabean Biryani": 14.0,
    "Veg & Egg Biryani": 22.0, "Vangi Bath": 14.0,
    # Breads & Breakfast Items
    "Chapati": 10.0, "Puri": 14.0, "Chole Bhature": 24.0, "Aloo Paratha": 18.0, "Masala Dosa": 20.0,
    "Idli": 12.0, "Vada": 14.0, "Upma": 10.0, "Pongal": 14.0, "Poha": 12.0, "Uthappam": 15.0,
    "Sheera": 15.0, "Egg Bhurji": 18.0, "Boiled Egg": 7.0,
    # Snacks & Evening Tiffin
    "Samosa": 12.0, "Pani Puri": 20.0, "Punugulu": 16.0, "Veg Noodles": 18.0, "Pasta": 18.0,
    "Masala Bhel": 15.0, "Girmit": 16.0,
    # Accompaniments, Sweets & Drinks
    "Curd": 10.0, "Mix Veg Raita": 12.0, "Ginger Tea": 6.0, "Tea": 6.0, "Coffee": 7.0,
    "Lassi": 12.0, "Buttermilk": 8.0, "Badam Milk": 14.0, "Lemonade": 8.0, "Jaljeera": 8.0, "Rasna": 8.0,
    "Papad": 3.0, "Pickle": 2.0, "Salad": 5.0, "Boiled Groundnut": 8.0, "Boiled Sweet Corn": 10.0,
    "Matki Sprouts": 8.0, "Green Moong Sprouts": 8.0, "Black Chana Sprouts": 8.0,
    "Chutney": 5.0, "Coconut Chutney": 5.0, "Motichoor Ladoo": 12.0, "Gulab Jamun": 14.0,
    "Jalebi": 14.0, "Ice Cream": 15.0, "Shevai Kheer": 16.0
}

# Baseline take-rate priors for cold-start fallbacks
FALLBACK_TAKE_RATES = {
    # Staples
    "Rice": 0.90,
    "Plain Rice": 0.90,
    "Curd Rice": 0.88,
    "Dal": 0.85,
    "Chapati": 0.80,
    "Chapati / Fulka": 0.80,
    "Puri": 0.88,
    "Vegetable Curry": 0.78,
    "Curd": 0.65,
    "Paneer Curry": 0.94,

    # Breakfast
    "Idli": 0.84,
    "Vada": 0.90,
    "Dosa": 0.95,
    "Masala Dosa": 0.96,
    "Uthappam": 0.84,
    "Upma": 0.65,
    "Pongal": 0.74,
    "Poha": 0.68,
    "Aloo Paratha": 0.92,
    "Vangi Bath": 0.76,
    "Sheera": 0.88,

    # Snacks
    "Samosa": 0.88,
    "Pani Puri": 0.92,
    "Punugulu": 0.82,
    "Veg Noodles": 0.86,
    "Pasta": 0.84,
    "Masala Bhel": 0.78,
    "Girmit": 0.80,
    "Ginger Tea": 0.75,

    # Specific Curries & Special items
    "Kadai Paneer": 0.95,
    "Paneer Kolhapuri": 0.95,
    "Shahi Paneer": 0.96,
    "Paneer Butter Masala": 0.96,
    "Chole Bhature": 0.94,
    "Veg Pulav": 0.92,
    "Veg & Egg Biryani": 0.94,
    "Soyabean Biryani": 0.86,
    "Tomato Rice": 0.85,
    "Onion Rice": 0.88,
    "Puliogare": 0.85,
    "Jeera Rice": 0.86,
    "Khichadi": 0.78,
    "Aloo 65": 0.90
}

# Dish category mapping for the official IIIT Dharwad menu to model training categories
DISH_CATEGORY_MAPPING = {
    # Rice varieties
    "Curd Rice": "Rice",
    "Veg Pulav": "Rice",
    "Puliogare": "Rice",
    "Onion Rice": "Rice",
    "Tomato Rice": "Rice",
    "Jeera Rice": "Rice",
    "Plain Rice": "Rice",
    "Rice": "Rice",
    "Khichadi": "Rice",
    "Soyabean Biryani": "Rice",
    "Vangi Bath": "Rice",

    # Dal varieties
    "Dal Methi": "Dal",
    "Dal Tadka": "Dal",
    "Dal Makhani": "Dal",
    "Dal Fry": "Dal",
    "Dal Palak": "Dal",
    "Dal Pancha ratan": "Dal",
    "Moong Dal": "Dal",
    "Masoor Dal": "Dal",
    "Channa Dal": "Dal",
    "Dal": "Dal",
    "Rasam": "Dal",

    # Paneer & high-demand curries
    "Kadai Paneer": "Paneer Curry",
    "Paneer Kholapuri": "Paneer Curry",
    "Shahi Paneer": "Paneer Curry",
    "Panner Butter Masala": "Paneer Curry",
    "Paneer Curry": "Paneer Curry",

    # Veg curries & gravies
    "Brinjal Bhartha": "Vegetable Curry",
    "White Pese Masala": "Vegetable Curry",
    "Jeera Aloo": "Vegetable Curry",
    "Chawli Masala": "Vegetable Curry",
    "Brinjal Tomato": "Vegetable Curry",
    "Kadhi Pakoda": "Vegetable Curry",
    "Bhendi Sabji": "Vegetable Curry",
    "Veg Kadai": "Vegetable Curry",
    "Ivy Gourd Fry": "Vegetable Curry",
    "Cabbage Baji": "Vegetable Curry",
    "Aloo 65": "Vegetable Curry",
    "Soyabean Sabzi Gravy": "Vegetable Curry",
    "Sorakay Chana": "Vegetable Curry",
    "Veg Kholapuri": "Vegetable Curry",
    "Aloo Mutter Sabzi": "Vegetable Curry",
    "Vegetable Curry": "Vegetable Curry",

    # Breads
    "Chapati / Fulka": "Chapati",
    "Chapati": "Chapati",
    "Puri": "Chapati",
    "Chole Bhature": "Chapati",
    "Aloo Paratha": "Chapati",
    "Aloo Partha": "Chapati",

    # Breakfast staples
    "Masala Dosa": "Dosa",
    "Dosa": "Dosa",
    "Uthappamm": "Idli",
    "Idli": "Idli",
    "Vada": "Idli",
    "Upma": "Upma",
    "Sewaiya Upma": "Upma",
    "Pongal": "Pongal",
    "Poha": "Pongal",

    # Dairy / Extras
    "Curd": "Curd",
    "Buttermilk": "Curd",
    "Lassi": "Curd"
}


def load_demand_model(model_path: str = DEFAULT_MODEL_PATH):
    """Safely load the trained joblib model pipeline."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    return None


def get_canonical_dish(dish_name: str) -> str:
    """Map an official menu dish to its nearest base category for ML inference."""
    return DISH_CATEGORY_MAPPING.get(dish_name.strip(), dish_name.strip())


class HistoricalDemandResult(float):
    """Dual float-and-tuple result object for bulletproof compatibility."""
    is_fallback: bool

    def __new__(cls, value: float, is_fallback: bool = False):
        obj = super().__new__(cls, float(value))
        obj.is_fallback = is_fallback
        return obj

    def __iter__(self):
        yield float(self)
        yield self.is_fallback

    def __getitem__(self, index):
        if index == 0:
            return float(self)
        elif index == 1:
            return self.is_fallback
        raise IndexError("HistoricalDemandResult index out of range")

    def __len__(self):
        return 2


def get_historical_demand_for_dish(
    meal_date: Optional[str] = None,
    meal_type: str = "Lunch",
    dish_name: Optional[str] = None,
    expected_attendance: int = 450,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> HistoricalDemandResult:
    """
    Compute historical demand using ONLY records chronologically strictly before meal_date.
    Checks for exact dish name first, then canonical category.
    Supports both (meal_date, meal_type, dish_name) AND (dish_name, meal_type) calling signatures.
    Returns HistoricalDemandResult (acts as both float and (value, is_fallback) tuple).
    """
    # Detect if meal_date is actually dish_name (e.g. called as get_historical_demand_for_dish(dish, meal_type))
    if meal_date is not None and dish_name is None:
        if not ("-" in str(meal_date) and len(str(meal_date)) >= 8):
            dish_name = meal_date
            meal_date = datetime.now().strftime("%Y-%m-%d")

    if dish_name is None:
        dish_name = kwargs.get("dish") or kwargs.get("dish_name") or "Plain Rice"
    if meal_date is None:
        meal_date = datetime.now().strftime("%Y-%m-%d")

    canonical = get_canonical_dish(dish_name)

    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            # 1. Try exact dish name
            cursor.execute("""
                SELECT quantity_served FROM meal_records
                WHERE dish_name = ? AND meal_type = ? AND meal_date < ?
                ORDER BY meal_date DESC LIMIT 14
            """, (dish_name, meal_type, meal_date))
            rows = cursor.fetchall()
            if rows and len(rows) >= 3:
                values = [r["quantity_served"] for r in rows]
                return HistoricalDemandResult(round(float(np.mean(values)), 1), False)

            # 2. Try canonical category (e.g. 'Rice', 'Dal', 'Paneer Curry')
            if canonical != dish_name:
                cursor.execute("""
                    SELECT quantity_served FROM meal_records
                    WHERE dish_name = ? AND meal_type = ? AND meal_date < ?
                    ORDER BY meal_date DESC LIMIT 14
                """, (canonical, meal_type, meal_date))
                rows_canon = cursor.fetchall()
                if rows_canon and len(rows_canon) >= 2:
                    values = [r["quantity_served"] for r in rows_canon]
                    return HistoricalDemandResult(round(float(np.mean(values)), 1), False)
    except Exception:
        pass

    # Fallback to configured domain priors
    take_rate = FALLBACK_TAKE_RATES.get(canonical, FALLBACK_TAKE_RATES.get(dish_name, 0.75))
    fallback_val = round(float(expected_attendance * take_rate), 1)
    return HistoricalDemandResult(fallback_val, True)


def get_polling_forecast_inputs(
    meal_date: str,
    meal_type: str,
    default_expected: int = 450,
    default_registered: int = 380,
    eligible_students: int = 450,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Fetch student attendance polling counts (Yes, No, Not sure) from SQLite.
    Calculate weighted attendance:
      Yes = 100%, Not Sure = 50%, No = 0%.
    If polling data exists:
      Populates registered_students with weighted_attendance and expected_attendance with eligible_students.
    If no polling data exists:
      Falls back to manual input defaults.
    """
    intent_counts = get_attendance_intent_counts(
        meal_date=meal_date,
        meal_type=meal_type,
        eligible_students=eligible_students,
        db_path=db_path
    )

    yes_cnt = int(intent_counts.get("yes_count", 0))
    no_cnt = int(intent_counts.get("no_count", 0))
    not_sure_cnt = int(intent_counts.get("not_sure_count", 0))
    total_responses = int(intent_counts.get("total_responses", 0))
    eligible = int(intent_counts.get("eligible_students", eligible_students))

    # Weighted attendance formula: Yes = 100%, Not Sure = 50%, No = 0%
    weighted_attendance = int(round(yes_cnt * 1.0 + not_sure_cnt * 0.50 + no_cnt * 0.0))

    has_polling = total_responses > 0

    if has_polling:
        registered_students = weighted_attendance
        expected_attendance = eligible
        data_source = "SQLite Student Polling"
    else:
        registered_students = default_registered
        expected_attendance = default_expected
        data_source = "Manual Fallback"

    return {
        "has_polling": has_polling,
        "yes_count": yes_cnt,
        "no_count": no_cnt,
        "not_sure_count": not_sure_cnt,
        "total_responses": total_responses,
        "eligible_students": eligible,
        "weighted_attendance": weighted_attendance,
        "registered_students": registered_students,
        "expected_attendance": expected_attendance,
        "data_source": data_source,
        "intent_breakdown": intent_counts
    }


def calculate_dish_waste_pattern(
    dish_name: str,
    meal_type: str,
    meal_date: Optional[str] = None,
    lookback_limit: int = 14,
    min_records: int = 3,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Analyze historical prepared, served, leftover, and discarded quantities from SQLite
    to identify persistent food waste patterns for a specific dish and meal period.
    Strictly considers records chronologically before meal_date to prevent data leakage.
    Requires at least min_records (default 3) to prevent erratic single-day overreactions.
    """
    clean_dish = dish_name.strip()
    records = get_dish_waste_history(
        dish_name=clean_dish,
        meal_type=meal_type,
        before_date=meal_date,
        limit=lookback_limit,
        db_path=db_path
    )

    # If insufficient history for exact dish name, check canonical category
    canonical_dish = get_canonical_dish(clean_dish)
    if len(records) < min_records and canonical_dish != clean_dish:
        cat_records = get_dish_waste_history(
            dish_name=canonical_dish,
            meal_type=meal_type,
            before_date=meal_date,
            limit=lookback_limit,
            db_path=db_path
        )
        if len(cat_records) >= min_records:
            records = cat_records

    sample_count = len(records)
    has_sufficient_history = sample_count >= min_records

    if not has_sufficient_history:
        return {
            "dish_name": clean_dish,
            "meal_type": meal_type,
            "sample_count": sample_count,
            "min_records_required": min_records,
            "has_sufficient_history": False,
            "avg_prepared": 0.0,
            "avg_served": 0.0,
            "avg_leftover": 0.0,
            "avg_discarded": 0.0,
            "avg_waste_pct": 0.0,
            "avg_leftover_pct": 0.0,
            "waste_trend": "Insufficient History",
            "waste_summary": f"Insufficient History ({sample_count}/{min_records} records)",
            "latest_discarded": records[0]["discarded_quantity"] if sample_count > 0 else 0,
            "latest_waste_pct": round(float(records[0]["discarded_quantity"] / max(records[0]["quantity_prepared"], 1) * 100), 1) if sample_count > 0 else 0.0,
            "records": records
        }

    prep_vals = [r["quantity_prepared"] for r in records]
    served_vals = [r["quantity_served"] for r in records]
    leftover_vals = [r["leftover_quantity"] for r in records]
    discarded_vals = [r["discarded_quantity"] for r in records]

    waste_pcts = [round(float(d / max(p, 1) * 100), 2) for d, p in zip(discarded_vals, prep_vals)]
    leftover_pcts = [round(float(l / max(p, 1) * 100), 2) for l, p in zip(leftover_vals, prep_vals)]

    avg_prepared = round(float(np.mean(prep_vals)), 1)
    avg_served = round(float(np.mean(served_vals)), 1)
    avg_leftover = round(float(np.mean(leftover_vals)), 1)
    avg_discarded = round(float(np.mean(discarded_vals)), 1)
    avg_waste_pct = round(float(np.mean(waste_pcts)), 1)
    avg_leftover_pct = round(float(np.mean(leftover_pcts)), 1)

    # Classify waste trend
    if avg_waste_pct >= 12.0:
        waste_trend = "High Waste"
    elif avg_waste_pct <= 3.5:
        waste_trend = "Low Waste"
    else:
        waste_trend = "Balanced Waste"

    waste_summary = f"{avg_waste_pct:.1f}% avg waste ({avg_discarded:.0f} serv discarded across {sample_count} meals)"

    return {
        "dish_name": clean_dish,
        "meal_type": meal_type,
        "sample_count": sample_count,
        "min_records_required": min_records,
        "has_sufficient_history": True,
        "avg_prepared": avg_prepared,
        "avg_served": avg_served,
        "avg_leftover": avg_leftover,
        "avg_discarded": avg_discarded,
        "avg_waste_pct": avg_waste_pct,
        "avg_leftover_pct": avg_leftover_pct,
        "waste_trend": waste_trend,
        "waste_summary": waste_summary,
        "latest_discarded": discarded_vals[0],
        "latest_waste_pct": waste_pcts[0],
        "records": records
    }


def compute_adaptive_waste_adjustment(
    predicted_demand: float,
    base_recommendation: int,
    waste_pattern: Dict[str, Any],
    min_records: int = 3,
    max_reduction_pct: float = 15.0,
    max_increase_pct: float = 10.0,
    target_waste_pct: float = 5.0,
    capacity_cap: int = 550
) -> Dict[str, Any]:
    """
    Adaptive food waste feedback loop.
    Adjusts base ML recommendations by learning from multi-day historical waste patterns.
    Rules:
      1. If insufficient history (< min_records), maintains base recommendation (0 adjustment).
      2. If high waste, dampens excess overpreparation while never reducing below predicted demand.
      3. If low waste, slightly boosts safety buffer to avert shortages.
      4. Limits are bounded by configurable max_reduction_pct and max_increase_pct.
    """
    if not waste_pattern.get("has_sufficient_history", False):
        return {
            "adjusted_preparation_quantity": base_recommendation,
            "adjustment_servings": 0,
            "adjustment_pct": 0.0,
            "feedback_status": "Insufficient History (Base ML Maintained)",
            "feedback_explanation": f"Insufficient historical records ({waste_pattern.get('sample_count', 0)}/{min_records}). Maintained base ML recommendation.",
            "bounded_by_floor": False
        }

    avg_waste_pct = waste_pattern.get("avg_waste_pct", 0.0)
    avg_leftover_pct = waste_pattern.get("avg_leftover_pct", 0.0)
    sample_count = waste_pattern.get("sample_count", 0)

    # Minimum safety floor: never drop below predicted demand so dining hall does not face shortage
    floor_prep = int(np.ceil(predicted_demand))

    if avg_waste_pct > target_waste_pct:
        # High waste: overpreparation detected across multiple past records
        excess_waste = avg_waste_pct - target_waste_pct
        # Dampen reduction: 0.6 factor applied to excess waste percentage, bounded by max_reduction_pct
        reduction_pct = min(max_reduction_pct, excess_waste * 0.6)
        raw_cut = int(round(base_recommendation * (reduction_pct / 100.0)))

        # Bounded adjustment: ensure adjusted_prep does not drop below floor_prep
        adjusted_prep = max(floor_prep, base_recommendation - raw_cut)
        adjusted_prep = int(np.clip(adjusted_prep, 10, capacity_cap))
        adjustment_servings = adjusted_prep - base_recommendation
        adjustment_pct = round((adjustment_servings / max(base_recommendation, 1)) * 100.0, 1)

        bounded_by_floor = (base_recommendation - raw_cut) < floor_prep

        floor_note = f" (bounded by predicted demand floor of {floor_prep} servings)" if bounded_by_floor else ""
        explanation = (
            f"High waste pattern detected (avg {avg_waste_pct:.1f}% discarded over {sample_count} past meals). "
            f"Adaptive loop trimmed preparation by {abs(adjustment_servings)} servings ({adjustment_pct:+.1f}%){floor_note} "
            f"to reduce waste while protecting diner volume."
        )

        return {
            "adjusted_preparation_quantity": adjusted_prep,
            "adjustment_servings": adjustment_servings,
            "adjustment_pct": adjustment_pct,
            "feedback_status": "Waste Dampened (High Historical Waste)",
            "feedback_explanation": explanation,
            "bounded_by_floor": bounded_by_floor
        }

    elif avg_waste_pct <= 2.5 and avg_leftover_pct <= 4.0:
        # Very low waste / stockout risk: plates licked clean, risk of shortage
        boost_pct = min(max_increase_pct, (2.5 - avg_waste_pct) * 1.5 + 2.0)
        raw_boost = int(round(base_recommendation * (boost_pct / 100.0)))
        adjusted_prep = min(capacity_cap, base_recommendation + max(1, raw_boost))
        adjustment_servings = adjusted_prep - base_recommendation
        adjustment_pct = round((adjustment_servings / max(base_recommendation, 1)) * 100.0, 1)

        explanation = (
            f"Low waste pattern detected (avg {avg_waste_pct:.1f}% discarded over {sample_count} past meals, high consumption). "
            f"Adaptive loop added +{adjustment_servings} protective buffer servings ({adjustment_pct:+.1f}%) to safeguard against food shortages."
        )

        return {
            "adjusted_preparation_quantity": adjusted_prep,
            "adjustment_servings": adjustment_servings,
            "adjustment_pct": adjustment_pct,
            "feedback_status": "Buffer Boosted (Low Waste / High Consumption)",
            "feedback_explanation": explanation,
            "bounded_by_floor": False
        }

    else:
        # Balanced / normal waste
        return {
            "adjusted_preparation_quantity": base_recommendation,
            "adjustment_servings": 0,
            "adjustment_pct": 0.0,
            "feedback_status": "Balanced (Optimal Waste Range)",
            "feedback_explanation": f"Waste is within optimal range (avg {avg_waste_pct:.1f}% discarded over {sample_count} past meals). Base recommendation is optimal.",
            "bounded_by_floor": False
        }


def generate_dish_recommendation(
    meal_date: str,
    meal_type: str,
    dish_name: str,
    expected_attendance: int = 450,
    registered_students: int = 400,
    safety_margin_pct: float = 5.0,
    capacity_cap: int = 550,
    model_path: str = DEFAULT_MODEL_PATH,
    db_path: str = DEFAULT_DB_PATH,
    persist_prediction: bool = False,
    intent_breakdown: Optional[Dict[str, int]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Generate demand prediction and preparation recommendation for an official menu item.
    Follows:
      Recommended preparation = predicted demand adjusted by safety margin
      Whole number of servings, capped at mess capacity.
    """
    # Accommodate flexible kwargs from UI
    if "intent_count" in kwargs and intent_breakdown is None:
        intent_breakdown = {"yes_count": int(kwargs["intent_count"]), "not_sure_count": 0}

    # 1. Parse date and day of week
    try:
        dt = datetime.strptime(meal_date, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
        meal_date = dt.strftime("%Y-%m-%d")

    day_of_week = kwargs.get("day_of_week", dt.strftime("%A"))
    is_weekend = int(day_of_week in ["Saturday", "Sunday"])

    # 2. Attendance intent consideration
    # If detailed intent breakdown is available, adjust effective registered count:
    # 100% of 'Yes' + ~50% of 'Not sure' attend, 'No' are excluded.
    if intent_breakdown:
        yes_c = intent_breakdown.get("yes_count", registered_students)
        not_sure_c = intent_breakdown.get("not_sure_count", 0)
        effective_registered = yes_c + int(round(not_sure_c * 0.50))
    else:
        effective_registered = registered_students

    # 3. Derive historical demand strictly from past dates
    hist_demand, is_fallback = get_historical_demand_for_dish(
        meal_date=meal_date,
        meal_type=meal_type,
        dish_name=dish_name,
        expected_attendance=expected_attendance,
        db_path=db_path
    )

    # 4. Use exact dish name representation matching model training
    clean_dish = dish_name.strip()
    canonical_dish = get_canonical_dish(clean_dish)
    reg_ratio = round(effective_registered / max(expected_attendance, 1), 4)

    feature_row = pd.DataFrame([{
        "day_of_week": day_of_week,
        "meal_type": meal_type,
        "dish_name": clean_dish,
        "expected_attendance": int(expected_attendance),
        "registered_students": int(effective_registered),
        "historical_demand": float(hist_demand),
        "registration_ratio": float(reg_ratio),
        "is_weekend": is_weekend
    }])

    # 5. Model Prediction
    model = kwargs.get("model") or load_demand_model(model_path)
    model_loaded = model is not None

    if model_loaded:
        try:
            pred_raw = float(model.predict(feature_row)[0])
            predicted_demand = max(10.0, round(pred_raw, 1))
        except Exception:
            predicted_demand = hist_demand
            model_loaded = False
    else:
        # Fallback heuristic using domain prior
        take_rate = FALLBACK_TAKE_RATES.get(clean_dish, FALLBACK_TAKE_RATES.get(canonical_dish, 0.75))
        heuristic = effective_registered * take_rate * 0.95 + (expected_attendance - effective_registered) * 0.10
        predicted_demand = round(float(np.clip(heuristic, 10, capacity_cap)), 1)

    # 6. Safety Margin & Base Preparation Recommendation
    margin_multiplier = 1.0 + (max(0.0, float(safety_margin_pct)) / 100.0)
    raw_recommendation = round(predicted_demand * margin_multiplier)
    base_recommendation = int(np.clip(raw_recommendation, 10, capacity_cap))

    # 7. Adaptive Food Waste Feedback Loop
    min_records = int(kwargs.get("min_records", 3))
    max_reduction_pct = float(kwargs.get("max_reduction_pct", 15.0))
    max_increase_pct = float(kwargs.get("max_increase_pct", 10.0))
    target_waste_pct = float(kwargs.get("target_waste_pct", 5.0))

    waste_pattern = calculate_dish_waste_pattern(
        dish_name=dish_name,
        meal_type=meal_type,
        meal_date=meal_date,
        lookback_limit=kwargs.get("lookback_limit", 14),
        min_records=min_records,
        db_path=db_path
    )

    adaptive_adj = compute_adaptive_waste_adjustment(
        predicted_demand=predicted_demand,
        base_recommendation=base_recommendation,
        waste_pattern=waste_pattern,
        min_records=min_records,
        max_reduction_pct=max_reduction_pct,
        max_increase_pct=max_increase_pct,
        target_waste_pct=target_waste_pct,
        capacity_cap=capacity_cap
    )

    adjusted_preparation = adaptive_adj["adjusted_preparation_quantity"]
    recommended_preparation = adjusted_preparation
    safety_margin_units = max(0, recommended_preparation - int(round(predicted_demand)))

    # Previous waste summary dictionary
    prev_waste_summary = {
        "avg_discarded": waste_pattern["avg_discarded"],
        "avg_waste_pct": waste_pattern["avg_waste_pct"],
        "sample_count": waste_pattern["sample_count"],
        "waste_trend": waste_pattern["waste_trend"],
        "latest_discarded": waste_pattern.get("latest_discarded", 0),
        "latest_waste_pct": waste_pattern.get("latest_waste_pct", 0.0),
        "display_str": f"{waste_pattern['avg_waste_pct']:.1f}% ({waste_pattern['avg_discarded']:.0f} serv)" if waste_pattern["has_sufficient_history"] else "N/A (Insufficient history)"
    }

    # 8. Non-technical explanation for mess staff
    source_tag = "ML Model (Random Forest)" if model_loaded else "Historical Fallback Heuristic"
    hist_tag = "Prior Mess Records" if not is_fallback else "Official Menu Fallback Prior"
    feedback_note = f" Adaptive Waste Feedback: {adaptive_adj['feedback_status']} ({adaptive_adj['adjustment_servings']:+} servings)."
    explanation = (
        f"For official menu dish '{dish_name}' on {day_of_week} {meal_type}, based on {effective_registered} "
        f"active registered students and {hist_tag} ({hist_demand} servings), {source_tag} estimated "
        f"{predicted_demand:.0f} servings needed. Adding a +{safety_margin_pct:.1f}% safety buffer gave "
        f"a base recommendation of {base_recommendation} servings.{feedback_note} "
        f"Final adjusted preparation: {recommended_preparation} servings (capped at {capacity_cap}). Final decision rests with mess operator."
    )

    if persist_prediction:
        save_prediction(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish_name,
            predicted_demand=predicted_demand,
            safety_margin_pct=safety_margin_pct,
            recommended_preparation=recommended_preparation,
            actual_demand=None,
            db_path=db_path
        )

    cost_per_serving = float(kwargs.get("cost_per_serving", DEFAULT_DISH_COSTS.get(clean_dish, 15.0)))
    estimated_prep_cost = round(float(recommended_preparation * cost_per_serving), 2)
    batch_weight_kg = round(float(recommended_preparation * SERVING_WEIGHT_KG), 2)

    return {
        "dish_name": dish_name,
        "canonical_category": canonical_dish,
        "meal_type": meal_type,
        "meal_date": meal_date,
        "day_of_week": day_of_week,
        "predicted_demand": predicted_demand,
        "safety_margin_pct": safety_margin_pct,
        "safety_margin_units": safety_margin_units,
        "base_recommendation": base_recommendation,
        "current_recommendation": base_recommendation,
        "recommended_preparation": recommended_preparation,
        "adjusted_preparation_quantity": adjusted_preparation,
        "adjustment_servings": adaptive_adj["adjustment_servings"],
        "adjustment_pct": adaptive_adj["adjustment_pct"],
        "feedback_status": adaptive_adj["feedback_status"],
        "feedback_explanation": adaptive_adj["feedback_explanation"],
        "waste_pattern": waste_pattern,
        "previous_waste": prev_waste_summary,
        "cost_per_serving": cost_per_serving,
        "estimated_preparation_cost": estimated_prep_cost,
        "batch_weight_kg": batch_weight_kg,
        "historical_demand": hist_demand,
        "is_fallback": is_fallback,
        "registered_students": effective_registered,
        "expected_attendance": expected_attendance,
        "weighted_attendance": effective_registered,
        "intent_breakdown": intent_breakdown,
        "model_loaded": model_loaded,
        "data_source": "Random Forest ML" if model_loaded else "Manual Fallback Prior",
        "explanation": explanation,
        "advice": explanation
    }


def generate_menu_recommendations(
    meal_date: str,
    meal_type: str,
    dish_list: Optional[List[str]] = None,
    expected_attendance: Optional[int] = None,
    registered_students: Optional[int] = None,
    safety_margin_pct: float = 5.0,
    capacity_cap: int = 550,
    model_path: str = DEFAULT_MODEL_PATH,
    db_path: str = DEFAULT_DB_PATH,
    persist: bool = False,
    intent_breakdown: Optional[Dict[str, int]] = None,
    model: Optional[Any] = None,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Generate ML-based recommendations for all dishes on the official menu for the selected date & meal.
    Automatically integrates SQLite student attendance polling and weighted attendance when available.
    """
    # 1. Parse date for day of week
    try:
        dt = datetime.strptime(meal_date, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
        meal_date = dt.strftime("%Y-%m-%d")
    day_of_week = dt.strftime("%A")

    # 2. Automatically query student polling if intent_breakdown is not provided
    if intent_breakdown is None:
        polling_ctx = get_polling_forecast_inputs(
            meal_date=meal_date,
            meal_type=meal_type,
            default_expected=expected_attendance if expected_attendance is not None else 450,
            default_registered=registered_students if registered_students is not None else 380,
            eligible_students=450,
            db_path=db_path
        )
        if polling_ctx["has_polling"]:
            intent_breakdown = polling_ctx["intent_breakdown"]
            if registered_students is None:
                registered_students = polling_ctx["registered_students"]
            if expected_attendance is None:
                expected_attendance = polling_ctx["expected_attendance"]
        else:
            if registered_students is None:
                registered_students = 380
            if expected_attendance is None:
                expected_attendance = 450
    else:
        if registered_students is None:
            registered_students = 380
        if expected_attendance is None:
            expected_attendance = 450

    # 3. Look up official menu if not provided
    if dish_list is None or len(dish_list) == 0:
        menu_info = get_menu_for_meal(day_of_week, meal_type)
        dish_list = menu_info.get("dishes", ["Rice", "Dal", "Chapati"])

    # 4. Preload model once for efficiency across all menu dishes
    if model is None:
        model = load_demand_model(model_path)

    results = []
    for dish in dish_list:
        cost = kwargs.get("cost_per_serving", DEFAULT_DISH_COSTS.get(dish, 15.0))
        rec = generate_dish_recommendation(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish,
            expected_attendance=int(expected_attendance),
            registered_students=int(registered_students),
            safety_margin_pct=safety_margin_pct,
            capacity_cap=capacity_cap,
            model_path=model_path,
            db_path=db_path,
            persist_prediction=persist,
            intent_breakdown=intent_breakdown,
            model=model,
            cost_per_serving=cost,
            day_of_week=day_of_week,
            **kwargs
        )
        results.append(rec)
    return results


if __name__ == "__main__":
    recs = generate_menu_recommendations(
        meal_date="2026-11-05",
        meal_type="Lunch",
        expected_attendance=480,
        registered_students=410,
        safety_margin_pct=5.0
    )
    print("Official Menu Recommendations for 2026-11-05 Lunch:")
    for r in recs:
        print(f"Dish: {r['dish_name']:<18} | Predicted: {r['predicted_demand']:<5} | Recommended: {r['recommended_preparation']} servings")
