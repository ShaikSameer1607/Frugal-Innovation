"""
action_center.py
================
AI Action Center for MessMind AI.
Generates empirical, actionable operational directives and waste intelligence
derived directly from SQLite meal records, student attendance intent, and
the Scikit-learn Random Forest recommendation engine.

Zero external LLM dependencies, zero cloud API costs.
All insights are strictly calculated from actual database records.
Provides transparent severity ratings (High, Medium, Low) and explicit
insufficient-data safeguards.
"""

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from database import (
    DEFAULT_DB_PATH,
    get_connection,
    get_meal_records,
    get_menu_for_meal,
    SERVING_WEIGHT_KG
)
from recommendations import (
    load_demand_model,
    generate_dish_recommendation,
    get_polling_forecast_inputs,
    calculate_dish_waste_pattern,
    DEFAULT_DISH_COSTS,
    DEFAULT_MODEL_PATH
)


def get_next_meal_period(now_dt: Optional[datetime] = None) -> Tuple[str, str, str]:
    """
    Determine the next upcoming meal period and date based on current time.
    Returns: (meal_date_str, day_of_week, meal_period)
    """
    if now_dt is None:
        now_dt = datetime.now()

    hour = now_dt.hour
    if hour < 10:
        target_dt = now_dt
        meal_period = "Breakfast"
    elif hour < 15:
        target_dt = now_dt
        meal_period = "Lunch"
    elif hour < 18:
        target_dt = now_dt
        meal_period = "Snacks"
    elif hour < 22:
        target_dt = now_dt
        meal_period = "Dinner"
    else:
        # After 10 PM, next meal is tomorrow's Breakfast
        target_dt = now_dt + timedelta(days=1)
        meal_period = "Breakfast"

    return target_dt.strftime("%Y-%m-%d"), target_dt.strftime("%A"), meal_period


def get_high_waste_dishes_insight(
    db_path: str = DEFAULT_DB_PATH,
    min_records: int = 3,
    data_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insight 1: Identify which dishes consistently generate high food waste.
    Requires at least min_records per dish to prevent one-day outliers.
    Supports filtering by data_source ('REAL', 'SYNTHETIC', 'DEMO', None).
    """
    df = get_meal_records(data_source=data_source, db_path=db_path)
    if df.empty or len(df) < min_records:
        return {
            "id": "high_waste_dishes",
            "title": "High Food Waste Dishes",
            "severity": "Low",
            "is_sufficient_data": False,
            "insufficient_data_reason": f"Insufficient historical records in SQLite ({len(df)} total records found, minimum {min_records} required).",
            "summary": "Historical data is insufficient to identify consistent dish-level waste patterns.",
            "supporting_data": {"dishes": [], "total_records": len(df)},
            "recommended_actions": [
                "Continue recording daily kitchen food preparation and leftovers to build historical baseline.",
                "Ensure staff logs both quantity served and quantity discarded for every shift."
            ]
        }

    # Group by dish_name
    dish_grp = df.groupby("dish_name").agg(
        record_count=("id", "count"),
        total_prepared=("quantity_prepared", "sum"),
        total_served=("quantity_served", "sum"),
        total_discarded=("discarded_quantity", "sum"),
        total_cost_lost=("estimated_discarded_food_cost", "sum")
    ).reset_index()

    # Filter dishes meeting minimum occurrences
    qualified = dish_grp[dish_grp["record_count"] >= min_records].copy()
    if qualified.empty:
        return {
            "id": "high_waste_dishes",
            "title": "High Food Waste Dishes",
            "severity": "Low",
            "is_sufficient_data": False,
            "insufficient_data_reason": f"No single dish has reached the minimum {min_records} service records required for statistical pattern analysis.",
            "summary": "No dish has sufficient repeated service history yet.",
            "supporting_data": {"dishes": [], "total_records": len(df)},
            "recommended_actions": [
                "Log multiple shifts for regular menu dishes before establishing permanent cooking reductions."
            ]
        }

    qualified["waste_percentage"] = (qualified["total_discarded"] / qualified["total_prepared"].clip(lower=1) * 100.0).round(1)
    qualified["avg_discarded_per_shift"] = (qualified["total_discarded"] / qualified["record_count"]).round(1)
    qualified = qualified.sort_values(by=["waste_percentage", "total_cost_lost"], ascending=[False, False])

    # Top high-waste dishes (waste percentage >= 10%)
    high_waste = qualified[qualified["waste_percentage"] >= 10.0]
    top_dishes = qualified.head(5).to_dict(orient="records")

    if not high_waste.empty:
        highest = high_waste.iloc[0]
        max_waste_pct = highest["waste_percentage"]
        if max_waste_pct >= 20.0 or highest["total_cost_lost"] >= 2000.0:
            severity = "High"
        else:
            severity = "Medium"

        summary = (
            f"Found {len(high_waste)} dish(es) with chronic waste exceeding 10%. "
            f"'{highest['dish_name']}' is the primary culprit with {highest['waste_percentage']}% average waste "
            f"({highest['avg_discarded_per_shift']} servings discarded per meal, ₹{highest['total_cost_lost']:,.0f} total lost)."
        )
        actions = [
            f"Downsize batch preparation for '{highest['dish_name']}' by 10–15% using adaptive feedback.",
            f"Audit portion ladle sizes for top waste items ({', '.join([d['dish_name'] for d in top_dishes[:3]])}).",
            "Engage hostel student mess committee to review seasoning, spice level, or recipe appeal."
        ]
    else:
        severity = "Low"
        summary = f"All {len(qualified)} evaluated dishes exhibit healthy waste levels under 10%."
        actions = [
            "Maintain current kitchen batch preparation guidelines.",
            "Continue periodic monitoring to detect emerging waste shifts."
        ]

    return {
        "id": "high_waste_dishes",
        "title": "High Food Waste Dishes",
        "severity": severity,
        "is_sufficient_data": True,
        "insufficient_data_reason": None,
        "summary": summary,
        "supporting_data": {
            "dishes": top_dishes,
            "high_waste_count": len(high_waste),
            "evaluated_dishes_count": len(qualified)
        },
        "recommended_actions": actions
    }


def get_meal_period_waste_insight(
    db_path: str = DEFAULT_DB_PATH,
    min_records: int = 4,
    data_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insight 2: Identify which meal periods generate the most discarded food.
    Requires at least min_records and coverage across multiple shifts.
    Supports filtering by data_source ('REAL', 'SYNTHETIC', 'DEMO', None).
    """
    df = get_meal_records(data_source=data_source, db_path=db_path)
    if df.empty or len(df) < min_records or df["meal_type"].nunique() < 2:
        return {
            "id": "meal_period_waste",
            "title": "Peak Waste Dining Periods",
            "severity": "Low",
            "is_sufficient_data": False,
            "insufficient_data_reason": f"Insufficient dining shift history in SQLite (requires records from at least 2 distinct meal shifts, found {df['meal_type'].nunique() if not df.empty else 0}).",
            "summary": "Historical data is insufficient to compare waste across dining shifts.",
            "supporting_data": {"meal_shifts": []},
            "recommended_actions": [
                "Log meal preparation and leftovers across all 4 dining periods (Breakfast, Lunch, Snacks, Dinner)."
            ]
        }

    meal_grp = df.groupby("meal_type").agg(
        shift_count=("id", "count"),
        total_prepared=("quantity_prepared", "sum"),
        total_served=("quantity_served", "sum"),
        total_discarded=("discarded_quantity", "sum"),
        total_cost_lost=("estimated_discarded_food_cost", "sum")
    ).reset_index()

    total_all_discarded = max(1, meal_grp["total_discarded"].sum())
    meal_grp["waste_share_pct"] = (meal_grp["total_discarded"] / total_all_discarded * 100.0).round(1)
    meal_grp["avg_discarded_per_shift"] = (meal_grp["total_discarded"] / meal_grp["shift_count"]).round(1)
    meal_grp["waste_rate_pct"] = (meal_grp["total_discarded"] / meal_grp["total_prepared"].clip(lower=1) * 100.0).round(1)
    meal_grp = meal_grp.sort_values(by="total_discarded", ascending=False)

    worst_shift = meal_grp.iloc[0]
    worst_name = worst_shift["meal_type"]
    share_pct = worst_shift["waste_share_pct"]

    if share_pct >= 38.0 or worst_shift["avg_discarded_per_shift"] >= 35.0:
        severity = "High"
    elif share_pct >= 28.0:
        severity = "Medium"
    else:
        severity = "Low"

    summary = (
        f"'{worst_name}' produces the largest share of food waste, accounting for {share_pct}% "
        f"of all discarded food ({worst_shift['total_discarded']:,} servings discarded, "
        f"avg {worst_shift['avg_discarded_per_shift']} servings/shift, ₹{worst_shift['total_cost_lost']:,.0f} lost)."
    )

    actions = [
        f"Apply a 10%–15% conservative preparation trim specifically for {worst_name} cooking batches.",
        f"Check student attendance intent polls prior to {worst_name} prep to catch off-campus dining trends.",
        "Repurpose unserved, pristine batch surplus from earlier shifts rather than overpreparing fresh stock."
    ]

    return {
        "id": "meal_period_waste",
        "title": "Peak Waste Dining Periods",
        "severity": severity,
        "is_sufficient_data": True,
        "insufficient_data_reason": None,
        "summary": summary,
        "supporting_data": {
            "meal_shifts": meal_grp.to_dict(orient="records"),
            "worst_shift": worst_name,
            "worst_shift_share_pct": share_pct
        },
        "recommended_actions": actions
    }


def get_prep_adjustment_insight(
    db_path: str = DEFAULT_DB_PATH,
    min_records: int = 3,
    data_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insight 3: Identify specific dishes needing reduced or increased preparation.
    Uses multi-record consumption and waste ratios.
    Supports filtering by data_source ('REAL', 'SYNTHETIC', 'DEMO', None).
    """
    df = get_meal_records(data_source=data_source, db_path=db_path)
    if df.empty or len(df) < min_records:
        return {
            "id": "prep_adjustments",
            "title": "Preparation Batch Resizing Directives",
            "severity": "Low",
            "is_sufficient_data": False,
            "insufficient_data_reason": f"Insufficient meal history in SQLite ({len(df)} records, minimum {min_records} required).",
            "summary": "Insufficient data to safely recommend batch resizing without risk of shortages.",
            "supporting_data": {"reduce_dishes": [], "increase_dishes": []},
            "recommended_actions": [
                "Maintain baseline preparation quantities until at least 3 shifts are logged per dish."
            ]
        }

    dish_stats = df.groupby(["dish_name", "meal_type"]).agg(
        record_count=("id", "count"),
        avg_prepared=("quantity_prepared", "mean"),
        avg_served=("quantity_served", "mean"),
        avg_discarded=("discarded_quantity", "mean"),
        total_prepared=("quantity_prepared", "sum"),
        total_discarded=("discarded_quantity", "sum")
    ).reset_index()

    qualified = dish_stats[dish_stats["record_count"] >= min_records].copy()
    if qualified.empty:
        return {
            "id": "prep_adjustments",
            "title": "Preparation Batch Resizing Directives",
            "severity": "Low",
            "is_sufficient_data": False,
            "insufficient_data_reason": f"No dish has reached the minimum {min_records} recorded shifts.",
            "summary": "Insufficient repeated dish history for batch adjustment.",
            "supporting_data": {"reduce_dishes": [], "increase_dishes": []},
            "recommended_actions": [
                "Continue logging meal shifts to unlock automated preparation sizing."
            ]
        }

    qualified["waste_pct"] = (qualified["total_discarded"] / qualified["total_prepared"].clip(lower=1) * 100.0).round(1)
    qualified["consumption_rate"] = (qualified["avg_served"] / qualified["avg_prepared"].clip(lower=1) * 100.0).round(1)

    # Criteria:
    # Reduce: waste_pct >= 12.0%
    # Increase: consumption_rate >= 97.5% and waste_pct <= 2.5% (running out of food)
    to_reduce = qualified[qualified["waste_pct"] >= 12.0].sort_values(by="waste_pct", ascending=False)
    to_increase = qualified[(qualified["consumption_rate"] >= 97.5) & (qualified["waste_pct"] <= 2.5)].sort_values(by="consumption_rate", ascending=False)

    reduce_list = []
    for _, r in to_reduce.head(5).iterrows():
        # Suggested trim: min(15%, (waste_pct - 5%) * 0.5)
        trim_pct = min(15.0, round((r["waste_pct"] - 5.0) * 0.5, 1))
        servings_cut = int(round(r["avg_prepared"] * (trim_pct / 100.0)))
        reduce_list.append({
            "dish_name": r["dish_name"],
            "meal_type": r["meal_type"],
            "waste_pct": r["waste_pct"],
            "avg_prepared": round(r["avg_prepared"], 0),
            "suggested_cut_servings": servings_cut,
            "suggested_cut_pct": trim_pct
        })

    increase_list = []
    for _, r in to_increase.head(5).iterrows():
        boost_servings = int(round(r["avg_prepared"] * 0.05))
        increase_list.append({
            "dish_name": r["dish_name"],
            "meal_type": r["meal_type"],
            "consumption_rate": r["consumption_rate"],
            "avg_prepared": round(r["avg_prepared"], 0),
            "suggested_boost_servings": max(5, boost_servings),
            "suggested_boost_pct": 5.0
        })

    if not to_reduce.empty:
        severity = "High" if to_reduce.iloc[0]["waste_pct"] >= 20.0 else "Medium"
        summary = (
            f"Identified {len(reduce_list)} dish(es) requiring reduced cooking batches to curtail surplus, "
            f"and {len(increase_list)} dish(es) requiring increased safety buffers to prevent stockouts."
        )
    elif not to_increase.empty:
        severity = "Medium"
        summary = f"Kitchen preparation is tight. {len(increase_list)} high-demand dish(es) risk stockouts and need higher safety buffers."
    else:
        severity = "Low"
        summary = "All evaluated dishes show balanced preparation and consumption quantities."

    actions = []
    for item in reduce_list[:2]:
        actions.append(f"Trim {item['dish_name']} ({item['meal_type']}) by ~{item['suggested_cut_servings']} servings (-{item['suggested_cut_pct']}%) to curb {item['waste_pct']}% waste.")
    for item in increase_list[:2]:
        actions.append(f"Increase {item['dish_name']} ({item['meal_type']}) by +{item['suggested_boost_servings']} buffer servings to protect against diner stockouts.")
    if not actions:
        actions.append("Maintain existing cooking quantities; preparation levels are aligned with student consumption.")

    return {
        "id": "prep_adjustments",
        "title": "Preparation Batch Resizing Directives",
        "severity": severity,
        "is_sufficient_data": True,
        "insufficient_data_reason": None,
        "summary": summary,
        "supporting_data": {
            "reduce_dishes": reduce_list,
            "increase_dishes": increase_list,
            "total_evaluated": len(qualified)
        },
        "recommended_actions": actions
    }


def get_financial_loss_insight(
    db_path: str = DEFAULT_DB_PATH,
    data_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insight 4: Estimated monetary loss due to discarded food and financial culprits.
    Supports filtering by data_source ('REAL', 'SYNTHETIC', 'DEMO', None).
    """
    df = get_meal_records(data_source=data_source, db_path=db_path)
    if df.empty or df["estimated_discarded_food_cost"].sum() <= 0:
        return {
            "id": "financial_loss",
            "title": "Financial Food Waste Impact",
            "severity": "Low",
            "is_sufficient_data": False,
            "insufficient_data_reason": "No costed discarded food records found in SQLite database.",
            "summary": "No financial loss data available to evaluate.",
            "supporting_data": {"total_cost_lost": 0.0, "culprits": []},
            "recommended_actions": [
                "Record daily kitchen logs with verified item ingredient unit costs."
            ]
        }

    total_lost = float(df["estimated_discarded_food_cost"].sum())
    total_days = max(1, df["meal_date"].nunique())
    daily_avg_lost = round(total_lost / total_days, 1)
    monthly_projected_lost = round(daily_avg_lost * 30, 0)

    # Financial culprit dishes
    culprits = df.groupby("dish_name").agg(
        cost_lost=("estimated_discarded_food_cost", "sum"),
        discarded_servings=("discarded_quantity", "sum")
    ).reset_index().sort_values(by="cost_lost", ascending=False)

    top_culprits = culprits.head(5).to_dict(orient="records")
    top_culprit_name = top_culprits[0]["dish_name"] if top_culprits else "N/A"
    top_culprit_cost = top_culprits[0]["cost_lost"] if top_culprits else 0.0

    if monthly_projected_lost >= 25000.0 or daily_avg_lost >= 800.0:
        severity = "High"
    elif monthly_projected_lost >= 10000.0:
        severity = "Medium"
    else:
        severity = "Low"

    summary = (
        f"Total recorded food waste financial loss is ₹{total_lost:,.0f} "
        f"(averaging ₹{daily_avg_lost:,.0f}/day, projected ₹{monthly_projected_lost:,.0f}/month). "
        f"Top financial culprit is '{top_culprit_name}' with ₹{top_culprit_cost:,.0f} lost."
    )

    actions = [
        f"Focus kitchen portion control on high-cost dishes like '{top_culprit_name}'.",
        f"Potential monthly savings of ~₹{monthly_projected_lost * 0.40:,.0f} achievable through a 40% waste reduction via AI targets.",
        "Implement second-serving-on-demand policy for high-cost proteins and paneer items."
    ]

    return {
        "id": "financial_loss",
        "title": "Financial Food Waste Impact",
        "severity": severity,
        "is_sufficient_data": True,
        "insufficient_data_reason": None,
        "summary": summary,
        "supporting_data": {
            "total_cost_lost": total_lost,
            "daily_avg_lost": daily_avg_lost,
            "monthly_projected_lost": monthly_projected_lost,
            "total_days_evaluated": total_days,
            "culprits": top_culprits
        },
        "recommended_actions": actions
    }


def get_next_meal_action_insight(
    current_dt: Optional[datetime] = None,
    db_path: str = DEFAULT_DB_PATH,
    model: Optional[Any] = None,
    data_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Insight 5: Specific, operational recommendations for the upcoming meal.
    Combines live polling, weekly menu, ML demand forecasting, and adaptive feedback.
    """
    next_date, day_of_week, meal_period = get_next_meal_period(current_dt)

    # Menu items
    menu_info = get_menu_for_meal(day_of_week, meal_period)
    dish_list = menu_info.get("dishes", ["Plain Rice", "Dal Fry"])

    # Query live polling from SQLite
    polling_ctx = get_polling_forecast_inputs(
        meal_date=next_date,
        meal_type=meal_period,
        default_expected=450,
        default_registered=380,
        eligible_students=450,
        db_path=db_path
    )
    has_polling = polling_ctx["has_polling"]
    weighted_att = polling_ctx["weighted_attendance"]

    # Preload model if not provided
    if model is None:
        model = load_demand_model()

    dish_recs = []
    total_prep_servings = 0
    total_prep_cost = 0.0
    high_waste_alerts = []

    for dish in dish_list:
        cost = DEFAULT_DISH_COSTS.get(dish, 15.0)
        rec = generate_dish_recommendation(
            meal_date=next_date,
            meal_type=meal_period,
            dish_name=dish,
            expected_attendance=polling_ctx["expected_attendance"],
            registered_students=polling_ctx["registered_students"],
            intent_breakdown=polling_ctx["intent_breakdown"] if has_polling else None,
            safety_margin_pct=5.0,
            db_path=db_path,
            model=model,
            cost_per_serving=cost
        )

        target = rec["adjusted_preparation_quantity"]
        total_prep_servings += target
        total_prep_cost += rec["estimated_preparation_cost"]

        if rec.get("feedback_status") == "Waste Dampened (High Historical Waste)":
            high_waste_alerts.append(f"{dish} ({rec['adjustment_servings']} servings trimmed due to past waste)")

        dish_recs.append({
            "dish_name": dish,
            "predicted_demand": rec["predicted_demand"],
            "base_rec": rec["current_recommendation"],
            "adjusted_prep": target,
            "adjustment_servings": rec["adjustment_servings"],
            "feedback_status": rec["feedback_status"],
            "unit_cost": cost,
            "prep_cost": rec["estimated_preparation_cost"],
            "weight_kg": rec["batch_weight_kg"]
        })

    # Severity classification
    if high_waste_alerts:
        severity = "High"
    elif has_polling and weighted_att < 250:
        severity = "Medium"  # Significant drop in expected turnout
    else:
        severity = "Low"

    polling_note = f" (Live Polling: {weighted_att} weighted diners)" if has_polling else " (Manual Baseline)"
    summary = (
        f"Upcoming service: {day_of_week} {meal_period} ({next_date}){polling_note}. "
        f"Recommended cooking: {total_prep_servings:,} servings across {len(dish_list)} dishes "
        f"(estimated budget: ₹{total_prep_cost:,.0f})."
    )

    actions = [
        f"Cook target portions: {', '.join([f'{d['dish_name']} ({d['adjusted_prep']} serv)' for d in dish_recs[:3]])}.",
        f"Review adaptive trims: {', '.join(high_waste_alerts)}" if high_waste_alerts else "All dishes operating with standard +5% safety buffer.",
        f"Verify actual attendance against {weighted_att} polled diners before opening second cooking batches."
    ]

    return {
        "id": "next_meal_action",
        "title": f"Next Meal Action Directive ({meal_period})",
        "severity": severity,
        "is_sufficient_data": True,
        "insufficient_data_reason": None,
        "summary": summary,
        "supporting_data": {
            "target_date": next_date,
            "day_of_week": day_of_week,
            "meal_period": meal_period,
            "has_polling": has_polling,
            "weighted_attendance": weighted_att,
            "total_prep_servings": total_prep_servings,
            "total_prep_cost": total_prep_cost,
            "dishes": dish_recs,
            "high_waste_alerts": high_waste_alerts
        },
        "recommended_actions": actions
    }


def generate_action_center_insights(
    db_path: str = DEFAULT_DB_PATH,
    model: Optional[Any] = None,
    data_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compile all 5 actionable insights into an integrated AI Action Center payload.
    Provides executive summary, priority level, and categorized directives.
    Separates real operational records from synthetic/demo records.
    """
    insight_high_waste = get_high_waste_dishes_insight(db_path=db_path, data_source=data_source)
    insight_meal_period = get_meal_period_waste_insight(db_path=db_path, data_source=data_source)
    insight_prep_adjust = get_prep_adjustment_insight(db_path=db_path, data_source=data_source)
    insight_financial = get_financial_loss_insight(db_path=db_path, data_source=data_source)
    insight_next_meal = get_next_meal_action_insight(db_path=db_path, model=model, data_source=data_source)

    insights = [
        insight_next_meal,
        insight_high_waste,
        insight_meal_period,
        insight_prep_adjust,
        insight_financial
    ]

    # Calculate overall Action Center priority level
    severities = [ins["severity"] for ins in insights if ins.get("is_sufficient_data")]
    if "High" in severities:
        overall_priority = "High"
    elif "Medium" in severities:
        overall_priority = "Medium"
    else:
        overall_priority = "Low"

    active_directives_count = sum(len(ins["recommended_actions"]) for ins in insights)

    return {
        "generated_timestamp": datetime.now().isoformat(),
        "overall_priority": overall_priority,
        "active_directives_count": active_directives_count,
        "data_source": data_source or "ALL",
        "insights": {
            "high_waste_dishes": insight_high_waste,
            "meal_period_waste": insight_meal_period,
            "prep_adjustments": insight_prep_adjust,
            "financial_loss": insight_financial,
            "next_meal_action": insight_next_meal
        },
        "insights_list": insights
    }

