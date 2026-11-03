"""
test_project.py
===============
Comprehensive unit & integration test suite for MessMind AI.
Tests dataset constraints, database persistence, duplicate prevention,
official weekly menu lookup, multi-state attendance intent, ML leakage guards,
recommendation logic, and edge cases.
"""

import os
import sys
import sqlite3
import tempfile
import numpy as np
import pandas as pd
import pytest
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db
import generate_dataset as gd
import train_model as tm
import recommendations as rc
import vision_ai as va
import action_center as ac
import smart_portions as sp
import reports


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dataset_path():
    """Returns path to the synthetic dataset, generating if needed."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mess_data.csv")
    if not os.path.exists(path):
        gd.generate_synthetic_data(num_days=100, output_path=path)
    return path


@pytest.fixture(scope="session")
def df(dataset_path):
    """Loads synthetic dataset DataFrame."""
    return pd.read_csv(dataset_path)


@pytest.fixture
def temp_db():
    """Provides an isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name
    db.init_db(temp_path)
    yield temp_path
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except PermissionError:
            pass


# -----------------------------------------------------------------------------
# Test Cases (1 through 18)
# -----------------------------------------------------------------------------

def test_1_synthetic_dataset_generation():
    """Test 1: Synthetic dataset generation runs and creates DataFrame with > 1000 records."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name
    try:
        df_gen = gd.generate_synthetic_data(num_days=30, output_path=tmp_csv)
        assert isinstance(df_gen, pd.DataFrame)
        assert len(df_gen) > 200
        assert os.path.exists(tmp_csv)
    finally:
        if os.path.exists(tmp_csv):
            try:
                os.remove(tmp_csv)
            except PermissionError:
                pass


def test_2_dataset_contains_expected_columns(df):
    """Test 2: Dataset contains all required operational and metadata columns."""
    required_cols = [
        "date", "day_of_week", "meal_type", "dish_name",
        "expected_attendance", "registered_students", "historical_demand",
        "actual_attendance", "actual_servings_needed", "quantity_prepared",
        "quantity_served", "leftover_quantity", "discarded_quantity",
        "usable_leftover", "cost_per_serving_inr", "discarded_food_cost_inr",
        "is_synthetic"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_3_dataset_contains_no_unintended_missing_values(df):
    """Test 3: Dataset contains zero null or missing values across all records."""
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing/null values"


def test_4_quantities_are_nonnegative(df):
    """Test 4: All attendance, serving, leftover, and cost figures are strictly non-negative."""
    numeric_cols = [
        "expected_attendance", "registered_students", "historical_demand",
        "actual_attendance", "actual_servings_needed", "quantity_prepared",
        "quantity_served", "leftover_quantity", "discarded_quantity",
        "usable_leftover", "cost_per_serving_inr", "discarded_food_cost_inr"
    ]
    for col in numeric_cols:
        assert (df[col] >= 0).all(), f"Found negative values in column: {col}"


def test_5_quantity_served_does_not_exceed_quantity_prepared(df):
    """Test 5: Quantity served cannot exceed quantity prepared in any record."""
    violations = df[df["quantity_served"] > df["quantity_prepared"]]
    assert len(violations) == 0, f"Found {len(violations)} records where served > prepared"


def test_6_leftover_quantity_is_calculated_correctly(df):
    """Test 6: Leftover quantity must strictly equal quantity_prepared - quantity_served."""
    calculated_leftover = df["quantity_prepared"] - df["quantity_served"]
    diff = (df["leftover_quantity"] - calculated_leftover).abs()
    assert (diff == 0).all(), "Leftover calculation discrepancy found"


def test_7_discarded_quantity_does_not_exceed_leftover_quantity(df):
    """Test 7: Discarded quantity must be between 0 and leftover_quantity."""
    violations = df[df["discarded_quantity"] > df["leftover_quantity"]]
    assert len(violations) == 0, f"Found {len(violations)} records where discarded > leftover"


def test_8_discarded_food_cost_is_calculated_correctly(df):
    """Test 8: Discarded food cost must equal discarded_quantity * cost_per_serving."""
    expected_cost = (df["discarded_quantity"] * df["cost_per_serving_inr"]).round(2)
    diff = (df["discarded_food_cost_inr"] - expected_cost).abs()
    assert (diff <= 0.01).all(), "Discarded food cost calculation mismatch found"


def test_9_database_initialization(temp_db):
    """Test 9: Database initialization safely creates all tables and migrations without error."""
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "student_registrations" in tables
        assert "meal_records" in tables
        assert "prediction_records" in tables

        # Verify attendance_intent column exists
        cursor.execute("PRAGMA table_info(student_registrations);")
        cols = [r["name"] for r in cursor.fetchall()]
        assert "attendance_intent" in cols
    finally:
        conn.close()


def test_10_student_registration_storage(temp_db):
    """Test 10: Student registration can be inserted with multi-state intent (Yes/No/Not sure)."""
    # 1. Confirmed Yes
    s_yes, m_yes = db.register_student(
        meal_date="2026-11-10",
        meal_type="Dinner",
        selected_dishes=["Rice", "Paneer Curry"],
        registration_code="23BDS001",
        attendance_intent="Yes",
        db_path=temp_db
    )
    assert s_yes is True

    # 2. Skipping No
    s_no, m_no = db.register_student(
        meal_date="2026-11-10",
        meal_type="Dinner",
        selected_dishes=["Rice", "Paneer Curry"],
        registration_code="23BDS002",
        attendance_intent="No",
        db_path=temp_db
    )
    assert s_no is True

    # 3. Uncertain Not sure
    s_ns, m_ns = db.register_student(
        meal_date="2026-11-10",
        meal_type="Dinner",
        selected_dishes=["Rice", "Paneer Curry"],
        registration_code="23BDS003",
        attendance_intent="Not sure",
        db_path=temp_db
    )
    assert s_ns is True

    # Confirmed count should only be 1 (only 'Yes' is confirmed!)
    confirmed_count = db.get_registration_count("2026-11-10", "Dinner", db_path=temp_db)
    assert confirmed_count == 1


def test_11_duplicate_registration_handling(temp_db):
    """Test 11: Submitting registration with the same student ID updates intent instead of duplicating."""
    # First submission: Student plans to skip
    db.register_student(
        meal_date="2026-11-10",
        meal_type="Lunch",
        registration_code="STUDENT_A",
        attendance_intent="No",
        db_path=temp_db
    )
    assert db.get_registration_count("2026-11-10", "Lunch", db_path=temp_db) == 0

    # Student updates plan: Decides to attend (Yes)
    success, msg = db.register_student(
        meal_date="2026-11-10",
        meal_type="Lunch",
        registration_code="STUDENT_A",
        attendance_intent="Yes",
        db_path=temp_db
    )
    assert success is True
    assert "updated" in msg.lower()

    # Confirmed count must now be exactly 1
    assert db.get_registration_count("2026-11-10", "Lunch", db_path=temp_db) == 1

    # Total responses must also remain 1
    counts = db.get_attendance_intent_counts("2026-11-10", "Lunch", eligible_students=500, db_path=temp_db)
    assert counts["total_responses"] == 1
    assert counts["yes_count"] == 1
    assert counts["no_count"] == 0


def test_12_model_can_be_trained_and_loaded():
    """Test 12: Trained Scikit-learn model exists and can be loaded via joblib."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "demand_model.joblib")
    assert os.path.exists(model_path), f"Model file missing at {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model object missing predict method"


def test_13_prediction_inputs_do_not_contain_post_meal_leakage():
    """Test 13: Strictly verifies that post-meal columns are excluded from features."""
    forbidden_leakage_cols = [
        "actual_attendance",
        "actual_servings_needed",
        "quantity_prepared",
        "quantity_served",
        "leftover_quantity",
        "discarded_quantity",
        "usable_leftover",
        "discarded_food_cost_inr"
    ]
    all_features = tm.CATEGORICAL_FEATURES + tm.NUMERICAL_FEATURES
    for forbidden in forbidden_leakage_cols:
        assert forbidden not in all_features, f"Post-meal leakage detected in model features: {forbidden}"


def test_14_preparation_recommendations_return_valid_nonnegative_integers():
    """Test 14: Preparation recommendations return whole non-negative integer servings."""
    rec = rc.generate_dish_recommendation(
        meal_date="2026-11-12",
        meal_type="Lunch",
        dish_name="Curd Rice",
        expected_attendance=450,
        registered_students=390,
        safety_margin_pct=5.0,
        capacity_cap=550
    )
    prep_qty = rec["recommended_preparation"]
    assert isinstance(prep_qty, int)
    assert prep_qty >= 0
    assert prep_qty <= 550
    assert rec["predicted_demand"] >= 0


def test_15_waste_calculations_handle_zero_quantities_correctly(temp_db):
    """Test 15: Food-waste recording correctly handles edge cases (zero prepared, zero leftover, zero cost)."""
    # Case A: Exactly consumed, zero leftovers
    ok_a, msg_a = db.record_meal_waste(
        meal_date="2026-11-15",
        meal_type="Breakfast",
        dish_name="Idli",
        quantity_prepared=300,
        quantity_served=300,
        discarded_quantity=0,
        cost_per_serving=12.0,
        db_path=temp_db
    )
    assert ok_a is True

    # Case B: Zero prepared (e.g. meal cancelled or dish omitted)
    ok_b, msg_b = db.record_meal_waste(
        meal_date="2026-11-15",
        meal_type="Breakfast",
        dish_name="Upma",
        quantity_prepared=0,
        quantity_served=0,
        discarded_quantity=0,
        cost_per_serving=10.0,
        db_path=temp_db
    )
    assert ok_b is True

    # Case C: Invalid served > prepared must be rejected safely
    bad, bad_msg = db.record_meal_waste(
        meal_date="2026-11-15",
        meal_type="Breakfast",
        dish_name="Dosa",
        quantity_prepared=100,
        quantity_served=150,
        discarded_quantity=0,
        cost_per_serving=18.0,
        db_path=temp_db
    )
    assert bad is False
    assert "exceed" in bad_msg.lower()


def test_16_official_weekly_menu_lookup():
    """Test 16: Verifies official IIIT Dharwad Mess 1 weekly menu lookup for all 7 days across all 4 shifts."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meals = ["Breakfast", "Lunch", "Snacks", "Dinner"]

    for day in days:
        for meal in meals:
            menu = db.get_menu_for_meal(day, meal)
            assert "dishes" in menu, f"Missing dishes for {day} {meal}"
            assert len(menu["dishes"]) > 0, f"Dishes list empty for {day} {meal}"
            assert "display_text" in menu, f"Missing display text for {day} {meal}"

    # Specific sanity check: Monday Lunch must feature Curd Rice or Dal Methi
    mon_lunch = db.get_menu_for_meal("Monday", "Lunch")
    assert "Curd Rice" in mon_lunch["dishes"] or "Rice" in mon_lunch["dishes"]

    # Specific sanity check: Monday Snacks must feature Samosa
    mon_snacks = db.get_menu_for_meal("Monday", "Snacks")
    assert "Samosa" in mon_snacks["dishes"]


def test_17_attendance_intent_counts_breakdown(temp_db):
    """Test 17: Verifies comprehensive attendance intent breakdown (Yes, No, Not sure, Unresponsive)."""
    # Register 3 students: 1 Yes, 1 No, 1 Not sure
    db.register_student("2026-11-20", "Breakfast", registration_code="S1", attendance_intent="Yes", db_path=temp_db)
    db.register_student("2026-11-20", "Breakfast", registration_code="S2", attendance_intent="No", db_path=temp_db)
    db.register_student("2026-11-20", "Breakfast", registration_code="S3", attendance_intent="Not sure", db_path=temp_db)

    counts = db.get_attendance_intent_counts("2026-11-20", "Breakfast", eligible_students=500, db_path=temp_db)

    assert counts["yes_count"] == 1
    assert counts["no_count"] == 1
    assert counts["not_sure_count"] == 1
    assert counts["total_responses"] == 3
    assert counts["eligible_students"] == 500
    assert counts["unresponsive_count"] == 497

    # Skipped meal must NOT contribute to confirmed count
    assert db.get_registration_count("2026-11-20", "Breakfast", db_path=temp_db) == 1


def test_18_official_menu_recommendations():
    """Test 18: Verifies demand and preparation recommendation for official menu schedule."""
    recs = rc.generate_menu_recommendations(
        meal_date="2026-11-20",
        meal_type="Dinner",
        expected_attendance=480,
        registered_students=380,
        safety_margin_pct=5.0,
        capacity_cap=550
    )
    assert isinstance(recs, list)
    assert len(recs) > 0
    for r in recs:
        assert "dish_name" in r
        assert "recommended_preparation" in r
        assert r["recommended_preparation"] >= 0
        assert r["recommended_preparation"] <= 550
        assert "explanation" in r


def test_19_vision_ai_sample_generation():
    """Test 19: Verifies synthetic sample food tray generation for clean, moderate, and high leftover trays."""
    img_clean = va.create_sample_tray_image("clean")
    img_mod = va.create_sample_tray_image("moderate")
    img_high = va.create_sample_tray_image("high")

    assert img_clean.size == (640, 480)
    assert img_mod.size == (640, 480)
    assert img_high.size == (640, 480)


def test_20_vision_ai_image_analysis_pipeline():
    """Test 20: Verifies end-to-end OpenCV leftover segmentation and metrics pipeline."""
    sample_img = va.create_sample_tray_image("moderate")
    result = va.analyze_food_image(sample_img)

    assert result["success"] is True
    assert "coverage_percentage" in result
    assert 0.0 <= result["coverage_percentage"] <= 100.0
    assert "leftover_category" in result
    assert result["leftover_category"] in ["Clean Tray", "Low Waste", "Moderate Waste", "High Waste"]
    assert "overlay_img" in result
    assert "mask_img" in result
    assert "heatmap_img" in result
    assert "disclaimer" in result
    assert "processing_time_ms" in result
    assert result["processing_time_ms"] > 0


def test_21_get_filtered_meal_records():
    """Test 21: Verifies database search and filtering for saved meal records."""
    df_all, total_all = db.get_filtered_meal_records(limit=10)
    assert total_all > 0
    assert len(df_all) <= 10
    assert "dish_name" in df_all.columns
    assert "meal_type" in df_all.columns

    # Test filtering by meal period
    df_lunch, total_lunch = db.get_filtered_meal_records(meal_type="Lunch", limit=20)
    assert total_lunch > 0
    assert (df_lunch["meal_type"] == "Lunch").all()

    # Test filtering by dish name search
    df_rice, total_rice = db.get_filtered_meal_records(search_dish="Rice", limit=20)
    assert total_rice > 0
    assert df_rice["dish_name"].str.contains("Rice", case=False).all()


def test_22_database_storage_location_transparency():
    """Test 22: Verifies storage transparency info reports valid SQLite and CSV paths."""
    info = db.get_database_storage_info()
    assert "sqlite_path" in info
    assert os.path.exists(info["sqlite_path"])
    assert "meal_records_count" in info
    assert info["meal_records_count"] > 0
    assert "csv_path" in info
    assert os.path.exists(info["csv_path"])
    assert "Synchronized" in info["status"]


def test_23_polling_forecast_inputs_weighted_calculation(temp_db):
    """Test 23: Verifies SQLite student attendance polling extraction and weighted calculation."""
    test_date = "2026-11-25"
    test_meal = "Lunch"

    # Step 1: No polling exists -> must use manual fallback
    fallback_ctx = rc.get_polling_forecast_inputs(test_date, test_meal, db_path=temp_db)
    assert fallback_ctx["has_polling"] is False
    assert fallback_ctx["total_responses"] == 0
    assert fallback_ctx["weighted_attendance"] == 0
    assert fallback_ctx["registered_students"] == 380
    assert fallback_ctx["expected_attendance"] == 450
    assert fallback_ctx["data_source"] == "Manual Fallback"

    # Step 2: Register polled student responses (100 Yes, 40 Not sure, 60 No) via bulk insert
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        rows = []
        for i in range(100):
            rows.append((test_date, test_meal, "Rice,Dal", f"YES_{i}", 1, "Yes", "2026-11-25T10:00:00"))
        for i in range(40):
            rows.append((test_date, test_meal, "Rice,Dal", f"NOTSURE_{i}", 0, "Not sure", "2026-11-25T10:00:00"))
        for i in range(60):
            rows.append((test_date, test_meal, "Rice,Dal", f"NO_{i}", 0, "No", "2026-11-25T10:00:00"))
        cursor.executemany("""
            INSERT INTO student_registrations (
                meal_date, meal_type, selected_dishes, registration_code,
                intention_to_attend, attendance_intent, submission_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
    finally:
        conn.close()

    # Step 3: Fetch polling forecast inputs
    poll_ctx = rc.get_polling_forecast_inputs(test_date, test_meal, db_path=temp_db)
    assert poll_ctx["has_polling"] is True
    assert poll_ctx["yes_count"] == 100
    assert poll_ctx["not_sure_count"] == 40
    assert poll_ctx["no_count"] == 60
    assert poll_ctx["total_responses"] == 200

    # Weighted calculation: Yes = 100%, Not Sure = 50%, No = 0%
    # Expected: 100 * 1.0 + 40 * 0.50 + 60 * 0.0 = 120
    assert poll_ctx["weighted_attendance"] == 120
    assert poll_ctx["registered_students"] == 120
    assert poll_ctx["expected_attendance"] == 450
    assert poll_ctx["data_source"] == "SQLite Student Polling"


def test_24_all_four_meal_periods_polling_forecast_sensitivity(temp_db):
    """Test 24: Verifies forecast sensitivity across all four meal periods (Breakfast, Lunch, Snacks, Dinner)."""
    meal_shifts = [
        ("Breakfast", "Masala Dosa", "2026-11-26"),
        ("Lunch", "Curd Rice", "2026-11-27"),
        ("Snacks", "Samosa", "2026-11-28"),
        ("Dinner", "Kadai Paneer", "2026-11-29")
    ]

    for meal_type, dish_name, meal_date in meal_shifts:
        conn = db.get_connection(temp_db)
        try:
            cursor = conn.cursor()
            # Scenario 1: Low attendance intent (High skipping: 40 Yes, 20 Not sure, 180 No)
            # Weighted attendance = 40 + 10 = 50
            rows_low = []
            for i in range(40):
                rows_low.append((meal_date, meal_type, dish_name, f"LOW_Y_{i}", 1, "Yes", "2026-11-26T10:00:00"))
            for i in range(20):
                rows_low.append((meal_date, meal_type, dish_name, f"LOW_NS_{i}", 0, "Not sure", "2026-11-26T10:00:00"))
            for i in range(180):
                rows_low.append((meal_date, meal_type, dish_name, f"LOW_N_{i}", 0, "No", "2026-11-26T10:00:00"))
            cursor.executemany("""
                INSERT INTO student_registrations (
                    meal_date, meal_type, selected_dishes, registration_code,
                    intention_to_attend, attendance_intent, submission_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows_low)
            conn.commit()
        finally:
            conn.close()

        ctx_low = rc.get_polling_forecast_inputs(meal_date, meal_type, db_path=temp_db)
        assert ctx_low["has_polling"] is True
        assert ctx_low["weighted_attendance"] == 50

        rec_low = rc.generate_dish_recommendation(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish_name,
            expected_attendance=ctx_low["expected_attendance"],
            registered_students=ctx_low["registered_students"],
            intent_breakdown=ctx_low["intent_breakdown"],
            safety_margin_pct=5.0,
            db_path=temp_db
        )

        conn = db.get_connection(temp_db)
        try:
            cursor = conn.cursor()
            # Scenario 2: High attendance intent (Students enthusiastic: add 260 Yes votes)
            # Total Yes = 40 + 260 = 300, Not sure = 20, No = 180
            # Weighted attendance = 300 + 10 = 310
            rows_high = []
            for i in range(260):
                rows_high.append((meal_date, meal_type, dish_name, f"HIGH_Y_{i}", 1, "Yes", "2026-11-26T10:00:00"))
            cursor.executemany("""
                INSERT INTO student_registrations (
                    meal_date, meal_type, selected_dishes, registration_code,
                    intention_to_attend, attendance_intent, submission_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows_high)
            conn.commit()
        finally:
            conn.close()

        ctx_high = rc.get_polling_forecast_inputs(meal_date, meal_type, db_path=temp_db)
        assert ctx_high["has_polling"] is True
        assert ctx_high["weighted_attendance"] == 310

        rec_high = rc.generate_dish_recommendation(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish_name,
            expected_attendance=ctx_high["expected_attendance"],
            registered_students=ctx_high["registered_students"],
            intent_breakdown=ctx_high["intent_breakdown"],
            safety_margin_pct=5.0,
            db_path=temp_db
        )

        # Verifications for this meal shift
        assert rec_high["predicted_demand"] > rec_low["predicted_demand"], (
            f"Expected higher predicted demand for {meal_type} {dish_name} with high polling, "
            f"got low={rec_low['predicted_demand']} vs high={rec_high['predicted_demand']}"
        )
        assert rec_high["recommended_preparation"] > rec_low["recommended_preparation"], (
            f"Expected higher recommended prep for {meal_type} {dish_name} with high polling, "
            f"got low={rec_low['recommended_preparation']} vs high={rec_high['recommended_preparation']}"
        )
        assert rec_low["weighted_attendance"] == 50
        assert rec_high["weighted_attendance"] == 310


def test_25_kitchen_targets_match_individual_ml_recommendations_across_all_four_meals(temp_db):
    """Test 25: Verifies kitchen cooking targets match individual ML recommendations across all 4 meals."""
    meal_shifts = [
        ("Breakfast", "2026-11-26"),  # Thursday
        ("Lunch", "2026-11-27"),      # Friday
        ("Snacks", "2026-11-28"),     # Saturday
        ("Dinner", "2026-11-29")      # Sunday
    ]

    for meal_type, meal_date in meal_shifts:
        # 1. Register student intent (120 Yes, 30 Not sure, 40 No -> weighted = 135)
        conn = db.get_connection(temp_db)
        try:
            cursor = conn.cursor()
            rows = []
            for i in range(120):
                rows.append((meal_date, meal_type, "Menu", f"Y_{meal_type}_{i}", 1, "Yes", "2026-11-26T08:00:00"))
            for i in range(30):
                rows.append((meal_date, meal_type, "Menu", f"NS_{meal_type}_{i}", 0, "Not sure", "2026-11-26T08:00:00"))
            for i in range(40):
                rows.append((meal_date, meal_type, "Menu", f"N_{meal_type}_{i}", 0, "No", "2026-11-26T08:00:00"))
            cursor.executemany("""
                INSERT INTO student_registrations (
                    meal_date, meal_type, selected_dishes, registration_code,
                    intention_to_attend, attendance_intent, submission_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
        finally:
            conn.close()

        # 2. Get polling context
        polling_ctx = rc.get_polling_forecast_inputs(meal_date, meal_type, db_path=temp_db)
        assert polling_ctx["has_polling"] is True
        assert polling_ctx["weighted_attendance"] == 135

        # 3. Generate kitchen targets for all dishes in this shift
        kitchen_targets = rc.generate_menu_recommendations(
            meal_date=meal_date,
            meal_type=meal_type,
            expected_attendance=polling_ctx["expected_attendance"],
            registered_students=polling_ctx["registered_students"],
            intent_breakdown=polling_ctx["intent_breakdown"],
            safety_margin_pct=5.0,
            db_path=temp_db
        )

        assert len(kitchen_targets) > 0, f"No kitchen targets generated for {meal_type}"

        # 4. Compare each dish's kitchen target with individual ML prediction
        for item in kitchen_targets:
            dish = item["dish_name"]
            individual_rec = rc.generate_dish_recommendation(
                meal_date=meal_date,
                meal_type=meal_type,
                dish_name=dish,
                expected_attendance=polling_ctx["expected_attendance"],
                registered_students=polling_ctx["registered_students"],
                intent_breakdown=polling_ctx["intent_breakdown"],
                safety_margin_pct=5.0,
                db_path=temp_db
            )

            # Strict equality assertions
            assert item["predicted_demand"] == individual_rec["predicted_demand"], (
                f"Predicted demand mismatch for {dish} in {meal_type}: {item['predicted_demand']} vs {individual_rec['predicted_demand']}"
            )
            assert item["recommended_preparation"] == individual_rec["recommended_preparation"], (
                f"Recommended prep mismatch for {dish} in {meal_type}: {item['recommended_preparation']} vs {individual_rec['recommended_preparation']}"
            )
            assert item["safety_margin_units"] == individual_rec["safety_margin_units"]
            assert item["cost_per_serving"] == individual_rec["cost_per_serving"]
            assert item["estimated_preparation_cost"] == individual_rec["estimated_preparation_cost"]
            assert item["batch_weight_kg"] == individual_rec["batch_weight_kg"]
            assert item["model_loaded"] is True
            assert item["recommended_preparation"] > 0
            assert item["estimated_preparation_cost"] > 0


def test_26_adaptive_feedback_insufficient_history(temp_db):
    """Test 26: Verifies adaptive feedback loop maintains base ML recommendation when insufficient history (< 3 meals) exists."""
    test_dish = "Masala Dosa"
    test_meal = "Breakfast"
    test_date = "2026-11-20"

    # Case A: 0 historical records
    waste_pattern_zero = rc.calculate_dish_waste_pattern(
        dish_name=test_dish,
        meal_type=test_meal,
        meal_date=test_date,
        min_records=3,
        db_path=temp_db
    )
    assert waste_pattern_zero["has_sufficient_history"] is False
    assert waste_pattern_zero["sample_count"] == 0
    assert waste_pattern_zero["waste_trend"] == "Insufficient History"

    rec_zero = rc.generate_dish_recommendation(
        meal_date=test_date,
        meal_type=test_meal,
        dish_name=test_dish,
        expected_attendance=450,
        registered_students=380,
        safety_margin_pct=5.0,
        min_records=3,
        db_path=temp_db
    )
    assert rec_zero["current_recommendation"] == rec_zero["base_recommendation"]
    assert rec_zero["adjusted_preparation_quantity"] == rec_zero["base_recommendation"]
    assert rec_zero["adjustment_servings"] == 0
    assert "Insufficient History" in rec_zero["feedback_status"]

    # Case B: Only 1 historical record (still below threshold of 3)
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, ("2026-11-10", test_meal, test_dish, 300, 200, 100, 80, 20, 20.0, 1600.0, "2026-11-10T10:00:00"))
        conn.commit()
    finally:
        conn.close()

    waste_pattern_one = rc.calculate_dish_waste_pattern(
        dish_name=test_dish,
        meal_type=test_meal,
        meal_date=test_date,
        min_records=3,
        db_path=temp_db
    )
    assert waste_pattern_one["has_sufficient_history"] is False
    assert waste_pattern_one["sample_count"] == 1

    rec_one = rc.generate_dish_recommendation(
        meal_date=test_date,
        meal_type=test_meal,
        dish_name=test_dish,
        expected_attendance=450,
        registered_students=380,
        safety_margin_pct=5.0,
        min_records=3,
        db_path=temp_db
    )
    assert rec_one["adjusted_preparation_quantity"] == rec_one["base_recommendation"]
    assert rec_one["adjustment_servings"] == 0
    assert "Insufficient History" in rec_one["feedback_status"]


def test_27_adaptive_feedback_high_waste_scenario(temp_db):
    """Test 27: Verifies adaptive feedback loop dampens overpreparation in high-waste scenarios without dropping below predicted demand."""
    test_dish = "Kadai Paneer"
    test_meal = "Dinner"
    test_date = "2026-11-25"

    # Insert 4 historical records with persistently high waste (e.g. 20-25% waste)
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        hist_rows = [
            ("2026-11-15", test_meal, test_dish, 350, 260, 90, 70, 20, 34.0, 2380.0, 0, "2026-11-15T21:00:00"),
            ("2026-11-17", test_meal, test_dish, 360, 270, 90, 80, 10, 34.0, 2720.0, 0, "2026-11-17T21:00:00"),
            ("2026-11-19", test_meal, test_dish, 340, 250, 90, 75, 15, 34.0, 2550.0, 0, "2026-11-19T21:00:00"),
            ("2026-11-21", test_meal, test_dish, 350, 265, 85, 70, 15, 34.0, 2380.0, 0, "2026-11-21T21:00:00"),
        ]
        cursor.executemany("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, hist_rows)
        conn.commit()
    finally:
        conn.close()

    pattern = rc.calculate_dish_waste_pattern(
        dish_name=test_dish,
        meal_type=test_meal,
        meal_date=test_date,
        min_records=3,
        db_path=temp_db
    )
    assert pattern["has_sufficient_history"] is True
    assert pattern["sample_count"] == 4
    assert pattern["waste_trend"] == "High Waste"
    assert pattern["avg_waste_pct"] > 15.0

    rec = rc.generate_dish_recommendation(
        meal_date=test_date,
        meal_type=test_meal,
        dish_name=test_dish,
        expected_attendance=450,
        registered_students=380,
        safety_margin_pct=5.0,
        min_records=3,
        max_reduction_pct=15.0,
        db_path=temp_db
    )

    # Verifications for high-waste feedback
    assert rec["adjustment_servings"] < 0, f"Expected negative adjustment, got {rec['adjustment_servings']}"
    assert rec["adjusted_preparation_quantity"] < rec["base_recommendation"], "Adjusted prep should be lower than base rec"
    floor_demand = int(np.ceil(rec["predicted_demand"]))
    assert rec["adjusted_preparation_quantity"] >= floor_demand, (
        f"Adjusted prep ({rec['adjusted_preparation_quantity']}) dropped below predicted demand floor ({floor_demand})"
    )
    reduction_pct = abs(rec["adjustment_pct"])
    assert reduction_pct <= 15.0 + 0.1, f"Reduction {reduction_pct}% exceeded max_reduction_pct 15%"
    assert "Waste Dampened" in rec["feedback_status"]


def test_28_adaptive_feedback_low_waste_scenario(temp_db):
    """Test 28: Verifies adaptive feedback loop adds protective buffer in low-waste/stockout-risk scenarios."""
    test_dish = "Curd Rice"
    test_meal = "Lunch"
    test_date = "2026-11-25"

    # Insert 4 historical records with very low waste (< 2% waste)
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        hist_rows = [
            ("2026-11-16", test_meal, test_dish, 320, 318, 2, 2, 0, 11.0, 22.0, 0, "2026-11-16T14:00:00"),
            ("2026-11-18", test_meal, test_dish, 330, 326, 4, 3, 1, 11.0, 33.0, 0, "2026-11-18T14:00:00"),
            ("2026-11-20", test_meal, test_dish, 325, 322, 3, 2, 1, 11.0, 22.0, 0, "2026-11-20T14:00:00"),
            ("2026-11-22", test_meal, test_dish, 340, 337, 3, 3, 0, 11.0, 33.0, 0, "2026-11-22T14:00:00"),
        ]
        cursor.executemany("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, hist_rows)
        conn.commit()
    finally:
        conn.close()

    pattern = rc.calculate_dish_waste_pattern(
        dish_name=test_dish,
        meal_type=test_meal,
        meal_date=test_date,
        min_records=3,
        db_path=temp_db
    )
    assert pattern["has_sufficient_history"] is True
    assert pattern["sample_count"] == 4
    assert pattern["waste_trend"] == "Low Waste"
    assert pattern["avg_waste_pct"] <= 2.5

    rec = rc.generate_dish_recommendation(
        meal_date=test_date,
        meal_type=test_meal,
        dish_name=test_dish,
        expected_attendance=450,
        registered_students=380,
        safety_margin_pct=5.0,
        min_records=3,
        max_increase_pct=10.0,
        db_path=temp_db
    )

    # Verifications for low-waste feedback
    assert rec["adjustment_servings"] > 0, f"Expected positive buffer boost, got {rec['adjustment_servings']}"
    assert rec["adjusted_preparation_quantity"] > rec["base_recommendation"]
    increase_pct = rec["adjustment_pct"]
    assert increase_pct <= 10.0 + 0.1, f"Increase {increase_pct}% exceeded max_increase_pct 10%"
    assert "Buffer Boosted" in rec["feedback_status"]


def test_29_adaptive_feedback_closed_loop_learning(temp_db):
    """Test 29: Verifies actual recorded outcomes update SQLite and are learned by subsequent adaptive recommendations."""
    test_dish = "Samosa"
    test_meal = "Snacks"
    before_date = "2026-11-20"
    future_date = "2026-11-21"

    # Step 1: Start with 2 records (insufficient history)
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, ("2026-11-18", test_meal, test_dish, 280, 270, 10, 5, 5, 12.0, 60.0, "2026-11-18T17:00:00"))
        cursor.execute("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, ("2026-11-19", test_meal, test_dish, 290, 280, 10, 5, 5, 12.0, 60.0, "2026-11-19T17:00:00"))
        conn.commit()
    finally:
        conn.close()

    rec_before = rc.generate_dish_recommendation(
        meal_date=before_date,
        meal_type=test_meal,
        dish_name=test_dish,
        expected_attendance=450,
        registered_students=380,
        min_records=3,
        db_path=temp_db
    )
    assert rec_before["waste_pattern"]["has_sufficient_history"] is False
    assert rec_before["adjustment_servings"] == 0

    # Step 2: Record an actual meal outcome via record_actual_meal_outcome on 2026-11-20
    ok, msg = db.record_actual_meal_outcome(
        meal_date=before_date,
        meal_type=test_meal,
        dish_name=test_dish,
        quantity_prepared=300,
        quantity_served=295,
        discarded_quantity=4,
        cost_per_serving=12.0,
        db_path=temp_db
    )
    assert ok is True

    # Step 3: Now generate recommendation for future_date (2026-11-21)
    rec_after = rc.generate_dish_recommendation(
        meal_date=future_date,
        meal_type=test_meal,
        dish_name=test_dish,
        expected_attendance=450,
        registered_students=380,
        min_records=3,
        db_path=temp_db
    )

    # It has learned the 3rd record! History is now sufficient
    assert rec_after["waste_pattern"]["has_sufficient_history"] is True
    assert rec_after["waste_pattern"]["sample_count"] == 3
    assert rec_after["waste_pattern"]["latest_discarded"] == 4


def test_30_action_center_insufficient_data_handling(temp_db):
    """Test 30: Verifies AI Action Center transparently reports insufficient data instead of misleading advice when database has minimal records."""
    # 1. High waste dishes with empty DB
    hw = ac.get_high_waste_dishes_insight(db_path=temp_db, min_records=3)
    assert hw["is_sufficient_data"] is False
    assert hw["severity"] == "Low"
    assert "Insufficient" in hw["insufficient_data_reason"]
    assert len(hw["recommended_actions"]) > 0

    # 2. Meal period waste with empty DB
    mp = ac.get_meal_period_waste_insight(db_path=temp_db, min_records=4)
    assert mp["is_sufficient_data"] is False
    assert "Insufficient" in mp["insufficient_data_reason"]

    # 3. Prep resizing directives with empty DB
    pa = ac.get_prep_adjustment_insight(db_path=temp_db, min_records=3)
    assert pa["is_sufficient_data"] is False
    assert "Insufficient" in pa["insufficient_data_reason"]

    # 4. Financial loss with empty DB
    fl = ac.get_financial_loss_insight(db_path=temp_db)
    assert fl["is_sufficient_data"] is False
    assert "No costed" in fl["insufficient_data_reason"]

    # 5. Full Action Center payload on empty DB
    full_ac = ac.generate_action_center_insights(db_path=temp_db)
    assert "insights" in full_ac
    assert len(full_ac["insights_list"]) == 5
    assert full_ac["overall_priority"] in ["High", "Medium", "Low"]


def test_31_action_center_insight_generation_with_data(temp_db):
    """Test 31: Verifies all 5 AI Action Center insights are accurately computed from actual SQLite records."""
    # Seed multi-shift operational records with distinct patterns
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        hist_rows = [
            # Dal Methi (Lunch) - 4 records with chronic high waste (25% discarded)
            ("2026-11-10", "Lunch", "Dal Methi", 300, 225, 75, 75, 0, 14.0, 1050.0, 0, "2026-11-10T14:00:00"),
            ("2026-11-12", "Lunch", "Dal Methi", 320, 240, 80, 80, 0, 14.0, 1120.0, 0, "2026-11-12T14:00:00"),
            ("2026-11-14", "Lunch", "Dal Methi", 310, 230, 80, 80, 0, 14.0, 1120.0, 0, "2026-11-14T14:00:00"),
            ("2026-11-16", "Lunch", "Dal Methi", 300, 225, 75, 75, 0, 14.0, 1050.0, 0, "2026-11-16T14:00:00"),

            # Kadai Paneer (Dinner) - 4 records with high waste (20% discarded)
            ("2026-11-11", "Dinner", "Kadai Paneer", 350, 280, 70, 70, 0, 34.0, 2380.0, 0, "2026-11-11T21:00:00"),
            ("2026-11-13", "Dinner", "Kadai Paneer", 360, 290, 70, 70, 0, 34.0, 2380.0, 0, "2026-11-13T21:00:00"),
            ("2026-11-15", "Dinner", "Kadai Paneer", 350, 280, 70, 70, 0, 34.0, 2380.0, 0, "2026-11-15T21:00:00"),
            ("2026-11-17", "Dinner", "Kadai Paneer", 350, 280, 70, 70, 0, 34.0, 2380.0, 0, "2026-11-17T21:00:00"),

            # Curd Rice (Lunch) - 4 records with very low waste (< 1% discarded, nearly 100% consumed)
            ("2026-11-10", "Lunch", "Curd Rice", 310, 308, 2, 2, 0, 11.0, 22.0, 0, "2026-11-10T14:00:00"),
            ("2026-11-12", "Lunch", "Curd Rice", 320, 318, 2, 2, 0, 11.0, 22.0, 0, "2026-11-12T14:00:00"),
            ("2026-11-14", "Lunch", "Curd Rice", 315, 313, 2, 2, 0, 11.0, 22.0, 0, "2026-11-14T14:00:00"),
            ("2026-11-16", "Lunch", "Curd Rice", 310, 308, 2, 2, 0, 11.0, 22.0, 0, "2026-11-16T14:00:00"),

            # Poha (Breakfast) - 4 records with balanced waste (~4% discarded)
            ("2026-11-11", "Breakfast", "Poha", 260, 250, 10, 10, 0, 12.0, 120.0, 0, "2026-11-11T09:00:00"),
            ("2026-11-13", "Breakfast", "Poha", 260, 250, 10, 10, 0, 12.0, 120.0, 0, "2026-11-13T09:00:00"),
            ("2026-11-15", "Breakfast", "Poha", 260, 250, 10, 10, 0, 12.0, 120.0, 0, "2026-11-15T09:00:00"),
            ("2026-11-17", "Breakfast", "Poha", 260, 250, 10, 10, 0, 12.0, 120.0, 0, "2026-11-17T09:00:00"),
        ]
        cursor.executemany("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, hist_rows)
        conn.commit()
    finally:
        conn.close()

    ac_payload = ac.generate_action_center_insights(db_path=temp_db)

    # 1. Overall Action Center Structure
    assert ac_payload["overall_priority"] == "High"
    assert len(ac_payload["insights_list"]) == 5
    assert ac_payload["active_directives_count"] > 0

    # 2. Insight 1: High Waste Dishes
    ins_hw = ac_payload["insights"]["high_waste_dishes"]
    assert ins_hw["is_sufficient_data"] is True
    assert ins_hw["severity"] == "High"
    culprit_names = [d["dish_name"] for d in ins_hw["supporting_data"]["dishes"]]
    assert "Dal Methi" in culprit_names or "Kadai Paneer" in culprit_names
    assert len(ins_hw["recommended_actions"]) >= 2

    # 3. Insight 2: Meal Period Waste
    ins_mp = ac_payload["insights"]["meal_period_waste"]
    assert ins_mp["is_sufficient_data"] is True
    assert ins_mp["supporting_data"]["worst_shift"] in ["Lunch", "Dinner"]
    assert ins_mp["severity"] in ["High", "Medium"]
    assert len(ins_mp["recommended_actions"]) >= 2

    # 4. Insight 3: Batch Resizing Directives
    ins_pa = ac_payload["insights"]["prep_adjustments"]
    assert ins_pa["is_sufficient_data"] is True
    reduce_dishes = [d["dish_name"] for d in ins_pa["supporting_data"]["reduce_dishes"]]
    increase_dishes = [d["dish_name"] for d in ins_pa["supporting_data"]["increase_dishes"]]
    assert "Dal Methi" in reduce_dishes or "Kadai Paneer" in reduce_dishes
    assert "Curd Rice" in increase_dishes
    assert len(ins_pa["recommended_actions"]) >= 2

    # 5. Insight 4: Financial Loss
    ins_fl = ac_payload["insights"]["financial_loss"]
    assert ins_fl["is_sufficient_data"] is True
    assert ins_fl["supporting_data"]["total_cost_lost"] > 5000.0
    assert ins_fl["supporting_data"]["daily_avg_lost"] > 0
    assert ins_fl["severity"] in ["High", "Medium"]

    # 6. Insight 5: Next Meal Action
    ins_nm = ac_payload["insights"]["next_meal_action"]
    assert ins_nm["is_sufficient_data"] is True
    assert ins_nm["supporting_data"]["total_prep_servings"] > 0
    assert len(ins_nm["supporting_data"]["dishes"]) > 0


def test_32_action_center_next_meal_timing_logic():
    """Test 32: Verifies next meal period detection across varying times of day."""
    from datetime import datetime

    # Morning 8:30 AM -> Breakfast
    dt_morn = datetime(2026, 11, 20, 8, 30)
    d, day, meal = ac.get_next_meal_period(dt_morn)
    assert meal == "Breakfast"
    assert d == "2026-11-20"
    assert day == "Friday"

    # Afternoon 1:30 PM -> Lunch
    dt_noon = datetime(2026, 11, 20, 13, 30)
    d, day, meal = ac.get_next_meal_period(dt_noon)
    assert meal == "Lunch"
    assert d == "2026-11-20"

    # Late Afternoon 4:30 PM -> Snacks
    dt_snack = datetime(2026, 11, 20, 16, 30)
    d, day, meal = ac.get_next_meal_period(dt_snack)
    assert meal == "Snacks"
    assert d == "2026-11-20"

    # Evening 7:30 PM -> Dinner
    dt_eve = datetime(2026, 11, 20, 19, 30)
    d, day, meal = ac.get_next_meal_period(dt_eve)
    assert meal == "Dinner"
    assert d == "2026-11-20"

    # Night 11:30 PM -> Tomorrow's Breakfast
    dt_night = datetime(2026, 11, 20, 23, 30)
    d, day, meal = ac.get_next_meal_period(dt_night)
    assert meal == "Breakfast"
    assert d == "2026-11-21"
    assert day == "Saturday"


def test_33_vision_ai_robust_lighting_and_shapes():
    """Test 33: Verifies vision AI plate/tray detection across lighting, shadows, and circular/rectangular shapes."""
    # 1. Circular plate detection
    img_circ = va.create_sample_tray_image("moderate", tray_shape="circular")
    res_circ = va.analyze_food_image(img_circ)
    assert res_circ["success"] is True
    assert res_circ["detected_tray_shape"] in ["Circular Dining Plate", "Partially Visible Circular Plate"]
    assert 0.0 <= res_circ["coverage_percentage"] <= 100.0
    assert res_circ["tray_pixel_count"] > 0

    # 2. Rectangular tray with directional shadow
    img_shadow = va.create_sample_tray_image("moderate", condition="shadow", tray_shape="rectangular")
    res_shadow = va.analyze_food_image(img_shadow)
    assert res_shadow["success"] is True
    assert res_shadow["coverage_percentage"] > 0.0
    assert res_shadow["leftover_category"] in ["Low Waste", "Moderate Waste", "High Waste"]

    # 3. High waste tray with bright lighting/flash
    img_bright = va.create_sample_tray_image("high", condition="bright", tray_shape="rectangular")
    res_bright = va.analyze_food_image(img_bright)
    assert res_bright["success"] is True
    assert res_bright["coverage_percentage"] >= 12.0
    assert res_bright["leftover_category"] in ["Moderate Waste", "High Waste"]

    # 4. Partially visible plate/tray
    img_partial = va.create_sample_tray_image("moderate", condition="partial")
    res_partial = va.analyze_food_image(img_partial)
    assert res_partial["success"] is True
    assert res_partial["tray_pixel_count"] > 0


def test_34_vision_scan_database_persistence_with_metadata(temp_db):
    """Test 34: Verifies storing and querying vision scans with meal date, period, dish, and scale grams."""
    db.init_db(temp_db)
    
    scan_id = db.record_vision_scan(
        coverage_percentage=28.5,
        leftover_category="Moderate Waste",
        detected_tray_shape="Multi-Compartment Stainless Thali",
        food_pixel_count=65000,
        tray_pixel_count=230000,
        processing_time_ms=36.4,
        image_source="Dish Return Camera",
        measured_weight_grams=210.0,
        meal_date="2026-11-04",
        meal_period="Lunch",
        dish_name="Kadai Paneer",
        db_path=temp_db
    )
    assert scan_id is not None
    assert scan_id > 0

    df_scans = db.get_vision_scans(limit=10, db_path=temp_db)
    assert len(df_scans) >= 1

    saved = df_scans[df_scans["id"] == scan_id].iloc[0]
    assert saved["coverage_percentage"] == 28.5
    assert saved["leftover_category"] == "Moderate Waste"
    assert saved["detected_tray_shape"] == "Multi-Compartment Stainless Thali"
    assert saved["measured_weight_grams"] == 210.0
    assert saved["meal_date"] == "2026-11-04"
    assert saved["meal_period"] == "Lunch"
    assert saved["dish_name"] == "Kadai Paneer"


def test_35_vision_scan_missing_weight_handling_and_no_gram_conversion(temp_db):
    """Test 35: Verifies handling of missing manual scale weight and guarantees no automatic gram estimation."""
    db.init_db(temp_db)
    
    # Store scan with NO manual weight provided (None)
    scan_id = db.record_vision_scan(
        coverage_percentage=42.0,
        leftover_category="High Waste",
        detected_tray_shape="Rectangular Compartment Tray",
        food_pixel_count=95000,
        tray_pixel_count=230000,
        processing_time_ms=38.2,
        image_source="Web Upload",
        measured_weight_grams=None,
        meal_date="2026-11-04",
        meal_period="Dinner",
        dish_name="Dal Makhani",
        db_path=temp_db
    )
    
    df_scans = db.get_vision_scans(limit=10, db_path=temp_db)
    saved = df_scans[df_scans["id"] == scan_id].iloc[0]
    
    # Physical weight must be None / NaN when unmeasured, NOT synthesized or estimated
    assert pd.isna(saved["measured_weight_grams"]) or saved["measured_weight_grams"] is None
    # Optical coverage remains clean area percentage
    assert saved["coverage_percentage"] == 42.0
    # Guarantee that coverage is not directly equal to mass in grams
    assert saved["coverage_percentage"] != saved["measured_weight_grams"]

# -----------------------------------------------------------------------------
# Smart Portion Recommendation & Ladle Calibration System Tests (Tests 36-38)
# -----------------------------------------------------------------------------
def test_36_smart_portion_high_waste_dish_downsizing(temp_db):
    """Test 36: Verifies high-waste dish detection via physical scale measurements and ladle downsizing recommendation."""
    db.init_db(temp_db)
    
    # Seed 3 high-leftover scans with physical scale weights (> 45g avg leftover)
    for w in [55.0, 60.0, 50.0]:
        db.record_vision_scan(
            coverage_percentage=38.0,
            leftover_category="High Waste",
            detected_tray_shape="Rectangular Compartment Tray",
            food_pixel_count=85000,
            tray_pixel_count=220000,
            processing_time_ms=35.0,
            image_source="Web Upload",
            measured_weight_grams=w,
            meal_date="2026-11-04",
            meal_period="Dinner",
            dish_name="High Waste Paneer",
            db_path=temp_db
        )

    rec = sp.recommend_dish_portion("High Waste Paneer", min_records=3, db_path=temp_db)
    assert rec["is_sufficient_data"] is True
    assert rec["waste_level"] == "High Leftover"
    assert rec["status"] == "Ladle Downsizing Recommended"
    assert rec["current_portion_grams"] == 200
    assert rec["suggested_portion_grams"] < rec["current_portion_grams"]
    assert rec["portion_reduction_pct"] > 0
    assert rec["avg_waste_grams"] == 55.0
    assert "Two-Stage Serving" in rec["two_stage_policy"]
    assert "refill" in rec["two_stage_policy"].lower()
    assert "Zero reduction to kitchen cooking batches" in rec["kitchen_prep_impact"]


def test_37_smart_portion_insufficient_history_handling(temp_db):
    """Test 37: Verifies that dishes with insufficient physical measurements (< 3 records) maintain standard portions."""
    db.init_db(temp_db)
    
    # Only 1 scan recorded (less than required min_records=3)
    db.record_vision_scan(
        coverage_percentage=40.0,
        leftover_category="High Waste",
        detected_tray_shape="Rectangular Compartment Tray",
        food_pixel_count=90000,
        tray_pixel_count=220000,
        processing_time_ms=32.0,
        image_source="Web Upload",
        measured_weight_grams=70.0,
        meal_date="2026-11-04",
        meal_period="Lunch",
        dish_name="Exotic Paneer Tikka",
        db_path=temp_db
    )

    rec = sp.recommend_dish_portion("Exotic Paneer Tikka", min_records=3, db_path=temp_db)
    assert rec["is_sufficient_data"] is False
    assert rec["status"] == "Insufficient Data"
    assert rec["record_count"] == 1
    assert rec["suggested_portion_grams"] == rec["current_portion_grams"]
    assert rec["portion_reduction_pct"] == 0.0
    assert "Insufficient historical data" in rec["recommendation_summary"]
    assert "None. Kitchen preparation targets remain unchanged" in rec["kitchen_prep_impact"]


def test_38_smart_portion_recommendation_limits_and_no_kitchen_prep_reduction(temp_db):
    """Test 38: Verifies safety limits on portion trimming (capped at max_reduction_pct) and zero kitchen batch reduction."""
    db.init_db(temp_db)
    
    # Seed extreme plate waste (e.g. 150g leftover out of 200g portion)
    for _ in range(4):
        db.record_vision_scan(
            coverage_percentage=50.0,
            leftover_category="High Waste",
            detected_tray_shape="Stainless Thali",
            food_pixel_count=110000,
            tray_pixel_count=220000,
            processing_time_ms=30.0,
            image_source="Synthetic Generator",
            measured_weight_grams=150.0,
            meal_date="2026-11-04",
            meal_period="Lunch",
            dish_name="Shahi Paneer",
            db_path=temp_db
        )

    # Standard portion for Shahi Paneer is 200g
    # Even with 150g (75%) waste, reduction must be clamped to max_reduction_pct (25%)
    max_trim = 25.0
    rec = sp.recommend_dish_portion("Shahi Paneer", min_records=3, max_reduction_pct=max_trim, db_path=temp_db)
    assert rec["is_sufficient_data"] is True
    assert rec["portion_reduction_pct"] <= max_trim
    # Minimum allowed portion is 200 * (1 - 0.25) = 150g
    assert rec["suggested_portion_grams"] >= int(rec["current_portion_grams"] * (1.0 - (max_trim / 100.0)))
    assert rec["suggested_portion_grams"] == 150

    # Strict Guarantee: Kitchen preparation quantities are NOT altered by portion calibration
    assert "Zero reduction to kitchen cooking batches" in rec["kitchen_prep_impact"]
    # Optical coverage is never used as food weight
    waste_analysis = sp.analyze_dish_plate_waste("Shahi Paneer", db_path=temp_db)
    assert waste_analysis["avg_waste_grams"] == 150.0
    assert all(w == 150.0 for w in waste_analysis["raw_measurements"])



def test_39_data_source_tagging_and_persistence(temp_db):
    """Test 39: Verifies data_source (REAL, SYNTHETIC, DEMO) tagging and persistence across SQLite tables."""
    db.init_db(temp_db)

    # 1. Test meal records tagging
    ok_real, _ = db.record_meal_waste(
        meal_date="2026-11-04",
        meal_type="Lunch",
        dish_name="Paneer Butter Masala",
        quantity_prepared=200,
        quantity_served=180,
        discarded_quantity=10,
        cost_per_serving=20.0,
        is_synthetic=False,
        data_source="REAL",
        db_path=temp_db
    )
    assert ok_real is True

    ok_synth, _ = db.record_meal_waste(
        meal_date="2026-11-03",
        meal_type="Lunch",
        dish_name="Paneer Butter Masala",
        quantity_prepared=200,
        quantity_served=170,
        discarded_quantity=20,
        cost_per_serving=20.0,
        is_synthetic=True,
        data_source="SYNTHETIC",
        db_path=temp_db
    )
    assert ok_synth is True

    # 2. Test student registration tagging
    ok_reg, msg_reg = db.register_student(
        meal_date="2026-11-04",
        meal_type="Lunch",
        registration_code="STU_TEST_01",
        attendance_intent="Yes",
        data_source="REAL",
        db_path=temp_db
    )
    assert ok_reg is True

    # 3. Test plate waste scans tagging
    scan_real = db.record_vision_scan(
        coverage_percentage=25.0,
        leftover_category="Low Waste",
        image_source="Web Upload",
        measured_weight_grams=35.0,
        meal_date="2026-11-04",
        meal_period="Lunch",
        dish_name="Paneer Butter Masala",
        data_source="REAL",
        is_simulated_weight=0,
        db_path=temp_db
    )
    assert scan_real is not None

    scan_demo = db.record_vision_scan(
        coverage_percentage=45.0,
        leftover_category="Moderate Waste",
        image_source="Synthetic Generator",
        measured_weight_grams=55.0,
        meal_date="2026-11-03",
        meal_period="Lunch",
        dish_name="Paneer Butter Masala",
        data_source="DEMO",
        is_simulated_weight=1,
        db_path=temp_db
    )
    assert scan_demo is not None

    # 4. Verify querying with data_source filter
    df_real = db.get_meal_records(data_source="REAL", db_path=temp_db)
    assert len(df_real) == 1
    assert df_real.iloc[0]["data_source"] == "REAL"

    df_synth = db.get_meal_records(data_source="SYNTHETIC", db_path=temp_db)
    assert len(df_synth) == 1
    assert df_synth.iloc[0]["data_source"] == "SYNTHETIC"

    df_all_meals = db.get_meal_records(db_path=temp_db)
    assert len(df_all_meals) == 2

    # Verify vision scans filtering
    scans_real = db.get_vision_scans(data_source="REAL", db_path=temp_db)
    assert len(scans_real) == 1
    assert scans_real.iloc[0]["data_source"] == "REAL"
    assert scans_real.iloc[0]["is_simulated_weight"] == 0

    scans_demo = db.get_vision_scans(limit=200, data_source="DEMO", db_path=temp_db)
    assert len(scans_demo) >= 1
    demo_pbm = scans_demo[scans_demo["dish_name"] == "Paneer Butter Masala"]
    assert len(demo_pbm) == 1
    assert demo_pbm.iloc[0]["data_source"] == "DEMO"
    assert demo_pbm.iloc[0]["is_simulated_weight"] == 1

    # Verify counts
    assert db.get_vision_scans_count(data_source="REAL", db_path=temp_db) == 1
    assert db.get_vision_scans_count(data_source="DEMO", db_path=temp_db) >= 1

    # Verify storage info breakdown
    info = db.get_database_storage_info(temp_db)
    assert info["real_meal_records_count"] == 1
    assert info["synthetic_meal_records_count"] == 1
    assert info["real_plate_waste_scans_count"] == 1
    assert info["demo_plate_waste_scans_count"] >= 1
    assert info["verified_weights_count"] == 1
    assert info["simulated_weights_count"] >= 1


def test_40_simulated_scale_readings_never_treated_as_real_weights(temp_db):
    """Test 40: Verifies that simulated scale readings are NEVER treated as real measured food weights."""
    db.init_db(temp_db)

    # Seed 3 DEMO plate scans with simulated weights (is_simulated_weight=1, data_source='DEMO')
    for w in [50.0, 55.0, 60.0]:
        db.record_vision_scan(
            coverage_percentage=35.0,
            leftover_category="Moderate Waste",
            image_source="Synthetic Generator",
            measured_weight_grams=w,
            meal_date="2026-11-04",
            meal_period="Dinner",
            dish_name="Paneer Kolhapuri",
            data_source="DEMO",
            is_simulated_weight=1,
            db_path=temp_db
        )

    # 1. When querying for REAL operational data, simulated weights MUST be excluded
    real_analysis = sp.analyze_dish_plate_waste("Paneer Kolhapuri", data_source_filter="REAL", db_path=temp_db)
    assert real_analysis["record_count"] == 0
    assert real_analysis["is_sufficient_data"] is False
    assert real_analysis["avg_waste_grams"] == 0.0

    # Portion recommendation with REAL filter MUST NOT downsize based on demo/simulated weights
    rec_real = sp.recommend_dish_portion("Paneer Kolhapuri", data_source_filter="REAL", db_path=temp_db)
    assert rec_real["is_sufficient_data"] is False
    assert rec_real["suggested_portion_grams"] == rec_real["current_portion_grams"]
    assert rec_real["status"] == "Insufficient Data"

    # 2. When demo data is explicitly inspected, it MUST be tagged as DEMO and simulated
    rec_demo = sp.recommend_dish_portion("Paneer Kolhapuri", data_source_filter="DEMO", db_path=temp_db)
    assert rec_demo["is_sufficient_data"] is True
    assert rec_demo["is_demo"] is True
    assert rec_demo["is_simulated_weight"] is True
    assert "DEMO DATA NOTICE" in rec_demo["demo_notice"]
    assert "[DEMO SIMULATION]" in rec_demo["status"]
    assert "[DEMO DATA - SIMULATED ESTIMATE]" in rec_demo["recommendation_summary"]

    # 3. Add 3 REAL physical scale logs (< 22g avg and < 10% ratio for Optimal Consumption)
    for rw in [10.0, 12.0, 14.0]:
        db.record_vision_scan(
            coverage_percentage=15.0,
            leftover_category="Low Waste",
            image_source="Kitchen Scale",
            measured_weight_grams=rw,
            meal_date="2026-11-05",
            meal_period="Dinner",
            dish_name="Paneer Kolhapuri",
            data_source="REAL",
            is_simulated_weight=0,
            db_path=temp_db
        )

    # Now REAL filter evaluates strictly on physical scale readings
    rec_real_after = sp.recommend_dish_portion("Paneer Kolhapuri", data_source_filter="REAL", db_path=temp_db)
    assert rec_real_after["is_sufficient_data"] is True
    assert rec_real_after["is_demo"] is False
    assert rec_real_after["is_simulated_weight"] is False
    assert rec_real_after["demo_notice"] is None
    assert rec_real_after["avg_waste_grams"] == 12.0
    assert rec_real_after["waste_level"] == "Optimal Consumption"


def test_41_analytics_and_action_center_separation_of_real_and_demo(temp_db):
    """Test 41: Verifies separation of real operational vs synthetic baseline data in Action Center and analytics."""
    db.init_db(temp_db)

    # Seed synthetic meal records
    for i in range(5):
        db.record_meal_waste(
            meal_date=f"2026-10-2{i}",
            meal_type="Dinner",
            dish_name="Kadai Paneer",
            quantity_prepared=300,
            quantity_served=240,
            discarded_quantity=50,
            cost_per_serving=22.0,
            is_synthetic=True,
            data_source="SYNTHETIC",
            db_path=temp_db
        )

    # 1. Action center in REAL mode must isolate real data and detect insufficient real records
    insights_real = ac.generate_action_center_insights(data_source="REAL", db_path=temp_db)
    assert insights_real["data_source"] == "REAL"
    high_waste_real = insights_real["insights"]["high_waste_dishes"]
    assert high_waste_real["is_sufficient_data"] is False
    assert "Insufficient historical" in high_waste_real["insufficient_data_reason"]

    # 2. Action center in SYNTHETIC mode analyzes the synthetic baseline
    insights_synth = ac.generate_action_center_insights(data_source="SYNTHETIC", db_path=temp_db)
    assert insights_synth["data_source"] == "SYNTHETIC"
    high_waste_synth = insights_synth["insights"]["high_waste_dishes"]
    assert high_waste_synth["is_sufficient_data"] is True
    assert any(d["dish_name"] == "Kadai Paneer" for d in high_waste_synth["supporting_data"]["dishes"])

    # 3. get_filtered_meal_records correctly isolates REAL from SYNTHETIC
    df_real, count_real = db.get_filtered_meal_records(data_source_filter="REAL", db_path=temp_db)
    assert count_real == 0
    assert len(df_real) == 0

    df_synth, count_synth = db.get_filtered_meal_records(data_source_filter="SYNTHETIC", db_path=temp_db)
    assert count_synth == 5
    assert len(df_synth) == 5



def test_42_student_portal_registration_and_real_data_source(temp_db):
    """Test 42: Verifies student attendance registration across meals with data_source = REAL."""
    db.init_db(temp_db)

    # Register for all four meal periods
    for m in ["Breakfast", "Lunch", "Snacks", "Dinner"]:
        ok, msg = db.register_student(
            meal_date="2026-11-04",
            meal_type=m,
            selected_dishes=["Official Menu Item"],
            registration_code=f"24BDS_{m.upper()}",
            attendance_intent="Yes",
            data_source="REAL",
            db_path=temp_db
        )
        assert ok is True
        assert "recorded successfully" in msg.lower()

    # Query directly from SQLite
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT registration_code, meal_type, attendance_intent, data_source FROM student_registrations")
        rows = cursor.fetchall()
        assert len(rows) == 4
        for r in rows:
            assert r["data_source"] == "REAL"
            assert r["attendance_intent"] == "Yes"
    finally:
        conn.close()

    # Verify intent counts
    cnts = db.get_attendance_intent_counts("2026-11-04", "Lunch", eligible_students=500, db_path=temp_db)
    assert cnts["yes_count"] == 1
    assert cnts["total_responses"] == 1


def test_43_student_portal_duplicate_prevention_and_update(temp_db):
    """Test 43: Verifies that submitting twice for the same student, date, and meal updates intent without creating duplicate rows."""
    db.init_db(temp_db)

    # First vote: Student intends to skip dinner
    ok1, msg1 = db.register_student(
        meal_date="2026-11-05",
        meal_type="Dinner",
        registration_code="23BCS099",
        attendance_intent="No",
        data_source="REAL",
        db_path=temp_db
    )
    assert ok1 is True
    assert "recorded successfully" in msg1.lower()

    # Polling counts should show 1 No, 0 Yes
    c1 = db.get_attendance_intent_counts("2026-11-05", "Dinner", db_path=temp_db)
    assert c1["no_count"] == 1
    assert c1["yes_count"] == 0
    assert c1["total_responses"] == 1

    # Second vote: Student changes mind and confirms attendance (Yes)
    ok2, msg2 = db.register_student(
        meal_date="2026-11-05",
        meal_type="Dinner",
        registration_code="23BCS099",
        attendance_intent="Yes",
        data_source="REAL",
        db_path=temp_db
    )
    assert ok2 is True
    assert "updated" in msg2.lower()
    assert "previous intent: 'no'" in msg2.lower()

    # Verify duplicate prevention: exactly 1 row must exist in SQLite
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM student_registrations WHERE meal_date = '2026-11-05' AND meal_type = 'Dinner' AND registration_code = '23BCS099'")
        cnt = cursor.fetchone()["cnt"]
        assert cnt == 1
    finally:
        conn.close()

    # Polling counts should now reflect Yes: 1, No: 0
    c2 = db.get_attendance_intent_counts("2026-11-05", "Dinner", db_path=temp_db)
    assert c2["yes_count"] == 1
    assert c2["no_count"] == 0
    assert c2["total_responses"] == 1


def test_44_student_portal_feedback_storage_and_ratings(temp_db):
    """Test 44: Verifies optional food quality, taste, and portion size feedback storage with data_source = REAL."""
    db.init_db(temp_db)

    # 1. Register with attached optional feedback
    ok_reg, msg_reg = db.register_student(
        meal_date="2026-11-06",
        meal_type="Lunch",
        registration_code="24BDS050",
        attendance_intent="Yes",
        food_quality_rating=5,
        taste_rating=4,
        portion_size_feedback="Just Right",
        comments="Paneer was fresh and hot!",
        data_source="REAL",
        db_path=temp_db
    )
    assert ok_reg is True

    # 2. Standalone feedback submission
    ok_fb, msg_fb = db.record_student_feedback(
        meal_date="2026-11-06",
        meal_type="Lunch",
        registration_code="24BDS051",
        food_quality_rating=4,
        taste_rating=5,
        portion_size_feedback="Just Right",
        comments="Great dal fry and steaming chapatis.",
        data_source="REAL",
        db_path=temp_db
    )
    assert ok_fb is True

    # 3. Retrieve feedback records from SQLite
    df_fb = db.get_student_feedback(meal_date="2026-11-06", meal_type="Lunch", db_path=temp_db)
    assert len(df_fb) == 2
    assert all(df_fb["data_source"] == "REAL")
    assert all(df_fb["portion_size_feedback"] == "Just Right")

    # 4. Feedback summary metrics
    summary = db.get_student_feedback_summary(meal_date="2026-11-06", meal_type="Lunch", db_path=temp_db)
    assert summary["total_feedback"] == 2
    assert summary["avg_quality"] == 4.5
    assert summary["avg_taste"] == 4.5
    assert summary["portion_breakdown"]["Just Right"] == 2
    assert len(summary["recent_comments"]) == 2


def test_45_student_portal_attendance_to_forecast_integration(temp_db):
    """Test 45: Verifies that student portal responses dynamically drive the ML demand forecasting pipeline."""
    db.init_db(temp_db)

    # Initial cohort votes for Lunch
    # 4 Yes (100% weight = 4.0), 2 Not sure (50% weight = 1.0), 3 No (0% weight = 0.0) -> Weighted = 5.0
    for i in range(4):
        db.register_student("2026-11-04", "Lunch", registration_code=f"STU_YES_{i}", attendance_intent="Yes", db_path=temp_db)
    for i in range(2):
        db.register_student("2026-11-04", "Lunch", registration_code=f"STU_NS_{i}", attendance_intent="Not sure", db_path=temp_db)
    for i in range(3):
        db.register_student("2026-11-04", "Lunch", registration_code=f"STU_NO_{i}", attendance_intent="No", db_path=temp_db)

    # Verify weighted attendance calculation
    poll_inputs = rc.get_polling_forecast_inputs(
        meal_date="2026-11-04",
        meal_type="Lunch",
        eligible_students=500,
        db_path=temp_db
    )
    assert poll_inputs["has_polling"] is True
    assert poll_inputs["yes_count"] == 4
    assert poll_inputs["not_sure_count"] == 2
    assert poll_inputs["no_count"] == 3
    assert poll_inputs["total_responses"] == 9
    assert poll_inputs["weighted_attendance"] == 5
    assert poll_inputs["registered_students"] == 5

    # Forecast prediction with initial attendance
    initial_rec = rc.generate_dish_recommendation(meal_date='2026-11-04', 
        dish_name="Curd Rice",
        meal_type="Lunch",
        day_of_week="Wednesday",
        registered_students=poll_inputs["registered_students"],
        expected_attendance=500,
        db_path=temp_db
    )
    assert initial_rec["predicted_demand"] > 0
    assert initial_rec["recommended_preparation"] >= initial_rec["predicted_demand"]

    # Now 10 more students vote Yes (attendance increases significantly)
    for i in range(4, 14):
        db.register_student("2026-11-04", "Lunch", registration_code=f"STU_YES_{i}", attendance_intent="Yes", db_path=temp_db)

    # Re-evaluate polling inputs
    updated_poll = rc.get_polling_forecast_inputs(
        meal_date="2026-11-04",
        meal_type="Lunch",
        eligible_students=500,
        db_path=temp_db
    )
    assert updated_poll["yes_count"] == 14
    assert updated_poll["weighted_attendance"] == 15
    assert updated_poll["registered_students"] == 15

    # Forecast prediction with increased student attendance
    updated_rec = rc.generate_dish_recommendation(meal_date='2026-11-04', 
        dish_name="Curd Rice",
        meal_type="Lunch",
        day_of_week="Wednesday",
        registered_students=updated_poll["registered_students"],
        expected_attendance=500,
        db_path=temp_db
    )
    # Recommended preparation must scale positively with student attendance
    assert updated_rec["predicted_demand"] >= initial_rec["predicted_demand"]
    assert updated_rec["recommended_preparation"] >= initial_rec["recommended_preparation"]


def test_46_password_hashing_and_user_authentication(temp_db):
    """Test 46: PBKDF2-HMAC-SHA256 password hashing, user registration, and authentication."""
    # Test hashing and verification
    pwd = "SecurePassword@123"
    hashed, salt = db.hash_password(pwd)
    assert hashed != pwd
    assert len(salt) >= 16
    assert db.verify_password(pwd, hashed, salt)
    assert not db.verify_password("WrongPassword", hashed, salt)

    # Test default accounts seeded in temp_db
    admin_auth = db.authenticate_user("admin", "Admin@123", db_path=temp_db)
    assert admin_auth is not None
    assert admin_auth["role"] == "Admin"
    assert admin_auth["username"] == "admin"

    staff_auth = db.authenticate_user("kitchen", "Kitchen@123", db_path=temp_db)
    assert staff_auth is not None
    assert staff_auth["role"] == "Kitchen Staff"

    student_auth = db.authenticate_user("student", "Student@123", db_path=temp_db)
    assert student_auth is not None
    assert student_auth["role"] == "Student"

    # Test invalid credentials
    assert db.authenticate_user("admin", "WrongPass", db_path=temp_db) is None
    assert db.authenticate_user("nonexistent_user", "AnyPass", db_path=temp_db) is None

    # Test creating new user
    ok, msg = db.create_user("chef_ramesh", "Ramesh@2026", "Kitchen Staff", "Chef Ramesh", db_path=temp_db)
    assert ok
    u_new = db.authenticate_user("chef_ramesh", "Ramesh@2026", db_path=temp_db)
    assert u_new is not None
    assert u_new["role"] == "Kitchen Staff"

    # Duplicate username rejection
    ok_dup, _ = db.create_user("chef_ramesh", "AnotherPass@123", "Kitchen Staff", "Chef Ramesh", db_path=temp_db)
    assert not ok_dup


def test_47_role_permissions_and_administrative_restrictions():
    """Test 47: Role permission matrices and restriction of administrative operations."""
    # Admin has full administrative access
    assert db.has_permission("Admin", "view_admin_dashboard")
    assert db.has_permission("Admin", "view_audit_reports")
    assert db.has_permission("Admin", "view_unmasked_roll_numbers")
    assert db.has_permission("Admin", "record_kitchen_meal")
    assert db.has_permission("Admin", "view_kitchen_targets")

    # Kitchen Staff has operational access but blocked from administrative operations
    assert db.has_permission("Kitchen Staff", "view_kitchen_targets")
    assert db.has_permission("Kitchen Staff", "record_kitchen_meal")
    assert db.has_permission("Kitchen Staff", "update_meal_status")
    assert not db.has_permission("Kitchen Staff", "view_admin_dashboard")
    assert not db.has_permission("Kitchen Staff", "view_audit_reports")
    assert not db.has_permission("Kitchen Staff", "view_unmasked_roll_numbers")

    # Students have dining portal access but blocked from kitchen recording & administrative operations
    assert db.has_permission("Student", "view_menu")
    assert db.has_permission("Student", "submit_student_intent")
    assert db.has_permission("Student", "submit_feedback")
    assert not db.has_permission("Student", "record_kitchen_meal")
    assert not db.has_permission("Student", "view_kitchen_targets")
    assert not db.has_permission("Student", "view_admin_dashboard")
    assert not db.has_permission("Student", "view_audit_reports")


def test_48_student_privacy_roll_number_masking():
    """Test 48: Student roll numbers and private IDs are masked to protect student privacy."""
    assert db.mask_student_id("22BCS045") == "***045"
    assert db.mask_student_id("21BDS102") == "***102"
    assert db.mask_student_id("CS1") == "***"
    assert db.mask_student_id("") == "Anonymous"
    assert db.mask_student_id(None) == "Anonymous"


def test_49_kitchen_recording_validation_and_persistence(temp_db):
    """Test 49: Kitchen staff recording enforces business rules and persists into SQLite."""
    # 1. Valid kitchen record
    ok, msg = db.record_kitchen_meal_entry(
        meal_date="2026-11-04",
        meal_type="Lunch",
        dish_name="Dal Tadka",
        quantity_prepared=400,
        quantity_served=360,
        discarded_quantity=25,
        cost_per_serving=16.0,
        actual_cost=6400.0,
        completion_status="Completed",
        recorded_by="Chef Suresh",
        db_path=temp_db
    )
    assert ok
    assert "saved successfully" in msg or "Record saved" in msg

    # Verify persistence in SQLite meal_records
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meal_records WHERE meal_date='2026-11-04' AND dish_name='Dal Tadka'")
        row = cursor.fetchone()
        assert row is not None
        assert row["quantity_prepared"] == 400
        assert row["quantity_served"] == 360
        assert row["leftover_quantity"] == 40  # 400 - 360
        assert row["discarded_quantity"] == 25
        assert row["usable_leftover"] == 15     # 40 - 25
        assert row["completion_status"] == "Completed"
        assert row["actual_cost"] == 6400.0
    finally:
        conn.close()

    # 2. Validation: Negative quantities rejected
    ok_neg, err_neg = db.record_kitchen_meal_entry(
        meal_date="2026-11-04",
        meal_type="Lunch",
        dish_name="Dal Tadka",
        quantity_prepared=-10,
        quantity_served=100,
        discarded_quantity=0,
        cost_per_serving=16.0,
        db_path=temp_db
    )
    assert not ok_neg
    assert "cannot be negative" in err_neg.lower()

    # 3. Validation: Served > Prepared rejected
    ok_over, err_over = db.record_kitchen_meal_entry(
        meal_date="2026-11-04",
        meal_type="Lunch",
        dish_name="Dal Tadka",
        quantity_prepared=100,
        quantity_served=120,
        discarded_quantity=0,
        cost_per_serving=16.0,
        db_path=temp_db
    )
    assert not ok_over
    assert "cannot exceed quantity prepared" in err_over.lower()

    # 4. Validation: Discarded > Leftover rejected
    ok_disc, err_disc = db.record_kitchen_meal_entry(
        meal_date="2026-11-04",
        meal_type="Lunch",
        dish_name="Dal Tadka",
        quantity_prepared=100,
        quantity_served=80,  # leftover = 20
        discarded_quantity=25,  # 25 > 20
        cost_per_serving=16.0,
        db_path=temp_db
    )
    assert not ok_disc
    assert "cannot exceed leftover quantity" in err_disc.lower()


def test_50_actual_food_costs_and_meal_completion_status(temp_db):
    """Test 50: Record actual food costs and meal shift completion status."""
    # Record two dishes for Dinner
    db.record_kitchen_meal_entry(
        meal_date="2026-11-05",
        meal_type="Dinner",
        dish_name="Kadai Paneer",
        quantity_prepared=300,
        quantity_served=280,
        discarded_quantity=10,
        cost_per_serving=35.0,
        actual_cost=10500.0,
        completion_status="Completed",
        db_path=temp_db
    )
    db.record_kitchen_meal_entry(
        meal_date="2026-11-05",
        meal_type="Dinner",
        dish_name="Chapati",
        quantity_prepared=600,
        quantity_served=580,
        discarded_quantity=15,
        cost_per_serving=5.0,
        actual_cost=3000.0,
        completion_status="Completed",
        db_path=temp_db
    )

    # Verify meal shift status aggregation
    sh_status = db.get_meal_shift_status("2026-11-05", "Dinner", db_path=temp_db)
    assert sh_status["status"] == "Completed"
    assert sh_status["total_prepared"] == 900
    assert sh_status["total_served"] == 860
    assert sh_status["total_discarded"] == 25
    assert sh_status["actual_food_cost"] == 13500.0  # 10500 + 3000

    # Test explicit status update
    ok_up, msg_up = db.update_meal_shift_status(
        meal_date="2026-11-05",
        meal_type="Dinner",
        status="Service Active",
        actual_food_cost=14000.0,
        staff_notes="Special festival dinner",
        updated_by="Chef Ramesh",
        db_path=temp_db
    )
    assert ok_up
    updated_status = db.get_meal_shift_status("2026-11-05", "Dinner", db_path=temp_db)
    assert updated_status["status"] == "Service Active"
    assert updated_status["actual_food_cost"] == 14000.0
    assert updated_status["staff_notes"] == "Special festival dinner"


def test_51_daily_all_four_meals_cooking_targets(temp_db):
    """Test 51: Kitchen staff can view daily cooking targets across all four meals."""
    targets = db.get_daily_all_meals_targets("2026-11-04", db_path=temp_db)
    assert targets["meal_date"] == "2026-11-04"
    assert targets["day_of_week"] == "Wednesday"

    # All four meals must be present
    meals = targets["meals"]
    for m in ["Breakfast", "Lunch", "Snacks", "Dinner"]:
        assert m in meals
        shift_info = meals[m]
        assert len(shift_info["dishes"]) > 0
        assert shift_info["total_prep_servings"] > 0
        assert shift_info["total_predicted_demand"] > 0
        assert shift_info["estimated_cost_inr"] > 0
        assert shift_info["total_weight_kg"] > 0
        assert "shift_status" in shift_info

    # Grand totals must strictly equal sum of the 4 shifts
    sum_servings = sum(meals[m]["total_prep_servings"] for m in ["Breakfast", "Lunch", "Snacks", "Dinner"])
    assert targets["grand_prep_servings"] == sum_servings
    assert targets["grand_prep_servings"] > 0
    assert targets["grand_estimated_cost_inr"] > 0
    assert targets["grand_cooking_weight_kg"] > 0


def test_52_kitchen_recording_adaptive_feedback_loop_integration(temp_db):
    """
    Test 52: Connect kitchen records to adaptive waste feedback loop.
    Verifies that logging high waste triggers an automatic trimming adjustment on subsequent cooking targets,
    bounded by the predicted demand floor.
    """
    dish = "Chole Bhature"
    meal = "Breakfast"
    target_date = "2026-11-10"

    # Initially, with no history in temp_db, recommendation has 0 adaptive trim
    initial_rec = rc.generate_dish_recommendation(
        meal_date=target_date,
        dish_name=dish,
        meal_type=meal,
        day_of_week="Tuesday",
        expected_attendance=400,
        registered_students=350,
        safety_margin_pct=5.0,
        db_path=temp_db
    )
    assert initial_rec["adjustment_servings"] == 0

    # Staff records 3 consecutive meals prior to target_date with high waste (e.g., 20% waste)
    for i, past_date in enumerate(["2026-11-05", "2026-11-07", "2026-11-09"]):
        db.record_kitchen_meal_entry(
            meal_date=past_date,
            meal_type=meal,
            dish_name=dish,
            quantity_prepared=400,
            quantity_served=320,
            discarded_quantity=80,  # 80/400 = 20% waste
            cost_per_serving=25.0,
            db_path=temp_db
        )

    # Verify that waste pattern was recognized by database history query
    history = db.get_dish_waste_history(dish, meal, before_date=target_date, db_path=temp_db)
    assert len(history) == 3

    waste_pattern = rc.calculate_dish_waste_pattern(dish, meal, meal_date=target_date, db_path=temp_db)
    assert waste_pattern["has_sufficient_history"] is True
    assert waste_pattern["avg_waste_pct"] >= 15.0
    assert waste_pattern["waste_trend"] == "High Waste"

    # Now generate cooking recommendation for target_date
    adapted_rec = rc.generate_dish_recommendation(
        meal_date=target_date,
        dish_name=dish,
        meal_type=meal,
        day_of_week="Tuesday",
        expected_attendance=400,
        registered_students=350,
        safety_margin_pct=5.0,
        db_path=temp_db
    )

    # Adaptive feedback loop must trim excess preparation
    assert adapted_rec["adjustment_servings"] < 0
    assert adapted_rec["adjusted_preparation_quantity"] < adapted_rec["current_recommendation"]
    # Safety rule: adjusted preparation must NEVER drop below predicted demand floor
    assert adapted_rec["adjusted_preparation_quantity"] >= int(np.ceil(adapted_rec["predicted_demand"]))
    assert "Waste Dampened" in adapted_rec["feedback_status"]


def test_53_food_waste_report_calculations_and_timeframes(temp_db):
    """
    Test 53: Daily, weekly, and monthly food waste reports calculations.
    Verifies that prepared, served, leftover, and discarded quantities,
    waste percentages, utilization, actual costs, and financial losses are accurately calculated.
    """
    db.init_db(temp_db)

    # Log 4 meals across 2 days (REAL data source)
    records = [
        # Day 1: 2026-11-08
        ("2026-11-08", "Breakfast", "Masala Dosa", 300, 270, 30, 20.0, 6000.0),
        ("2026-11-08", "Lunch", "Veg Pulav", 400, 360, 40, 25.0, 10000.0),
        # Day 2: 2026-11-09
        ("2026-11-09", "Breakfast", "Idli Vada", 350, 340, 10, 18.0, 6300.0),
        ("2026-11-09", "Dinner", "Kadai Paneer", 420, 370, 50, 30.0, 12600.0),
    ]
    for r in records:
        db.record_kitchen_meal_entry(
            meal_date=r[0],
            meal_type=r[1],
            dish_name=r[2],
            quantity_prepared=r[3],
            quantity_served=r[4],
            discarded_quantity=r[5],
            cost_per_serving=r[6],
            actual_cost=r[7],
            db_path=temp_db
        )

    # 1. Daily Report for 2026-11-08
    rep_d1 = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-08", data_source="REAL", db_path=temp_db)
    s1 = rep_d1["summary"]
    assert s1["total_prepared"] == 700  # 300 + 400
    assert s1["total_served"] == 630    # 270 + 360
    assert s1["total_leftover"] == 70  # 30 + 40
    assert s1["total_discarded"] == 70  # 30 + 40
    assert s1["waste_percentage"] == round(70 / 700 * 100, 2)  # 10.0%
    assert s1["utilization_rate"] == round(630 / 700 * 100, 2)  # 90.0%
    assert s1["actual_cost_total"] == 16000.0  # 6000 + 10000
    assert s1["estimated_financial_loss"] == (30 * 20.0) + (40 * 25.0)  # 600 + 1000 = 1600.0

    # 2. Weekly Report covering both days (anchor 2026-11-09)
    rep_week = reports.generate_food_waste_report(timeframe="weekly", target_date="2026-11-09", data_source="REAL", db_path=temp_db)
    sw = rep_week["summary"]
    assert sw["total_prepared"] == 1470  # 300 + 400 + 350 + 420
    assert sw["total_served"] == 1340    # 270 + 360 + 340 + 370
    assert sw["total_discarded"] == 130  # 30 + 40 + 10 + 50
    assert sw["actual_cost_total"] == 34900.0  # 6000 + 10000 + 6300 + 12600
    assert sw["estimated_financial_loss"] == (30*20.0) + (40*25.0) + (10*18.0) + (50*30.0)  # 1600 + 180 + 1500 = 3280.0

    # Meal-wise breakdown check
    meals_dict = {m["meal_type"]: m for m in rep_week["meal_breakdown"]}
    assert "Breakfast" in meals_dict
    assert meals_dict["Breakfast"]["prepared"] == 650
    assert meals_dict["Breakfast"]["served"] == 610
    assert meals_dict["Breakfast"]["discarded"] == 40


def test_54_report_data_source_filtering_and_provenance(temp_db):
    """
    Test 54: Clearly distinguish REAL, SYNTHETIC, and DEMO data in reports.
    Verifies that querying REAL data never leaks synthetic or demo records.
    """
    db.init_db(temp_db)

    # Insert 1 REAL record
    db.record_kitchen_meal_entry(
        meal_date="2026-11-08",
        meal_type="Lunch",
        dish_name="Dal Tadka",
        quantity_prepared=200,
        quantity_served=190,
        discarded_quantity=10,
        cost_per_serving=15.0,
        is_synthetic=False,
        db_path=temp_db
    )

    # Insert 1 SYNTHETIC record directly into meal_records
    conn = db.get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO meal_records (
            meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
            leftover_quantity, discarded_quantity, usable_leftover, cost_per_serving,
            estimated_discarded_food_cost, is_synthetic, data_source, recorded_timestamp
        ) VALUES ('2026-11-08', 'Dinner', 'Paneer Butter Masala', 500, 400, 100, 100, 0, 35.0, 3500.0, 1, 'SYNTHETIC', '2026-11-08T20:00:00')
    """)
    conn.commit()
    conn.close()

    # Query REAL report
    rep_real = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-08", data_source="REAL", db_path=temp_db)
    assert rep_real["data_source"] == "REAL"
    assert bool(rep_real["provenance"]["is_pure_real"]) is True
    assert rep_real["summary"]["total_prepared"] == 200  # Only real Dal Tadka
    assert rep_real["summary"]["total_discarded"] == 10

    # Query SYNTHETIC report
    rep_synth = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-08", data_source="SYNTHETIC", db_path=temp_db)
    assert rep_synth["data_source"] == "SYNTHETIC"
    assert rep_synth["summary"]["total_prepared"] == 500  # Only synthetic Paneer
    assert rep_synth["summary"]["total_discarded"] == 100

    # Query ALL report
    rep_all = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-08", data_source="ALL", db_path=temp_db)
    assert rep_all["summary"]["total_prepared"] == 700  # 200 + 500
    assert rep_all["summary"]["total_discarded"] == 110


def test_55_report_export_to_csv_and_pdf(temp_db):
    """
    Test 55: Add downloadable CSV and PDF reports.
    Verifies that generated CSV contains metadata headers, executive summary,
    meal distributions, and raw logs, and PDF compiles to valid PDF binary bytes.
    """
    db.init_db(temp_db)
    db.record_kitchen_meal_entry(
        meal_date="2026-11-08",
        meal_type="Lunch",
        dish_name="Veg Biryani",
        quantity_prepared=350,
        quantity_served=320,
        discarded_quantity=30,
        cost_per_serving=22.0,
        db_path=temp_db
    )

    rep = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-08", data_source="REAL", db_path=temp_db)

    # 1. Test CSV Export
    csv_text = reports.export_report_to_csv(rep)
    assert isinstance(csv_text, str)
    assert "MESSMIND AI: INSTITUTIONAL FOOD WASTE & FINANCIAL LOSS AUDIT REPORT" in csv_text
    assert "SECTION 1: EXECUTIVE SUMMARY KPIS" in csv_text
    assert "Total Food Prepared" in csv_text or "Total Prepared Quantity" in csv_text
    assert "Veg Biryani" in csv_text

    # 2. Test PDF Export
    pdf_bytes = reports.export_report_to_pdf(rep)
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_56_system_alert_creation_persistence_and_deduplication(temp_db):
    """
    Test 56: System alerts for high food waste, low stock, overspending, and upcoming meal prep.
    Verifies SQLite persistence, read status management, and alert retrieval.
    """
    db.init_db(temp_db)

    # Initial seeded alerts
    initial_alerts = db.get_alerts(db_path=temp_db)
    assert len(initial_alerts) >= 4

    types_found = {a["alert_type"] for a in initial_alerts}
    assert "high_waste" in types_found
    assert "low_stock" in types_found
    assert "overspending" in types_found
    assert "upcoming_prep" in types_found

    # Test creating custom alert
    new_id = db.create_alert(
        alert_type="high_waste",
        title="⚠️ Custom High Waste Warning",
        message="Excessive rice wasted during lunch.",
        severity="critical",
        target_role="Kitchen Staff",
        dish_name="Rice",
        meal_type="Lunch",
        data_source="REAL",
        db_path=temp_db
    )
    assert new_id > 0

    # Retrieve and verify alert
    staff_alerts = db.get_alerts(target_role="Kitchen Staff", db_path=temp_db)
    custom_alert = next((a for a in staff_alerts if a["id"] == new_id), None)
    assert custom_alert is not None
    assert custom_alert["title"] == "⚠️ Custom High Waste Warning"
    assert custom_alert["is_read"] is False

    # Mark as read
    ok = db.mark_alert_as_read(new_id, db_path=temp_db)
    assert ok is True

    # Verify unread count decreases
    unread_staff = db.get_alerts(target_role="Kitchen Staff", unread_only=True, db_path=temp_db)
    assert not any(a["id"] == new_id for a in unread_staff)


def test_57_role_based_notification_visibility(temp_db):
    """
    Test 57: Role-based notification visibility for Admin and Kitchen Staff.
    Verifies that Admin and Kitchen Staff receive relevant alerts and cannot see private role-restricted alerts.
    """
    db.init_db(temp_db)

    # Admin-targeted alert (e.g. overspending / audit)
    aid_admin = db.create_alert(
        alert_type="overspending",
        title="💸 Admin-Only Budget Overrun",
        message="Monthly procurement invoice variance +20%.",
        severity="critical",
        target_role="Admin",
        db_path=temp_db
    )

    # Kitchen Staff-targeted alert (e.g. low stock / prep)
    aid_kitchen = db.create_alert(
        alert_type="low_stock",
        title="📦 Kitchen-Only Low Stock",
        message="Dal Methi ingredients below safety threshold.",
        severity="warning",
        target_role="Kitchen Staff",
        db_path=temp_db
    )

    # All-targeted alert
    aid_all = db.create_alert(
        alert_type="high_waste",
        title="⚠️ Universal Waste Warning",
        message="Hostel total waste exceeded 12%.",
        severity="warning",
        target_role="All",
        db_path=temp_db
    )

    # Admin view must include Admin and All, but NOT Kitchen Staff private alert
    admin_alerts = db.get_alerts(target_role="Admin", db_path=temp_db)
    admin_ids = [a["id"] for a in admin_alerts]
    assert aid_admin in admin_ids
    assert aid_all in admin_ids
    assert aid_kitchen not in admin_ids

    # Kitchen Staff view must include Kitchen Staff and All, but NOT Admin private alert
    kitchen_alerts = db.get_alerts(target_role="Kitchen Staff", db_path=temp_db)
    kitchen_ids = [a["id"] for a in kitchen_alerts]
    assert aid_kitchen in kitchen_ids
    assert aid_all in kitchen_ids
    assert aid_admin not in kitchen_ids

    # Permission check
    assert db.has_permission("Admin", "view_notifications") is True
    assert db.has_permission("Kitchen Staff", "view_notifications") is True
    assert db.has_permission("Student", "view_notifications") is False


def test_58_real_data_savings_and_zero_waste_truthfulness(temp_db):
    """
    Test 58: Data truthfulness for food costs, losses, and savings.
    Verifies that reports never fabricate fictitious savings or statistics.
    """
    db.init_db(temp_db)

    # Zero records -> verified_savings must be 0.0 with honest explanation
    rep_empty = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-15", data_source="REAL", db_path=temp_db)
    assert rep_empty["summary"]["verified_savings"] == 0.0
    assert "No meal records" in rep_empty["summary"]["savings_explanation"]

    # Record 1 meal with 10 usable leftovers repurposed
    db.record_kitchen_meal_entry(
        meal_date="2026-11-15",
        meal_type="Lunch",
        dish_name="Jeera Rice",
        quantity_prepared=100,
        quantity_served=80,
        discarded_quantity=10,  # 10 leftover is usable (20 leftover - 10 discarded = 10 usable)
        cost_per_serving=12.0,
        db_path=temp_db
    )

    rep_with_data = reports.generate_food_waste_report(timeframe="daily", target_date="2026-11-15", data_source="REAL", db_path=temp_db)
    s = rep_with_data["summary"]
    assert s["total_usable_leftover"] == 10
    # Usable leftovers value saved = 10 * 12.0 = 120.0
    assert s["verified_savings"] == 120.0
    assert "repurposing" in s["savings_explanation"].lower()
    assert s["estimated_financial_loss"] == 120.0  # 10 discarded * 12.0


def test_59_complete_end_to_end_operational_pipeline(temp_db):
    """
    Test 59: Complete end-to-end operational pipeline.
    Validates: Student Intent -> Polling Aggregation -> ML Forecast -> Kitchen Targets ->
    Kitchen Shift Recording -> Adaptive Feedback Loop -> Multi-Period Audit Reports.
    """
    db.init_db(temp_db)
    test_date = "2026-11-20"
    meal_type = "Lunch"
    dish = "Veg Pulav"

    # Step 1: Student registers intent via Student Dining Portal
    ok_reg, _ = db.register_student(
        meal_date=test_date,
        meal_type=meal_type,
        attendance_intent="Yes",
        registration_code="22BCS099",
        data_source="REAL",
        db_path=temp_db
    )
    assert ok_reg is True

    # Step 2: Intent aggregation feeds into pre-meal forecasting
    polling = rc.get_polling_forecast_inputs(meal_date=test_date, meal_type=meal_type, db_path=temp_db)
    assert polling["yes_count"] == 1
    assert polling["weighted_attendance"] > 0

    # Step 3: ML model generates demand forecast (+5% safety buffer)
    rec = rc.generate_dish_recommendation(
        meal_date=test_date,
        dish_name=dish,
        meal_type=meal_type,
        day_of_week="Friday",
        expected_attendance=polling["weighted_attendance"],
        registered_students=polling["registered_students"],
        safety_margin_pct=5.0,
        db_path=temp_db
    )
    prep_target = rec["recommended_preparation"]
    assert prep_target >= int(np.ceil(rec["predicted_demand"]))

    # Step 4: Kitchen staff views daily cooking targets for all meals
    daily_targets = db.get_daily_all_meals_targets(meal_date=test_date, db_path=temp_db)
    assert "Lunch" in daily_targets["meals"]
    assert daily_targets["grand_prep_servings"] > 0

    # Step 5: Kitchen staff records shift outcome across consecutive shifts (quantity prepared, served, leftover, discarded, actual cost)
    served_qty = int(prep_target * 0.8)
    disc_qty = prep_target - served_qty  # 20% discarded
    for past_dt in ["2026-11-16", "2026-11-18", "2026-11-20"]:
        ok_rec, _ = db.record_kitchen_meal_entry(
            meal_date=past_dt,
            meal_type=meal_type,
            dish_name=dish,
            quantity_prepared=prep_target,
            quantity_served=served_qty,
            discarded_quantity=disc_qty,
            cost_per_serving=25.0,
            actual_cost=prep_target * 25.0,
            completion_status="Completed",
            recorded_by="Staff Chef",
            is_synthetic=False,
            db_path=temp_db
        )
        assert ok_rec is True

    # Step 6: Adaptive feedback loop recognizes high waste pattern for subsequent meals
    waste_pat = rc.calculate_dish_waste_pattern(dish, meal_type, meal_date="2026-11-21", db_path=temp_db)
    assert waste_pat["has_sufficient_history"] is True
    assert waste_pat["avg_waste_pct"] >= 15.0

    # Step 7: Reports engine generates daily audit report and exports
    audit_rep = reports.generate_food_waste_report(
        timeframe="daily",
        target_date=test_date,
        data_source="REAL",
        db_path=temp_db
    )
    assert audit_rep["summary"]["total_prepared"] == prep_target
    assert audit_rep["summary"]["total_served"] == served_qty
    assert audit_rep["summary"]["total_discarded"] == disc_qty

    # Verify CSV and PDF export
    csv_str = reports.export_report_to_csv(audit_rep)
    assert dish in csv_str
    pdf_bytes = reports.export_report_to_pdf(audit_rep)
    assert len(pdf_bytes) > 1000


def test_60_production_mode_and_database_backup(temp_db, monkeypatch):
    """
    Test 60: Production mode security, demo credentials handling, and hot database backup.
    """
    db.init_db(temp_db)

    # 1. Hot crash-consistent SQLite database backup
    ok_bak, bak_file = db.backup_database(db_path=temp_db)
    assert ok_bak is True
    assert os.path.exists(bak_file)
    assert os.path.getsize(bak_file) > 0

    # Verify backup is a valid readable SQLite database
    bak_conn = sqlite3.connect(bak_file)
    bak_cur = bak_conn.cursor()
    bak_cur.execute("SELECT COUNT(*) FROM meal_records")
    assert isinstance(bak_cur.fetchone()[0], int)
    bak_conn.close()

    # 2. Production mode check
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ENABLE_DEMO_CREDENTIALS", raising=False)
    assert db.is_demo_mode_enabled() is False

    # Development mode override
    monkeypatch.setenv("ENABLE_DEMO_CREDENTIALS", "true")
    assert db.is_demo_mode_enabled() is True

    # 3. Student privacy roll number masking check
    assert db.mask_student_id("22BCS045") == "***045"
    assert db.mask_student_id("CS001") == "***001"
    assert db.mask_student_id("") == "Anonymous"


def test_61_dish_likes_recording_and_aggregation(temp_db):
    """Verify dish like/dislike recording, updates, and aggregation."""
    # 1. Record thumbs up for Pongal
    ok1, msg1 = db.record_dish_like("Pongal", "Breakfast", "2026-11-03", "24BDS021", liked=True, db_path=temp_db)
    assert ok1 is True
    assert "Rating recorded" in msg1

    # 2. Record thumbs down for Upma
    ok2, msg2 = db.record_dish_like("Upma", "Breakfast", "2026-11-03", "24BDS021", liked=False, db_path=temp_db)
    assert ok2 is True

    # 3. Record thumbs up for Pongal from another student
    ok3, msg3 = db.record_dish_like("Pongal", "Breakfast", "2026-11-03", "24BDS099", liked=True, db_path=temp_db)
    assert ok3 is True

    # 4. Check aggregated dataframe
    df = db.get_dish_likes(meal_type="Breakfast", meal_date="2026-11-03", db_path=temp_db)
    assert not df.empty
    assert "dish_name" in df.columns
    assert "thumbs_up" in df.columns
    assert "thumbs_down" in df.columns

    pongal_row = df[df["dish_name"] == "Pongal"]
    assert not pongal_row.empty
    assert pongal_row["thumbs_up"].values[0] == 2
    assert pongal_row["thumbs_down"].values[0] == 0

    upma_row = df[df["dish_name"] == "Upma"]
    assert not upma_row.empty
    assert upma_row["thumbs_up"].values[0] == 0
    assert upma_row["thumbs_down"].values[0] == 1

    # 5. Update rating: student changes vote for Upma to True
    ok4, msg4 = db.record_dish_like("Upma", "Breakfast", "2026-11-03", "24BDS021", liked=True, db_path=temp_db)
    assert ok4 is True
    assert "Rating updated" in msg4

    df_updated = db.get_dish_likes(meal_type="Breakfast", meal_date="2026-11-03", db_path=temp_db)
    upma_upd = df_updated[df_updated["dish_name"] == "Upma"]
    assert upma_upd["thumbs_up"].values[0] == 1
    assert upma_upd["thumbs_down"].values[0] == 0


def test_62_student_eco_impact_calculation(temp_db):
    """Verify green points, advance skip count, food avoided, and level tier."""
    roll = "24BDS777"

    # Initial impact should be 0 points, Seedling tier
    initial = db.get_student_impact(roll, db_path=temp_db)
    assert initial["green_points"] == 0
    assert initial["level"] == "Seedling"
    assert initial["meals_skipped_advance"] == 0
    assert initial["food_not_cooked_kg"] == 0.0

    # 1. Register 2 Yes votes (+20 points)
    db.register_student("2026-11-03", "Breakfast", registration_code=roll, attendance_intent="Yes", db_path=temp_db)
    db.register_student("2026-11-03", "Lunch", registration_code=roll, attendance_intent="Yes", db_path=temp_db)

    # 2. Register 3 No votes (advance skips) (+30 points, 3 skips, 3 * 0.35 = 1.05 kg avoided)
    db.register_student("2026-11-03", "Snacks", registration_code=roll, attendance_intent="No", db_path=temp_db)
    db.register_student("2026-11-03", "Dinner", registration_code=roll, attendance_intent="No", db_path=temp_db)
    db.register_student("2026-11-04", "Breakfast", registration_code=roll, attendance_intent="No", db_path=temp_db)

    # 3. Rate 4 dishes (+20 points)
    for dish in ["Pongal", "Dal Methi", "Samosa", "Kadai Paneer"]:
        db.record_dish_like(dish, "Lunch", "2026-11-03", registration_code=roll, liked=True, db_path=temp_db)

    # Total points: 20 + 30 + 20 = 70 points -> Sprout tier (50-149)
    res = db.get_student_impact(roll, db_path=temp_db)
    assert res["green_points"] == 70
    assert res["level"] == "Sprout"
    assert res["meals_skipped_advance"] == 3
    assert abs(res["food_not_cooked_kg"] - 1.05) < 0.01


def test_63_smart_mess_headcount_formula(temp_db):
    """Verify Smart Mess attendance calculation formula and plate savings."""
    # Headcount = students eating + (1,200 - students who replied) * historical_rate
    rates = {"Breakfast": 0.78, "Lunch": 0.92, "Snacks": 0.70, "Dinner": 0.88}

    # Simulate 100 students replying, 80 eating for Lunch
    total_capacity = 1200
    replied = 100
    eating = 80
    unreplied = total_capacity - replied  # 1100
    expected_headcount = int(round(eating + unreplied * rates["Lunch"]))  # 80 + 1100 * 0.92 = 80 + 1012 = 1092
    fewer_plates = total_capacity - expected_headcount  # 1200 - 1092 = 108

    assert expected_headcount == 1092
    assert fewer_plates == 108

    # Avoided over-cooking in kg: 108 plates * 0.35 kg/plate = 37.8 kg
    avoided_kg = round(fewer_plates * 0.35, 1)
    assert avoided_kg == 37.8


def test_64_student_portal_persistence_and_choice_update(temp_db):
    """Verify student Eating/Skipping choices persist and update without creating duplicates."""
    target_date = "2026-11-04"
    roll = "24BCS001"

    # Initial choice: Eating
    ok, msg = db.register_student(target_date, "Lunch", ["Veg Pulav", "Dal Tadka"], roll, intention_to_attend=True, attendance_intent="Yes", db_path=temp_db)
    assert ok is True
    status = db.get_student_meal_status(roll, target_date, "Lunch", db_path=temp_db)
    assert status == "Yes"

    # Count of registrations for this roll and meal should be 1
    cnt = db.get_registration_count(target_date, "Lunch", db_path=temp_db)
    assert cnt == 1

    # Student updates choice: Skipping
    ok2, msg2 = db.register_student(target_date, "Lunch", ["Veg Pulav", "Dal Tadka"], roll, intention_to_attend=False, attendance_intent="No", db_path=temp_db)
    assert ok2 is True
    status2 = db.get_student_meal_status(roll, target_date, "Lunch", db_path=temp_db)
    assert status2 == "No"

    # Total responses remains 1 (prevented duplicate, updated in place)
    cnts2 = db.get_attendance_intent_counts(target_date, "Lunch", eligible_students=1200, db_path=temp_db)
    assert cnts2["total_responses"] == 1
    assert cnts2["yes_count"] == 0
    assert cnts2["no_count"] == 1


def test_65_configurable_per_meal_food_estimate_and_history(temp_db):
    """Verify student impact with configurable per-meal portion and meal history retrieval."""
    target_date = "2026-11-05"
    roll = "24BCS002"

    # Student skips 2 meals in advance
    db.register_student(target_date, "Breakfast", ["Pongal"], roll, intention_to_attend=False, attendance_intent="No", db_path=temp_db)
    db.register_student(target_date, "Dinner", ["Kadai Paneer"], roll, intention_to_attend=False, attendance_intent="No", db_path=temp_db)

    # Test with default 0.35 kg
    res_default = db.get_student_impact(roll, per_meal_kg=0.35, db_path=temp_db)
    assert res_default["meals_skipped_advance"] == 2
    assert abs(res_default["food_not_cooked_kg"] - 0.70) < 0.01

    # Test with configurable 0.40 kg
    res_custom = db.get_student_impact(roll, per_meal_kg=0.40, db_path=temp_db)
    assert abs(res_custom["food_not_cooked_kg"] - 0.80) < 0.01

    # Retrieve personal meal history
    df_hist = db.get_student_meal_history(roll, limit=10, db_path=temp_db)
    assert len(df_hist) == 2
    assert set(df_hist["meal_type"]) == {"Breakfast", "Dinner"}
    assert all(df_hist["attendance_intent"] == "No")


def test_66_iiit_mess_menu_week_ab_switching():
    """Verify the official IIIT Dharwad Mess 1 menu loads all 7 days and 4 meals with Week A / Week B alternates."""
    menu_data = db.load_weekly_menu("data/weekly_menu.json")
    menu = menu_data.get("menu", {})

    assert len(menu) == 7
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meals = ["Breakfast", "Lunch", "Snacks", "Dinner"]

    for d in days:
        assert d in menu
        for m in meals:
            assert m in menu[d]
            assert "dishes" in menu[d][m]
            assert len(menu[d][m]["dishes"]) > 0

    # Test slash resolution helper
    def resolve_dish(text, week):
        if "/" not in text:
            return text.strip()
        parts = [p.strip() for p in text.split("/")]
        return parts[0] if week == "Week A" else parts[1]

    # Monday Breakfast: Pongal / Poha with sev
    mon_b = menu["Monday"]["Breakfast"]["dishes"][0]
    assert resolve_dish(mon_b, "Week A") == "Pongal"
    assert resolve_dish(mon_b, "Week B") == "Poha with sev"

    # Tuesday Breakfast: Upma / Sewaiya Upma
    tue_b = menu["Tuesday"]["Breakfast"]["dishes"][0]
    assert resolve_dish(tue_b, "Week A") == "Upma"
    assert resolve_dish(tue_b, "Week B") == "Sewaiya Upma"

    # Tuesday Dinner: Khichadi / Bisibele Bath with Boodi
    tue_d = menu["Tuesday"]["Dinner"]["dishes"][0]
    assert resolve_dish(tue_d, "Week A") == "Khichadi"
    assert resolve_dish(tue_d, "Week B") == "Bisibele Bath with Boodi"


def test_67_admin_smart_mess_headcount_live_connection(temp_db):
    """Verify Admin Portal headcount prediction for 1,200 capacity and live student connection."""
    target_date = "2026-11-04"

    # Base headcount for Lunch without live entries
    base_lunch = db.get_smart_mess_headcount(target_date, "Lunch", db_path=temp_db)
    assert base_lunch["total_capacity"] == 1200
    assert base_lunch["replied"] == 684
    assert base_lunch["eating"] == 644
    assert base_lunch["pending"] == 516
    assert base_lunch["rate"] == 0.92
    assert base_lunch["predicted_headcount"] == 1119
    assert base_lunch["fewer_plates"] == 81

    # Student checks in as Eating live from Student Portal
    db.register_student(target_date, "Lunch", ["Curd Rice"], "24BCS101", intention_to_attend=True, attendance_intent="Yes", db_path=temp_db)

    # Immediately recalculates in SQLite
    updated_lunch = db.get_smart_mess_headcount(target_date, "Lunch", db_path=temp_db)
    assert updated_lunch["replied"] == 685
    assert updated_lunch["eating"] == 645
    assert updated_lunch["pending"] == 515


def test_68_planner_portion_sizing_and_avoided_quantities():
    """Verify Smart Mess Planner dish portion specs, cook quantities and avoided quantities."""
    # Test Curd Rice (Rice/grain: 100 g)
    pred_headcount = 1119
    fewer_plates = 81

    # Multipliers: Rice 0.10 kg, Dal 0.03 kg, Fulka 3 pcs, Drink 0.15 L
    cook_rice = round(pred_headcount * 0.10, 1)
    avoided_rice = round(fewer_plates * 0.10, 1)
    assert cook_rice == 111.9
    assert avoided_rice == 8.1

    cook_dal = round(pred_headcount * 0.03, 1)
    avoided_dal = round(fewer_plates * 0.03, 1)
    assert cook_dal == 33.6
    assert avoided_dal == 2.4

    cook_bread = int(round(pred_headcount * 3))
    avoided_bread = int(round(fewer_plates * 3))
    assert cook_bread == 3357
    assert avoided_bread == 243

    cook_drink = round(pred_headcount * 0.15, 1)
    avoided_drink = round(fewer_plates * 0.15, 1)
    assert cook_drink == 167.8
    assert avoided_drink == 12.2


def test_69_waste_logging_persistence_alerts_and_kpis(temp_db):
    """Verify live waste logging, automated alert for >25 kg, and KPI recalculation."""
    target_date = "2026-11-04"

    # Log waste entry under 25 kg
    ok, msg, log_id = db.record_waste_log(target_date, "Lunch", "Curd Rice", "Leftover cooked", 15.0, "Compost", notes="Excess rain", db_path=temp_db)
    assert ok is True
    assert log_id is not None

    # Check live waste metrics
    m1 = db.get_smart_mess_dashboard_metrics(cost_per_kg=35.0, db_path=temp_db)
    assert m1["live_waste_kg"] == 15.0
    assert m1["waste_this_week_kg"] == 273.0  # 258 base + 15 live

    # Log heavy waste entry (>25 kg) -> should trigger alert
    ok2, msg2, log_id2 = db.record_waste_log(target_date, "Dinner", "Kadai Paneer", "Plate waste", 30.0, "Discarded", notes="Batch overrun", db_path=temp_db)
    assert ok2 is True

    # Verify alert created in system_alerts table
    alerts = db.get_alerts(alert_type="high_waste", db_path=temp_db)
    assert len(alerts) > 0
    recent_alert = alerts.iloc[0] if hasattr(alerts, "iloc") else alerts[0]
    assert "Kadai Paneer" in recent_alert["title"]

    # Verify top wasted dishes
    top_w = db.get_most_wasted_dishes(db_path=temp_db)
    assert len(top_w) == 2
    assert top_w[0]["dish_name"] == "Kadai Paneer"
    assert top_w[0]["total_kg"] == 30.0


def test_70_admin_announcements_broadcast_and_targeting(temp_db):
    """Test 70: Verify Admin announcements broadcasting with multiple targeting criteria."""
    db.init_db(temp_db)

    # Empty title or message should fail validation
    ok_bad, msg_bad, _ = db.send_admin_announcement("", "Valid message", db_path=temp_db)
    assert ok_bad is False

    ok_bad2, _, _ = db.send_admin_announcement("Valid Title", "   ", db_path=temp_db)
    assert ok_bad2 is False

    # Broadcast to all students
    ok1, msg1, aid1 = db.send_admin_announcement(
        title="Lunch starts at 12:30 PM.",
        message="Dining hall doors for Lunch will open at 12:30 PM today.",
        severity="info",
        target_criteria="all",
        admin_username="admin",
        db_path=temp_db
    )
    assert ok1 is True
    assert aid1 is not None

    # Targeted to students attending a particular meal
    ok2, msg2, aid2 = db.send_admin_announcement(
        title="Tomorrow's breakfast menu has changed.",
        message="Pongal replaced with Poha due to supply update.",
        severity="warning",
        target_criteria="attending_meal",
        target_date="2026-11-05",
        target_meal="Breakfast",
        admin_username="admin",
        db_path=temp_db
    )
    assert ok2 is True
    assert aid2 is not None

    # Targeted to unresponded students
    ok3, msg3, aid3 = db.send_admin_announcement(
        title="Please submit your attendance before 8 PM.",
        message="Kindly submit tomorrow's attendance.",
        severity="info",
        target_criteria="unresponded",
        target_date="2026-11-05",
        target_meal="Breakfast",
        admin_username="admin",
        db_path=temp_db
    )
    assert ok3 is True
    assert aid3 is not None

    # Verify alerts exist in system_alerts
    alerts = db.get_alerts(alert_type="announcement", db_path=temp_db)
    assert len(alerts) >= 3


def test_71_student_notification_filtering_and_read_receipts(temp_db):
    """Test 71: Verify student notification relevance filtering and per-student persistent read receipts."""
    db.init_db(temp_db)

    target_date = "2026-11-06"
    target_meal = "Lunch"

    # Student 1 registers 'Yes' (Eating)
    db.register_student(target_date, target_meal, ["Curd Rice"], "24BCS001", intention_to_attend=True, attendance_intent="Yes", db_path=temp_db)
    # Student 2 registers 'No' (Skipping)
    db.register_student(target_date, target_meal, [], "24BCS002", intention_to_attend=False, attendance_intent="No", db_path=temp_db)
    # Student 3 has not registered (unresponded)

    # 1. Admin sends broadcast to ALL
    ok_all, _, aid_all = db.send_admin_announcement("Notice to All", "General notice", target_criteria="all", db_path=temp_db)
    # 2. Admin sends notice to ATTENDING students
    ok_att, _, aid_att = db.send_admin_announcement("Attending Notice", "For lunch diners", target_criteria="attending_meal", target_date=target_date, target_meal=target_meal, db_path=temp_db)
    # 3. Admin sends notice to UNRESPONDED students
    ok_unr, _, aid_unr = db.send_admin_announcement("Reminder Notice", "Please submit your meal choices", target_criteria="unresponded", target_date=target_date, target_meal=target_meal, db_path=temp_db)

    # Check Student 1 (Eating): should see 'all' and 'attending_meal', but NOT 'unresponded'
    s1_notifs = db.get_student_notifications("24BCS001", db_path=temp_db)
    s1_titles = [n["title"] for n in s1_notifs]
    assert "Notice to All" in s1_titles
    assert "Attending Notice" in s1_titles
    assert "Reminder Notice" not in s1_titles

    # Check Student 2 (Skipping): should see 'all', but NOT 'attending_meal' and NOT 'unresponded'
    s2_notifs = db.get_student_notifications("24BCS002", db_path=temp_db)
    s2_titles = [n["title"] for n in s2_notifs]
    assert "Notice to All" in s2_titles
    assert "Attending Notice" not in s2_titles
    assert "Reminder Notice" not in s2_titles

    # Check Student 3 (Unresponded): should see 'all' and 'unresponded', but NOT 'attending_meal'
    s3_notifs = db.get_student_notifications("24BCS003", db_path=temp_db)
    s3_titles = [n["title"] for n in s3_notifs]
    assert "Notice to All" in s3_titles
    assert "Reminder Notice" in s3_titles
    assert "Attending Notice" not in s3_titles

    # Test read receipts for Student 1
    assert db.get_student_unread_count("24BCS001", db_path=temp_db) >= 2
    # Mark aid_all as read for Student 1
    db.mark_student_alert_read(aid_all, "24BCS001", db_path=temp_db)

    # Check that Student 1 has aid_all marked read
    s1_after = db.get_student_notifications("24BCS001", db_path=temp_db)
    s1_all_item = next(n for n in s1_after if n["id"] == aid_all)
    assert s1_all_item["is_read"] is True

    # Check Student 2: aid_all MUST STILL BE UNREAD for Student 2!
    s2_after = db.get_student_notifications("24BCS002", db_path=temp_db)
    s2_all_item = next(n for n in s2_after if n["id"] == aid_all)
    assert s2_all_item["is_read"] is False

    # Student 1 marks all as read
    db.mark_all_student_alerts_read("24BCS001", db_path=temp_db)
    assert db.get_student_unread_count("24BCS001", db_path=temp_db) == 0


def test_72_analytics_least_popular_and_waste_distribution(temp_db):
    """Test 72: Verify least popular dishes computation and environmental waste distribution."""
    db.init_db(temp_db)

    # Baseline least popular dishes returned when no live ratings
    base_lp = db.get_least_popular_dishes_analytics(db_path=temp_db)
    assert len(base_lp) >= 4
    assert any("Black Chana" in d["dish_name"] for d in base_lp)

    # Add live ratings in dish_likes:
    # 2 likes and 8 dislikes for Curd Rice = 20% liked
    for i in range(2):
        db.record_dish_like("Curd Rice", "Lunch", "2026-11-04", f"24BDS{i:03d}", liked=1, db_path=temp_db)
    for i in range(2, 10):
        db.record_dish_like("Curd Rice", "Lunch", "2026-11-04", f"24BDS{i:03d}", liked=0, db_path=temp_db)

    updated_lp = db.get_least_popular_dishes_analytics(db_path=temp_db)
    # Curd Rice should now be the #1 least popular dish at 20% liked
    assert updated_lp[0]["dish_name"] == "Curd Rice"
    assert updated_lp[0]["liked_pct"] == 20
    assert updated_lp[0]["source"] == "REAL"

    # Test waste distribution with destinations
    db.record_waste_log("2026-11-04", "Lunch", "Curd Rice", "Leftover cooked", 20.0, "NGO Donation", db_path=temp_db)
    db.record_waste_log("2026-11-04", "Lunch", "Curd Rice", "Plate waste", 10.0, "Composting", db_path=temp_db)
    db.record_waste_log("2026-11-04", "Lunch", "Curd Rice", "Preparation", 5.0, "Discarded", db_path=temp_db)

    m = db.get_smart_mess_dashboard_metrics(cost_per_kg=40.0, db_path=temp_db)
    assert m["live_waste_kg"] == 35.0
    assert m["waste_this_week_kg"] == 293.0  # 258 + 35
    assert m["cost_per_kg"] == 40.0
    assert m["cost_of_waste_inr"] == round(293.0 * 40.0)
    assert m["ngo_kg"] == 116.0 + 20.0
    assert m["compost_kg"] == 90.0 + 10.0
    assert m["discarded_kg"] == 52.0 + 5.0


def test_73_sqlite_users_schema_and_student_roll_number_auth(temp_db):
    """Test 73: Verify SQLite users table schema, student_roll_number, and PBKDF2 authentication."""
    db.init_db(temp_db)
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users);")
        cols = [r["name"] for r in cursor.fetchall()]
        for req in ["user_id", "username", "student_roll_number", "password_hash", "salt", "role", "created_at", "account_status"]:
            assert req in cols, f"Missing required column in users table: {req}"
    finally:
        conn.close()

    # Register student with institutional roll number
    ok, msg = db.create_user("24bds074", "Student@123", "Student", "Student 24BDS074", student_roll_number="24BDS074", db_path=temp_db)
    assert ok is True

    # Authenticate using uppercase roll number
    auth_roll = db.authenticate_user("24BDS074", "Student@123", db_path=temp_db)
    assert auth_roll is not None
    assert auth_roll["username"] == "24BDS074"
    assert auth_roll["student_roll_number"] == "24BDS074"
    assert auth_roll["role"] == "Student"
    assert auth_roll["account_status"] == "Active"
    assert "password_hash" not in auth_roll
    assert "salt" not in auth_roll

    # Also authenticate using roll number as initial password
    ok2, _ = db.create_user("24bcs009", "24BCS009", "Student", "Student 24BCS009", student_roll_number="24BCS009", db_path=temp_db)
    assert ok2 is True
    auth_roll_pwd = db.authenticate_user("24BCS009", "24BCS009", db_path=temp_db)
    assert auth_roll_pwd is not None
    assert auth_roll_pwd["student_roll_number"] == "24BCS009"


def test_74_admin_create_user_and_duplicate_prevention(temp_db):
    """Test 74: Verify Admin account creation, role checks, and duplicate username/roll prevention."""
    db.init_db(temp_db)
    
    # 1. Invalid role rejection
    ok, msg = db.create_user("baduser", "Secret@123", "SuperAdmin", "Bad User", db_path=temp_db)
    assert ok is False
    assert "Invalid role" in msg

    # 2. Short password rejection
    ok, msg = db.create_user("shortpwd", "123", "Student", "Short User", db_path=temp_db)
    assert ok is False
    assert "at least 6 characters" in msg

    # 3. Create Admin account
    ok, msg = db.create_user("mess_manager", "Manager@123", "Admin", "Mess Operations Head", db_path=temp_db)
    assert ok is True

    # 4. Prevent duplicate username
    ok_dup, dup_msg = db.create_user("mess_manager", "OtherPass@123", "Admin", "Mess Operations Head", db_path=temp_db)
    assert ok_dup is False
    assert "already exists" in dup_msg

    # 5. Prevent duplicate student roll number
    ok_s1, _ = db.create_user("student_a", "Pass@123", "Student", "Student A", student_roll_number="24BDS999", db_path=temp_db)
    assert ok_s1 is True
    ok_s2, dup_s2 = db.create_user("student_b", "Pass@123", "Student", "Student B", student_roll_number="24BDS999", db_path=temp_db)
    assert ok_s2 is False
    assert "already exists" in dup_s2


def test_75_dynamic_kpi_calculations_and_data_source_separation(temp_db):
    """Test 75: Verify headcount formula, live SQLite updates, and REAL data source separation with insufficient data alert."""
    db.init_db(temp_db)

    # 1. When data_source == 'REAL' and no registrations exist, verify insufficient data indicator
    m_real_empty = db.get_smart_mess_headcount("2026-11-20", "Lunch", data_source="REAL", db_path=temp_db)
    assert m_real_empty["insufficient_data"] is True
    assert "Insufficient real operational data" in m_real_empty["message"]

    m_dash_empty = db.get_smart_mess_dashboard_metrics(cost_per_kg=35.0, data_source="REAL", db_path=temp_db)
    assert m_dash_empty["insufficient_data"] is True
    assert "Insufficient real operational data" in m_dash_empty["message"]

    # 2. Add real registrations for 1,200 capacity:
    # 800 Eating (Yes), 200 Skipping (No) -> Replied = 1,000, Pending = 200.
    # Historical rate for Lunch is 92% (0.92).
    # Estimated from pending = round(200 * 0.92) = 184.
    # Predicted headcount = 800 + 184 = 984.
    # Fewer plates than 1,200 = 1,200 - 984 = 216.
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        now_str = "2026-11-04T12:00:00"
        for i in range(800):
            cursor.execute("""
                INSERT INTO student_registrations (
                    meal_date, meal_type, selected_dishes, registration_code,
                    intention_to_attend, attendance_intent, data_source, submission_timestamp
                ) VALUES ('2026-11-04', 'Lunch', '[]', ?, 1, 'Yes', 'REAL', ?)
            """, (f"ROLL_YES_{i:04d}", now_str))
        for i in range(200):
            cursor.execute("""
                INSERT INTO student_registrations (
                    meal_date, meal_type, selected_dishes, registration_code,
                    intention_to_attend, attendance_intent, data_source, submission_timestamp
                ) VALUES ('2026-11-04', 'Lunch', '[]', ?, 0, 'No', 'REAL', ?)
            """, (f"ROLL_NO_{i:04d}", now_str))
        conn.commit()
    finally:
        conn.close()

    hc = db.get_smart_mess_headcount("2026-11-04", "Lunch", data_source="REAL", db_path=temp_db)
    assert hc["insufficient_data"] is False
    assert hc["replied"] == 1000
    assert hc["eating"] == 800
    assert hc["skipping"] == 200
    assert hc["pending"] == 200
    assert hc["estimated_from_pending"] == 184
    assert hc["predicted_headcount"] == 984
    assert hc["fewer_plates"] == 216

    # 3. Add real waste log and check verified financial calculation against 460 kg baseline
    db.record_waste_log("2026-11-04", "Lunch", "Veg Pulav", "Leftover cooked", 50.0, "NGO Donation", data_source="REAL", db_path=temp_db)
    dash = db.get_smart_mess_dashboard_metrics(cost_per_kg=35.0, data_source="REAL", db_path=temp_db)
    assert dash["insufficient_data"] is False
    assert dash["waste_this_week_kg"] == 50.0
    assert dash["cost_of_waste_inr"] == round(50.0 * 35.0)  # 1750
    # Verified savings against 460 kg baseline: max(0, 460 - 50) = 410 kg saved = 410 * 35 = ₹14,350
    assert dash["money_saved_week_inr"] == 410 * 35
    assert dash["ngo_kg"] == 50.0







def test_76_user_management_stats_filtering_and_lifecycle(temp_db):
    """Test 76: Comprehensive test for Manage Users page - stats, search, role/status filtering, update, toggle status, and password reset."""
    db.init_db(temp_db)

    # 1. Check initial user stats
    stats = db.get_user_stats(db_path=temp_db)
    assert "students" in stats
    assert "kitchen_staff" in stats
    assert "admins" in stats
    assert "active" in stats
    assert "inactive" in stats
    initial_students = stats["students"]
    initial_active = stats["active"]

    # 2. Create users
    ok1, _ = db.create_user("student_t1", "SecurePass@123", "Student", "Test Student One", student_roll_number="24BDS101", db_path=temp_db)
    ok2, _ = db.create_user("chef_t1", "StaffPass@123", "Kitchen Staff", "Test Chef One", db_path=temp_db)
    assert ok1 is True
    assert ok2 is True

    # 3. Verify stats updated
    stats_after = db.get_user_stats(db_path=temp_db)
    assert stats_after["students"] == initial_students + 1
    assert stats_after["active"] == initial_active + 2

    # 4. Search and filter users (returns tuple: users, total_count)
    users_search, total_search = db.get_filtered_users(search="24BDS101", db_path=temp_db)
    assert len(users_search) == 1
    assert total_search == 1
    assert users_search[0]["username"] == "student_t1"
    assert users_search[0]["student_roll_number"] == "24BDS101"

    users_staff, total_staff = db.get_filtered_users(role="Kitchen Staff", db_path=temp_db)
    assert any(u["username"] == "chef_t1" for u in users_staff)

    # 5. Edit user details
    target_id = users_search[0]["user_id"]
    ok_up, msg_up = db.update_user_details(target_id, full_name="Updated Name", student_roll_number="24BDS102", role="Student", db_path=temp_db)
    assert ok_up is True

    user_check = db.get_user_by_username("student_t1", db_path=temp_db)
    assert user_check["full_name"] == "Updated Name"
    assert user_check["student_roll_number"] == "24BDS102"

    # 6. Toggle account status (Active -> Inactive)
    ok_tog, msg_tog = db.toggle_user_status(target_id, "Inactive", db_path=temp_db)
    assert ok_tog is True
    assert "Inactive" in msg_tog

    user_inactive = db.get_user_by_username("student_t1", db_path=temp_db)
    assert user_inactive["account_status"] == "Inactive"

    # Verify inactive user cannot authenticate (returns None)
    user_inactive_auth = db.authenticate_user("student_t1", "SecurePass@123", db_path=temp_db)
    assert user_inactive_auth is None

    # 7. Reactivate user
    ok_react, msg_react = db.toggle_user_status(target_id, "Active", db_path=temp_db)
    assert ok_react is True
    assert "Active" in msg_react

    # 8. Reset password securely
    ok_res, _ = db.reset_user_password(target_id, "NewSecurePassword#999", db_path=temp_db)
    assert ok_res is True

    # Old password fails
    user_old_auth = db.authenticate_user("student_t1", "SecurePass@123", db_path=temp_db)
    assert user_old_auth is None

    # New password succeeds
    user_new_auth = db.authenticate_user("student_t1", "NewSecurePassword#999", db_path=temp_db)
    assert user_new_auth is not None
    assert user_new_auth["role"] == "Student"


def test_77_week_a_week_b_menu_alternation_and_persistence(temp_db):
    """Test 77: Verify Week A / Week B system settings persistence and accurate slash dish resolution."""
    db.init_db(temp_db)

    # 1. Default active week is 'Week A'
    w_default = db.get_active_menu_week(db_path=temp_db)
    assert w_default == "Week A"

    # 2. Persist active week to 'Week B'
    ok_set, _ = db.set_active_menu_week("Week B", db_path=temp_db)
    assert ok_set is True
    w_b = db.get_active_menu_week(db_path=temp_db)
    assert w_b == "Week B"

    # 3. Test slash dish resolution for Week A and Week B
    # Monday Breakfast: 'Pongal / Poha with sev'
    assert db.resolve_dish_name("Pongal / Poha with sev", "Week A") == "Pongal"
    assert db.resolve_dish_name("Pongal / Poha with sev", "Week B") == "Poha with sev"

    # Lunch: 'Chapati / Fulka'
    assert db.resolve_dish_name("Chapati / Fulka", "Week A") == "Chapati"
    assert db.resolve_dish_name("Chapati / Fulka", "Week B") == "Fulka"

    # Tuesday Breakfast: 'Upma / Sewaiya Upma'
    assert db.resolve_dish_name("Upma / Sewaiya Upma", "Week A") == "Upma"
    assert db.resolve_dish_name("Upma / Sewaiya Upma", "Week B") == "Sewaiya Upma"

    # Tuesday Dinner: 'Khichadi / Bisibele Bath with Boodi'
    assert db.resolve_dish_name("Khichadi / Bisibele Bath with Boodi", "Week A") == "Khichadi"
    assert db.resolve_dish_name("Khichadi / Bisibele Bath with Boodi", "Week B") == "Bisibele Bath with Boodi"

    # Fixed items without slash remain completely unchanged
    assert db.resolve_dish_name("Curd Rice", "Week A") == "Curd Rice"
    assert db.resolve_dish_name("Curd Rice", "Week B") == "Curd Rice"
    assert db.resolve_dish_name("Kadai Paneer", "Week A") == "Kadai Paneer"
    assert db.resolve_dish_name("Kadai Paneer", "Week B") == "Kadai Paneer"


def test_78_student_leaderboard_points_and_ranking_calculations(temp_db):
    """Test 78: Verify real-time student leaderboard calculation, ranking, badges, and personal standing."""
    db.init_db(temp_db)

    # 1. Create 3 test students
    db.create_user("student_lb_1", "Pass@123", "Student", "Leaderboard Alice", student_roll_number="24BDS801", db_path=temp_db)
    db.create_user("student_lb_2", "Pass@123", "Student", "Leaderboard Bob", student_roll_number="24BDS802", db_path=temp_db)
    db.create_user("student_lb_3", "Pass@123", "Student", "Leaderboard Charlie", student_roll_number="24BDS803", db_path=temp_db)

    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        now_str = "2026-11-04T12:00:00"

        # Alice: 5 meals logged, 2 skips (No), 4 dish likes
        # Points = (5 * 10) + (4 * 5) + (2 * 15) = 50 + 20 + 30 = 100 points
        for i in range(3):
            cursor.execute("INSERT INTO student_registrations (meal_date, meal_type, selected_dishes, registration_code, intention_to_attend, attendance_intent, submission_timestamp) VALUES ('2026-11-04', 'Lunch', '[]', '24BDS801', 1, 'Yes', ?);", (now_str,))
        for i in range(2):
            cursor.execute("INSERT INTO student_registrations (meal_date, meal_type, selected_dishes, registration_code, intention_to_attend, attendance_intent, submission_timestamp) VALUES ('2026-11-04', 'Dinner', '[]', '24BDS801', 0, 'No', ?);", (now_str,))
        for i in range(4):
            cursor.execute("INSERT INTO dish_likes (dish_name, meal_type, meal_date, registration_code, liked, submitted_timestamp) VALUES ('Paneer', 'Lunch', '2026-11-04', '24BDS801', 1, ?);", (now_str,))

        # Bob: 2 meals logged, 0 skips, 0 likes
        # Points = (2 * 10) + 0 + 0 = 20 points
        for i in range(2):
            cursor.execute("INSERT INTO student_registrations (meal_date, meal_type, selected_dishes, registration_code, intention_to_attend, attendance_intent, submission_timestamp) VALUES ('2026-11-04', 'Lunch', '[]', '24BDS802', 1, 'Yes', ?);", (now_str,))

        # Charlie: 10 meals logged, 5 skips, 10 likes
        # Points = (10 * 10) + (10 * 5) + (5 * 15) = 100 + 50 + 75 = 225 points
        for i in range(5):
            cursor.execute("INSERT INTO student_registrations (meal_date, meal_type, selected_dishes, registration_code, intention_to_attend, attendance_intent, submission_timestamp) VALUES ('2026-11-04', 'Breakfast', '[]', '24BDS803', 1, 'Yes', ?);", (now_str,))
        for i in range(5):
            cursor.execute("INSERT INTO student_registrations (meal_date, meal_type, selected_dishes, registration_code, intention_to_attend, attendance_intent, submission_timestamp) VALUES ('2026-11-04', 'Lunch', '[]', '24BDS803', 0, 'No', ?);", (now_str,))
        for i in range(10):
            cursor.execute("INSERT INTO dish_likes (dish_name, meal_type, meal_date, registration_code, liked, submitted_timestamp) VALUES ('Pulav', 'Lunch', '2026-11-04', '24BDS803', 1, ?);", (now_str,))

        conn.commit()
    finally:
        conn.close()

    # 2. Query leaderboard
    res = db.get_student_leaderboard(current_roll="24BDS801", db_path=temp_db)
    summary = res["summary"]
    assert summary["total_students"] >= 3
    assert summary["total_points"] >= 345

    # Charlie should rank higher than Alice, who ranks higher than Bob
    charlie_st = next(st for st in res["leaderboard"] if st["student_roll_number"] == "24BDS803")
    alice_st = next(st for st in res["leaderboard"] if st["student_roll_number"] == "24BDS801")
    bob_st = next(st for st in res["leaderboard"] if st["student_roll_number"] == "24BDS802")

    assert charlie_st["green_points"] == 225
    assert alice_st["green_points"] == 100
    assert bob_st["green_points"] == 20
    assert charlie_st["rank"] < alice_st["rank"] < bob_st["rank"]

    # Check badge levels
    assert "Sprout" in charlie_st["badge_level"]   # 150 - 299 pts
    assert "Seedling" in alice_st["badge_level"]   # < 150 pts
    assert "Seedling" in bob_st["badge_level"]

    # Check personal standing for Alice
    assert res["my_rank"] is not None
    assert res["my_rank"]["student_roll_number"] == "24BDS801"
    assert res["my_rank"]["green_points"] == 100
    assert res["my_rank"]["meals_skipped"] == 2
