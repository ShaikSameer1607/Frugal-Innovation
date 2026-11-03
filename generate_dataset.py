"""
generate_dataset.py
===================
Synthetic Dataset Generator for MessMind AI:
An AI-Based Food Waste Prevention and Demand Recommendation System
Tailored specifically for the Official Weekly Menu of IIIT Dharwad Mess 1.

Generates ~3,500–4,000 realistic meal-and-dish records over 300 days strictly
matching the actual day-by-day menu schedule (Monday to Sunday) for Breakfast, Lunch, and Dinner.
Enforces non-negative counts, leftover equations, zero leakage, and transparent synthetic labeling.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set fixed random seed for 100% reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Detailed configuration for all official IIIT Dharwad Mess 1 dishes:
OFFICIAL_DISH_PROFILES = {
    # Monday
    "Pongal": {"meal": "Breakfast", "take_rate": 0.74, "cost": 14.0, "spoilage": 0.60},
    "Poha": {"meal": "Breakfast", "take_rate": 0.68, "cost": 12.0, "spoilage": 0.55},
    "Curd Rice": {"meal": "Lunch", "take_rate": 0.82, "cost": 11.0, "spoilage": 0.70},
    "Dal Methi": {"meal": "Lunch", "take_rate": 0.85, "cost": 14.0, "spoilage": 0.50},
    "Brinjal Bhartha": {"meal": "Lunch", "take_rate": 0.72, "cost": 16.0, "spoilage": 0.65},
    "Chapati": {"meal": "Lunch", "take_rate": 0.80, "cost": 12.0, "spoilage": 0.40},
    "Plain Rice": {"meal": "Lunch", "take_rate": 0.90, "cost": 9.0, "spoilage": 0.35},
    "Kadai Paneer": {"meal": "Dinner", "take_rate": 0.95, "cost": 34.0, "spoilage": 0.85},
    "Rasam": {"meal": "Dinner", "take_rate": 0.76, "cost": 10.0, "spoilage": 0.40},
    "Dal Pancha Ratan": {"meal": "Dinner", "take_rate": 0.82, "cost": 14.0, "spoilage": 0.50},
    "Sheera": {"meal": "Dinner", "take_rate": 0.88, "cost": 15.0, "spoilage": 0.60},

    # Tuesday
    "Upma": {"meal": "Breakfast", "take_rate": 0.65, "cost": 10.0, "spoilage": 0.50},
    "Tuesday Sheera": {"dish": "Sheera", "meal": "Breakfast", "take_rate": 0.86, "cost": 15.0, "spoilage": 0.60},
    "Veg Pulav": {"meal": "Lunch", "take_rate": 0.92, "cost": 14.0, "spoilage": 0.50},
    "Dal Tadka": {"meal": "Lunch", "take_rate": 0.86, "cost": 14.0, "spoilage": 0.50},
    "White Peas Masala": {"meal": "Lunch", "take_rate": 0.78, "cost": 16.0, "spoilage": 0.60},
    "Puri": {"meal": "Lunch", "take_rate": 0.88, "cost": 14.0, "spoilage": 0.45},
    "Curd": {"meal": "Lunch", "take_rate": 0.65, "cost": 10.0, "spoilage": 0.75},
    "Khichadi": {"meal": "Dinner", "take_rate": 0.78, "cost": 12.0, "spoilage": 0.50},
    "Bhendi Sabji": {"meal": "Dinner", "take_rate": 0.74, "cost": 18.0, "spoilage": 0.65},
    "Veg Kadai": {"meal": "Dinner", "take_rate": 0.80, "cost": 18.0, "spoilage": 0.65},
    "Channa Dal": {"meal": "Dinner", "take_rate": 0.82, "cost": 14.0, "spoilage": 0.50},

    # Wednesday
    "Idli": {"meal": "Breakfast", "take_rate": 0.84, "cost": 12.0, "spoilage": 0.65},
    "Vada": {"meal": "Breakfast", "take_rate": 0.90, "cost": 14.0, "spoilage": 0.70},
    "Onion Rice": {"meal": "Lunch", "take_rate": 0.88, "cost": 12.0, "spoilage": 0.50},
    "Dal Makhani": {"meal": "Lunch", "take_rate": 0.90, "cost": 16.0, "spoilage": 0.60},
    "Jeera Aloo": {"meal": "Lunch", "take_rate": 0.78, "cost": 14.0, "spoilage": 0.50},
    "Chawli Masala": {"meal": "Lunch", "take_rate": 0.72, "cost": 15.0, "spoilage": 0.60},
    "Soyabean Biryani": {"meal": "Dinner", "take_rate": 0.86, "cost": 14.0, "spoilage": 0.55},
    "Paneer Kolhapuri": {"meal": "Dinner", "take_rate": 0.95, "cost": 34.0, "spoilage": 0.85},
    "Masoor Dal": {"meal": "Dinner", "take_rate": 0.80, "cost": 14.0, "spoilage": 0.50},

    # Thursday
    "Aloo Paratha": {"meal": "Breakfast", "take_rate": 0.92, "cost": 18.0, "spoilage": 0.60},
    "Puliogare": {"meal": "Lunch", "take_rate": 0.85, "cost": 12.0, "spoilage": 0.50},
    "Dal Fry": {"meal": "Lunch", "take_rate": 0.86, "cost": 14.0, "spoilage": 0.50},
    "Brinjal Tomato": {"meal": "Lunch", "take_rate": 0.70, "cost": 15.0, "spoilage": 0.65},
    "Kadhi Pakoda": {"meal": "Lunch", "take_rate": 0.76, "cost": 16.0, "spoilage": 0.70},
    "Ivy Gourd Fry": {"meal": "Dinner", "take_rate": 0.68, "cost": 16.0, "spoilage": 0.60},
    "Cabbage Baji": {"meal": "Dinner", "take_rate": 0.72, "cost": 14.0, "spoilage": 0.60},
    "Thursday Dal": {"dish": "Dal", "meal": "Dinner", "take_rate": 0.82, "cost": 14.0, "spoilage": 0.50},

    # Friday
    "Masala Dosa": {"meal": "Breakfast", "take_rate": 0.96, "cost": 20.0, "spoilage": 0.80},
    "Veg & Egg Biryani": {"meal": "Lunch", "take_rate": 0.94, "cost": 22.0, "spoilage": 0.60},
    "Mix Veg Raita": {"meal": "Lunch", "take_rate": 0.72, "cost": 12.0, "spoilage": 0.75},
    "Black Chana Masala": {"meal": "Lunch", "take_rate": 0.78, "cost": 15.0, "spoilage": 0.60},
    "Tomato Rice": {"meal": "Dinner", "take_rate": 0.85, "cost": 12.0, "spoilage": 0.50},
    "Aloo 65": {"meal": "Dinner", "take_rate": 0.90, "cost": 16.0, "spoilage": 0.65},
    "Soyabean Sabzi Gravy": {"meal": "Dinner", "take_rate": 0.75, "cost": 15.0, "spoilage": 0.60},

    # Saturday
    "Uthappam": {"meal": "Breakfast", "take_rate": 0.84, "cost": 15.0, "spoilage": 0.65},
    "Dal Palak": {"meal": "Lunch", "take_rate": 0.82, "cost": 14.0, "spoilage": 0.55},
    "Chole Bhature": {"meal": "Lunch", "take_rate": 0.94, "cost": 24.0, "spoilage": 0.65},
    "Jeera Rice": {"meal": "Dinner", "take_rate": 0.86, "cost": 12.0, "spoilage": 0.45},
    "Shahi Paneer": {"meal": "Dinner", "take_rate": 0.96, "cost": 34.0, "spoilage": 0.85},
    "Dal Kolhapuri": {"meal": "Dinner", "take_rate": 0.82, "cost": 14.0, "spoilage": 0.55},

    # Sunday
    "Vangi Bath": {"meal": "Breakfast", "take_rate": 0.76, "cost": 14.0, "spoilage": 0.55},
    "Aloo Mutter Sabzi": {"meal": "Breakfast", "take_rate": 0.74, "cost": 15.0, "spoilage": 0.60},
    "Paneer Butter Masala": {"meal": "Lunch", "take_rate": 0.96, "cost": 34.0, "spoilage": 0.85},
    "Moong Dal": {"meal": "Lunch", "take_rate": 0.80, "cost": 14.0, "spoilage": 0.50},
    "Egg Curry": {"meal": "Lunch", "take_rate": 0.85, "cost": 22.0, "spoilage": 0.70},
    "Sorakay Chana": {"meal": "Dinner", "take_rate": 0.70, "cost": 15.0, "spoilage": 0.60},
    "Veg Kolhapuri": {"meal": "Dinner", "take_rate": 0.82, "cost": 18.0, "spoilage": 0.65},

    # Snacks (Monday - Sunday)
    "Samosa": {"meal": "Snacks", "take_rate": 0.88, "cost": 12.0, "spoilage": 0.50},
    "Pani Puri": {"meal": "Snacks", "take_rate": 0.92, "cost": 20.0, "spoilage": 0.60},
    "Punugulu": {"meal": "Snacks", "take_rate": 0.82, "cost": 16.0, "spoilage": 0.60},
    "Veg Noodles": {"meal": "Snacks", "take_rate": 0.86, "cost": 18.0, "spoilage": 0.55},
    "Pasta": {"meal": "Snacks", "take_rate": 0.84, "cost": 18.0, "spoilage": 0.55},
    "Masala Bhel": {"meal": "Snacks", "take_rate": 0.78, "cost": 14.0, "spoilage": 0.50},
    "Girmit": {"meal": "Snacks", "take_rate": 0.80, "cost": 14.0, "spoilage": 0.50},
    "Ginger Tea": {"meal": "Snacks", "take_rate": 0.75, "cost": 8.0, "spoilage": 0.30},
}

# Day-to-Dish schedule matching IIIT Dharwad Mess 1 Menu exactly:
WEEKLY_SCHEDULE = {
    "Monday": {
        "Breakfast": ["Pongal", "Poha"],
        "Lunch": ["Curd Rice", "Dal Methi", "Brinjal Bhartha", "Chapati", "Plain Rice"],
        "Snacks": ["Samosa", "Ginger Tea"],
        "Dinner": ["Kadai Paneer", "Rasam", "Dal Pancha Ratan", "Sheera", "Plain Rice", "Chapati"]
    },
    "Tuesday": {
        "Breakfast": ["Upma", "Tuesday Sheera"],
        "Lunch": ["Veg Pulav", "Dal Tadka", "White Peas Masala", "Puri", "Curd", "Plain Rice"],
        "Snacks": ["Pani Puri", "Ginger Tea"],
        "Dinner": ["Khichadi", "Bhendi Sabji", "Veg Kadai", "Channa Dal", "Plain Rice", "Chapati"]
    },
    "Wednesday": {
        "Breakfast": ["Idli", "Vada"],
        "Lunch": ["Onion Rice", "Dal Makhani", "Jeera Aloo", "Chawli Masala", "Chapati", "Curd", "Plain Rice"],
        "Snacks": ["Punugulu", "Ginger Tea"],
        "Dinner": ["Soyabean Biryani", "Paneer Kolhapuri", "Rasam", "Masoor Dal", "Plain Rice", "Chapati"]
    },
    "Thursday": {
        "Breakfast": ["Aloo Paratha"],
        "Lunch": ["Puliogare", "Dal Fry", "Brinjal Tomato", "Kadhi Pakoda", "Chapati", "Plain Rice"],
        "Snacks": ["Veg Noodles", "Ginger Tea"],
        "Dinner": ["Ivy Gourd Fry", "Cabbage Baji", "Thursday Dal", "Plain Rice", "Chapati"]
    },
    "Friday": {
        "Breakfast": ["Masala Dosa"],
        "Lunch": ["Veg & Egg Biryani", "Mix Veg Raita", "Black Chana Masala", "Chapati", "Curd", "Plain Rice"],
        "Snacks": ["Pasta", "Ginger Tea"],
        "Dinner": ["Tomato Rice", "Aloo 65", "Soyabean Sabzi Gravy", "Dal Tadka", "Plain Rice", "Chapati"]
    },
    "Saturday": {
        "Breakfast": ["Uthappam"],
        "Lunch": ["Curd Rice", "Dal Palak", "Chole Bhature", "Plain Rice"],
        "Snacks": ["Masala Bhel", "Ginger Tea"],
        "Dinner": ["Jeera Rice", "Shahi Paneer", "Dal Kolhapuri", "Curd", "Plain Rice", "Chapati"]
    },
    "Sunday": {
        "Breakfast": ["Vangi Bath", "Aloo Mutter Sabzi"],
        "Lunch": ["Paneer Butter Masala", "Moong Dal", "Egg Curry", "Chapati", "Plain Rice"],
        "Snacks": ["Girmit", "Ginger Tea"],
        "Dinner": ["Sorakay Chana", "Veg Kolhapuri", "Moong Dal", "Plain Rice", "Chapati"]
    }
}


def generate_synthetic_data(num_days: int = 300, output_path: str = "data/mess_data.csv") -> pd.DataFrame:
    """
    Generate synthetic dataset matching the official IIIT Dharwad Mess 1 weekly menu.
    """
    records = []
    start_date = datetime.now().date() - timedelta(days=num_days)
    history_tracker = {}

    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        day_of_week = current_date.strftime("%A")
        is_weekend = day_of_week in ["Saturday", "Sunday"]
        is_friday = day_of_week == "Friday"

        day_menu = WEEKLY_SCHEDULE[day_of_week]

        for meal_type in ["Breakfast", "Lunch", "Snacks", "Dinner"]:
            # Baseline expected attendance: ~500 hostel student capacity
            if meal_type == "Breakfast":
                expected_base = 310 if is_weekend else 425
            elif meal_type == "Lunch":
                expected_base = 360 if is_weekend else 480
            elif meal_type == "Snacks":
                expected_base = 290 if is_weekend else 360
            else:  # Dinner
                expected_base = 370 if is_weekend else (440 if is_friday else 485)

            expected_attendance = int(np.clip(np.random.normal(expected_base, 14), 220, 540))

            # Advance registration intent (QR code submissions):
            reg_ratio = np.random.uniform(0.68, 0.86) if is_weekend else np.random.uniform(0.82, 0.95)
            registered_students = int(np.clip(expected_attendance * reg_ratio + np.random.normal(0, 6), 80, expected_attendance))

            # Actual attendance: combination of registered students (~90% show-up) + walk-ins
            show_up_rate = np.random.uniform(0.88, 0.96)
            walk_in_rate = np.random.uniform(0.04, 0.12)
            actual_attendance = int(round(registered_students * show_up_rate + (expected_attendance - registered_students) * walk_in_rate))
            actual_attendance = max(60, min(actual_attendance, 550))

            dishes_for_session = day_menu[meal_type]

            for dish_key in dishes_for_session:
                cfg = OFFICIAL_DISH_PROFILES.get(dish_key, {
                    "take_rate": 0.80, "cost": 15.0, "spoilage": 0.60
                })
                # Display name (strips prefixes like 'Tuesday Sheera' or 'Thursday Dal')
                dish_display_name = cfg.get("dish", dish_key)
                hist_key = (dish_display_name, meal_type)

                # Chronological historical demand strictly prior to this day
                past_servings = history_tracker.get(hist_key, [])
                if len(past_servings) >= 3:
                    historical_demand = float(np.mean(past_servings[-12:]))
                elif len(past_servings) > 0:
                    historical_demand = float(np.mean(past_servings))
                else:
                    historical_demand = float(round(expected_base * cfg["take_rate"]))

                historical_demand = round(historical_demand, 1)

                # Take-rate noise for this specific meal
                take_rate = np.clip(np.random.normal(cfg["take_rate"], 0.04), 0.35, 1.0)
                actual_servings_needed = int(round(actual_attendance * take_rate))
                actual_servings_needed = max(10, actual_servings_needed)

                # Record into history for FUTURE dates only (zero leakage)
                if hist_key not in history_tracker:
                    history_tracker[hist_key] = []
                history_tracker[hist_key].append(actual_servings_needed)

                # Traditional mess prep heuristic (pre-MessMind flawed behavior):
                # Staff cook using a crude flat buffer (+10% to +20%) on registered students
                overprep_factor = np.random.choice([1.08, 1.15, 1.22, 1.02, 0.96], p=[0.42, 0.32, 0.14, 0.08, 0.04])
                heuristic_basis = max(registered_students * cfg["take_rate"], historical_demand)
                quantity_prepared = int(round(heuristic_basis * overprep_factor + np.random.normal(0, 4)))
                quantity_prepared = max(actual_servings_needed - 12, quantity_prepared)
                quantity_prepared = max(10, quantity_prepared)

                # Quantity served cannot exceed prepared, and cannot exceed need
                quantity_served = min(actual_servings_needed, quantity_prepared)

                # Leftover = prepared - served
                leftover_quantity = quantity_prepared - quantity_served

                # Discarded vs usable:
                if leftover_quantity > 0:
                    spoilage_prob = cfg["spoilage"]
                    discard_ratio = np.clip(np.random.normal(spoilage_prob, 0.12), 0.10, 1.0)
                    discarded_quantity = int(round(leftover_quantity * discard_ratio))
                    discarded_quantity = min(leftover_quantity, max(0, discarded_quantity))
                else:
                    discarded_quantity = 0

                usable_leftover = leftover_quantity - discarded_quantity
                cost_per_serving = float(cfg["cost"])
                discarded_food_cost_inr = round(float(discarded_quantity * cost_per_serving), 2)

                # Validate strict constraints
                assert quantity_prepared >= 0
                assert quantity_served >= 0
                assert quantity_served <= quantity_prepared
                assert leftover_quantity == quantity_prepared - quantity_served
                assert 0 <= discarded_quantity <= leftover_quantity
                assert usable_leftover == leftover_quantity - discarded_quantity

                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "day_of_week": day_of_week,
                    "meal_type": meal_type,
                    "dish_name": dish_display_name,
                    "expected_attendance": expected_attendance,
                    "registered_students": registered_students,
                    "historical_demand": historical_demand,
                    "actual_attendance": actual_attendance,
                    "actual_servings_needed": actual_servings_needed,
                    "quantity_prepared": quantity_prepared,
                    "quantity_served": quantity_served,
                    "leftover_quantity": leftover_quantity,
                    "discarded_quantity": discarded_quantity,
                    "usable_leftover": usable_leftover,
                    "cost_per_serving_inr": cost_per_serving,
                    "discarded_food_cost_inr": discarded_food_cost_inr,
                    "is_synthetic": True,
                    "data_source": "IIIT Dharwad Mess 1 Menu Simulation (Synthetic Baseline)"
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def print_dataset_summary(df: pd.DataFrame, output_path: str):
    """Print comprehensive summary of generated dataset."""
    print("=" * 80)
    print("MESSMIND AI — IIIT DHARWAD MESS 1 DATASET GENERATION SUMMARY")
    print("=" * 80)
    print(f"Destination: {output_path}")
    print(f"Total Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Unique Dates: {df['date'].nunique()} ({df['date'].min()} to {df['date'].max()})")
    print(f"Unique Scheduled Dishes: {df['dish_name'].nunique()}")
    print(f"Dishes List: {sorted(df['dish_name'].unique().tolist())}")
    print("-" * 80)
    print("Columns & Missing Values:")
    for col, count in df.isnull().sum().items():
        print(f"  - {col:<26}: {count} missing")
    print("-" * 80)
    print("Sample Records:")
    print(df[["date", "day_of_week", "meal_type", "dish_name", "registered_students", "actual_servings_needed", "quantity_prepared", "discarded_quantity", "discarded_food_cost_inr"]].head(5).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mess_data.csv")
    df_gen = generate_synthetic_data(num_days=300, output_path=out_file)
    print_dataset_summary(df_gen, out_file)
