"""
train_model.py
==============
Machine Learning Training Pipeline for MessMind AI.
Trains a Scikit-learn RandomForestRegressor to predict actual servings needed
prior to meal preparation without post-meal data leakage.
Compares performance against simple historical and mean baselines.
"""

import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

CATEGORICAL_FEATURES = ["day_of_week", "meal_type", "dish_name"]
NUMERICAL_FEATURES = [
    "expected_attendance",
    "registered_students",
    "historical_demand",
    "registration_ratio",
    "is_weekend"
]
TARGET = "actual_servings_needed"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive pre-meal available features without data leakage."""
    data = df.copy()
    # Registration ratio: how strongly students committed relative to expected capacity
    expected_safe = data["expected_attendance"].replace(0, 1)
    data["registration_ratio"] = (data["registered_students"] / expected_safe).round(4)
    # Weekend indicator
    data["is_weekend"] = data["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return data


def train_and_evaluate(
    data_path: str = "data/mess_data.csv",
    model_save_path: str = "models/demand_model.joblib",
    metrics_save_path: str = "models/model_metrics.json"
):
    """
    Train Scikit-learn regression pipeline with chronological train/test split.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    # Sort chronologically by date
    df = df.sort_values(by=["date", "meal_type", "dish_name"]).reset_index(drop=True)

    # Chronological train/test split (80% train, 20% test based on unique dates)
    unique_dates = df["date"].drop_duplicates().tolist()
    split_idx = int(len(unique_dates) * 0.80)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    X_train = train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_test = test_df[TARGET]

    print("=" * 80)
    print("MESSMIND AI — MODEL TRAINING & EVALUATION")
    print("=" * 80)
    print(f"Total Records: {len(df):,}")
    print(f"Training Records: {len(train_df):,} ({len(train_dates)} dates: {min(train_dates)} to {max(train_dates)})")
    print(f"Held-out Test Records: {len(test_df):,} ({len(test_dates)} dates: {min(test_dates)} to {max(test_dates)})")
    print(f"Pre-meal Features: {CATEGORICAL_FEATURES + NUMERICAL_FEATURES}")
    print(f"Target Variable: '{TARGET}' (strictly pre-meal prediction)")
    print("-" * 80)

    # Build preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_model)
    ])

    # Train model
    print("Training RandomForestRegressor pipeline...")
    pipeline.fit(X_train, y_train)
    y_pred_rf = pipeline.predict(X_test)

    # Baselines
    # Baseline 1: Historical demand (what a naive manager would use: past rolling average)
    y_pred_hist_baseline = test_df["historical_demand"].values

    # Baseline 2: Dummy mean regressor
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_mean_baseline = dummy.predict(X_test)

    # Metrics calculation
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_r2 = float(r2_score(y_test, y_pred_rf))

    hist_mae = float(mean_absolute_error(y_test, y_pred_hist_baseline))
    hist_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_hist_baseline)))
    hist_r2 = float(r2_score(y_test, y_pred_hist_baseline))

    mean_mae = float(mean_absolute_error(y_test, y_pred_mean_baseline))
    mean_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_mean_baseline)))
    mean_r2 = float(r2_score(y_test, y_pred_mean_baseline))

    print("Model Evaluation on Chronologically Held-Out Test Data:")
    print(f"  RandomForest Model   -> MAE: {rf_mae:.2f} servings | RMSE: {rf_rmse:.2f} servings | R²: {rf_r2:.3f}")
    print(f"  Historical Baseline  -> MAE: {hist_mae:.2f} servings | RMSE: {hist_rmse:.2f} servings | R²: {hist_r2:.3f}")
    print(f"  Mean Baseline        -> MAE: {mean_mae:.2f} servings | RMSE: {mean_rmse:.2f} servings | R²: {mean_r2:.3f}")
    print("-" * 80)

    # Sample comparison
    comparison_df = test_df[["date", "meal_type", "dish_name", "registered_students", "historical_demand"]].copy()
    comparison_df["Actual_Needed"] = y_test.values
    comparison_df["Predicted_RF"] = np.round(y_pred_rf).astype(int)
    comparison_df["Error_RF"] = comparison_df["Predicted_RF"] - comparison_df["Actual_Needed"]
    comparison_df["Hist_Baseline"] = np.round(y_pred_hist_baseline).astype(int)

    print("Sample Test Set Predictions (First 8 Instances):")
    print(comparison_df.head(8).to_string(index=False))
    print("=" * 80)

    # Save model and preprocessing pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"Trained pipeline successfully saved to: {model_save_path}")

    # Prepare metrics payload
    metrics_payload = {
        "model_type": "RandomForestRegressor Pipeline (OneHotEncoder + StandardScaler)",
        "train_timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "dataset_records_total": len(df),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "split_strategy": "Chronological (First 80% dates for train, final 20% for test)",
        "features_used": CATEGORICAL_FEATURES + NUMERICAL_FEATURES,
        "target": TARGET,
        "rf_metrics": {
            "mae": round(rf_mae, 3),
            "rmse": round(rf_rmse, 3),
            "r2": round(rf_r2, 4)
        },
        "historical_baseline_metrics": {
            "mae": round(hist_mae, 3),
            "rmse": round(hist_rmse, 3),
            "r2": round(hist_r2, 4)
        },
        "mean_baseline_metrics": {
            "mae": round(mean_mae, 3),
            "rmse": round(mean_rmse, 3),
            "r2": round(mean_r2, 4)
        },
        "sample_predictions": comparison_df.head(10).to_dict(orient="records"),
        "synthetic_notice": (
            "These metrics demonstrate prototype performance on synthetic hostel data. "
            "They illustrate model feasibility under simulated demand patterns and do not "
            "guarantee identical predictive performance on live college mess attendance."
        )
    }

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_save_path}")

    return pipeline, metrics_payload


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_evaluate(
        data_path=os.path.join(base_dir, "data", "mess_data.csv"),
        model_save_path=os.path.join(base_dir, "models", "demand_model.joblib"),
        metrics_save_path=os.path.join(base_dir, "models", "model_metrics.json")
    )
