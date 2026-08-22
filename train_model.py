"""
Training Pipeline - Pearls AQI Predictor
Pulls historical + live features from the Hopsworks feature store,
engineers a couple of extra features (lag, rolling average, AQI
change rate), trains Ridge Regression and Random Forest, evaluates
both with RMSE / MAE / R^2, and saves the better model to the
Hopsworks Model Registry.

Run: python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Weather columns are ~99% missing right now (see fetch_features.py notes),
# so we train on pollutants + time features only for this first version.
FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_rolling_3", "aqi_change_rate",
]
TARGET_COL = "aqi"


def load_data() -> pd.DataFrame:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)

    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from feature store.")
    return df


def engineer_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling average, and AQI change-rate features (as required by the spec)."""
    df = df.copy()
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_rolling_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()

    # Fill remaining pollutant gaps with the column median (simple, defensible baseline)
    pollutant_cols = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    for col in pollutant_cols:
        df[col] = df[col].fillna(df[col].median())

    # Drop rows where we still don't have a target or the new lag/rolling features
    # (the first couple of rows won't have a valid lag/rolling window)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.15):
    """Split chronologically (NOT randomly) since this is a forecasting problem."""
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    return train_df, test_df


def evaluate(y_true, y_pred, model_name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name:20s} | RMSE: {rmse:7.3f} | MAE: {mae:7.3f} | R2: {r2:7.3f}")
    return {"model_name": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def train_and_evaluate(train_df, test_df):
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    results = []
    trained_models = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results.append(evaluate(y_test, ridge_preds, "Ridge Regression"))
    trained_models["Ridge Regression"] = ridge

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append(evaluate(y_test, rf_preds, "Random Forest"))
    trained_models["Random Forest"] = rf

    return results, trained_models


def save_best_model(results, trained_models, train_df):
    best = min(results, key=lambda r: r["rmse"])
    best_name = best["model_name"]
    best_model = trained_models[best_name]

    print(f"\nBest model: {best_name} (RMSE={best['rmse']:.3f})")

    local_path = "aqi_model.pkl"
    joblib.dump(best_model, local_path)

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    mr = project.get_model_registry()

    model = mr.sklearn.create_model(
        name="aqi_predictor_model",
        metrics={"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]},
        description=f"Best model ({best_name}) for Karachi AQI prediction",
        input_example=train_df[FEATURE_COLS].iloc[:1],
    )
    model.save(local_path)
    print("Model saved to Hopsworks Model Registry.")


def main():
    df = load_data()
    df = engineer_extra_features(df)
    train_df, test_df = time_based_split(df)

    print("\n--- Model comparison ---")
    results, trained_models = train_and_evaluate(train_df, test_df)

    save_best_model(results, trained_models, train_df)


if __name__ == "__main__":
    main()