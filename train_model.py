"""
Training Pipeline v2 - Pearls AQI Predictor (real forecasting)
Pulls features from Hopsworks, engineers lag/rolling/change-rate
features, then trains a SEPARATE model for each forecast horizon
(24h, 48h, 72h ahead) so the dashboard can show a genuine 3-day
forecast instead of just estimating the current hour.

Key differences from the first version:
- Target is AQI N HOURS IN THE FUTURE, not the current hour's AQI.
- pm25 is dropped as an input feature, since historical 'aqi' values
  were computed directly FROM pm25 (EPA formula) - using it as a
  feature let the model "cheat" by reversing that formula instead of
  learning real pollution/time patterns.

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# pm25 intentionally excluded - see module docstring.
# Weather columns excluded too - still ~99% missing (see fetch_features.py notes).
FEATURE_COLS = [
    "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_rolling_3", "aqi_change_rate",
]
TARGET_COL = "aqi"
HORIZONS_HOURS = [24, 48, 72]  # next-day, 2-day, 3-day ahead


def load_data() -> pd.DataFrame:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)

    df = fg.read()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from feature store.")
    return df


def engineer_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling average, and AQI change-rate features (as required by the spec)."""
    df = df.copy()

    # The live AQICN station only ever reports pm25 - pm10/o3/no2/so2/co
    # are always NaN for live rows. Imputing those with a median gets
    # WORSE over time as more live-only rows accumulate (dilutes the
    # dataset with guessed values). Instead, drop rows missing real
    # pollutant data BEFORE computing lag/rolling features, so those
    # features aren't computed across artificial gaps in the timeline.
    pollutant_cols = ["pm10", "o3", "no2", "so2", "co"]
    df = df.dropna(subset=pollutant_cols).sort_values("timestamp").reset_index(drop=True)

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_rolling_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()

    # Weather IS reported live by AQICN, so a light median fill there is fine.
    weather_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in weather_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def make_horizon_dataset(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """
    Shift the target 'horizon_hours' rows into the future, so each row's
    features are used to predict AQI that many hours ahead (assumes
    roughly hourly-spaced data, which matches our pipeline's cadence).
    """
    horizon_df = df.copy()
    horizon_df["target"] = horizon_df["aqi"].shift(-horizon_hours)
    horizon_df = horizon_df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    return horizon_df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.15):
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_neural_network(X_train, y_train, X_test, y_test):
    """A small feed-forward network - deep learning is included here for
    genuine comparison against the classical models, as called for by the
    project spec. With only a few hundred training rows, a deep model is
    unlikely to beat Ridge/RF, and that outcome itself is a valid, reportable
    finding rather than a failure."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0,
    )

    preds = model.predict(X_test_scaled, verbose=0).flatten()
    return preds


def evaluate(y_true, y_pred, label: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:20s} | RMSE: {rmse:7.3f} | MAE: {mae:7.3f} | R2: {r2:7.3f}")
    return {"model_name": label, "rmse": rmse, "mae": mae, "r2": r2}


def train_horizon_model(df: pd.DataFrame, horizon_hours: int):
    horizon_df = make_horizon_dataset(df, horizon_hours)

    if len(horizon_df) < 20:
        print(f"  Not enough rows ({len(horizon_df)}) to train the {horizon_hours}h model yet. Skipping.")
        return None

    train_df, test_df = time_based_split(horizon_df)
    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

    print(f"\n--- {horizon_hours}h-ahead forecast (train={len(train_df)}, test={len(test_df)}) ---")

    results, trained_models = [], {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate(y_test, ridge.predict(X_test), "Ridge Regression"))
    trained_models["Ridge Regression"] = ridge

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate(y_test, rf.predict(X_test), "Random Forest"))
    trained_models["Random Forest"] = rf

    # Neural network: trained and evaluated for genuine comparison, but not
    # currently eligible for production deployment - the dashboard loads
    # scikit-learn .pkl models, and with this little data it hasn't beaten
    # the classical models anyway (see printed comparison below).
    nn_preds = train_neural_network(X_train, y_train, X_test, y_test)
    nn_result = evaluate(y_test, nn_preds, "Neural Network (TF)")
    results.append(nn_result)

    deployable_results = [r for r in results if r["model_name"] != "Neural Network (TF)"]
    best = min(deployable_results, key=lambda r: r["rmse"])
    best_model = trained_models[best["model_name"]]
    print(f"  -> Best deployable model for {horizon_hours}h: {best['model_name']} (RMSE={best['rmse']:.3f})")
    if nn_result["rmse"] < best["rmse"]:
        print(f"  (Note: Neural Network scored lower RMSE={nn_result['rmse']:.3f} but is not deployed - see script notes.)")

    return {
        "horizon_hours": horizon_hours,
        "best_result": best,
        "best_model": best_model,
        "sample_input": train_df[FEATURE_COLS].iloc[:1],
    }


def save_model_to_registry(project, horizon_result: dict):
    horizon_hours = horizon_result["horizon_hours"]
    best = horizon_result["best_result"]
    model_name = f"aqi_predictor_{horizon_hours}h"

    local_path = f"aqi_model_{horizon_hours}h.pkl"
    joblib.dump(horizon_result["best_model"], local_path)

    mr = project.get_model_registry()
    model = mr.sklearn.create_model(
        name=model_name,
        metrics={"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]},
        description=f"{best['model_name']} - predicts AQI {horizon_hours}h ahead for Karachi",
        input_example=horizon_result["sample_input"],
    )
    model.save(local_path)
    print(f"Saved '{model_name}' to the Model Registry.")


def main():
    df = load_data()
    df = engineer_extra_features(df)

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )

    for horizon_hours in HORIZONS_HOURS:
        result = train_horizon_model(df, horizon_hours)
        if result is not None:
            save_model_to_registry(project, result)


if __name__ == "__main__":
    main()