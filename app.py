"""
Dashboard - Pearls AQI Predictor
Streamlit app that loads the latest features + the 3 saved forecast
models (24h/48h/72h) from Hopsworks, shows the current AQI, a 3-day
forecast, a historical trend chart, and a hazard alert banner.

Run: streamlit run app.py
"""

import os
import glob
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_COLS = [
    "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_rolling_3", "aqi_change_rate",
]
HORIZONS = [24, 48, 72]

st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="centered")


def aqi_category(aqi: float):
    """US EPA AQI category, label + color, for display."""
    if aqi is None:
        return "Unknown", "gray"
    if aqi <= 50:
        return "Good", "green"
    if aqi <= 100:
        return "Moderate", "yellow"
    if aqi <= 150:
        return "Unhealthy (Sensitive Groups)", "orange"
    if aqi <= 200:
        return "Unhealthy", "red"
    if aqi <= 300:
        return "Very Unhealthy", "purple"
    return "Hazardous", "maroon"


@st.cache_resource
def get_project():
    return hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)


@st.cache_data(ttl=600)
def load_recent_features(_project, n_rows: int = 72):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    df = fg.read()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df.tail(n_rows).reset_index(drop=True)


@st.cache_resource
def load_model(_project, horizon_hours: int):
    model_name = f"aqi_predictor_{horizon_hours}h"
    mr = _project.get_model_registry()
    models = mr.get_models(name=model_name)
    if not models:
        return None
    best = max(models, key=lambda m: m.version)
    model_dir = best.download()
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        return None
    return joblib.load(pkl_files[0])


def build_current_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Build the single-row feature vector (matching training schema) from the latest data."""
    latest = df.iloc[-1].copy()

    aqi_lag_1 = df["aqi"].iloc[-2] if len(df) >= 2 else df["aqi"].iloc[-1]
    aqi_rolling_3 = df["aqi"].tail(3).mean()
    aqi_change_rate = df["aqi"].iloc[-1] - df["aqi"].iloc[-2] if len(df) >= 2 else 0.0

    base_cols = ["pm10", "o3", "no2", "so2", "co",
                 "temperature", "humidity", "pressure", "wind_speed",
                 "hour", "day", "month", "day_of_week"]
    row = {col: latest.get(col) for col in base_cols}
    row["aqi_lag_1"] = aqi_lag_1
    row["aqi_rolling_3"] = aqi_rolling_3
    row["aqi_change_rate"] = aqi_change_rate

    feature_df = pd.DataFrame([row])[FEATURE_COLS]

    # If the latest row is missing any values (e.g. the live station only
    # reports pm25), fall back to the recent median for that column - same
    # approach used during training, so predictions don't break on NaN.
    for col in FEATURE_COLS:
        if pd.isna(feature_df[col].iloc[0]):
            fallback = df[col].median() if col in df.columns else None
            feature_df.at[0, col] = fallback if pd.notna(fallback) else 0

    return feature_df


def main():
    st.title("🌫️ Karachi AQI Forecast")
    st.caption("Pearls AQI Predictor — live data, 3-day forecast")

    project = get_project()
    df = load_recent_features(project)

    if df.empty:
        st.error("No data found in the feature store yet.")
        return

    current_aqi = df["aqi"].iloc[-1]
    category, color = aqi_category(current_aqi)

    st.metric("Current AQI (Karachi)", f"{current_aqi:.0f}")
    st.markdown(f"**Status:** :{color}[{category}]")

    if current_aqi is not None and current_aqi > 150:
        st.error("⚠️ Hazardous air quality — limit outdoor activity.")

    st.subheader("3-Day Forecast")
    feature_vector = build_current_feature_vector(df)

    cols = st.columns(len(HORIZONS))
    for col, horizon in zip(cols, HORIZONS):
        model = load_model(project, horizon)
        with col:
            if model is None:
                st.write(f"**+{horizon}h**")
                st.write("Model not available")
                continue
            pred = model.predict(feature_vector)[0]
            pred_category, pred_color = aqi_category(pred)
            st.write(f"**+{horizon}h**")
            st.metric(label="", value=f"{pred:.0f}")
            st.markdown(f":{pred_color}[{pred_category}]")

    st.subheader("Recent AQI Trend")
    trend_df = df[["timestamp", "aqi"]].set_index("timestamp")
    st.line_chart(trend_df)

    with st.expander("Show raw recent data"):
        st.dataframe(df[["timestamp", "aqi", "pm25", "pm10", "temperature", "humidity"]])


if __name__ == "__main__":
    main()