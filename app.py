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
import shap
import matplotlib.pyplot as plt
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

# Palette: dark slate background with a teal/cyan accent (matches .streamlit/config.toml)
ACCENT = "#2DD4BF"
CATEGORY_COLORS = {
    "Good": "#22C55E",
    "Moderate": "#EAB308",
    "Unhealthy (Sensitive Groups)": "#F97316",
    "Unhealthy": "#EF4444",
    "Very Unhealthy": "#A855F7",
    "Hazardous": "#7F1D1D",
    "Unknown": "#64748B",
}

st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="centered")

st.markdown(
    f"""
    <style>
    .card {{
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }}
    .forecast-card {{
        background-color: #1E293B;
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        border-top: 4px solid {ACCENT};
    }}
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-top: 6px;
    }}
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 10px;
        color: #F1F5F9;
    }}
    .metric-big {{
        font-size: 3rem;
        font-weight: 800;
        color: {ACCENT};
        line-height: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def aqi_category(aqi: float):
    """US EPA AQI category, for display."""
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy (Sensitive Groups)"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def badge_html(category: str) -> str:
    color = CATEGORY_COLORS.get(category, "#64748B")
    return f'<span class="badge" style="background-color:{color};">{category}</span>'


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
    best = max(models, key=lambda m: int(m.version))
    model_dir = best.download()
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        return None
    return joblib.load(pkl_files[0])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag/rolling/change-rate features across the whole dataframe (not just one row) -
    needed both for the current prediction and for SHAP's background dataset."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_rolling_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()
    return df


def build_current_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Build the single-row feature vector (matching training schema) from the latest data.
    Expects df to already have aqi_lag_1/aqi_rolling_3/aqi_change_rate (see add_engineered_features)."""
    latest = df.iloc[-1].copy()
    row = {col: latest.get(col) for col in FEATURE_COLS}
    feature_df = pd.DataFrame([row])[FEATURE_COLS]

    for col in FEATURE_COLS:
        if pd.isna(feature_df[col].iloc[0]):
            fallback = df[col].median() if col in df.columns else None
            feature_df.at[0, col] = fallback if pd.notna(fallback) else 0

    return feature_df


def main():
    st.markdown(
        "<h1 style='margin-bottom:0;'>🌫️ Karachi AQI Forecast</h1>"
        "<p style='color:#94A3B8; margin-top:4px;'>Pearls AQI Predictor — live data, 3-day forecast</p>",
        unsafe_allow_html=True,
    )

    project = get_project()
    df = load_recent_features(project)

    if df.empty:
        st.error("No data found in the feature store yet.")
        return

    current_aqi = df["aqi"].iloc[-1]
    category = aqi_category(current_aqi)

    st.markdown(
        f"""
        <div class="card">
            <div style="color:#94A3B8; font-size:0.95rem;">Current AQI — Karachi</div>
            <div class="metric-big">{current_aqi:.0f}</div>
            {badge_html(category)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_aqi is not None and current_aqi > 150:
        st.error("⚠️ Hazardous air quality — limit outdoor activity.")

    df = add_engineered_features(df)

    st.markdown('<div class="section-title">3-Day Forecast</div>', unsafe_allow_html=True)
    feature_vector = build_current_feature_vector(df)

    cols = st.columns(len(HORIZONS))
    for col, horizon in zip(cols, HORIZONS):
        model = load_model(project, horizon)
        with col:
            if model is None:
                st.markdown(
                    f'<div class="forecast-card"><b>+{horizon}h</b><br>Model not available</div>',
                    unsafe_allow_html=True,
                )
                continue
            pred = model.predict(feature_vector)[0]
            pred_category = aqi_category(pred)
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div style="color:#94A3B8; font-weight:600;">+{horizon}h</div>
                    <div style="font-size:2rem; font-weight:800; color:{ACCENT};">{pred:.0f}</div>
                    {badge_html(pred_category)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent AQI Trend</div>', unsafe_allow_html=True)
    trend_df = df[["timestamp", "aqi"]].set_index("timestamp")
    st.line_chart(trend_df, color=ACCENT)

    st.markdown('<div class="section-title">Why this forecast? (Feature Importance)</div>', unsafe_allow_html=True)
    st.caption("Which features most influence the 24h-ahead prediction, based on recent data.")
    model_24h = load_model(project, 24)
    shap_background = df.dropna(subset=["aqi_lag_1"]).reset_index(drop=True)
    if model_24h is not None and len(shap_background) >= 10:
        try:
            background = shap_background[FEATURE_COLS].copy()
            for col in FEATURE_COLS:
                median = background[col].median()
                background[col] = background[col].fillna(median if pd.notna(median) else 0)

            explainer = shap.Explainer(model_24h, background)
            shap_values = explainer(background)

            plt.style.use("dark_background")
            fig, ax = plt.subplots(facecolor="#0F172A")
            ax.set_facecolor("#0F172A")
            shap.summary_plot(shap_values, background, plot_type="bar", show=False, color=ACCENT)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.info(f"SHAP explanation not available right now ({e}).")
    else:
        st.info("Not enough data yet to compute feature importance.")

    with st.expander("Show raw recent data"):
        st.dataframe(df[["timestamp", "aqi", "pm25", "pm10", "temperature", "humidity"]])


if __name__ == "__main__":
    main()