"""
Dashboard - Pearls AQI Predictor
Streamlit app that loads the latest features + the 3 saved forecast
models (24h/48h/72h) from Hopsworks, shows the current AQI, a 3-day
forecast, a historical trend chart with AQI threshold bands, and
SHAP-based feature importance.

Run: streamlit run app.py
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# Official US EPA AQI category colors and breakpoints - the standard any
# air-quality dashboard uses, rather than an arbitrary decorative palette.
# Each entry also carries the text color needed for readable contrast on
# that background (e.g. yellow needs dark text, not light).
AQI_BANDS = [
    (0, 50, "Good", "#00E400", "#1B1B1F"),
    (50, 100, "Moderate", "#FFDE33", "#1B1B1F"),
    (100, 150, "Unhealthy (Sensitive Groups)", "#FF9933", "#1B1B1F"),
    (150, 200, "Unhealthy", "#CC0033", "#FFFFFF"),
    (200, 300, "Very Unhealthy", "#660099", "#FFFFFF"),
    (300, 500, "Hazardous", "#7E0023", "#FFFFFF"),
]

# Palette accents (beige/navy/sage/rust) used for chrome around the
# AQI colors, which stay standard for readability.
ACCENT_NAVY = "#242953"
ACCENT_RUST = "#5C2E1F"
BG_CREAM = "#FBF3E7"

st.set_page_config(page_title="Karachi AQI Forecast", page_icon="📊", layout="wide")


def aqi_category_color(aqi: float):
    if aqi is None or pd.isna(aqi):
        return "Unknown", "#9E9E9E", "#FFFFFF"
    for low, high, label, color, text_color in AQI_BANDS:
        if low < aqi <= high or (low == 0 and aqi <= high):
            return label, color, text_color
    return "Hazardous", AQI_BANDS[-1][3], AQI_BANDS[-1][4]


@st.cache_resource
def get_project():
    return hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)


@st.cache_data(ttl=600)
def load_recent_features(_project, n_rows: int = 168):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    df = fg.read()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df.tail(n_rows).reset_index(drop=True)


@st.cache_resource
def load_model_with_metrics(_project, horizon_hours: int):
    model_name = f"aqi_predictor_{horizon_hours}h"
    mr = _project.get_model_registry()
    models = mr.get_models(name=model_name)
    if not models:
        return None, None
    best = max(models, key=lambda m: int(m.version))
    model_dir = best.download()
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        return None, None
    model = joblib.load(pkl_files[0])
    return model, best.training_metrics


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_rolling_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()
    return df


def build_current_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.iloc[-1].copy()
    row = {col: latest.get(col) for col in FEATURE_COLS}
    feature_df = pd.DataFrame([row])[FEATURE_COLS]

    for col in FEATURE_COLS:
        if pd.isna(feature_df[col].iloc[0]):
            fallback = df[col].median() if col in df.columns else None
            feature_df.at[0, col] = fallback if pd.notna(fallback) else 0

    return feature_df


def plot_trend_with_bands(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG_CREAM)
    ax.set_facecolor(BG_CREAM)

    y_max = max(220, df["aqi"].max() * 1.15)
    for low, high, label, color, _ in AQI_BANDS:
        if low >= y_max:
            continue
        ax.axhspan(low, min(high, y_max), color=color, alpha=0.15, lw=0)

    ax.plot(df["timestamp"], df["aqi"], color=ACCENT_NAVY, linewidth=1.8)
    ax.scatter(df["timestamp"].iloc[[-1]], df["aqi"].iloc[[-1]], color=ACCENT_NAVY, s=32, zorder=5)

    ax.set_ylim(0, y_max)
    ax.set_ylabel("AQI (US EPA)", color="#1B1B1F")
    ax.tick_params(colors="#1B1B1F")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#1B1B1F")
    ax.grid(axis="y", color="#D9CBB5", linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main():
    st.title("Karachi AQI Forecast")
    st.caption("Pearls AQI Predictor — feature pipeline, forecasting models, and explainability, refreshed hourly.")

    project = get_project()
    df = load_recent_features(project)

    if df.empty:
        st.error("No data found in the feature store yet.")
        return

    df = add_engineered_features(df)
    current_aqi = df["aqi"].iloc[-1]
    last_updated = df["timestamp"].iloc[-1]
    category, bg_color, text_color = aqi_category_color(current_aqi)

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.metric("Current AQI — Karachi", f"{current_aqi:.0f}", help="US EPA Air Quality Index scale, 0-500.")
        st.markdown(
            f'<span style="background-color:{bg_color}; color:{text_color}; '
            f'padding:3px 12px; border-radius:4px; font-weight:600; font-size:0.85rem;">'
            f'{category}</span>',
            unsafe_allow_html=True,
        )
    with top_right:
        st.caption(f"Last reading: {last_updated.strftime('%b %d, %Y %H:%M UTC')}")
        st.caption(f"Rows in feature store window: {len(df)}")

    if current_aqi is not None and current_aqi > 150:
        st.warning("Air quality is in the Unhealthy range or worse. Limiting prolonged outdoor exertion is advisable.")

    st.divider()
    st.subheader("3-Day Forecast")

    feature_vector = build_current_feature_vector(df)
    cols = st.columns(len(HORIZONS))
    for col, horizon in zip(cols, HORIZONS):
        model, metrics = load_model_with_metrics(project, horizon)
        with col:
            if model is None:
                st.write(f"+{horizon}h — model not available")
                continue
            pred = model.predict(feature_vector)[0]
            pred_category, pred_bg, pred_text = aqi_category_color(pred)
            rmse = metrics.get("rmse") if metrics else None

            st.metric(f"+{horizon}h", f"{pred:.0f}")
            st.markdown(
                f'<span style="background-color:{pred_bg}; color:{pred_text}; '
                f'padding:2px 10px; border-radius:4px; font-size:0.8rem; font-weight:600;">'
                f'{pred_category}</span>',
                unsafe_allow_html=True,
            )
            if rmse is not None:
                st.caption(f"Model RMSE: ±{rmse:.1f}")

    st.caption(
        "Forecasts carry meaningful uncertainty (see RMSE above) — training data currently spans "
        "roughly one month for one city, which limits accuracy at longer horizons."
    )

    st.divider()
    st.subheader("Recent AQI Trend")
    st.pyplot(plot_trend_with_bands(df))
    st.caption("Shaded bands mark US EPA AQI categories (Good → Hazardous).")

    st.divider()
    st.subheader("Feature Importance (24h model)")
    st.caption("Mean absolute SHAP value per feature — which inputs most influence the 24h-ahead prediction.")

    model_24h, _ = load_model_with_metrics(project, 24)
    shap_background = df.dropna(subset=["aqi_lag_1"]).reset_index(drop=True)
    if model_24h is not None and len(shap_background) >= 10:
        try:
            background = shap_background[FEATURE_COLS].copy()
            for col in FEATURE_COLS:
                median = background[col].median()
                background[col] = background[col].fillna(median if pd.notna(median) else 0)

            explainer = shap.Explainer(model_24h, background)
            shap_values = explainer(background)

            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            importance = pd.Series(mean_abs_shap, index=FEATURE_COLS).sort_values()

            fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_CREAM)
            ax.set_facecolor(BG_CREAM)
            ax.barh(importance.index, importance.values, color=ACCENT_RUST)
            ax.set_xlabel("mean(|SHAP value|)", color="#1B1B1F")
            ax.tick_params(colors="#1B1B1F")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#1B1B1F")
            ax.grid(axis="x", color="#D9CBB5", linewidth=0.8)
            fig.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP explanation not available right now ({e}).")
    else:
        st.info("Not enough data yet to compute feature importance.")

    st.divider()
    with st.expander("Raw recent data"):
        st.dataframe(
            df[["timestamp", "aqi", "pm25", "pm10", "temperature", "humidity"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()