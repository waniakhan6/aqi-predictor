"""
Exploratory Data Analysis - Pearls AQI Predictor
Pulls all available features from Hopsworks and generates plots
identifying trends: AQI distribution, pollutant correlations,
hourly/daily/monthly seasonality patterns.

Run: python eda_analysis.py
Output: PNG plots saved to ./eda_plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import hopsworks

load_dotenv()

OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAVY = "#242953"
RUST = "#5C2E1F"
CREAM = "#FBF3E7"

plt.rcParams["figure.facecolor"] = CREAM
plt.rcParams["axes.facecolor"] = CREAM


def load_data() -> pd.DataFrame:
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    df = fg.read()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows for EDA.")
    return df


def plot_aqi_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["aqi"].dropna(), bins=30, color=NAVY, edgecolor="white")
    ax.set_title("Distribution of AQI Readings")
    ax.set_xlabel("AQI (US EPA scale)")
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/aqi_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "pressure", "wind_speed"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Between AQI, Pollutants, and Weather")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_hourly_pattern(df: pd.DataFrame):
    hourly_avg = df.groupby("hour")["aqi"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly_avg.index, hourly_avg.values, marker="o", color=NAVY)
    ax.set_title("Average AQI by Hour of Day")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Average AQI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9CBB5")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/hourly_pattern.png", dpi=150)
    plt.close(fig)


def plot_day_of_week_pattern(df: pd.DataFrame):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_avg = df.groupby("day_of_week")["aqi"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([day_names[i] for i in daily_avg.index], daily_avg.values, color=RUST)
    ax.set_title("Average AQI by Day of Week")
    ax.set_ylabel("Average AQI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9CBB5")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/day_of_week_pattern.png", dpi=150)
    plt.close(fig)


def plot_pm25_vs_aqi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["pm25"], df["aqi"], alpha=0.4, color=NAVY, s=18)
    ax.set_title("PM2.5 Concentration vs AQI")
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_ylabel("AQI")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pm25_vs_aqi.png", dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame):
    print("\n--- Summary Statistics ---")
    print(df[["aqi", "pm25", "pm10", "temperature", "humidity"]].describe())

    print("\n--- Key Observations ---")
    print(f"Mean AQI: {df['aqi'].mean():.1f}")
    print(f"Max AQI recorded: {df['aqi'].max():.0f}")
    print(f"Min AQI recorded: {df['aqi'].min():.0f}")
    worst_hour = df.groupby('hour')['aqi'].mean().idxmax()
    print(f"Hour of day with worst average AQI: {worst_hour}:00")


def main():
    df = load_data()
    plot_aqi_distribution(df)
    plot_correlation_heatmap(df)
    plot_hourly_pattern(df)
    plot_day_of_week_pattern(df)
    plot_pm25_vs_aqi(df)
    print_summary(df)
    print(f"\nAll plots saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()