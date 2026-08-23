"""
Weather Backfill - Pearls AQI Predictor
Fetches REAL historical weather for Karachi from Open-Meteo's free
archive API (no key needed), merges it onto the existing feature
group rows by (date, hour), and re-inserts (upserts) so temperature/
humidity/pressure/wind_speed are finally populated for training.

Run: python backfill_weather.py
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

LAT, LON = 24.8607, 67.0011


def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Pull hourly historical weather from Open-Meteo (free, no API key required)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
        "&wind_speed_unit=ms&timezone=UTC"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()["hourly"]

    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(payload["time"]),
        "temperature": payload["temperature_2m"],
        "humidity": payload["relative_humidity_2m"],
        "pressure": payload["surface_pressure"],
        "wind_speed": payload["wind_speed_10m"],
    })
    weather_df["date"] = weather_df["timestamp"].dt.date.astype(str)
    weather_df["hour"] = weather_df["timestamp"].dt.hour
    return weather_df


def main():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)

    df = fg.read()
    print(f"Loaded {len(df)} existing rows.")

    start_date = df["date"].min()
    end_date = df["date"].max()
    print(f"Fetching weather for {start_date} to {end_date}...")

    weather_df = fetch_historical_weather(start_date, end_date)
    print(f"Fetched {len(weather_df)} hourly weather records.")

    # Drop the old (mostly empty) weather columns, then merge in the real ones by (date, hour)
    df = df.drop(columns=["temperature", "humidity", "pressure", "wind_speed"])
    merged = df.merge(weather_df[["date", "hour", "temperature", "humidity", "pressure", "wind_speed"]],
                       on=["date", "hour"], how="left")

    numeric_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    merged[numeric_cols] = merged[numeric_cols].astype(float)

    missing = merged["temperature"].isna().sum()
    print(f"Rows still missing weather after merge: {missing} (likely today's not-yet-archived hours)")

    fg.insert(merged)
    print(f"Upserted {len(merged)} rows with real weather data into 'aqi_features' v2.")


if __name__ == "__main__":
    main()