"""
Backfill Pipeline - Pearls AQI Predictor
Fetches HISTORICAL pollutant data for Karachi from OpenWeather's free
Air Pollution History API, engineers the same features as the live
pipeline, and bulk-inserts them into the Hopsworks feature store.

Note: OpenWeather's free tier gives historical POLLUTANTS but not
historical WEATHER (temp/humidity/wind history is a paid product).
So weather columns will be NaN for backfilled rows - that's expected,
not a bug. The model will mainly learn from pollutant + time features.

Run: python backfill_features.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

CITY = "karachi"
# Karachi coordinates (OpenWeather needs lat/lon, not a city name)
LAT, LON = 24.8607, 67.0011

DAYS_BACK = 30  # how many days of history to pull


def pm25_to_aqi(pm25: float) -> float:
    """
    Convert a PM2.5 concentration (µg/m3) to a US EPA AQI value (0-500),
    using the standard EPA breakpoint table. This keeps the backfilled
    'aqi' values on the SAME scale as the live AQICN pipeline (which
    already reports AQI, not raw concentration).
    """
    if pm25 is None or pd.isna(pm25):
        return None

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                (aqi_high - aqi_low) / (c_high - c_low) * (pm25 - c_low) + aqi_low
            )

    return 500  # cap at max if it's off-the-chart hazardous


def fetch_historical_pollution(start_ts: int, end_ts: int) -> list:
    """Call OpenWeather's Air Pollution History API for a unix timestamp range."""
    url = (
        "http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()
    return payload.get("list", [])


def build_feature_rows(records: list) -> pd.DataFrame:
    """Turn OpenWeather's hourly records into the same feature schema as the live pipeline."""
    rows = []
    for record in records:
        dt = datetime.utcfromtimestamp(record["dt"])
        components = record.get("components", {})
        pm25 = components.get("pm2_5")

        rows.append({
            "city": CITY,
            "timestamp": dt,
            "date": dt.date().isoformat(),
            "aqi": pm25_to_aqi(pm25),
            "pm25": pm25,
            "pm10": components.get("pm10"),
            "o3": components.get("o3"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "co": components.get("co"),
            # Not available from the free historical endpoint - left as NaN
            "temperature": None,
            "humidity": None,
            "pressure": None,
            "wind_speed": None,
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "day_of_week": dt.weekday(),
        })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "pressure", "wind_speed",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def push_to_feature_store(df: pd.DataFrame):
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    # version=2 with a composite primary key, so each (city, date, hour)
    # gets its own row instead of overwriting a single "city" row.
    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        description="Hourly AQI, pollutant, and weather features for Karachi",
        primary_key=["city", "date", "hour"],
        event_time="timestamp",
        online_enabled=True,
        time_travel_format="HUDI",
    )

    feature_group.insert(df)
    print(f"Inserted {len(df)} historical row(s) into feature group 'aqi_features' (v2).")


def main():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_BACK)

    print(f"Backfilling {CITY} data from {start_time.date()} to {end_time.date()}...")

    all_rows = []
    # OpenWeather allows large ranges in one call, but we chunk by week
    # to keep each request small and easy to debug if one fails.
    chunk_start = start_time
    while chunk_start < end_time:
        chunk_end = min(chunk_start + timedelta(days=7), end_time)
        print(f"  Fetching {chunk_start.date()} to {chunk_end.date()}...")

        records = fetch_historical_pollution(
            int(chunk_start.timestamp()), int(chunk_end.timestamp())
        )
        all_rows.extend(records)

        chunk_start = chunk_end
        time.sleep(1)  # be polite to the free-tier rate limit

    print(f"Fetched {len(all_rows)} hourly records total.")

    df = build_feature_rows(all_rows)
    print(df.head())
    print(f"... {len(df)} rows total")

    push_to_feature_store(df)


if __name__ == "__main__":
    main()