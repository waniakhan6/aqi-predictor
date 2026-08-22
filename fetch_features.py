"""
Feature Pipeline - Pearls AQI Predictor
Fetches current AQI + weather data for Karachi from AQICN,
engineers features, and pushes them to the Hopsworks feature store.

Run manually first to test: python fetch_features.py
Later, this same script is what GitHub Actions will run every hour.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

CITY = "karachi"


def fetch_raw_data(city: str) -> dict:
    """Call the AQICN API for the given city and return the raw JSON 'data' block."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "ok":
        raise ValueError(f"AQICN API error: {payload}")

    return payload["data"]


def build_feature_row(raw: dict) -> dict:
    """Turn the raw AQICN response into a flat row of engineered features."""
    iaqi = raw.get("iaqi", {})
    now = datetime.utcnow()

    row = {
        # Identifiers / timestamp
        "city": CITY,
        "timestamp": now,
        "date": now.date().isoformat(),

        # Target + pollutants (use .get with None default -- not every station reports every pollutant)
        "aqi": raw.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),

        # Weather
        "temperature": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
        "pressure": iaqi.get("p", {}).get("v"),
        "wind_speed": iaqi.get("w", {}).get("v"),

        # Time-based features
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),  # 0 = Monday
    }
    return row


def push_to_feature_store(df: pd.DataFrame):
    """Connect to Hopsworks and insert the row into (or create) the AQI feature group."""
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

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
    print(f"Inserted {len(df)} row(s) into feature group 'aqi_features'.")


def main():
    print(f"Fetching AQI data for {CITY}...")
    raw = fetch_raw_data(CITY)

    row = build_feature_row(raw)
    df = pd.DataFrame([row])

    # Force pollutant/weather columns to float so a missing reading becomes
    # NaN (a proper numeric "empty") instead of None, which Hopsworks can't
    # infer a column type from.
    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "pressure", "wind_speed",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)

    print(df)

    push_to_feature_store(df)


if __name__ == "__main__":
    main()