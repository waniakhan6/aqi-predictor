"""
Quick check: is weather data actually in the feature store or not?
Run: python check_weather.py
"""
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=2)

df = fg.read()
print(f"Total rows: {len(df)}")
print("\nMissing values per weather column:")
print(df[["temperature", "humidity", "pressure", "wind_speed"]].isna().sum())
print("\nSample of actual values:")
print(df[["timestamp", "temperature", "humidity", "pressure", "wind_speed"]].head(10))