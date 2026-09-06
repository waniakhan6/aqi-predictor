# Pearls AQI Predictor

A serverless, end-to-end machine learning pipeline that forecasts Karachi's Air Quality Index (AQI) up to **3 days ahead**  built as a data science internship project.

## Overview

This project automates the full lifecycle of an ML system: hourly data collection, feature engineering, multi-horizon forecasting, daily retraining, and an interactive dashboard with explainability — all running on free-tier serverless infrastructure with no manual intervention required.

## Features

- **Live data pipeline** — pulls real-time AQI and weather data for Karachi every hour via the AQICN API
- **Historical backfill** — ~30 days of hourly pollutant data (OpenWeather) and weather data (Open-Meteo)
- **Multi-horizon forecasting** — separate Ridge Regression / Random Forest models for 24h, 48h, and 72h-ahead AQI predictions
- **Automated retraining** — models retrain daily via GitHub Actions, no manual runs needed
- **Interactive dashboard** — Streamlit app showing current AQI, 3-day forecast, historical trend with US EPA AQI reference bands, and hazard alerts
- **Explainability** — SHAP feature importance so predictions aren't a black box
- **Full CI/CD** — hourly and daily GitHub Actions workflows using repository secrets

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn (Ridge Regression, Random Forest) |
| Feature Store / Model Registry | Hopsworks |
| Automation | GitHub Actions |
| Dashboard | Streamlit |
| Explainability | SHAP |
| Data Sources | AQICN API, OpenWeather API, Open-Meteo API |

## Project Structure

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # Hourly live data collection
│   └── training_pipeline.yml     # Daily model retraining
├── .streamlit/
│   └── config.toml               # Dashboard theme
├── fetch_features.py             # Live feature pipeline (AQICN)
├── backfill_features.py          # Historical pollutant backfill (OpenWeather)
├── backfill_weather.py           # Historical weather backfill (Open-Meteo)
├── train_model.py                # Multi-horizon training pipeline
├── app.py                        # Streamlit dashboard
├── check_weather.py              # Diagnostic: verify weather data in feature store
├── check_model_versions.py       # Diagnostic: inspect Model Registry versions
├── requirements.txt
├── .env.example                  # Template for required environment variables
└── Pearls_AQI_Predictor_Report.pdf   # Full project report
```

## How It Works

1. **Feature Pipeline** (`fetch_features.py`) runs hourly, fetching live AQI + weather data and writing engineered features (time-based + derived) to the Hopsworks feature store.
2. **Training Pipeline** (`train_model.py`) runs daily, pulling all available features, engineering lag/rolling/change-rate features, training models per forecast horizon, and registering the best one to the Model Registry.
3. **Dashboard** (`app.py`) loads the latest features and registered models to display the current AQI, forecast, trend, and SHAP explainability.

## Setup

```bash
# Clone the repo
git clone https://github.com/waniakhan6/pearls-aqi-predictor.git
cd pearls-aqi-predictor

# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your own API keys
cp .env.example .env
```

Required environment variables (see `.env.example`):
- `AQICN_API_TOKEN` — from [aqicn.org](https://aqicn.org/data-platform/token/)
- `HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT_NAME` — from [app.hopsworks.ai](https://app.hopsworks.ai)

Run locally:
```bash
python fetch_features.py     # pull live data
python train_model.py        # train/retrain models
streamlit run app.py         # launch dashboard
```

## Results

| Horizon | Best Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 24h | Ridge Regression | 4.99 | 4.35 | -0.34 |
| 48h | Random Forest | 8.95 | 7.80 | -4.90 |
| 72h | Random Forest | 8.48 | 7.58 | -5.32 |

Negative R² values are reported transparently — a known limitation of training on ~1 month of single-city data, detailed in the full report along with the engineering challenges encountered and resolved along the way.

## Full Report

See [`Pearls_AQI_Predictor_Report.pdf`](./Pearls_AQI_Predictor_Report.pdf) for the complete write-up: methodology, challenges and solutions, evaluation, limitations, and future work.

## Author

Wania Khan BS Computer Science, Jinnah University for Women, Karachi
