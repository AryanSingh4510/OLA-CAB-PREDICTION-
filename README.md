# OLA-CAB-PREDICTION-
# Ola Cab Fare Prediction (AI/ML)

> Production-ready, interview-friendly ML project to predict Ola cab fares from trip details. Includes data prep, model training, experiment tracking, evaluation, and an optional FastAPI microservice for deployment.

---

## 🚀 Overview

This repository builds a supervised regression model to estimate the **cab fare** before a ride is booked. It covers:

* Data ingestion & cleaning
* Feature engineering (time, location, surge proxies, driver supply/demand signals)
* Model training (baseline → advanced)
* Evaluation & error analysis
* Inference service (FastAPI) + Docker
* Reproducibility (conda, seeds) & experiment tracking (MLflow supported)

> **Why it’s interview-ready:** Clear structure, crisp metrics, ablation notes, and deployment story you can demo quickly.

---

## 🧭 Problem Statement

Given trip request details (pickup/drop coordinates or zones, timestamp, trip distance/duration estimates, cab category), predict the **fare** in INR. Optionally, estimate **ETA** and **surge multiplier** as extensions.

**Input (examples):**

* `pickup_lat, pickup_lng, drop_lat, drop_lng`
* `requested_timestamp`
* `estimated_distance_km`, `estimated_duration_min`
* `cab_type` (Micro, Mini, Prime, Auto, Bike, etc.)
* Optional: `rain`, `temp`, `holiday`, `event_density`, `traffic_level`

**Output:**

* `predicted_fare_inr` (float)

**Business Impact:**

* Better price transparency, reduced cancellations, improved matching & revenue forecasting.

---

## 📦 Project Structure

```
ola-cab-fare-prediction/
├─ data/
│  ├─ raw/                  # raw CSV/Parquet (ignored by git)
│  ├─ interim/              # after cleaning/standardization
│  └─ processed/            # modeling-ready sets (train/val/test)
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_engineering.ipynb
│  └─ 03_modeling.ipynb
├─ src/
│  ├─ config.py             # paths, constants, SEED
│  ├─ data_prep.py          # loaders, cleaners, splitters
│  ├─ features.py           # feature builders/transformers
│  ├─ models.py             # model zoo & training loops
│  ├─ evaluate.py           # metrics & error analysis
│  ├─ utils.py              # helpers (logging, timing)
│  └─ service/
│     ├─ app.py             # FastAPI service
│     └─ schemas.py         # pydantic I/O
├─ experiments/
│  ├─ mlflow/               # tracking (optional)
│  └─ results/              # metrics, plots
├─ Dockerfile
├─ environment.yml          # conda env
├─ requirements.txt         # pip alt
├─ README.md
└─ LICENSE
```

---

## 🗂️ Dataset

You can use any ride-fare dataset with time & geo fields. Typical columns:

| column                     | type     | description                             |
| -------------------------- | -------- | --------------------------------------- |
| `trip_id`                  | string   | unique id                               |
| `pickup_lat`, `pickup_lng` | float    | pickup coordinates                      |
| `drop_lat`, `drop_lng`     | float    | drop coordinates                        |
| `requested_timestamp`      | datetime | ride request time (local)               |
| `estimated_distance_km`    | float    | pre-ride estimate or haversine distance |
| `estimated_duration_min`   | float    | pre-ride estimate (optional)            |
| `cab_type`                 | category | e.g., Micro, Mini, Prime, Auto, Bike    |
| `fare_inr`                 | float    | **target**                              |

> If your data lacks `estimated_distance_km`, compute **haversine\_km** from lat/lng.

**Data Privacy:** Ensure coordinates are anonymized or binned into zones (geohash).

---

## 🧹 Data Cleaning

* Remove rows with impossible coordinates, zero/negative distance, or extreme outliers (IQR or z-score).
* Snap timestamps to local timezone; derive **hour-of-day, day-of-week, month, is\_weekend, holiday**.
* Optionally grid coordinates into **geohash** (5–7) for pickup/drop zones.
* Handle missing values (median/most-frequent or model-based imputer).

---

## 🏗️ Feature Engineering

* **Geo features:** havers
