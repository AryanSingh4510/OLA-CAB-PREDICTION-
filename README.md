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

* **Geo features:** haversine distance, bearing, pickup/drop geohash, zone popularity.
* **Time features:** hour, dayofweek, month, is\_weekend, festive/holiday flags.
* **Demand–Supply proxies:** rolling pickup density in zone, active drivers per zone (if available).
* **Traffic/weather:** joined via timestamp & zone (optional).
* **Cab category:** one-hot or target-encoder.
* **Interactions:** distance×hour, distance×cab\_type, traffic×hour.
* **Scaling:** Robust/StandardScaler for linear models; tree models handle raw.

---

## 🔧 Models

Start simple → grow:

* **Baselines:**

  * Mean/median fare
  * Linear Regression / Ridge / Lasso
* **Non-linear:**

  * Random Forest, XGBoost/LightGBM, CatBoost
* **Neural (optional):**

  * MLP on tabular features

> Default recommended: **LightGBM** or **CatBoost** for strong tabular performance + fast training.

---

## 📊 Metrics

Primary: **RMSE, MAE, MAPE** on a held-out **chronological** test split.

Report:

* Overall metrics (val/test)
* By **cab\_type**, **distance buckets**, **time-of-day**
* Error distribution plots & calibration

Target interview numbers (illustrative):

* MAE ≤ ₹25–₹45 for short trips
* MAPE ≤ 12–18% overall

---

## 🧪 Reproducible Training (CLI)

```bash
# 1) Create env
conda env create -f environment.yml
conda activate ola-ml

# 2) Prepare data
python -m src.data_prep --raw data/raw/trips.csv --out data/processed/ --test-size 0.2 --time-col requested_timestamp

# 3) Feature engineering
python -m src.features --in data/processed/train.parquet --out data/processed/features/

# 4) Train (LightGBM example)
python -m src.models train \
  --algo lightgbm \
  --train data/processed/features/train.parquet \
  --valid data/processed/features/valid.parquet \
  --model-out experiments/results/model_lgbm.pkl \
  --mlflow-uri experiments/mlflow

# 5) Evaluate
python -m src.evaluate \
  --model experiments/results/model_lgbm.pkl \
  --test data/processed/features/test.parquet \
  --report experiments/results/report.json \
  --plots experiments/results/plots/
```

**Key CLI Flags:** `--seed`, `--n-estimators`, `--learning-rate`, `--max-depth`, `--l2`, `--early-stopping`.

---

## 🧭 Notebooks

* `01_eda.ipynb` – sanity checks, distributions, leakage scan
* `02_feature_engineering.ipynb` – build & validate features
* `03_modeling.ipynb` – experiments & comparisons

> Keep notebooks deterministic by fixing seeds and snapshotting processed data.

---

## 🧰 FastAPI Inference Service

**Run locally:**

```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload
```

**Sample Request:**

```json
POST /predict
{
  "pickup_lat": 28.6139,
  "pickup_lng": 77.2090,
  "drop_lat": 28.4595,
  "drop_lng": 77.0266,
  "requested_timestamp": "2025-08-29T09:30:00+05:30",
  "estimated_distance_km": 15.2,
  "estimated_duration_min": 38,
  "cab_type": "Prime Sedan"
}
```

**Sample Response:**

```json
{
  "predicted_fare_inr": 342.75,
  "model": "lightgbm_v1",
  "features_used": ["haversine_km", "hour", "dayofweek", "cab_type", "zone_popularity", "interactions"]
}
```

---

## 🐳 Docker (Optional)

```bash
# Build
docker build -t ola-fare:latest .
# Run
docker run -p 8000:8000 ola-fare:latest
```

---

## 📈 Experiment Tracking (Optional)

* Enable **MLflow** by setting `--mlflow-uri` and auto-logging for LightGBM/XGBoost.
* Log params, metrics, artifacts (plots), and the model.
* Use tags like `algo=lgbm`, `features=v3`, `split=chronological`.

---

## 🔍 Error Analysis

* Slice by `distance` (0–3, 3–8, 8–20, 20+ km)
* Slice by `hour` (rush vs off-peak)
* **Shapley** plots for feature attribution (optional)
* Investigate high-MAPE tails; consider capping surge or adding traffic proxies

---

## 🧪 Ablations (Examples)

* With vs without **geohash**
* Add **weather** & **holiday** flags
* Distance only vs distance+time vs distance+time+category

Document what moves metrics the most.

---

## ⚠️ Assumptions & Limitations

* Training data represents the target city (distribution shift hurts).
* True surge, tolls, coupons, and platform fees may be unavailable; we approximate via time/zone demand.
* ETA and realized duration may differ from pre-ride estimates.

---

## 🧯 Monitoring (Prod)

* Track **data drift** (feature means, PSI/KS)
* Shadow deploy new model; compare **MAE/MAPE** online
* Log request feature ranges; alert on anomalies

---

## 📝 How to Reproduce

1. Place raw CSV/Parquet in `data/raw/`.
2. Run CLI steps in order (prep → features → train → evaluate).
3. Start API and hit `/predict`.

> Set `PYTHONHASHSEED=0` and use fixed `SEED` from `src/config.py`.

---

## 🛠️ Tech Stack

* Python 3.10+
* pandas, numpy, scikit-learn
* lightgbm / xgboost / catboost
* fastapi, pydantic, uvicorn
* mlflow (optional), shap (optional)

---

## 📜 License

MIT (replace if needed).

---

## 🤝 Contributing

PRs welcome! Open an issue for bugs or feature requests.

---

## 📚 References (generic)

* Tabular ML best practices (feature scaling, leakage checks)
* Gradient boosting for regression (LGBM/XGB)
* Geospatial features (haversine, geohash)


> If you use this for placements, rehearse a 2–3 min walkthrough (problem → features → model → metrics → demo).
