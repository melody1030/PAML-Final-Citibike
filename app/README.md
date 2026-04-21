# CitiBike Demand Predictor — Streamlit App

A web interface for the Citi Bike NYC hourly demand-prediction model.
Trains a pure-NumPy multiple linear regression on the 2025 hourly neighborhood
dataset and serves predictions through a point-and-click sidebar.

## How to run

From the **project root** (the folder containing `datasets/` and `app/`):

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Streamlit will open `http://localhost:8501` in your browser.

First launch takes ~30–60 seconds while the app loads the ~60 MB CSV and fits
the linear regression (both are cached across reruns within the session).

## What the app does

- **Sidebar inputs:** borough, neighborhood (NTA2020), date, hour of day.
- **Main panel:** predicted hourly ride count, historical average for the same
  hour/month, delta vs. average, hourly demand curve, and the model's
  held-out test metrics (RMSE / MAE / R²).

## Project structure

```
app/
├── app.py          # Streamlit entry point (UI, layout, caching)
├── model.py        # Feature engineering + pure-NumPy LinearModel
├── data_utils.py   # CSV loader and dropdown helpers
├── requirements.txt
└── README.md
```

## Data & model

- Data: `datasets/hourly_neighborhood_2025.csv` (hourly ride counts per NTA,
  2025).
- Features: cyclical encoding for `hour` and `dow`, one-hot for `NTA2020` and
  `BoroName`, plus raw `month` and `is_weekend`.
- Target: `ride_count`, `log1p`-transformed; predictions are inverted with
  `expm1`.
- Training: chronological 70/15/15 split, train-set-only standardization,
  Normal Equation solved with `numpy.linalg.pinv`.
- No scikit-learn, XGBoost, TensorFlow, or PyTorch — per the project's
  from-scratch constraint.

## Next milestone

- Integrate the NumPy ANN from `ann_demand_prediction.ipynb` and expose a
  model toggle in the sidebar.
- Wire in weather features from `weather_knyc_2025.csv`.
- Add a side-by-side model comparison view.
