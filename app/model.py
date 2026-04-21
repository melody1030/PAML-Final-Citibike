"""
Feature engineering and a pure-NumPy Multiple Linear Regression model.

Implements training and prediction without scikit-learn, XGBoost, TensorFlow,
or PyTorch — per the project's constraint that models be built from scratch
using only NumPy and Pandas.

The model predicts hourly Citi Bike ride_count per neighborhood (NTA2020),
following the preprocessing pipeline from `offline_experiments.ipynb`:
    - cyclical encoding for hour and day-of-week (sin/cos pairs)
    - raw month and is_weekend
    - one-hot encoding for NTA2020 and BoroName
    - log1p transform on the target
    - chronological 70/15/15 split
    - standardization using training-set mean/std only
    - closed-form Normal Equation: W = pinv(X^T X) @ X^T y
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Columns that come out of the feature engineering step, in a stable order.
NUMERIC_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month", "is_weekend"]


def build_features(df: pd.DataFrame, nta_values: List[str], boro_values: List[str]) -> pd.DataFrame:
    """Build the feature matrix from raw dataframe columns.

    Cyclical encoding lets the model treat adjacent hours (e.g. 23 and 0) as
    close together even though their raw integer values are far apart. The
    NTA/Boro one-hot columns are created against fixed vocabularies so that
    train/val/test/inference rows all produce the same column layout.
    """
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2 * np.pi * df["hour"].astype(float) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * df["hour"].astype(float) / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * df["dow"].astype(float) / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * df["dow"].astype(float) / 7.0)
    out["month"] = df["month"].astype(float)
    out["is_weekend"] = df["is_weekend"].astype(int).astype(float)

    # One-hot encode BoroName against a fixed vocabulary.
    for boro in boro_values:
        out[f"boro_{boro}"] = (df["BoroName"].values == boro).astype(float)

    # One-hot encode NTA2020 against a fixed vocabulary.
    for nta in nta_values:
        out[f"nta_{nta}"] = (df["NTA2020"].values == nta).astype(float)

    return out


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score standardize. A tiny epsilon is added to std to avoid div-by-zero
    on constant columns (a known issue with sparse one-hot features in some splits).
    """
    return (X - mean) / (std + 1e-7)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


@dataclass
class LinearModel:
    """Pure-NumPy multiple linear regression trained with the Normal Equation.

    Training is done in log1p-space; `predict` inverts the transform with expm1
    so callers always work in the original ride_count scale.
    """

    weights: np.ndarray            # (D+1,) including bias
    feature_columns: List[str]     # order of columns the model expects
    feature_mean: np.ndarray       # (D,) standardization means
    feature_std: np.ndarray        # (D,) standardization stds
    nta_vocabulary: List[str]
    boro_vocabulary: List[str]

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """Predict ride_count on the original scale from un-standardized features."""
        X_norm = standardize(X_raw, self.feature_mean, self.feature_std)
        X_b = add_bias(X_norm)
        y_log = X_b @ self.weights
        return np.expm1(y_log)

    def predict_from_row(self, feature_row: pd.DataFrame) -> float:
        """Convenience for single-row inference from a dataframe row matching
        the model's feature_columns order."""
        X = feature_row[self.feature_columns].to_numpy(dtype=float)
        return float(self.predict(X)[0])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Regression metrics implemented from scratch: RMSE, MAE, R-squared."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_linear_regression(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[LinearModel, Dict[str, Dict[str, float]]]:
    """Fit the linear model end-to-end and return (model, split_metrics).

    Steps mirror `multiple_regression_demand_prediction.ipynb`:
        1. Build features with fixed NTA/Boro vocabularies
        2. Log1p the target
        3. Chronological split on unique hour_slot values
        4. Standardize with train-set statistics only
        5. Solve the Normal Equation in closed form
        6. Report RMSE / MAE / R² on train, val, test (original scale)
    """
    df = df.copy()
    df["hour_slot"] = pd.to_datetime(df["hour_slot"])

    nta_vocabulary = sorted(df["NTA2020"].dropna().unique().tolist())
    boro_vocabulary = sorted(df["BoroName"].dropna().unique().tolist())

    X_df = build_features(df, nta_vocabulary, boro_vocabulary)
    feature_columns = X_df.columns.tolist()

    y_raw = df["ride_count"].astype(float).to_numpy()
    y_log = np.log1p(y_raw)

    # Chronological split on unique timestamps.
    ordered_slots = np.sort(df["hour_slot"].unique())
    train_end = ordered_slots[int(train_frac * len(ordered_slots))]
    val_end = ordered_slots[int((train_frac + val_frac) * len(ordered_slots))]

    train_mask = (df["hour_slot"] < train_end).to_numpy()
    val_mask = ((df["hour_slot"] >= train_end) & (df["hour_slot"] < val_end)).to_numpy()
    test_mask = (df["hour_slot"] >= val_end).to_numpy()

    X_all = X_df.to_numpy(dtype=float)
    X_train, X_val, X_test = X_all[train_mask], X_all[val_mask], X_all[test_mask]
    y_train_log = y_log[train_mask]
    y_train_raw = y_raw[train_mask]
    y_val_raw = y_raw[val_mask]
    y_test_raw = y_raw[test_mask]

    # Train-set standardization statistics only — no leakage.
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)

    X_train_norm = standardize(X_train, feature_mean, feature_std)
    X_train_b = add_bias(X_train_norm)

    # Closed-form Normal Equation with pseudoinverse for numerical stability.
    weights = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train_log

    model = LinearModel(
        weights=weights,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        nta_vocabulary=nta_vocabulary,
        boro_vocabulary=boro_vocabulary,
    )

    metrics = {
        "train": evaluate(y_train_raw, model.predict(X_train)),
        "val": evaluate(y_val_raw, model.predict(X_val)),
        "test": evaluate(y_test_raw, model.predict(X_test)),
    }

    return model, metrics
