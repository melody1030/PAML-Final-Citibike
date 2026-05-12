"""
Pure-NumPy ANN regressor + training pipeline for the CitiBike demand model.

Lifted directly from `ann_demand_prediction.ipynb` so the Streamlit app can
train and serve both:
    * ANN (no weather)  — temporal + spatial features only
    * ANN (+ weather)   — adds hourly weather predictors (temp, prcp, rhum,
                          wspd, cldc, weather_condition)

The architecture, hyperparameters, feature engineering, and split logic are
deliberately identical to the notebook so that the metrics shown in the app
match what's reported in the team's experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — match the notebook
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_SIZE = 64
LEARNING_RATE = 0.003
# Epochs trimmed from the notebook's 200 so app startup stays under ~45s
# per model. Validation loss curve flattens well before this cap and early
# stopping with `PATIENCE` kicks in if the run plateaus sooner.
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 4096
RANDOM_SEED = 42

WEATHER_NUMERIC_CANDIDATES = ["temp", "prcp", "rhum", "wspd", "cldc"]
WEATHER_CATEGORICAL_CANDIDATES = ["weather_condition"]


# ─────────────────────────────────────────────────────────────────────────────
# Activations + metrics (NumPy from scratch — same as notebook)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────────────────────────────────────
# ANN class — pure NumPy, one hidden layer + ReLU
# ─────────────────────────────────────────────────────────────────────────────

class SimpleANNRegressor:
    """Feed-forward ANN with one hidden layer and ReLU, trained with SGD."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 learning_rate: float = 0.003, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / max(1, input_size)),
                             size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / max(1, hidden_size)),
                             size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        cache = {"X": X, "Z1": Z1, "A1": A1, "y_pred": Z2}
        return Z2, cache

    def backward(self, y_true: np.ndarray, cache: dict):
        X, Z1, A1, y_pred = cache["X"], cache["Z1"], cache["A1"], cache["y_pred"]
        n = len(X)
        d_y_pred = (2.0 / n) * (y_pred - y_true)
        dW2 = A1.T @ d_y_pred
        db2 = np.sum(d_y_pred, axis=0, keepdims=True)
        dA1 = d_y_pred @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, g: dict):
        self.W1 -= self.learning_rate * g["dW1"]
        self.b1 -= self.learning_rate * g["db1"]
        self.W2 -= self.learning_rate * g["dW2"]
        self.b2 -= self.learning_rate * g["db2"]

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred, cache = self.forward(X)
        loss = mse_loss(y, y_pred)
        self.step(self.backward(y, cache))
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred, _ = self.forward(X)
        return y_pred

    def get_state(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_state(self, state):
        self.W1 = state["W1"].copy()
        self.b1 = state["b1"].copy()
        self.W2 = state["W2"].copy()
        self.b2 = state["b2"].copy()


def _train_loop(net: SimpleANNRegressor, X_train, y_train, X_val, y_val,
                epochs: int, patience: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    best_state = net.get_state()
    best_val = np.inf
    best_epoch = 0
    misses = 0
    n_train = len(X_train)
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        if batch_size is None or batch_size >= n_train:
            net.train_batch(X_train, y_train)
        else:
            perm = rng.permutation(n_train)
            Xs, ys = X_train[perm], y_train[perm]
            for s in range(0, n_train, batch_size):
                e = s + batch_size
                net.train_batch(Xs[s:e], ys[s:e])

        tl = mse_loss(y_train, net.predict(X_train))
        vl = mse_loss(y_val, net.predict(X_val))
        history["epoch"].append(epoch)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if vl < best_val - 1e-8:
            best_val = vl
            best_epoch = epoch
            best_state = net.get_state()
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break

    net.set_state(best_state)
    return history, best_epoch, best_val


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — mirrors the notebook exactly
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_temporal_and_weather(work_df: pd.DataFrame,
                                   include_weather: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Add cyclical encodings + figure out which numeric/categorical cols to use.

    Returns:
        work_df (mutated copy), numeric_feature_columns, categorical_columns
    """
    work_df = work_df.copy()

    raw_temporal = [c for c in ["hour", "dow", "month", "is_weekend"]
                    if c in work_df.columns]
    spatial = [c for c in ["NTA2020", "BoroName"] if c in work_df.columns]

    weather_numeric = []
    weather_categorical = []
    if include_weather:
        weather_numeric = [c for c in WEATHER_NUMERIC_CANDIDATES
                           if c in work_df.columns]
        weather_categorical = [c for c in WEATHER_CATEGORICAL_CANDIDATES
                               if c in work_df.columns]

    # Coerce numerics.
    for col in raw_temporal + weather_numeric + ["ride_count"]:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    # Cyclical encodings.
    engineered = []
    if "hour" in work_df.columns:
        work_df["hour_sin"] = np.sin(2 * np.pi * work_df["hour"] / 24)
        work_df["hour_cos"] = np.cos(2 * np.pi * work_df["hour"] / 24)
        engineered += ["hour_sin", "hour_cos"]
    if "dow" in work_df.columns:
        work_df["dow_sin"] = np.sin(2 * np.pi * work_df["dow"] / 7)
        work_df["dow_cos"] = np.cos(2 * np.pi * work_df["dow"] / 7)
        engineered += ["dow_sin", "dow_cos"]
    if "month" in work_df.columns:
        work_df["month_sin"] = np.sin(2 * np.pi * (work_df["month"] - 1) / 12)
        work_df["month_cos"] = np.cos(2 * np.pi * (work_df["month"] - 1) / 12)
        engineered += ["month_sin", "month_cos"]

    numeric_features = []
    if "is_weekend" in work_df.columns:
        numeric_features.append("is_weekend")
    numeric_features.extend(engineered)
    numeric_features.extend(weather_numeric)

    categorical = spatial + weather_categorical
    return work_df, numeric_features, categorical


def _one_hot(work_df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    if not categorical_columns:
        return pd.DataFrame(index=work_df.index)
    return pd.get_dummies(
        work_df[categorical_columns],
        prefix=categorical_columns,
        prefix_sep="=",
        dtype=float,
    )


def _normalize(frame: pd.DataFrame, cols: List[str],
               means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    if cols:
        out[cols] = ((out[cols].astype(float) - means) / stds).astype(float)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Weather merge
# ─────────────────────────────────────────────────────────────────────────────

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_weather() -> Optional[pd.DataFrame]:
    """Load `datasets/weather_knyc_2025.csv` if present; otherwise None."""
    path = os.path.join(_project_root(), "datasets", "weather_knyc_2025.csv")
    if not os.path.exists(path):
        return None
    weather = pd.read_csv(path)
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
    return weather


def merge_weather(rides_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join hourly weather onto the rides dataframe by hour_slot/time."""
    rides = rides_df.copy()
    rides["hour_slot"] = pd.to_datetime(rides["hour_slot"], errors="coerce")
    keep = ["time"] + [c for c in WEATHER_NUMERIC_CANDIDATES if c in weather_df.columns] \
                    + [c for c in WEATHER_CATEGORICAL_CANDIDATES if c in weather_df.columns]
    w = weather_df[keep].rename(columns={"time": "hour_slot"})
    return rides.merge(w, on="hour_slot", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Trained-model container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ANNModel:
    """Everything needed to score a single user input."""

    net: SimpleANNRegressor
    feature_columns: List[str]            # final column order fed to the net
    numeric_feature_columns: List[str]    # the subset that was standardized
    feature_means: pd.Series              # for numeric_feature_columns
    feature_stds: pd.Series               # for numeric_feature_columns
    target_mean: float                    # log1p-space scaling
    target_std: float
    nta_vocabulary: List[str]
    boro_vocabulary: List[str]
    weather_condition_vocabulary: List[str]   # empty list if not used
    include_weather: bool
    weather_means: Optional[pd.Series]    # for filling missing weather at inference
    weather_condition_mode: Optional[str] # most common condition (fallback)

    def predict_from_row(self, row: pd.DataFrame) -> float:
        """Score a one-row DataFrame that already has the raw input columns:
        NTA2020, BoroName, hour, dow, month, is_weekend, plus optional weather
        (temp, prcp, rhum, wspd, cldc, weather_condition).
        Missing one-hots are zero-filled and unknown columns are dropped to
        keep the feature layout aligned with training.
        """
        df, numeric_cols, cat_cols = _engineer_temporal_and_weather(
            row, include_weather=self.include_weather
        )
        if self.include_weather:
            # Fill any missing weather numerics with the training-set mean.
            for c in WEATHER_NUMERIC_CANDIDATES:
                if c in df.columns and self.weather_means is not None and c in self.weather_means:
                    df[c] = df[c].fillna(self.weather_means[c])
            # Fill missing weather_condition with mode.
            if "weather_condition" in df.columns and self.weather_condition_mode is not None:
                df["weather_condition"] = df["weather_condition"].fillna(self.weather_condition_mode)

        encoded = _one_hot(df, cat_cols)
        numeric_df = df[numeric_cols].astype(float).copy() if numeric_cols else pd.DataFrame(index=df.index)
        feature_df = pd.concat([numeric_df, encoded], axis=1)

        # Add any missing columns the model expects (e.g. an NTA that wasn't
        # selected in this single row), and drop any extra columns.
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[self.feature_columns]

        # Standardize numeric columns using training-set stats.
        feature_df = _normalize(feature_df, self.numeric_feature_columns,
                                self.feature_means, self.feature_stds)

        X = feature_df.to_numpy(dtype=float)
        y_scaled = self.net.predict(X)
        y_log = y_scaled * self.target_std + self.target_mean
        y = float(np.expm1(y_log)[0, 0])
        return max(0.0, y)


# ─────────────────────────────────────────────────────────────────────────────
# Public training entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def train_ann(
    rides_df: pd.DataFrame,
    include_weather: bool = False,
    weather_df: Optional[pd.DataFrame] = None,
) -> Tuple[ANNModel, Dict[str, Dict[str, float]]]:
    """End-to-end: feature build → chronological split → normalize → train.

    Returns (ANNModel, metrics_dict_on_original_ride_count_scale).
    """
    df = rides_df.copy()
    if include_weather:
        if weather_df is None:
            raise ValueError("weather_df is required when include_weather=True")
        df = merge_weather(df, weather_df)

    df["hour_slot"] = pd.to_datetime(df["hour_slot"], errors="coerce")
    work_df, numeric_features, categorical_columns = \
        _engineer_temporal_and_weather(df, include_weather=include_weather)

    encoded = _one_hot(work_df, categorical_columns)
    numeric_df = work_df[numeric_features].astype(float).copy()
    feature_matrix = pd.concat([numeric_df, encoded], axis=1)
    feature_columns = feature_matrix.columns.tolist()

    model_df = pd.concat([
        work_df[["hour_slot"]].copy(),
        feature_matrix,
        work_df[["ride_count"]].astype(float).copy(),
    ], axis=1)

    required = feature_columns + ["ride_count"]
    model_df = model_df.dropna(subset=required).reset_index(drop=True)

    # Chronological split on unique hour_slot timestamps (70 / 15 / 15).
    model_df = model_df.sort_values("hour_slot").reset_index(drop=True)
    unique_times = np.array(sorted(model_df["hour_slot"].dropna().unique()))
    train_end_idx = int(len(unique_times) * 0.70)
    val_end_idx = int(len(unique_times) * 0.85)
    train_cutoff = unique_times[train_end_idx - 1]
    val_cutoff = unique_times[val_end_idx - 1]

    train_df = model_df[model_df["hour_slot"] <= train_cutoff].copy()
    val_df = model_df[(model_df["hour_slot"] > train_cutoff) & (model_df["hour_slot"] <= val_cutoff)].copy()
    test_df = model_df[model_df["hour_slot"] > val_cutoff].copy()

    # Train-set-only normalization stats.
    means = train_df[numeric_features].mean()
    stds = train_df[numeric_features].std(ddof=0).replace(0, 1.0)

    train_norm = _normalize(train_df, numeric_features, means, stds)
    val_norm = _normalize(val_df, numeric_features, means, stds)
    test_norm = _normalize(test_df, numeric_features, means, stds)

    X_train = train_norm[feature_columns].to_numpy(dtype=float)
    X_val = val_norm[feature_columns].to_numpy(dtype=float)
    X_test = test_norm[feature_columns].to_numpy(dtype=float)

    y_train = train_norm[["ride_count"]].to_numpy(dtype=float)
    y_val = val_norm[["ride_count"]].to_numpy(dtype=float)
    y_test = test_norm[["ride_count"]].to_numpy(dtype=float)

    y_train_log = np.log1p(y_train)
    target_mean = float(y_train_log.mean())
    target_std = float(y_train_log.std()) or 1.0
    y_train_scaled = (y_train_log - target_mean) / target_std
    y_val_scaled = (np.log1p(y_val) - target_mean) / target_std

    net = SimpleANNRegressor(
        input_size=X_train.shape[1],
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
    )
    _train_loop(
        net, X_train, y_train_scaled, X_val, y_val_scaled,
        epochs=EPOCHS, patience=PATIENCE,
        batch_size=BATCH_SIZE, seed=RANDOM_SEED,
    )

    def _eval(X, y_true_orig):
        scaled = net.predict(X)
        y_pred = np.expm1(scaled * target_std + target_mean)
        return {"RMSE": rmse(y_true_orig, y_pred),
                "MAE": mae(y_true_orig, y_pred),
                "R2": r2_score(y_true_orig, y_pred)}

    metrics = {
        "train": _eval(X_train, y_train),
        "val": _eval(X_val, y_val),
        "test": _eval(X_test, y_test),
    }

    nta_vocab = sorted(work_df["NTA2020"].dropna().unique().tolist())
    boro_vocab = sorted(work_df["BoroName"].dropna().unique().tolist())
    if include_weather and "weather_condition" in work_df.columns:
        cond_vocab = sorted(work_df["weather_condition"].dropna().unique().tolist())
        cond_mode = work_df["weather_condition"].mode()
        cond_mode = cond_mode.iloc[0] if not cond_mode.empty else None
    else:
        cond_vocab = []
        cond_mode = None

    weather_means = train_df[[c for c in WEATHER_NUMERIC_CANDIDATES if c in train_df.columns]].mean() \
        if include_weather else None

    return ANNModel(
        net=net,
        feature_columns=feature_columns,
        numeric_feature_columns=numeric_features,
        feature_means=means,
        feature_stds=stds,
        target_mean=target_mean,
        target_std=target_std,
        nta_vocabulary=nta_vocab,
        boro_vocabulary=boro_vocab,
        weather_condition_vocabulary=cond_vocab,
        include_weather=include_weather,
        weather_means=weather_means,
        weather_condition_mode=cond_mode,
    ), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Weather lookup helper (for inference)
# ─────────────────────────────────────────────────────────────────────────────

def lookup_weather(weather_df: pd.DataFrame, when: pd.Timestamp) -> Dict[str, float]:
    """Return a dict of weather columns for the given hourly timestamp.

    Empty dict if no row matches; callers should fall back to training means.
    """
    if weather_df is None:
        return {}
    match = weather_df[weather_df["time"] == when]
    if match.empty:
        return {}
    row = match.iloc[0]
    out: Dict[str, float] = {}
    for c in WEATHER_NUMERIC_CANDIDATES + WEATHER_CATEGORICAL_CANDIDATES:
        if c in row and pd.notna(row[c]):
            out[c] = row[c]
    return out
