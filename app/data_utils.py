"""
Data loading and lookup helpers for the Streamlit app.

Keeps CSV loading and dropdown-option helpers in one place so the Streamlit
layer can stay focused on UI.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


# Resolve the dataset path relative to the project root so the app works
# both when launched from the repo root and from inside the app/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "hourly_neighborhood_2025.csv")


def load_hourly_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the hourly neighborhood dataset and parse the datetime column.

    The CSV is ~60MB and contains ~1M rows of hourly ride counts per NTA.
    The caller is expected to cache this.
    """
    df = pd.read_csv(path)
    df["hour_slot"] = pd.to_datetime(df["hour_slot"])
    return df


def borough_to_nta_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build {borough -> [(nta_code, nta_name), ...]} for cascading dropdowns."""
    groups: Dict[str, List[str]] = {}
    for boro, sub in df.groupby("BoroName"):
        # Pair each NTA code with a human-readable name, de-duplicated and sorted.
        pairs = sub[["NTA2020", "NTAName"]].drop_duplicates().sort_values("NTAName")
        groups[boro] = list(pairs.itertuples(index=False, name=None))
    return groups


def historical_hourly_mean(
    df: pd.DataFrame, nta_code: str, month: int
) -> pd.Series:
    """Average ride_count by hour-of-day for a given NTA & month.

    Used to draw the historical baseline curve alongside the model prediction.
    """
    subset = df[(df["NTA2020"] == nta_code) & (df["month"] == month)]
    if subset.empty:
        # Fall back to any month's data for that NTA so the chart still renders.
        subset = df[df["NTA2020"] == nta_code]
    return subset.groupby("hour")["ride_count"].mean()
