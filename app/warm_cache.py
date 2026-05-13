"""
One-shot script: train both ANN variants and pickle them to disk so the
deployed Streamlit app starts instantly (no training spinner on cold start).

Run once before deploying (or whenever the model code changes):

    cd PAML-Final-Citibike
    python app/warm_cache.py

This produces:
    app/cache/ann_no_weather.pkl
    app/cache/ann_weather.pkl

Commit both pickle files alongside the app code so they ship with the deployed
build. Delete `app/cache/` if you ever want the deployed app to retrain from
scratch on its next cold start.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ann_model import (  # noqa: E402  (path setup must come first)
    cache_path_for,
    load_weather,
    save_trained,
    train_ann,
)
from data_utils import load_hourly_data  # noqa: E402


def _fmt_metrics(metrics):
    return " | ".join(
        f"{split}: RMSE={m['RMSE']:.1f} MAE={m['MAE']:.1f} R²={m['R2']:.3f}"
        for split, m in metrics.items()
    )


def main() -> None:
    print("[1/3] Loading hourly ride data...")
    t0 = time.time()
    df = load_hourly_data()
    print(f"      {len(df):,} rows in {time.time() - t0:.1f}s")

    # No-weather ANN
    out_no_weather = cache_path_for("ann_no_weather")
    print("\n[2/3] Training ANN (no weather) ...")
    t0 = time.time()
    trained_nw = train_ann(df, include_weather=False)
    print(f"      done in {time.time() - t0:.1f}s")
    print(f"      {_fmt_metrics(trained_nw[1])}")
    save_trained(trained_nw, out_no_weather)
    size_mb = os.path.getsize(out_no_weather) / (1024 * 1024)
    print(f"      saved -> {out_no_weather} ({size_mb:.2f} MB)")

    # Weather ANN
    out_weather = cache_path_for("ann_weather")
    print("\n[3/3] Training ANN (+ weather) ...")
    weather_df = load_weather()
    if weather_df is None:
        print("      WARNING: weather CSV not found; skipping weather ANN cache.")
        return
    t0 = time.time()
    trained_w = train_ann(df, include_weather=True, weather_df=weather_df)
    print(f"      done in {time.time() - t0:.1f}s")
    print(f"      {_fmt_metrics(trained_w[1])}")
    save_trained(trained_w, out_weather)
    size_mb = os.path.getsize(out_weather) / (1024 * 1024)
    print(f"      saved -> {out_weather} ({size_mb:.2f} MB)")

    print("\nAll models cached. The deployed app will now load instantly.")


if __name__ == "__main__":
    main()
