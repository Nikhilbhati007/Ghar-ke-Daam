"""Phase 3 - One pipeline, one file.

Usage:
    python phase3_train.py clean.csv --model hgb
    (--model choices: ridge | rf | hgb | lgbm   -> use the winner from Phase 2)

Output: models/house_bundle.joblib containing
    point  : pipeline predicting log(price)
    low    : pipeline for the 10th percentile (range floor)
    high   : pipeline for the 90th percentile (range ceiling)
    offset : calibration widening, in log units, so the band really covers ~80%
    meta   : categories, limits, metrics, library versions, training date

Preprocessing lives INSIDE each pipeline, so the website can never feed the
model mismatched columns again.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from phase2_modeling import CAT, NUM, load, preprocessor

RANGE_LOW, RANGE_HIGH = 0.10, 0.90


def make_point_model(kind):
    if kind == "ridge":
        return Pipeline([("prep", preprocessor(True)), ("m", Ridge(alpha=1.0))])
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return Pipeline([("prep", preprocessor(False)),
                         ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42))])
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return Pipeline([("prep", preprocessor(False)),
                         ("m", LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42))])
    return Pipeline([("prep", preprocessor(False)),
                     ("m", HistGradientBoostingRegressor(learning_rate=0.06, max_iter=500, random_state=42))])


def make_quantile_model(alpha):
    return Pipeline([("prep", preprocessor(False)),
                     ("m", HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=300, random_state=42))])


def main(path, kind, out_dir):
    df = load(path)
    X, y_log, y = df[CAT + NUM], df["log_price"], df["price"]

    # 70% train / 15% calibration / 15% test
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y_log, test_size=0.30, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.50, random_state=42)

    point = make_point_model(kind).fit(X_tr, y_tr)
    low = make_quantile_model(RANGE_LOW).fit(X_tr, y_tr)
    high = make_quantile_model(RANGE_HIGH).fit(X_tr, y_tr)

    # Conformal calibration: how far outside the band do calibration points fall?
    scores = np.maximum(low.predict(X_cal) - y_cal, y_cal - high.predict(X_cal))
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (RANGE_HIGH - RANGE_LOW)) / n)
    offset = float(np.quantile(scores, level))

    # Honest evaluation on data the models never saw
    pred = np.exp(point.predict(X_te))
    true = np.exp(y_te)
    ape = np.abs(pred - true) / true
    lo_te, hi_te = low.predict(X_te) - offset, high.predict(X_te) + offset
    coverage = float(((y_te >= lo_te) & (y_te <= hi_te)).mean())
    metrics = {
        "model": kind,
        "test_rows": int(len(y_te)),
        "MAE_lakh": round(mean_absolute_error(true, pred) / 1e5, 2),
        "MAPE_pct": round(float(ape.mean() * 100), 2),
        "median_APE_pct": round(float(np.median(ape) * 100), 2),
        "R2_rupees": round(r2_score(true, pred), 4),
        "R2_log": round(r2_score(y_te, np.log(pred)), 4),
        "range_coverage_pct": round(coverage * 100, 1),
        "range_offset_log": round(offset, 4),
    }
    print("Held-out test results:")
    for k, v in metrics.items():
        print(f"  {k:<20}{v}")

    # Refit all three models on 100% of the data for the final bundle
    point = make_point_model(kind).fit(X, y_log)
    low = make_quantile_model(RANGE_LOW).fit(X, y_log)
    high = make_quantile_model(RANGE_HIGH).fit(X, y_log)

    meta = {
        "features": CAT + NUM,
        "locations": sorted(df["location"].unique().tolist()),
        "furnishing": sorted(df["Furnishing"].dropna().unique().tolist()),
        "transaction": sorted(df["Transaction"].dropna().unique().tolist()),
        "area_limits_sqft": [float(df["area_sqft"].min()), float(df["area_sqft"].max())],
        "bhk_limits": [int(df["bhk"].min()), int(df["bhk"].max())],
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    joblib.dump({"point": point, "low": low, "high": high, "offset": offset, "meta": meta}, out / "house_bundle.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out / 'house_bundle.joblib'} and metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="clean.csv")
    ap.add_argument("--model", choices=["ridge", "rf", "hgb", "lgbm"], default="hgb")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    main(a.csv, a.model, a.out)
