"""Train Hierarchical House Price Model (State -> City -> Area).

Includes:
- State, City, Area hierarchy
- 3-tier confidence ladder (High: >=20, Medium: 5-19, Low: <5 / city fallback)
- Conformal quantile regression (10th and 90th percentiles)
- Complete metadata bundle with state, city, area benchmarks and hierarchies
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

ROOT_DIR = Path(__file__).resolve().parent

RANGE_LOW = 0.10
RANGE_HIGH = 0.90

CAT_FEATURES = ["state", "location", "area", "Furnishing", "Transaction"]
NUM_FEATURES = ["bhk", "bathrooms", "log_area", "floor_no", "total_floors"]


def build_pipeline(model):
    """Builds preprocessor with min_frequency=5 to handle long-tail areas cleanly."""
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUM_FEATURES),
        ("cat", cat_transformer, CAT_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("reg", model),
    ])


def main():
    print("Loading clean hierarchy dataset...")
    df = pd.read_csv(ROOT_DIR / "clean.csv")
    if "log_area" not in df.columns:
        df["log_area"] = np.log(df["area_sqft"])
    
    # Ensure state is present
    if "state" not in df.columns:
        with open(ROOT_DIR / "data" / "city_state_mapping.json") as f:
            c2s = json.load(f)
        df["state"] = df["location"].map(c2s).fillna("Other")

    X = df[CAT_FEATURES + NUM_FEATURES]
    y_log = df["log_price"].values
    y_rupees = df["price"].values

    # Pre-compute area listing counts per (city, area)
    area_counts = df.groupby(["location", "area"]).size().to_dict()
    # Format keys as "city::area"
    area_counts_json = {f"{c}::{a}": int(cnt) for (c, a), cnt in area_counts.items()}

    # 70% train / 15% calibration / 15% test
    X_tr, X_rest, y_tr, y_rest, rup_tr, rup_rest = train_test_split(
        X, y_log, y_rupees, test_size=0.30, random_state=42
    )
    X_cal, X_te, y_cal, y_te, rup_cal, rup_te = train_test_split(
        X_rest, y_rest, rup_rest, test_size=0.50, random_state=42
    )

    print("\nTraining LightGBM Point Estimator...")
    point_model = LGBMRegressor(
        n_estimators=450,
        learning_rate=0.06,
        num_leaves=35,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    point_pipe = build_pipeline(point_model).fit(X_tr, y_tr)

    print("Training Quantile Regressors (10th and 90th percentiles)...")
    low_model = LGBMRegressor(
        objective="quantile",
        alpha=RANGE_LOW,
        n_estimators=350,
        learning_rate=0.06,
        num_leaves=35,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    low_pipe = build_pipeline(low_model).fit(X_tr, y_tr)

    high_model = LGBMRegressor(
        objective="quantile",
        alpha=RANGE_HIGH,
        n_estimators=350,
        learning_rate=0.06,
        num_leaves=35,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    high_pipe = build_pipeline(high_model).fit(X_tr, y_tr)

    # Conformal Calibration Offset on calibration set
    cal_low = low_pipe.predict(X_cal)
    cal_high = high_pipe.predict(X_cal)
    scores = np.maximum(cal_low - y_cal, y_cal - cal_high)
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (RANGE_HIGH - RANGE_LOW)) / n)
    offset = float(np.quantile(scores, level))

    # Evaluate on unseen test set
    pred_log = point_pipe.predict(X_te)
    pred_rup = np.exp(pred_log)
    ape = np.abs(pred_rup - rup_te) / rup_te

    te_low = low_pipe.predict(X_te) - offset
    te_high = high_pipe.predict(X_te) + offset
    coverage = float(((y_te >= te_low) & (y_te <= te_high)).mean())

    metrics = {
        "model": "lgbm_hierarchical",
        "test_rows": int(len(y_te)),
        "MAE_lakh": round(mean_absolute_error(rup_te, pred_rup) / 1e5, 2),
        "MAPE_pct": round(float(ape.mean() * 100), 2),
        "median_APE_pct": round(float(np.median(ape) * 100), 2),
        "R2_rupees": round(r2_score(rup_te, pred_rup), 4),
        "R2_log": round(r2_score(y_te, pred_log), 4),
        "range_coverage_pct": round(coverage * 100, 1),
        "range_offset_log": round(offset, 4),
    }

    print("\nHeld-out Test Results with Hierarchy:")
    for k, v in metrics.items():
        print(f"  {k:<22}{v}")

    # Refit final models on 100% of data for serving
    print("\nRefitting champion models on 100% of dataset...")
    point_final = build_pipeline(point_model).fit(X, y_log)
    low_final = build_pipeline(low_model).fit(X, y_log)
    high_final = build_pipeline(high_model).fit(X, y_log)

    # Load pre-built benchmarks and metadata
    with open(ROOT_DIR / "models" / "hierarchy_metadata.json", "r", encoding="utf-8") as f:
        hierarchy_metadata = json.load(f)
    with open(ROOT_DIR / "models" / "hierarchy_benchmarks.json", "r", encoding="utf-8") as f:
        hierarchy_benchmarks = json.load(f)
    with open(ROOT_DIR / "data" / "city_state_mapping.json", "r", encoding="utf-8") as f:
        city_to_state = json.load(f)

    # State to Cities mapping
    state_to_cities = {}
    for city, st in city_to_state.items():
        state_to_cities.setdefault(st, []).append(city)
    for st in state_to_cities:
        state_to_cities[st] = sorted(state_to_cities[st])

    bundle = {
        "point": point_final,
        "low": low_final,
        "high": high_final,
        "offset": offset,
        "meta": {
            "features": CAT_FEATURES + NUM_FEATURES,
            "states": sorted(list(state_to_cities.keys())),
            "cities": sorted(df["location"].unique().tolist()),
            "city_to_state": city_to_state,
            "state_to_cities": state_to_cities,
            "furnishing": sorted(df["Furnishing"].dropna().unique().tolist()),
            "transaction": sorted(df["Transaction"].dropna().unique().tolist()),
            "area_limits_sqft": [float(df["area_sqft"].min()), float(df["area_sqft"].max())],
            "bhk_limits": [int(df["bhk"].min()), int(df["bhk"].max())],
            "area_counts": area_counts_json,
            "hierarchy_metadata": hierarchy_metadata,
            "hierarchy_benchmarks": hierarchy_benchmarks,
            "metrics": metrics,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    }

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "house_bundle.joblib")
    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSuccessfully dumped bundle to {models_dir / 'house_bundle.joblib'}")


if __name__ == "__main__":
    main()
