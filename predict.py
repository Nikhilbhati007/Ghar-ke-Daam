"""Loads models/house_bundle.joblib and predicts with hierarchical confidence ladder.

Usage:
    from predict import predict, METADATA
    predict({
        "state": "Maharashtra",
        "location": "thane",
        "area": "kolshet",
        "area_sqft": 850,
        "bhk": 2,
        "bathrooms": 2,
        "Furnishing": "Semi-Furnished",
        "Transaction": "Resale"
    })
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BUNDLE = joblib.load(Path(__file__).parent / "models" / "house_bundle.joblib")
METADATA = BUNDLE["meta"]

# Backward compatibility alias
if "locations" not in METADATA and "cities" in METADATA:
    METADATA["locations"] = METADATA["cities"]

AREA_MIN, AREA_MAX = METADATA["area_limits_sqft"]


class InputError(ValueError):
    """Raised with a message that is safe to show to the user."""


def _validate(row):
    city = row.get("location")
    if not city or city not in METADATA["cities"]:
        raise InputError("Unknown city.")

    # Validate state if passed
    state = row.get("state")
    if state:
        expected_state = METADATA.get("city_to_state", {}).get(city)
        if expected_state and state.lower() != expected_state.lower():
            raise InputError(f"City '{city.title()}' is not in State '{state}'. It is in '{expected_state}'.")

    if row.get("Furnishing") not in METADATA["furnishing"]:
        raise InputError("Unknown furnishing option.")
    if row.get("Transaction") not in METADATA["transaction"]:
        raise InputError("Unknown transaction type.")
    try:
        area_sqft = float(row["area_sqft"])
    except (KeyError, TypeError, ValueError):
        raise InputError("Enter a valid carpet area.")
    if not AREA_MIN <= area_sqft <= AREA_MAX:
        raise InputError(f"Area must be between {AREA_MIN:.0f} and {AREA_MAX:.0f} sq ft.")


def predict(row):
    _validate(row)
    city = row["location"]
    state = row.get("state") or METADATA.get("city_to_state", {}).get(city, "Other")
    
    # Area fallback ladder
    area_input = row.get("area")
    if area_input:
        area_clean = str(area_input).strip().lower()
    else:
        area_clean = "other"

    area_key = f"{city}::{area_clean}"
    count = METADATA.get("area_counts", {}).get(area_key, 0)

    if area_clean == "other" or count < 5:
        confidence = "Low"
        confidence_label = "City-level estimate (area unlisted or < 5 listings)"
        area_used = "other"
        sample_count = count
    elif count < 20:
        confidence = "Medium"
        confidence_label = f"Blended area estimate based on {count} listings"
        area_used = area_clean
        sample_count = count
    else:
        confidence = "High"
        confidence_label = f"High-confidence area estimate based on {count} listings"
        area_used = area_clean
        sample_count = count

    data = {
        "state": state,
        "location": city,
        "area": area_used,
        "Furnishing": row["Furnishing"],
        "Transaction": row["Transaction"],
        "bhk": row.get("bhk", np.nan),
        "bathrooms": row.get("bathrooms", np.nan),
        "log_area": np.log(float(row["area_sqft"])),
        "floor_no": row.get("floor_no", np.nan),
        "total_floors": row.get("total_floors", np.nan),
    }

    X = pd.DataFrame([data], columns=METADATA["features"])
    mid = float(np.exp(BUNDLE["point"].predict(X)[0]))
    lo = float(np.exp(BUNDLE["low"].predict(X)[0] - BUNDLE["offset"]))
    hi = float(np.exp(BUNDLE["high"].predict(X)[0] + BUNDLE["offset"]))

    # Keep estimate within interval
    lo, hi = min(lo, mid), max(hi, mid)

    return {
        "price": mid,
        "low": lo,
        "high": hi,
        "per_sqft": mid / float(row["area_sqft"]),
        "confidence": confidence,
        "confidence_label": confidence_label,
        "sample_count": sample_count,
        "area_used": area_used,
        "state": state,
        "city": city,
    }
