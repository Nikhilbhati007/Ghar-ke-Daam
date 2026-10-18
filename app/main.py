"""Backend API for House Price Predictor (Ghar ka Daam) with Hierarchical Valuation.

Endpoints:
    GET  /               - Web interface with State -> City -> Area selectors
    GET  /health         - Operational health check
    GET  /metadata       - Hierarchical metadata (States -> Cities -> Areas)
    GET  /areas          - Searchable area endpoint (?city=...&q=...)
    POST /predict        - Valuate property with 3-tier confidence & multi-level benchmarks
    POST /compare-areas  - Compare same specs across neighbouring areas in the chosen city
    POST /compare        - Compare across major Indian metros
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, Field, ValidationError

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predict import predict as run_predict, METADATA, InputError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("house_price_api")

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "app" / "templates"),
    static_folder=str(ROOT_DIR / "app" / "static"),
)

# Load benchmarks
BENCHMARKS = METADATA.get("hierarchy_benchmarks", {})
STATE_BENCHMARKS = BENCHMARKS.get("state", {})
CITY_BENCHMARKS = BENCHMARKS.get("city", {})
AREA_BENCHMARKS = BENCHMARKS.get("area", {})
HIERARCHY_TREE = METADATA.get("hierarchy_metadata", {})


class PropertyInput(BaseModel):
    """Pydantic validation schema for incoming prediction requests."""
    state: Optional[str] = Field(default=None, description="State / Union Territory")
    location: str = Field(..., description="City or district name")
    area: Optional[str] = Field(default=None, description="Locality, sector, or neighborhood name")
    area_sqft: float = Field(..., gt=0, description="Carpet area in square feet")
    bhk: Optional[float] = Field(default=2.0, ge=1, le=10, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(default=2.0, ge=1, le=10, description="Number of bathrooms")
    Furnishing: str = Field(..., description="Furnishing status")
    Transaction: str = Field(..., description="Transaction type")
    floor_no: Optional[float] = Field(default=None, description="Floor number of the unit")
    total_floors: Optional[float] = Field(default=None, description="Total floors in building")


def format_inr(amount: float) -> str:
    """Formats amount in Indian Rupees (Lac/Cr)."""
    if amount >= 1e7:
        return f"{amount / 1e7:.2f} Cr"
    elif amount >= 1e5:
        return f"{amount / 1e5:.2f} Lac"
    return f"Rs {amount:,.0f}"


@app.route("/", methods=["GET"])
def index():
    """Serves the frontend template."""
    return render_template("index.html", metadata=METADATA)


@app.route("/health", methods=["GET"])
def health():
    """Operational health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": METADATA.get("metrics", {}).get("model", "lgbm_hierarchical"),
        "sklearn_version": METADATA.get("sklearn_version", "unknown"),
        "trained_at": METADATA.get("trained_at", "unknown"),
    }), 200


@app.route("/metadata", methods=["GET"])
def metadata():
    """Returns dynamic hierarchy (States -> Cities -> Areas) and validation constraints."""
    return jsonify({
        "states": METADATA.get("states", []),
        "cities": METADATA.get("cities", []),
        "locations": METADATA.get("cities", []),
        "state_to_cities": METADATA.get("state_to_cities", {}),
        "hierarchy": HIERARCHY_TREE,
        "furnishing": METADATA.get("furnishing", []),
        "transaction": METADATA.get("transaction", []),
        "area_limits_sqft": METADATA.get("area_limits_sqft", [100.0, 8000.0]),
        "bhk_limits": METADATA.get("bhk_limits", [1, 10]),
        "metrics": METADATA.get("metrics", {}),
    }), 200


@app.route("/areas", methods=["GET"])
def get_areas():
    """Searchable area query: /areas?city=thane&q=kol"""
    city = request.args.get("city", "").strip().lower()
    q = request.args.get("q", "").strip().lower()

    if not city or city not in METADATA.get("cities", []):
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    state = METADATA.get("city_to_state", {}).get(city, "")
    city_data = HIERARCHY_TREE.get(state, {}).get(city, {})
    areas = city_data.get("areas", [])

    if q:
        filtered = [a for a in areas if q in a["name"].lower() or q in a["id"].lower()]
    else:
        filtered = areas

    return jsonify({
        "city": city,
        "state": state,
        "count": len(filtered),
        "areas": filtered
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Valuates property with 3-tier confidence and multi-level benchmarks."""
    start_time = time.time()
    raw_json = request.get_json(silent=True)

    if not raw_json:
        return jsonify({"error": "Missing or invalid JSON payload."}), 400

    try:
        validated = PropertyInput.model_validate(raw_json)
    except ValidationError as e:
        first_err = e.errors()[0]
        field = ".".join(str(loc) for loc in first_err["loc"])
        msg = f"Validation error for field '{field}': {first_err['msg']}"
        return jsonify({"error": msg, "details": e.errors()}), 400

    data_dict = validated.model_dump()
    try:
        res = run_predict(data_dict)
    except InputError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f'"Prediction execution failed: {str(e)}"')
        return jsonify({"error": f"Internal prediction failure: {str(e)}"}), 500

    duration_ms = (time.time() - start_time) * 1000.0
    ppsf = res["per_sqft"]
    city = res["city"]
    state = res["state"]
    area_used = res["area_used"]

    # 3-Tier Multi-Level Benchmarking
    city_med = CITY_BENCHMARKS.get(city, {}).get("median")
    state_med = STATE_BENCHMARKS.get(state, {}).get("median")
    area_key = f"{city}::{area_used}"
    area_med = AREA_BENCHMARKS.get(area_key, {}).get("median")

    benchmarks = {
        "city": {
            "name": city.title(),
            "median_ppsf": city_med,
            "median_formatted": f"Rs {city_med:,.0f} / sq ft" if city_med else "N/A",
            "diff_pct": round(((ppsf - city_med) / city_med) * 100.0, 1) if city_med else 0.0,
        },
        "state": {
            "name": state,
            "median_ppsf": state_med,
            "median_formatted": f"Rs {state_med:,.0f} / sq ft" if state_med else "N/A",
            "diff_pct": round(((ppsf - state_med) / state_med) * 100.0, 1) if state_med else 0.0,
        },
    }

    if area_used != "other" and area_med:
        benchmarks["area"] = {
            "name": area_used.title(),
            "median_ppsf": area_med,
            "median_formatted": f"Rs {area_med:,.0f} / sq ft",
            "diff_pct": round(((ppsf - area_med) / area_med) * 100.0, 1),
        }

    response_payload = {
        "price": round(res["price"], 2),
        "low": round(res["low"], 2),
        "high": round(res["high"], 2),
        "per_sqft": round(ppsf, 2),
        "confidence": res["confidence"],
        "confidence_label": res["confidence_label"],
        "sample_count": res["sample_count"],
        "area_used": res["area_used"].title(),
        "state": state,
        "city": city.title(),
        "formatted": {
            "price": format_inr(res["price"]),
            "low": format_inr(res["low"]),
            "high": format_inr(res["high"]),
            "per_sqft": f"Rs {ppsf:,.2f} / sq ft",
            "range": f"{format_inr(res['low'])} - {format_inr(res['high'])}",
        },
        "benchmarks": benchmarks,
        "latency_ms": round(duration_ms, 2),
    }

    logger.info(
        f'"Valuation complete for {state} -> {city} -> {area_used} | '
        f'Price={response_payload["formatted"]["price"]} | Conf={res["confidence"]} | Latency={duration_ms:.2f}ms"'
    )

    return jsonify(response_payload), 200


@app.route("/compare-areas", methods=["POST"])
def compare_areas():
    """Compares current property specifications across top areas in the same city."""
    raw_json = request.get_json(silent=True)
    if not raw_json:
        return jsonify({"error": "Missing payload"}), 400

    try:
        validated = PropertyInput.model_validate(raw_json)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    city = validated.location
    state = validated.state or METADATA.get("city_to_state", {}).get(city, "")
    city_data = HIERARCHY_TREE.get(state, {}).get(city, {})
    areas = city_data.get("areas", [])

    if not areas:
        return jsonify({"areas": []}), 200

    # Pick top 6 areas by listing count + ensure current area is included
    selected_areas = [a["id"] for a in areas[:6] if a["id"] != "other"]
    if validated.area and validated.area.lower() not in selected_areas and validated.area.lower() != "other":
        selected_areas.insert(0, validated.area.lower())

    comparisons = []
    base_dict = validated.model_dump()

    for area_name in selected_areas[:7]:
        d = dict(base_dict)
        d["area"] = area_name
        try:
            r = run_predict(d)
            comparisons.append({
                "area": area_name.title(),
                "price": round(r["price"], 2),
                "formatted": format_inr(r["price"]),
                "per_sqft": round(r["per_sqft"], 2),
                "confidence": r["confidence"],
                "sample_count": r["sample_count"],
                "is_current": (validated.area and area_name.lower() == validated.area.lower()),
            })
        except Exception:
            continue

    # Sort by price descending
    comparisons.sort(key=lambda x: x["price"], reverse=True)
    return jsonify({"city": city.title(), "areas": comparisons}), 200


@app.route("/compare", methods=["POST"])
def compare():
    """Compares price of current property configuration across major Indian metros."""
    raw_json = request.get_json(silent=True)
    if not raw_json:
        return jsonify({"error": "Missing payload"}), 400

    try:
        validated = PropertyInput.model_validate(raw_json)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    metros = ["thane", "mumbai", "new-delhi", "gurgaon", "bangalore", "hyderabad", "pune", "ahmedabad", "kolkata"]
    target_cities = [c for c in metros if c in METADATA.get("cities", [])]
    if validated.location not in target_cities and validated.location in METADATA.get("cities", []):
        target_cities.insert(0, validated.location)

    comparisons = []
    base_data = validated.model_dump()
    base_data["area"] = "other"  # fair comparison across whole cities

    for city in target_cities[:8]:
        d = dict(base_data)
        d["location"] = city
        d["state"] = METADATA.get("city_to_state", {}).get(city, "Other")
        try:
            r = run_predict(d)
            comparisons.append({
                "city": city.title(),
                "location_id": city,
                "price": round(r["price"], 2),
                "formatted": format_inr(r["price"]),
                "per_sqft": round(r["per_sqft"], 2),
                "is_current": (city == validated.location),
            })
        except Exception:
            continue

    return jsonify({"comparisons": comparisons}), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
