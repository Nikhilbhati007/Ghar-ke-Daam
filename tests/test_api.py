"""Pytest test suite for Hierarchical Backend API (State -> City -> Area).

Covers:
- /health, /metadata, and /areas endpoints
- Valid hierarchical predictions and confidence tier assignment
- City-level fallback when area is omitted or has < 5 listings
- State-to-city geographic validation
- Multi-level benchmark calculation (Area, City, State)
- Local area comparison endpoint (/compare-areas)
- Metro comparison endpoint (/compare)
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_renders_template(client):
    """GET / must render the full HTML interface."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Ghar ka Daam" in html
    assert "price-form" in html


def test_health_endpoint(client):
    """GET /health must return status 200 and 'healthy'."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_metadata_endpoint(client):
    """GET /metadata must return hierarchical tree."""
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.get_json()
    assert "states" in data
    assert "Maharashtra" in data["states"]
    assert "cities" in data
    assert "thane" in data["cities"]
    assert "hierarchy" in data
    assert "area_limits_sqft" in data


def test_areas_endpoint(client):
    """GET /areas?city=thane returns list of areas with listing counts."""
    res = client.get("/areas?city=thane")
    assert res.status_code == 200
    data = res.get_json()
    assert data["city"] == "thane"
    assert data["state"] == "Maharashtra"
    assert len(data["areas"]) > 0
    assert any("ghodbunder" in a["id"] or "manpada" in a["id"] for a in data["areas"])


def test_predict_high_confidence_area(client):
    """POST /predict with high-volume area returns 200 and 'High' confidence."""
    payload = {
        "state": "Uttar Pradesh",
        "location": "greater-noida",
        "area": "sector 1",
        "area_sqft": 1000.0,
        "bhk": 2.0,
        "bathrooms": 2.0,
        "Furnishing": "Semi-Furnished",
        "Transaction": "Resale",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["confidence"] == "High"
    assert data["sample_count"] >= 20
    assert data["price"] > 0
    assert "benchmarks" in data
    assert "city" in data["benchmarks"]
    assert "state" in data["benchmarks"]


def test_predict_city_fallback_low_confidence(client):
    """POST /predict with no area returns 200 and 'Low' confidence fallback."""
    payload = {
        "location": "greater-noida",
        "area_sqft": 1000.0,
        "bhk": 2.0,
        "bathrooms": 2.0,
        "Furnishing": "Semi-Furnished",
        "Transaction": "Resale",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["confidence"] == "Low"
    assert "City-level" in data["confidence_label"]


def test_predict_state_city_mismatch(client):
    """POST /predict with city not in stated state returns 400 Bad Request."""
    payload = {
        "state": "Delhi",
        "location": "thane",
        "area_sqft": 900.0,
        "bhk": 2.0,
        "bathrooms": 2.0,
        "Furnishing": "Semi-Furnished",
        "Transaction": "Resale",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert "not in State 'Delhi'" in data["error"]


def test_compare_areas_endpoint(client):
    """POST /compare-areas returns ranked areas in the same city."""
    payload = {
        "location": "thane",
        "area_sqft": 950.0,
        "bhk": 2.0,
        "bathrooms": 2.0,
        "Furnishing": "Semi-Furnished",
        "Transaction": "Resale",
    }
    res = client.post("/compare-areas", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "areas" in data
    assert len(data["areas"]) >= 3
    assert data["city"] == "Thane"


def test_compare_metro_endpoint(client):
    """POST /compare returns cross-metro valuations."""
    payload = {
        "location": "thane",
        "area_sqft": 1000.0,
        "bhk": 2.0,
        "bathrooms": 2.0,
        "Furnishing": "Semi-Furnished",
        "Transaction": "Resale",
    }
    res = client.post("/compare", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "comparisons" in data
    assert len(data["comparisons"]) >= 5


def test_predict_area_below_minimum(client):
    """POST /predict with area below minimum bound returns 400."""
    payload = {
        "location": "thane",
        "area_sqft": 45.0,
        "bhk": 1.0,
        "bathrooms": 1.0,
        "Furnishing": "Unfurnished",
        "Transaction": "Resale",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert "Area must be between" in data["error"]


def test_predict_missing_required_field(client):
    """POST /predict without required field returns 400."""
    payload = {
        "location": "thane",
        "bhk": 2.0,
        "bathrooms": 2.0,
        "Furnishing": "Semi-Furnished",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
