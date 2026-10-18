"""Hierarchical dataset builder: State -> City -> Area.

Extracts normalized area names from listing titles and society names,
integrates the state mapping, and computes hierarchical benchmark statistics.
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_cleaning import convert_amount, to_sqft, parse_bhk, parse_floor, MIN_AREA, MAX_AREA

ROOT_DIR = Path(__file__).resolve().parent

# Load City to State Mapping
with open(ROOT_DIR / "data" / "city_state_mapping.json", "r", encoding="utf-8") as f:
    CITY_STATE = json.load(f)


def normalize_area_text(area_raw: str, city: str, society: str = None) -> str:
    """Cleans and standardizes area strings into uniform neighborhood names."""
    if not isinstance(area_raw, str):
        return "other"
    
    t = area_raw.strip().lower()
    
    # Strip society prefix if present
    if isinstance(society, str) and society.strip():
        soc = society.strip().lower()
        if t.startswith(soc):
            t = t[len(soc):].strip()
            
    # Remove city name and common variations
    city_clean = city.lower().replace("-", " ")
    t = re.sub(rf"\b{re.escape(city_clean)}\b.*$", "", t)
    t = re.sub(rf"\b{re.escape(city.lower())}\b.*$", "", t)
    
    # Standardize 'sector-xx' or 'sector xx'
    t = re.sub(r"\bsec(?:tor)?[\s\-_]*([0-9]+[a-z]?)\b", r"sector \1", t)
    t = re.sub(r"\bphase[\s\-_]*([0-9]+[a-z]?)\b", r"phase \1", t)
    t = re.sub(r"\bblock[\s\-_]*([0-9a-z]+)\b", r"block \1", t)
    
    # Remove leading/trailing non-alphanumeric chars
    t = re.sub(r"^[^a-z0-9]+", "", t)
    t = re.sub(r"[^a-z0-9]+$", "", t)
    
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    
    # If the remaining text is trivial or too short, fallback to society or 'other'
    if len(t) < 3:
        if isinstance(society, str) and len(society.strip()) >= 3:
            t = re.sub(r"\s+", " ", society.strip().lower())
        else:
            return "other"
            
    return t


def extract_area_from_title(title: str, society: str, city: str) -> str:
    """Extracts neighborhood/sector from Title string."""
    if not isinstance(title, str):
        return "other"
        
    t = title.strip()
    
    # Match 'for sale in ...' or 'for sale ...'
    m = re.search(r"for sale in\s+(.+)", t, re.IGNORECASE)
    if not m:
        m = re.search(r"for sale\s+(.+)", t, re.IGNORECASE)
        
    if not m:
        # Fallback to society if no 'for sale' matched
        if isinstance(society, str) and len(society.strip()) >= 3:
            return normalize_area_text(society, city)
        return "other"
        
    raw_area = m.group(1).strip()
    return normalize_area_text(raw_area, city, society=society)


def build_hierarchy_dataset(raw_csv_path="house_prices.csv", out_csv_path="data/processed/clean_hierarchy.csv"):
    print("Loading raw CSV:", raw_csv_path)
    df = pd.read_csv(raw_csv_path, on_bad_lines="skip")
    print(f"Raw rows: {len(df):,}")

    # 1. Sale listings only
    df = df[df["Transaction"].isin(["New Property", "Resale"])].copy()

    # 2. Conversions & parsing
    df["price"] = df["Amount(in rupees)"].apply(convert_amount)
    df["area_sqft"] = df["Carpet Area"].apply(to_sqft)
    df["bhk"] = df["Title"].apply(parse_bhk)
    floors = df["Floor"].apply(parse_floor)
    df["floor_no"] = floors.str[0]
    df["total_floors"] = floors.str[1]
    df["bathrooms"] = pd.to_numeric(df["Bathroom"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    # 3. Valid price, area, bhk
    df = df.dropna(subset=["price", "area_sqft", "bhk"])
    df = df[df["area_sqft"].between(MIN_AREA, MAX_AREA)]

    # 4. Bathroom median fill
    df["bathrooms"] = df["bathrooms"].fillna(df.groupby("bhk")["bathrooms"].transform("median"))
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # 5. Extract and normalize Area
    print("Extracting and normalizing area names...")
    df["area"] = [
        extract_area_from_title(title, soc, city)
        for title, soc, city in zip(df["Title"], df["Society"], df["location"])
    ]

    # 6. Map State from City
    df["state"] = df["location"].map(CITY_STATE).fillna("Other")

    # 7. Deduplicate listings (including area in the uniqueness key)
    dup_cols = ["Title", "location", "area", "price", "area_sqft", "floor_no"]
    df = df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
    print(f"Rows after deduplication: {len(df):,}")

    # 8. Per-city outlier removal on log(ppsf)
    df["ppsf"] = df["price"] / df["area_sqft"]
    df["log_ppsf"] = np.log(df["ppsf"])
    q1 = df.groupby("location")["log_ppsf"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby("location")["log_ppsf"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df = df[(df["log_ppsf"] >= q1 - 2.0 * iqr) & (df["log_ppsf"] <= q3 + 2.0 * iqr)].reset_index(drop=True)
    print(f"Rows after per-city outlier removal: {len(df):,}")

    # 9. Rare Area Aggregation (< 5 listings in city -> 'other')
    area_counts = df.groupby(["location", "area"])["price"].transform("count")
    df.loc[area_counts < 5, "area"] = "other"
    print(f"Unique areas after min-count 5 threshold: {df['area'].nunique():,}")

    # 10. Compute Log Target
    df["log_price"] = np.log(df["price"])

    # Keep relevant hierarchy features
    cols = ["state", "location", "area", "bhk", "bathrooms", "area_sqft", "floor_no", "total_floors",
            "Furnishing", "Transaction", "price", "log_price", "ppsf"]
    df = df[cols].reset_index(drop=True)

    # Save processed CSV
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    # Also save to root clean.csv to stay in sync
    df.to_csv(ROOT_DIR / "clean.csv", index=False)
    print(f"Saved cleaned hierarchy dataset to {out_path} and root clean.csv")

    # 11. Compute Hierarchical Statistics & Benchmarks
    print("\nComputing Hierarchical Benchmarks (State -> City -> Area)...")
    
    # State-level medians
    state_benchmarks = df.groupby("state")["ppsf"].agg(
        median="median",
        count="count"
    ).round(0).to_dict(orient="index")

    # City-level medians
    city_benchmarks = df.groupby("location")["ppsf"].agg(
        median="median",
        count="count"
    ).round(0).to_dict(orient="index")

    # Area-level medians (by city and area)
    area_stats = df.groupby(["location", "area"])["ppsf"].agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        count="count"
    ).round(0).reset_index()

    # Build nested metadata structure for frontend: State -> Cities -> Areas
    hierarchy_tree = {}
    for state, cities in df.groupby("state")["location"].unique().items():
        hierarchy_tree[state] = {}
        for city in sorted(cities):
            city_df = df[df["location"] == city]
            areas = (
                city_df.groupby("area")["ppsf"]
                .agg(count="count", median="median")
                .reset_index()
                .sort_values(by="count", ascending=False)
            )
            area_list = []
            for _, r in areas.iterrows():
                area_list.append({
                    "name": r["area"].title(),
                    "id": r["area"],
                    "count": int(r["count"]),
                    "median_ppsf": float(round(r["median"], 0)),
                })
            hierarchy_tree[state][city] = {
                "city_name": city.title(),
                "city_id": city,
                "count": len(city_df),
                "median_ppsf": float(round(city_df["ppsf"].median(), 0)),
                "areas": area_list
            }

    # Save metadata files
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "hierarchy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy_tree, f, indent=2)
    print(f"Saved {models_dir / 'hierarchy_metadata.json'}")

    benchmarks_data = {
        "state": state_benchmarks,
        "city": city_benchmarks,
        "area": {
            f"{r['location']}::{r['area']}": {
                "median": r["median"],
                "q25": r["q25"],
                "q75": r["q75"],
                "count": int(r["count"])
            }
            for _, r in area_stats.iterrows()
        }
    }
    with open(models_dir / "hierarchy_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(benchmarks_data, f, indent=2)
    print(f"Saved {models_dir / 'hierarchy_benchmarks.json'}")

    # Inspect Sample Output for top 5 cities
    print("\n--- Top Areas in Key Cities ---")
    top_cities = ["greater-noida", "thane", "gurgaon", "mumbai", "pune"]
    for city in top_cities:
        if city in df["location"].values:
            print(f"\nCity: {city.title()} (Total Listings: {len(df[df['location']==city]):,})")
            top_areas = df[df["location"] == city]["area"].value_counts().head(7)
            for a_name, a_cnt in top_areas.items():
                med_ppsf = df[(df["location"] == city) & (df["area"] == a_name)]["ppsf"].median()
                print(f"   • {a_name.title():<28} Count: {a_cnt:<4} | Median PPSF: Rs {med_ppsf:,.0f}/sqft")

    return df


if __name__ == "__main__":
    build_hierarchy_dataset()
