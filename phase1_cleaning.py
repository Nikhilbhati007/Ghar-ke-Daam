"""Phase 1 - Data foundation for the house price project.

Usage (in a notebook or terminal):
    from phase1_cleaning import build_clean_dataset
    df = build_clean_dataset("house_prices.csv", out_path="data/processed/clean.csv")

Steps, in order:
 1. Keep sale listings only (drop Rent/Lease)
 2. Convert price text ("1.4 Cr", "42 Lac") to rupees
 3. Standardise carpet area to sq ft (unreliable units become NaN)
 4. Parse BHK from Title
 5. Parse floor number / total floors from Floor
 6. Remove duplicates and per-city outliers
 7. Add log-price target
"""
import re

import numpy as np
import pandas as pd

TO_SQFT = {"sqft": 1.0, "sqyrd": 9.0, "sqm": 10.7639}
MIN_AREA, MAX_AREA = 100, 8000


# ---------- 2. price ----------
def convert_amount(text):
    if not isinstance(text, str):
        return np.nan
    t = text.lower().strip()
    if "call for price" in t:
        return np.nan
    m = re.search(r"(\d+\.?\d*)", t.replace(",", ""))
    if not m:
        return np.nan
    val = float(m.group(1))
    if "cr" in t:
        return val * 1e7
    if "lac" in t:
        return val * 1e5
    return val


# ---------- 3. area ----------
def to_sqft(text):
    """Only sqft / sqyrd / sqm are trusted. acre, bigha, kanal, marla... -> NaN."""
    if not isinstance(text, str):
        return np.nan
    t = text.lower().strip()
    m = re.search(r"(\d+\.?\d*)", t.replace(",", ""))
    if not m:
        return np.nan
    for unit, factor in TO_SQFT.items():
        if unit in t:
            return float(m.group(1)) * factor
    return np.nan


# ---------- 4. BHK ----------
def parse_bhk(title):
    """'3 BHK Ready to Occupy Flat for sale in X' -> 3. Studio -> 1."""
    if not isinstance(title, str):
        return np.nan
    m = re.search(r"(\d+)\s*bhk", title.lower())
    if m:
        return int(m.group(1))
    if "studio" in title.lower():
        return 1
    return np.nan


# ---------- 5. floor ----------
def parse_floor(text):
    """'7 out of 20' -> (7, 20). 'Ground out of 5' -> (0, 5). Basement -> (-1, n)."""
    if not isinstance(text, str):
        return (np.nan, np.nan)
    t = text.lower().strip()
    total = re.search(r"out of\s*(\d+)", t)
    total = int(total.group(1)) if total else np.nan
    if t.startswith("ground"):
        return (0, total)
    if "basement" in t:
        return (-1, total)
    m = re.match(r"(\d+)", t)
    return (int(m.group(1)) if m else np.nan, total)


# ---------- main pipeline ----------
def build_clean_dataset(csv_path, out_path=None, verbose=True):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    log = {"raw rows": len(df)}

    # 1. sale listings only
    df = df[df["Transaction"].isin(["New Property", "Resale"])].copy()
    log["after sale-only filter"] = len(df)

    # 2-5. conversions and new features
    df["price"] = df["Amount(in rupees)"].apply(convert_amount)
    df["area_sqft"] = df["Carpet Area"].apply(to_sqft)
    df["bhk"] = df["Title"].apply(parse_bhk)
    floors = df["Floor"].apply(parse_floor)
    df["floor_no"] = floors.str[0]
    df["total_floors"] = floors.str[1]
    df["bathrooms"] = pd.to_numeric(df["Bathroom"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    # rows we cannot train on: no price or no usable area or missing BHK
    df = df.dropna(subset=["price", "area_sqft", "bhk"])
    df = df[df["area_sqft"].between(MIN_AREA, MAX_AREA)]
    log["after price/area/bhk validity"] = len(df)

    # fill bathrooms from the median of the same BHK, then overall median
    df["bathrooms"] = df["bathrooms"].fillna(df.groupby("bhk")["bathrooms"].transform("median"))
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # 6a. duplicates (same listing scraped twice would leak into the test set)
    dup_cols = ["Title", "location", "price", "area_sqft", "floor_no"]
    df = df.drop_duplicates(subset=dup_cols)
    log["after dedupe"] = len(df)

    # 6b. per-city outliers on log price per sq ft (IQR rule, factor 2.0)
    df["ppsf"] = df["price"] / df["area_sqft"]
    df["log_ppsf"] = np.log(df["ppsf"])
    q1 = df.groupby("location")["log_ppsf"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby("location")["log_ppsf"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df = df[(df["log_ppsf"] >= q1 - 2.0 * iqr) & (df["log_ppsf"] <= q3 + 2.0 * iqr)]
    log["after per-city outlier removal"] = len(df)

    # 7. log target
    df["log_price"] = np.log(df["price"])

    keep = ["location", "bhk", "bathrooms", "area_sqft", "floor_no", "total_floors",
            "Furnishing", "Transaction", "price", "log_price"]
    df = df[keep].reset_index(drop=True)

    if verbose:
        print("Row count at each step:")
        for k, v in log.items():
            print(f"  {k:<34}{v:>9,}")
        print("\nMissing values left:\n", df.isna().sum()[lambda s: s > 0])
    if out_path:
        import os
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    import sys
    build_clean_dataset(sys.argv[1], out_path=sys.argv[2] if len(sys.argv) > 2 else None)
