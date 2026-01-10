from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

_UNIT_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)")

def _parse_unit_multiplier(unit: str) -> float:
    '''
    Parse common WFP unit strings and return multiplier to convert to **kg**.

    Examples:
      "kg" -> 1
      "1 kg" -> 1
      "100kg" -> 100
      "50 KG" -> 50
      "g" -> 0.001
      "100 g" -> 0.1
    If unrecognized, returns 1 (assume already per-kg).
    '''
    if not isinstance(unit, str) or unit.strip() == "":
        return 1.0
    u = unit.strip().lower().replace("kgs", "kg").replace("kilogram", "kg")

    # Exact units
    if u in {"kg", "1kg", "1 kg"}:
        return 1.0
    if u in {"g", "gram", "grams"}:
        return 0.001

    m = _UNIT_RE.search(u.replace("/", " "))
    if not m:
        return 1.0
    num = float(m.group("num"))
    unit_str = m.group("unit").lower()
    if unit_str in {"kg"}:
        return num
    if unit_str in {"g"}:
        return num * 0.001
    # unknown, fallback
    return 1.0

@dataclass
class PreparedData:
    series: Dict[str, pd.Series]          # key -> pandas Series indexed by datetime
    meta: pd.DataFrame                   # metadata per series key

def load_and_prepare(
    url: str,
    country: str,
    currency: str,
    market: str,
    n_commodities: int,
    date_col: str,
    commodity_col: str,
    market_col: str,
    price_col: str,
    unit_col: str,
    freq: str,
    agg: str,
) -> PreparedData:
    df = pd.read_csv(url)
    df = df.copy()

    # Basic filtering
    if "country" in df.columns:
        df = df[df["country"].astype(str).str.strip().str.lower() == country.lower()]
    if "currency" in df.columns:
        df = df[df["currency"].astype(str).str.strip().str.upper() == currency.upper()]
    df = df[df[market_col].astype(str).str.strip() == market]

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, commodity_col, price_col])

    # Normalize to BDT/kg
    mult = df[unit_col].apply(_parse_unit_multiplier) if unit_col in df.columns else 1.0
    df["_price_per_kg"] = df[price_col].astype(float) / mult.astype(float)

    # Pick top commodities by coverage
    counts = df.groupby(commodity_col)["_price_per_kg"].count().sort_values(ascending=False)
    top = counts.head(n_commodities).index.tolist()
    df = df[df[commodity_col].isin(top)]

    # Resample each commodity to a regular grid
    series = {}
    meta_rows = []
    for comm, g in df.groupby(commodity_col):
        g = g.sort_values(date_col)
        s = g.set_index(date_col)["_price_per_kg"].astype(float)
        if agg == "mean":
            s = s.resample(freq).mean()
        else:
            s = s.resample(freq).median()
        s = s.interpolate(limit_direction="both")
        key = f"{market}__{comm}"
        series[key] = s
        meta_rows.append({"key": key, "market": market, "commodity": comm, "n": int(s.dropna().shape[0])})

    meta = pd.DataFrame(meta_rows).sort_values("n", ascending=False).reset_index(drop=True)
    return PreparedData(series=series, meta=meta)
