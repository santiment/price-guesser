"""Load and resolve metrics from metrics_output CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

METRICS_DIR = Path(__file__).parent / "metrics_output"


def get_available_metrics(metrics_dir: str | Path | None = None) -> list[str]:
    """Get list of metric column names from BTC.csv (has most metrics)."""
    base = Path(metrics_dir) if metrics_dir else METRICS_DIR
    path = base / "BTC.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, nrows=0)
    return [c for c in df.columns if c != "timestamp"]


def symbol_has_valid_metric(symbol: str, metric: str, metrics_dir: str | Path | None = None) -> bool:
    """
    Check if a symbol has valid (non-empty, not all zeros) data for the given metric.
    """
    df = load_metrics(symbol, metrics_dir)
    if df is None or metric not in df.columns:
        return False
    vals = df[metric]
    if vals.isna().all():
        return False
    valid = vals.dropna()
    if len(valid) == 0:
        return False
    if (valid == 0).all():
        return False
    return True


def filter_symbols_by_metric(
    symbols: list[str],
    metric: str,
    metrics_dir: str | Path | None = None,
) -> list[str]:
    """
    Return only symbols that have valid (non-empty, not all zeros) data for the metric.
    """
    return [s for s in symbols if symbol_has_valid_metric(s, metric, metrics_dir)]


def load_metrics(symbol: str, metrics_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Load metrics CSV for a symbol. Returns None if not found."""
    base = Path(metrics_dir) if metrics_dir else METRICS_DIR
    path = base / f"{symbol.upper()}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    stripped = re.sub(r"^10{3,}", "", symbol.upper())
    if stripped and stripped != symbol.upper():
        path = base / f"{stripped}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    return None
