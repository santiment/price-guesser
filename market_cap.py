"""Market cap data for crypto assets. Loads from market_caps.csv."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

MarketCapTier = Literal["majors", "large_caps", "mid_caps", "small_caps"]

# Thresholds in USD
MAJORS_MIN = 5_000_000_000      # > $5B
LARGE_CAP_MIN = 500_000_000    # $500M - $5B
LARGE_CAP_MAX = 5_000_000_000
MID_CAP_MIN = 50_000_000       # $50M - $500M
MID_CAP_MAX = 500_000_000
SMALL_CAP_MAX = 50_000_000     # < $50M

DEFAULT_CSV_PATH = Path(__file__).parent / "market_caps.csv"


def get_market_caps(csv_path: Path | str | None = None) -> dict[str, float]:
    """
    Load symbol -> market_cap (USD) from CSV.
    CSV format: symbol,market_cap
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    result = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                sym = parts[0].strip().upper()
                try:
                    mc = float(parts[1].strip())
                    if sym:
                        result[sym] = mc
                except ValueError:
                    continue
    return result


def _lookup_symbol(sym: str, market_caps: dict[str, float]) -> float | None:
    """Get market cap for symbol. Strip leading digits (Binance prefix) if needed."""
    sym_upper = sym.upper()
    mc = market_caps.get(sym_upper) or market_caps.get(sym)
    if mc is not None:
        return mc
    # Binance: tickers like 1000PEPE - strip 1000 or greater powers of 10 only
    stripped = re.sub(r"^10{3,}", "", sym_upper)
    if stripped and stripped != sym_upper:
        return market_caps.get(stripped)
    return None


def _classify_market_cap(market_cap: float) -> MarketCapTier | None:
    if market_cap >= MAJORS_MIN:
        return "majors"
    if LARGE_CAP_MIN <= market_cap < LARGE_CAP_MAX:
        return "large_caps"
    if MID_CAP_MIN <= market_cap < MID_CAP_MAX:
        return "mid_caps"
    if market_cap < SMALL_CAP_MAX and market_cap > 0:
        return "small_caps"
    return None


def filter_symbols_by_market_cap(
    symbols: list[str],
    market_caps: dict[str, float],
    selected_tiers: list[MarketCapTier],
) -> list[str]:
    """
    Filter symbols by selected market cap tiers.
    Empty selected_tiers = no filter (all symbols).
    Symbols without market cap data are excluded when filter is applied.
    """
    if not selected_tiers:
        return symbols

    result = []
    for sym in symbols:
        mc = _lookup_symbol(sym, market_caps)
        if mc is None:
            continue
        tier = _classify_market_cap(mc)
        if tier in selected_tiers:
            result.append(sym)
    return result
