"""Regime classification engine for market condition filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from config import GameConfig, DEFAULT_CONFIG


@dataclass
class Regime:
    """Structured regime tags for a window."""

    trend: Literal["trending", "ranging", "neutral"]
    volatility: Literal["low", "medium", "high"]
    prior_move: Literal["strong_up", "strong_down", "mild"]


def _compute_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute ADX(period) for OHLC data."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed with Wilder's (EMA-like, alpha=1/period)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.fillna(0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def _classify_trend(
    adx: float,
    config: GameConfig,
) -> Literal["trending", "ranging", "neutral"]:
    if adx > config.adx_threshold_trending:
        return "trending"
    if adx < config.adx_threshold_ranging:
        return "ranging"
    return "neutral"


def _classify_volatility(
    vol: float,
    low_p33: float,
    high_p67: float,
) -> Literal["low", "medium", "high"]:
    if vol <= low_p33:
        return "low"
    if vol >= high_p67:
        return "high"
    return "medium"


def _classify_prior_move(pct_return: float) -> Literal["strong_up", "strong_down", "mild"]:
    if pct_return > 0.20:
        return "strong_up"
    if pct_return < -0.20:
        return "strong_down"
    return "mild"


def compute_regimes_for_asset(
    df: pd.DataFrame,
    config: GameConfig | None = None,
    step: int = 1,
) -> pd.DataFrame:
    """
    Pre-compute regime for every valid 60-day window in the asset's data.
    Returns DataFrame with columns: start_idx, end_idx, trend, volatility, prior_move.
    """
    config = config or DEFAULT_CONFIG
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    window_candles = config.window_candles
    vol_window = config.volatility_days * config.candles_per_day()

    # Returns for volatility
    df["returns"] = df["close"].pct_change()

    # Rolling 14-day realized volatility (std of returns)
    vol_window = min(vol_window, len(df) // 4)  # fallback if small
    vol_window = max(vol_window, 14)
    df["realized_vol"] = df["returns"].rolling(vol_window).std()

    # ADX
    df["adx"] = _compute_adx(df, config.adx_period)

    # Need enough data: window + forward for the last window
    min_len = window_candles + config.forward_candles + max(vol_window, config.adx_period * 2)
    if len(df) < min_len:
        return pd.DataFrame(columns=["start_idx", "end_idx", "trend", "volatility", "prior_move"])

    # Collect all valid windows
    windows = []
    vol_values = []

    for start in range(vol_window, len(df) - window_candles - config.forward_candles + 1, step):
        end = start + window_candles
        window_df = df.iloc[start:end]

        # ADX at end of window (last value)
        adx_val = window_df["adx"].iloc[-1]
        if pd.isna(adx_val) or adx_val <= 0:
            continue

        # Realized vol at end of window
        vol_val = window_df["realized_vol"].iloc[-1]
        if pd.isna(vol_val) or vol_val <= 0:
            continue

        # Prior move: % return over window
        entry_price = window_df["close"].iloc[0]
        exit_price = window_df["close"].iloc[-1]
        pct_return = (exit_price - entry_price) / entry_price if entry_price else 0

        trend = _classify_trend(adx_val, config)
        prior_move = _classify_prior_move(pct_return)

        windows.append({
            "start_idx": start,
            "end_idx": end,
            "trend": trend,
            "volatility_raw": vol_val,
            "prior_move": prior_move,
            "pct_return": pct_return,
        })
        vol_values.append(vol_val)

    if not windows:
        return pd.DataFrame(columns=["start_idx", "end_idx", "trend", "volatility", "prior_move"])

    # Bucket volatility: low (bottom 33%), high (top 33%), medium
    vol_arr = pd.Series(vol_values)
    low_p33 = vol_arr.quantile(0.33)
    high_p67 = vol_arr.quantile(0.67)

    result = []
    for w in windows:
        vol_class = _classify_volatility(w["volatility_raw"], low_p33, high_p67)
        result.append({
            "start_idx": w["start_idx"],
            "end_idx": w["end_idx"],
            "trend": w["trend"],
            "volatility": vol_class,
            "prior_move": w["prior_move"],
        })

    return pd.DataFrame(result)


def filter_windows_by_regime(
    windows_df: pd.DataFrame,
    trend_filter: list[str] | None,
    volatility_filter: list[str] | None,
    prior_move_filter: list[str] | None,
) -> pd.DataFrame:
    """Filter windows by selected regime filters. Empty filter = no constraint."""
    out = windows_df.copy()
    if trend_filter:
        out = out[out["trend"].isin(trend_filter)]
    if volatility_filter:
        out = out[out["volatility"].isin(volatility_filter)]
    if prior_move_filter:
        out = out[out["prior_move"].isin(prior_move_filter)]
    return out.reset_index(drop=True)
