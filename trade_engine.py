"""Trade simulation engine for direction predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from config import GameConfig, DEFAULT_CONFIG


Prediction = Literal["UP", "DOWN", "SAME"]


def classify_outcome(
    entry_price: float,
    exit_price: float,
    threshold: float,
) -> Prediction:
    """Classify price move as UP, DOWN, or SAME based on threshold."""
    if entry_price <= 0:
        return "SAME"
    pct = (exit_price - entry_price) / entry_price
    if pct > threshold:
        return "UP"
    if pct < -threshold:
        return "DOWN"
    return "SAME"


def get_trade_return(
    prediction: Prediction,
    entry_price: float,
    exit_price: float,
) -> float:
    """
    Simulate trade return:
    UP -> long: (exit - entry) / entry
    DOWN -> short: (entry - exit) / entry
    SAME -> flat: 0
    """
    if entry_price <= 0:
        return 0.0
    if prediction == "UP":
        return (exit_price - entry_price) / entry_price
    if prediction == "DOWN":
        return (entry_price - exit_price) / entry_price
    return 0.0


@dataclass
class TradeResult:
    """Result of a simulated trade."""

    prediction: Prediction
    actual: Prediction
    correct: bool
    trade_return: float
    entry_price: float
    exit_price: float


def simulate_trade(
    df: pd.DataFrame,
    start_idx: int,
    prediction: Prediction,
    config: GameConfig | None = None,
) -> TradeResult | None:
    """
    Simulate trade: enter at close of window, hold for forward_days, exit.
    Returns TradeResult or None if insufficient data.
    """
    config = config or DEFAULT_CONFIG
    end_idx = start_idx + config.window_candles
    exit_idx = end_idx + config.forward_candles - 1

    if exit_idx >= len(df):
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    entry_price = float(df.iloc[end_idx - 1]["close"])
    exit_price = float(df.iloc[exit_idx]["close"])

    actual = classify_outcome(
        entry_price,
        exit_price,
        config.threshold,
    )
    correct = prediction == actual
    trade_return = get_trade_return(prediction, entry_price, exit_price)

    return TradeResult(
        prediction=prediction,
        actual=actual,
        correct=correct,
        trade_return=trade_return,
        entry_price=entry_price,
        exit_price=exit_price,
    )


def get_future_prices(
    df: pd.DataFrame,
    start_idx: int,
    config: GameConfig | None = None,
) -> pd.DataFrame | None:
    """Get the future price series for the forward window (for reveal chart)."""
    config = config or DEFAULT_CONFIG
    end_idx = start_idx + config.window_candles
    exit_idx = end_idx + config.forward_candles - 1

    if exit_idx >= len(df):
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    future = df.iloc[end_idx : exit_idx + 1][["datetime", "open", "high", "low", "close"]].copy()
    future.columns = [c.lower() for c in future.columns]
    return future


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Compute max drawdown from equity curve (cumulative returns)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd


def compute_sharpe(
    returns: list[float],
    risk_free: float = 0.0,
) -> float:
    """Annualized Sharpe ratio (simplified). Assumes 4h returns."""
    if not returns or len(returns) < 2:
        return 0.0
    import pandas as pd
    s = pd.Series(returns)
    excess = s - risk_free
    if excess.std() == 0:
        return 0.0
    # Annualization: ~6*365 periods per year for 4h
    ann_factor = (6 * 365) ** 0.5
    return float(excess.mean() / excess.std() * ann_factor)
