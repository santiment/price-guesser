# Crypto Direction Game

A Streamlit-based game for predicting crypto price direction with regime filtering, trade simulation, timed mode, and bias analytics.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Features

- **Regime Classification**: Filter training by market condition (trend, volatility, prior move)
- **Trade Simulation Mode**: Simulate long/short/flat positions, track PnL, Sharpe, drawdown
- **Timed Mode**: Answer within N seconds or auto-submit as incorrect
- **Bias Dashboard**: Prediction bias, streak sensitivity, regime performance, trade behavior
- **Future Reveal**: 14-day chart with ±5% lines, entry/exit markers

## Data

Place 4h OHLCV CSV files in `candles_4h/` named `{SYMBOL}USDT.csv` (e.g. `BTCUSDT.csv`).

## Config

All parameters configurable via sidebar → Advanced Config:
- `window_days` (60), `forward_days` (14), `threshold` (0.05)
- `adx_threshold_trending` (25), `adx_threshold_ranging` (20)
- `timer_seconds` (5)
