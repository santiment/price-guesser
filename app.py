"""
Crypto Direction Game - Streamlit App

A modular Streamlit-based game for predicting crypto price direction
with regime filtering, trade simulation, timed mode, and bias analytics.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import streamlit as st

from config import GameConfig, DEFAULT_CONFIG
from regimes import compute_regimes_for_asset, filter_windows_by_regime
from trade_engine import classify_outcome, simulate_trade
from stats_tracker import SessionStats
from bias_analytics import render_bias_dashboard
from market_cap import get_market_caps, filter_symbols_by_market_cap


# ---------------------------------------------------------------------------
# Config & Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_market_caps() -> dict[str, float]:
    """Load market caps from market_caps.csv."""
    return get_market_caps()


@st.cache_data
def load_candles(symbol: str, candles_dir: str) -> pd.DataFrame | None:
    """Load candle data for a symbol."""
    path = Path(candles_dir) / f"{symbol}USDT.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data
def load_all_regimes(
    symbols: tuple[str, ...],
    candles_dir: str,
    window_days: int,
    forward_days: int,
    threshold: float,
    adx_trending: float,
    adx_ranging: float,
    volatility_days: int,
    step: int = 6,
) -> dict[str, pd.DataFrame]:
    """Pre-compute regimes for assets. Cached. step=6 samples ~1 window/day for 4h data."""
    config = GameConfig(
        window_days=window_days,
        forward_days=forward_days,
        threshold=threshold,
        adx_threshold_trending=adx_trending,
        adx_threshold_ranging=adx_ranging,
        volatility_days=volatility_days,
    )
    result = {}
    for sym in symbols:
        df = load_candles(sym, candles_dir)
        if df is not None and len(df) > 100:
            regimes = compute_regimes_for_asset(df, config, step=step)
            if len(regimes) > 0:
                result[sym] = regimes
    return result


def get_available_symbols(candles_dir: str) -> list[str]:
    """List available symbols from candles dir."""
    path = Path(candles_dir)
    if not path.exists():
        return []
    symbols = []
    for f in path.glob("*USDT.csv"):
        symbols.append(f.stem.replace("USDT", ""))
    return sorted(symbols)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def init_session_state():
    if "stats" not in st.session_state:
        st.session_state.stats = SessionStats()
    if "current_round" not in st.session_state:
        st.session_state.current_round = None
    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False
    if "timer_start" not in st.session_state:
        st.session_state.timer_start = None
    if "game_ready" not in st.session_state:
        st.session_state.game_ready = False
    if "pool" not in st.session_state:
        st.session_state.pool = []


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_equity_curve(stats: SessionStats):
    """Mini equity curve in sidebar."""
    if not stats.trade_returns:
        return
    import plotly.graph_objects as go
    equity = stats.equity_curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity,
        mode="lines",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
    ))
    fig.update_layout(
        title="Equity Curve",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showticklabels=False),
        yaxis=dict(title=""),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_unified_chart(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    config: GameConfig,
    symbol: str,
    show_future: bool = False,
    entry_price: float | None = None,
    exit_price: float | None = None,
    reveal: bool = False,
):
    """
    Single chart: historical (60-day) + optional future (14-day) attached to the right.
    When reveal=False: hide asset name and time axis. Reveal after submit.
    """
    import plotly.graph_objects as go

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    hist = df.iloc[start_idx:end_idx]
    fig = go.Figure(data=[
        go.Candlestick(
            x=hist["datetime"],
            open=hist["open"],
            high=hist["high"],
            low=hist["low"],
            close=hist["close"],
            name="OHLC",
        )
    ])

    if show_future:
        exit_idx = end_idx + config.forward_candles - 1
        if exit_idx < len(df):
            future = df.iloc[end_idx : exit_idx + 1]
            fig.add_trace(go.Scatter(
                x=future["datetime"],
                y=future["close"],
                mode="lines+markers",
                name="Future",
                line=dict(color="#00d4aa", width=2),
            ))
            if entry_price is not None and exit_price is not None:
                up_line = entry_price * (1 + config.threshold)
                down_line = entry_price * (1 - config.threshold)
                fig.add_hline(y=up_line, line_dash="dash", line_color="green", annotation_text="+5%")
                fig.add_hline(y=down_line, line_dash="dash", line_color="red", annotation_text="-5%")
                fig.add_hline(y=entry_price, line_dash="dot", line_color="gray", annotation_text="Entry")
                fig.add_hline(y=exit_price, line_dash="dot", line_color="orange", annotation_text="Exit")

    layout = dict(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    if reveal:
        layout["title"] = f"{symbol} — 60-day window" + (" + 14-day future" if show_future else "")
    else:
        layout["title"] = ""
        layout["xaxis"] = dict(showticklabels=False)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Crypto Direction Game", layout="wide", initial_sidebar_state="expanded")
    st.title("Crypto Direction Game")
    init_session_state()

    config = DEFAULT_CONFIG
    candles_dir = config.candles_dir
    symbols = get_available_symbols(candles_dir)
    if not symbols:
        st.error("No candle data found. Place CSV files in candles_4h/")
        return

    # Load market caps from CSV for filter
    market_caps = load_market_caps()

    # Sidebar: Config (form for explicit Apply)
    with st.sidebar:
        st.header("Settings")

        with st.form("settings_form", clear_on_submit=False):
            mode = st.radio("Mode", ["Classification Mode", "Trade Simulation Mode"], index=0)
            trade_mode = mode == "Trade Simulation Mode"
            timed_mode = st.checkbox("Timed Mode", value=False)

            st.subheader("Regime Filters")
            trend_opts = ["trending", "ranging", "neutral"]
            vol_opts = ["low", "medium", "high"]
            prior_opts = ["strong_up", "strong_down", "mild"]
            trend_filter = st.multiselect("Trend", trend_opts, default=[])
            volatility_filter = st.multiselect("Volatility", vol_opts, default=[])
            prior_move_filter = st.multiselect("Prior Move", prior_opts, default=[])

            st.subheader("Market Cap")
            cap_opts = ["majors", "large_caps", "mid_caps", "small_caps"]
            cap_labels = {
                "majors": "Majors (>$5B)",
                "large_caps": "Large caps ($500M–$5B)",
                "mid_caps": "Mid caps ($50M–$500M)",
                "small_caps": "Small caps (<$50M)",
            }
            market_cap_filter = st.multiselect(
                "Market cap tiers",
                options=cap_opts,
                format_func=lambda x: cap_labels[x],
                default=[],
            )

            filtered_symbols = (
                filter_symbols_by_market_cap(symbols, market_caps, market_cap_filter)
                if market_cap_filter and market_caps
                else symbols
            )
            if not filtered_symbols:
                filtered_symbols = symbols

            selected_symbol = st.selectbox("Asset", filtered_symbols, index=0)
            num_assets = st.slider(
                "Assets to sample from",
                1,
                len(filtered_symbols),
                min(5, len(filtered_symbols)),
            )

            with st.expander("Advanced Config"):
                cfg_window = st.number_input("window_days", 30, 120, config.window_days, key="cfg_window")
                cfg_forward = st.number_input("forward_days", 7, 30, config.forward_days, key="cfg_forward")
                cfg_thresh = st.number_input("threshold", 0.01, 0.20, float(config.threshold), 0.01, key="cfg_thresh")
                cfg_adx_t = st.number_input("ADX trending", 15.0, 40.0, config.adx_threshold_trending, 1.0, key="cfg_adx_t")
                cfg_adx_r = st.number_input("ADX ranging", 10.0, 25.0, config.adx_threshold_ranging, 1.0, key="cfg_adx_r")
                cfg_vol = st.number_input("volatility_days", 7, 30, config.volatility_days, key="cfg_vol")
                cfg_timer = int(st.number_input("timer_seconds", 1, 30, config.timer_seconds, key="cfg_timer"))

            config = GameConfig(
                window_days=cfg_window,
                forward_days=cfg_forward,
                threshold=cfg_thresh,
                adx_threshold_trending=cfg_adx_t,
                adx_threshold_ranging=cfg_adx_r,
                volatility_days=cfg_vol,
                timer_seconds=cfg_timer,
            )

            apply_clicked = st.form_submit_button("Apply Settings & Prepare Game")

        if apply_clicked:
            with st.spinner("Computing regimes (may take a moment)..."):
                sample_symbols = [selected_symbol] + [
                    s for s in filtered_symbols if s != selected_symbol
                ][: num_assets - 1]
                sample_symbols = tuple(sample_symbols[:num_assets])

                all_regimes = load_all_regimes(
                    sample_symbols,
                    candles_dir,
                    config.window_days,
                    config.forward_days,
                    config.threshold,
                    config.adx_threshold_trending,
                    config.adx_threshold_ranging,
                    config.volatility_days,
                    step=6,
                )
                pool = []
                for sym, regimes_df in all_regimes.items():
                    filtered = filter_windows_by_regime(
                        regimes_df,
                        trend_filter or None,
                        volatility_filter or None,
                        prior_move_filter or None,
                    )
                    for _, row in filtered.iterrows():
                        pool.append((sym, int(row["start_idx"]), dict(row)))
                st.session_state.pool = pool
                st.session_state.game_config = config
                st.session_state.game_ready = len(pool) > 0
            if st.session_state.game_ready:
                st.success(f"Ready! {len(pool)} windows available.")
            else:
                st.warning("No windows match filters. Relax filters and apply again.")
            st.rerun()

    config = st.session_state.get("game_config", config)
    pool = st.session_state.get("pool", [])

    # Sidebar: Stats & Equity
    with st.sidebar:
        stats = st.session_state.stats
        st.subheader("Stats")
        st.metric("Accuracy", f"{stats.accuracy:.1%}")
        if trade_mode:
            st.metric("Cumulative PnL", f"{stats.cumulative_pnl:.2%}")
            st.metric("Win Rate", f"{stats.win_rate:.1%}")
            st.metric("Sharpe", f"{stats.sharpe_ratio:.2f}")
            st.metric("Max DD", f"{stats.max_drawdown:.1%}")
            st.metric("Avg Win", f"{stats.avg_win:.2%}")
            st.metric("Avg Loss", f"{stats.avg_loss:.2%}")
            render_equity_curve(stats)
        if timed_mode:
            st.metric("Timed Accuracy", f"{stats.timed_correct / stats.timed_total:.1%}" if stats.timed_total else "N/A")
            st.metric("Untimed Accuracy", f"{stats.untimed_correct / stats.untimed_total:.1%}" if stats.untimed_total else "N/A")

    # Main area: Round or Start
    if st.session_state.current_round is None:
        if not st.session_state.game_ready or not pool:
            st.info(
                "Configure settings in the sidebar and click **Apply Settings & Prepare Game** to start."
            )
            st.divider()
            render_bias_dashboard(st.session_state.stats, st)
            return

        st.subheader("Ready to play")
        if st.button("Start Round", type="primary", use_container_width=True):
            sym, start_idx, regime = random.choice(pool)
            st.session_state.current_round = {
                "symbol": sym,
                "start_idx": start_idx,
                "regime": regime,
            }
            st.session_state.answer_submitted = False
            if timed_mode:
                st.session_state.timer_start = pd.Timestamp.now()
            st.rerun()
        st.divider()
        render_bias_dashboard(st.session_state.stats, st)
        return

    # In round
    round_data = st.session_state.current_round
    symbol = round_data["symbol"]
    start_idx = round_data["start_idx"]
    regime = round_data["regime"]

    df = load_candles(symbol, candles_dir)
    if df is None:
        st.error("Failed to load data")
        st.session_state.current_round = None
        st.rerun()
        return
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    end_idx = start_idx + config.window_candles
    window_df = df.iloc[start_idx:end_idx]

    # Timer (if timed mode)
    if timed_mode and not st.session_state.answer_submitted:
        elapsed = (pd.Timestamp.now() - st.session_state.timer_start).total_seconds()
        remaining = max(0, config.timer_seconds - elapsed)
        if remaining <= 0:
            # Auto-submit as "No Answer"
            entry_price = float(window_df.iloc[-1]["close"])
            exit_price = float(df.iloc[end_idx + config.forward_candles - 1]["close"])
            actual = classify_outcome(entry_price, exit_price, config.threshold)
            regime_key = f"{regime.get('trend', '?')}/{regime.get('volatility', '?')}/{regime.get('prior_move', '?')}"
            st.session_state.stats.record(
                prediction=None,
                actual=actual,
                correct=False,
                trade_return=0.0,
                regime_key=regime_key,
                was_timed=True,
            )
            st.session_state.answer_submitted = True
            st.session_state.pending_result = {
                "prediction": None,
                "actual": actual,
                "correct": False,
                "trade_return": 0.0,
                "trade_result": type("TradeResult", (), {"entry_price": entry_price, "exit_price": exit_price})(),
                "was_timed": True,
                "timed_out": True,
            }
            st.rerun()
        st.progress(remaining / config.timer_seconds)
        st.caption(f"Time remaining: {remaining:.1f}s")

    # Unified chart (historical only when guessing; + future when revealed)
    show_future = st.session_state.answer_submitted and "pending_result" in st.session_state
    entry_px = None
    exit_px = None
    if show_future and "pending_result" in st.session_state:
        pr = st.session_state.pending_result
        tr = pr.get("trade_result")
        if tr is not None:
            entry_px = getattr(tr, "entry_price", None)
            exit_px = getattr(tr, "exit_price", None)
        if entry_px is None:
            entry_px = float(df.iloc[end_idx - 1]["close"])
            exit_px = float(df.iloc[end_idx + config.forward_candles - 1]["close"])
    render_unified_chart(
        df, start_idx, end_idx, config, symbol,
        show_future=show_future,
        entry_price=entry_px,
        exit_price=exit_px,
        reveal=show_future,
    )

    # Buttons (only if not submitted)
    if not st.session_state.answer_submitted:
        st.markdown("""
        <style>
        section.main [data-testid="column"]:nth-of-type(1) button { background-color: #22c55e !important; color: white !important; }
        section.main [data-testid="column"]:nth-of-type(2) button { background-color: #6b7280 !important; color: white !important; }
        section.main [data-testid="column"]:nth-of-type(3) button { background-color: #ef4444 !important; color: white !important; }
        </style>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("↑ UP", use_container_width=True, key="btn_up"):
                submit_answer("UP", df, symbol, start_idx, regime, config, trade_mode, timed_mode)
        with col2:
            if st.button("→ SAME", use_container_width=True, key="btn_same"):
                submit_answer("SAME", df, symbol, start_idx, regime, config, trade_mode, timed_mode)
        with col3:
            if st.button("↓ DOWN", use_container_width=True, key="btn_down"):
                submit_answer("DOWN", df, symbol, start_idx, regime, config, trade_mode, timed_mode)
        return

    # Result reveal
    if "pending_result" in st.session_state:
        res = st.session_state.pending_result
        actual = res["actual"]
        correct = res["correct"]
        trade_return = res["trade_return"]
        trade_result = res.get("trade_result")
        timed_out = res.get("timed_out", False)

        if timed_out:
            st.warning("Time's up! Counted as incorrect.")
        else:
            if correct:
                st.success(f"Correct! Actual: {actual}")
            else:
                st.error(f"Incorrect. Actual: {actual}")
        if trade_mode and trade_result is not None:
            st.metric("Trade Return", f"{trade_return:.2%}")

        # Chart with future is already shown above (unified chart)

        pool = st.session_state.get("pool", [])
        config = st.session_state.get("game_config", DEFAULT_CONFIG)
        if st.button("Next Round", type="primary") and pool:
            _start_next_round(pool, config, timed_mode)
            st.rerun()
        st.divider()
        render_bias_dashboard(st.session_state.stats, st)
        return


def _start_next_round(pool: list, config: GameConfig, timed_mode: bool) -> None:
    """Clear result and start a new round from pool."""
    del st.session_state.pending_result
    sym, start_idx, regime = random.choice(pool)
    st.session_state.current_round = {
        "symbol": sym,
        "start_idx": start_idx,
        "regime": regime,
    }
    st.session_state.answer_submitted = False
    if timed_mode:
        st.session_state.timer_start = pd.Timestamp.now()


def submit_answer(
    prediction: str,
    df: pd.DataFrame,
    symbol: str,
    start_idx: int,
    regime: dict,
    config: GameConfig,
    trade_mode: bool,
    timed_mode: bool,
):
    """Process answer and store result."""
    end_idx = start_idx + config.window_candles
    exit_idx = end_idx + config.forward_candles - 1
    if exit_idx >= len(df):
        st.error("Insufficient data")
        return

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    entry_price = float(df.iloc[end_idx - 1]["close"])
    exit_price = float(df.iloc[exit_idx]["close"])
    actual = classify_outcome(entry_price, exit_price, config.threshold)
    correct = prediction == actual

    trade_return = 0.0
    trade_result = None
    if trade_mode:
        trade_result = simulate_trade(df, start_idx, prediction, config)
        if trade_result:
            trade_return = trade_result.trade_return

    regime_key = f"{regime.get('trend', '?')}/{regime.get('volatility', '?')}/{regime.get('prior_move', '?')}"

    stats = st.session_state.stats
    stats.record(
        prediction=prediction,
        actual=actual,
        correct=correct,
        trade_return=trade_return,
        regime_key=regime_key,
        was_timed=timed_mode,
    )

    st.session_state.answer_submitted = True
    st.session_state.pending_result = {
        "prediction": prediction,
        "actual": actual,
        "correct": correct,
        "trade_return": trade_return,
        "trade_result": trade_result,
        "was_timed": timed_mode,
        "timed_out": False,
    }
    st.rerun()


if __name__ == "__main__":
    main()
