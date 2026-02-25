"""Bias & behavior analytics for the crypto direction game."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stats_tracker import SessionStats

if TYPE_CHECKING:
    import streamlit as st


def render_bias_dashboard(stats: SessionStats, st_module) -> None:
    """Render the bias & behavior dashboard section."""
    st_module.subheader("Bias & Behavior Dashboard")

    if stats.total_rounds == 0:
        st_module.info("Complete some rounds to see bias analytics.")
        return

    col1, col2, col3 = st_module.columns(3)

    with col1:
        st_module.markdown("**Prediction Bias**")
        total = stats.total_rounds
        pred_dist = stats.prediction_distribution
        up_pct = 100 * pred_dist.get("UP", 0) / total if total else 0
        down_pct = 100 * pred_dist.get("DOWN", 0) / total if total else 0
        same_pct = 100 * pred_dist.get("SAME", 0) / total if total else 0
        st_module.metric("UP %", f"{up_pct:.1f}%")
        st_module.metric("DOWN %", f"{down_pct:.1f}%")
        st_module.metric("SAME %", f"{same_pct:.1f}%")

    with col2:
        st_module.markdown("**Streak Sensitivity**")
        st_module.metric("Accuracy after 2+ wins", f"{stats.streak_after_wins_accuracy:.1%}")
        st_module.metric("Accuracy after 2+ losses", f"{stats.streak_after_losses_accuracy:.1%}")

    with col3:
        st_module.markdown("**Regime Performance**")
        if stats.regime_performance:
            for regime_key, perf in list(stats.regime_performance.items())[:5]:
                acc = perf["correct"] / perf["total"] if perf["total"] else 0
                st_module.metric(regime_key, f"{acc:.1%}")
        else:
            st_module.caption("No regime data yet")

    st_module.markdown("**Trade Behavior**")
    tb1, tb2, tb3 = st_module.columns(3)
    tb1.metric("Avg return after UP", f"{stats.avg_return_after_up:.2%}")
    tb2.metric("Avg return after DOWN", f"{stats.avg_return_after_down:.2%}")
    tb3.metric("Avg return after SAME", f"{stats.avg_return_after_same:.2%}")
