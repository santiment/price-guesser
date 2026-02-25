"""Session statistics tracker for the crypto direction game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trade_engine import compute_max_drawdown, compute_sharpe

Prediction = Literal["UP", "DOWN", "SAME"]


@dataclass
class SessionStats:
    """Extended session statistics."""

    total_rounds: int = 0
    correct: int = 0
    accuracy: float = 0.0
    class_accuracy: dict = field(default_factory=lambda: {"UP": 0.0, "DOWN": 0.0, "SAME": 0.0})
    confusion_matrix: dict = field(default_factory=lambda: {
        "UP": {"UP": 0, "DOWN": 0, "SAME": 0},
        "DOWN": {"UP": 0, "DOWN": 0, "SAME": 0},
        "SAME": {"UP": 0, "DOWN": 0, "SAME": 0},
    })
    current_streak: int = 0
    best_streak: int = 0
    current_losing_streak: int = 0

    # Trade simulation
    cumulative_pnl: float = 0.0
    trade_returns: list[float] = field(default_factory=list)

    # Regime performance: {regime_key: {correct, total}}
    regime_performance: dict = field(default_factory=dict)

    # Prediction distribution: {UP, DOWN, SAME} counts
    prediction_distribution: dict = field(default_factory=lambda: {"UP": 0, "DOWN": 0, "SAME": 0})

    # Streak sensitivity
    streak_after_wins_accuracy: float = 0.0
    streak_after_losses_accuracy: float = 0.0

    # Timed mode
    timed_correct: int = 0
    timed_total: int = 0
    untimed_correct: int = 0
    untimed_total: int = 0

    # Trade behavior: avg return by prediction
    avg_return_after_up: float = 0.0
    avg_return_after_down: float = 0.0
    avg_return_after_same: float = 0.0
    _return_sum_up: float = 0.0
    _return_sum_down: float = 0.0
    _return_sum_same: float = 0.0

    # Streak tracking for sensitivity
    _streak_after_wins_correct: int = 0
    _streak_after_wins_total: int = 0
    _streak_after_losses_correct: int = 0
    _streak_after_losses_total: int = 0

    # Win/loss stats
    wins: int = 0
    losses: int = 0
    total_win_amount: float = 0.0
    total_loss_amount: float = 0.0

    def record(
        self,
        prediction: Prediction | None,
        actual: Prediction,
        correct: bool,
        trade_return: float = 0.0,
        regime_key: str | None = None,
        was_timed: bool = False,
        was_streak_after_wins: bool | None = None,  # auto-computed from current state
        was_streak_after_losses: bool | None = None,
    ) -> None:
        """Record a round result."""
        self.total_rounds += 1
        was_streak_after_wins = self.current_streak >= 2
        was_streak_after_losses = self.current_losing_streak >= 2

        if correct:
            self.correct += 1
            self.current_streak += 1
            self.current_losing_streak = 0
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0
            self.current_losing_streak += 1

        if trade_return > 0:
            self.wins += 1
            self.total_win_amount += trade_return
        elif trade_return < 0:
            self.losses += 1
            self.total_loss_amount += abs(trade_return)

        self.accuracy = self.correct / self.total_rounds if self.total_rounds else 0.0

        # Confusion matrix (prediction -> actual)
        if prediction is not None:
            self.confusion_matrix[prediction][actual] += 1
            self.prediction_distribution[prediction] = self.prediction_distribution.get(prediction, 0) + 1

        # Class accuracy (for each actual class)
        for p in ["UP", "DOWN", "SAME"]:
            total_p = sum(self.confusion_matrix[k][p] for k in ["UP", "DOWN", "SAME"])
            correct_p = self.confusion_matrix[p][p]
            self.class_accuracy[p] = correct_p / total_p if total_p else 0.0

        # Trade
        self.cumulative_pnl += trade_return
        self.trade_returns.append(trade_return)

        # Regime
        if regime_key:
            if regime_key not in self.regime_performance:
                self.regime_performance[regime_key] = {"correct": 0, "total": 0}
            self.regime_performance[regime_key]["total"] += 1
            if correct:
                self.regime_performance[regime_key]["correct"] += 1

        # Timed
        if was_timed:
            self.timed_total += 1
            if correct:
                self.timed_correct += 1
        else:
            self.untimed_total += 1
            if correct:
                self.untimed_correct += 1

        # Streak sensitivity (computed from state before this round)
        if was_streak_after_wins:
            self._streak_after_wins_total += 1
            if correct:
                self._streak_after_wins_correct += 1
        if was_streak_after_losses:
            self._streak_after_losses_total += 1
            if correct:
                self._streak_after_losses_correct += 1

        # Trade behavior
        if prediction == "UP":
            self._return_sum_up += trade_return
            n = self.prediction_distribution.get("UP", 0)
            self.avg_return_after_up = self._return_sum_up / n if n else 0.0
        elif prediction == "DOWN":
            self._return_sum_down += trade_return
            n = self.prediction_distribution.get("DOWN", 0)
            self.avg_return_after_down = self._return_sum_down / n if n else 0.0
        elif prediction == "SAME":
            self._return_sum_same += trade_return
            n = self.prediction_distribution.get("SAME", 0)
            self.avg_return_after_same = self._return_sum_same / n if n else 0.0

        # Recompute streak accuracies
        self.streak_after_wins_accuracy = (
            self._streak_after_wins_correct / self._streak_after_wins_total
            if self._streak_after_wins_total else 0.0
        )
        self.streak_after_losses_accuracy = (
            self._streak_after_losses_correct / self._streak_after_losses_total
            if self._streak_after_losses_total else 0.0
        )

    @property
    def max_drawdown(self) -> float:
        """Max drawdown from equity curve."""
        if not self.trade_returns:
            return 0.0
        equity = []
        cum = 1.0
        for r in self.trade_returns:
            cum *= 1 + r
            equity.append(cum)
        return compute_max_drawdown(equity)

    @property
    def sharpe_ratio(self) -> float:
        """Session Sharpe ratio."""
        return compute_sharpe(self.trade_returns)

    @property
    def win_rate(self) -> float:
        """Win rate (trades with non-zero return: wins / (wins + losses))."""
        total_trades = self.wins + self.losses
        if total_trades == 0:
            return 0.0
        return self.wins / total_trades

    @property
    def avg_win(self) -> float:
        """Average winning trade return."""
        return self.total_win_amount / self.wins if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade return (as positive number)."""
        return self.total_loss_amount / self.losses if self.losses else 0.0

    @property
    def equity_curve(self) -> list[float]:
        """Cumulative equity curve (1 + cumulative return)."""
        out = [1.0]
        for r in self.trade_returns:
            out.append(out[-1] * (1 + r))
        return out
