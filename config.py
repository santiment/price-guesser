"""Configuration for the crypto direction game."""

from dataclasses import dataclass, field


@dataclass
class GameConfig:
    """Game parameters - all configurable."""

    # Window & forward
    window_days: int = 60
    forward_days: int = 14
    threshold: float = 0.05

    # ADX regime thresholds
    adx_threshold_trending: float = 25.0
    adx_threshold_ranging: float = 20.0
    adx_period: int = 14

    # Volatility lookback (days) - 14-day realized vol
    volatility_days: int = 14

    # Timed mode
    timer_seconds: int = 5

    # Paths
    candles_dir: str = "candles_4h"

    def candles_per_day(self) -> int:
        """4h candles = 6 per day."""
        return 6

    @property
    def window_candles(self) -> int:
        return self.window_days * self.candles_per_day()

    @property
    def forward_candles(self) -> int:
        return self.forward_days * self.candles_per_day()


# Default config instance
DEFAULT_CONFIG = GameConfig()
