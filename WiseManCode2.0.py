"""
WiseManCode2.0
-----------------
Python adaptation of the "Three Wise Men" strategy described in the repository.
The script focuses on structure and core logic, allowing further integration with
backtesting engines or live trading platforms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional
import math


@dataclass
class Candle:
    """Basic OHLC candle representation."""

    time: datetime
    open: float
    high: float
    low: float
    close: float

    @property
    def median(self) -> float:
        """Return the candle's median price used for backlog."""
        return (self.high + self.low) / 2.0


@dataclass
class Trade:
    """Store information about an individual trade."""

    direction: int  # 1 for buy, -1 for sell
    entry: float
    stop_loss: float
    take_profit: float
    risk: float
    opened_bar: int
    stage: int = 0  # 0 original,1 50%,2 25%,3 breakeven
    closed: bool = False
    result: float = 0.0
    close_bar: Optional[int] = None


class WiseManStrategy:
    """Implementation skeleton of the Three Wise Men trading logic."""

    def __init__(
        self,
        avg_period: int = 5,
        backlog_size: int = 20,
        n_range: int = 2,
        percent: float = 0.5,
        wave_tolerance: float = 5.0,
        trades_checking: int = 10,
        atr_period: int = 14,
        stop_atr: List[float] = [2.0, 1.5, 1.0],
        rr: List[float] = [2.0, 1.5, 1.0],
        half_sl_bars: int = 5,
        tclose_bars: int = 30,
        risk_percent: float = 1.0,
    ) -> None:
        self.avg_period = avg_period
        self.backlog_size = backlog_size
        self.n_range = n_range
        self.percent = percent
        self.wave_tolerance = wave_tolerance
        self.trades_checking = trades_checking
        self.atr_period = atr_period
        self.stop_atr = stop_atr
        self.rr = rr
        self.half_sl_bars = half_sl_bars
        self.tclose_bars = tclose_bars
        self.risk_percent = risk_percent / 100.0

        self.backlog: Deque[float] = deque(maxlen=backlog_size)
        self.wave_active: bool = False
        self.wave_up: bool = False
        self.trade_bars_left: int = 0
        self.wave_points: List[Optional[float]] = [None] * 9  # index 0..8
        self.trades: List[Trade] = []

        # statistics for optimization
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_trades: int = 0
        self.total_profit: float = 0.0
        self.equity_curve: List[float] = []
        self.start_equity: float = 0.0

    # ------------------------------------------------------------------
    # Data and wave detection helpers
    # ------------------------------------------------------------------
    def on_new_candle(self, candle: Candle) -> None:
        """Main entry executed at the open of each new candle."""

        if self.start_time is None:
            self.start_time = candle.time
            self.start_equity = 1.0  # placeholder for equity tracking
            self.equity_curve = [self.start_equity]

        self.backlog.append(candle.median)

        if len(self.backlog) < self.backlog_size:
            return

        if not self.wave_active:
            self._detect_wave()
        else:
            self._check_for_trades(candle)

    def _detect_wave(self) -> None:
        """Detect if current backlog resembles the synthetic example wave."""

        example = self._create_example_wave(B=1.0, S=0.0001, B1=-1.0, B2=1.0)
        p = self._pearson_corr(example)
        d = self._mae_first_diff(example)

        pearson_threshold = 0.8  # configurable
        mae_threshold = 0.5  # configurable

        if abs(p) >= pearson_threshold and abs(d) <= mae_threshold:
            self.wave_active = True
            self.wave_up = p > 0
            self.trade_bars_left = self.trades_checking
            self._prepare_wave_points(example)

    def _create_example_wave(self, B: float, S: float, B1: float, B2: float) -> List[float]:
        """Generate example wave as described in the specification."""

        N = self.backlog_size
        dx = (B2 - B1) / float(N - 1)
        example = []
        for i in range(N):
            x = B1 + i * dx
            example.append(S * (x ** 3 - B * x))
        offset = self.backlog[0] - example[0]
        return [e + offset for e in example]

    def _pearson_corr(self, example: List[float]) -> float:
        """Compute Pearson correlation between example and backlog."""

        x_mean = sum(example) / len(example)
        y_mean = sum(self.backlog) / len(self.backlog)
        num = 0.0
        denom_x = 0.0
        denom_y = 0.0
        for ex, ba in zip(example, self.backlog):
            dx = ex - x_mean
            dy = ba - y_mean
            num += dx * dy
            denom_x += dx * dx
            denom_y += dy * dy
        denom = math.sqrt(denom_x) * math.sqrt(denom_y)
        if denom == 0:
            return 0.0
        return num / denom

    def _mae_first_diff(self, example: List[float]) -> float:
        """Mean absolute error on first differences."""
        diffs = []
        for i in range(1, len(example)):
            delta_e = example[i] - example[i - 1]
            delta_b = self.backlog[i] - self.backlog[i - 1]
            diffs.append(abs(delta_e - delta_b))
        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)

    # ------------------------------------------------------------------
    # Trade setup and management
    # ------------------------------------------------------------------
    def _prepare_wave_points(self, example: List[float]) -> None:
        """Initialize wave points. Point 3 is the local extremum in example."""
        idx = 1
        if self.wave_up:
            # local minimum
            for i in range(1, len(example) - 1):
                if example[i - 1] > example[i] < example[i + 1]:
                    idx = i
                    break
        else:
            for i in range(1, len(example) - 1):
                if example[i - 1] < example[i] > example[i + 1]:
                    idx = i
                    break
        self.wave_points = [None] * 9
        self.wave_points[3] = self.backlog[idx]

    def _check_for_trades(self, candle: Candle) -> None:
        """Scan for wise man points and manage open trades."""
        self.trade_bars_left -= 1

        price = candle.median

        # Update wave points
        self._update_points(price)

        # Manage open trades and open new ones when points trigger
        self._manage_trades(candle)

        if self.trade_bars_left <= 0 and not self.trades:
            self.wave_active = False

    def _update_points(self, price: float) -> None:
        """Update points 5..8 sequentially based on price movement."""
        if self.wave_points[5] is None:
            if (self.wave_up and (self.wave_points[5] is None or price > self.wave_points[5])) or (
                not self.wave_up and (self.wave_points[5] is None or price < self.wave_points[5])
            ):
                self.wave_points[5] = price
        elif self.wave_points[6] is None:
            p3 = self.wave_points[3]
            p5 = self.wave_points[5]
            target = p3 + (p5 - p3) * self.percent
            if (self.wave_up and price <= target) or (not self.wave_up and price >= target):
                self.wave_points[6] = price
                self._open_trade(0, price)
        elif self.wave_points[7] is None:
            if (self.wave_up and price > self.wave_points[6]) or (
                not self.wave_up and price < self.wave_points[6]
            ):
                self.wave_points[7] = price
                self._open_trade(1, price)
        elif self.wave_points[8] is None:
            if (self.wave_up and abs(price - self.wave_points[5]) <= self.wave_tolerance) or (
                not self.wave_up and abs(price - self.wave_points[5]) <= self.wave_tolerance
            ):
                self.wave_points[8] = price
                self._open_trade(2, price)

    def _open_trade(self, idx: int, price: float) -> None:
        """Open a wise man trade using ATR-based stop and RR-based target."""
        atr = self._get_atr()
        p3 = self.wave_points[3]
        direction = 1 if self.wave_up else -1
        stop = p3 - direction * self.stop_atr[idx] * atr
        take = price + direction * self.rr[idx] * (price - stop)

        risk = self.risk_percent  # risk as fraction of equity, placeholder
        trade = Trade(direction, price, stop, take, risk, opened_bar=self.trade_bars_left)
        self.trades.append(trade)
        self.total_trades += 1

    def _get_atr(self) -> float:
        """Placeholder ATR computation using backlog data."""
        if len(self.backlog) < 2:
            return 0.0
        true_ranges = [abs(self.backlog[i] - self.backlog[i - 1]) for i in range(1, len(self.backlog))]
        return sum(true_ranges[-self.atr_period :]) / min(len(true_ranges), self.atr_period)

    def _manage_trades(self, candle: Candle) -> None:
        """Update stop-loss levels and close trades based on price."""
        price = candle.median
        for trade in list(self.trades):
            # simulate stop-loss / take-profit hit
            if (trade.direction == 1 and price <= trade.stop_loss) or (
                trade.direction == -1 and price >= trade.stop_loss
            ):
                trade.closed = True
                trade.result = -trade.risk
            elif (trade.direction == 1 and price >= trade.take_profit) or (
                trade.direction == -1 and price <= trade.take_profit
            ):
                trade.closed = True
                trade.result = trade.risk * self.rr[trade.stage if trade.stage < 3 else 2]

            # stop-loss management in stages
            bars_open = trade.opened_bar - self.trade_bars_left
            if not trade.closed and bars_open % self.half_sl_bars == 0 and trade.stage < 3:
                # move stop towards entry
                move = (trade.entry - trade.stop_loss) * 0.5
                trade.stop_loss += move
                trade.stage += 1

            if not trade.closed and bars_open >= self.tclose_bars:
                trade.closed = True
                trade.result = 0.0

            if trade.closed:
                self.total_profit += trade.result
                self.trades.remove(trade)

    # ------------------------------------------------------------------
    # Optimization objective
    # ------------------------------------------------------------------
    def optimization_score(self, weights: List[float]) -> float:
        """Custom optimization score as described in the specification."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        total_bars = int((self.end_time - self.start_time).total_seconds())  # placeholder
        months = max(1, (self.end_time.year - self.start_time.year) * 12 + self.end_time.month - self.start_time.month)
        T = self.total_trades / max(1, total_bars)
        P = (self.total_profit / max(1.0, self.start_equity)) / months
        D = self._max_drawdown() / 100.0
        Wt, Wp, Wd = weights
        score = (T * Wt + P * Wp - D * Wd) * 100.0
        return score

    def _max_drawdown(self) -> float:
        """Compute max drawdown from equity curve."""
        peak = self.equity_curve[0]
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
        return max_dd


if __name__ == "__main__":
    # Simple usage demonstration
    strategy = WiseManStrategy()
    # The user should feed candle data through on_new_candle within a backtest loop.
    # Example:
    # for candle in historical_data:
    #     strategy.on_new_candle(candle)
    pass
