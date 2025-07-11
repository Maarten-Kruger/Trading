"""Optimized main entry point for running the hover breakout strategy.

This script executes the simulation without generating any plots or PDF
reports. It's intended for quick runs where only the raw metrics are
needed.
"""

from hover_breakout_test import (
    load_data,
    simulate_strategy,
    calculate_metrics,
    LOOKBACK,
    RANGE_THRESHOLD_PIPS,
    STOP_LOSS_PIPS,
    TAKE_PROFIT_PIPS,
    HOLD_PERIOD,
    SPREAD_PIPS,
    RISK_PER_TRADE,
    INITIAL_EQUITY,
    BARS_PER_CHECK,
)


def main():
    """Run the hover breakout strategy without generating graphics."""
    df = load_data('EURUSD_M1_Data.csv')
    trade_df, equity_curve, times = simulate_strategy(
        df,
        LOOKBACK,
        RANGE_THRESHOLD_PIPS,
        STOP_LOSS_PIPS,
        TAKE_PROFIT_PIPS,
        HOLD_PERIOD,
        SPREAD_PIPS,
        RISK_PER_TRADE,
        INITIAL_EQUITY,
        BARS_PER_CHECK,
    )
    metrics = calculate_metrics(trade_df, equity_curve)
    print(metrics)


if __name__ == '__main__':
    main()
