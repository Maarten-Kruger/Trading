import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from phase1 import load_and_preprocess_data, find_triggers

def run_backtest(df, trigger_indices):
    """
    Template module for backtesting. Runs a standardized simulation using
    parameters defined in config.py.
    """
    balance = config.STARTING_BALANCE
    equity_curve = [balance]
    timestamps = [df['Time'].iloc[0]]
    trades = []

    # Track the highest balance for drawdown calculation
    peak_balance = balance
    drawdowns = [0.0]

    for trigger_idx in trigger_indices:
        # Example constraints: Make sure we have data to trade
        if trigger_idx >= len(df) - 1:
            continue

        trigger_row = df.iloc[trigger_idx]
        entry_price = trigger_row['Close']
        spread_points = trigger_row['Spread']

        # Calculate risk amount
        risk_amount = balance * (config.RISK_PER_TRADE_PERCENT / 100.0)

        # --- DUMMY LOGIC START (Remove for empty template) ---
        # Dummy Entry Logic: Always buy at the close of the trigger candle
        # Dummy Stop Loss: 50 points
        # Dummy Take Profit: 100 points
        stop_loss_points = 50
        take_profit_points = 100

        # Account for spread on entry (buy at ask)
        entry_price_with_spread = entry_price + (spread_points * config.POINT_VALUE)

        # Calculate position size based on risk and stop loss
        # Note: In real Forex, points (like spread) and pip/point value calculations
        # differ heavily by asset class. This dummy math uses standard integer points.
        if stop_loss_points > 0 and config.POINT_VALUE > 0:
            position_size = risk_amount / (stop_loss_points * config.POINT_VALUE)
        else:
            position_size = 0.01 # Fallback

        # Simulate trade outcome (simplified for dummy purposes)
        # Look forward to see if TP or SL hits first
        trade_result_pips = 0
        exit_time = None

        for i in range(trigger_idx + 1, len(df)):
            future_row = df.iloc[i]
            # Check Stop Loss (Low goes below entry - SL)
            # Rough conversion of points to price (Assuming 1 point = 0.00001 for 5-digit broker Forex)
            # This is highly simplified dummy logic.
            point_value_approx = 0.00001

            sl_price = entry_price_with_spread - (stop_loss_points * point_value_approx)
            tp_price = entry_price_with_spread + (take_profit_points * point_value_approx)

            if future_row['Low'] <= sl_price:
                trade_result_pips = -stop_loss_points
                exit_time = future_row['Time']
                break
            elif future_row['High'] >= tp_price:
                trade_result_pips = take_profit_points
                exit_time = future_row['Time']
                break

        # If trade never closed before end of dataset
        if exit_time is None:
            continue

        # Calculate PnL
        pnl = trade_result_pips * position_size * config.POINT_VALUE
        balance += pnl
        # --- DUMMY LOGIC END ---

        equity_curve.append(balance)
        timestamps.append(exit_time)
        trades.append({
            'Entry_Time': trigger_row['Time'],
            'Exit_Time': exit_time,
            'PnL': pnl,
            'Balance': balance
        })

        # Calculate Drawdown
        if balance > peak_balance:
            peak_balance = balance
        drawdown_pct = ((peak_balance - balance) / peak_balance) * 100
        drawdowns.append(drawdown_pct)

    return timestamps, equity_curve, drawdowns, trades

def generate_simulation_report(timestamps, equity_curve, drawdowns, file_name):
    """
    Generates the final summary image with Equity Curve and Drawdowns.
    """
    strategy_name = config.STRATEGY_NAME
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    output_image = f"{strategy_name}_{base_file_name}_Simulation.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Equity Curve
    ax1.plot(timestamps, equity_curve, color='blue', label='Balance')
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Balance ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Drawdown Bar Graph
    # We negate drawdowns to show them pointing downwards
    ax2.bar(timestamps, [-d for d in drawdowns], color='red', label='Drawdown (%)', width=0.05 if len(timestamps) < 50 else 0.5)
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'Backtest Simulation: {strategy_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_image, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return output_image
