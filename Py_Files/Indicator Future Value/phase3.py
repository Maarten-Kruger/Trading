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

        # Place the real entry and exit rules here to see what happens when we run the simulation.
        

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
