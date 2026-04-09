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

    OPTIMIZATION GUIDELINES:
    - Avoid using `df.iloc[idx]` inside the loop, as creating Pandas Series row objects is very slow.
    - Instead, extract the required columns into NumPy arrays before the loop (e.g. `close_prices = df['Close'].values`)
      and index the arrays directly (`close_prices[idx]`).
    - Alternatively, use `df.at[idx, 'Column']` or `df.iat[row, col]` for fast scalar lookups.
    - If the backtest logic is highly complex, consider using Numba (`@njit`) to compile the backtest loop into C-speed machine code.
    """
    balance = config.STARTING_BALANCE
    equity_curve = [balance]
    timestamps = [df['Time'].iloc[0]]
    trades = []

    # Track the highest balance for drawdown calculation
    peak_balance = balance
    drawdowns = [0.0]

    # Pre-extract numpy arrays for fast access
    times = df['Time'].values
    closes = df['Close'].values
    spreads = df['Spread'].values
    df_len = len(df)

    for trigger_info in trigger_indices:
        # Extract integer index
        trigger_idx = trigger_info[0] if isinstance(trigger_info, tuple) else trigger_info

        # Example constraints: Make sure we have data to trade
        if trigger_idx >= df_len - 1:
            continue

        # Fast scalar access using numpy arrays instead of df.iloc
        entry_time = times[trigger_idx]
        entry_price = closes[trigger_idx]
        spread_points = spreads[trigger_idx]

        # Calculate risk amount
        risk_amount = balance * (config.RISK_PER_TRADE_PERCENT / 100.0)

        # ---------------------------------------------------------------------------------------------------
        # REQUIREMENT FOR CUSTOM STRATEGY CODE:
        # Place the real entry and exit rules here to see what happens when we run the simulation.
        #
        # For this template, we implement a dummy strategy that just holds for 1 candle
        # and has a simulated random PnL to allow the code to run out-of-the-box.
        # ---------------------------------------------------------------------------------------------------

        # Template dummy logic
        exit_idx = trigger_idx + 1
        exit_time = times[exit_idx]

        # Simulated trade logic (replace with real rules)
        import random
        # 50/50 win or lose risk amount
        pnl = risk_amount if random.random() > 0.5 else -risk_amount
        
        balance += pnl

        equity_curve.append(balance)
        timestamps.append(exit_time)
        trades.append({
            'Entry_Time': entry_time,
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
