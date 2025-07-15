import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta


def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # generate synthetic data for demonstration
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="30T")
        price = 1.10 + np.cumsum(np.random.randn(len(dates)) * 0.0001)
        df = pd.DataFrame({
            "Time": dates,
            "Open": price,
            "High": price + np.random.rand(len(dates)) * 0.0002,
            "Low": price - np.random.rand(len(dates)) * 0.0002,
            "Close": price + np.random.randn(len(dates)) * 0.00005,
        })
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def hover_breakout_backtest(
    df: pd.DataFrame,
    lookback: int = 10,
    range_threshold: float = 0.0005,
    tp_pips: float = 0.0010,
    sl_pips: float = 0.0005,
    hold_candles: int = 10,
    spread_pips: float = 0.0002,
    risk_percent: float = 0.01,
    starting_equity: float = 10000.0,
):
    # Support passing a single dictionary of parameters. This mirrors how
    # the function was invoked in earlier revisions of the script where a
    # params dict was supplied as the second argument. If `lookback` is a
    # dictionary we extract the values from it to avoid a type error.
    if isinstance(lookback, dict):
        params = lookback
        lookback = params.get("lookback", lookback)
        range_threshold = params.get("range_threshold", range_threshold)
        tp_pips = params.get("tp_pips", tp_pips)
        sl_pips = params.get("sl_pips", sl_pips)
        hold_candles = params.get("hold_candles", hold_candles)
        spread_pips = params.get("spread_pips", spread_pips)
        risk_percent = params.get("risk_percent", risk_percent)
        starting_equity = params.get("starting_equity", starting_equity)

    pip_unit = 0.0001
    risk_per_trade = starting_equity * risk_percent
    equity = starting_equity
    equity_curve = []
    trades = []

    for idx in range(lookback, len(df) - hold_candles):
        past_high = df['High'].iloc[idx - lookback:idx].max()
        past_low = df['Low'].iloc[idx - lookback:idx].min()
        if past_high - past_low > range_threshold:
            continue
        close = df['Close'].iloc[idx]
        direction = None
        if close > past_high:
            direction = 'long'
        elif close < past_low:
            direction = 'short'
        if direction is None:
            continue
        entry_price = close
        tp_price = entry_price + (tp_pips if direction == 'long' else -tp_pips)
        sl_price = entry_price - (sl_pips if direction == 'long' else -sl_pips)

        exit_price = df['Close'].iloc[idx + hold_candles]
        exit_time = df['Time'].iloc[idx + hold_candles]
        for j in range(1, hold_candles + 1):
            high = df['High'].iloc[idx + j]
            low = df['Low'].iloc[idx + j]
            if direction == 'long':
                if low <= sl_price:
                    exit_price = sl_price
                    exit_time = df['Time'].iloc[idx + j]
                    break
                if high >= tp_price:
                    exit_price = tp_price
                    exit_time = df['Time'].iloc[idx + j]
                    break
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_time = df['Time'].iloc[idx + j]
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    exit_time = df['Time'].iloc[idx + j]
                    break
        pip_diff = (exit_price - entry_price) / pip_unit
        if direction == 'short':
            pip_diff *= -1
        pip_diff -= spread_pips / pip_unit
        trade_profit = risk_per_trade * (pip_diff / (sl_pips / pip_unit))
        equity += trade_profit
        equity_curve.append((df['Time'].iloc[idx], equity))
        trades.append({
            'Time Open': df['Time'].iloc[idx],
            'Open Price': entry_price,
            'Time Close': exit_time,
            'Close Price': exit_price,
            'Take Profit Price': tp_price,
            'Stop Loss Price': sl_price,
            'Profit': trade_profit,
        })

    equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    if trades:
        trade_df = pd.DataFrame(trades)
    else:
        # Ensure all expected columns exist even when no trades were generated
        trade_df = pd.DataFrame(columns=[
            'Time Open',
            'Open Price',
            'Time Close',
            'Close Price',
            'Take Profit Price',
            'Stop Loss Price',
            'Profit',
        ])

    wins = trade_df[trade_df['Profit'] > 0]
    losses = trade_df[trade_df['Profit'] <= 0]
    total_trades = len(trade_df)
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    expectancy = trade_df['Profit'].mean() / risk_per_trade * 100 if total_trades else 0
    avg_win = wins['Profit'].mean() / starting_equity * 100 if len(wins) else 0
    avg_loss = losses['Profit'].mean() / starting_equity * 100 if len(losses) else 0

    # maximum drawdown
    peaks = equity_df['Equity'].cummax()
    drawdowns = (equity_df['Equity'] - peaks) / peaks
    max_drawdown = drawdowns.min() * 100

    stats = {
        'Total Trades': total_trades,
        'Win Rate %': win_rate,
        'Expectancy %': expectancy,
        'Average Win %': avg_win,
        'Average Loss %': avg_loss,
        'Max Drawdown %': max_drawdown,
        'Final Equity': equity,
    }

    return equity_df, trade_df, stats


def plot_equity_curve(equity_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df['Time'], equity_df['Equity'])
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.title('Equity Curve Over Time')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_pdf(output_pdf: str, params: dict, stats: dict, graph_path: str):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Hover Breakout Strategy Backtest")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, "Parameters:")
    y -= 15
    for k, v in params.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 15

    y -= 10
    c.drawString(50, y, "Results:")
    y -= 15
    for k, v in stats.items():
        c.drawString(70, y, f"{k}: {v:.2f}")
        y -= 15

    y -= 20
    c.drawImage(graph_path, 50, y - 250, width=500, height=250)
    c.showPage()
    c.save()


def main():
    data_path = 'EURUSD_M30_Data.csv'
    df = load_data(data_path)

    params = {
        'lookback': 6,
        'range_threshold': 0.004,
        'tp_pips': 0.0040,
        'sl_pips': 0.0009,
        'hold_candles': 12,
        'spread_pips': 0.0002,
        'risk_percent': 0.03,
        'starting_equity': 10000.0,
    }

    equity_df, trade_df, stats = hover_breakout_backtest(df, **params)

    graph_path = 'equity_curve.png'
    plot_equity_curve(equity_df, graph_path)

    trade_log_path = 'tradelog_hover_breakout_strategy.csv'
    trade_df.to_csv(trade_log_path, index=False)

    output_pdf = 'hover_breakout_results.pdf'
    save_pdf(output_pdf, params, stats, graph_path)

    print("Backtest complete.")
    print(stats)
    print(f"Trade log saved to {trade_log_path}")
    print(f"PDF report saved to {output_pdf}")


if __name__ == '__main__':
    main()