import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PIP_VALUE = 10  # $10 per pip per standard lot


def load_trades(filename: str = 'tradelog.csv') -> pd.DataFrame:
    """Load trade log from CSV."""
    df = pd.read_csv(filename, parse_dates=['Time Open', 'Time Close'])
    df.sort_values('Time Close', inplace=True)
    return df


def load_params(filename: str = 'strategy_params.json') -> dict:
    """Load strategy parameters from a JSON file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def calculate_stats(
    trades: pd.DataFrame,
    *,
    starting_equity: float = 10000.0,
    risk_factor: float = 0.01,
    leverage: float = 1.0,
):
    """Calculate metrics and simulate account equity."""
    equity = starting_equity
    equity_curve = [equity]
    tp_hits = 0
    sl_hits = 0
    partial_hits = 0
    wins = []
    losses = []

    for _, row in trades.iterrows():
        pip_diff = row['Pip PnL']
        status = row['Status']
        if status == 'tp':
            tp_hits += 1
        elif status == 'sl':
            sl_hits += 1
        elif status == 'partial':
            partial_hits += 1

        trade_size = equity * risk_factor
        lot_size = (trade_size * leverage) / 100000
        pnl = PIP_VALUE * lot_size * pip_diff
        equity += pnl
        equity_curve.append(equity)
        if pnl >= 0:
            wins.append(pnl)
        else:
            losses.append(pnl)

    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    avg_win = np.mean(wins) / starting_equity * 100 if wins else 0
    avg_loss = abs(np.mean(losses)) / starting_equity * 100 if losses else 0

    curve = np.array(equity_curve)
    if len(curve) > 1:
        peaks = np.maximum.accumulate(curve)
        drawdowns = (peaks - curve) / peaks * 100
        max_drawdown = drawdowns.max()
    else:
        max_drawdown = 0.0

    stats = {
        'Total Trades': total_trades,
        'Partial Hits': partial_hits,
        'TP Hits': tp_hits,
        'SL Hits': sl_hits,
        'Win Rate': round(win_rate, 2),
        'Average Win Size': round(avg_win, 2),
        'Average Loss Size': round(avg_loss, 2),
        'Max Drawdown': round(max_drawdown, 2),
        'Final Equity': round(equity_curve[-1], 2),
    }
    return stats, list(zip(trades['Time Close'].tolist(), equity_curve[1:]))


def create_pdf_report(
    stats: dict,
    equity_curve: list,
    params: dict,
    filename: str = 'report.pdf',
) -> None:
    """Generate a simple PDF report with equity graph."""
    # Save equity curve plot
    if equity_curve:
        times, values = zip(*equity_curve)
        plt.figure(figsize=(6, 3))
        plt.plot(times, values, label='Equity')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.title('Equity Curve')
        plt.tight_layout()
        plt.savefig('equity_curve.png')
        plt.close()

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(40, y, 'Backtest Results')
    y -= 20
    for key, val in params.items():
        c.drawString(40, y, f'{key}: {val}')
        y -= 15
    y -= 10
    for key, val in stats.items():
        c.drawString(40, y, f'{key}: {val}')
        y -= 15
    if equity_curve:
        y -= 20
        c.drawImage('equity_curve.png', 40, y - 200, width=500, height=200)
    c.save()


if __name__ == '__main__':
    trades = load_trades()
    params = {
        'Starting Equity': 10000.0,
        'Risk Factor': 0.01,
        'Leverage': 1.0,
    }
    params.update(load_params())
    stats, curve = calculate_stats(
        trades,
        starting_equity=params['Starting Equity'],
        risk_factor=params['Risk Factor'],
        leverage=params['Leverage'],
    )
    create_pdf_report(stats, curve, params)
