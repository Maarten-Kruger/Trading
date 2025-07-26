import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PIP_VALUE = 10  # $10 per pip per standard lot


def load_trades(filename: str = 'tradelog.csv') -> pd.DataFrame:
    """Load trade log from CSV."""
    df = pd.read_csv(filename, parse_dates=['Time Open', 'Time Close'])
    df.sort_values('Time Close', inplace=True)
    return df


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
        pip_diff = row['Pip Difference']
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
        drawdowns = (peaks - curve) / starting_equity * 100
        max_drawdown = drawdowns.max()
    else:
        max_drawdown = 0.0

    stats = {
        'Total Trades': total_trades,
        'Partial Hits': partial_hits,
        'TP Hits': tp_hits,
        'SL Hits': sl_hits,
        'Win Rate': win_rate,
        'Average Win Size': avg_win,
        'Average Loss Size': avg_loss,
        'Max Drawdown': max_drawdown,
        'Final Equity': equity_curve[-1],
    }
    return stats, list(zip(trades['Time Close'].tolist(), equity_curve[1:]))


def create_pdf_report(stats: dict, filename: str = 'report.pdf') -> None:
    """Generate a simple PDF report."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(40, y, 'Backtest Results')
    y -= 20
    for key, val in stats.items():
        c.drawString(40, y, f'{key}: {val}')
        y -= 15
    c.save()


if __name__ == '__main__':
    trades = load_trades()
    s, _ = calculate_stats(trades)
    create_pdf_report(s)
