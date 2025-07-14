import itertools
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Default strategy parameters
STARTING_EQUITY = 10000
RISK_PERCENT = 0.01
SPREAD = 0.0002
DATA_FILE = 'EURUSD_M30_Data.csv'

DEFAULT_PARAMS = {
    'BACK_CANDLES': 5,
    'RANGE_PIPS': 10,
    'TP_PIPS': 20,
    'SL_PIPS': 10,
    'FUTURE_CANDLES': 3
}

# Parameter ranges to test
PARAM_GRID = {
    'BACK_CANDLES': [2, 5, 10],
    'RANGE_PIPS': [5, 10, 20, 40],
    'TP_PIPS': [10, 20, 40],
    'SL_PIPS': [5, 10, 20],
    'FUTURE_CANDLES': [2, 3, 5, 8]
}


def run_strategy(back_candles, range_pips, tp_pips, sl_pips, future_candles):
    """Execute the hover breakout strategy and return performance metrics."""
    df = pd.read_csv(DATA_FILE, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)

    equity = STARTING_EQUITY
    risk_amount = STARTING_EQUITY * RISK_PERCENT

    equity_curve = []
    trades = []

    for idx in range(back_candles, len(df) - future_candles):
        window = df.iloc[idx - back_candles:idx]
        if (window['High'].max() - window['Low'].min()) <= range_pips / 10000:
            current_close = df['Close'].iloc[idx]
            range_high = window['High'].max()
            range_low = window['Low'].min()

            direction = 0
            if current_close > range_high:
                direction = 1
            elif current_close < range_low:
                direction = -1
            if direction == 0:
                continue

            entry_price = current_close + direction * (SPREAD / 2)
            tp_price = entry_price + direction * tp_pips / 10000
            sl_price = entry_price - direction * sl_pips / 10000
            entry_time = df['Time'].iloc[idx]
            exit_price = df['Close'].iloc[idx + future_candles] - direction * (SPREAD / 2)
            exit_time = df['Time'].iloc[idx + future_candles]

            for j in range(1, future_candles + 1):
                high = df['High'].iloc[idx + j]
                low = df['Low'].iloc[idx + j]
                if direction == 1:
                    if high >= tp_price:
                        exit_price = tp_price
                        exit_time = df['Time'].iloc[idx + j]
                        break
                    if low <= sl_price:
                        exit_price = sl_price
                        exit_time = df['Time'].iloc[idx + j]
                        break
                else:
                    if low <= tp_price:
                        exit_price = tp_price
                        exit_time = df['Time'].iloc[idx + j]
                        break
                    if high >= sl_price:
                        exit_price = sl_price
                        exit_time = df['Time'].iloc[idx + j]
                        break

            pnl_pips = (exit_price - entry_price) * direction * 10000
            pnl_money = (pnl_pips / sl_pips) * risk_amount
            equity += pnl_money
            equity_curve.append(equity)
            trades.append(pnl_money)

    total_trades = len(trades)
    win_rate = (sum(1 for p in trades if p > 0) / total_trades * 100) if total_trades else 0
    expectancy = (np.mean(trades) / STARTING_EQUITY * 100) if trades else 0
    avg_win = (np.mean([p for p in trades if p > 0]) / STARTING_EQUITY * 100) if any(p > 0 for p in trades) else 0
    avg_loss = (np.mean([abs(p) for p in trades if p < 0]) / STARTING_EQUITY * 100) if any(p < 0 for p in trades) else 0

    if equity_curve:
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = 100 * (peaks - equity_curve) / STARTING_EQUITY
        max_drawdown = float(np.max(drawdowns))
    else:
        max_drawdown = 0.0

    return {
        'Final Equity': equity,
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Average Win Size': avg_win,
        'Average Loss Size': avg_loss,
        'Max Drawdown': max_drawdown,
    }


def generate_report(grid, results, defaults, pdf_name):
    """Create a summary PDF of the optimization run."""
    c = canvas.Canvas(pdf_name, pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(40, y, 'Hover Breakout Optimization Results')
    y -= 20

    c.drawString(40, y, 'Default Parameters:')
    y -= 15
    for k, v in defaults.items():
        c.drawString(60, y, f'{k} = {v}')
        y -= 15

    y -= 10
    c.drawString(40, y, 'Parameter Grid Tested:')
    y -= 15
    for k, vals in grid.items():
        c.drawString(60, y, f'{k}: {vals}')
        y -= 15

    y -= 10
    c.drawString(40, y, 'Top 10 Results:')
    y -= 15
    for i, res in enumerate(results[:10], 1):
        text = (f"{i}) Final Equity: {res['Final Equity']:.2f}, "
                f"Max DD: {res['Max Drawdown']:.2f}%, "
                f"Trades: {res['Total Trades']}")
        c.drawString(60, y, text)
        y -= 15
        param_str = ', '.join(f'{k}={v}' for k, v in res['Params'].items())
        c.drawString(80, y, param_str)
        y -= 15
        if y < 60:
            c.showPage()
            y = height - 40

    c.save()


def optimize():
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    results = []
    for values in combos:
        params = dict(zip(keys, values))
        metrics = run_strategy(**params)
        metrics['Params'] = params
        results.append(metrics)

    results.sort(key=lambda r: (-r['Final Equity'], -r['Total Trades'], r['Max Drawdown']))
    generate_report(PARAM_GRID, results, DEFAULT_PARAMS, 'hover_breakout_optimization.pdf')


if __name__ == '__main__':
    optimize()
