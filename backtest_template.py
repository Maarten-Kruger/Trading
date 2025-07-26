import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def load_data(filename: str = 'data.csv') -> pd.DataFrame:
    """Load and sort price data from CSV."""
    df = pd.read_csv(filename, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)
    return df


def run_strategy(
    df: pd.DataFrame,
    strategy_func,
    *,
    starting_equity: float = 10000.0,
    risk_percent: float = 0.02,
    name: str = 'strategy',
):
    """Execute a trading strategy function and save the trades."""
    trade_log, equity_curve = strategy_func(
        df, starting_equity=starting_equity, risk_percent=risk_percent
    )
    pd.DataFrame(trade_log).to_csv(f"trades_{name}.csv", index=False)
    return trade_log, equity_curve


def calculate_stats(
    trade_log: list,
    equity_curve: list,
    *,
    starting_equity: float = 10000.0,
) -> dict:
    """Calculate statistics from a trade log and equity curve."""
    total_trades = len(trade_log)
    tp_hits = sum(1 for t in trade_log if t.get('Outcome') == 'tp')
    sl_hits = sum(1 for t in trade_log if t.get('Outcome') == 'sl')
    partials = sum(1 for t in trade_log if t.get('Outcome') == 'partial')
    wins = [t for t in trade_log if t['Profit/Loss'] > 0]
    profits = [t['Profit/Loss'] for t in trade_log]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    avg_win = (
        np.mean([p for p in profits if p > 0]) / starting_equity * 100
        if wins
        else 0
    )
    avg_loss = (
        np.mean([abs(p) for p in profits if p < 0]) / starting_equity * 100
        if len(profits) > len(wins)
        else 0
    )
    if equity_curve:
        eq_values = [e for _, e in equity_curve]
        peaks = np.maximum.accumulate(eq_values)
        drawdowns = 100 * (peaks - eq_values) / starting_equity
        max_drawdown = np.max(drawdowns)
    else:
        max_drawdown = 0
        eq_values = []
    return {
        'Total Trades': total_trades,
        'TP Hits': tp_hits,
        'SL Hits': sl_hits,
        'Partial Hits': partials,
        'Win Rate': win_rate,
        'Average Win Size': avg_win,
        'Average Loss Size': avg_loss,
        'Max Drawdown': max_drawdown,
        'Final Equity': eq_values[-1] if eq_values else starting_equity,
    }


def generate_report(
    name: str,
    stats: dict,
    trade_log: list,
    equity_curve: list,
):
    """Create graphs and compile a PDF summary."""
    if equity_curve:
        times = [t for t, _ in equity_curve]
        eq_values = [e for _, e in equity_curve]
        plt.figure(figsize=(10, 4))
        plt.plot(times, eq_values)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.tight_layout()
        plt.savefig(f'equity_curve_{name}.png')
        plt.close()
    else:
        plt.figure()
        plt.savefig(f'equity_curve_{name}.png')
        plt.close()

    c = canvas.Canvas(f'{name}_results.pdf', pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(40, y, f'{name} Strategy Results')
    y -= 20
    for key, val in stats.items():
        c.drawString(40, y, f'{key}: {val}')
        y -= 15
    y -= 20
    c.drawImage(f'equity_curve_{name}.png', 40, y - 300, width=500, height=300)
    c.save()
    pd.DataFrame(trade_log).to_csv(f'trades_{name}.csv', index=False)


if __name__ == '__main__':
    df = load_data('EURUSD_M30_Data.csv')

    def example_strategy(data, starting_equity=10000, risk_percent=0.02):
        # Placeholder strategy logic
        equity = starting_equity
        trade_log = []
        equity_curve = []
        # Implement strategy here
        return trade_log, equity_curve

    trades, curve = run_strategy(df, example_strategy, name='example')
    stats = calculate_stats(trades, curve)
    generate_report('example', stats, trades, curve)
