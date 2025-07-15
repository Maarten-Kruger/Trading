import pandas as pd
import itertools
from hover_breakout_test import backtest, compute_stats, PARAMS, START_EQUITY
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def optimize(df, param_ranges):
    results = []
    keys = list(param_ranges.keys())
    for values in itertools.product(*param_ranges.values()):
        params = PARAMS.copy()
        params.update(dict(zip(keys, values)))
        trades, times, equity = backtest(df, params)
        stats = compute_stats(trades)
        profit = equity[-1] - START_EQUITY if equity else 0
        results.append({
            'lookback_candles': params['lookback_candles'],
            'range_pips': params['range_pips'],
            'distance_pips': params['distance_pips'],
            'rr': params['rr'],
            'lookahead_candles': params['lookahead_candles'],
            'Profit': profit,
            'Trades': len(trades),
            'Max Drawdown': stats['Max Drawdown']
        })
    return pd.DataFrame(results)


def create_pdf(grid, defaults, top_df, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    c.setFont('Helvetica-Bold', 14)
    c.drawString(50, y, 'Hover Breakout Optimization')
    y -= 20
    c.setFont('Helvetica', 10)
    c.drawString(50, y, 'Default Parameters:')
    y -= 15
    for k, v in defaults.items():
        c.drawString(60, y, f'{k}: {v}')
        y -= 12
    y -= 10
    c.drawString(50, y, 'Parameter Grid Tested:')
    y -= 15
    for k, v in grid.items():
        c.drawString(60, y, f'{k}: {v}')
        y -= 12
    y -= 10
    c.drawString(50, y, 'Top 10 Results:')
    y -= 15
    headers = ['lookback', 'range', 'distance', 'rr', 'lookahead', 'profit', 'trades', 'max DD']
    c.drawString(50, y, ' | '.join(f'{h:>8}' for h in headers))
    y -= 12
    for _, row in top_df.iterrows():
        row_str = f"{int(row['lookback_candles']):>8} | {int(row['range_pips']):>8} | {int(row['distance_pips']):>8} | {row['rr']:>8} | {int(row['lookahead_candles']):>8} | {row['Profit']:>8.2f} | {int(row['Trades']):>8} | {row['Max Drawdown']:>7.2f}"
        c.drawString(50, y, row_str)
        y -= 12
        if y < 50:
            c.showPage()
            y = 750
    c.showPage()
    c.save()


def main():
    df = pd.read_csv('EURUSD_M30_Data.csv')
    grid = {
        'lookback_candles': [5, 10, 20, 50],
        'range_pips': [2, 5, 10, 20],
        'distance_pips': [5, 10, 20, 50],
        'rr': [1, 2, 3, 5],
        'lookahead_candles': [5, 8, 15, 30]
    }
    results = optimize(df, grid)
    results.to_csv('hover_breakout_optimization_results.csv', index=False)
    top10 = results.sort_values('Profit', ascending=False).head(10)
    create_pdf(grid, PARAMS, top10, 'hover_breakout_optimization.pdf')


if __name__ == '__main__':
    main()
