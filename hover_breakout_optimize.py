import itertools
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# reuse the backtest implementation
import hover_breakout_backtest as backtest_mod

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
    """Execute the hover breakout strategy using the backtest module."""
    return backtest_mod.backtest(
        back_candles=back_candles,
        range_pips=range_pips,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        future_candles=future_candles,
        generate_files=False,
    )


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
        metrics = run_strategy(
            back_candles=params['BACK_CANDLES'],
            range_pips=params['RANGE_PIPS'],
            tp_pips=params['TP_PIPS'],
            sl_pips=params['SL_PIPS'],
            future_candles=params['FUTURE_CANDLES'],
        )
        metrics['Params'] = params
        results.append(metrics)

    results.sort(key=lambda r: (-r['Final Equity'], -r['Total Trades'], r['Max Drawdown']))
    generate_report(PARAM_GRID, results, DEFAULT_PARAMS, 'hover_breakout_optimization.pdf')


if __name__ == '__main__':
    optimize()
