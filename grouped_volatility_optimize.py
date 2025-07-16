import itertools
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import grouped_volatility_backtest as backtest_mod

DEFAULT_PARAMS = {
    'BACK_CANDLES': backtest_mod.BACK_CANDLES,
    'CANDLE_SIZE_PIPS': backtest_mod.CANDLE_SIZE_PIPS,
    'TP_PIPS': backtest_mod.TP_PIPS,
    'SL_PIPS': backtest_mod.SL_PIPS,
    'FUTURE_CANDLES': backtest_mod.FUTURE_CANDLES,
    'FOLLOW_DIRECTION': backtest_mod.FOLLOW_DIRECTION,
}

# Parameter ranges including some extreme values
PARAM_GRID = {
    'BACK_CANDLES': [1, DEFAULT_PARAMS['BACK_CANDLES'], 10, 50],
    'CANDLE_SIZE_PIPS': [5, DEFAULT_PARAMS['CANDLE_SIZE_PIPS'], 25, 50],
    'TP_PIPS': [10, DEFAULT_PARAMS['TP_PIPS'], 50],
    'SL_PIPS': [5, DEFAULT_PARAMS['SL_PIPS'], 50],
    'FUTURE_CANDLES': [2, DEFAULT_PARAMS['FUTURE_CANDLES'], 50],
    'FOLLOW_DIRECTION': [True, False],
}

def run_strategy(back_candles, candle_size_pips, tp_pips, sl_pips, future_candles, follow_direction):
    """Execute the grouped volatility strategy using the backtest module."""
    return backtest_mod.backtest(
        back_candles=back_candles,
        candle_size_pips=candle_size_pips,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        future_candles=future_candles,
        follow_direction=follow_direction,
        generate_files=False,
    )

def generate_report(grid, results, defaults, pdf_name):
    """Create a summary PDF of the optimization run."""
    c = canvas.Canvas(pdf_name, pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(40, y, 'Grouped Volatility Optimization Results')
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
        text = (
            f"{i}) Final Equity: {res['Final Equity']:.2f}, "
            f"Max DD: {res['Max Drawdown']:.2f}%, "
            f"Trades: {res['Total Trades']}"
        )
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
            candle_size_pips=params['CANDLE_SIZE_PIPS'],
            tp_pips=params['TP_PIPS'],
            sl_pips=params['SL_PIPS'],
            future_candles=params['FUTURE_CANDLES'],
            follow_direction=params['FOLLOW_DIRECTION'],
        )
        metrics['Params'] = params
        results.append(metrics)

    results.sort(key=lambda r: (-r['Final Equity'], -r['Total Trades'], r['Max Drawdown']))
    generate_report(PARAM_GRID, results, DEFAULT_PARAMS, 'grouped_volatility_optimization.pdf')

if __name__ == '__main__':
    optimize()
