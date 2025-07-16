import itertools
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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
    'BACK_CANDLES': [1, DEFAULT_PARAMS['BACK_CANDLES'], 3],
    'CANDLE_SIZE_PIPS': [14, DEFAULT_PARAMS['CANDLE_SIZE_PIPS'], 16],
    'TP_PIPS': [7, DEFAULT_PARAMS['TP_PIPS'], 9],
    'SL_PIPS': [2, DEFAULT_PARAMS['SL_PIPS'], 4],
    'FUTURE_CANDLES': [10, DEFAULT_PARAMS['FUTURE_CANDLES'], 14],
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
    """Create a neat PDF summarizing the optimization run."""
    doc = SimpleDocTemplate(pdf_name, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph('Grouped Volatility Optimization Results', styles['Title']))
    elements.append(Spacer(1, 12))

    # Defaults section
    elements.append(Paragraph('Default Parameters', styles['Heading2']))
    defaults_table = [['Parameter', 'Value']] + [[k, v] for k, v in defaults.items()]
    t = Table(defaults_table, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # Grid section
    elements.append(Paragraph('Parameter Grid Tested', styles['Heading2']))
    grid_table = [['Parameter', 'Values']] + [[k, ', '.join(map(str, v))] for k, v in grid.items()]
    t = Table(grid_table, hAlign='LEFT', colWidths=[120, 360])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # Results section
    elements.append(Paragraph('Top 10 Results', styles['Heading2']))
    results_table = [['#', 'Final Equity', 'Max DD', 'Trades', 'Parameters']]
    for i, res in enumerate(results[:10], 1):
        params_str = ', '.join(f'{k}={v}' for k, v in res['Params'].items())
        params_para = Paragraph(params_str, styles['Normal'])
        results_table.append([
            i,
            f"{res['Final Equity']:.2f}",
            f"{res['Max Drawdown']:.2f}%",
            res['Total Trades'],
            params_para,
        ])

    t = Table(results_table, hAlign='LEFT', colWidths=[30, 80, 80, 60, 230])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 1), (3, -1), 'RIGHT'),
    ]))
    elements.append(t)

    doc.build(elements)

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
