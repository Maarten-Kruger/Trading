import itertools
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from main_optimized import (
    load_data,
    simulate_strategy,
    calculate_metrics,
    LOOKBACK,
    RANGE_THRESHOLD_PIPS,
    STOP_LOSS_PIPS,
    TAKE_PROFIT_PIPS,
    HOLD_PERIOD,
    SPREAD_PIPS,
    RISK_PER_TRADE,
    INITIAL_EQUITY,
    BARS_PER_CHECK,
)

DEFAULT_PARAMS = {
    'Lookback': LOOKBACK,
    'Range Threshold (pips)': RANGE_THRESHOLD_PIPS,
    'Stop Loss (pips)': STOP_LOSS_PIPS,
    'Take Profit (pips)': TAKE_PROFIT_PIPS,
    'Hold Period (bars)': HOLD_PERIOD,
    'Spread (pips)': SPREAD_PIPS,
    'Risk Per Trade': RISK_PER_TRADE,
    'Initial Equity': INITIAL_EQUITY,
}

PARAM_GRID = {
    'Lookback':           [4,   8,   12],
    'Range Threshold':    [20,  60,  100],   # in pips
    'Stop Loss':          [5,   10,  20],    # in pips
    'Take Profit':        [20,  40,  80],    # in pips
    'Hold Period':        [6,   12,  24],    # bars
    'Risk Per Trade':     [0.005, 0.01, 0.02] # fraction of equity
}


def generate_opt_report(grid, results, defaults, output_pdf):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    elements = []
    elements.append(Paragraph('Hover Breakout Optimization', styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph('Default Parameters', styles['Heading2']))
    for k, v in defaults.items():
        elements.append(Paragraph(f"{k}: {v}", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph('Parameter Grid Tested', styles['Heading2']))
    for k, vals in grid.items():
        elements.append(Paragraph(f"{k}: {vals}", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph('Top 50 Results', styles['Heading2']))
    for i, res in enumerate(results[:50], 1):
        eq = res['Final Equity']
        dd = res['Max Drawdown'] * 100
        exp = res['Expectancy'] * 100
        param_str = ', '.join(f"{k}: {v}" for k, v in res['Params'].items())
        text = f"{i}) Final Equity: {eq:.2f}, Max DD: {dd:.2f}%, Expectancy: {exp:.2f}%"
        elements.append(Paragraph(text, styles['Normal']))
        elements.append(Paragraph(param_str, styles['Normal']))
        elements.append(Spacer(1, 6))

    doc.build(elements)


def main():
    df = load_data('EURUSD_M1_Data.csv')
    combos = list(itertools.product(
        PARAM_GRID['Lookback'],
        PARAM_GRID['Range Threshold (pips)'],
        PARAM_GRID['Stop Loss (pips)'],
        PARAM_GRID['Take Profit (pips)'],
        PARAM_GRID['Hold Period (bars)'],
        PARAM_GRID['Risk Per Trade']
    ))

    results = []
    for params in combos:
        lookback, rng, sl, tp, hold, risk = params
        trade_df, eq_curve, _ = simulate_strategy(
            df,
            lookback,
            rng,
            sl,
            tp,
            hold,
            SPREAD_PIPS,
            risk,
            INITIAL_EQUITY,
            BARS_PER_CHECK,
        )
        metrics = calculate_metrics(trade_df, eq_curve)
        results.append({
            'Final Equity': eq_curve[-1],
            'Max Drawdown': metrics['Max Drawdown'],
            'Expectancy': metrics['Expectancy'],
            'Params': {
                'Lookback': lookback,
                'Range Threshold (pips)': rng,
                'Stop Loss (pips)': sl,
                'Take Profit (pips)': tp,
                'Hold Period (bars)': hold,
                'Spread (pips)': SPREAD_PIPS,
                'Risk Per Trade': risk,
                'Initial Equity': INITIAL_EQUITY,
            }
        })

    results.sort(key=lambda x: (-x['Final Equity'], x['Max Drawdown']))
    generate_opt_report(PARAM_GRID, results, DEFAULT_PARAMS, 'Hover_Breakout_Optimization.pdf')


if __name__ == '__main__':
    main()
