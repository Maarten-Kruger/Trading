import itertools
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from hover_breakout_backtester import load_market_data, hover_breakout_strategy
from stats_calculator import calculate_stats

DEFAULT_PARAMS = {
    "Back Candles": 10,
    "Range Pips": 8,
    "TP Pips": 12,
    "SL Pips": 20,
    "Future Candles": 12,
    "Spread": 0.0002,
}

# Wider parameter grid for thorough search
PARAM_GRID = {
    "Back Candles": [5, 10, 20, 50],
    "Range Pips": [4, 8, 12, 20],
    "TP Pips": [8, 12, 20, 40],
    "SL Pips": [10, 20, 40, 80],
    "Future Candles": [6, 12, 24, 48],
}



def run_test(df: pd.DataFrame, params: dict) -> dict:
    """Run backtest for a single parameter combination and return metrics."""
    trades = hover_breakout_strategy(
        df,
        back_candles=params["Back Candles"],
        range_pips=params["Range Pips"],
        tp_pips=params["TP Pips"],
        sl_pips=params["SL Pips"],
        future_candles=params["Future Candles"],
        spread=DEFAULT_PARAMS["Spread"],
        save_log=False,
    )
    stats, _ = calculate_stats(trades)
    return {
        "Params": params,
        "Final Equity": stats["Final Equity"],
        "Total Trades": stats["Total Trades"],
        "Max Drawdown": stats["Max Drawdown"],
    }


def generate_pdf(results: list) -> None:
    """Create summary PDF with grid, defaults and top results."""
    doc = SimpleDocTemplate("hover_breakout_optimization.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Hover Breakout Optimization Results", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Default Parameters", styles["Heading2"]))
    defaults_tbl = Table([[k, v] for k, v in DEFAULT_PARAMS.items()])
    defaults_tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ]))
    elements.append(defaults_tbl)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Parameter Grid Tested", styles["Heading2"]))
    grid_data = [["Parameter", "Values"]]
    for key, vals in PARAM_GRID.items():
        grid_data.append([key, str(vals)])
    grid_tbl = Table(grid_data)
    grid_tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ]))
    elements.append(grid_tbl)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Top 10 Results", styles["Heading2"]))
    header = ["Final Equity", "Max DD", "Trades", "Parameters"]
    table_data = [header]
    for res in results[:10]:
        param_str = ", ".join(f"{k}={v}" for k, v in res["Params"].items())
        table_data.append([
            f"{res['Final Equity']:.2f}",
            f"{res['Max Drawdown']:.2f}%",
            str(res['Total Trades']),
            param_str,
        ])
    result_tbl = Table(table_data, colWidths=[80, 60, 60, 260])
    result_tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    elements.append(result_tbl)

    doc.build(elements)


def optimize() -> None:
    df = load_market_data()
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    results = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        metrics = run_test(df, params)
        results.append(metrics)

    results.sort(key=lambda r: (-r["Final Equity"], -r["Total Trades"], r["Max Drawdown"]))
    generate_pdf(results)

    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    optimize()
