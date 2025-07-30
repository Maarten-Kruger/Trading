import itertools
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from grouped_volatility_backtester import load_market_data, grouped_volatility_strategy
from stats_calculator import calculate_stats

DEFAULT_PARAMS = {
    "Back Candles": 20,
    "Candle Size Pips": 30,
    "TP Pips": 30,
    "SL Pips": 30,
    "Future Candles": 20,
    "Follow Direction": True,
    "Spread": 0.0002,
}

# Parameter grid mixes sensible and extreme values
PARAM_GRID = {
    "Back Candles": [25, 30],             # centered on 20
    "Candle Size Pips": [20, 25],         # centered on 30
    "TP Pips": [45, 50],                  # centered on 40
    "SL Pips": [25],                      # centered on 25
    "Future Candles": [35, 40],           # centered on 30
    "Follow Direction": [False]           # keep fixed to the optimal value
}

def run_test(df: pd.DataFrame, params: dict) -> dict:
    """Run backtest for a single parameter combination and return metrics."""
    trades = grouped_volatility_strategy(
        df,
        back_candles=params["Back Candles"],
        candle_size_pips=params["Candle Size Pips"],
        tp_pips=params["TP Pips"],
        sl_pips=params["SL Pips"],
        future_candles=params["Future Candles"],
        follow_direction=params["Follow Direction"],
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
    """Create a PDF summarizing defaults, parameter grid and top results."""
    doc = SimpleDocTemplate("grouped_volatility_optimization.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Grouped Volatility Optimization Results", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Default Parameters", styles["Heading2"]))
    defaults_table = [[k, v] for k, v in DEFAULT_PARAMS.items()]
    t = Table(defaults_table)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Parameter Grid Tested", styles["Heading2"]))
    grid_data = [["Parameter", "Values"]]
    for key, vals in PARAM_GRID.items():
        grid_data.append([key, str(vals)])
    grid_tbl = Table(grid_data, colWidths=[120, 360])
    grid_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(grid_tbl)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Top 10 Results", styles["Heading2"]))
    res_table = [["#", "Final Equity", "Max DD", "Trades", "Parameters"]]
    for i, res in enumerate(results[:10], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in res["Params"].items())
        res_table.append([
            i,
            f"{res['Final Equity']:.2f}",
            f"{res['Max Drawdown']:.2f}%",
            res['Total Trades'],
            Paragraph(param_str, styles["Normal"]),
        ])
    res_tbl = Table(res_table, colWidths=[30, 80, 80, 60, 260])
    res_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 1), (3, -1), "RIGHT"),
    ]))
    elements.append(res_tbl)

    doc.build(elements)


def optimize() -> None:
    df = load_market_data()
    keys = list(PARAM_GRID.keys())
    combos = itertools.product(*[PARAM_GRID[k] for k in keys])

    results = []
    for values in combos:
        params = dict(zip(keys, values))
        metrics = run_test(df, params)
        results.append(metrics)

    results.sort(key=lambda r: (-r["Final Equity"], -r["Total Trades"], r["Max Drawdown"]))
    generate_pdf(results)

    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    optimize()
