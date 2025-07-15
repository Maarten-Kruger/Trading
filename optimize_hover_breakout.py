import itertools
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from hover_breakout_strategy import load_data, hover_breakout_backtest


def save_optimization_pdf(pdf_path: str, grid: dict, defaults: dict, top10: pd.DataFrame):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Hover Breakout Strategy Optimization")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, "Default Parameters:")
    y -= 15
    for k, v in defaults.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 15

    y -= 10
    c.drawString(50, y, "Parameter Grid:")
    y -= 15
    for k, v in grid.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 15

    y -= 10
    c.drawString(50, y, "Top 10 Results:")
    y -= 15
    cols = ["lookback", "range_threshold", "tp_pips", "sl_pips", "hold_candles", "Final Equity", "Total Trades", "Max Drawdown %"]
    for _, row in top10.iterrows():
        line = ", ".join(f"{col}: {row[col]:.4f}" if isinstance(row[col], float) else f"{col}: {row[col]}" for col in cols)
        c.drawString(70, y, line)
        y -= 15
        if y < 60:
            c.showPage()
            y = height - 50
    c.save()


def optimize():
    data_path = "EURUSD_M30_Data.csv"
    df = load_data(data_path)

    defaults = {
        "lookback": 10,
        "range_threshold": 0.0005,
        "tp_pips": 0.0010,
        "sl_pips": 0.0005,
        "hold_candles": 10,
        "spread_pips": 0.0002,
        "risk_percent": 0.01,
        "starting_equity": 10000.0,
    }

    parameter_grid = {
        "lookback": [5, 10, 20, 50],
        "range_threshold": [0.0002, 0.0005, 0.001, 0.005],
        "tp_pips": [0.0005, 0.0010, 0.002, 0.01],
        "sl_pips": [0.0003, 0.0005, 0.001, 0.005],
        "hold_candles": [5, 10, 20, 50],
    }

    results = []
    for values in itertools.product(*parameter_grid.values()):
        params = dict(zip(parameter_grid.keys(), values))
        params.update({
            "spread_pips": defaults["spread_pips"],
            "risk_percent": defaults["risk_percent"],
            "starting_equity": defaults["starting_equity"],
        })
        equity_df, trade_df, stats = hover_breakout_backtest(df, **params)
        row = {**params, **stats}
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("optimization_results.csv", index=False)

    top10 = results_df.sort_values(by="Final Equity", ascending=False).head(10)

    save_optimization_pdf("hover_breakout_optimization.pdf", parameter_grid, defaults, top10)
    print("Optimization complete. Results saved to hover_breakout_optimization.pdf")


if __name__ == "__main__":
    optimize()
