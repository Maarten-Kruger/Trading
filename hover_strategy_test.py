import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Image,
    Spacer,
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import os


def load_data(path="EURUSD_M30_Data.csv"):
    """Load OHLC data from CSV."""
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [c.strip().title() for c in df.columns]
    return df


def backtest(df, lookback=5, hover_range=0.004, tp=0.003, sl=0.0009, max_hold=12, spread=0.0002):
    """Hover breakout backtest.

    A trade is opened when the previous *lookback* bars stay within
    *hover_range*. If the next bar closes outside this range we enter in
    the breakout direction. The trade exits on TP/SL or after *max_hold*
    bars. Spread is deducted from entry and exit prices.
    """
    trades = []
    half_spread = spread / 2

    for i in range(lookback, len(df) - max_hold):
        prior_high = df.loc[i - lookback:i - 1, "High"].max()
        prior_low = df.loc[i - lookback:i - 1, "Low"].min()

        if prior_high - prior_low > hover_range:
            continue

        close = df.loc[i, "Close"]
        direction = 0
        if close > prior_high:
            direction = 1
        elif close < prior_low:
            direction = -1
        else:
            continue

        entry = df.loc[i, "Open"] + half_spread * direction
        target = entry + tp * direction
        stop = entry - sl * direction

        exit_price = None
        exit_bar = None
        for j in range(i + 1, i + max_hold + 1):
            high = df.loc[j, "High"]
            low = df.loc[j, "Low"]
            if direction == 1:
                if high >= target:
                    exit_price = target - half_spread
                    exit_bar = j
                    break
                if low <= stop:
                    exit_price = stop - half_spread
                    exit_bar = j
                    break
            else:
                if low <= target:
                    exit_price = target + half_spread
                    exit_bar = j
                    break
                if high >= stop:
                    exit_price = stop + half_spread
                    exit_bar = j
                    break

        if exit_price is None:
            exit_price = df.loc[i + max_hold, "Close"] - half_spread * direction
            exit_bar = i + max_hold

        pnl = (exit_price - entry) * direction
        trades.append({
            "entry_bar": i,
            "exit_bar": exit_bar,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "pnl": pnl,
        })

    return trades


def compute_metrics(trades, initial_balance=10000):
    """Return basic performance metrics for the trade list."""
    pnls = [t["pnl"] for t in trades]
    total = len(pnls)
    wins = sum(p > 0 for p in pnls)
    losses = total - wins
    win_rate = wins / total if total else 0

    avg_win = sum(p for p in pnls if p > 0) / wins if wins else 0
    avg_loss = -sum(p for p in pnls if p <= 0) / losses if losses else 0
    rr = avg_win / avg_loss if avg_loss else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    kelly = win_rate - (1 - win_rate) / rr if rr else 0

    net_profit = sum(pnls)
    equity = initial_balance
    peak = equity
    max_dd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "max_drawdown": max_dd,
        "expectancy": expectancy,
        "kelly": kelly,
    }


def simulate_equity(trades, kelly_fraction, start_balance=10000):
    """Simulate account growth using the Kelly fraction for stake sizing."""
    eq = start_balance
    curve = [eq]
    for t in trades:
        stake = eq * max(0, kelly_fraction)
        eq += t["pnl"] * stake
        curve.append(eq)
    return curve


def _svg_line_chart(values, width=600, height=300, pad=10, max_points=300):
    """Return an SVG line chart string sampling long series for readability."""
    if not values:
        return "<svg></svg>"

    step = max(1, len(values) // max_points)
    sampled = values[::step]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])

    max_v = max(sampled)
    min_v = min(sampled)
    x_scale = (width - 2 * pad) / (len(sampled) - 1) if len(sampled) > 1 else 1
    y_scale = (height - 2 * pad) / (max_v - min_v) if max_v != min_v else 1
    points = []
    for i, v in enumerate(sampled):
        x = pad + i * x_scale
        y = height - pad - (v - min_v) * y_scale
        points.append(f"{x:.2f},{y:.2f}")
    pts = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="blue" stroke-width="2" points="{pts}"/>'
        '</svg>'
    )


def write_html_report(metrics, equity_curve, explanations, output_path="hover_strategy_report.html"):
    """Write metrics and equity curve to a standalone HTML report."""
    metrics_rows = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td><td>{explanations.get(k, '')}</td></tr>" for k, v in metrics.items()
    )

    df_curve = pd.DataFrame({"Balance": equity_curve})
    table_curve = df_curve.to_html(index=True)

    svg = _svg_line_chart(equity_curve)

    html = f"""
<html>
<head>
<title>Hover Breakout Strategy Report</title>
<style>
body {{font-family: Arial, sans-serif; margin: 40px;}}
h1 {{color: #333;}}
table {{border-collapse: collapse; width: 80%; margin-bottom: 20px;}}
th, td {{border: 1px solid #ccc; padding: 8px; text-align: center;}}
th {{background: #eee;}}
</style>
</head>
<body>
<h1>Hover Breakout Strategy Report</h1>
<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Description</th></tr>
{metrics_rows}
</table>
<h2>Equity Curve</h2>
{svg}
<h2>Balance Over Time</h2>
{table_curve}
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


def write_pdf_report(metrics, equity_curve, explanations, output_path="hover_strategy_report.pdf"):
    """Write metrics and equity curve to a PDF file using ReportLab."""
    # Save equity curve plot to a temporary image
    img_path = "_equity_curve.png"
    plt.figure(figsize=(6, 3))
    plt.plot(equity_curve)
    plt.title("Demo Account Growth")
    plt.xlabel("Trade #")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    elements.append(Paragraph("Hover Breakout Strategy Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    data = [["Metric", "Value", "Description"]]
    for k, v in metrics.items():
        val = f"{v*100:.2f}%" if k == "win_rate" else v
        data.append([k, val, explanations.get(k, "")])

    table = Table(data)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Image(img_path, width=400, height=200))

    doc.build(elements)
    if os.path.exists(img_path):
        os.remove(img_path)
    print(f"PDF report saved to {output_path}")


def main():
    df = load_data()
    trades = backtest(df)
    metrics = compute_metrics(trades)

    explanations = {
        "total_trades": "Number of executed trades",
        "wins": "Trades closed in profit",
        "losses": "Trades closed at a loss",
        "win_rate": "Winning trades / total",
        "max_drawdown": "Largest equity drop",
        "expectancy": "Average pnl per trade",
        "kelly": "Risk fraction suggested by Kelly",
    }

    for k, exp in explanations.items():
        val = metrics.get(k, 0)
        if k == "win_rate":
            val = f"{val*100:.2f}%"
        print(f"{k}: {val} - {exp}")

    eq_curve = simulate_equity(trades, metrics["kelly"], 10000)
    write_html_report(metrics, eq_curve, explanations)
    write_pdf_report(metrics, eq_curve, explanations)
    plt.figure(figsize=(8, 4))
    plt.plot(eq_curve)
    plt.title("Demo Account Growth")
    plt.xlabel("Trade #")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
