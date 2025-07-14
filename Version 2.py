import argparse
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import os

def load_data(path):
    """Read OHLC data CSV."""
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [c.strip().title() for c in df.columns]
    return df

def backtest(df, lookback, hover_range, tp, sl, max_hold, spread):
    """Run hover breakout backtest."""
    trades = []
    half_spread = spread / 2
    for i in range(lookback, len(df) - max_hold):
        window_high = df.loc[i - lookback:i - 1, "High"].max()
        window_low = df.loc[i - lookback:i - 1, "Low"].min()
        if window_high - window_low > hover_range:
            continue
        close = df.loc[i, "Close"]
        direction = 0
        if close > window_high:
            direction = 1
        elif close < window_low:
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

def compute_metrics(trades):
    """Return metrics for trade list."""
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
    equity = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "expectancy": expectancy,
        "kelly": kelly,
    }

def simulate_equity(trades, risk_pct, start_balance=10000):
    """Simulate account growth risking a fraction per trade."""
    eq = start_balance
    curve = [eq]
    for t in trades:
        stake = eq * risk_pct
        eq += t["pnl"] * stake
        curve.append(eq)
    return curve

def create_pdf(metrics, curve, output):
    """Generate PDF report with metrics and equity curve."""
    img = "_equity.png"
    plt.figure(figsize=(6,3))
    plt.plot(curve)
    plt.title("Demo Account Growth")
    plt.xlabel("Trade #")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig(img)
    plt.close()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output, pagesize=letter)
    elems = []
    elems.append(Paragraph("Hover Breakout Strategy Report", styles["Title"]))
    elems.append(Spacer(1,12))
    data = [["Metric", "Value", "Description"],
            ["total_trades", metrics["total_trades"], "Number of executed trades"],
            ["wins", metrics["wins"], "Trades closed in profit"],
            ["losses", metrics["losses"], "Trades closed with loss"],
            ["win_rate", f"{metrics['win_rate']*100:.2f}%", "Winning percentage"],
            ["max_drawdown", f"{metrics['max_drawdown']:.5f}", "Largest equity drop"],
            ["expectancy", f"{metrics['expectancy']:.5f}", "Avg result per trade"],
            ["kelly", f"{metrics['kelly']:.2f}", "Kelly fraction"],
           ]
    table = Table(data)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    elems.append(table)
    elems.append(Spacer(1,12))
    elems.append(Image(img, width=400, height=200))
    doc.build(elems)
    os.remove(img)

def main():
    parser = argparse.ArgumentParser(description="Hover Breakout backtest")
    parser.add_argument("--csv", default="EURUSD_M30_Data.csv")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--hover_range", type=float, default=0.004)
    parser.add_argument("--tp", type=float, default=0.003)
    parser.add_argument("--sl", type=float, default=0.0009)
    parser.add_argument("--max_hold", type=int, default=12)
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--risk_pct", type=float, default=0.05,
                        help="fraction of equity to risk per trade")
    parser.add_argument("--pdf", default="hover_report.pdf",
                        help="output PDF path")
    args = parser.parse_args()
    df = load_data(args.csv)
    trades = backtest(df, args.lookback, args.hover_range, args.tp,
                      args.sl, args.max_hold, args.spread)
    metrics = compute_metrics(trades)
    curve = simulate_equity(trades, args.risk_pct, 10000)
    create_pdf(metrics, curve, args.pdf)
    for k, v in metrics.items():
        if k == "win_rate":
            print(f"{k}: {v*100:.2f}%")
        else:
            print(f"{k}: {v}")
    print(f"PDF report saved to {args.pdf}")

if __name__ == "__main__":
    main()