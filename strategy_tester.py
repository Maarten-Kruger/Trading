import pandas as pd
import matplotlib.pyplot as plt


def load_data(path="data.csv"):
    """Load OHLC data from CSV into a DataFrame."""
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [c.strip().title() for c in df.columns]
    return df


def example_strategy(df, spread=0.0002):
    """Simple placeholder strategy. Replace with your own logic.

    Goes long whenever today's close is greater than yesterday's close and
    exits at the next close. Returns a list of trade dictionaries with pnl in
    price units adjusted for spread.
    """
    trades = []
    for i in range(1, len(df) - 1):
        if df.loc[i, "Close"] > df.loc[i - 1, "Close"]:
            entry = df.loc[i + 1, "Open"] + spread / 2
            exit_ = df.loc[i + 1, "Close"] - spread / 2
            pnl = exit_ - entry
            trades.append({
                "entry_bar": i + 1,
                "exit_bar": i + 1,
                "entry_price": entry,
                "exit_price": exit_,
                "pnl": pnl,
            })
    return trades


def compute_metrics(trades, initial_balance=10000):
    """Calculate performance metrics for a list of trades."""
    pnls = [t["pnl"] for t in trades]
    total = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
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


def simulate_account(trades, kelly_fraction, start=10000):
    """Simulate account growth using Kelly sizing."""
    equity = start
    curve = [equity]
    for t in trades:
        stake = equity * kelly_fraction
        equity += t["pnl"] * stake
        curve.append(equity)
    return curve


def main():
    df = load_data("EURUSD_M30_Data.csv")
    trades = example_strategy(df)
    metrics = compute_metrics(trades)

    print("Total trades - number of executed trades:", metrics["total_trades"])
    print("Winning trades - trades with positive PnL:", metrics["wins"])
    print("Losing trades - trades with zero or negative PnL:", metrics["losses"])
    print("Win rate - percentage of wins:", round(metrics["win_rate"] * 100, 2), "%")
    print("Maximum drawdown - worst equity drop:", round(metrics["max_drawdown"], 5))
    print("Expectancy - average profit per trade:", round(metrics["expectancy"], 5))
    print("Kelly criterion - suggested risk fraction:", round(metrics["kelly"], 3))

    curve = simulate_account(trades, max(0, metrics["kelly"]))
    plt.figure(figsize=(8, 4))
    plt.plot(curve)
    plt.title("Demo Account Equity")
    plt.xlabel("Trade #")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    print("Equity curve saved to equity_curve.png")


if __name__ == "__main__":
    main()
