import argparse
from itertools import product

from hover_backtest_pdf import (
    load_data,
    backtest,
    compute_metrics,
    simulate_equity,
    create_pdf,
)


def parse_args():
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser(
        description="Grid search optimizer for the Hover Breakout strategy"
    )
    parser.add_argument("--csv", default="EURUSD_M30_Data.csv")
    parser.add_argument("--lookback", nargs="+", type=int, default=[5, 6, 7, 8, 9, 10])
    parser.add_argument("--hover_range", nargs="+", type=float, default=[0.002, 0.003, 0.004])
    parser.add_argument("--tp", nargs="+", type=float, default=[0.002, 0.0025, 0.003])
    parser.add_argument("--sl", nargs="+", type=float, default=[0.0009, 0.0012, 0.0015])
    parser.add_argument("--max_hold", nargs="+", type=int, default=[8, 10, 12, 14])
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--risk_pct", type=float, default=0.01, help="equity risk per trade")
    parser.add_argument("--pdf", default="hover_optimize_report.pdf", help="output PDF report")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.csv)

    best_params = None
    best_metrics = None
    best_curve = None
    best_balance = None

    for combo in product(
        args.lookback, args.hover_range, args.tp, args.sl, args.max_hold
    ):
        lb, hr, tp_val, sl_val, mh = combo
        trades = backtest(df, lb, hr, tp_val, sl_val, mh, args.spread)
        metrics = compute_metrics(trades)
        curve = simulate_equity(trades, args.risk_pct, 10000)
        final_balance = curve[-1]
        if best_balance is None or final_balance > best_balance:
            best_balance = final_balance
            best_params = {
                "lookback": lb,
                "hover_range": hr,
                "tp": tp_val,
                "sl": sl_val,
                "max_hold": mh,
            }
            best_metrics = metrics
            best_curve = curve

    if best_params is None:
        print("No trades generated with given parameters")
        return

    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print("\nBest Metrics:")
    for k, v in best_metrics.items():
        if k == "win_rate":
            print(f"{k}: {v*100:.2f}%")
        else:
            print(f"{k}: {v}")

    print(f"Final balance: {best_balance:.2f}")

    create_pdf(best_metrics, best_curve, args.pdf)
    print(f"PDF report saved to {args.pdf}")


if __name__ == "__main__":
    main()
