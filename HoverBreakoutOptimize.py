import argparse
from itertools import product
from Hover_Breakout_Test import load_data, run_backtest, compute_metrics


def write_html_report(best_params, best_metrics, output_path="hover_optimize_report.html"):
    """Write optimization results to an HTML file."""
    params_rows = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in best_params.items()
    )
    metrics_rows = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in best_metrics.items()
    )

    html = f"""
<html>
<head>
<title>Hover Breakout Optimization Report</title>
<style>
body {{font-family: Arial, sans-serif; margin: 40px;}}
h1 {{color: #333;}}
table {{border-collapse: collapse; width: 60%; margin-bottom: 20px;}}
th, td {{border: 1px solid #ccc; padding: 8px; text-align: center;}}
th {{background: #eee;}}
</style>
</head>
<body>
<h1>Hover Breakout Optimization Report</h1>
<h2>Best Parameters</h2>
<table>
{params_rows}
</table>
<h2>Metrics</h2>
<table>
{metrics_rows}
</table>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search for hover breakout parameters")
    parser.add_argument('--data', default='EURUSD_M30_Data.csv', help='CSV file with OHLC data')
    parser.add_argument('--lookback', nargs='+', type=int, default=[5, 6, 7, 8, 9, 10])
    parser.add_argument('--hover_range', nargs='+', type=float, default=[0.0020, 0.0030, 0.0040])
    parser.add_argument('--tp', nargs='+', type=float, default=[0.0020, 0.0025, 0.0030])
    parser.add_argument('--sl', nargs='+', type=float, default=[0.0009, 0.0012, 0.0015])
    parser.add_argument('--max_hold', nargs='+', type=int, default=[8, 10, 12, 14])
    parser.add_argument('--spread', nargs='+', type=float, default=[0.0002, 0.0003])
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args.data)

    keys = ['lookback', 'hover_range', 'tp', 'sl', 'max_hold', 'spread']
    grid = {k: getattr(args, k) for k in keys}

    best_params = None
    best_metrics = None
    for combo in product(*grid.values()):
        params = dict(zip(keys, combo))
        trades = run_backtest(data, **params)
        metrics = compute_metrics(trades)
        if best_metrics is None or metrics['net_profit'] > best_metrics['net_profit']:
            best_params = params
            best_metrics = metrics

    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print("\nMetrics:")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")

    write_html_report(best_params, best_metrics)


if __name__ == '__main__':
    main()
