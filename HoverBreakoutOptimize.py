import argparse
from itertools import product
from Hover_Breakout_Test import load_data, run_backtest, compute_metrics


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


if __name__ == '__main__':
    main()
