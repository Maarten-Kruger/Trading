import csv
import itertools

# ============================================
# Hover Breakout Strategy Backtest (No pandas/numpy)
# ============================================
# Theory: When price hovers within a range for N bars, it often breaks out
# with high volatility. Enter on breakout and exit on TP/SL or after M bars.

def load_data(path):
    """
    Load OHLC data from CSV with columns: Date, Open, High, Low, Close, Volume.
    Returns a dict of lists: {'open', 'high', 'low', 'close'}.
    """
    opens, highs, lows, closes = [], [], [], []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            opens.append(float(row['Open']))
            highs.append(float(row['High']))
            lows.append(float(row['Low']))
            closes.append(float(row['Close']))
    return {'open': opens, 'high': highs, 'low': lows, 'close': closes}


def run_backtest(data, lookback, hover_range, tp, sl, max_hold, spread=0.0002):
    """
    Run the hover breakout backtest on raw price lists.

    Parameters:
        data        : dict with 'open', 'high', 'low', 'close' lists
        lookback    : bars to check hover
        hover_range : max price range during lookback
        tp          : take-profit distance (in price units)
        sl          : stop-loss distance
        max_hold    : max bars to hold before exit
        spread      : total bid-ask spread

    Returns:
        trades : list of trade dicts with entry, exit, pnl
    """
    trades = []
    opens = data['open']
    highs = data['high']
    lows  = data['low']
    closes = data['close']

    length = len(closes)
    for i in range(lookback, length - max_hold):
        window = closes[i-lookback:i]
        if max(window) - min(window) <= hover_range:
            cur_open = opens[i]
            half_spread = spread / 2

            # breakout up
            if closes[i] > max(window):
                entry_price = cur_open + half_spread
                target = entry_price + tp
                stop   = entry_price - sl
                direction = 1
            # breakout down
            elif closes[i] < min(window):
                entry_price = cur_open - half_spread
                target = entry_price - tp
                stop   = entry_price + sl
                direction = -1
            else:
                continue

            exit_price = None
            exit_bar = None
            # scan future bars for TP or SL
            for j in range(i+1, i+1+max_hold):
                if direction == 1:
                    if highs[j] >= target:
                        exit_price = target - half_spread
                        exit_bar = j
                        break
                    if lows[j] <= stop:
                        exit_price = stop - half_spread
                        exit_bar = j
                        break
                else:
                    if lows[j] <= target:
                        exit_price = target + half_spread
                        exit_bar = j
                        break
                    if highs[j] >= stop:
                        exit_price = stop + half_spread
                        exit_bar = j
                        break
            # if neither hit, exit at market after max_hold
            if exit_price is None:
                exit_price = closes[i+max_hold] - direction * half_spread
                exit_bar = i + max_hold

            pnl = (exit_price - entry_price) * direction
            trades.append({
                'entry_bar': i,
                'exit_bar': exit_bar,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl
            })
    return trades


def compute_metrics(trades, initial_equity=10000):
    """
    Compute performance metrics from trades list using pure Python.
    Returns a dict of metrics.
    """
    pnls = [t['pnl'] for t in trades]
    total_trades = len(pnls)
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x <= 0)
    win_rate = wins / total_trades if total_trades else 0
    net_profit = sum(pnls)

    # equity curve and max drawdown
    equity = initial_equity
    peak = initial_equity
    max_dd = 0
    for x in pnls:
        equity += x
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # average win/loss
    avg_win = sum(x for x in pnls if x > 0) / wins if wins else 0
    avg_loss = -sum(x for x in pnls if x <= 0) / losses if losses else 0
    rr = avg_win / avg_loss if avg_loss else 0
    expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss)
    kelly = win_rate - (1 - win_rate) / rr if rr else 0

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'net_profit': net_profit,
        'max_drawdown': max_dd,
        'expectancy': expectancy,
        'kelly': kelly
    }


def sensitivity_analysis(data, params_grid, top_n=5):
    """
    Grid search over parameter combinations without pandas.
    Returns a list of top_n result dicts sorted by net_profit.
    """
    results = []
    keys = list(params_grid.keys())
    for combo in itertools.product(*params_grid.values()):
        p = dict(zip(keys, combo))
        trades = run_backtest(data, **p)
        metrics = compute_metrics(trades)
        results.append({**p, **metrics})
    # sort and return top_n
    sorted_res = sorted(results, key=lambda x: x['net_profit'], reverse=True)
    return sorted_res[:top_n]


def write_html_report(metrics, top_sets, output_path="hover_backtest_report.html"):
    """Write metrics and top parameter sets to an HTML file."""
    rows_metrics = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in metrics.items()
    )

    if top_sets:
        headers = top_sets[0].keys()
        head_row = "".join(f"<th>{h}</th>" for h in headers)
        body_rows = []
        for row in top_sets:
            cells = "".join(f"<td>{row[h]}</td>" for h in headers)
            body_rows.append(f"<tr>{cells}</tr>")
        params_table = (
            f"<table><tr>{head_row}</tr>" + "".join(body_rows) + "</table>"
        )
    else:
        params_table = "<p>No parameter sets</p>"

    html = f"""
<html>
<head>
<title>Hover Breakout Backtest Report</title>
<style>
body {{font-family: Arial, sans-serif; margin: 40px;}}
h1 {{color: #333;}}
table {{border-collapse: collapse; width: 80%; margin-bottom: 20px;}}
th, td {{border: 1px solid #ccc; padding: 8px; text-align: center;}}
th {{background: #eee;}}
</style>
</head>
<body>
<h1>Hover Breakout Backtest Report</h1>
<h2>Metrics</h2>
<table>
{rows_metrics}
</table>
<h2>Top Parameter Sets</h2>
{params_table}
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


def main():
    data = load_data('EURUSD_M30_Data.csv')

    # example parameters
    params = {
        'lookback': 5,
        'hover_range': 0.004,
        'tp': 0.003,
        'sl': 0.0009,
        'max_hold': 12,
        'spread': 0.0002
    }
    trades = run_backtest(data, **params)
    metrics = compute_metrics(trades)

    print("Backtest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # sensitivity analysis
    grid = {
        'lookback': [5, 10, 15],
        'hover_range': [0.0010, 0.0020, 0.0030],
        'tp': [0.0020, 0.0040, 0.0060],
        'sl': [0.0010, 0.0020],
        'max_hold': [5, 10]
    }
    top_sets = sensitivity_analysis(data, grid)
    print("\nTop Parameter Sets by Net Profit:")
    for res in top_sets:
        print(res)

    # create html report
    write_html_report(metrics, top_sets)


if __name__ == '__main__':
    main()
