import csv
import argparse
from typing import List, Dict

# Default parameters that can be edited directly in the code. Command line
# arguments override these values if provided.
DEFAULT_CONFIG = {
    'csv': 'EURUSD_M30_Data.csv',  # path to the OHLC data
    'period': 6,                  # lookback bars defining the consolidation range
    'threshold': 0.0038,              # fixed range width. 0 -> derive from ATR
    'atr_mult': 2.0,               # ATR multiplier when threshold is 0
    'duration': 12,                 # forward bars to check TP/SL
    'risk': 0.0009,                # stop loss distance in price units
    'rr': 2.777777,                     # reward-to-risk ratio
    'spread': 0.0002,              # bid/ask spread in price units
}


def load_data(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"])
            })
    return rows


def run_strategy(
    rows: List[Dict[str, float]],
    range_period: int,
    threshold: float,
    duration: int,
    risk: float,
    reward: float,
    spread: float,
    atr_mult: float | None = None,
) -> List[float]:
    """Run breakout strategy with fixed forward duration.

    A trade is opened at the close after a consolidation period. The last
    `range_period` candles must fit within `threshold` (or ATR-based value).
    The next `duration` candles are then checked for a take profit or stop
    loss. If neither is hit, the position is closed for the partial move.
    """

    trades: List[float] = []
    for i in range(range_period, len(rows) - duration):
        high_back = max(r["High"] for r in rows[i - range_period : i])
        low_back = min(r["Low"] for r in rows[i - range_period : i])
        if threshold <= 0 and atr_mult is not None:
            avg_range = sum(r["High"] - r["Low"] for r in rows[i - range_period : i]) / range_period
            effective_thr = avg_range * atr_mult
        else:
            effective_thr = threshold

        if high_back - low_back > effective_thr:
            continue

        entry_price = rows[i]["Close"]
        mid = (high_back + low_back) / 2
        long_trade = entry_price >= mid

        if long_trade:
            entry = entry_price + spread / 2
            tp = entry + reward
            sl = entry - risk
            for j in range(1, duration + 1):
                bar = rows[i + j]
                if bar["Low"] <= sl:
                    trades.append(-risk)
                    break
                if bar["High"] >= tp:
                    trades.append(reward)
                    break
            else:
                trades.append(rows[i + duration]["Close"] - entry)
        else:
            entry = entry_price - spread / 2
            tp = entry - reward
            sl = entry + risk
            for j in range(1, duration + 1):
                bar = rows[i + j]
                if bar["High"] >= sl:
                    trades.append(-risk)
                    break
                if bar["Low"] <= tp:
                    trades.append(reward)
                    break
            else:
                trades.append(entry - rows[i + duration]["Close"])
    return trades


def analyze_trades(trades: List[float]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    result['wins'] = len(wins)
    result['losses'] = len(losses)
    total = len(wins) + len(losses)
    result['win_rate'] = len(wins) / total if total else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    result['avg_win'] = avg_win
    result['avg_loss'] = avg_loss
    result['risk_reward'] = (avg_win / avg_loss) if avg_loss else 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    result['max_drawdown'] = max_dd
    result['expectancy'] = result['win_rate'] * avg_win - (1 - result['win_rate']) * avg_loss
    r = (avg_win / avg_loss) if avg_loss else 0.0
    result['kelly'] = result['win_rate'] - (1 - result['win_rate']) / r if r else 0.0
    return result


def sensitivity_analysis(
    rows: List[Dict[str, float]],
    range_period: int,
    threshold: float,
    duration: int,
    risk: float,
    rr_values: List[float],
    spread: float,
    atr_mult: float | None,
) -> List[Dict[str, float]]:
    results = []
    for rr in rr_values:
        trades = run_strategy(
            rows,
            range_period,
            threshold,
            duration,
            risk,
            risk * rr,
            spread,
            atr_mult,
        )
        metrics = analyze_trades(trades)
        metrics["rr_multiplier"] = rr
        results.append(metrics)
    return results


def main():
    parser = argparse.ArgumentParser(description="Test range breakout strategy")
    parser.add_argument('--csv', default=DEFAULT_CONFIG['csv'], help='CSV file path')
    parser.add_argument('--period', type=int, default=DEFAULT_CONFIG['period'], help='Lookback bars for range')
    parser.add_argument('--duration', type=int, default=DEFAULT_CONFIG['duration'], help='Forward bars to evaluate TP/SL')
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_CONFIG['threshold'],
        help='Maximum range width. If 0, use ATR-based threshold.'
    )
    parser.add_argument(
        '--atr-mult',
        type=float,
        default=DEFAULT_CONFIG['atr_mult'],
        help='Multiplier for ATR when threshold is 0'
    )
    parser.add_argument('--risk', type=float, default=DEFAULT_CONFIG['risk'], help='Stop loss in price units')
    parser.add_argument('--rr', type=float, default=DEFAULT_CONFIG['rr'], help='Reward-to-risk ratio')
    parser.add_argument('--spread', type=float, default=DEFAULT_CONFIG['spread'], help='Spread in price units')
    args = parser.parse_args()

    # Merge CLI arguments with the default configuration so parameters can be
    # edited in this file while remaining overridable from the command line.
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))

    rows = load_data(config['csv'])
    trades = run_strategy(
        rows,
        config['period'],
        config['threshold'],
        config['duration'],
        config['risk'],
        config['risk'] * config['rr'],
        config['spread'],
        config['atr_mult'],
    )
    metrics = analyze_trades(trades)
    print(f"Total trades: {metrics['wins'] + metrics['losses']}")
    print(f"Winning trades: {metrics['wins']}")
    print(f"Losing trades: {metrics['losses']}")
    print(f"Win rate: {metrics['win_rate'] * 100:.2f}%")
    print(f"Average win: {metrics['avg_win']:.5f}")
    print(f"Average loss: {metrics['avg_loss']:.5f}")
    print(f"Risk-Reward ratio: {metrics['risk_reward']:.2f}")
    print(f"Maximum drawdown: {metrics['max_drawdown']:.5f}")
    print(f"Expectancy: {metrics['expectancy']:.5f}")
    print(f"Kelly criterion: {metrics['kelly']:.2f}")

    rr_values = [1.0, 1.5, 2.0, 3.0]
    sa_results = sensitivity_analysis(
        rows,
        config['period'],
        config['threshold'],
        config['duration'],
        config['risk'],
        rr_values,
        config['spread'],
        config['atr_mult'],
    )
    print("\nSensitivity Analysis (RR multiplier -> Expectancy):")
    for res in sa_results:
        print(f"RR {res['rr_multiplier']}: {res['expectancy']:.5f}")

if __name__ == '__main__':
    main()
