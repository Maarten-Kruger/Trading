import csv
import argparse
from typing import List, Dict


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
    risk: float,
    reward: float,
    spread: float,
    atr_mult: float | None = None,
) -> List[float]:
    trades: List[float] = []
    i = range_period
    while i < len(rows):
        recent_high = max(r["High"] for r in rows[i - range_period : i])
        recent_low = min(r["Low"] for r in rows[i - range_period : i])
        if threshold <= 0 and atr_mult is not None:
            avg_range = sum(r["High"] - r["Low"] for r in rows[i - range_period : i]) / range_period
            dynamic_threshold = avg_range * atr_mult
        else:
            dynamic_threshold = threshold
        if recent_high - recent_low <= dynamic_threshold:
            price = rows[i]['Close']
            if price > recent_high:
                entry = price + spread / 2
                stop = entry - risk
                take = entry + reward
                j = i + 1
                while j < len(rows):
                    if rows[j]['Low'] <= stop:
                        trades.append(-risk)
                        break
                    if rows[j]['High'] >= take:
                        trades.append(reward)
                        break
                    j += 1
                i = j
            elif price < recent_low:
                entry = price - spread / 2
                stop = entry + risk
                take = entry - reward
                j = i + 1
                while j < len(rows):
                    if rows[j]['High'] >= stop:
                        trades.append(-risk)
                        break
                    if rows[j]['Low'] <= take:
                        trades.append(reward)
                        break
                    j += 1
                i = j
            else:
                i += 1
        else:
            i += 1
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
    parser.add_argument('--csv', default='EURUSD_M30_Data.csv', help='CSV file path')
    parser.add_argument('--period', type=int, default=10, help='Lookback bars for range')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Maximum range width. If 0, use ATR-based threshold.'
    )
    parser.add_argument(
        '--atr-mult',
        type=float,
        default=2.0,
        help='Multiplier for ATR when threshold is 0'
    )
    parser.add_argument('--risk', type=float, default=0.0010, help='Stop loss in price units')
    parser.add_argument('--rr', type=float, default=2.0, help='Reward-to-risk ratio')
    parser.add_argument('--spread', type=float, default=0.0002, help='Spread in price units')
    args = parser.parse_args()

    rows = load_data(args.csv)
    trades = run_strategy(
        rows,
        args.period,
        args.threshold,
        args.risk,
        args.risk * args.rr,
        args.spread,
        args.atr_mult,
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
        args.period,
        args.threshold,
        args.risk,
        rr_values,
        args.spread,
        args.atr_mult,
    )
    print("\nSensitivity Analysis (RR multiplier -> Expectancy):")
    for res in sa_results:
        print(f"RR {res['rr_multiplier']}: {res['expectancy']:.5f}")

if __name__ == '__main__':
    main()
