import csv
import argparse
from typing import List, Tuple, Optional


def load_data(csv_file: str) -> List[dict]:
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close'])
            })
        return rows


def compute_results(rows: List[dict], P3: float, Q3: int, R3: int, P6: float, Q6: float) -> Tuple[int, int, int, float]:
    results: List[Optional[float]] = []
    for idx in range(len(rows)):
        if idx - Q3 < 0 or idx + R3 >= len(rows):
            results.append(None)
            continue
        close = rows[idx]['Close']
        past_closes = [rows[j]['Close'] for j in range(idx - Q3, idx)]
        if min(past_closes) >= close - P3 and max(past_closes) <= close:
            future_highs = [rows[j]['High'] for j in range(idx + 1, idx + R3 + 1)]
            if max(future_highs) > close + Q6:
                results.append(1)
            else:
                future_lows = [rows[j]['Low'] for j in range(idx + 1, idx + R3 + 1)]
                if min(future_lows) < close - P6:
                    results.append(-1)
                else:
                    results.append(rows[idx + R3]['Close'] - close)
        elif min(past_closes) >= close and max(past_closes) <= close + P3:
            future_highs = [rows[j]['High'] for j in range(idx + 1, idx + R3 + 1)]
            if max(future_highs) > close + P6:
                results.append(-1)
            else:
                future_lows = [rows[j]['Low'] for j in range(idx + 1, idx + R3 + 1)]
                if min(future_lows) < close - Q6:
                    results.append(1)
                else:
                    results.append(close - rows[idx + R3]['Close'])
        else:
            results.append(None)

    hits = [r for r in results if r is not None]
    total_hits = len(hits)
    total_plus1 = sum(1 for r in hits if r == 1)
    total_minus1 = sum(1 for r in hits if r == -1)
    other_sum = sum(r for r in hits if r not in {1, -1})
    return total_hits, total_plus1, total_minus1, other_sum


def main():
    parser = argparse.ArgumentParser(description="Calculate hits from EURUSD data.")
    parser.add_argument('--csv', default='EURUSD_M30_Data.csv', help='CSV file with OHLC data')
    parser.add_argument('--P3', type=float, required=True, help='Range amount')
    parser.add_argument('--Q3', type=int, required=True, help='Lookback length')
    parser.add_argument('--R3', type=int, required=True, help='Duration')
    parser.add_argument('--P6', type=float, required=True, help='Stop loss')
    parser.add_argument('--Q6', type=float, required=True, help='Take profit')
    args = parser.parse_args()

    rows = load_data(args.csv)
    hits, plus1, minus1, others = compute_results(rows, args.P3, args.Q3, args.R3, args.P6, args.Q6)
    print(f'Total counted hits: {hits}')
    print(f'Total +1: {plus1}')
    print(f'Total -1: {minus1}')
    print(f'Sum of other hits: {others}')


if __name__ == '__main__':
    main()
