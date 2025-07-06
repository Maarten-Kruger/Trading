import csv
import argparse
from typing import List, Tuple, Optional

# ----- Configuration -----
# Edit these values as needed. They also serve as defaults for CLI arguments.
CSV_FILE = 'EURUSD_M30_Data.csv'
P3 = 0.0002  # Range amount
Q3 = 5        # Lookback length
R3 = 3        # Duration
P6 = 0.0003  # Stop loss
Q6 = 0.0004  # Take profit


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
    parser.add_argument('--csv', default=CSV_FILE, help='CSV file with OHLC data')
    parser.add_argument('--P3', type=float, default=P3, help='Range amount')
    parser.add_argument('--Q3', type=int, default=Q3, help='Lookback length')
    parser.add_argument('--R3', type=int, default=R3, help='Duration')
    parser.add_argument('--P6', type=float, default=P6, help='Stop loss')
    parser.add_argument('--Q6', type=float, default=Q6, help='Take profit')
    args = parser.parse_args()

    rows = load_data(args.csv)
    hits, plus1, minus1, others = compute_results(rows, args.P3, args.Q3, args.R3, args.P6, args.Q6)
    print(f'Total counted hits: {hits}')
    print(f'Total +1: {plus1}')
    print(f'Total -1: {minus1}')
    print(f'Sum of other hits: {others}')


if __name__ == '__main__':
    main()
