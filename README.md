# Trading

This repository stores a sample dataset of EUR/USD 30 minute bars and a Python
script for calculating "hits" based on simple trading rules.

## Dataset

`EURUSD_M30_Data.csv` contains 10,000 rows of candlestick data ranging from
September 2024 until July 2025. Each row has `Time`, `Open`, `High`, `Low` and
`Close` columns.

## Script: `compute_hits.py`

The `compute_hits.py` script reads the CSV file and applies a basic trading
strategy. It counts how many times certain price patterns are met, reports the
number of +1 and -1 outcomes and sums any remaining results.

The script accepts optional arguments to override the default parameters:

```
--csv PATH   path to the CSV file (default: EURUSD_M30_Data.csv)
--P3 FLOAT   range amount
--Q3 INT     lookback length
--R3 INT     duration in bars
--P6 FLOAT   stop loss amount
--Q6 FLOAT   take profit amount
```

Run the script with Python 3:

```bash
python3 compute_hits.py
```

You can also provide your own parameters, for example:

```bash
python3 compute_hits.py --P3 0.0003 --Q3 10 --R3 3
```

The program then prints a summary similar to:

```
Total counted hits: 16
Total +1: 6
Total -1: 7
Sum of other hits: 8.999999999992347e-05
```

No additional dependencies are required beyond the Python standard library.

## Script: `strategy_test.py`

`strategy_test.py` evaluates a breakout strategy that triggers after a period of
consolidation. By default the script determines a "tight" range using the
average candle range (ATR) of the lookback period multiplied by a factor of two.
This avoids situations where the strategy produces no trades because the fixed
threshold is too small for the dataset.

Run the script with:

```bash
python3 strategy_test.py
```

Important arguments:

```
--period INT       lookback bars for range (default: 10)
--threshold FLOAT  fixed range width. Use 0 to derive from ATR
--atr-mult FLOAT   ATR multiplier when threshold is 0 (default: 2.0)
--risk FLOAT       stop loss distance
--rr FLOAT         reward-to-risk ratio
--spread FLOAT     spread in price units
```

Example using a fixed threshold:

```bash
python3 strategy_test.py --threshold 0.002 --period 10
```

The program prints metrics such as win rate, risk-reward ratio and expectancy
along with a simple sensitivity analysis.
