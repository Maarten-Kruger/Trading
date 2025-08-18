# Trading

Collection of MetaTrader 5 Expert Advisors.

## Grouped Volatility

`1.2 GroupedVolatility.mq5` now evaluates candle patterns even when other
positions remain open, allowing multiple trades to run concurrently when the
setup reappears on new bars.

## Candlestick Simulator

`visualize_ohlc.py` provides a small matplotlib-based viewer for OHLC CSV data.
Scroll up to move forward through candles and scroll down to go backward.

Run it with a path to the data file:

```bash
python visualize_ohlc.py path/to/data.csv
```

Or simply execute the script and enter the path when prompted:

```bash
python visualize_ohlc.py
Path to OHLC CSV file: path/to/data.csv
```
