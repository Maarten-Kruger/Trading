# Trading

This repository includes a backtesting script for the Hover Breakout Strategy.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the backtest

Run the strategy script using Python 3:

```bash
python3 hover_breakout_strategy.py
```

The script expects a file named `EURUSD_M30_Data.csv` in the same directory. If the file is not present, synthetic data will be generated for demonstration purposes.

## Optimizing parameters

To search for profitable parameter combinations, run:

```bash
python3 optimize_hover_breakout.py
```

This will test a grid of parameter values and produce `hover_breakout_optimization.pdf` summarizing the best results.
