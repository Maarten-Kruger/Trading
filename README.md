# Trading

This repository stores sample datasets of EUR/USD prices and Python scripts for
testing different strategies. The main strategy now operates on one minute data
while checking every 30 candles (30 minutes) for patterns. When a breakout is
detected the strategy opens the trade on that same one-minute candle. Each trade
risks a fixed amount based on the starting equity so account growth remains
realistic during long simulations.

The optimization script explores a grid of parameters and now ranks the results
by final equity, the total number of trades executed and finally the maximum
drawdown. This helps surface configurations that are both profitable and
active.

Package list:
Pandas, NumPy, MatLab and ReportLab


