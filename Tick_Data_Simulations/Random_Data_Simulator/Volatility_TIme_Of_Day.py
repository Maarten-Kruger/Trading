import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the tick data
df = pd.read_csv('US500.cash_Jan.csv', sep='\t')

# 2. Forward fill any missing bid/ask values (common in tick data)
df['<BID>'] = df['<BID>'].ffill()
df['<ASK>'] = df['<ASK>'].ffill()

# 3. Calculate the Mid price
df['Mid'] = (df['<BID>'] + df['<ASK>']) / 2

# 4. Extract the Hour of the day from the <TIME> column
# The format is HH:MM:SS.mmm, so we can slice the first two characters
df['Hour'] = df['<TIME>'].str[:2].astype(int)

# 5. Group by Date and Hour to calculate the highest and lowest price per hour
hourly_stats = df.groupby(['<DATE>', 'Hour'])['Mid'].agg(['max', 'min'])

# 6. Calculate the hourly range in Pips (1 Pip = 0.0001 for EUR/USD)
hourly_stats['Range_Pips'] = (hourly_stats['max'] - hourly_stats['min']) * 10000

# 7. Average the range across all days for each specific hour
avg_hourly_volatility = hourly_stats.groupby('Hour')['Range_Pips'].mean()

# 8. Graph the volatility profile
plt.figure(figsize=(12, 6))
avg_hourly_volatility.plot(kind='bar', color="#45B360", edgecolor='black')

plt.title('Average Intraday Volatility by Hour of Day (EURUSD)', fontsize=14)
plt.xlabel('Hour of Day (Server Time)', fontsize=12)
plt.ylabel('Average Hourly Range (Pips)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()