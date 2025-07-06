import csv

P3 = 0.005  # range amount
Q3 = 10     # length (lookback)
R3 = 10     # duration (lookahead)
P6 = 0.002  # stop loss
Q6 = 0.003  # take profit

with open('EURUSD_M30_Data.csv', newline='') as f:
    reader = csv.DictReader(f)
    data = list(reader)

close = [float(row['Close']) for row in data]
high = [float(row['High']) for row in data]
low = [float(row['Low']) for row in data]

n = len(data)
results = [''] * n

for i in range(Q3, n - R3):
    current_close = close[i]
    prev_closes = close[i-Q3:i]
    cond1 = min(prev_closes) >= current_close - P3 and max(prev_closes) <= current_close
    cond2 = min(prev_closes) >= current_close and max(prev_closes) <= current_close + P3
    if cond1:
        fut_highs = high[i+1:i+1+R3]
        fut_lows = low[i+1:i+1+R3]
        if max(fut_highs) > current_close + Q6:
            results[i] = 1
        elif min(fut_lows) < current_close - P6:
            results[i] = -1
        else:
            results[i] = close[i+R3] - current_close
    elif cond2:
        fut_highs = high[i+1:i+1+R3]
        fut_lows = low[i+1:i+1+R3]
        if max(fut_highs) > current_close + P6:
            results[i] = -1
        elif min(fut_lows) < current_close - Q6:
            results[i] = 1
        else:
            results[i] = current_close - close[i+R3]


# results for each row are kept in the 'results' list; no output CSV is generated

# Summarize results
hits = [r for r in results if r != '']
total_hits = len(hits)
total_pos = sum(1 for r in hits if r == 1)
total_neg = sum(1 for r in hits if r == -1)
other_sum = sum(r for r in hits if r not in (-1, 1))

print('Summary:')
print('Total hits:', total_hits)
print('Total -1:', total_neg)
print('Total 1:', total_pos)
print('Sum of other numbers:', other_sum)
