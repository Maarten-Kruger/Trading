import pandas as pd
from phase1 import load_and_preprocess_data, find_triggers, generate_images
from phase3 import run_backtest, generate_simulation_report

# Create mock data
data = {
    'Time': pd.date_range(start='2024-01-01', periods=100, freq='h'),
    'Open': [1.0] * 100,
    'High': [1.01] * 100,
    'Low': [0.99] * 100,
    'Close': [1.0] * 100,
    'Volume': [1000] * 100,
    'Spread': [10] * 100
}

# Create a trend to trigger SMA crossover
for i in range(50, 100):
    data['Close'][i] = 1.05
    data['High'][i] = 1.06

df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)

# Run phase 1
df_loaded = load_and_preprocess_data("test_data.csv")
triggers = find_triggers(df_loaded)
generate_images(df_loaded, triggers, "test_data.csv")

# Run phase 3
timestamps, equity_curve, drawdowns, trades = run_backtest(df_loaded, triggers)
generate_simulation_report(timestamps, equity_curve, drawdowns, "test_data.csv")

print("Test complete.")
