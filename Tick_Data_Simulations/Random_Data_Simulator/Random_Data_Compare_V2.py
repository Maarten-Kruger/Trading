import pandas as pd
import numpy as np
import json

# ==========================================
# CONFIGURATION & VARIABLES
# ==========================================
FILE_PATH = 'BTCUSD_Jan.csv'         # Path to your tick data file
OUTPUT_HTML = 'BTCUSD_50_Exness.html' # Name of the generated HTML file

POINT_MULTIPLIER = 1            # Conversion to points (100,000 for 5-decimal pairs like EURUSD)
THRESHOLD_POINTS = 500                # 5 pips = 50 points. Change this to test different breakout sizes
MAX_TICK_JUMP_CAP = 500               # Caps extreme outliers in single-tick data errors
HISTOGRAM_BINS = 200                 # Number of bins for the smooth histogram
START_HOUR = 8                       # Filter ticks starting at this hour (e.g. 8 for 08:00)
END_HOUR = 19                        # Filter ticks ending at this hour (e.g. 17 for 17:59)
BLOCK_MINUTES = 15                    # Size of time blocks in minutes
SAMPLE_BLOCKS = 10000                # Number of time blocks to show in sample OHLC path
FAST_JUMP_LIMIT_SECONDS = 1200       # Threshold to consider a jump "fast" for overlay purposes (e.g. 10.5 seconds)


# ==========================================
# 1. LOAD DATA & GET PROBABILITY PROFILE
# ==========================================
print(f"Loading data from {FILE_PATH}...")
# Note: Adjust separator or loading logic if your CSV structure requires it.
try:
    df = pd.read_csv(FILE_PATH, sep='\t')
    # Combine date and time to create a datetime column
    if '<DATE>' in df.columns and '<TIME>' in df.columns:
        df['Datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S.%f')
    elif 'Datetime' not in df.columns:
        # Fallback if specific format missing but another time column might exist
        df['Datetime'] = pd.date_range("2026-01-01", periods=len(df), freq="1s")

    # Drop rows where <BID> is missing to ensure equal lengths
    df = df.dropna(subset=['<BID>'])

    # Filter by time of day
    df = df[(df['Datetime'].dt.hour >= START_HOUR) & (df['Datetime'].dt.hour <= END_HOUR)]

    prices = df['<BID>'].values * POINT_MULTIPLIER
    timestamps = df['Datetime'].values
except (KeyError, FileNotFoundError):
    # Fallback dummy data generation if file isn't found/formatted right so the script still runs
    print("Could not find <BID> column or file, generating synthetic data for demonstration.")
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, 100000)) * 10
    timestamps = pd.date_range("2026-01-01", periods=len(prices), freq="1s").values

N = len(prices)

print(f"Loaded {N} ticks. Calculating real market tick profile...")
# Get actual diffs
diff_actual = np.diff(prices)

# ==========================================
# 2. SIMULATE RANDOM WALK
# ==========================================
print("Simulating Random Walk using randomly shuffled actual ticks...")
np.random.seed(42)  # For reproducibility
shuffled_diffs = np.random.permutation(diff_actual)
sim_prices = np.insert(np.cumsum(shuffled_diffs) + prices[0], 0, prices[0])

# ==========================================
# 3. MEASURE DURATIONS & MAE & CALCULATE STATS
# ==========================================
print(f"Measuring duration and MAE to hit {THRESHOLD_POINTS} points...")
def get_blocks_data(price_array, threshold, timestamp_array):
    durations = []
    maes = []
    directions = []
    end_times = [] # <--- NEW: Track when the jump completes
    ref_price = price_array[0]
    ref_idx = 0

    # We want to keep track of the max and min seen so far in this block
    max_price = ref_price
    min_price = ref_price

    for i in range(1, len(price_array)):
        current_price = price_array[i]

        if current_price > max_price:
            max_price = current_price
        if current_price < min_price:
            min_price = current_price

        diff = current_price - ref_price

        if abs(diff) >= threshold:
            # duration in seconds
            time_diff = (timestamp_array[i] - timestamp_array[ref_idx])
            durations.append(time_diff / np.timedelta64(1, 's'))
            end_times.append(timestamp_array[i]) # <--- NEW: Record the time

            if diff >= threshold:
                # Upward breakout: MAE is the maximum unfavorable (downward) movement.
                # So we look at the lowest point reached compared to ref_price.
                mae = abs(min_price - ref_price)
                directions.append(1)
            else:
                # Downward breakout: MAE is the maximum unfavorable (upward) movement.
                # So we look at the highest point reached compared to ref_price.
                mae = abs(max_price - ref_price)
                directions.append(-1)

            maes.append(mae)

            # Reset block stats
            ref_price = current_price
            ref_idx = i
            max_price = current_price
            min_price = current_price
    return durations, maes, directions, end_times

durations_actual, mae_actual, dirs_actual, end_times_actual = get_blocks_data(prices, THRESHOLD_POINTS, timestamps)
durations_sim, mae_sim, dirs_sim, end_times_sim = get_blocks_data(sim_prices, THRESHOLD_POINTS, timestamps)

def calc_stats(data_array):
    if not data_array:
        return {"trials": 0, "mean": 0, "median": 0, "std": 0, "p25": 0, "p75": 0, "p95": 0}
    return {
        "trials": len(data_array),
        "mean": np.mean(data_array),
        "median": np.median(data_array),
        "std": np.std(data_array),
        "p25": np.percentile(data_array, 25),
        "p75": np.percentile(data_array, 75),
        "p95": np.percentile(data_array, 95)
    }

actual_stats = calc_stats(durations_actual)
sim_stats = calc_stats(durations_sim)
actual_mae_stats = calc_stats(mae_actual)
sim_mae_stats = calc_stats(mae_sim)

print(f"Actual {THRESHOLD_POINTS}-point moves: {actual_stats['trials']}")
print(f"Simulated {THRESHOLD_POINTS}-point moves: {sim_stats['trials']}")

# ==========================================
# 4. BUILD SMOOTH HISTOGRAM DATA
# ==========================================
print("Generating distribution curves...")
# Define bin range based on 95th percentile to drop extremely long tails
max_dur = max(actual_stats['p95'], sim_stats['p95']) if actual_stats['trials'] > 0 else 100
bins = np.linspace(0, max_dur * 1.2, HISTOGRAM_BINS)

# We use count instead of density so the Y-axis matches trial volume better
hist_actual, _ = np.histogram(durations_actual, bins=bins)
hist_sim, _ = np.histogram(durations_sim, bins=bins)


bin_centers = (bins[:-1] + bins[1:]) / 2

# Package into JSON format for Chart.js
output_data = []
for i in range(len(bin_centers)):
    output_data.append({
        "duration_secs": round(bin_centers[i], 1),
        "Actual": float(hist_actual[i]),
        "Simulated": float(hist_sim[i])
    })

json_data_string = json.dumps(output_data)

# --- MAE Histogram Data ---
max_mae = max(actual_mae_stats['p95'], sim_mae_stats['p95']) if actual_mae_stats['trials'] > 0 else 100
bins_mae = np.linspace(0, max_mae * 1.2, HISTOGRAM_BINS)

hist_mae_act, _ = np.histogram(mae_actual, bins=bins_mae)
hist_mae_sim, _ = np.histogram(mae_sim, bins=bins_mae)


bin_centers_mae = (bins_mae[:-1] + bins_mae[1:]) / 2

mae_output_data = []
for i in range(len(bin_centers_mae)):
    mae_output_data.append({
        "mae_points": round(bin_centers_mae[i], 1),
        "Actual": float(hist_mae_act[i]),
        "Simulated": float(hist_mae_sim[i])
    })

mae_json_string = json.dumps(mae_output_data)

pips_target = int(THRESHOLD_POINTS / 10)
title_str = FILE_PATH.split('.')[0].replace('_', ' ')

# ==========================================
# 4.6 CALCULATE STREAKS (TARGET HITS)
# ==========================================
print("Calculating target hit streaks...")
def get_target_streaks(price_array, threshold):
    directions = []
    ref_price = price_array[0]
    
    # First, record the direction (+1 or -1) every time the threshold is hit
    for i in range(1, len(price_array)):
        diff = price_array[i] - ref_price
        if abs(diff) >= threshold:
            directions.append(np.sign(diff))
            ref_price = price_array[i]
            
    if not directions: return []
    
    # Next, calculate the lengths of consecutive identical directions
    directions = np.array(directions)
    changes = np.where(directions[:-1] != directions[1:])[0] + 1
    runs = np.split(directions, changes)
    return [len(run) for run in runs]

streaks_actual = get_target_streaks(prices, THRESHOLD_POINTS)
streaks_sim = get_target_streaks(sim_prices, THRESHOLD_POINTS)

act_lens, act_counts = np.unique(streaks_actual, return_counts=True)
sim_lens, sim_counts = np.unique(streaks_sim, return_counts=True)

act_map = dict(zip(act_lens, act_counts))
sim_map = dict(zip(sim_lens, sim_counts))

max_streak = int(max(max(act_lens, default=0), max(sim_lens, default=0)))

streak_data = []
for i in range(1, max_streak + 1):
    streak_data.append({
        "streak_len": i,
        "Actual": int(act_map.get(i, 0)),
        "Simulated": int(sim_map.get(i, 0))
    })

streak_json_string = json.dumps(streak_data)

actual_streak_stats = calc_stats(streaks_actual)
sim_streak_stats = calc_stats(streaks_sim)


# ==========================================
# 4.7 OHLC TIME BLOCKS & SIZE DISTRIBUTION
# ==========================================
print(f"Calculating OHLC Time Blocks & Size Metrics ({BLOCK_MINUTES} mins)...")

df_act = pd.DataFrame({'price': prices}, index=timestamps)
df_sim = pd.DataFrame({'price': sim_prices}, index=timestamps)

def calc_ohlc_metrics(df, block_minutes):
    ohlc = df['price'].resample(f'{block_minutes}min').ohlc().dropna()
    ohlc['total_size'] = ohlc['high'] - ohlc['low']
    ohlc['body'] = abs(ohlc['close'] - ohlc['open'])
    ohlc['head'] = ohlc['high'] - np.maximum(ohlc['open'], ohlc['close'])
    ohlc['tail'] = np.minimum(ohlc['open'], ohlc['close']) - ohlc['low']
    return ohlc

ohlc_act = calc_ohlc_metrics(df_act, BLOCK_MINUTES)
ohlc_sim = calc_ohlc_metrics(df_sim, BLOCK_MINUTES)

max_size_ohlc = max(np.percentile(ohlc_act['total_size'], 95) if len(ohlc_act) > 0 else 100,
                    np.percentile(ohlc_sim['total_size'], 95) if len(ohlc_sim) > 0 else 100)

bins_size = np.linspace(0, max_size_ohlc * 1.2, 50)
bin_centers_size = (bins_size[:-1] + bins_size[1:]) / 2

ohlc_act['bin'] = pd.cut(ohlc_act['total_size'], bins=bins_size, labels=bin_centers_size)
ohlc_sim['bin'] = pd.cut(ohlc_sim['total_size'], bins=bins_size, labels=bin_centers_size)

grouped_act_mean = ohlc_act.groupby('bin', observed=False)[['body', 'head', 'tail']].mean().fillna(0)
grouped_sim_mean = ohlc_sim.groupby('bin', observed=False)[['body', 'head', 'tail']].mean().fillna(0)

count_act = ohlc_act.groupby('bin', observed=False).size()
count_sim = ohlc_sim.groupby('bin', observed=False).size()

size_dist_data = []
for i, b in enumerate(bin_centers_size):
    size_dist_data.append({
        "size_bin": round(float(b), 1),
        "act_body": float(grouped_act_mean.loc[b, 'body']),
        "act_head": float(grouped_act_mean.loc[b, 'head']),
        "act_tail": float(grouped_act_mean.loc[b, 'tail']),
        "act_count": int(count_act.loc[b]),
        "sim_body": float(grouped_sim_mean.loc[b, 'body']),
        "sim_head": float(grouped_sim_mean.loc[b, 'head']),
        "sim_tail": float(grouped_sim_mean.loc[b, 'tail']),
        "sim_count": int(count_sim.loc[b]),
    })

size_dist_json = json.dumps(size_dist_data)

# Global Size Stats
size_stats = {
    'act_body': ohlc_act['body'].mean() if len(ohlc_act) > 0 else 0,
    'act_head': ohlc_act['head'].mean() if len(ohlc_act) > 0 else 0,
    'act_tail': ohlc_act['tail'].mean() if len(ohlc_act) > 0 else 0,
    'sim_body': ohlc_sim['body'].mean() if len(ohlc_sim) > 0 else 0,
    'sim_head': ohlc_sim['head'].mean() if len(ohlc_sim) > 0 else 0,
    'sim_tail': ohlc_sim['tail'].mean() if len(ohlc_sim) > 0 else 0,
}

# Take sample for OHLC Path Line Overlay
sample_act_closes = ohlc_act['close'].head(SAMPLE_BLOCKS).tolist()
sample_sim_closes = ohlc_sim['close'].head(SAMPLE_BLOCKS).tolist()
sample_timestamps = ohlc_act.index[:SAMPLE_BLOCKS].values

# --- MAP FAST JUMPS TO OHLC SAMPLE ---
actual_fast_jumps = [None] * len(sample_act_closes)

# Filter jump times that happened in <= 10.5 seconds
fast_jump_times = [
    end_times_actual[i] for i in range(len(durations_actual)) 
    if durations_actual[i] <= FAST_JUMP_LIMIT_SECONDS
]

# Map them onto our chart's time blocks
for jt in fast_jump_times:
    # Check if the jump occurred within our sample timeframe
    if jt >= sample_timestamps[0] and jt <= sample_timestamps[-1]:
        # Find which OHLC block this timestamp falls into
        idx = np.searchsorted(sample_timestamps, jt) - 1
        if 0 <= idx < len(actual_fast_jumps):
            actual_fast_jumps[idx] = sample_act_closes[idx]

sample_json = json.dumps({
    "actual": sample_act_closes,
    "simulated": sample_sim_closes,
    "actual_fast_jumps": actual_fast_jumps
})


# ==========================================
# 5. GENERATE HTML WIDGET (LIGHT THEME DASHBOARD)
# ==========================================
print(f"Writing HTML to {OUTPUT_HTML}...")

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tick Volatility: Actual vs Simulated</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-body: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --color-actual: #3b82f6;
            --color-actual-bg: rgba(59, 130, 246, 0.15);
            --color-sim: #f97316;
            --color-sim-bg: rgba(249, 115, 22, 0.15);
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            line-height: 1.5;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 25px;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: var(--text-main);
        }}
        .subtitle {{
            font-size: 14px;
            color: var(--text-muted);
            margin: 0;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card-header {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            width: 100%;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
        }}
        .stat-col h3 {{
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            margin: 0 0 12px 0;
        }}
        .title-actual {{ color: var(--color-actual); }}
        .title-sim {{ color: var(--color-sim); }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid var(--border-color);
            font-size: 14px;
        }}
        .stat-row:last-child {{ border-bottom: none; }}
        .stat-label {{ color: var(--text-muted); }}
        .stat-value {{ font-weight: 600; color: var(--text-main); }}
        
        /* Insights & Setup Boxes */
        .insight-box {{
            background-color: #f1f5f9;
            border-left: 3px solid #8b5cf6;
            padding: 16px 20px;
            border-radius: 4px;
            margin-top: 24px;
            font-size: 14px;
        }}
        .setup-section {{
            font-size: 14px;
            color: var(--text-muted);
        }}
        .setup-section p {{ margin-top: 10px; margin-bottom: 10px; }}
        strong {{ color: var(--text-main); font-weight: 600; }}
        
        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: 1fr; gap: 20px; }}
        }}
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>{title_str} — Ticks to ±{pips_target} Pips</h1>
        <p class="subtitle">Actual market data vs. Monte Carlo simulation using empirical tick-move probabilities</p>
    </div>

    <div class="card">
        <div class="card-header">Duration Distribution</div>
        <div class="chart-container">
            <canvas id="volatilityChart"></canvas>
        </div>

    <div class="card">
        <div class="card-header">Streak Distribution (Consecutive Hits in Same Direction)</div>
        <div class="chart-container">
            <canvas id="streakChart"></canvas>
        </div>
    </div>

    <div class="card">
        <div class="card-header">Maximum Adverse Excursion (MAE) Distribution</div>
        <div class="chart-container">
            <canvas id="maeChart"></canvas>
        </div>
    </div>

    <div class="card">
        <div class="card-header">Time Block Size Distribution (Head, Body, Tail)</div>
        <div class="chart-container">
            <canvas id="sizeChart"></canvas>
        </div>
    </div>

    <div class="card">
        <div class="card-header">Sample Path Overlay</div>
        <div class="chart-container">
            <canvas id="sampleChart"></canvas>
        </div>
    </div>

    <div class="card">
        <div class="card-header">Summary Statistics</div>
        
        <div class="stats-grid">
            <div class="stat-col">
                <h3 class="title-actual">Actual EURUSD</h3>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">Duration (Seconds)</div>
                <div class="stat-row">
                    <span class="stat-label">Trials</span>
                    <span class="stat-value">{actual_stats['trials']:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mean</span>
                    <span class="stat-value">{actual_stats['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Median</span>
                    <span class="stat-value">{actual_stats['median']:.0f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Std Dev</span>
                    <span class="stat-value">{actual_stats['std']:.1f}</span>
                </div>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">MAE (Points)</div>
                <div class="stat-row">
                    <span class="stat-label">Mean</span>
                    <span class="stat-value">{actual_mae_stats['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Median</span>
                    <span class="stat-value">{actual_mae_stats['median']:.0f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Std Dev</span>
                    <span class="stat-value">{actual_mae_stats['std']:.1f}</span>
                </div>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">Streaks</div>
                <div class="stat-row">
                    <span class="stat-label">Mean Length</span>
                    <span class="stat-value">{actual_streak_stats['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Max Length</span>
                    <span class="stat-value">{max(act_lens, default=0):.0f}</span>
                </div>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">Time Block Avg Size (Points)</div>
                <div class="stat-row">
                    <span class="stat-label">Body</span>
                    <span class="stat-value">{size_stats['act_body']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Head</span>
                    <span class="stat-value">{size_stats['act_head']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Tail</span>
                    <span class="stat-value">{size_stats['act_tail']:.1f}</span>
                </div>

            </div>

            <div class="stat-col">
                <h3 class="title-sim">Simulated</h3>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">Duration (Seconds)</div>
                <div class="stat-row">
                    <span class="stat-label">Trials</span>
                    <span class="stat-value">{sim_stats['trials']:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mean</span>
                    <span class="stat-value">{sim_stats['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Median</span>
                    <span class="stat-value">{sim_stats['median']:.0f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Std Dev</span>
                    <span class="stat-value">{sim_stats['std']:.1f}</span>
                </div>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">MAE (Points)</div>
                <div class="stat-row">
                    <span class="stat-label">Mean</span>
                    <span class="stat-value">{sim_mae_stats['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Median</span>
                    <span class="stat-value">{sim_mae_stats['median']:.0f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Std Dev</span>
                    <span class="stat-value">{sim_mae_stats['std']:.1f}</span>
                </div>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">Streaks</div>
                <div class="stat-row">
                    <span class="stat-label">Mean Length</span>
                    <span class="stat-value">{sim_streak_stats['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Max Length</span>
                    <span class="stat-value">{max(sim_lens, default=0):.0f}</span>
                </div>

                <div style="font-weight: 600; font-size: 13px; margin: 15px 0 5px 0; color: var(--text-main); border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">Time Block Avg Size (Points)</div>
                <div class="stat-row">
                    <span class="stat-label">Body</span>
                    <span class="stat-value">{size_stats['sim_body']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Head</span>
                    <span class="stat-value">{size_stats['sim_head']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Tail</span>
                    <span class="stat-value">{size_stats['sim_tail']:.1f}</span>
                </div>

            </div>
        </div>

        <div class="insight-box">
            <strong>What the gap tells you:</strong> The actual market reaches ±{pips_target} pips <em>later</em> than pure random (mean {actual_stats['mean']:.1f} vs {sim_stats['mean']:.1f} seconds). Both distributions share the same tick-size profile — the only thing that differs is direction. The heavier right tail in real data reflects <strong>mean-reversion micro-structure</strong>: consecutive ticks often reverse (market makers ping-ponging the spread), causing the price to "waste" ticks oscillating near the origin before eventually trending far enough to hit the target.
        </div>
    </div>

    <div class="card">
        <div class="card-header">Experiment Setup</div>
        <div class="setup-section">
            <p><strong>What is being measured?</strong></p>
            <p>We walk forward tick-by-tick through the data series and record <strong>how many seconds it takes until the price has moved ±{pips_target} pips ({THRESHOLD_POINTS} points) away from the start</strong>. The count increments by 1 per second elapsed, not per pip.</p>
            <p>The same experiment is repeated on a <strong>synthetic random walk</strong> that uses the exact same per-tick move distribution observed in the real data — but with no sequential dependence (each tick is drawn independently with a 50/50 directional probability).</p>
        </div>
    </div>
</div>

<script>
    const rawData = {json_data_string};

    const labels = rawData.map(d => d.duration_secs);
    const actualData = rawData.map(d => d.Actual);
    const simData = rawData.map(d => d.Simulated);

    const ctx = document.getElementById('volatilityChart').getContext('2d');
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [
                {{
                    label: 'Actual Data',
                    data: actualData,
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 0,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0
                }},
                {{
                    label: 'Simulated',
                    data: simData,
                    backgroundColor: 'rgba(249, 115, 22, 0.7)',
                    borderColor: 'rgba(249, 115, 22, 1)',
                    borderWidth: 0,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                tooltip: {{ 
                    callbacks: {{ 
                        label: function(context) {{ return context.dataset.label + ': ' + context.parsed.y.toFixed(2); }} 
                    }} 
                }},
                legend: {{ 
                    position: 'top',
                    align: 'start',
                    labels: {{
                        usePointStyle: true,
                        boxWidth: 8,
                        font: {{ family: "-apple-system, system-ui, sans-serif", size: 13, weight: '500' }}
                    }}
                }}
            }},
            scales: {{
                x: {{ 
                    title: {{ display: true, text: 'Seconds to reach ±{pips_target} pips', color: '#64748b', font: {{ size: 12 }} }},
                    grid: {{ display: false }},
                    ticks: {{ color: '#64748b' }}
                }},
                y: {{ 
                    title: {{ display: true, text: 'Count', color: '#64748b', font: {{ size: 12 }} }},
                    grid: {{ color: '#f1f5f9' }},
                    ticks: {{ color: '#64748b' }},
                    beginAtZero: true 
                }}
            }}
        }}
    }});

    // --- STREAK CHART ---
    const rawStreakData = {streak_json_string};
    const streakLabels = rawStreakData.map(d => d.streak_len);
    const streakAct = rawStreakData.map(d => d.Actual);
    const streakSim = rawStreakData.map(d => d.Simulated);

    const ctxStreak = document.getElementById('streakChart').getContext('2d');
    new Chart(ctxStreak, {{
        type: 'bar',
        data: {{
            labels: streakLabels,
            datasets: [
                {{
                    label: 'Actual Data',
                    data: streakAct,
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                }},
                {{
                    label: 'Simulated',
                    data: streakSim,
                    backgroundColor: 'rgba(249, 115, 22, 0.7)',
                    borderColor: 'rgba(249, 115, 22, 1)',
                    borderWidth: 1
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', align: 'start', labels: {{ usePointStyle: true, boxWidth: 8 }} }}
            }},
            scales: {{
                x: {{ 
                    title: {{ display: true, text: 'Consecutive Hits', color: '#64748b' }},
                    grid: {{ display: false }}
                }},
                y: {{ 
                    type: 'linear', 
                    title: {{ display: true, text: 'Frequency', color: '#64748b' }},
                    grid: {{ color: '#f1f5f9' }},
                    beginAtZero: true
                }}
            }}
        }}
    }});

    // --- MAE CHART ---
    const rawMaeData = {mae_json_string};
    const maeLabels = rawMaeData.map(d => d.mae_points);
    const maeAct = rawMaeData.map(d => d.Actual);
    const maeSim = rawMaeData.map(d => d.Simulated);

    const ctxMae = document.getElementById('maeChart').getContext('2d');
    new Chart(ctxMae, {{
        type: 'bar',
        data: {{
            labels: maeLabels,
            datasets: [
                {{
                    label: 'Actual Data MAE',
                    data: maeAct,
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 0,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0
                }},
                {{
                    label: 'Simulated MAE',
                    data: maeSim,
                    backgroundColor: 'rgba(249, 115, 22, 0.7)',
                    borderColor: 'rgba(249, 115, 22, 1)',
                    borderWidth: 0,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{ return context.dataset.label + ': ' + context.parsed.y.toFixed(2); }}
                    }}
                }},
                legend: {{ position: 'top', align: 'start', labels: {{ usePointStyle: true, boxWidth: 8 }} }}
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Maximum Adverse Excursion (Points)', color: '#64748b' }},
                    grid: {{ display: false }}
                }},
                y: {{
                    type: 'linear',
                    title: {{ display: true, text: 'Frequency', color: '#64748b' }},
                    grid: {{ color: '#f1f5f9' }},
                    beginAtZero: true
                }}
            }}
        }}
    }});

    // --- SIZE DISTRIBUTION CHART ---
    const rawSizeData = {size_dist_json};
    const sizeLabels = rawSizeData.map(d => d.size_bin);

    const ctxSize = document.getElementById('sizeChart').getContext('2d');
    new Chart(ctxSize, {{
        type: 'bar',
        data: {{
            labels: sizeLabels,
            datasets: [
                {{
                    label: 'Actual - Tail',
                    data: rawSizeData.map(d => d.act_tail),
                    backgroundColor: 'rgba(59, 130, 246, 0.4)',
                    stack: 'actual',
                    borderWidth: 0
                }},
                {{
                    label: 'Actual - Body',
                    data: rawSizeData.map(d => d.act_body),
                    backgroundColor: 'rgba(59, 130, 246, 0.9)',
                    stack: 'actual',
                    borderWidth: 0
                }},
                {{
                    label: 'Actual - Head',
                    data: rawSizeData.map(d => d.act_head),
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    stack: 'actual',
                    borderWidth: 0
                }},
                {{
                    label: 'Simulated - Tail',
                    data: rawSizeData.map(d => d.sim_tail),
                    backgroundColor: 'rgba(249, 115, 22, 0.4)',
                    stack: 'simulated',
                    borderWidth: 0
                }},
                {{
                    label: 'Simulated - Body',
                    data: rawSizeData.map(d => d.sim_body),
                    backgroundColor: 'rgba(249, 115, 22, 0.9)',
                    stack: 'simulated',
                    borderWidth: 0
                }},
                {{
                    label: 'Simulated - Head',
                    data: rawSizeData.map(d => d.sim_head),
                    backgroundColor: 'rgba(249, 115, 22, 0.6)',
                    stack: 'simulated',
                    borderWidth: 0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', align: 'start', labels: {{ usePointStyle: true, boxWidth: 8 }} }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const dataset = context.dataset;
                            const index = context.dataIndex;
                            const isActual = dataset.stack === 'actual';
                            const count = isActual ? rawSizeData[index].act_count : rawSizeData[index].sim_count;
                            return `${{dataset.label}}: ${{context.parsed.y.toFixed(2)}} (Count: ${{count}})`;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Total Block Size (Points)', color: '#64748b' }},
                    grid: {{ display: false }}
                }},
                y: {{
                    title: {{ display: true, text: 'Average Size Components (Points)', color: '#64748b' }},
                    grid: {{ color: '#f1f5f9' }},
                    beginAtZero: true
                }}
            }}
        }}
    }});

    // --- SAMPLE PATH CHART ---
    const rawSampleData = {sample_json};
    const numSamples = Math.min(rawSampleData.actual.length, rawSampleData.simulated.length);
    const sampleLabels = Array.from({{length: numSamples}}, (_, i) => i + 1);

    const ctxSample = document.getElementById('sampleChart').getContext('2d');
    new Chart(ctxSample, {{
        type: 'line',
        data: {{
            labels: sampleLabels,
            datasets: [
                {{
                    type: 'scatter',
                    label: 'Fast Jumps (>{int(THRESHOLD_POINTS / 10)} pips in <{FAST_JUMP_LIMIT_SECONDS}s)',
                    data: rawSampleData.actual_fast_jumps,
                    backgroundColor: 'rgba(239, 68, 68, 1)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    pointStyle: 'circle',
                    pointRadius: 2,
                    pointHoverRadius: 3,
                    showLine: false,
                    order: 0
                }},
                {{
                    label: 'Actual Data (Close)',
                    data: rawSampleData.actual,
                    borderColor: 'rgba(59, 130, 246, 0.8)',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }},
                {{
                    label: 'Simulated (Close)',
                    data: rawSampleData.simulated,
                    borderColor: 'rgba(249, 115, 22, 0.8)',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', align: 'start', labels: {{ usePointStyle: true, boxWidth: 8 }} }}
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Time Block Index', color: '#64748b' }},
                    grid: {{ display: false }},
                    ticks: {{ maxTicksLimit: 10 }}
                }},
                y: {{
                    title: {{ display: true, text: 'Price', color: '#64748b' }},
                    grid: {{ color: '#f1f5f9' }}
                }}
            }}
        }}
    }});
</script>

</body>
</html>
"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Done! Open the generated HTML file in your web browser.")