import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import re
from datetime import timedelta

# --- CONFIGURATION ---
# Set this to the folder where your CSV files are located. 
# "." means the current directory where the script is running.
DATA_DIR = "."
OUT_DIR = os.path.join(DATA_DIR, "news_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load and filter News
news_file = os.path.join(DATA_DIR, "Signals_And_News.csv")
print("Loading news events...")
news_df = pd.read_csv(news_file)

# Filter High/Medium Impact
news_df = news_df[news_df['Impact'].isin(['High', 'Medium'])]

# Identify the exact time column name dynamically (e.g., 'Time (+2:00 GMT)')
time_col = [c for c in news_df.columns if 'Time' in c][0]

# Mapping symbols to the tick data files you provided
symbol_to_file = {
    'EUR': 'EURUSDm_202606050000_202606142203.csv',
    'GBP': 'GBPUSDm_202605312106_202606142203.csv',
    'JPY': 'USDJPYm_202605312105_202606142203.csv',
    'NZD': 'NZDUSDm_202605312105_202606142203.csv',
    'AUD': 'AUDUSDm_202605312105_202606142203.csv',
    'CAD': 'USDCADm_202605312105_202606142203.csv',
    'CHF': 'USDCHFm_202605312106_202606142203.csv',
    'USD': 'EURUSDm_202606050000_202606142203.csv'  # Using EURUSD as a proxy to view USD impact
}

news_df['TargetFile'] = news_df['Symbol'].map(symbol_to_file)
news_df = news_df.dropna(subset=['TargetFile'])

# Parse Datetime
def parse_dt(row):
    try:
        dt_str = f"{row['Date']} {row[time_col]}"
        return pd.to_datetime(dt_str)
    except Exception:
        return pd.NaT

news_df['Datetime'] = news_df.apply(parse_dt, axis=1)
news_df = news_df.dropna(subset=['Datetime'])

generated_files = []

# Function to safely create a valid filename for the saved images
def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))

# 2. Iterate through target files and plot
for file_name, group in news_df.groupby('TargetFile'):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: {file_name} not found in directory. Skipping.")
        continue
        
    print(f"\nLoading tick data: {file_name}...")
    try:
        # Load the tick data (tab-separated)
        tdf = pd.read_csv(file_path, sep='\t')
        tdf.columns = [c.strip() for c in tdf.columns]
        
        # Parse datetime: Format is YYYY.MM.DD HH:MM:SS.ms
        tdf['Datetime'] = pd.to_datetime(tdf['<DATE>'] + ' ' + tdf['<TIME>'], format='%Y.%m.%d %H:%M:%S.%f')
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        continue
        
    print(f"Successfully loaded {len(tdf)} ticks. Processing news events...")
    
    for idx, row in group.iterrows():
        edt = row['Datetime']
        event_name = row['Name']
        symbol = row['Symbol']
        impact = row['Impact']
        
        start_dt = edt - timedelta(hours=1)
        end_dt = edt + timedelta(hours=1)
        
        # Filter tick data strictly around the 2-hour window of the event
        mask = (tdf['Datetime'] >= start_dt) & (tdf['Datetime'] <= end_dt)
        df_slice = tdf[mask].copy()
        
        if df_slice.empty:
            continue
            
        # Compute Mid price and Spread
        df_slice['Mid'] = (df_slice['<ASK>'] + df_slice['<BID>']) / 2.0
        df_slice['Spread'] = df_slice['<ASK>'] - df_slice['<BID>']
        df_slice.set_index('Datetime', inplace=True)
        
        # Resample into 6-second candles
        ohlc = df_slice['Mid'].resample('6s').ohlc().dropna()
        spread_avg = df_slice['Spread'].resample('6s').mean().reindex(ohlc.index).fillna(0)
        
        if ohlc.empty:
            continue
            
        # Prepare Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        x = np.arange(len(ohlc))
        
        up = ohlc['close'] >= ohlc['open']
        down = ohlc['close'] < ohlc['open']
        
        # Draw the 6-second Candles
        width = 0.8
        ax1.bar(x[up], ohlc['close'][up] - ohlc['open'][up], bottom=ohlc['open'][up], color='forestgreen', width=width)
        ax1.vlines(x[up], ohlc['low'][up], ohlc['high'][up], color='forestgreen', linewidth=1)
        
        ax1.bar(x[down], ohlc['open'][down] - ohlc['close'][down], bottom=ohlc['close'][down], color='firebrick', width=width)
        ax1.vlines(x[down], ohlc['low'][down], ohlc['high'][down], color='firebrick', linewidth=1)
        
        # Plot Event Line (Blue Dashed Line at the exact minute of the news)
        event_idx_arr = np.where(ohlc.index >= edt)[0]
        if len(event_idx_arr) > 0:
            event_idx = event_idx_arr[0]
            ax1.axvline(x=event_idx, color='dodgerblue', linestyle='--', linewidth=1.5, label='News Event Time')
            ax2.axvline(x=event_idx, color='dodgerblue', linestyle='--', linewidth=1.5)
            ax1.legend()

        ax1.set_title(f"[{impact} Impact] {symbol}: {event_name} - {edt.strftime('%Y-%m-%d %H:%M')}\n1 Hour Before / 1 Hour After (6s Candles)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Price (Mid)")
        ax1.grid(True, alpha=0.2)
        
        # Draw Spread in the bottom panel
        ax2.plot(x, spread_avg, color='purple', linewidth=1.5)
        ax2.set_ylabel("Spread (points)")
        ax2.grid(True, alpha=0.2)
        
        # Formatter for X axis to show readable time
        tick_indices = np.linspace(0, len(ohlc) - 1, 15, dtype=int)
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels([ohlc.index[i].strftime('%H:%M:%S') for i in tick_indices], rotation=30, ha='right')
        
        plt.tight_layout()
        
        # Save figure to the news_plots directory
        out_name = f"{symbol}_{impact}_{safe_filename(event_name)}_{edt.strftime('%Y%m%d_%H%M')}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        generated_files.append(out_path)

# 3. Zip the generated files
zip_path = os.path.join(DATA_DIR, "High_Medium_Impact_News_Charts.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in generated_files:
        zf.write(f, os.path.basename(f))

print(f"\nSuccess! Generated {len(generated_files)} charts.")
print(f"They are saved in the '{OUT_DIR}' folder and compressed into '{zip_path}'.")