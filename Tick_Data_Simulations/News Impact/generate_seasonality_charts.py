import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_DIR = "."
OUT_DIR = os.path.join(DATA_DIR, "seasonality_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# List of the tick data files to process
files = [
    'EURUSDm_202606050000_202606142203.csv',
    'GBPUSDm_202605312106_202606142203.csv',
    'USDJPYm_202605312105_202606142203.csv',
    'NZDUSDm_202605312105_202606142203.csv',
    'AUDUSDm_202605312105_202606142203.csv',
    'USDCADm_202605312105_202606142203.csv',
    'USDCHFm_202605312106_202606142203.csv',
    'EURUSD_Jan.csv',
    'BTCUSDm_Jan.csv',
    'BTCUSDm_202601012100_202606172041.csv' 
]

# RVI Parameters
SD_PERIOD = 10
EMA_PERIOD = 14
TIMEFRAME = '15min'  # Resample tick data to 15-minute bars (lowercase for modern Pandas)

for file_name in files:
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: {file_name} not found. Skipping.")
        continue
        
    print(f"Loading and processing: {file_name}...")
    symbol = file_name.split('m_')[0]  # Extract pair name like 'EURUSD'
    
    try:
        # Load the tick data
        tdf = pd.read_csv(file_path, sep='\t')
        tdf.columns = [c.strip() for c in tdf.columns]
        
        # Parse datetime
        tdf['Datetime'] = pd.to_datetime(tdf['<DATE>'] + ' ' + tdf['<TIME>'], format='%Y.%m.%d %H:%M:%S.%f')
        tdf.set_index('Datetime', inplace=True)
        
        # Calculate Mid Price
        tdf['Mid'] = (tdf['<ASK>'] + tdf['<BID>']) / 2.0
        
        # Resample into regular OHLC bars
        ohlc = tdf['Mid'].resample(TIMEFRAME).ohlc().dropna()
        
        if ohlc.empty:
            print(f"Not enough data to resample for {symbol}.")
            continue
            
        # 1. Calculate Volatility (High - Low)
        ohlc['Volatility'] = ohlc['high'] - ohlc['low']
        
        # 2. Calculate RVI (Relative Volatility Index)
        close_diff = ohlc['close'].diff()
        sd = ohlc['close'].rolling(window=SD_PERIOD).std()
        
        # Up and Down standard deviations
        up = pd.Series(np.where(close_diff > 0, sd, 0), index=ohlc.index).fillna(0)
        down = pd.Series(np.where(close_diff < 0, sd, 0), index=ohlc.index).fillna(0)
        
        # Smooth with EMA
        ema_up = up.ewm(span=EMA_PERIOD, adjust=False).mean()
        ema_down = down.ewm(span=EMA_PERIOD, adjust=False).mean()
        
        # RVI Formula
        rvi = 100 * (ema_up / (ema_up + ema_down))
        ohlc['RVI'] = rvi.fillna(50) # Fill NaN with neutral 50
        
        # Clean up any remaining NaNs from rolling windows
        ohlc.dropna(inplace=True)
        
        # 3. Group by Time of Day to find intraday seasonality averages
        ohlc['TimeOfDay'] = ohlc.index.time
        grouped = ohlc.groupby('TimeOfDay')[['Volatility', 'RVI']].mean()
        
        # Convert Time to string for easier plotting on the X-axis
        x_labels = [t.strftime('%H:%M') for t in grouped.index]
        x = np.arange(len(grouped))
        
        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Top Panel: Average Volatility
        ax1.plot(x, grouped['Volatility'], color='darkorange', linewidth=2)
        ax1.set_title(f"{symbol} - Average Intraday Seasonality ({TIMEFRAME} bars)", fontsize=14, fontweight='bold')
        ax1.set_ylabel(f"Avg Volatility (High-Low spread)")
        ax1.grid(True, alpha=0.3)
        
        # Bottom Panel: Average RVI
        ax2.plot(x, grouped['RVI'], color='teal', linewidth=2)
        ax2.axhline(50, color='gray', linestyle='--', alpha=0.7) # Neutral line
        ax2.set_ylabel("Average RVI")
        ax2.set_xlabel("Time of Day (Server Time)")
        ax2.grid(True, alpha=0.3)
        
        # Format X-axis to show legible time intervals
        tick_indices = np.linspace(0, len(grouped) - 1, 24, dtype=int) # Show roughly 24 ticks
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels([x_labels[i] for i in tick_indices], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save Plot
        out_name = f"{symbol}_Seasonality_Volatility_RVI.png"
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved plot: {out_name}")
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print(f"\nAll done! Seasonality charts saved in the '{OUT_DIR}' folder.")