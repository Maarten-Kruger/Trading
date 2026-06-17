import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
FILE_PATH = "EURUSD_Jan.csv"  # Replace with your MetaTrader 5 tick data CSV
START_HOUR = 8
END_HOUR = 18
CANDLE = '15min'
AMOUNT_CANDLES = 1000
START_DATE = '2026-01-08'

def load_and_prepare_data(filepath):
    print("Loading and parsing tick data...")
    
    # 1. Use STRICT tab separator so empty columns don't shift the data
    df = pd.read_csv(filepath, sep='\t')
    
    # Combine DATE and TIME into a single datetime index
    df['Datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S.%f')
    df.set_index('Datetime', inplace=True)
    
    # 2. Forward-fill the Bid and Ask columns. 
    # If MT5 leaves an Ask blank because only the Bid updated, this fills in the last known Ask.
    df['<BID>'] = df['<BID>'].ffill()
    df['<ASK>'] = df['<ASK>'].ffill()
    
    # Calculate Mid-price
    df['Mid'] = (df['<BID>'] + df['<ASK>']) / 2
    
    # 3. Aggregate to 1-second OHLC
    print("Resampling to 1-second intervals...")
    df_ohlc = df['Mid'].resample('1s').ohlc()
    df_ohlc['volume'] = df['<VOLUME>'].resample('1s').sum().fillna(0)
    
    # 4. Forward fill empty seconds (0 volume)
    df_ohlc['close'] = df_ohlc['close'].ffill()
    df_ohlc['open'] = df_ohlc['open'].fillna(df_ohlc['close'])
    df_ohlc['high'] = df_ohlc['high'].fillna(df_ohlc['close'])
    df_ohlc['low'] = df_ohlc['low'].fillna(df_ohlc['close'])
    
    # 5. Filter specific hours and remove weekends
    df_1s = df_ohlc.between_time(f'{START_HOUR:02d}:00', f'{END_HOUR:02d}:00')
    df_1s = df_1s[df_1s.index.dayofweek < 5] # 0-4 are Monday-Friday
    
    return df_1s

def generate_shuffled_dataset(df_1s):
    print("Aggregating to 1-Minute blocks and generating a CONTINUOUS walk...")
    
    # 1. Aggregate 1s to 1m
    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_1m = df_1s.resample('1min').agg(ohlc_dict).dropna()
    
    # 2. Calculate Close-to-Close differences for the M1 blocks
    # We group by date to ensure real-market overnight price gaps are NOT included as a 'movement'
    df_1m['close_diff'] = df_1m.groupby(df_1m.index.date)['close'].diff().fillna(0)
    
    # 3. Calculate Open, High, Low relative to the Close
    df_1m['rel_o'] = df_1m['open'] - df_1m['close']
    df_1m['rel_h'] = df_1m['high'] - df_1m['close']
    df_1m['rel_l'] = df_1m['low'] - df_1m['close']
    
    shapes = df_1m[['close_diff', 'rel_o', 'rel_h', 'rel_l', 'volume']]
    
    # Completely shuffle the 1-minute blocks
    shuffled_shapes = shapes.sample(frac=1).reset_index(drop=True)
    shuffled_shapes.index = df_1m.index  
    
    # 4. Build the continuous price walk
    # Anchor the very first candle of the ENTIRE dataset
    first_real_open = df_1m['open'].iloc[0]
    
    # Anchor the first close relative to the first shuffled open
    first_rel_o = shuffled_shapes['rel_o'].iloc[0]
    first_close = first_real_open - first_rel_o
    
    diffs = shuffled_shapes['close_diff'].copy()
    diffs.iloc[0] = 0 
    
    # ONE single cumulative sum across the whole dataset (Continuous Price Stitching)
    synth_close = first_close + diffs.cumsum()
    
    # Re-attach the Open, High, and Low
    synth_open = synth_close + shuffled_shapes['rel_o']
    synth_high = synth_close + shuffled_shapes['rel_h']
    synth_low = synth_close + shuffled_shapes['rel_l']
    
    df_shuffled = pd.DataFrame({
        'open': synth_open,
        'high': synth_high,
        'low': synth_low,
        'close': synth_close,
        'volume': shuffled_shapes['volume']
    })
    
    return df_shuffled



def process_and_graph():
    # 1. Prepare base data
    df_1s_ordered = load_and_prepare_data(FILE_PATH)
    df_1s_shuffled = generate_shuffled_dataset(df_1s_ordered)
    
    print(f"Aggregating to {CANDLE} candles...")
    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    
    # 2. Aggregate to M15 (dropna removes the overnight hours created by resampling)
    m15_orig = df_1s_ordered.resample(CANDLE).agg(ohlc_dict).dropna()
    m15_shuf = df_1s_shuffled.resample(CANDLE).agg(ohlc_dict).dropna()
    
    # 3. Filter by START_DATE and AMOUNT_CANDLES
    start_dt = pd.to_datetime(START_DATE)
    
    m15_orig = m15_orig[m15_orig.index >= start_dt].head(AMOUNT_CANDLES)
    m15_shuf = m15_shuf[m15_shuf.index >= start_dt].head(AMOUNT_CANDLES)
    
    print("Generating interactive charts...")
    # 4. Plotting
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Original Market Data (M15)", "Shuffled Synthetic Data (M15)"),
                        vertical_spacing=0.05)

    # Add Original Candlesticks
    fig.add_trace(go.Candlestick(x=m15_orig.index,
                                 open=m15_orig['open'], high=m15_orig['high'],
                                 low=m15_orig['low'], close=m15_orig['close'],
                                 name='Original'), row=1, col=1)

    # Add Shuffled Candlesticks
    fig.add_trace(go.Candlestick(x=m15_shuf.index,
                                 open=m15_shuf['open'], high=m15_shuf['high'],
                                 low=m15_shuf['low'], close=m15_shuf['close'],
                                 name='Shuffled'), row=2, col=1)

    # 5. Formatting the layout
    fig.update_layout(height=900, title_text="M15 OHLC Generation: Ordered vs Shuffled 1-Second Ticks",
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                      template="plotly_dark")
    
    
    # Hide weekends AND overnight gaps (17:00 to 08:00) to visually stitch the timeline
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]), 
        dict(bounds=[END_HOUR, START_HOUR], pattern="hour")
    ])
    
    # Add thin vertical lines at 08:00 for each day present in the sliced dataset
    unique_days = m15_orig.index.normalize().unique()
    for day in unique_days:
        start_of_day = day + pd.Timedelta(hours=START_HOUR)
        fig.add_vline(x=start_of_day, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)", row='all', col='all')

    fig.show()
'''

def process_and_graph():
    # 1. Prepare base data
    df_1s_ordered = load_and_prepare_data(FILE_PATH)
    df_1s_shuffled = generate_shuffled_dataset(df_1s_ordered)
    
    print(f"Aggregating to {CANDLE} candles...")
    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    
    # 2. Aggregate to M15 (dropna removes the overnight hours created by resampling)
    m15_orig = df_1s_ordered.resample(CANDLE).agg(ohlc_dict).dropna()
    m15_shuf = df_1s_shuffled.resample(CANDLE).agg(ohlc_dict).dropna()
    
    # 3. Filter by START_DATE and AMOUNT_CANDLES
    start_dt = pd.to_datetime(START_DATE)
    
    m15_orig = m15_orig[m15_orig.index >= start_dt].head(AMOUNT_CANDLES)
    m15_shuf = m15_shuf[m15_shuf.index >= start_dt].head(AMOUNT_CANDLES)
    
    print("Generating continuous interactive charts...")
    # 4. Plotting
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Original Market Data (M15) - Continuous", "Shuffled Synthetic Data (M15) - Continuous"),
                        vertical_spacing=0.05)

    # Add Original Candlesticks
    fig.add_trace(go.Candlestick(x=m15_orig.index,
                                 open=m15_orig['open'], high=m15_orig['high'],
                                 low=m15_orig['low'], close=m15_orig['close'],
                                 name='Original'), row=1, col=1)

    # Add Shuffled Candlesticks
    fig.add_trace(go.Candlestick(x=m15_shuf.index,
                                 open=m15_shuf['open'], high=m15_shuf['high'],
                                 low=m15_shuf['low'], close=m15_shuf['close'],
                                 name='Shuffled'), row=2, col=1)

    # 5. Formatting the layout
    fig.update_layout(height=900, title_text="M15 OHLC Generation: Ordered vs Shuffled 1-Minute Blocks",
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                      template="plotly_dark")
    
    # 6. Hide weekends AND overnight gaps to make the market completely continuous
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[END_HOUR, START_HOUR], pattern="hour")  # Hide hours outside trading session
        ]
    )
    
    # Add thin vertical lines at the start of each new trading session (08:00)
    unique_days = m15_orig.index.normalize().unique()
    for day in unique_days:
        start_of_day = day + pd.Timedelta(hours=START_HOUR)
        fig.add_vline(x=start_of_day, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)", row='all', col='all')

    fig.show()
    '''

if __name__ == "__main__":
    process_and_graph()