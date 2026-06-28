import pandas as pd
import plotly.graph_objects as go

# 1. Load data
df = pd.read_csv('EURUSD_Jan.csv', sep='\t')

# 2. Process datetime
df['Datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S.%f')
df.set_index('Datetime', inplace=True)

# 3. Resample to 1min OHLC using <BID>
ohlc_1m = df['<BID>'].resample('1min').ohlc()
ohlc_1m.dropna(inplace=True)

chunk_size = 50
num_chunks = len(ohlc_1m) // chunk_size

# 4. Create an interactive Plotly figure
fig = go.Figure()

for i in range(num_chunks):
    chunk = ohlc_1m.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
    normalized_path = chunk['close'] - chunk['close'].iloc[0]
    
    # We use Scattergl (WebGL) for better rendering performance with hundreds of lines
    fig.add_trace(go.Scattergl(
        x=list(range(chunk_size)),
        y=normalized_path,
        mode='lines',
        line=dict(color='rgba(0, 128, 128, 0.1)'), # Teal color with 5% opacity
        showlegend=False,
        hoverinfo='y', # Only show the Y value on hover to reduce clutter
        name=f'Chunk {i}'
    ))

# 5. Format the chart layout
fig.update_layout(
    title='EURUSD 50-Candle Overlays (1-Minute Intervals)',
    xaxis_title='Candle Sequence (0 to 49)',
    yaxis_title='Price Change from Chunk Start',
    template='plotly_white',
    hovermode='x' # Aligns hover tooltips vertically
)

# 6. Save directly to a playable HTML file
fig.write_html('interactive_monte_carlo.html')

print(f"Successfully generated 'interactive_monte_carlo.html' containing {num_chunks} overlapping paths.")