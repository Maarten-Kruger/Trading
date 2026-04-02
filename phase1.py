import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

def load_and_preprocess_data(file_path):
    """
    Template function to load CSV data and apply preprocessing.
    """
    df = pd.read_csv(file_path)

    # Ensure standard column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # In case Time is the first column but labeled differently
    if 'Time' not in df.columns and len(df.columns) > 0:
        df.rename(columns={df.columns[0]: 'Time'}, inplace=True)

    # Convert Time to datetime
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])

    # Calculate dummy indicator (Simple Moving Average crossover)
    # --- DUMMY LOGIC START (Remove for empty template) ---
    df['SMA_Fast'] = df['Close'].rolling(window=5).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=10).mean()
    # --- DUMMY LOGIC END ---

    return df

def find_triggers(df):
    """
    Template function to find the exact indices where the setup occurs.
    Returns a list of integer indices corresponding to the trigger candles.
    """
    trigger_indices = []

    # --- DUMMY LOGIC START (Remove for empty template) ---
    # Trigger when Fast SMA crosses above Slow SMA
    for i in range(1, len(df)):
        if pd.isna(df['SMA_Slow'].iloc[i]):
            continue

        if (df['SMA_Fast'].iloc[i] > df['SMA_Slow'].iloc[i]) and \
           (df['SMA_Fast'].iloc[i-1] <= df['SMA_Slow'].iloc[i-1]):
            trigger_indices.append(i)
    # --- DUMMY LOGIC END ---

    return trigger_indices

def generate_images(df, trigger_indices, file_name):
    """
    Generates static images for each trigger and saves them in a structured folder.
    """
    strategy_name = config.STRATEGY_NAME
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    output_dir = f"{strategy_name}_{base_file_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labels_file = os.path.join(output_dir, 'labels.csv')
    if not os.path.exists(labels_file):
        # Create empty labels file if it doesn't exist
        with open(labels_file, 'w') as f:
            f.write("Image_Name,Time,Label\n")

    # Read existing labels to avoid overwriting and to know what to append
    existing_labels = pd.DataFrame()
    if os.path.exists(labels_file) and os.path.getsize(labels_file) > 0:
        try:
            existing_labels = pd.read_csv(labels_file)
        except:
            pass

    generated_count = 0
    new_labels = []

    for i, trigger_idx in enumerate(trigger_indices):
        start_idx = max(0, trigger_idx - config.CANDLES_BEFORE_TRIGGER)
        end_idx = min(len(df) - 1, trigger_idx + config.CANDLES_AFTER_TRIGGER)

        # Ensure we have enough data to plot
        if start_idx >= end_idx:
            continue

        window_df = df.iloc[start_idx:end_idx+1]

        # Format the image name
        image_name = f"sample_{i+1:03d}.png"
        image_path = os.path.join(output_dir, image_name)

        trigger_time = df['Time'].iloc[trigger_idx]

        # If it's not in existing labels, we add it to the list to append later
        if existing_labels.empty or image_name not in existing_labels['Image_Name'].values:
            new_labels.append({"Image_Name": image_name, "Time": trigger_time, "Label": "Unclassified"})

        # Only generate image if it doesn't exist
        if os.path.exists(image_path):
            continue

        # Plotting
        # Template Section: make_subplots can be adjusted to add indicator rows below
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                            row_width=[0.2, 0.7])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=window_df['Time'],
                                     open=window_df['Open'],
                                     high=window_df['High'],
                                     low=window_df['Low'],
                                     close=window_df['Close'],
                                     name='Price'), row=1, col=1)

        # Highlight trigger candle
        trigger_row = df.iloc[trigger_idx]
        fig.add_trace(go.Scatter(x=[trigger_row['Time']],
                                 y=[trigger_row['High']],
                                 mode='markers',
                                 marker=dict(symbol='triangle-down', size=15, color='black'),
                                 name='Trigger'), row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(x=window_df['Time'], y=window_df['Volume'], name='Volume'), row=2, col=1)

        # Update layout
        fig.update_layout(title=f'Setup at {trigger_time}', xaxis_rangeslider_visible=False)

        # Save image
        fig.write_image(image_path, engine="kaleido")
        generated_count += 1

    # Append new labels
    if new_labels:
        new_df = pd.DataFrame(new_labels)
        if existing_labels.empty:
            new_df.to_csv(labels_file, index=False)
        else:
            new_df.to_csv(labels_file, mode='a', header=False, index=False)

    return output_dir, generated_count
