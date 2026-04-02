import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
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

    # ---------------------------------------------------------------------------------------------------
    # REQUIREMENT FOR CUSTOM STRATEGY CODE:
    # 1. Implement your strategy logic in this function to calculate necessary indicators.
    # 2. Append new columns to the dataframe `df` containing the results of your indicator calculations.
    # 3. Return the fully preprocessed dataframe.
    # ---------------------------------------------------------------------------------------------------

    # Calculate indicator/setup here

    return df

def find_triggers(df):
    """
    Template function to find the exact indices where the setup occurs.
    Returns a list of tuples containing integer indices and the signal direction ("Up" or "Down").
    Example: [(10, "Up"), (25, "Down"), (42, "Up")]
    """
    trigger_indices = []

    # ---------------------------------------------------------------------------------------------------
    # REQUIREMENT FOR CUSTOM STRATEGY CODE:
    # 1. Iterate through the dataframe or use vectorized operations to find where your setup triggers.
    # 2. For each trigger, determine if it is an "Up" signal (expecting price to rise) or "Down" signal.
    # 3. Append a tuple of (index, "Signal_Direction") to the `trigger_indices` list.
    #    - `index` must be an integer representing the row index in the dataframe.
    #    - `Signal_Direction` must be a string, exactly "Up" or "Down".
    # ---------------------------------------------------------------------------------------------------

    # Place the trigger logic here. For example, let's say we want to trigger on a simple condition:
    # if Close > Open for a bullish candle, we consider it a trigger (this is just an example, replace with actual logic)

    return trigger_indices

def generate_images(df, trigger_indices, file_name, progress_bar=None, progress_text=None):
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
            f.write("Image_Name,Time,Signal,Label\n")

    # Read existing labels to avoid overwriting and to know what to append
    existing_labels = pd.DataFrame()
    if os.path.exists(labels_file) and os.path.getsize(labels_file) > 0:
        try:
            existing_labels = pd.read_csv(labels_file)
        except:
            pass

    generated_count = 0
    new_labels = []

    total_triggers = len(trigger_indices)

    for i, trigger_info in enumerate(trigger_indices):
        # Handle both old format (just integer index) and new format (tuple with index and signal)
        if isinstance(trigger_info, tuple):
            trigger_idx, signal = trigger_info
        else:
            trigger_idx = trigger_info
            signal = "Unknown"

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
            new_labels.append({"Image_Name": image_name, "Time": trigger_time, "Signal": signal, "Label": "Unclassified"})

        # Only generate image if it doesn't exist
        if os.path.exists(image_path):
            if progress_bar is not None:
                progress_bar.progress((i + 1) / total_triggers)
            if progress_text is not None:
                progress_text.text(f"Generating image {i + 1} of {total_triggers}...")
            continue

        # Plotting using mplfinance
        # Set Time as index for mplfinance
        plot_df = window_df.set_index('Time')

        # Create an addplot for the trigger marker
        # Determine the position of the trigger candle in the window
        trigger_pos = trigger_idx - start_idx

        # We need an array of the same length as plot_df, with nan everywhere except the trigger
        marker_data = [float('nan')] * len(plot_df)
        if 0 <= trigger_pos < len(plot_df):
            # Place marker slightly above the high or below the low based on signal
            if signal == "Up":
                marker_data[trigger_pos] = float(plot_df.iloc[trigger_pos]['Low']) * 0.9995
                marker = '^'
                color = 'green'
            elif signal == "Down":
                marker_data[trigger_pos] = float(plot_df.iloc[trigger_pos]['High']) * 1.0005
                marker = 'v'
                color = 'red'
            else:
                marker_data[trigger_pos] = float(plot_df.iloc[trigger_pos]['High']) * 1.0005
                marker = 'v'
                color = 'black'

        ap = mpf.make_addplot(marker_data, type='scatter', markersize=100, marker=marker, color=color)

        # Save image directly using mpf.plot
        mpf.plot(plot_df,
                 type='candle',
                 volume=True,
                 addplot=ap,
                 title=f'Setup at {trigger_time}',
                 savefig=dict(fname=image_path, dpi=100, bbox_inches='tight'),
                 style='yahoo',
                 warn_too_much_data=1000)

        plt.close('all') # Ensure resources are freed
        generated_count += 1

        if progress_bar is not None:
            progress_bar.progress((i + 1) / total_triggers)
        if progress_text is not None:
            progress_text.text(f"Generating image {i + 1} of {total_triggers}...")

    # Append new labels
    if new_labels:
        new_df = pd.DataFrame(new_labels)
        if existing_labels.empty:
            new_df.to_csv(labels_file, index=False)
        else:
            new_df.to_csv(labels_file, mode='a', header=False, index=False)

    return output_dir, generated_count
