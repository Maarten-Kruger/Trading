import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import config
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def _generate_single_image(args):
    """Helper function to generate a single image for multiprocessing."""
    plot_df, signal, trigger_pos, image_path, trigger_time = args

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
    return True

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
        df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M')

    # ---------------------------------------------------------------------------------------------------
    # REQUIREMENT FOR CUSTOM STRATEGY CODE:
    # 1. Implement your strategy logic in this function to calculate necessary indicators.
    # 2. Append new columns to the dataframe `df` containing the results of your indicator calculations.
    # 3. Return the fully preprocessed dataframe.
    #
    # OPTIMIZATION GUIDELINES:
    # - Avoid using `df.apply()` or row-by-row iteration (e.g., `iterrows()`), as they are extremely slow on large datasets.
    # - Strictly use vectorized operations with Pandas and NumPy for calculations.
    # - If complex looping is absolutely necessary, consider using Numba (`@njit`) to compile the logic.
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
    #
    # OPTIMIZATION GUIDELINES:
    # - Strongly prefer vectorized boolean conditions to find triggers instead of iterating through the dataframe with loops.
    # - Example: `up_indices = df.index[(df['Close'] > df['Open']) & (df['Volume'] > 100)]`
    # - Use list comprehensions or fast array operations to construct the final list of tuples.
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
    existing_image_names = set()
    if os.path.exists(labels_file) and os.path.getsize(labels_file) > 0:
        try:
            existing_df = pd.read_csv(labels_file)
            if 'Image_Name' in existing_df.columns:
                existing_image_names = set(existing_df['Image_Name'].tolist())
        except:
            pass

    generated_count = 0

    total_triggers = len(trigger_indices)

    # Pre-set index for mplfinance to avoid setting it inside the loop
    df_indexed = df.set_index('Time')

    new_labels = []
    tasks = []

    # First pass: collect tasks and update labels
    for i, trigger_info in enumerate(trigger_indices):
        if isinstance(trigger_info, tuple):
            trigger_idx, signal = trigger_info
        else:
            trigger_idx = trigger_info
            signal = "Unknown"

        start_idx = max(0, trigger_idx - config.CANDLES_BEFORE_TRIGGER)
        end_idx = min(len(df) - 1, trigger_idx + config.CANDLES_AFTER_TRIGGER)

        if start_idx >= end_idx:
            continue

        image_name = f"sample_{i+1:03d}.png"
        image_path = os.path.join(output_dir, image_name)
        trigger_time = df['Time'].iloc[trigger_idx]

        if image_name not in existing_image_names:
            new_labels.append({
                "Image_Name": image_name,
                "Time": trigger_time,
                "Signal": signal,
                "Label": "Unclassified"
            })
            existing_image_names.add(image_name)

        if not os.path.exists(image_path):
            plot_df = df_indexed.iloc[start_idx:end_idx+1]
            trigger_pos = trigger_idx - start_idx
            tasks.append((plot_df, signal, trigger_pos, image_path, trigger_time))

    # Determine optimal number of workers
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    # Process images in parallel
    if tasks:
        completed = total_triggers - len(tasks)
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(_generate_single_image, task): task for task in tasks}

            for future in as_completed(futures):
                try:
                    future.result()
                    generated_count += 1
                except Exception as e:
                    print(f"Error generating image: {e}")

                completed += 1
                if progress_bar is not None:
                    progress_bar.progress(completed / total_triggers)
                if progress_text is not None:
                    progress_text.text(f"Generating image {completed} of {total_triggers}...")
    else:
        # If no tasks to process, still update the progress bar to 100%
        if progress_bar is not None:
            progress_bar.progress(1.0)
        if progress_text is not None:
            progress_text.text(f"All {total_triggers} images already exist.")

    # Append all new labels to CSV at once
    if new_labels:
        new_labels_df = pd.DataFrame(new_labels)
        write_header = not os.path.exists(labels_file) or os.path.getsize(labels_file) == 0
        new_labels_df.to_csv(labels_file, mode='a', header=write_header, index=False)

    return output_dir, generated_count
