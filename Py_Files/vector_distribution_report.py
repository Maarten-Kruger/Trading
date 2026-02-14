
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
SMOOTHING_WINDOW = 573  # Controls the smoothness of the moving average line
# ---------------------

def read_csv_robust(filepath):
    """
    Reads a CSV file robustly, handling different separators and decimal formats.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            if not header: return pd.DataFrame()

        sep = ';' if ';' in header else ','
        decimal = ',' if sep == ';' else '.'

        # Read with detected separator
        df = pd.read_csv(filepath, sep=sep, decimal=decimal, engine='c')
    except:
        try:
             # Fallback to python engine
             df = pd.read_csv(filepath, sep=None, engine='python')
        except:
             return pd.DataFrame()

    # Convert numeric columns that might be objects due to comma decimals
    cols_to_numeric = ['Result', 'Profit', 'Trades']
    for c in cols_to_numeric:
        if c in df.columns and df[c].dtype == 'object':
             df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    # Convert other object columns that look numeric
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        try:
            # Check if column is actually numeric
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='ignore')
        except:
            pass

    return df

def get_date_from_filename(filename):
    """
    Extracts a date for sorting from the filename.
    Looks for an 8-digit sequence starting with '20'.
    """
    base = os.path.basename(filename)
    parts = base.replace('.', ' ').replace('_', ' ').split()
    for p in parts:
        if len(p) == 8 and p.startswith("20") and p.isdigit():
            return int(p)
    return float('inf')

def generate_plot(results, filename, window_size, y_min=None, y_max=None):
    """
    Generates a matplotlib plot for the results distribution.
    """
    plt.figure(figsize=(12, 6))

    # X-axis is just the index (Rank)
    x = np.arange(len(results))

    # Plot the raw data as a filled area/bar
    plt.fill_between(x, results, color='royalblue', alpha=0.8, label='Result')

    # Calculate and plot smooth line (Moving Average)
    if len(results) > window_size:
        smooth = pd.Series(results).rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(x, smooth, color='orange', linewidth=2, label=f'Trend ({window_size} avg)')

    plt.title(f"Result Distribution: {filename}")
    plt.xlabel("Vector Rank (Based on First File)")
    plt.ylabel("Result")

    # Set fixed Y-axis scale if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    smooth_values = smooth.values if 'smooth' in locals() else None
    return img_base64, smooth_values

def generate_overlay_plot(trends_list, y_min=None, y_max=None):
    """
    Generates a matplotlib plot with multiple overlayed trend lines.
    Older lines have lower opacity.
    """
    plt.figure(figsize=(12, 6))

    num_trends = len(trends_list)

    # Check if there is anything to plot
    if num_trends > 0:
        # Assuming all trends have the same length (x-axis)
        # Verify valid trend exists
        valid_trends = [t for t in trends_list if t is not None]
        if valid_trends:
            x = np.arange(len(valid_trends[0]))

            for i, trend in enumerate(trends_list):
                if trend is None: continue

                # Calculate alpha: increasing for newer lines
                # Range alpha from 0.2 to 1.0
                alpha = 0.2 + (0.8 * (i / (num_trends - 1))) if num_trends > 1 else 1.0

                plt.plot(x, trend, color='orange', linewidth=2, alpha=alpha)

    plt.title(f"Previous {num_trends} Trends Overlay")
    plt.xlabel("Vector Rank (Based on First File)")
    plt.ylabel("Result")

    # Set fixed Y-axis scale if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64

def main():
    # Use Agg backend for headless environment
    plt.switch_backend('Agg')

    print("--- Vector Distribution Report Generator ---")

    # Get target directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("Enter the path to the folder containing CSVs: ").strip()

    if not target_dir or not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # Find and sort CSV files
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    csv_files.sort(key=get_date_from_filename)
    print(f"Found {len(csv_files)} files. Processing...")

    # 1. Process First File (Master Order)
    first_file = csv_files[0]
    print(f"Reading Master File: {os.path.basename(first_file)}")

    master_df = read_csv_robust(first_file)
    if master_df.empty or 'Trades' not in master_df.columns or 'Result' not in master_df.columns:
        print("Error: First file is invalid (missing 'Trades' or 'Result' columns).")
        return

    # Determine Y-axis scaling based on the master file
    y_max = master_df['Result'].max()
    y_min = master_df['Result'].min()

    # Add a little padding (e.g., 5%)
    y_range = y_max - y_min
    y_max += y_range * 0.05
    y_min -= y_range * 0.05

    print(f"Fixed Y-Axis Scale established: [{y_min:.2f}, {y_max:.2f}]")

    # Identify Vector Columns (everything after 'Trades')
    cols = list(master_df.columns)
    try:
        trades_idx = cols.index('Trades')
        vector_cols = cols[trades_idx+1:]
    except ValueError:
        print("Error: 'Trades' column not found.")
        return

    if not vector_cols:
        print("Error: No vector columns found after 'Trades'.")
        return

    print(f" identified {len(vector_cols)} vector parameters.")

    # Sort Master DF by Result Descending to establish rank
    master_df.sort_values(by='Result', ascending=False, inplace=True)

    # Keep only vector columns in the master DataFrame for merging
    # We add a 'MasterRank' to preserve order after merge if needed,
    # but since we iterate through the master list, we can just left join.
    master_vectors = master_df[vector_cols].copy()
    master_vectors.reset_index(drop=True, inplace=True)
    master_vectors['Master_Index'] = master_vectors.index # 0 to N

    # 2. Process All Files
    BACK_GRAPH = 10
    all_trends = [] # Stores smooth trend lines (numpy arrays)

    # We will build HTML parts dynamically
    html_standard_rows = ""
    html_sliding_rows = ""

    first_filename = os.path.basename(csv_files[0]) if csv_files else 'N/A'
    processed_count = 0

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        # Read current file
        current_df = read_csv_robust(filepath)

        if current_df.empty:
            print(f"Warning: {filename} is empty or unreadable. Skipping.")
            continue

        # Ensure 'Result' and vector cols exist
        if 'Result' not in current_df.columns:
            print(f"Warning: 'Result' column missing in {filename}. Skipping.")
            continue

        # Merge current data onto master vectors
        # Left join on vector columns
        # This aligns current results to the Master Rank
        merged_df = pd.merge(master_vectors, current_df, on=vector_cols, how='left')

        # Fill missing results with 0
        merged_df['Result'] = merged_df['Result'].fillna(0.0)

        # Sort by Master Index to ensure X-axis is consistent
        merged_df.sort_values(by='Master_Index', ascending=True, inplace=True)

        # Get Y-values
        results = merged_df['Result'].values

        # Generate Plot with fixed scaling and global smoothing window
        img_std, smooth_data = generate_plot(results, filename, SMOOTHING_WINDOW, y_min=y_min, y_max=y_max)

        processed_count += 1

        # Add to Standard View HTML
        html_standard_rows += f"""
            <div class="plot-container">
                <h3>{filename}</h3>
                <img src="data:image/png;base64,{img_std}" alt="Plot for {filename}">
            </div>
        """

        # Handle Sliding Window Logic
        all_trends.append(smooth_data)

        if len(all_trends) > BACK_GRAPH:
            # Get previous BACK_GRAPH trends (excluding current)
            prev_trends = all_trends[-(BACK_GRAPH + 1) : -1]

            # Generate Overlay Plot
            img_overlay = generate_overlay_plot(prev_trends, y_min=y_min, y_max=y_max)

            # Add to Sliding View HTML
            html_sliding_rows += f"""
            <div class="plot-container">
                <h3>{filename} (Sliding Window Analysis)</h3>
                <div style="display: flex; gap: 20px; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="margin-bottom: 5px;">Previous {BACK_GRAPH} Trends Overlay</h4>
                        <img src="data:image/png;base64,{img_overlay}" style="width: 100%; border: 1px solid #eee;">
                    </div>
                    <div style="flex: 1;">
                        <h4 style="margin-bottom: 5px;">Current Week</h4>
                        <img src="data:image/png;base64,{img_std}" style="width: 100%; border: 1px solid #eee;">
                    </div>
                </div>
            </div>
            """

    # 3. Generate HTML Report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vector Distribution Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .plot-container {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: #fff; }}
            img {{ max-width: 100%; height: auto; }}
            h1 {{ color: #333; }}
            h3 {{ color: #555; margin-top: 0; }}
            details {{ margin-bottom: 20px; text-align: left; border: 1px solid #ccc; border-radius: 5px; padding: 10px; }}
            summary {{ cursor: pointer; font-weight: bold; font-size: 1.2em; padding: 10px; background-color: #f9f9f9; }}
            summary:hover {{ background-color: #f0f0f0; }}
            .section-content {{ padding: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vector Distribution Report</h1>
            <p>Results ordered by vector rank in the first file ({first_filename}).</p>
            <p>Fixed Scale (First File): [{y_min:.2f}, {y_max:.2f}] | Smoothing Window: {SMOOTHING_WINDOW}</p>
            <p>Total Files: {processed_count} | Back Graph Window: {BACK_GRAPH}</p>

            <details open>
                <summary>Standard View</summary>
                <div class="section-content">
                    {html_standard_rows}
                </div>
            </details>

            <details>
                <summary>Sliding Window View (Last {BACK_GRAPH} Overlay)</summary>
                <div class="section-content">
                    {html_sliding_rows}
                </div>
            </details>
        </div>
    </body>
    </html>
    """

    output_path = os.path.join(target_dir, "Vector_Distribution_Report.html")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    main()
