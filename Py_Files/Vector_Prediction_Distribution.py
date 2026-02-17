
import os
import sys
import glob
import pandas as pd
import numpy as np
import warnings
import json
from sklearn.neighbors import KDTree

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
HYPERCUBE = 1           # Hypercube size (steps) for averaging neighbors
FILE_LOOKBACK = 5       # Number of past files to average for prediction
TOP_N = 100             # Number of top predicted vectors to evaluate
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

def main():
    print("--- Vector Prediction Distribution Generator ---")

    # Get target directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # Ask user if not provided
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
    print(f"Found {len(csv_files)} files.")

    if len(csv_files) <= FILE_LOOKBACK:
        print(f"Error: Not enough files ({len(csv_files)}) for lookback of {FILE_LOOKBACK}.")
        return

    # 1. Process Master File (First File)
    first_file = csv_files[0]
    print(f"Reading Master File: {os.path.basename(first_file)}")

    master_df = read_csv_robust(first_file)
    if master_df.empty or 'Trades' not in master_df.columns or 'Result' not in master_df.columns:
        print("Error: First file is invalid (missing 'Trades' or 'Result' columns).")
        return

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

    print(f"Identified {len(vector_cols)} vector parameters.")

    # Keep only vector columns in the master DataFrame for merging
    master_vectors = master_df[vector_cols].copy()
    master_vectors.reset_index(drop=True, inplace=True)
    master_vectors['Master_Index'] = master_vectors.index # 0 to N

    # Store global params list
    global_params = master_vectors[vector_cols].astype(str).agg(', '.join, axis=1).tolist()

    # --- Hypercube Neighbor Pre-computation ---
    print(f"Pre-computing Hypercube Neighbors (Size={HYPERCUBE})...")

    # 1. Map parameters to Grid Indices
    grid_indices_list = []
    for col in vector_cols:
        unique_vals = np.sort(master_vectors[col].unique())
        indices = np.searchsorted(unique_vals, master_vectors[col].values)
        grid_indices_list.append(indices)

    # Transpose to get (N_samples, N_features)
    grid_coords = np.vstack(grid_indices_list).T

    # 2. Build KDTree
    print("Building KDTree...")
    tree = KDTree(grid_coords, metric='chebyshev')

    # 3. Query Neighbors
    print("Querying neighbors...")
    master_neighbor_indices = tree.query_radius(grid_coords, r=HYPERCUBE)
    print("Neighbor pre-computation complete.")
    # -------------------------------------------

    # 2. Pre-compute Hypercube Averages for ALL files
    print("Pre-computing Hypercube Averages for all files...")

    all_file_hypercube_avgs = [] # List of numpy arrays
    all_file_results = []        # List of numpy arrays
    all_file_profits = []        # List of numpy arrays
    all_file_dates = []          # List of file dates/names

    valid_files_indices = []     # Indices of files that were successfully processed

    for i, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        # print(f"Processing: {filename}") # Reduce verbosity if needed

        current_df = read_csv_robust(filepath)

        if current_df.empty:
            print(f"Warning: {filename} is empty or unreadable. Skipping.")
            all_file_hypercube_avgs.append(None)
            all_file_results.append(None)
            all_file_profits.append(None)
            continue

        if 'Result' not in current_df.columns or 'Profit' not in current_df.columns:
            print(f"Warning: Missing columns in {filename}. Skipping.")
            all_file_hypercube_avgs.append(None)
            all_file_results.append(None)
            all_file_profits.append(None)
            continue

        # Merge onto master vectors to ensure alignment
        merged_df = pd.merge(master_vectors, current_df, on=vector_cols, how='left')
        merged_df['Result'] = merged_df['Result'].fillna(0.0)
        merged_df['Profit'] = merged_df['Profit'].fillna(0.0)

        # Sort by Master Index
        merged_df.sort_values(by='Master_Index', ascending=True, inplace=True)

        results = merged_df['Result'].values
        profits = merged_df['Profit'].values

        # Calculate Hypercube Average
        hypercube_avgs = np.zeros_like(results)
        for idx, neighbor_idxs in enumerate(master_neighbor_indices):
            hypercube_avgs[idx] = np.mean(results[neighbor_idxs])

        all_file_hypercube_avgs.append(hypercube_avgs)
        all_file_results.append(results)
        all_file_profits.append(profits)
        all_file_dates.append(filename) # Use filename for display

        valid_files_indices.append(i)

    print(f"Pre-computation complete for {len(valid_files_indices)} valid files.")

    # 3. Prediction Loop
    print("Running Prediction Logic...")

    html_rows = ""
    # Data storage for JS
    # Structure: { filename: { sorted_indices: [], params: [], pred_scores: [], act_results: [], act_profits: [], pred_ranks: [] } }
    report_data = {}

    # We iterate through valid files
    # We need at least FILE_LOOKBACK previous valid files to make a prediction for the current one

    # Map from 'valid_index' to 'real_index' (in all lists)
    # valid_files_indices[k] gives the index in all_file_* lists

    for k in range(FILE_LOOKBACK, len(valid_files_indices)):
        current_real_idx = valid_files_indices[k]
        current_filename = all_file_dates[current_real_idx]

        # Identify the previous FILE_LOOKBACK files
        # We take the immediately preceding valid files
        lookback_indices = [valid_files_indices[j] for j in range(k - FILE_LOOKBACK, k)]

        print(f"Predicting for: {current_filename} (using lookback indices: {lookback_indices})")

        # Gather past averages
        past_avgs_list = []
        for past_idx in lookback_indices:
            past_avgs_list.append(all_file_hypercube_avgs[past_idx])

        # Calculate Prediction Score (Average of past Hypercube Averages)
        # element-wise mean
        prediction_scores = np.mean(np.vstack(past_avgs_list), axis=0)

        # Select Top N Indices based on Prediction Score
        # argsort is ascending, so we take the last TOP_N
        top_n_indices = np.argsort(prediction_scores)[-TOP_N:][::-1]

        # Get Actual Data for Current File
        current_results = all_file_results[current_real_idx]
        current_profits = all_file_profits[current_real_idx]

        # Extract data for Top N
        selected_indices = top_n_indices
        selected_pred_scores = prediction_scores[selected_indices]
        selected_act_results = current_results[selected_indices]
        selected_act_profits = current_profits[selected_indices]
        selected_pred_ranks = np.arange(1, TOP_N + 1) # Rank 1 is highest score

        # SORT by ACTUAL RESULT (Descending) as requested
        # We need to sort the parallel arrays
        sort_order = np.argsort(selected_act_results)[::-1]

        sorted_indices = selected_indices[sort_order]
        sorted_pred_scores = selected_pred_scores[sort_order]
        sorted_act_results = selected_act_results[sort_order]
        sorted_act_profits = selected_act_profits[sort_order]
        sorted_pred_ranks = selected_pred_ranks[sort_order]

        # Prepare Data for JS
        # Store params just for these N vectors to save space
        sorted_params = [global_params[idx] for idx in sorted_indices]

        safe_fname = current_filename.replace('.', '_').replace(' ', '_')

        report_data[current_filename] = {
            'indices': sorted_indices.tolist(),
            'params': sorted_params,
            'pred_scores': [round(x, 4) for x in sorted_pred_scores.tolist()],
            'act_results': [round(x, 4) for x in sorted_act_results.tolist()],
            'act_profits': [round(x, 2) for x in sorted_act_profits.tolist()],
            'pred_ranks': sorted_pred_ranks.tolist()
        }

        # Generate HTML Row
        html_rows += f"""
        <div class="plot-container" id="container-{safe_fname}">
            <details>
                <summary onclick="lazyLoadCharts('{current_filename}', '{safe_fname}')">
                    {current_filename} (Click to Load Prediction Analysis)
                </summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 3; height: 500px; position: relative;">
                            <h4>Top {TOP_N} Predicted Vectors (Sorted by Actual Performance)</h4>
                            <canvas id="chart-{safe_fname}"></canvas>
                        </div>
                        <div style="flex: 1; padding: 20px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
                            <h4 id="info-title-{safe_fname}">Select a point...</h4>
                            <div id="info-content-{safe_fname}" style="font-size: 0.9em;">
                                <p>Click on a data point in the graph to see vector details.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </details>
        </div>
        """

    # Serialize Data
    json_report_data = json.dumps(report_data)

    # 4. Generate HTML File
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vector Prediction Distribution</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; background-color: #f4f4f9; }}
            .container {{ max-width: 1600px; margin: 0 auto; padding-bottom: 50px; }}
            .plot-container {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #fff; text-align: left; }}
            summary {{ cursor: pointer; font-weight: bold; font-size: 1.1em; padding: 10px; background-color: #eee; }}
            summary:hover {{ background-color: #e0e0e0; }}
            .section-content {{ padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vector Prediction Distribution Report</h1>
            <p><strong>Config:</strong> Lookback={FILE_LOOKBACK} | Hypercube={HYPERCUBE} | Top N={TOP_N}</p>
            <p>Showing distribution of the Top {TOP_N} vectors predicted by the strategy, sorted by their ACTUAL result in the current file.</p>

            {html_rows}
        </div>

        <script>
            const reportData = {json_report_data};
            const charts = {{}};

            function lazyLoadCharts(filename, safeFname) {{
                if (charts[safeFname]) return;

                const data = reportData[filename];
                if (!data) return;

                // X-Axis: 1 to N
                const labels = Array.from({{length: data.act_results.length}}, (v, k) => k + 1);

                const ctx = document.getElementById('chart-' + safeFname).getContext('2d');

                const chart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Actual Result',
                            data: data.act_results,
                            borderColor: 'royalblue',
                            backgroundColor: 'rgba(65, 105, 225, 0.2)',
                            borderWidth: 2,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            fill: true,
                            tension: 0.2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false,
                        }},
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const idx = context.dataIndex;
                                        return [
                                            `Result: ${{data.act_results[idx]}}`,
                                            `Profit: ${{data.act_profits[idx]}}`,
                                            `Pred Rank: ${{data.pred_ranks[idx]}}`,
                                            `Pred Score: ${{data.pred_scores[idx]}}`
                                        ];
                                    }}
                                }}
                            }}
                        }},
                        onClick: (e) => {{
                            const points = chart.getElementsAtEventForMode(e, 'nearest', {{ intersect: true }}, true);
                            if (points.length) {{
                                const firstPoint = points[0];
                                const idx = firstPoint.index;
                                updateInfoPanel(filename, safeFname, idx);
                            }}
                        }}
                    }}
                }});

                charts[safeFname] = chart;
            }}

            function updateInfoPanel(filename, safeFname, idx) {{
                const data = reportData[filename];
                const infoTitle = document.getElementById('info-title-' + safeFname);
                const infoContent = document.getElementById('info-content-' + safeFname);

                infoTitle.textContent = `Vector Detail (Sorted Rank #${{idx + 1}})`;

                infoContent.innerHTML = `
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Vector ID</td><td>${{data.indices[idx]}}</td></tr>
                        <tr><td><strong>Actual Profit</strong></td><td style="color: ${{data.act_profits[idx] >= 0 ? 'green' : 'red'}}"><strong>${{data.act_profits[idx]}}</strong></td></tr>
                        <tr><td>Actual Result</td><td>${{data.act_results[idx]}}</td></tr>
                        <tr><td>Prediction Rank</td><td>${{data.pred_ranks[idx]}}</td></tr>
                        <tr><td>Prediction Score</td><td>${{data.pred_scores[idx]}}</td></tr>
                        <tr><td>Params</td><td style="font-size:0.8em; word-break:break-all;">${{data.params[idx]}}</td></tr>
                    </table>
                `;
            }}
        </script>
    </body>
    </html>
    """

    output_path = os.path.join(target_dir, "Vector_Prediction_Distribution.html")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    main()
