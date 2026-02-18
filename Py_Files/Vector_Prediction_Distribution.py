
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
INITIAL_EQUITY = 10000  # Initial account balance for simulation
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

def calculate_sharpe_ratio(returns):
    """Calculates annualized Sharpe Ratio assuming weekly data."""
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return 0.0
    return (mean_return / std_return) * np.sqrt(52)

def calculate_drawdowns(equity_curve):
    """Calculates Max Drawdown and Average Drawdown."""
    peak = equity_curve[0]
    drawdowns = []

    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        drawdowns.append(dd)

    max_dd = np.max(drawdowns) if drawdowns else 0
    avg_dd = np.mean(drawdowns) if drawdowns else 0
    return max_dd * 100, avg_dd * 100

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

    html_file_rows = ""
    # Data storage for JS
    # Structure: { filename: { sorted_indices: [], params: [], pred_scores: [], act_results: [], act_profits: [], pred_ranks: [] } }
    report_data = {}

    # Rank-based history storage
    # { rank_id (1..N): { filenames: [], results: [], profits: [], params: [] } }
    rank_history = {r: {'filenames': [], 'results': [], 'profits': [], 'params': []} for r in range(1, TOP_N + 1)}

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

        # --- Capture Rank-Based History (Before Sorting) ---
        for i, vec_idx in enumerate(top_n_indices):
            rank = i + 1
            if rank <= TOP_N:
                rank_history[rank]['filenames'].append(current_filename)
                rank_history[rank]['results'].append(round(current_results[vec_idx], 4))
                rank_history[rank]['profits'].append(round(current_profits[vec_idx], 2))
                rank_history[rank]['params'].append(global_params[vec_idx])
        # ---------------------------------------------------

        # Extract data for Top N
        selected_indices = top_n_indices
        selected_pred_scores = prediction_scores[selected_indices]
        selected_act_results = current_results[selected_indices]
        selected_act_profits = current_profits[selected_indices]
        selected_pred_ranks = np.arange(1, TOP_N + 1) # Rank 1 is highest score

        # SORT by ACTUAL RESULT (Descending) as requested for File View
        sort_order = np.argsort(selected_act_results)[::-1]

        sorted_indices = selected_indices[sort_order]
        sorted_pred_scores = selected_pred_scores[sort_order]
        sorted_act_results = selected_act_results[sort_order]
        sorted_act_profits = selected_act_profits[sort_order]
        sorted_pred_ranks = selected_pred_ranks[sort_order]

        # Prepare Data for JS
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

        # Generate HTML Row for File View
        html_file_rows += f"""
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

    # --- Generate Rank Analysis HTML ---
    html_rank_rows = ""
    rank_summary_rows = ""

    # Pre-calculate stats for each Rank
    for r in range(1, TOP_N + 1):
        rank_data = rank_history[r]
        profits = rank_data['profits']

        # Calculate Equity Curve
        equity = [INITIAL_EQUITY]
        current_balance = INITIAL_EQUITY
        for p in profits:
            current_balance += p
            equity.append(current_balance)

        rank_data['equity_curve'] = equity

        total_pl = current_balance - INITIAL_EQUITY
        max_dd, avg_dd = calculate_drawdowns(equity)
        sharpe = calculate_sharpe_ratio(profits)

        # Add to rank_data for JS
        rank_data['stats'] = {
            'total_pl': round(total_pl, 2),
            'max_dd': round(max_dd, 2),
            'avg_dd': round(avg_dd, 2),
            'sharpe': round(sharpe, 3)
        }

        # Append to summary table rows
        rank_summary_rows += f"""
        <tr>
            <td style="text-align:center;">{r}</td>
            <td style="text-align:center; color: {'green' if total_pl >= 0 else 'red'}">${total_pl:.2f}</td>
            <td style="text-align:center;">{max_dd:.2f}%</td>
            <td style="text-align:center;">{avg_dd:.2f}%</td>
            <td style="text-align:center;">{sharpe:.3f}</td>
        </tr>
        """

        rank_id = f"rank-{r}"

        html_rank_rows += f"""
        <div class="plot-container">
            <details>
                <summary onclick="lazyLoadRankChart({r})">
                    Prediction Rank {r} Performance
                </summary>
                <div class="section-content">

                    <!-- Stats Table (Individual) -->
                    <table style="width: 100%; margin-bottom: 20px; font-size: 0.9em; background: #f9f9f9;">
                        <tr>
                            <th style="text-align:center;">Total P/L ($)</th>
                            <th style="text-align:center;">Max Drawdown (%)</th>
                            <th style="text-align:center;">Avg Drawdown (%)</th>
                            <th style="text-align:center;">Sharpe Ratio</th>
                        </tr>
                        <tr>
                            <td style="text-align:center; font-weight:bold; color: { 'green' if total_pl >= 0 else 'red' }">${total_pl:.2f}</td>
                            <td style="text-align:center;">{max_dd:.2f}%</td>
                            <td style="text-align:center;">{avg_dd:.2f}%</td>
                            <td style="text-align:center;">{sharpe:.3f}</td>
                        </tr>
                    </table>

                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 3; display: flex; flex-direction: column; gap: 20px;">
                             <!-- Chart 1: Result over Time -->
                            <div style="height: 400px; position: relative;">
                                <h4 style="margin:0;">Rank {r} Result over Time</h4>
                                <canvas id="chart-rank-{r}"></canvas>
                            </div>

                            <!-- Chart 2: Equity Curve -->
                            <div style="height: 400px; position: relative;">
                                <h4 style="margin:0;">Simulated Account Equity ($10k Start)</h4>
                                <canvas id="chart-equity-{r}"></canvas>
                            </div>
                        </div>

                        <div style="flex: 1; padding: 20px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; height: fit-content;">
                            <h4 id="info-title-rank-{r}">Select a point...</h4>
                            <div id="info-content-rank-{r}" style="font-size: 0.9em;">
                                <p>Click on a data point in the graph to see details for this file.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </details>
        </div>
        """

    # Serialize Data
    json_report_data = json.dumps(report_data)
    json_rank_data = json.dumps(rank_history)

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

            .calc-box {{
                background: #eef;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid #ccd;
                border-radius: 5px;
                text-align: left;
            }}
            .calc-box input {{ padding: 8px; width: 200px; margin-right: 10px; }}
            .calc-box button {{ padding: 8px 15px; cursor: pointer; background: #007bff; color: #fff; border: none; border-radius: 4px; }}
            .calc-box button:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vector Prediction Distribution Report</h1>
            <p><strong>Config:</strong> Lookback={FILE_LOOKBACK} | Hypercube={HYPERCUBE} | Top N={TOP_N}</p>

            <details open style="margin-bottom: 40px; border: 2px solid #aaa;">
                <summary style="background-color: #ddd;">Section 1: File Analysis (Prediction vs Actual)</summary>
                <div class="section-content">

                    <div class="calc-box">
                        <label><strong>Count Results Above Threshold:</strong></label><br>
                        <div style="margin-top:5px;">
                            <input type="number" id="threshold-input" placeholder="Enter Result Threshold (e.g. 50)">
                            <button onclick="calculateThresholdStats()">Calculate</button>
                        </div>
                        <div id="threshold-results" style="margin-top: 15px; max-height: 200px; overflow-y: auto; display: none;"></div>
                    </div>

                    <p>Showing distribution of the Top {TOP_N} vectors predicted by the strategy, sorted by their ACTUAL result in each file.</p>
                    {html_file_rows}
                </div>
            </details>

            <details style="margin-bottom: 40px; border: 2px solid #aaa;">
                <summary style="background-color: #ddd;">Section 2: Rank Analysis (Performance over Time)</summary>
                <div class="section-content">
                    <p>Showing the actual result of specific Prediction Ranks across all files.</p>

                    <div style="max-height: 300px; overflow-y: auto; margin-bottom: 30px; border: 1px solid #ccc;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead style="position: sticky; top: 0; background: #eee; z-index: 10;">
                                <tr>
                                    <th style="text-align:center;">Rank</th>
                                    <th style="text-align:center;">Total P/L ($)</th>
                                    <th style="text-align:center;">Max Drawdown (%)</th>
                                    <th style="text-align:center;">Avg Drawdown (%)</th>
                                    <th style="text-align:center;">Sharpe Ratio</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rank_summary_rows}
                            </tbody>
                        </table>
                    </div>

                    {html_rank_rows}
                </div>
            </details>
        </div>

        <script>
            const reportData = {json_report_data};
            const rankData = {json_rank_data};
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

            function lazyLoadRankChart(rank) {{
                const chartId1 = 'rank-' + rank;
                const chartId2 = 'equity-' + rank;

                if (charts[chartId1]) return; // Assume if one loaded, both loaded

                const data = rankData[rank];
                if (!data) return;

                // --- Chart 1: Results over Time ---
                const ctx1 = document.getElementById('chart-rank-' + rank).getContext('2d');
                const chart1 = new Chart(ctx1, {{
                    type: 'line',
                    data: {{
                        labels: data.filenames,
                        datasets: [{{
                            label: `Rank ${{rank}} Result`,
                            data: data.results,
                            borderColor: 'seagreen',
                            backgroundColor: 'rgba(46, 139, 87, 0.2)',
                            borderWidth: 2,
                            pointRadius: 3,
                            fill: true,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'index', intersect: false }},
                        scales: {{
                            x: {{ display: false }} // Hide Labels as requested
                        }},
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    title: (ctx) => ctx[0].label,
                                    label: (ctx) => [`Result: ${{data.results[ctx.dataIndex]}}`, `Profit: ${{data.profits[ctx.dataIndex]}}`]
                                }}
                            }}
                        }},
                        onClick: (e) => {{
                            const points = chart1.getElementsAtEventForMode(e, 'nearest', {{ intersect: true }}, true);
                            if (points.length) updateRankInfoPanel(rank, points[0].index);
                        }}
                    }}
                }});
                charts[chartId1] = chart1;

                // --- Chart 2: Equity Curve ---
                // X-axis has one more point than filenames (Start)
                const equityLabels = ['Start'].concat(data.filenames);

                const ctx2 = document.getElementById('chart-equity-' + rank).getContext('2d');
                const chart2 = new Chart(ctx2, {{
                    type: 'line',
                    data: {{
                        labels: equityLabels,
                        datasets: [{{
                            label: 'Account Balance ($)',
                            data: data.equity_curve,
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0,
                            fill: true,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'index', intersect: false }},
                        scales: {{
                            x: {{ display: false }} // Hide Labels as requested
                        }},
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    title: (ctx) => ctx[0].label,
                                }}
                            }}
                        }}
                    }}
                }});
                charts[chartId2] = chart2;
            }}

            function updateRankInfoPanel(rank, idx) {{
                const data = rankData[rank];
                const infoTitle = document.getElementById('info-title-rank-' + rank);
                const infoContent = document.getElementById('info-content-rank-' + rank);

                const fname = data.filenames[idx];

                infoTitle.textContent = `Detail for Rank ${{rank}} on ${{fname.substring(0, 15)}}...`;

                infoContent.innerHTML = `
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Filename</td><td style="font-size:0.8em; word-break:break-all;">${{fname}}</td></tr>
                        <tr><td><strong>Actual Profit</strong></td><td style="color: ${{data.profits[idx] >= 0 ? 'green' : 'red'}}"><strong>${{data.profits[idx]}}</strong></td></tr>
                        <tr><td>Actual Result</td><td>${{data.results[idx]}}</td></tr>
                        <tr><td>Params</td><td style="font-size:0.8em; word-break:break-all;">${{data.params[idx]}}</td></tr>
                    </table>
                `;
            }}

            function calculateThresholdStats() {{
                const input = document.getElementById('threshold-input');
                const threshold = parseFloat(input.value);
                const display = document.getElementById('threshold-results');

                if (isNaN(threshold)) {{
                    alert("Please enter a valid numeric threshold.");
                    return;
                }}

                let html = '<table style="width:100%; border:1px solid #ddd;"><thead><tr><th>File</th><th>Count > ' + threshold + '</th><th>%</th></tr></thead><tbody>';

                const filenames = Object.keys(reportData).sort(); // Should sort naturally or we use stored order?
                // reportData keys are filenames. We can iterate them.

                for (const fname of filenames) {{
                    const rData = reportData[fname];
                    if (!rData) continue;

                    const count = rData.act_results.filter(r => r >= threshold).length;
                    const total = rData.act_results.length;
                    const pct = (total > 0) ? ((count / total) * 100).toFixed(1) : 0;

                    html += `<tr><td>${{fname}}</td><td>${{count}} / ${{total}}</td><td>${{pct}}%</td></tr>`;
                }}

                html += '</tbody></table>';
                display.innerHTML = html;
                display.style.display = 'block';
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
