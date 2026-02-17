
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import base64
import io
import warnings
import json
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
SMOOTHING_WINDOW = 573  # (Deprecated/Secondary) Controls the smoothness of the moving average line if needed
BACK_GRAPH = 10         # Number of previous graphs to overlay
TSNE_PERPLEXITY = 30    # Perplexity for t-SNE dimensionality reduction
HYPERCUBE = 1           # Hypercube size (steps) for averaging neighbors
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

def generate_histogram(data, title):
    """
    Generates a simple histogram for the distribution of results inside a hypercube.
    """
    plt.figure(figsize=(4, 3)) # Small figure for side-by-side
    plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=10)
    plt.xlabel("Result", fontsize=8)
    plt.ylabel("Count", fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

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

    # Sort Master DF by t-SNE 1D Embedding of Parameters
    print(f"Applying t-SNE dimensionality reduction (perplexity={TSNE_PERPLEXITY})...")

    # Extract Vector Parameters for t-SNE
    X = master_df[vector_cols].values

    tsne = TSNE(n_components=1, perplexity=TSNE_PERPLEXITY, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    master_df['tsne_1d'] = X_embedded[:, 0]

    # Sort by the 1D embedding
    master_df.sort_values(by='tsne_1d', ascending=True, inplace=True)
    print("Sorting complete based on t-SNE landscape.")

    # Keep only vector columns in the master DataFrame for merging
    master_vectors = master_df[vector_cols].copy()
    master_vectors.reset_index(drop=True, inplace=True)
    master_vectors['Master_Index'] = master_vectors.index # 0 to N

    # --- Hypercube Neighbor Pre-computation ---
    print(f"Pre-computing Hypercube Neighbors (Size={HYPERCUBE})...")

    # 1. Map parameters to Grid Indices
    grid_indices_list = []
    for col in vector_cols:
        # Get unique sorted values for this parameter
        unique_vals = np.sort(master_vectors[col].unique())
        # Map values to their index in unique_vals
        indices = np.searchsorted(unique_vals, master_vectors[col].values)
        grid_indices_list.append(indices)

    # Transpose to get (N_samples, N_features)
    grid_coords = np.vstack(grid_indices_list).T

    # 2. Build KDTree
    print("Building KDTree...")
    tree = KDTree(grid_coords, metric='chebyshev')

    # 3. Query Neighbors for all vectors
    # We query for HYPERCUBE radius (Chebyshev distance)
    print("Querying neighbors...")
    # returns array of objects (arrays of indices)
    master_neighbor_indices = tree.query_radius(grid_coords, r=HYPERCUBE)
    print("Neighbor pre-computation complete.")
    # -------------------------------------------

    # 2. Process All Files
    all_trends = [] # Stores hypercube avg trend lines (numpy arrays) for overlay logic

    # Store data for JS charts and tables
    # Structure:
    # {
    #   filename: {
    #     date: int,
    #     vectors: [{r, p, v}],
    #     chartData: {
    #        results: [float], // Raw results
    #        trend: [float]    // Hypercube Avg Trend
    #     },
    #     overlayData: [ // Array of previous trends
    #        [float], [float], ...
    #     ]
    #   }
    # }
    vector_data_store = {}

    html_sliding_rows = ""
    html_top3_rows = ""

    first_filename = os.path.basename(csv_files[0]) if csv_files else 'N/A'
    processed_count = 0

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        file_date = get_date_from_filename(filename)

        # Read current file
        current_df = read_csv_robust(filepath)

        if current_df.empty:
            print(f"Warning: {filename} is empty or unreadable. Skipping.")
            continue

        # Ensure 'Result', 'Profit' and vector cols exist
        if 'Result' not in current_df.columns or 'Profit' not in current_df.columns:
            print(f"Warning: 'Result' or 'Profit' column missing in {filename}. Skipping.")
            continue

        # Merge current data onto master vectors
        merged_df = pd.merge(master_vectors, current_df, on=vector_cols, how='left')

        # Fill missing numeric values with 0
        merged_df['Result'] = merged_df['Result'].fillna(0.0)
        merged_df['Profit'] = merged_df['Profit'].fillna(0.0)

        # Sort by Master Index to ensure X-axis is consistent
        merged_df.sort_values(by='Master_Index', ascending=True, inplace=True)

        # Get Y-values
        results = merged_df['Result'].values
        profits = merged_df['Profit'].values

        # --- Calculate Hypercube Average ---
        hypercube_avgs = np.zeros_like(results)

        for i, neighbor_idxs in enumerate(master_neighbor_indices):
            hypercube_avgs[i] = np.mean(results[neighbor_idxs])

        # -----------------------------------

        # Prepare vector data for this file to be embedded
        params_df = merged_df[vector_cols]
        params_list = params_df.astype(str).agg(', '.join, axis=1).tolist()

        file_vectors = []
        for r, p, v in zip(results, profits, params_list):
            file_vectors.append({'r': round(r, 4), 'p': round(p, 2), 'v': v})

        # Collect Overlay Data (Previous Trends)
        prev_trends_data = []
        if len(all_trends) > 0:
            # We want the last BACK_GRAPH trends
            start_idx = max(0, len(all_trends) - BACK_GRAPH)
            # Convert numpy arrays to lists for JSON serialization
            prev_trends_subset = all_trends[start_idx:]
            prev_trends_data = [t.tolist() for t in prev_trends_subset]

        vector_data_store[filename] = {
            'date': file_date,
            'vectors': file_vectors,
            'chartData': {
                'results': results.tolist(),
                'trend': hypercube_avgs.tolist()
            },
            'overlayData': prev_trends_data
        }

        # Store current trend for future overlays
        all_trends.append(hypercube_avgs)

        processed_count += 1

        # We only generate the HTML container here. The Chart.js rendering happens on load.
        # We need unique IDs for the canvases.

        safe_fname = filename.replace('.', '_').replace(' ', '_')

        # Only show Sliding Window if we have history (or just always show it, but empty overlay if first file)
        # Requirement: "Sliding Window View (Last x Overlay)"

        html_sliding_rows += f"""
        <div class="plot-container" id="container-{safe_fname}">
            <h3>{filename} (Sliding Window Analysis)</h3>

            <!-- Top Row: Overlay Graph + Controls -->
            <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 20px;">
                <div style="flex: 2; height: 400px; position: relative;">
                    <h4 style="margin-bottom: 5px;">Previous {BACK_GRAPH} Trends Overlay (Hypercube Avg)</h4>
                    <canvas id="overlay-chart-{safe_fname}"></canvas>
                    <button onclick="resetZoom('overlay-chart-{safe_fname}')" style="position:absolute; top:10px; right:10px; z-index:10; font-size:0.8em;">Reset Zoom</button>
                </div>
                <div style="flex: 1; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #ddd;">
                    <h4>Evaluate Vector Strategy</h4>
                    <p>Select a Vector Rank to evaluate for this week.</p>
                    <div style="margin-bottom: 15px;">
                        <label style="display:block; margin-bottom:5px; font-weight:bold;">Vector Rank (0-{len(master_vectors)-1}):</label>
                        <input type="number" id="input-{safe_fname}" class="rank-input" min="0" max="{len(master_vectors)-1}" placeholder="Enter Rank" style="width: 100%; padding: 8px; margin-bottom: 10px;">
                        <button onclick="submitVector('{filename}', '{safe_fname}')" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Submit</button>
                    </div>
                    <div id="result-{safe_fname}" style="margin-top: 10px; font-size: 0.9em; color: #555;"></div>
                </div>
            </div>

            <!-- Bottom Row: Current File Graph -->
            <div style="width: 100%; height: 400px; position: relative;">
                <h4 style="margin-bottom: 5px;">Current File: {filename}</h4>
                <canvas id="current-chart-{safe_fname}"></canvas>
                 <button onclick="resetZoom('current-chart-{safe_fname}')" style="position:absolute; top:10px; right:10px; z-index:10; font-size:0.8em;">Reset Zoom</button>
            </div>
        </div>
        """

        # --- Top 3 Hypercube Vectors ---
        # Identify top 3 vectors by Hypercube Average with EXCLUSION logic
        sorted_indices = np.argsort(hypercube_avgs)[::-1]

        top_selected_indices = []
        excluded_indices = set()

        for cand_idx in sorted_indices:
            if len(top_selected_indices) >= 3:
                break

            if cand_idx in excluded_indices:
                continue

            top_selected_indices.append(cand_idx)
            neighbors = master_neighbor_indices[cand_idx]
            excluded_indices.update(neighbors)

        top3_html_content = ""
        for rank_idx, vec_idx in enumerate(top_selected_indices):
            rank = rank_idx + 1
            vec_avg = hypercube_avgs[vec_idx]
            vec_raw_result = results[vec_idx]
            vec_profit = profits[vec_idx]
            vec_params = params_list[vec_idx]

            n_idxs = master_neighbor_indices[vec_idx]
            neighbor_results = results[n_idxs]

            img_hist = generate_histogram(neighbor_results, f"Rank {rank} (ID: {vec_idx}) Distribution")

            top3_html_content += f"""
            <div style="flex: 1; padding: 10px; border: 1px solid #eee; background: #fff;">
                <h5 style="margin: 0 0 10px 0; color: #333;">#{rank} (Vector ID: {vec_idx})</h5>
                <div style="font-size: 0.85em; margin-bottom: 10px; text-align: left;">
                    <div><strong>Hypercube Avg:</strong> {vec_avg:.4f}</div>
                    <div><strong>Raw Result:</strong> {vec_raw_result:.4f}</div>
                    <div><strong>Profit:</strong> {vec_profit:.2f}</div>
                    <div style="font-size: 0.8em; color: #666; word-break: break-all;">{vec_params}</div>
                </div>
                <img src="data:image/png;base64,{img_hist}" style="width: 100%;">
            </div>
            """

        html_top3_rows += f"""
        <div class="plot-container">
            <h3>{filename} - Top 3 Hypercube Vectors</h3>
            <div style="display: flex; gap: 20px;">
                {top3_html_content}
            </div>
        </div>
        """
        # -------------------------------

    # Serialize data for embedding
    # Use a customized JSON encoder or just standard json.dumps since we converted numpy to lists
    json_data = json.dumps(vector_data_store)

    # 3. Generate HTML Report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vector Distribution Report</title>
        <!-- Chart.js -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <!-- Hammer.js (required for zoom) -->
        <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
        <!-- Chart.js Zoom Plugin -->
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>

        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; background-color: #f4f4f9; }}
            .container {{ max-width: 1600px; margin: 0 auto; padding-bottom: 100px; }}
            .plot-container {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: #fff; }}
            img {{ max-width: 100%; height: auto; }}
            h1 {{ color: #333; }}
            h3 {{ color: #555; margin-top: 0; }}
            h4 {{ color: #777; }}
            h5 {{ color: #555; font-weight: bold; }}
            details {{ margin-bottom: 20px; text-align: left; border: 1px solid #ccc; border-radius: 5px; padding: 10px; background: #fff; }}
            summary {{ cursor: pointer; font-weight: bold; font-size: 1.2em; padding: 10px; background-color: #f9f9f9; }}
            summary:hover {{ background-color: #f0f0f0; }}
            .section-content {{ padding: 20px; }}

            /* Dashboard Styles */
            #dashboard {{ position: fixed; bottom: 0; left: 0; width: 100%; height: 350px; background: #fff; border-top: 2px solid #ccc; box-shadow: 0 -2px 10px rgba(0,0,0,0.2); z-index: 1000; display: flex; flex-direction: column; transition: height 0.3s; }}
            #dashboard.minimized {{ height: 40px; }}
            #dashboard-header {{ padding: 10px 20px; background: #333; color: #fff; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }}
            #dashboard-content {{ flex: 1; display: flex; overflow: hidden; padding: 10px; gap: 20px; }}
            .dash-panel {{ flex: 1; border: 1px solid #eee; padding: 10px; overflow-y: auto; display: flex; flex-direction: column; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}

            .save-btn {{ background-color: #28a745; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }}
            .save-btn:hover {{ background-color: #218838; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <div style="text-align:left;">
                    <h1>Vector Distribution Report</h1>
                    <p>Results ordered by vector rank in the first file ({first_filename}).</p>
                    <p>Fixed Scale: [{y_min:.2f}, {y_max:.2f}] | Hypercube Size: {HYPERCUBE} | Back Graph: {BACK_GRAPH}</p>
                </div>
                <button class="save-btn" onclick="saveReport()">Save Report Analysis</button>
            </div>

            <details open>
                <summary>Top 3 Vectors (Hypercube Analysis)</summary>
                <div class="section-content">
                    {html_top3_rows}
                </div>
            </details>

            <!-- Standard View REMOVED as per request -->

            <details open>
                <summary>Sliding Window View (Last {BACK_GRAPH} Overlay) - Interactive</summary>
                <div class="section-content">
                    {html_sliding_rows}
                </div>
            </details>
        </div>

        <!-- Dashboard -->
        <div id="dashboard">
            <div id="dashboard-header" onclick="toggleDashboard()">
                <span>Strategy Performance Dashboard</span>
                <span id="dash-toggle-icon">▼</span>
            </div>
            <div id="dashboard-content">
                <div class="dash-panel" style="flex: 1.2;">
                    <h4>Selected Vectors</h4>
                    <div style="flex:1; overflow-y:auto;">
                        <table id="trades-table">
                            <thead>
                                <tr>
                                    <th>File Date</th>
                                    <th>Filename</th>
                                    <th>Rank</th>
                                    <th>Profit</th>
                                    <th>Result</th>
                                    <th>Params</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>
                <div class="dash-panel" style="flex: 1.5;">
                    <h4>Cumulative Profit (Start $10,000)</h4>
                    <div style="position: relative; height: 100%; width: 100%;">
                        <canvas id="profitChart"></canvas>
                    </div>
                </div>
                <div class="dash-panel" style="flex: 0.8;">
                    <h4>Performance Metrics</h4>
                    <table id="metrics-table">
                        <tr><td>Total Profit</td><td id="m-total">0.00</td></tr>
                        <tr><td>Max Drawdown</td><td id="m-maxdd">0.00%</td></tr>
                        <tr><td>Avg Drawdown</td><td id="m-avgdd">0.00%</td></tr>
                        <tr><td>Sharpe Ratio (Est)</td><td id="m-sharpe">0.00</td></tr>
                        <tr><td>Files Traded</td><td id="m-count">0</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Embedded Data
            const vectorData = {json_data};

            // Global Chart Registry
            const charts = {{}}; // id -> Chart instance

            // State Management
            let selectedTrades = {{}}; // {{ filename: rank }}

            // Chart Instance for Dashboard
            let profitChart = null;

            // Initialize
            document.addEventListener('DOMContentLoaded', () => {{
                // Initialize Charts for Sliding Window
                initCharts();

                restoreState();
                updateDashboard();
            }});

            function initCharts() {{
                for (const [filename, data] of Object.entries(vectorData)) {{
                    const safeFname = filename.replace(/\\./g, '_').replace(/ /g, '_');

                    // X-Axis Labels (Rank 0 to N)
                    // We assume all files have same length vectors (Master list)
                    // Just create an array [0, 1, ..., N-1]
                    const len = data.chartData.results.length;
                    const labels = Array.from({{length: len}}, (v, k) => k);

                    // --- Overlay Chart ---
                    const ctxOverlay = document.getElementById('overlay-chart-' + safeFname);
                    if (ctxOverlay) {{
                        const datasets = [];

                        // Add previous trends (faded)
                        if (data.overlayData && data.overlayData.length > 0) {{
                            data.overlayData.forEach((trend, i) => {{
                                // Calculate alpha
                                const alpha = 0.2 + (0.6 * (i / data.overlayData.length));
                                // Randomish color but consistent?
                                // Let's just use grey/blue variants
                                datasets.push({{
                                    label: `Prev ${{i+1}}`,
                                    data: trend,
                                    borderColor: `rgba(100, 100, 100, ${{alpha}})`,
                                    borderWidth: 1,
                                    pointRadius: 0,
                                    fill: false,
                                    tension: 0.1
                                }});
                            }});
                        }}

                        // Add Current Trend (Orange) - Optional? Usually overlay implies comparison.
                        // Let's add current trend as bold orange
                        datasets.push({{
                             label: 'Current Trend',
                             data: data.chartData.trend,
                             borderColor: 'orange',
                             borderWidth: 2,
                             pointRadius: 0,
                             fill: false,
                             tension: 0.1
                        }});

                        const chartOverlay = new Chart(ctxOverlay, {{
                            type: 'line',
                            data: {{
                                labels: labels,
                                datasets: datasets
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    legend: {{ display: false }},
                                    zoom: {{
                                        pan: {{
                                            enabled: true,
                                            mode: 'x',
                                        }},
                                        zoom: {{
                                            wheel: {{ enabled: true }},
                                            pinch: {{ enabled: true }},
                                            mode: 'x',
                                        }}
                                    }}
                                }},
                                scales: {{
                                    x: {{ ticks: {{ maxTicksLimit: 20 }} }},
                                    y: {{ min: {y_min}, max: {y_max} }}
                                }}
                            }}
                        }});
                        charts['overlay-chart-' + safeFname] = chartOverlay;
                    }}

                    // --- Current File Chart ---
                    const ctxCurrent = document.getElementById('current-chart-' + safeFname);
                    if (ctxCurrent) {{
                        const chartCurrent = new Chart(ctxCurrent, {{
                            data: {{
                                labels: labels,
                                datasets: [
                                    {{
                                        type: 'line',
                                        label: 'Hypercube Avg',
                                        data: data.chartData.trend,
                                        borderColor: 'orange',
                                        borderWidth: 2,
                                        pointRadius: 0,
                                        tension: 0.1,
                                        order: 1
                                    }},
                                    {{
                                        type: 'scatter', // or 'bar', scatter is good for dense points, but 'line' with fill is what we had.
                                        // To mimic fill_between, we can use a line with fill: 'origin' or fill: true.
                                        // But 'Raw Result' is discrete per vector.
                                        // Let's use a filled line chart for raw data to match original look
                                        type: 'line',
                                        label: 'Raw Result',
                                        data: data.chartData.results,
                                        backgroundColor: 'rgba(65, 105, 225, 0.5)', // royalblue alpha 0.5
                                        borderColor: 'rgba(65, 105, 225, 0.8)',
                                        borderWidth: 0, // No line stroke, just fill?
                                        pointRadius: 1, // Small points
                                        fill: true, // Fill to bottom
                                        order: 2
                                    }}
                                ]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    zoom: {{
                                        pan: {{ enabled: true, mode: 'x' }},
                                        zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'x' }}
                                    }}
                                }},
                                scales: {{
                                    x: {{ ticks: {{ maxTicksLimit: 20 }} }},
                                    y: {{ min: {y_min}, max: {y_max} }}
                                }}
                            }}
                        }});
                        charts['current-chart-' + safeFname] = chartCurrent;
                    }}
                }}
            }}

            function resetZoom(chartId) {{
                if (charts[chartId]) {{
                    charts[chartId].resetZoom();
                }}
            }}

            function toggleDashboard() {{
                const d = document.getElementById('dashboard');
                d.classList.toggle('minimized');
                const icon = document.getElementById('dash-toggle-icon');
                icon.textContent = d.classList.contains('minimized') ? '▲' : '▼';
            }}

            function submitVector(filename, safeFname) {{
                const input = document.getElementById('input-' + safeFname);
                const rank = parseInt(input.value);
                const maxRank = vectorData[filename].vectors.length - 1;

                if (isNaN(rank) || rank < 0 || rank > maxRank) {{
                    alert('Invalid Rank. Please enter a value between 0 and ' + maxRank);
                    return;
                }}

                // Save to state
                selectedTrades[filename] = rank;

                // Provide feedback
                const data = vectorData[filename].vectors[rank];
                const resDiv = document.getElementById('result-' + safeFname);
                resDiv.innerHTML = `<strong>Selected Rank ${{rank}}</strong><br>Profit: ${{data.p}}<br>Result: ${{data.r}}`;
                resDiv.style.color = data.p >= 0 ? 'green' : 'red';

                updateDashboard();
            }}

            function updateDashboard() {{
                const tbody = document.querySelector('#trades-table tbody');
                tbody.innerHTML = '';

                // Convert state to array and sort by date
                let trades = [];
                for (const [fname, rank] of Object.entries(selectedTrades)) {{
                    if (vectorData[fname]) {{
                        const info = vectorData[fname];
                        trades.push({{
                            filename: fname,
                            date: info.date,
                            rank: rank,
                            data: info.vectors[rank]
                        }});
                    }}
                }}

                // Sort Chronologically
                trades.sort((a, b) => a.date - b.date);

                // Calculate Equity Curve
                let equity = 10000;
                let equityCurve = [10000];
                let labels = ['Start'];
                let returns = [];
                let peak = 10000;
                let drawdowns = [];

                trades.forEach(t => {{
                    // Update Table
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${{t.date}}</td>
                        <td title="${{t.filename}}">${{t.filename.substring(0, 20)}}...</td>
                        <td>${{t.rank}}</td>
                        <td style="color: ${{t.data.p >= 0 ? 'green' : 'red'}}">${{t.data.p.toFixed(2)}}</td>
                        <td>${{t.data.r.toFixed(4)}}</td>
                        <td style="font-size:0.8em">${{t.data.v}}</td>
                    `;
                    tbody.appendChild(row);

                    // Update Equity
                    equity += t.data.p;
                    equityCurve.push(equity);
                    labels.push(t.date.toString());

                    // Metrics Prep
                    if (equity > peak) peak = equity;
                    let dd = (peak - equity) / peak;
                    drawdowns.push(dd);

                    // Simple return approx (profit / previous equity) - assuming fixed lot size or compounding?
                    // User asked for plot from 10k account. We'll stick to absolute profit addition.
                    // For Sharpe, we can use the raw PnL series mean/std.
                    returns.push(t.data.p);
                }});

                // Update Metrics
                const totalProfit = equity - 10000;
                const maxDD = drawdowns.length > 0 ? Math.max(...drawdowns) * 100 : 0;
                const avgDD = drawdowns.length > 0 ? (drawdowns.reduce((a,b)=>a+b,0)/drawdowns.length) * 100 : 0;

                // Sharpe Calculation (Weekly)
                let sharpe = 0;
                if (returns.length > 1) {{
                    const mean = returns.reduce((a,b)=>a+b,0) / returns.length;
                    const variance = returns.reduce((a,b)=>a + Math.pow(b-mean, 2), 0) / (returns.length - 1);
                    const std = Math.sqrt(variance);
                    if (std > 0) {{
                        sharpe = (mean / std) * Math.sqrt(52);
                    }}
                }}

                document.getElementById('m-total').textContent = totalProfit.toFixed(2);
                document.getElementById('m-total').style.color = totalProfit >= 0 ? 'green' : 'red';
                document.getElementById('m-maxdd').textContent = maxDD.toFixed(2) + '%';
                document.getElementById('m-avgdd').textContent = avgDD.toFixed(2) + '%';
                document.getElementById('m-sharpe').textContent = sharpe.toFixed(3);
                document.getElementById('m-count').textContent = trades.length;

                // Update Chart
                updateChart(labels, equityCurve);
            }}

            function updateChart(labels, data) {{
                const ctx = document.getElementById('profitChart').getContext('2d');

                if (profitChart) {{
                    profitChart.destroy();
                }}

                profitChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Account Equity ($)',
                            data: data,
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ maxTicksLimit: 10 }} }},
                            y: {{ beginAtZero: false }}
                        }}
                    }}
                }});
            }}

            function saveReport() {{
                // Serialize current DOM to a string
                // We need to make sure the input values are preserved in the DOM before serializing

                // 1. Update input attributes
                for (const [fname, rank] of Object.entries(selectedTrades)) {{
                     // We need to find the safe ID for the input.
                     // The input ID is 'input-' + safeFname
                     // But we only have 'fname'. We need to convert it.
                     const safeFname = fname.replace(/\\./g, '_').replace(/ /g, '_');
                    const input = document.getElementById('input-' + safeFname);
                    if (input) input.setAttribute('value', rank);
                }}

                // 2. Clone and Clean
                // We want to save the file such that when opened, it loads the state.
                // The easiest way is to rely on 'selectedTrades' being hardcoded into the script?
                // No, simpler: We save the DOM. The 'restoreState' function will read from the inputs!

                const htmlContent = document.documentElement.outerHTML;
                const blob = new Blob([htmlContent], {{type: 'text/html'}});
                const url = URL.createObjectURL(blob);

                const a = document.createElement('a');
                a.href = url;
                a.download = 'Vector_Distribution_Report_Saved.html';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}

            function restoreState() {{
                // Read from inputs to restore 'selectedTrades' if this is a saved file
                const inputs = document.querySelectorAll('.rank-input');
                inputs.forEach(input => {{
                    if (input.value && input.value !== '') {{
                        // ID is 'input-safeFname'
                        const safeId = input.id.replace('input-', '');
                        // We need to map safeId back to original filename?
                        // Or just store safeId -> rank in selectedTrades?
                        // Actually, 'vectorData' keys are original filenames.
                        // We need to find which key corresponds to this safeId.

                        // Let's iterate vectorData to find match
                        for(const fname of Object.keys(vectorData)) {{
                            const s = fname.replace(/\\./g, '_').replace(/ /g, '_');
                            if (s === safeId) {{
                                selectedTrades[fname] = parseInt(input.value);

                                // Visual feedback
                                if (vectorData[fname]) {{
                                    const rank = parseInt(input.value);
                                    const data = vectorData[fname].vectors[rank];
                                    const resDiv = document.getElementById('result-' + safeId);
                                     if(resDiv && data) {{
                                        resDiv.innerHTML = `<strong>Selected Rank ${{rank}}</strong><br>Profit: ${{data.p}}<br>Result: ${{data.r}}`;
                                        resDiv.style.color = data.p >= 0 ? 'green' : 'red';
                                     }}
                                }}
                                break;
                            }}
                        }}
                    }}
                }});
            }}
        </script>
    </body>
    </html>
    """

    output_path = os.path.join(target_dir, "Vector_Distribution_Report.html")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    main()
