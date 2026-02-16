
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

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
SMOOTHING_WINDOW = 573  # Controls the smoothness of the moving average line
BACK_GRAPH = 10         # Number of previous graphs to overlay
TSNE_PERPLEXITY = 30    # Perplexity for t-SNE dimensionality reduction
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

                # Use random color
                color = np.random.rand(3,)

                # Calculate alpha: increasing for newer lines
                # Range alpha from 0.2 to 1.0
                alpha = 0.2 + (0.8 * (i / (num_trends - 1))) if num_trends > 1 else 1.0

                plt.plot(x, trend, color=color, linewidth=2, alpha=alpha)

    plt.title(f"Previous {num_trends} Trends Overlay")
    plt.xlabel("Vector Rank (Based on First File)")
    plt.ylabel("Result")

    # Set fixed Y-axis scale if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    # Detailed X-Axis and Grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # 5 minor ticks per major
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

    # Sort Master DF by t-SNE 1D Embedding of Parameters
    print(f"Applying t-SNE dimensionality reduction (perplexity={TSNE_PERPLEXITY})...")

    # Extract Vector Parameters for t-SNE
    X = master_df[vector_cols].values

    # Normalize/Scale if needed?
    # Usually good for t-SNE if ranges differ significantly, but we'll stick to raw for now unless requested.
    # The user asked to use variable parameter dimensionality.

    tsne = TSNE(n_components=1, perplexity=TSNE_PERPLEXITY, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    master_df['tsne_1d'] = X_embedded[:, 0]

    # Sort by the 1D embedding
    master_df.sort_values(by='tsne_1d', ascending=True, inplace=True)
    print("Sorting complete based on t-SNE landscape.")

    # Keep only vector columns in the master DataFrame for merging
    # We add a 'MasterRank' to preserve order after merge if needed,
    # but since we iterate through the master list, we can just left join.
    master_vectors = master_df[vector_cols].copy()
    master_vectors.reset_index(drop=True, inplace=True)
    master_vectors['Master_Index'] = master_vectors.index # 0 to N

    # 2. Process All Files
    all_trends = [] # Stores smooth trend lines (numpy arrays)
    vector_data_store = {} # Stores data for JS {filename: {date, vectors: [r, p, params]}}

    # We will build HTML parts dynamically
    html_standard_rows = ""
    html_sliding_rows = ""

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

        # Prepare vector data for this file to be embedded
        # We store a list of dicts: {r: result, p: profit, v: params_str}
        # To save space, we can just store a list of lists if needed, but dict is clearer for JS.
        # Let's optimize slightly: simple arrays

        # Create a params string for each row
        # We need to make sure we select the vector columns row by row
        params_df = merged_df[vector_cols]
        # Convert to string representation like "p1=v1, p2=v2" or just "v1, v2"
        # Let's just do comma separated values
        params_list = params_df.astype(str).agg(', '.join, axis=1).tolist()

        file_vectors = []
        for r, p, v in zip(results, profits, params_list):
            file_vectors.append({'r': round(r, 4), 'p': round(p, 2), 'v': v})

        vector_data_store[filename] = {
            'date': file_date,
            'vectors': file_vectors
        }

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

            # Add to Sliding View HTML (New Layout)
            html_sliding_rows += f"""
            <div class="plot-container" id="container-{filename.replace('.', '_')}">
                <h3>{filename} (Sliding Window Analysis)</h3>

                <!-- Top Row: Overlay Graph + Controls -->
                <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 20px;">
                    <div style="flex: 2;">
                        <h4 style="margin-bottom: 5px;">Previous {BACK_GRAPH} Trends Overlay</h4>
                        <img src="data:image/png;base64,{img_overlay}" style="width: 100%; border: 1px solid #eee;">
                    </div>
                    <div style="flex: 1; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #ddd;">
                        <h4>Evaluate Vector Strategy</h4>
                        <p>Select a Vector Rank to evaluate for this week.</p>
                        <div style="margin-bottom: 15px;">
                            <label style="display:block; margin-bottom:5px; font-weight:bold;">Vector Rank (0-{len(master_vectors)-1}):</label>
                            <input type="number" id="input-{filename}" class="rank-input" min="0" max="{len(master_vectors)-1}" placeholder="Enter Rank" style="width: 100%; padding: 8px; margin-bottom: 10px;">
                            <button onclick="submitVector('{filename}')" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Submit</button>
                        </div>
                        <div id="result-{filename}" style="margin-top: 10px; font-size: 0.9em; color: #555;"></div>
                    </div>
                </div>

                <!-- Bottom Row: Current File Graph -->
                <div style="width: 100%;">
                    <h4 style="margin-bottom: 5px;">Current File: {filename}</h4>
                    <img src="data:image/png;base64,{img_std}" style="width: 100%; border: 1px solid #eee;">
                </div>
            </div>
            """

    # Serialize data for embedding
    json_data = json.dumps(vector_data_store)

    # 3. Generate HTML Report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vector Distribution Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; background-color: #f4f4f9; }}
            .container {{ max-width: 1600px; margin: 0 auto; padding-bottom: 100px; }}
            .plot-container {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: #fff; }}
            img {{ max-width: 100%; height: auto; }}
            h1 {{ color: #333; }}
            h3 {{ color: #555; margin-top: 0; }}
            h4 {{ color: #777; }}
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
                    <p>Fixed Scale: [{y_min:.2f}, {y_max:.2f}] | Smoothing: {SMOOTHING_WINDOW} | Back Graph: {BACK_GRAPH}</p>
                </div>
                <button class="save-btn" onclick="saveReport()">Save Report Analysis</button>
            </div>

            <details>
                <summary>Standard View</summary>
                <div class="section-content">
                    {html_standard_rows}
                </div>
            </details>

            <details open>
                <summary>Sliding Window View (Last {BACK_GRAPH} Overlay)</summary>
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

            // State Management
            let selectedTrades = {{}}; // {{ filename: rank }}

            // Chart Instance
            let profitChart = null;

            // Initialize
            document.addEventListener('DOMContentLoaded', () => {{
                restoreState();
                updateDashboard();
            }});

            function toggleDashboard() {{
                const d = document.getElementById('dashboard');
                d.classList.toggle('minimized');
                const icon = document.getElementById('dash-toggle-icon');
                icon.textContent = d.classList.contains('minimized') ? '▲' : '▼';
            }}

            function submitVector(filename) {{
                const input = document.getElementById('input-' + filename);
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
                const resDiv = document.getElementById('result-' + filename);
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
                    const input = document.getElementById('input-' + fname);
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
                        const filename = input.id.replace('input-', '');
                        selectedTrades[filename] = parseInt(input.value);

                        // Also restore visual feedback
                        if (vectorData[filename]) {{
                            const rank = parseInt(input.value);
                             const data = vectorData[filename].vectors[rank];
                             const resDiv = document.getElementById('result-' + filename);
                             if(resDiv && data) {{
                                resDiv.innerHTML = `<strong>Selected Rank ${{rank}}</strong><br>Profit: ${{data.p}}<br>Result: ${{data.r}}`;
                                resDiv.style.color = data.p >= 0 ? 'green' : 'red';
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
