
import os
import sys
import glob
import pandas as pd
import numpy as np
import warnings
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import KDTree

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
HYPERCUBE = 1           # Hypercube size (steps) for averaging neighbors
FILE_LOOKBACK = 5       # Number of past files to average for prediction
TOP_N = 100             # Number of top predicted vectors to evaluate
INITIAL_EQUITY = 10000  # Initial account balance for simulation
SMOOTHING_WINDOW = 10   # Window for smooth average line
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

def generate_final_verdict_pdf(output_path, rank1_data):
    """Generates a PDF report for the Top Rank."""
    with PdfPages(output_path) as pdf:
        # Page 1: Equity Curve & Stats
        plt.figure(figsize=(11, 8.5))

        # Title
        plt.suptitle("Final Verdict Report: Top Rank Model", fontsize=20, weight='bold')

        # Text Stats
        stats_text = (
            f"Top Rank Model (Rank 1)\n"
            f"--------------------------------\n"
            f"Total Profit: ${rank1_data['stats']['total_pl']}\n"
            f"Max Drawdown: {rank1_data['stats']['max_dd']}%\n"
            f"Avg Drawdown: {rank1_data['stats']['avg_dd']}%\n"
            f"Sharpe Ratio: {rank1_data['stats']['sharpe']}\n"
            f"Total Trades: {len(rank1_data['profits'])}\n"
            f"Final Equity: ${rank1_data['equity_curve'][-1]:.2f}"
        )
        plt.figtext(0.1, 0.75, stats_text, fontsize=12, family='monospace', bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 10})

        # Equity Curve
        ax1 = plt.axes([0.1, 0.1, 0.8, 0.5])
        ax1.plot(rank1_data['equity_curve'], color='green', linewidth=2)
        ax1.set_title("Equity Curve ($10,000 Start)")
        ax1.set_xlabel("Time (Files)")
        ax1.set_ylabel("Account Balance ($)")
        ax1.grid(True, alpha=0.3)

        pdf.savefig()
        plt.close()

def main():
    # Use Agg backend
    plt.switch_backend('Agg')

    print("--- Vector Prediction Distribution Generator ---")

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("Enter the path to the folder containing CSVs: ").strip()

    if not target_dir or not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    csv_files.sort(key=get_date_from_filename)
    print(f"Found {len(csv_files)} files.")

    if len(csv_files) <= FILE_LOOKBACK:
        print(f"Error: Not enough files ({len(csv_files)}) for lookback of {FILE_LOOKBACK}.")
        return

    # 1. Process Master File
    first_file = csv_files[0]
    master_df = read_csv_robust(first_file)
    if master_df.empty or 'Trades' not in master_df.columns or 'Result' not in master_df.columns:
        print("Error: First file is invalid.")
        return

    cols = list(master_df.columns)
    try:
        trades_idx = cols.index('Trades')
        vector_cols = cols[trades_idx+1:]
    except ValueError:
        return

    master_vectors = master_df[vector_cols].copy()
    master_vectors.reset_index(drop=True, inplace=True)
    master_vectors['Master_Index'] = master_vectors.index
    global_params = master_vectors[vector_cols].astype(str).agg(', '.join, axis=1).tolist()

    # Pre-compute Hypercube Neighbors
    print(f"Pre-computing Hypercube Neighbors (Size={HYPERCUBE})...")
    grid_indices_list = []
    for col in vector_cols:
        unique_vals = np.sort(master_vectors[col].unique())
        indices = np.searchsorted(unique_vals, master_vectors[col].values)
        grid_indices_list.append(indices)
    grid_coords = np.vstack(grid_indices_list).T
    tree = KDTree(grid_coords, metric='chebyshev')
    master_neighbor_indices = tree.query_radius(grid_coords, r=HYPERCUBE)

    # Pre-compute Averages
    print("Pre-computing Hypercube Averages for all files...")
    all_file_hypercube_avgs = []
    all_file_results = []
    all_file_profits = []
    all_file_dates = []
    valid_files_indices = []

    for i, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        current_df = read_csv_robust(filepath)

        if current_df.empty or 'Result' not in current_df.columns:
            all_file_hypercube_avgs.append(None)
            all_file_results.append(None)
            all_file_profits.append(None)
            continue

        merged_df = pd.merge(master_vectors, current_df, on=vector_cols, how='left')
        merged_df['Result'] = merged_df['Result'].fillna(0.0)
        merged_df['Profit'] = merged_df['Profit'].fillna(0.0)
        merged_df.sort_values(by='Master_Index', ascending=True, inplace=True)

        results = merged_df['Result'].values
        profits = merged_df['Profit'].values

        hypercube_avgs = np.zeros_like(results)
        for idx, neighbor_idxs in enumerate(master_neighbor_indices):
            hypercube_avgs[idx] = np.mean(results[neighbor_idxs])

        all_file_hypercube_avgs.append(hypercube_avgs)
        all_file_results.append(results)
        all_file_profits.append(profits)
        all_file_dates.append(filename)
        valid_files_indices.append(i)

    # 3. Prediction Loop
    print("Running Prediction Logic...")
    html_file_rows = ""
    report_data = {}
    rank_history = {r: {'filenames': [], 'results': [], 'profits': [], 'params': []} for r in range(1, TOP_N + 1)}

    for k in range(FILE_LOOKBACK, len(valid_files_indices)):
        current_real_idx = valid_files_indices[k]
        current_filename = all_file_dates[current_real_idx]
        lookback_indices = [valid_files_indices[j] for j in range(k - FILE_LOOKBACK, k)]

        past_avgs_list = [all_file_hypercube_avgs[idx] for idx in lookback_indices]
        prediction_scores = np.mean(np.vstack(past_avgs_list), axis=0)

        # Select Top N Indices based on Prediction Score (Descending)
        top_n_indices = np.argsort(prediction_scores)[-TOP_N:][::-1]

        current_results = all_file_results[current_real_idx]
        current_profits = all_file_profits[current_real_idx]

        # Rank-based history (Rank 1 is index 0 of top_n_indices)
        for i, vec_idx in enumerate(top_n_indices):
            rank = i + 1
            if rank <= TOP_N:
                rank_history[rank]['filenames'].append(current_filename)
                rank_history[rank]['results'].append(round(current_results[vec_idx], 4))
                rank_history[rank]['profits'].append(round(current_profits[vec_idx], 2))
                rank_history[rank]['params'].append(global_params[vec_idx])

        # Prepare Data for File View

        sorted_indices = top_n_indices
        sorted_pred_scores = prediction_scores[sorted_indices]
        sorted_act_results = current_results[sorted_indices]
        sorted_act_profits = current_profits[sorted_indices]
        sorted_pred_ranks = np.arange(1, TOP_N + 1)
        sorted_params = [global_params[idx] for idx in sorted_indices]

        # Calculate Smooth Average Line
        smooth_series = pd.Series(sorted_act_results).rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        smooth_values = [round(x, 4) for x in smooth_series.fillna(0).tolist()]

        # Calculate Average Line
        avg_value = round(np.mean(sorted_act_results), 4)

        safe_fname = current_filename.replace('.', '_').replace(' ', '_')
        report_data[current_filename] = {
            'indices': sorted_indices.tolist(),
            'params': sorted_params,
            'pred_scores': [round(x, 4) for x in sorted_pred_scores.tolist()],
            'act_results': [round(x, 4) for x in sorted_act_results.tolist()],
            'act_profits': [round(x, 2) for x in sorted_act_profits.tolist()],
            'pred_ranks': sorted_pred_ranks.tolist(),
            'smooth': smooth_values,
            'avg': avg_value
        }

        html_file_rows += f"""
        <div class="plot-container" id="container-{safe_fname}">
            <details>
                <summary onclick="lazyLoadCharts('{current_filename}', '{safe_fname}')">
                    {current_filename} (Click to Load Prediction Analysis)
                </summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 3; height: 500px; position: relative;">
                            <h4>Distribution of Prediction Results (Sorted by Prediction Rank)</h4>
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

    # --- Process Rank History & Summary ---
    print("Processing Rank Analysis...")
    html_rank_rows = ""
    rank_stats_summary = []

    ranks_x = []
    max_dds_y = []
    avg_dds_y = []
    profits_y = []

    for r in range(1, TOP_N + 1):
        rank_data = rank_history[r]
        profits = rank_data['profits']

        equity = [INITIAL_EQUITY]
        current_balance = INITIAL_EQUITY
        for p in profits:
            current_balance += p
            equity.append(current_balance)

        rank_data['equity_curve'] = equity
        total_pl = current_balance - INITIAL_EQUITY
        max_dd, avg_dd = calculate_drawdowns(equity)
        sharpe = calculate_sharpe_ratio(profits)

        rank_data['stats'] = {
            'total_pl': round(total_pl, 2),
            'max_dd': round(max_dd, 2),
            'avg_dd': round(avg_dd, 2),
            'sharpe': round(sharpe, 3)
        }

        ranks_x.append(r)
        max_dds_y.append(max_dd)
        avg_dds_y.append(avg_dd)
        profits_y.append(total_pl)

        # Generate Interactive Chart container for Rank Analysis
        rank_safe_id = f"rank-{r}"

        html_rank_rows += f"""
        <div class="plot-container">
            <details>
                <summary onclick="lazyLoadRankChart({r})">Rank {r} Performance (Click to Expand)</summary>
                <div class="section-content">
                    <!-- Stats Table -->
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

                    <!-- Interactive Chart Container -->
                    <div style="height: 400px; position: relative;">
                        <h4 style="margin:0;">Simulated Account Equity ($10k Start)</h4>
                        <canvas id="chart-{rank_safe_id}"></canvas>
                    </div>
                </div>
            </details>
        </div>
        """

    # Prepare Summary Data for JS
    summary_data = {
        'ranks': ranks_x,
        'max_dd': max_dds_y,
        'avg_dd': avg_dds_y,
        'profit': profits_y,
        'profit_smooth': pd.Series(profits_y).rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean().fillna(0).tolist(),
        'max_dd_smooth': pd.Series(max_dds_y).rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean().fillna(0).tolist(),
    }

    # Generate Final Verdict PDF
    print("Generating Final Verdict PDF...")
    pdf_path = os.path.join(target_dir, "Final_Verdict.pdf")
    generate_final_verdict_pdf(pdf_path, rank_history[1])
    print(f"PDF Saved: {pdf_path}")

    # Serialize JSON for JS
    json_report_data = json.dumps(report_data)
    json_summary_data = json.dumps(summary_data)
    json_rank_history = json.dumps(rank_history)

    # HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vector Prediction Distribution Report</title>
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
            h2 {{ border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vector Prediction Distribution Report</h1>
            <p><strong>Config:</strong> Lookback={FILE_LOOKBACK} | Hypercube={HYPERCUBE} | Top N={TOP_N}</p>
            <p><a href="Final_Verdict.pdf" target="_blank" style="padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">Download Final Verdict PDF</a></p>

            <details open style="margin-bottom: 40px; border: 2px solid #aaa;">
                <summary style="background-color: #cce5ff;">Global Summary: Risk & Reward Distributions</summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1; height: 400px; position: relative;">
                            <h4 style="text-align:center;">Drawdown Distribution by Rank</h4>
                            <canvas id="summary-drawdown"></canvas>
                        </div>
                        <div style="flex: 1; height: 400px; position: relative;">
                            <h4 style="text-align:center;">Profit Distribution by Rank</h4>
                            <canvas id="summary-profit"></canvas>
                        </div>
                    </div>
                </div>
            </details>

            <details open style="margin-bottom: 40px; border: 2px solid #aaa;">
                <summary style="background-color: #ddd;">Section 1: File Analysis (Prediction vs Actual)</summary>
                <div class="section-content">
                    <p>Showing distribution of prediction results. X-Axis = Rank (1 is Best Prediction), Y-Axis = Actual Result.</p>
                    {html_file_rows}
                </div>
            </details>

            <details style="margin-bottom: 40px; border: 2px solid #aaa;">
                <summary style="background-color: #ddd;">Section 2: Rank Analysis (Performance over Time)</summary>
                <div class="section-content">
                    <p>Interactive analysis of specific Prediction Ranks across all files.</p>
                    {html_rank_rows}
                </div>
            </details>
        </div>

        <script>
            const reportData = {json_report_data};
            const summaryData = {json_summary_data};
            const rankHistory = {json_rank_history};
            const charts = {{}};

            // --- Render Summary Charts ---
            document.addEventListener('DOMContentLoaded', () => {{
                // Drawdown Chart
                new Chart(document.getElementById('summary-drawdown'), {{
                    type: 'line',
                    data: {{
                        labels: summaryData.ranks,
                        datasets: [
                            {{
                                label: 'Max Drawdown (%)',
                                data: summaryData.max_dd,
                                borderColor: 'rgba(255, 99, 132, 0.5)',
                                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                type: 'bar',
                                order: 2
                            }},
                             {{
                                label: 'Avg Drawdown (%)',
                                data: summaryData.avg_dd,
                                borderColor: 'rgba(54, 162, 235, 0.5)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                type: 'bar',
                                order: 3
                            }},
                            {{
                                label: 'Trend (Max DD)',
                                data: summaryData.max_dd_smooth,
                                borderColor: 'red',
                                borderWidth: 2,
                                pointRadius: 0,
                                type: 'line',
                                tension: 0.3,
                                order: 1
                            }}
                        ]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false }}
                }});

                // Profit Chart
                new Chart(document.getElementById('summary-profit'), {{
                    type: 'line',
                    data: {{
                        labels: summaryData.ranks,
                        datasets: [
                            {{
                                label: 'Total Profit ($)',
                                data: summaryData.profit,
                                borderColor: 'rgba(75, 192, 192, 0.6)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                type: 'bar',
                                order: 2
                            }},
                            {{
                                label: 'Trend (Profit)',
                                data: summaryData.profit_smooth,
                                borderColor: 'green',
                                borderWidth: 2,
                                pointRadius: 0,
                                type: 'line',
                                tension: 0.3,
                                order: 1
                            }}
                        ]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false }}
                }});
            }});

            function lazyLoadCharts(filename, safeFname) {{
                if (charts[safeFname]) return;
                const data = reportData[filename];
                if (!data) return;

                const labels = data.pred_ranks; // 1 to N

                const ctx = document.getElementById('chart-' + safeFname).getContext('2d');
                const chart = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        labels: labels,
                        datasets: [
                            {{
                                label: 'Actual Result',
                                data: data.act_results.map((y, i) => ({{x: labels[i], y: y}})),
                                backgroundColor: 'rgba(65, 105, 225, 0.5)',
                                borderColor: 'rgba(65, 105, 225, 0.8)',
                                borderWidth: 1,
                                pointRadius: 3,
                                order: 3
                            }},
                            {{
                                label: 'Smooth Avg',
                                data: data.smooth.map((y, i) => ({{x: labels[i], y: y}})),
                                borderColor: 'orange',
                                borderWidth: 2,
                                pointRadius: 0,
                                type: 'line',
                                tension: 0.2,
                                order: 1
                            }},
                            {{
                                label: 'Global Avg',
                                data: labels.map(l => ({{x: l, y: data.avg}})),
                                borderColor: 'gray',
                                borderDash: [5, 5],
                                borderWidth: 1,
                                pointRadius: 0,
                                type: 'line',
                                order: 2
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'nearest', intersect: true }},
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const idx = context.dataIndex;
                                        if (context.dataset.type === 'line') return context.dataset.label + ': ' + context.raw.y;
                                        return [
                                            `Rank: ${{context.raw.x}}`,
                                            `Result: ${{context.raw.y}}`,
                                            `Profit: ${{data.act_profits[idx]}}`,
                                            `Score: ${{data.pred_scores[idx]}}`
                                        ];
                                    }}
                                }}
                            }}
                        }},
                        onClick: (e) => {{
                             const points = chart.getElementsAtEventForMode(e, 'nearest', {{ intersect: true }}, true);
                            if (points.length) {{
                                const firstPoint = points[0];
                                if (firstPoint.datasetIndex === 0) {{ // Only scatter points
                                     updateInfoPanel(filename, safeFname, firstPoint.index);
                                }}
                            }}
                        }}
                    }}
                }});
                charts[safeFname] = chart;
            }}

            function lazyLoadRankChart(rank) {{
                const chartId = 'chart-rank-' + rank;
                if (charts[chartId]) return;

                const data = rankHistory[rank];
                if (!data) return;

                // X-Axis labels: Start + Filenames
                const labels = ['Start'].concat(data.filenames);

                const ctx = document.getElementById(chartId).getContext('2d');
                const chart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Equity ($)',
                            data: data.equity_curve,
                            borderColor: 'green',
                            backgroundColor: 'rgba(0, 128, 0, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.1,
                            pointRadius: 0,
                            pointHoverRadius: 5
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ mode: 'index', intersect: false }},
                        scales: {{
                            x: {{ display: false }} // Hide dense labels
                        }},
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    title: (ctx) => ctx[0].label
                                }}
                            }}
                        }}
                    }}
                }});
                charts[chartId] = chart;
            }}

            function updateInfoPanel(filename, safeFname, idx) {{
                const data = reportData[filename];
                const infoTitle = document.getElementById('info-title-' + safeFname);
                const infoContent = document.getElementById('info-content-' + safeFname);

                infoTitle.textContent = `Vector Detail (Rank #${{data.pred_ranks[idx]}})`;
                infoContent.innerHTML = `
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Vector ID</td><td>${{data.indices[idx]}}</td></tr>
                        <tr><td><strong>Actual Profit</strong></td><td style="color: ${{data.act_profits[idx] >= 0 ? 'green' : 'red'}}"><strong>${{data.act_profits[idx]}}</strong></td></tr>
                        <tr><td>Actual Result</td><td>${{data.act_results[idx]}}</td></tr>
                        <tr><td>Prediction Score</td><td>${{data.pred_scores[idx]}}</td></tr>
                        <tr><td>Params</td><td style="font-size:0.8em; word-break:break-all;">${{data.params[idx]}}</td></tr>
                    </table>
                `;
            }}
        </script>
    </body>
    </html>
    """

    output_html_path = os.path.join(target_dir, "Vector_Prediction_Distribution.html")
    with open(output_html_path, "w", encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated successfully: {output_html_path}")

if __name__ == "__main__":
    main()
