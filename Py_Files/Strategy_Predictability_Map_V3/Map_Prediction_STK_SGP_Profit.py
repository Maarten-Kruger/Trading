
import os
import sys
import glob
import warnings
import json
import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import datetime

# pip install numpy pandas polars matplotlib torch gpytorch scikit-learn pyarrow

# --- Configuration ---
TRAIN_WINDOW = 10       # Number of past samples (files) to train on (Walk-Forward Window)
TOP_N = 200             # Number of top predicted vectors to evaluate
INITIAL_EQUITY = 10000  # Initial account balance for simulation
SMOOTHING_WINDOW = 25   # Window for smooth average line
MAX_WORKERS = 4

PER_FILE = 2000
HIGH = 0.33
AVG = 0.33
LOW = 0.34

# SGP Configuration
INDUCING_POINTS = 1000   # Number of inducing points for Sparse Gaussian Process
TRAINING_MAX = 200       # Maximum number of iterations for GP optimization
TRAINING_THRESHOLD = 0.0001 # Early stopping threshold for ELBO Loss improvement
STABILITY_WEIGHT = 1.0  # Kappa (κ). Higher values prioritize stability (lower variance), lower values prioritize expected return.
NOISE_FACTOR = 0.6 # Minimum noise level to prevent overfitting (added to likelihood noise constraint). 
                    #Percentage, e.g. 0.1 means noise will assumed to be at least 10% of the variance of the training targets.


# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------------

def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    return str(datetime.timedelta(seconds=int(seconds)))

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

def read_csv_polars(filepath):
    """
    Reads a CSV file robustly using Polars, handling different separators and decimal formats.
    """
    try:
        # Detect separator by reading first line
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            if not header: return pl.DataFrame()

        sep = ';' if ';' in header else ','
        decimal = ',' if sep == ';' else '.' # Often related

        try:
            df = pl.read_csv(filepath, separator=sep, infer_schema_length=10000, ignore_errors=True)
        except:
             # Fallback: try different encoding
             df = pl.read_csv(filepath, separator=sep, encoding='latin1', ignore_errors=True)

        # Convert numeric columns that might be strings due to comma decimals
        cols_to_numeric = ['Result', 'Profit', 'Trades']

        for col in cols_to_numeric:
            if col in df.columns:
                if df[col].dtype == pl.Utf8:
                    # Replace comma with dot and cast
                    df = df.with_columns(
                        pl.col(col).str.replace(',', '.').cast(pl.Float64, strict=False).alias(col)
                    )
                elif df[col].dtype != pl.Float64 and df[col].dtype != pl.Int64:
                     df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Fill nulls with 0.0
        df = df.fill_null(0.0)
        return df

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pl.DataFrame()

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

class SpatioTemporalSGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_spatial_dims):
        # We use a Cholesky variational distribution for the inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))

        # We use the standard VariationalStrategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SpatioTemporalSGP, self).__init__(variational_strategy)

        # The mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Spatial Kernel (Matern 5/2 usually works well for parameter spaces)
        self.covar_module_spatial = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, active_dims=tuple(range(num_spatial_dims)))
        )

        # Temporal Kernel (Matern 1/2 or RBF for time)
        self.covar_module_temporal = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=(num_spatial_dims,))
        )

        # Composite Kernel (Multiply Spatial and Temporal to capture space-time interactions)
        self.covar_module = self.covar_module_spatial * self.covar_module_temporal

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_raw_features_polars(df_list, master_vectors):
    """
    Builds the dataset of (Spatial_Params, Time_Index, Result, Profit) for each file.
    Does not compute rolling windows or hypercubes.
    df_list: List of Polars DataFrames (one per file, sorted chronologically).
    master_vectors: Polars DataFrame of vector parameters.
    """
    processed_files = []
    print("Compiling raw spatial-temporal data...")

    vector_cols = [c for c in master_vectors.columns if c != 'Master_Index']

    for file_idx, df in enumerate(df_list):
        merged = master_vectors.join(df, on=vector_cols, how='left').fill_null(0.0)

        # We keep the spatial parameters for each row, plus add the file_idx (time)
        results = merged['Result'].to_numpy()
        profits = merged['Profit'].to_numpy()

        # We also need the actual parameter columns flattened out into this dataframe
        # However, to save memory and avoid redundant strings, we will rely on vector_idx
        # But we do need the raw parameters for the SGP features.

        # Determine the length safely based on the array lengths
        n_vectors = len(results)

        # Make a smaller dataframe to save memory
        file_df = pl.DataFrame({
            'File_Index': np.full(n_vectors, file_idx, dtype=np.int16),
            'Vector_Index': np.arange(n_vectors, dtype=np.int32),
            'Result': results.astype(np.float32),
            'Profit': profits.astype(np.float32)
        })

        processed_files.append(file_df)

    full_df = pl.concat(processed_files)

    # We want to predict Result of NEXT file (Target)
    full_df = full_df.sort(['Vector_Index', 'File_Index'])

    target_expr = pl.col("Result").shift(-1).alias("Result_Next")
    profit_next_expr = pl.col("Profit").shift(-1).alias("Profit_Next")

    full_df = full_df.with_columns([
        target_expr.over("Vector_Index"),
        profit_next_expr.over("Vector_Index")
    ])

    return full_df

def generate_html_report(target_dir, report_data, rank_history, time_stats=None):
    if time_stats is None:
        time_stats = {}

    # Process Rank Summary Stats
    ranks_x = []
    max_dds_y = []
    avg_dds_y = []
    profits_y = []

    html_rank_rows = ""

    # Sort ranks numerically
    sorted_ranks = sorted(rank_history.keys())

    # Build Rank Charts Data
    rank_charts_data = {}

    for r in sorted_ranks:
        data = rank_history[r]
        stats = data['stats']

        ranks_x.append(r)
        max_dds_y.append(stats['max_dd'])
        avg_dds_y.append(stats['avg_dd'])
        profits_y.append(stats['total_pl'])

        # Save chart data to big object
        rank_charts_data[r] = {
            'filenames': data['filenames'],
            'equity_curve': data['equity_curve']
        }

        # Add HTML rows
        rank_safe_id = f"rank-{r}"

        html_rank_rows += f"""
        <div class="plot-container">
            <details class="modern-details">
                <summary onclick="lazyLoadRankChart({r})">
                    <span>Rank {r} Performance</span>
                    <span class="icon">▼</span>
                </summary>
                <div class="section-content">
                    <table style="width: 100%; margin-bottom: 20px; font-size: 0.9em; background: #f9f9f9;">
                         <tr><th>Total P/L ($)</th><th>Max Drawdown (%)</th><th>Avg Drawdown (%)</th><th>Sharpe Ratio</th></tr>
                        <tr>
                            <td style="font-weight:bold; color: { 'green' if stats['total_pl'] >= 0 else 'red' }">${stats['total_pl']:.2f}</td>
                            <td>{stats['max_dd']:.2f}%</td>
                            <td>{stats['avg_dd']:.2f}%</td>
                            <td>{stats['sharpe']:.3f}</td>
                        </tr>
                    </table>
                    <div style="height: 300px;"><canvas id="chart-{rank_safe_id}"></canvas></div>
                </div>
            </details>
        </div>
        """

    # File Rows
    html_file_rows = ""
    for filename in report_data.keys():
        safe_fname = filename.replace('.', '_').replace(' ', '_')

        # Rules HTML (now best weights)
        weights_str = report_data[filename]['weights_info']
        top_preds = report_data[filename].get('top_preds_table', [])

        table_html = """
        <table style="width: 100%; margin-top: 15px; font-size: 0.85em;">
            <tr>
                <th>Rank</th>
                <th>Vector Index</th>
                <th>Predicted Mean (\u03bc)</th>
                <th>Uncertainty Std (\u03c3)</th>
                <th>Alpha Score (\u03b1)</th>
                <th>Actual Result</th>
            </tr>
        """
        for i, row in enumerate(top_preds):
            table_html += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{row['vector_idx']}</td>
                <td>{row['predicted_mean']:.4f}</td>
                <td>{row['uncertainty_std']:.4f}</td>
                <td>{row['alpha_score']:.4f}</td>
                <td>{row['actual_result']:.4f}</td>
            </tr>
            """
        table_html += "</table>"

        html_file_rows += f"""
        <div class="plot-container">
            <details class="modern-details">
                <summary onclick="lazyLoadCharts('{filename}', '{safe_fname}')">
                    <span>{filename}</span>
                    <span class="icon">▼</span>
                </summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        <div style="flex: 1 1 100%; height: 500px;">
                            <h4>Prediction Distribution (Sorted by Rank)</h4>
                            <canvas id="chart-{safe_fname}"></canvas>
                        </div>
                        <div style="flex: 1 1 45%; min-width: 400px; height: 350px;">
                            <h4>ELBO Loss History (Training)</h4>
                            <canvas id="loss-{safe_fname}"></canvas>
                        </div>
                        <div style="flex: 1 1 45%; min-width: 400px; height: 350px;">
                            <h4>Confidence (\u03c3) vs Predicted Result (\u03bc) - Top {len(top_preds)}</h4>
                            <canvas id="scatter-{safe_fname}"></canvas>
                        </div>
                        <div style="flex: 1 1 100%; padding: 15px; background: #f0f8ff; border: 1px solid #cce5ff; border-radius: 8px; margin-top: 30px;">
                            <h4 style="margin-top:0;">SGP Model Info</h4>
                            <div style="font-family: monospace; font-size: 1.0em; color: #333;">
                                {weights_str}
                            </div>
                        </div>
                    </div>

                    <details style="margin-top: 15px;">
                        <summary style="background-color: #f1f3f5; font-size: 0.95em;">Top {len(top_preds)} Predictions Table</summary>
                        <div style="max-height: 300px; overflow-y: auto;">
                            {table_html}
                        </div>
                    </details>
                </div>
            </details>
        </div>
        """

    # JSON Dumps
    json_summary = json.dumps({
        'ranks': ranks_x,
        'max_dd': max_dds_y,
        'avg_dd': avg_dds_y,
        'profit': profits_y
    })

    # Clean report data for JSON
    json_report = json.dumps(report_data)
    json_rank_hist = json.dumps(rank_charts_data)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Sequential Distribution Optimizer Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f9; }}
            .container {{ max-width: 1600px; margin: 0 auto; padding-bottom: 50px; }}
            .plot-container {{ margin-bottom: 15px; border: none; padding: 0; background: transparent; }}

            .modern-details {{
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                overflow: hidden;
            }}
            .modern-details summary {{
                cursor: pointer;
                font-weight: 600;
                font-size: 1.05em;
                padding: 15px 20px;
                background-color: #f8f9fa;
                border-bottom: 1px solid transparent;
                transition: all 0.2s ease;
                display: flex;
                justify-content: space-between;
                align-items: center;
                list-style: none; /* Hide default arrow */
            }}
            .modern-details summary::-webkit-details-marker {{
                display: none;
            }}
            .modern-details summary:hover {{
                background-color: #e9ecef;
            }}
            .modern-details[open] summary {{
                border-bottom: 1px solid #e0e0e0;
                background-color: #e9ecef;
            }}
            .modern-details summary .icon {{
                font-size: 0.8em;
                transition: transform 0.3s ease;
            }}
            .modern-details[open] summary .icon {{
                transform: rotate(180deg);
            }}
            .section-content {{ padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
            .config-box {{ background: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 5px; margin-bottom: 20px; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">Spatio-Temporal Kriging SGP Report</h1>

        
            <div class="config-box">
                <div style="display: flex; justify-content: space-between;">
                    <div style="flex: 2;">
                        <h3 style="margin-top:0;">Configuration & Parameters</h3>
                        <ul style="column-count: 2; list-style-type: none; padding: 0; margin: 0;">
                            <li><strong>Engine:</strong> GPyTorch + CUDA</li>
                            <li><strong>Train Window (Time):</strong> {TRAIN_WINDOW} files</li>
                            <li><strong>Top N Predictions:</strong> {TOP_N}</li>
                            <li><strong>Inducing Points:</strong> {INDUCING_POINTS}</li>
                            <li><strong>Training Max:</strong> {TRAINING_MAX}</li>
                            <li><strong>Training Threshold:</strong> {TRAINING_THRESHOLD}</li>
                            <li><strong>Stability Weight (\u03ba):</strong> {STABILITY_WEIGHT}</li>
                            <li><strong>Workers:</strong> {MAX_WORKERS}</li>
                            <li><strong>Samples Per File:</strong> {PER_FILE}</li>
                            <li><strong>Sampling Weights:</strong> High: {HIGH} | Avg: {AVG} | Low: {LOW}</li>
                        </ul>
                    </div>



                    <div style="flex: 1; border-left: 2px solid #ffeeba; padding-left: 20px;">
                        <h3 style="margin-top:0;">Performance Stats</h3>
                        <ul style="list-style-type: none; padding: 0; margin: 0;">
                            <li><strong>Total Time:</strong> {time_stats.get('total_time', 'N/A')}</li>
                            <li><strong>Avg Time per File:</strong> {time_stats.get('avg_time_per_file', 'N/A')}</li>
                            <li><strong>Steps per File:</strong> {time_stats.get('steps_per_file', 'N/A')}</li>
                            <li><strong>Avg Time per Step:</strong> {time_stats.get('avg_time_per_step', 'N/A')}</li>
                        </ul>
                    </div>
                </div>
            </div>

            <p style="text-align: center;"><strong>Score Formula:</strong> \u03b1(V) = \u03bc(V) - \u03ba \u00b7 \u03c3(V)</p>

            <details class="modern-details" open style="margin-bottom: 20px; border: 2px solid #6c757d;">
                <summary style="background-color: #6c757d; color: white;">
                    <span>Global Performance Summary</span>
                    <span class="icon">▼</span>
                </summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1; height: 400px;"><canvas id="summary-profit"></canvas></div>
                        <div style="flex: 1; height: 400px;"><canvas id="summary-drawdown"></canvas></div>
                    </div>
                </div>
            </details>

            <details class="modern-details" style="margin-bottom: 20px; border: 2px solid #007bff;">
                <summary style="background-color: #007bff; color: white;">
                    <span>File-by-File Analysis (Walk-Forward)</span>
                    <span class="icon">▼</span>
                </summary>
                <div class="section-content" style="background: #f8f9fa;">
                    {html_file_rows}
                </div>
            </details>

            <details class="modern-details" style="margin-bottom: 20px; border: 2px solid #17a2b8;">
                <summary style="background-color: #17a2b8; color: white;">
                    <span>Rank Analysis (Longitudinal)</span>
                    <span class="icon">▼</span>
                </summary>
                <div class="section-content" style="background: #f8f9fa;">
                    {html_rank_rows}
                </div>
            </details>
        </div>

        <script>
            const reportData = {json_report};
            const summaryData = {json_summary};
            const rankHistory = {json_rank_hist};
            const charts = {{}};

            // Summary Charts
            document.addEventListener('DOMContentLoaded', () => {{
                new Chart(document.getElementById('summary-profit'), {{
                    type: 'bar',
                    data: {{
                        labels: summaryData.ranks,
                        datasets: [{{
                            label: 'Total Profit ($)',
                            data: summaryData.profit,
                            backgroundColor: summaryData.profit.map(val => val < 0 ? 'rgba(255, 99, 132, 0.6)' : 'rgba(75, 192, 192, 0.6)')
                        }}]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: 'Profit by Rank' }} }} }}
                }});

                new Chart(document.getElementById('summary-drawdown'), {{
                    type: 'bar',
                    data: {{
                        labels: summaryData.ranks,
                        datasets: [{{
                            label: 'Max Drawdown (%)',
                            data: summaryData.max_dd,
                            backgroundColor: 'rgba(255, 99, 132, 0.6)'
                        }}]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: 'Risk by Rank' }} }} }}
                }});
            }});

            function lazyLoadCharts(filename, safeFname) {{
                if (charts[safeFname]) return;
                const data = reportData[filename];
                const ctx = document.getElementById('chart-' + safeFname).getContext('2d');
                const scatterCtx = document.getElementById('scatter-' + safeFname).getContext('2d');
                const lossCtx = document.getElementById('loss-' + safeFname).getContext('2d');

                charts['loss-' + safeFname] = new Chart(lossCtx, {{
                    type: 'line',
                    data: {{
                        labels: Array.from({{length: data.loss_history.length}}, (_, i) => i + 1),
                        datasets: [{{
                            label: 'ELBO Loss',
                            data: data.loss_history,
                            borderColor: 'red',
                            backgroundColor: 'rgba(255, 0, 0, 0.1)',
                            fill: true,
                            tension: 0.1,
                            pointRadius: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{ title: {{ display: true, text: 'Iteration' }} }},
                            y: {{ title: {{ display: true, text: 'Loss' }} }}
                        }}
                    }}
                }});

                charts[safeFname] = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [
                            {{
                                type: 'line',
                                label: 'Smooth Trend',
                                data: data.pred_ranks.map((r, i) => ({{x: r, y: data.smooth[i]}})),
                                borderColor: 'orange',
                                borderWidth: 2,
                                pointRadius: 0,
                                tension: 0.3
                            }},
                            {{
                                label: 'Actual Result',
                                data: data.pred_ranks.map((r, i) => ({{x: r, y: data.act_results[i]}})),
                                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                pointRadius: 2
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{ type: 'linear', title: {{ display: true, text: 'Prediction Rank' }} }},
                            y: {{ title: {{ display: true, text: 'Actual Result' }} }}
                        }}
                    }}
                }});

                charts['scatter-' + safeFname] = new Chart(scatterCtx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [{{
                            label: '\u03c3 vs \u03bc',
                            data: data.top_preds_table.map(row => ({{x: row.predicted_mean, y: row.uncertainty_std}})),
                            backgroundColor: 'rgba(153, 102, 255, 0.6)',
                            pointRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{ type: 'linear', title: {{ display: true, text: 'Predicted Mean (\u03bc)' }} }},
                            y: {{ type: 'linear', title: {{ display: true, text: 'Uncertainty (\u03c3)' }} }}
                        }},
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        let pt = data.top_preds_table[context.dataIndex];
                                        return `Idx: ${{pt.vector_idx}} | \u03bc: ${{pt.predicted_mean.toFixed(4)}} | \u03c3: ${{pt.uncertainty_std.toFixed(4)}}`;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }}

            function lazyLoadRankChart(rank) {{
                const id = 'chart-rank-' + rank;
                if (charts[id]) return;
                const data = rankHistory[rank];
                const ctx = document.getElementById(id).getContext('2d');

                charts[id] = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: ['Start', ...data.filenames],
                        datasets: [{{
                            label: 'Equity',
                            data: data.equity_curve,
                            segment: {{
                                borderColor: ctx => ctx.p0.parsed.y < 10000 || ctx.p1.parsed.y < 10000 ? 'red' : 'green',
                                backgroundColor: ctx => ctx.p0.parsed.y < 10000 || ctx.p1.parsed.y < 10000 ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 128, 0, 0.1)'
                            }},
                            fill: true,
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        elements: {{ point: {{ radius: 0, hitRadius: 10 }} }},
                        scales: {{
                            x: {{
                                ticks: {{ display: false }} // Hide labels to maximize space
                            }}
                        }}
                    }}
                }});
            }}
        </script>
    </body>
    </html>
    """

    with open(os.path.join(target_dir, "Map_Prediction_STK_SGP.html"), "w", encoding='utf-8') as f:
        f.write(html_content)


# --- Worker Function ---
def worker_process_file(task_args):
    """
    Independent worker function for parallel processing.
    """
    try:
        target_file_idx = task_args['target_file_idx']
        target_filename = task_args['target_filename']
        train_data_pd = task_args['train_data']
        pred_data_pd = task_args['pred_data']
        master_numpy = task_args['master_numpy'] # Used to look up spatial parameters

        # Set up Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare Training Data
        # Drop rows where target is NA/null
        train_data_pd = train_data_pd.dropna(subset=['Result_Next'])

        # Features: Spatial Parameters + Temporal Index
        # Extract Vector Indices to get spatial params
        train_v_indices = train_data_pd['Vector_Index'].values.astype(int)
        train_spatial = master_numpy[train_v_indices]
        train_temporal = train_data_pd['File_Index'].values.reshape(-1, 1)

        train_X_np = np.hstack([train_spatial, train_temporal])
        train_Y_np = train_data_pd['Result_Next'].values

        # Prepare Prediction Data (Input for target_file_idx)
        # We use the previous step's data to predict the target file.
        # But wait! For SGP predicting step K, the input is Spatial Params + Time K
        # The target file index IS time K. So we construct the test inputs using target_file_idx.
        test_v_indices = pred_data_pd['Vector_Index'].values.astype(int)
        test_spatial = master_numpy[test_v_indices]
        test_temporal = np.full((len(test_spatial), 1), target_file_idx) # Predict for Time K

        test_X_np = np.hstack([test_spatial, test_temporal])
        actual_results = pred_data_pd['Result_Next'].values # The actual results we are trying to predict
        actual_profits = pred_data_pd['Profit_Next'].values

        # Scale Data
        scaler_X = StandardScaler()
        train_X_scaled = scaler_X.fit_transform(train_X_np)
        test_X_scaled = scaler_X.transform(test_X_np)

        scaler_Y = StandardScaler()
        train_Y_scaled = scaler_Y.fit_transform(train_Y_np.reshape(-1, 1)).flatten()

        # Convert to PyTorch tensors
        train_X_tensor = torch.tensor(train_X_scaled, dtype=torch.float32, device=device)
        train_Y_tensor = torch.tensor(train_Y_scaled, dtype=torch.float32, device=device)
        test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32, device=device)

        # Initialize SGP Model
        num_dims = train_X_tensor.shape[1]

        # Select inducing points deterministically using KMeans clustering
        num_inducing = min(INDUCING_POINTS, train_X_tensor.shape[0])

        if num_inducing < train_X_tensor.shape[0]:
            kmeans = MiniBatchKMeans(n_clusters=num_inducing, random_state=42, batch_size=2048, n_init="auto")
            kmeans.fit(train_X_scaled)
            inducing_points_np = kmeans.cluster_centers_
            inducing_points = torch.tensor(inducing_points_np, dtype=torch.float32, device=device)
        else:
            inducing_points = train_X_tensor

        # Free some memory if possible
        del train_data_pd
        del pred_data_pd
        del train_X_np
        del train_Y_np
        del test_X_np
        del train_X_scaled
        del train_Y_scaled
        del test_X_scaled

        model = SpatioTemporalSGP(inducing_points=inducing_points, num_spatial_dims=num_dims-1).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        # Force the GP to assume a higher baseline of noise
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(NOISE_FACTOR) 
        ).to(device)

        # Optional: explicitly set the starting noise value before optimization
        likelihood.noise = NOISE_FACTOR

        # Train GP
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.1)

        # Loss for GP (Variational ELBO)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_Y_tensor.size(0))

        final_loss = 0.0
        prev_loss = float('inf')
        loss_history = []
        actual_iters = 0

        for i in range(TRAINING_MAX):
            optimizer.zero_grad()
            output = model(train_X_tensor)
            loss = -mll(output, train_Y_tensor)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            loss_history.append(final_loss)
            actual_iters += 1

            if abs(prev_loss - final_loss) < TRAINING_THRESHOLD:
                break
            prev_loss = final_loss

        # Prediction
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Process in chunks to avoid OOM
            chunk_size = 5000
            pred_mean_scaled_list = []
            pred_var_scaled_list = []
            for i in range(0, len(test_X_tensor), chunk_size):
                preds = likelihood(model(test_X_tensor[i:i+chunk_size]))
                pred_mean_scaled_list.append(preds.mean.cpu().numpy())
                pred_var_scaled_list.append(preds.variance.cpu().numpy())
            pred_mean_scaled = np.concatenate(pred_mean_scaled_list)
            pred_var_scaled = np.concatenate(pred_var_scaled_list)

        # Inverse transform mean and variance
        # Var(aX) = a^2 Var(X) => std(aX) = a * std(X)
        pred_mean = scaler_Y.inverse_transform(pred_mean_scaled.reshape(-1, 1)).flatten()
        pred_std = np.sqrt(pred_var_scaled) * scaler_Y.scale_[0]

        # Calculate Alpha (Ranking Score)
        # alpha(V) = mu(V) - kappa * sigma(V)
        alpha_scores = pred_mean - (STABILITY_WEIGHT * pred_std)

        # Format weights info to show ELBO Loss and info
        weights_info = (
            f"<b>Spatio-Temporal SGP Trained:</b><br>"
            f"Inducing Points: {num_inducing}<br>"
            f"Training Iterations: {actual_iters} / {TRAINING_MAX}<br>"
            f"Final ELBO Loss: {final_loss:.4f}<br>"
            f"Stability Weight (\u03ba): {STABILITY_WEIGHT}<br>"
        )

        # Generate Top N Output Table Data
        sorted_indices = np.argsort(alpha_scores)[::-1]
        top_n_indices = sorted_indices[:TOP_N]

        top_preds = []
        for idx in top_n_indices:
            top_preds.append({
                'vector_idx': int(idx),
                'predicted_mean': float(pred_mean[idx]),
                'uncertainty_std': float(pred_std[idx]),
                'alpha_score': float(alpha_scores[idx]),
                'actual_result': float(actual_results[idx])
            })

        return {
            'target_file_idx': target_file_idx,
            'target_filename': target_filename,
            'predicted_scores': alpha_scores,
            'actual_results': actual_results,
            'actual_profits': actual_profits,
            'best_fitness': -final_loss, # Using negative loss as proxy for fitness tracking
            'test_avg_res': 0.0, # Removed slope calc for simplicity, can add back if needed
            'test_slope': 0.0,
            'weights_info': weights_info,
            'top_preds_table': top_preds,
            'actual_iters': actual_iters,
            'loss_history': loss_history
        }

    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc(), 'target_file_idx': task_args['target_file_idx']}

# --- Main ---
def main():
    start_time_program = time.time()
    print("--- Spatio-Temporal Kriging (Sparse Gaussian Processes) Optimizer ---")
    print(f"Parallel Workers: {MAX_WORKERS}")
    print(f"SGP Config: INDUCING={INDUCING_POINTS}, MAX_ITER={TRAINING_MAX}, THRESHOLD={TRAINING_THRESHOLD}, STABILITY_WEIGHT={STABILITY_WEIGHT}")

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

    min_files = TRAIN_WINDOW + 2
    if len(csv_files) < min_files:
        print(f"Error: Not enough files. Need at least {min_files}.")
        return

    # 1. Load Master Vector Definitions
    print("Loading Master Vector Definitions...")
    first_file = csv_files[0]
    df_temp = read_csv_polars(first_file)
    cols = df_temp.columns
    try:
        trades_idx = cols.index('Trades')
        vector_cols = cols[trades_idx+1:]
    except ValueError:
        print("Error: 'Trades' column not found in first file.")
        return

    master_vectors = df_temp.select(vector_cols)
    master_vectors = master_vectors.with_columns(pl.lit(np.arange(len(master_vectors))).alias("Master_Index"))
    # Save raw numpy spatial parameters
    master_numpy = master_vectors.select(vector_cols).to_pandas().apply(pd.to_numeric, errors='coerce').fillna(0).values

    # 2. Load All Data and Build Features
    print("Loading all files...")
    df_list = []
    file_dates = []
    for f in csv_files:
        df_list.append(read_csv_polars(f))
        file_dates.append(os.path.basename(f))

    full_dataset = build_raw_features_polars(df_list, master_vectors)
    full_dataset = full_dataset.fill_null(0.0)

    # Convert entire dataset to Pandas ONCE to allow easy slicing and pickling for workers
    print("Converting Global Dataset to Pandas for Parallel Sharing...")
    full_dataset_pd = full_dataset.to_pandas()

    # 3. Prepare Parallel Tasks
    print("Preparing Tasks for Parallel Execution...")

    start_pred_idx = TRAIN_WINDOW + 1
    tasks = []

    for target_file_idx in range(start_pred_idx, len(csv_files)):
        target_filename = file_dates[target_file_idx]

        # Training Window: Ends at K-2 to predict K (using data from K-1)
        # Train: [K - Window - 1] to [K - 2]
        # Predict Input (to get spatial vector mapping): K - 1

        train_start = target_file_idx - TRAIN_WINDOW - 1
        train_end = target_file_idx - 2

        # For training data, downsample if too large, say 10,000 max samples uniformly
        # To avoid Out Of Memory errors on `multiprocessing` pass
        cols_to_keep = ['File_Index', 'Vector_Index', 'Result', 'Result_Next', 'Profit_Next']
        train_data_raw = full_dataset_pd[
            (full_dataset_pd['File_Index'] >= train_start) &
            (full_dataset_pd['File_Index'] <= train_end)
        ][cols_to_keep]

        # Deterministic sampling based on PER_FILE
        sampled_frames = []
        for f_idx in range(train_start, train_end + 1):
            file_data = train_data_raw[train_data_raw['File_Index'] == f_idx]
            n_total = len(file_data)

            if n_total == 0:
                continue

            # Sort by current Result
            file_data = file_data.sort_values(by='Result')

            target_per_file = min(PER_FILE, n_total)
            n_high = int(HIGH * target_per_file)
            n_avg = int(AVG * target_per_file)
            n_low = target_per_file - n_high - n_avg # Ensure they sum up correctly

            # Highest Results
            high_set = file_data.tail(n_high)

            # Lowest Results
            low_set = file_data.head(n_low)

            # Exclude already selected
            remaining = file_data.drop(high_set.index).drop(low_set.index)

            # Average Results
            if n_avg > 0 and len(remaining) > 0:
                mean_res = file_data['Result'].mean()
                remaining['DistToMean'] = (remaining['Result'] - mean_res).abs()
                remaining = remaining.sort_values(by='DistToMean')
                avg_set = remaining.head(min(n_avg, len(remaining))).drop(columns=['DistToMean'])
                sampled_frames.append(pd.concat([high_set, low_set, avg_set]))
            else:
                sampled_frames.append(pd.concat([high_set, low_set]))

        if sampled_frames:
            train_data = pd.concat(sampled_frames)
        else:
            train_data = pd.DataFrame(columns=cols_to_keep)

        pred_data = full_dataset_pd[full_dataset_pd['File_Index'] == (target_file_idx - 1)][cols_to_keep]

        tasks.append({
            'target_file_idx': target_file_idx,
            'target_filename': target_filename,
            'train_data': train_data,
            'pred_data': pred_data,
            'master_numpy': master_numpy
        })

    # Free the massive full_dataset
    del full_dataset_pd
    import gc
    gc.collect()

    data_loading_time = time.time() - start_time_program

    # Calculate steps per task
    N_rows = TRAIN_WINDOW * PER_FILE
    M_inducing = INDUCING_POINTS
    math_operations_per_iter = (M_inducing ** 3) + (N_rows * (M_inducing ** 2))
    steps_per_task = math_operations_per_iter * TRAINING_MAX

    print(f"Data loading completed in {format_time(data_loading_time)}.")
    print(f"Number of files to process: {len(tasks)}")
    print(f"Number of steps per task/worker: {steps_per_task:.2e}")

    # 5. Execute Parallel
    print(f"Submitting {len(tasks)} tasks to ProcessPoolExecutor (Workers={MAX_WORKERS})...")
    start_time_processing = time.time()

    rank_history = {r: {'filenames': [], 'results': [], 'profits': [], 'params': []} for r in range(1, TOP_N + 1)}
    report_data = {}

    global_params = master_vectors.select(vector_cols).to_pandas().astype(str).agg(', '.join, axis=1).tolist()

    # Use spawn context for CUDA safety if GPU enabled
    ctx = multiprocessing.get_context('spawn')

    def process_res(res):
        if 'error' in res:
            print(f"Task Failed (Idx {res['target_file_idx']}): {res['error']}")
            print(res['traceback'])
            return

        idx = res['target_file_idx']
        fname = res['target_filename']

        # Calculate Average Pred Mean and Std
        top_preds = res.get('top_preds_table', [])
        avg_pred_mean = np.mean([row['predicted_mean'] for row in top_preds]) if top_preds else 0.0
        avg_pred_std = np.mean([row['uncertainty_std'] for row in top_preds]) if top_preds else 0.0
        iters = res.get('actual_iters', 0)

        print(f"  Finished: {fname} | Iters: {iters} | ELBO Loss: {-res['best_fitness']:.4f} | Avg \u03bc(Top{TOP_N}): {avg_pred_mean:.4f} | Avg \u03c3(Top{TOP_N}): {avg_pred_std:.4f}")

        # Process Result
        scores = res['predicted_scores']
        act_res = res['actual_results']
        act_prof = res['actual_profits']

        # Sort
        sorted_indices = np.argsort(scores)[::-1]

        # Store History
        for i, vec_idx in enumerate(sorted_indices[:TOP_N]):
            rank = i + 1
            rank_history[rank]['filenames'].append(fname)
            rank_history[rank]['results'].append(float(act_res[vec_idx]))
            rank_history[rank]['profits'].append(float(act_prof[vec_idx]))
            rank_history[rank]['params'].append(global_params[vec_idx])

        # Report Data
        display_indices = sorted_indices[:2000]
        smooth_series = pd.Series(act_res[display_indices]).rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()

        # Convert top_preds_table values to native Python floats to avoid JSON serialization errors
        top_preds = res.get('top_preds_table', [])
        for row in top_preds:
            row['vector_idx'] = int(row['vector_idx'])
            row['predicted_mean'] = float(row['predicted_mean'])
            row['uncertainty_std'] = float(row['uncertainty_std'])
            row['alpha_score'] = float(row['alpha_score'])
            row['actual_result'] = float(row['actual_result'])

        report_data[fname] = {
            'pred_scores': [float(x) for x in np.round(scores[display_indices], 4)],
            'act_results': [float(x) for x in np.round(act_res[display_indices], 4)],
            'act_profits': [float(x) for x in np.round(act_prof[display_indices], 2)],
            'pred_ranks': list(range(1, len(display_indices) + 1)),
            'smooth': [float(x) for x in np.round(smooth_series.fillna(0).tolist(), 4)],
            'avg': float(np.round(np.mean(act_res[display_indices]), 4)),
            'weights_info': res['weights_info'],
            'top_preds_table': top_preds,
            'loss_history': res.get('loss_history', [])
        }

    # Process tasks in batches of size MAX_WORKERS
    num_tasks = len(tasks)
    total_processed = 0
    total_file_processing_time = 0.0

    print(f"\nStarting batch processing (Batch Size: {MAX_WORKERS})...")

    for i in range(0, num_tasks, MAX_WORKERS):
        batch_tasks = tasks[i:i+MAX_WORKERS]
        batch_start_time = time.time()

        # Spawn fresh workers FOR THIS BATCH ONLY to release VRAM
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
            # Submit batch
            futures = [executor.submit(worker_process_file, task) for task in batch_tasks]

            # Wait for the entire batch to finish
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                process_res(res)

        # Batch stats
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        total_file_processing_time += batch_duration

        files_in_batch = len(batch_tasks)
        total_processed += files_in_batch
        files_remaining = num_tasks - total_processed

        avg_time_per_file = total_file_processing_time / total_processed
        eta_seconds = files_remaining * avg_time_per_file
        total_program_time = time.time() - start_time_program

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Batch completed!")
        print(f"  Batch Time: {format_time(batch_duration)}")
        print(f"  Average Time Per File: {format_time(avg_time_per_file)}")
        print(f"  Files Remaining: {files_remaining}")
        print(f"  ETA for remaining files: {format_time(eta_seconds)}")
        print(f"  Total Running Time: {format_time(total_program_time)}\n")

    file_processing_time = time.time() - start_time_processing
    avg_time_per_file = file_processing_time / num_tasks if num_tasks > 0 else 0
    avg_time_per_step = avg_time_per_file / steps_per_task if steps_per_task > 0 else 0

    print(f"\nAll files processed.")
    print(f"Average time per step: {avg_time_per_step:.4e} seconds")

    print("\nGenerating Reports...")

    # Calculate stats
    for r in rank_history:
        data = rank_history[r]
        profits = data['profits']

        equity = [INITIAL_EQUITY]
        curr = INITIAL_EQUITY
        for p in profits:
            curr += p
            equity.append(curr)

        data['equity_curve'] = equity
        total_pl = curr - INITIAL_EQUITY
        max_dd, avg_dd = calculate_drawdowns(equity)
        sharpe = calculate_sharpe_ratio(profits)

        data['stats'] = {
            'total_pl': round(total_pl, 2),
            'max_dd': round(max_dd, 2),
            'avg_dd': round(avg_dd, 2),
            'sharpe': round(sharpe, 3)
        }

    time_stats = {
        'total_time': f"({format_time(data_loading_time)}) + ({format_time(file_processing_time)})",
        'avg_time_per_file': format_time(avg_time_per_file),
        'steps_per_file': steps_per_task,
        'avg_time_per_step': f"{avg_time_per_step:.4e} seconds"
    }

    generate_html_report(target_dir, report_data, rank_history, time_stats)
    print("HTML Report Generated.")

if __name__ == "__main__":
    main()
