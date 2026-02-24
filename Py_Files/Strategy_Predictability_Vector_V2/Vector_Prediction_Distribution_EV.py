
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
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import KDTree
from scipy.stats import spearmanr
import pygad

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
HYPERCUBE = 2           # Hypercube size (steps) for averaging neighbors
FEATURE_LOOKBACK = 5    # Number of past files to look back for feature calculation
TRAIN_WINDOW = 10       # Number of past samples (files) to train on (Walk-Forward Window)
TOP_N = 10000           # Number of top predicted vectors to evaluate
INITIAL_EQUITY = 10000  # Initial account balance for simulation
SMOOTHING_WINDOW = 25   # Window for smooth average line

# Genetic Algorithm Config
GA_NUM_GENERATIONS = 50
GA_SOL_PER_POP = 20
GA_NUM_PARENTS_MATING = 10
GA_MUTATION_PERCENT_GENES = 10

# ---------------------

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

class VectorOptimizer:
    def __init__(self, data_df, feature_cols, target_col='Result_Next'):
        """
        data_df: Polars DataFrame containing training data.
        feature_cols: List of column names to use as features.
        target_col: Column name of the target variable (Next Result).
        """
        self.data_pd = data_df.to_pandas() # PyGAD works best with numpy/pandas
        self.features = self.data_pd[feature_cols].values
        self.targets = self.data_pd[target_col].values
        self.feature_names = feature_cols
        self.num_features = len(feature_cols)

        # Genes per feature: [Rank_Weight, Threshold_Pct, Threshold_Weight]
        self.genes_per_feature = 3
        self.total_genes = self.num_features * self.genes_per_feature
        self.feature_ranks = None

    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Calculates Spearman Correlation between predicted scores and actual targets.
        """
        scores = self.calculate_scores(solution)

        # Calculate Correlation
        # We want high correlation with the Target (Result_Next)
        corr, _ = spearmanr(scores, self.targets)

        # Handle NaN
        if np.isnan(corr):
            return -1.0

        return corr

    def calculate_scores(self, solution):
        """
        Decodes the solution (genome) and calculates scores for all rows.
        solution: 1D array of weights/params.
        """
        # Initialize scores
        total_scores = np.zeros(len(self.features))

        for i in range(self.num_features):
            base_idx = i * self.genes_per_feature

            rank_weight = solution[base_idx]
            threshold_pct = solution[base_idx + 1] # -1.0 to 1.0 range from PyGAD, need to map to 0-1
            threshold_weight = solution[base_idx + 2]

            # Normalize threshold_pct from [-1, 1] to [0, 1]
            # Actually PyGAD range is configurable, I set -1 to 1.
            norm_threshold = (threshold_pct + 1) / 2

            # Term 1: Weighted Rank
            total_scores += self.feature_ranks[:, i] * rank_weight

            # Term 2: Threshold Boost
            mask = self.feature_ranks[:, i] > norm_threshold
            total_scores[mask] += threshold_weight

        return total_scores

    def precalculate_ranks(self):
        """
        Pre-calculates normalized ranks (0.0 to 1.0) for all features.
        """
        self.feature_ranks = np.zeros_like(self.features)
        for i in range(self.num_features):
            col_vals = self.features[:, i]
            # argsort().argsort() gives rank (0 to N-1)
            # Divide by N-1 to normalize to 0-1
            if len(col_vals) > 1:
                ranks = np.argsort(np.argsort(col_vals))
                self.feature_ranks[:, i] = ranks / (len(col_vals) - 1)
            else:
                self.feature_ranks[:, i] = 0.5

    def decode_rules(self, solution):
        """
        Returns a human-readable list of rules from the solution.
        """
        rules = []
        for i in range(self.num_features):
            base_idx = i * self.genes_per_feature
            rank_weight = solution[base_idx]
            threshold_pct_raw = solution[base_idx + 1]
            threshold_weight = solution[base_idx + 2]

            threshold_pct = (threshold_pct_raw + 1) / 2

            fname = self.feature_names[i]

            # Format:
            # Feature X: Rank Weight = 0.5, Boost if > 80% (+0.2)
            rule_str = f"<b>{fname}</b>:<br>"
            rule_str += f"&nbsp;&nbsp;Rank Weight: {rank_weight:.3f}<br>"
            if abs(threshold_weight) > 0.01:
                direction = "Score +=" if threshold_weight > 0 else "Score -="
                rule_str += f"&nbsp;&nbsp;IF Rank > {threshold_pct*100:.1f}% THEN {direction} {abs(threshold_weight):.3f}"
            else:
                rule_str += f"&nbsp;&nbsp;(Threshold Boost Negligible)"

            rules.append(rule_str)
        return rules

def build_features_polars(df_list, master_vectors, hypercube_neighbor_indices):
    """
    df_list: List of Polars DataFrames (one per file, sorted chronologically).
    master_vectors: Polars DataFrame of vector parameters.
    hypercube_neighbor_indices: List of neighbor indices for each vector (from KDTree).
    """

    processed_files = []
    print("Computing Hypercube Statistics...")

    # Pre-convert master vectors to numpy for join key check?
    # No, Polars join is fine.

    for file_idx, df in enumerate(df_list):
        vector_cols = [c for c in master_vectors.columns if c != 'Master_Index']
        merged = master_vectors.join(df, on=vector_cols, how='left').fill_null(0.0)

        results = merged['Result'].to_numpy()
        profits = merged['Profit'].to_numpy()

        hc_means = np.zeros_like(results)
        # hc_stds = np.zeros_like(results) # Unused in features currently, but good to have

        # Calculate HC Mean
        # Vectorized if possible, but list comprehension is robust
        # Optimization: neighbor_indices is a jagged array (list of arrays)
        # We can just loop.

        for v_idx, neighbors in enumerate(hypercube_neighbor_indices):
            if len(neighbors) > 0:
                hc_means[v_idx] = np.mean(results[neighbors])
            else:
                hc_means[v_idx] = results[v_idx]

        file_df = pl.DataFrame({
            'File_Index': file_idx,
            'Vector_Index': np.arange(len(results)),
            'Result': results,
            'Profit': profits,
            'Hypercube_Mean': hc_means
        })

        processed_files.append(file_df)

    full_df = pl.concat(processed_files)

    print("Computing Temporal Features (Rolling Windows)...")

    full_df = full_df.sort(['Vector_Index', 'File_Index'])

    # Feature Expressions
    res_mean_expr = pl.col("Result").rolling_mean(window_size=FEATURE_LOOKBACK).alias("Result_Mean")
    res_std_expr = pl.col("Result").rolling_std(window_size=FEATURE_LOOKBACK).alias("Result_Std")
    res_mom_expr = (pl.col("Result") - pl.col("Result").shift(1)).alias("Result_Momentum")

    hc_mean_expr = pl.col("Hypercube_Mean").rolling_mean(window_size=FEATURE_LOOKBACK).alias("HC_Mean_Mean")
    hc_mom_expr = (pl.col("Hypercube_Mean") - pl.col("Hypercube_Mean").shift(1)).alias("HC_Momentum")
    hc_std_temp_expr = pl.col("Hypercube_Mean").rolling_std(window_size=FEATURE_LOOKBACK).alias("HC_Temporal_Std")

    # Target: Result of NEXT file
    target_expr = pl.col("Result").shift(-1).alias("Result_Next")
    profit_next_expr = pl.col("Profit").shift(-1).alias("Profit_Next")

    full_df = full_df.with_columns([
        res_mean_expr.over("Vector_Index"),
        res_std_expr.over("Vector_Index"),
        res_mom_expr.over("Vector_Index"),
        hc_mean_expr.over("Vector_Index"),
        hc_mom_expr.over("Vector_Index"),
        hc_std_temp_expr.over("Vector_Index"),
        target_expr.over("Vector_Index"),
        profit_next_expr.over("Vector_Index")
    ])

    return full_df

def generate_final_verdict_pdf(output_path, rank1_data):
    """Generates a PDF report for the Top Rank."""
    with PdfPages(output_path) as pdf:
        plt.figure(figsize=(11, 8.5))
        plt.suptitle("Final Verdict Report: Top Rank Model", fontsize=20, weight='bold')

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

        ax1 = plt.axes([0.1, 0.1, 0.8, 0.5])
        ax1.plot(rank1_data['equity_curve'], color='green', linewidth=2)
        ax1.set_title("Equity Curve ($10,000 Start)")
        ax1.set_xlabel("Time (Files)")
        ax1.set_ylabel("Account Balance ($)")
        ax1.grid(True, alpha=0.3)

        pdf.savefig()
        plt.close()

def generate_html_report(target_dir, report_data, rank_history):
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

        # Only add HTML for top 100 to save DOM size? Or all?
        # User wants access to all? Lazy load handles it.
        rank_safe_id = f"rank-{r}"

        # Save chart data to big object
        rank_charts_data[r] = {
            'filenames': data['filenames'],
            'equity_curve': data['equity_curve']
        }

        html_rank_rows += f"""
        <div class="plot-container">
            <details>
                <summary onclick="lazyLoadRankChart({r})">Rank {r} Performance (Click to Expand)</summary>
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

        # Rules HTML
        rules_list = report_data[filename]['rules']
        rules_html = "<div style='margin-bottom:8px; border-bottom:1px solid #eee;'>" + "</div><div style='margin-bottom:8px; border-bottom:1px solid #eee;'>".join(rules_list) + "</div>"

        html_file_rows += f"""
        <div class="plot-container">
            <details>
                <summary onclick="lazyLoadCharts('{filename}', '{safe_fname}')">
                    {filename} (Click to Load Analysis)
                </summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 2; height: 500px;">
                            <h4>Prediction Distribution (Sorted by Rank)</h4>
                            <canvas id="chart-{safe_fname}"></canvas>
                        </div>
                        <div style="flex: 1; padding: 15px; background: #f0f8ff; border: 1px solid #cce5ff; border-radius: 8px; overflow-y: auto; max-height: 500px;">
                            <h4 style="margin-top:0;">Optimized Rules (Genome)</h4>
                            <div style="font-family: monospace; font-size: 0.85em;">
                                {rules_html}
                            </div>
                        </div>
                    </div>
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

    # Clean report data for JSON (remove rules to save space if needed, but they are text)
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
            .plot-container {{ margin-bottom: 15px; border: 1px solid #ddd; padding: 0; border-radius: 8px; background: #fff; overflow: hidden; }}
            summary {{ cursor: pointer; font-weight: 600; font-size: 1.05em; padding: 12px; background-color: #e9ecef; transition: background 0.2s; }}
            summary:hover {{ background-color: #dee2e6; }}
            .section-content {{ padding: 20px; border-top: 1px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">Sequential Distribution Optimizer Report</h1>
            <p style="text-align: center;"><strong>Engine:</strong> Polars + PyGAD | <strong>Fitness:</strong> Spearman Correlation | <strong>Lookback:</strong> {FEATURE_LOOKBACK}</p>

            <div style="text-align: center; margin-bottom: 30px;">
                 <a href="Final_Verdict.pdf" target="_blank" style="padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">Download Final Verdict PDF</a>
            </div>

            <details open style="margin-bottom: 40px; border: 2px solid #6c757d; border-radius: 8px;">
                <summary style="background-color: #6c757d; color: white;">Global Performance Summary</summary>
                <div class="section-content">
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1; height: 400px;"><canvas id="summary-profit"></canvas></div>
                        <div style="flex: 1; height: 400px;"><canvas id="summary-drawdown"></canvas></div>
                    </div>
                </div>
            </details>

            <h3>File-by-File Analysis (Walk-Forward)</h3>
            {html_file_rows}

            <h3>Rank Analysis (Longitudinal)</h3>
            {html_rank_rows}
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
                            backgroundColor: 'rgba(75, 192, 192, 0.6)'
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
                            borderColor: 'green',
                            fill: true,
                            backgroundColor: 'rgba(0, 128, 0, 0.1)'
                        }}]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, elements: {{ point: {{ radius: 0, hitRadius: 10 }} }} }}
                }});
            }}
        </script>
    </body>
    </html>
    """

    with open(os.path.join(target_dir, "Vector_Prediction_Distribution_EV.html"), "w", encoding='utf-8') as f:
        f.write(html_content)

def main():
    print("--- Sequential Distribution Optimizer (Polars + PyGAD) ---")

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # Default for testing? Or prompt?
        # Let's prompt.
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

    # Need minimum files
    min_files = FEATURE_LOOKBACK + TRAIN_WINDOW
    if len(csv_files) < min_files:
        print(f"Error: Not enough files. Need at least {min_files}.")
        return

    # --- 1. Load Master Vector Definitions ---
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

    # KDTree Prep
    master_numpy = master_vectors.select(vector_cols).to_pandas().apply(pd.to_numeric, errors='coerce').fillna(0).values

    # --- 2. Build KDTree ---
    print(f"Building KDTree (Hypercube={HYPERCUBE})...")
    tree = KDTree(master_numpy, metric='chebyshev')
    neighbor_indices = tree.query_radius(master_numpy, r=HYPERCUBE)

    # --- 3. Load All Data and Build Features ---
    print("Loading all files...")
    df_list = []
    file_dates = []
    for f in csv_files:
        df_list.append(read_csv_polars(f))
        file_dates.append(os.path.basename(f))

    full_dataset = build_features_polars(df_list, master_vectors, neighbor_indices)
    full_dataset = full_dataset.fill_null(0.0)

    # --- 4. Walk-Forward Optimization Loop ---
    print("Starting Walk-Forward Optimization...")

    feature_cols = [
        "Result_Mean", "Result_Std", "Result_Momentum",
        "HC_Mean_Mean", "HC_Momentum", "HC_Temporal_Std"
    ]

    rank_history = {r: {'filenames': [], 'results': [], 'profits': [], 'params': []} for r in range(1, TOP_N + 1)}
    report_data = {}
    best_rules_log = {}

    start_pred_idx = FEATURE_LOOKBACK + TRAIN_WINDOW

    for target_file_idx in range(start_pred_idx, len(csv_files)):
        target_filename = file_dates[target_file_idx]
        print(f"\nProcessing Target: {target_filename} (File {target_file_idx+1}/{len(csv_files)})")

        train_start = target_file_idx - TRAIN_WINDOW - 1
        train_end = target_file_idx - 2

        print(f"  Training on Files: {train_start} to {train_end}")

        train_data = full_dataset.filter(
            (pl.col("File_Index") >= train_start) &
            (pl.col("File_Index") <= train_end)
        )

        optimizer = VectorOptimizer(train_data, feature_cols, target_col='Result_Next')
        optimizer.precalculate_ranks()

        def fitness_wrapper(ga_instance, solution, solution_idx):
            return optimizer.fitness_func(ga_instance, solution, solution_idx)

        ga_instance = pygad.GA(
            num_generations=GA_NUM_GENERATIONS,
            num_parents_mating=GA_NUM_PARENTS_MATING,
            fitness_func=fitness_wrapper,
            sol_per_pop=GA_SOL_PER_POP,
            num_genes=optimizer.total_genes,
            init_range_low=-1.0,
            init_range_high=1.0,
            mutation_percent_genes=GA_MUTATION_PERCENT_GENES,
            keep_parents=2,
            suppress_warnings=True
        )

        if target_file_idx == start_pred_idx:
             print("  Running Genetic Algorithm...")

        ga_instance.run()

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        print(f"  Best Fitness (Spearman): {best_solution_fitness:.4f}")

        rules = optimizer.decode_rules(best_solution)
        best_rules_log[target_filename] = rules

        # Predict for Target
        pred_input_file_idx = target_file_idx - 1
        pred_data = full_dataset.filter(pl.col("File_Index") == pred_input_file_idx)

        predictor = VectorOptimizer(pred_data, feature_cols, target_col='Result_Next')
        predictor.precalculate_ranks()

        predicted_scores = predictor.calculate_scores(best_solution)

        # Target Result is 'Result_Next' of the pred_data
        actual_results = predictor.targets
        actual_profits = pred_data['Profit_Next'].to_numpy()

        sorted_indices = np.argsort(predicted_scores)[::-1]

        global_params = master_vectors.select(vector_cols).to_pandas().astype(str).agg(', '.join, axis=1).tolist()

        for i, idx in enumerate(sorted_indices[:TOP_N]):
            rank = i + 1
            rank_history[rank]['filenames'].append(target_filename)
            rank_history[rank]['results'].append(float(actual_results[idx]))
            rank_history[rank]['profits'].append(float(actual_profits[idx]))
            rank_history[rank]['params'].append(global_params[idx])

        display_indices = sorted_indices[:2000]
        smooth_series = pd.Series(actual_results[display_indices]).rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()

        report_data[target_filename] = {
            'pred_scores': np.round(predicted_scores[display_indices], 4).tolist(),
            'act_results': np.round(actual_results[display_indices], 4).tolist(),
            'act_profits': np.round(actual_profits[display_indices], 2).tolist(),
            'pred_ranks': list(range(1, len(display_indices) + 1)),
            'smooth': np.round(smooth_series.fillna(0).tolist(), 4).tolist(),
            'avg': round(np.mean(actual_results[display_indices]), 4),
            'rules': best_rules_log[target_filename]
        }

    print("\nGenerating Reports...")

    # Calculate stats for all ranks before generating reports
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

    pdf_path = os.path.join(target_dir, "Final_Verdict.pdf")
    generate_final_verdict_pdf(pdf_path, rank_history[1])
    print(f"PDF Saved: {pdf_path}")

    generate_html_report(target_dir, report_data, rank_history)
    print("HTML Report Generated.")

if __name__ == "__main__":
    main()
