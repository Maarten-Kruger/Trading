
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
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
HYPERCUBE = 2           # Hypercube size (steps) for averaging neighbors
FEATURE_LOOKBACK = 5    # Number of past files to look back for feature calculation
TRAIN_WINDOW = 10       # Number of past samples (files) to train on
TOP_N = 10000           # Number of top predicted vectors to evaluate
INITIAL_EQUITY = 10000  # Initial account balance for simulation
SMOOTHING_WINDOW = 25   # Window for smooth average line
EMA_WEIGHT = 0.6        # Weight for Exponential Moving Average
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

def generate_diagnostics_report(output_dir, all_predictions, feature_importance_map):
    """Generates a separate HTML report for model diagnostics."""

    # 1. Prediction Accuracy Scatter
    # Sample down if too large
    if len(all_predictions['pred']) > 20000:
        indices = np.random.choice(len(all_predictions['pred']), 20000, replace=False)
        preds = np.array(all_predictions['pred'])[indices]
        actuals = np.array(all_predictions['actual'])[indices]
    else:
        preds = np.array(all_predictions['pred'])
        actuals = np.array(all_predictions['actual'])

    # 2. Residuals
    residuals = actuals - preds

    # 3. Feature Importance Data
    feats = list(feature_importance_map.keys())
    scores = list(feature_importance_map.values())

    # Sort by importance
    sorted_idx = np.argsort(scores)[::-1]
    sorted_feats = [feats[i] for i in sorted_idx]
    sorted_scores = [scores[i] for i in sorted_idx]

    # JS Data
    diag_data = {
        'scatter': [{'x': float(p), 'y': float(a)} for p, a in zip(preds, actuals)],
        'residuals': [float(r) for r in residuals],
        'feat_names': sorted_feats,
        'feat_scores': [float(s) for s in sorted_scores]
    }

    # Feature Descriptions Glossary
    glossary_html = """
    <div style="width: 80%; margin: 0 auto 40px auto; background: #f9f9f9; padding: 20px; border: 1px solid #ddd;">
        <h3>Feature Glossary</h3>
        <table style="width:100%; text-align:left; border-collapse: collapse;">
            <tr><th style="padding:8px; border-bottom:1px solid #ccc;">Feature Name</th><th style="padding:8px; border-bottom:1px solid #ccc;">Description</th></tr>
            <tr><td style="padding:8px;"><strong>Trend_Slope</strong></td><td style="padding:8px;">Linear regression slope of the result history over the lookback period (Direction of performance).</td></tr>
            <tr><td style="padding:8px;"><strong>Risk_Adjusted_Return</strong></td><td style="padding:8px;">Mean result divided by standard deviation over the lookback period (Sharpe-like metric).</td></tr>
            <tr><td style="padding:8px;"><strong>Hypercube_Stability_Mean</strong></td><td style="padding:8px;">Average standard deviation of spatial neighbors over time (Spatial stability).</td></tr>
            <tr><td style="padding:8px;"><strong>Temporal_Stability_Std</strong></td><td style="padding:8px;">Standard deviation of the result history itself (Temporal stability).</td></tr>
            <tr><td style="padding:8px;"><strong>EMA_Prev</strong></td><td style="padding:8px;">Exponential Moving Average of the Hypercube Average at the previous time step.</td></tr>
            <tr><td style="padding:8px;"><strong>Param_*</strong></td><td style="padding:8px;">Static parameter value defining the vector (e.g., MA Period).</td></tr>
            <tr><td style="padding:8px;"><strong>Res_Lag_*</strong></td><td style="padding:8px;">Raw historical result at a specific lag (e.g., Lag 1 = Previous File).</td></tr>
            <tr><td style="padding:8px;"><strong>HC_Avg_Lag_*</strong></td><td style="padding:8px;">Raw historical hypercube average at a specific lag.</td></tr>
        </table>
    </div>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vector Model Diagnostics</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            .chart-box {{ width: 80%; margin: 0 auto 40px auto; height: 400px; border: 1px solid #ccc; padding: 10px; }}
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">GBDT Model Diagnostics</h1>

        {glossary_html}

        <div class="chart-box">
            <h3>Feature Importance</h3>
            <canvas id="featChart"></canvas>
        </div>

        <div class="chart-box">
            <h3>Prediction vs Actual</h3>
            <canvas id="scatterChart"></canvas>
        </div>

        <div class="chart-box">
            <h3>Residual Distribution (Actual - Pred)</h3>
            <canvas id="residChart"></canvas>
        </div>

        <script>
            const data = {json.dumps(diag_data)};

            // Feature Importance
            new Chart(document.getElementById('featChart'), {{
                type: 'bar',
                data: {{
                    labels: data.feat_names,
                    datasets: [{{
                        label: 'Gain / Importance',
                        data: data.feat_scores,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)'
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});

            // Scatter
            new Chart(document.getElementById('scatterChart'), {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Pred vs Actual',
                        data: data.scatter,
                        backgroundColor: 'rgba(255, 99, 132, 0.5)'
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Predicted Score' }} }},
                        y: {{ title: {{ display: true, text: 'Actual Result' }} }}
                    }}
                }}
            }});

            // Residual Histogram (Approximate with Bar)
            // Binning logic for JS? Let's just pass raw data and use a plugin or simple line
            // For simplicity, let's just show the residuals as a line chart (series)
            // Actually, let's just bin it in Python to save space?
            // Nah, let's just plot the series for now to see stability.
            new Chart(document.getElementById('residChart'), {{
                type: 'line',
                data: {{
                    labels: data.residuals.map((_, i) => i),
                    datasets: [{{
                        label: 'Residuals over time (Sample)',
                        data: data.residuals,
                        borderColor: 'purple',
                        borderWidth: 1,
                        pointRadius: 0
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
        </script>
    </body>
    </html>
    """

    with open(os.path.join(output_dir, "Vector_Model_Diagnostics.html"), "w") as f:
        f.write(html_content)
    print("Diagnostics report generated.")

def calculate_slope(y):
    """Calculate slope of linear regression for 1D array y."""
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n)
    # Slope formula: (n*Sum(xy) - Sum(x)*Sum(y)) / (n*Sum(x^2) - (Sum(x))^2)
    # Using numpy covariance for speed? Or polyfit
    # Vectorized approach for multiple vectors?
    # This function expects a single 1D array.
    try:
        slope, _ = np.polyfit(x, y, 1)
        return slope
    except:
        return 0.0

def build_features(target_idx, all_results, all_hypercube_avgs, all_hypercube_stds, all_emas, vector_static_features, num_vectors):
    """
    Builds feature matrix X for a specific time step `target_idx`.
    Features are derived from history [target_idx - FEATURE_LOOKBACK, target_idx - 1].
    """

    # 1. Collect History Arrays
    # We need arrays of shape (FEATURE_LOOKBACK, num_vectors)
    hist_results = []
    hist_hc_avgs = []
    hist_hc_stds = []

    for lag in range(1, FEATURE_LOOKBACK + 1):
        past_idx = target_idx - lag
        hist_results.append(all_results[past_idx])
        hist_hc_avgs.append(all_hypercube_avgs[past_idx])
        hist_hc_stds.append(all_hypercube_stds[past_idx])

    # Shape: (FEATURE_LOOKBACK, num_vectors)
    hist_results = np.array(hist_results)
    hist_hc_avgs = np.array(hist_hc_avgs)
    hist_hc_stds = np.array(hist_hc_stds)

    # --- Feature Engineering ---
    features_list = []
    feature_names = []

    # A. Static Vector Params
    features_list.append(vector_static_features)
    feature_names.extend([f"Param_{i}" for i in range(vector_static_features.shape[1])])

    # B. Trend (Slope of Results)
    # Vectorized Polyfit?
    # Slope = (Mean(xy) - Mean(x)Mean(y)) / Var(x)
    # x is 0, 1, 2... Lookback-1
    # We can precompute x constants.
    x = np.arange(FEATURE_LOOKBACK)
    mx = x.mean()
    vx = x.var()
    # For each vector (column), calculate covariance with x
    # Mean(y) per vector
    my = hist_results.mean(axis=0)
    # Mean(xy)
    # We broadcast x across the 0-axis
    mxy = (hist_results.T * x).T.mean(axis=0)

    slopes = (mxy - mx * my) / vx if vx != 0 else np.zeros(num_vectors)
    features_list.append(slopes.reshape(-1, 1))
    feature_names.append("Trend_Slope")

    # C. Risk-Adjusted Return (Mean / Std of Results)
    stds = hist_results.std(axis=0)
    # Avoid div by zero
    safe_stds = np.where(stds == 0, 1.0, stds)
    risk_adj = my / safe_stds
    features_list.append(risk_adj.reshape(-1, 1))
    feature_names.append("Risk_Adjusted_Return")

    # D. Hypercube Stability (Mean of Hypercube Stds)
    # Represents spatial stability over time
    hc_stability = hist_hc_stds.mean(axis=0)
    features_list.append(hc_stability.reshape(-1, 1))
    feature_names.append("Hypercube_Stability_Mean")

    # E. Temporal Stability (Std of Results)
    features_list.append(stds.reshape(-1, 1))
    feature_names.append("Temporal_Stability_Std")

    # F. Recent EMA (at t-1)
    # all_emas has data up to the latest file.
    # We need EMA available *before* target_idx, i.e., at target_idx-1
    ema_prev = all_emas[target_idx - 1]
    features_list.append(ema_prev.reshape(-1, 1))
    feature_names.append("EMA_Prev")

    # G. Lags (Raw History)
    # Add raw results and hypercube avgs from history
    # Flatten history: Lag1_Res, Lag1_HC, Lag2_Res...
    for i in range(FEATURE_LOOKBACK):
        # i=0 is Lag1 (most recent) because we appended in order target-1...target-Lookback
        # Actually my loop above was: target-1, target-2...
        # So hist_results[0] is target-1 (Lag 1)
        features_list.append(hist_results[i].reshape(-1, 1))
        feature_names.append(f"Res_Lag_{i+1}")

        features_list.append(hist_hc_avgs[i].reshape(-1, 1))
        feature_names.append(f"HC_Avg_Lag_{i+1}")

    # Combine
    X = np.hstack(features_list)
    return X, feature_names

def main():
    # Use Agg backend
    plt.switch_backend('Agg')

    print("--- Vector Prediction Distribution Generator (GBDT) ---")

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

    # Need at least FEATURE_LOOKBACK + TRAIN_WINDOW files to start predicting
    # We predict file K. We need K >= FEATURE_LOOKBACK + TRAIN_WINDOW + 1 (if 0-indexed)
    min_files_needed = FEATURE_LOOKBACK + TRAIN_WINDOW + 1
    if len(csv_files) < min_files_needed:
        print(f"Error: Not enough files ({len(csv_files)}). Need at least {min_files_needed}.")
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
    num_vectors = len(master_vectors)

    # Create Static Features Matrix (Vector Params)
    # Ensure they are numeric
    # master_vectors is currently valid types? read_csv_robust might leave them as is if they looked like numbers?
    # Let's force numeric
    vector_static_features_df = master_vectors[vector_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    vector_static_features = vector_static_features_df.values # (num_vectors, num_params)

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
    print("Pre-computing Hypercube Stats for all files...")
    all_file_hypercube_avgs = []
    all_file_hypercube_stds = []
    all_file_results = []
    all_file_profits = []
    all_file_dates = []
    valid_files_indices = []

    for i, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        current_df = read_csv_robust(filepath)

        if current_df.empty or 'Result' not in current_df.columns:
            # Fallback for empty/corrupt
            zero_arr = np.zeros(num_vectors)
            all_file_hypercube_avgs.append(zero_arr)
            all_file_hypercube_stds.append(zero_arr)
            all_file_results.append(zero_arr)
            all_file_profits.append(zero_arr)
            continue

        merged_df = pd.merge(master_vectors, current_df, on=vector_cols, how='left')
        merged_df['Result'] = merged_df['Result'].fillna(0.0)
        merged_df['Profit'] = merged_df['Profit'].fillna(0.0)
        merged_df.sort_values(by='Master_Index', ascending=True, inplace=True)

        results = merged_df['Result'].values
        profits = merged_df['Profit'].values

        hypercube_avgs = np.zeros_like(results)
        hypercube_stds = np.zeros_like(results)

        for idx, neighbor_idxs in enumerate(master_neighbor_indices):
            neighbors = results[neighbor_idxs]
            mean_val = np.mean(neighbors)
            std_val = np.std(neighbors)
            hypercube_avgs[idx] = mean_val
            hypercube_stds[idx] = std_val

        all_file_hypercube_avgs.append(hypercube_avgs)
        all_file_hypercube_stds.append(hypercube_stds)
        all_file_results.append(results)
        all_file_profits.append(profits)
        all_file_dates.append(filename)
        valid_files_indices.append(i)

    # Calculate EMA History
    print("Calculating EMA History...")
    all_file_emas = []
    current_ema = np.zeros(num_vectors)
    if len(valid_files_indices) > 0:
         current_ema = all_file_hypercube_avgs[valid_files_indices[0]].copy()

    for k in range(len(valid_files_indices)):
        real_idx = valid_files_indices[k]
        val = all_file_hypercube_avgs[real_idx]
        if k == 0:
            current_ema = val
        else:
            current_ema = EMA_WEIGHT * val + (1 - EMA_WEIGHT) * current_ema
        all_file_emas.append(current_ema.copy())

    # 3. Prediction Loop
    print("Running GBDT Prediction Logic...")
    html_file_rows = ""
    report_data = {}
    rank_history = {r: {'filenames': [], 'results': [], 'profits': [], 'params': []} for r in range(1, TOP_N + 1)}

    # Diagnostics Container
    all_predictions_diag = {'pred': [], 'actual': []}
    latest_feature_importance = {}

    start_prediction_idx = FEATURE_LOOKBACK + TRAIN_WINDOW + 1

    for k in range(start_prediction_idx, len(valid_files_indices)):
        current_real_idx = valid_files_indices[k]
        current_filename = all_file_dates[current_real_idx]
        print(f"Predicting for file {k}: {current_filename}")

        # --- Train Model ---
        # Train on window: [k - TRAIN_WINDOW, k - 1]
        X_train_list = []
        y_train_list = []

        train_range = range(k - TRAIN_WINDOW, k)

        feature_names = [] # Catch names from first iteration

        for t in train_range:
            # Target
            y_t = all_file_results[valid_files_indices[t]]

            # Features (built from t-LOOKBACK to t-1)
            # t is the index of the file we are predicting in training
            X_t, f_names = build_features(t, all_file_results, all_file_hypercube_avgs,
                                          all_file_hypercube_stds, all_file_emas,
                                          vector_static_features, num_vectors)
            if not feature_names:
                feature_names = f_names

            X_train_list.append(X_t)
            y_train_list.append(y_t)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            n_jobs=-1,
            tree_method="hist",
            random_state=42
        )
        model.fit(X_train, y_train)

        # Save feature importance
        if k == len(valid_files_indices) - 1: # Save last one
            latest_feature_importance = dict(zip(feature_names, model.feature_importances_))

        # --- Predict Current Step (k) ---
        X_pred, _ = build_features(k, all_file_results, all_file_hypercube_avgs,
                                   all_file_hypercube_stds, all_file_emas,
                                   vector_static_features, num_vectors)

        prediction_scores = model.predict(X_pred)

        # Diagnostics: Accumulate
        current_results = all_file_results[current_real_idx]
        current_profits = all_file_profits[current_real_idx]

        all_predictions_diag['pred'].extend(prediction_scores.tolist())
        all_predictions_diag['actual'].extend(current_results.tolist())

        # Select Top N Indices based on Prediction Score (Descending)
        top_n_indices = np.argsort(prediction_scores)[-TOP_N:][::-1]

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

    # Generate Diagnostics Report
    print("Generating Diagnostics Report...")
    generate_diagnostics_report(target_dir, all_predictions_diag, latest_feature_importance)

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
        <title>Vector Prediction Distribution Report (GBDT)</title>
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
            <h1>Vector Prediction Distribution Report (GBDT)</h1>
            <p><strong>Config:</strong> Lookback={FEATURE_LOOKBACK} | Hypercube={HYPERCUBE} | Top N={TOP_N} | Model=XGBoost</p>
            <p>
                <a href="Final_Verdict.pdf" target="_blank" style="padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">Download Final Verdict PDF</a>
                <a href="Vector_Model_Diagnostics.html" target="_blank" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin-left:10px;">View Model Diagnostics</a>
            </p>

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
