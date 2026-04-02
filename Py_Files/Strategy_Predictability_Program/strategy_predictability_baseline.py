
import pandas as pd
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import io
import base64
import warnings
import time
import traceback
import concurrent.futures
from collections import deque

# Suppress Warnings
warnings.filterwarnings("ignore")

# Configuration
RESULT_CUTOFF = 20
VECTOR_INPUT = 10  # Lookback window size
TRAINING_WINDOW = 30 # Not strictly used for training here, but defines the window logic
MAX_WORKERS = 1

# --- Helper Functions ---

def read_csv_robust(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            if not header: return pd.DataFrame()

        sep = ';' if ';' in header else ','
        decimal = ',' if sep == ';' else '.'
        df = pd.read_csv(filepath, sep=sep, decimal=decimal, engine='c')
    except:
        try:
             df = pd.read_csv(filepath, sep=None, engine='python')
        except:
             return pd.DataFrame()

    cols_to_numeric = ['Result', 'Profit', 'Trades']
    for c in cols_to_numeric:
        if c in df.columns and df[c].dtype == 'object':
             df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        try:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='ignore')
        except:
            pass

    return df

def process_file_load(filepath):
    try:
        df = read_csv_robust(filepath)
        if df.empty or 'Trades' not in df.columns or 'Result' not in df.columns:
            return None

        if 'Trades' in df.columns:
            trades_idx = df.columns.get_loc("Trades")
            var_cols = df.columns[trades_idx+1:].tolist()
        else:
            var_cols = []

        return {
            'path': filepath,
            'df': df,
            'var_cols': var_cols,
            'unique_vals': {col: df[col].dropna().unique() for col in var_cols}
        }
    except:
        return None

def preprocess_file_data(fdata, config):
    df = fdata['df']
    var_cols = fdata['var_cols']

    best_vector = {'params': {}, 'Result': 0}
    if not df.empty and 'Result' in df.columns:
        # Filter by cutoff
        valid_df = df[df['Result'] > RESULT_CUTOFF]

        # If no valid vectors, we might take the absolute best, or None
        # Original logic implies "Best Vector" filtering.
        # Requirements says: "Filter vectors... > RESULT_CUTOFF... identify Best Vector by Hypercube"
        # The original code just took df['Result'].idxmax().
        # I will stick to the original code's implementation for simplicity unless specified otherwise,
        # but the prompt asked to keep "baseline prediction graph".
        # Original code:
        # best_idx = df['Result'].idxmax()

        if not df.empty:
             best_idx = df['Result'].idxmax()
             best_row = df.loc[best_idx]
             best_vector = {
                 'params': best_row[var_cols].to_dict(),
                 'Result': best_row['Result']
             }

    return {
        'best_vector': best_vector
    }

def lookup_stats(df, params, config):
    if df.empty:
        return {'Result': 0, 'Profit': 0, 'found': False}

    # 1. Try Exact/Close Match
    mask = np.ones(len(df), dtype=bool)
    for col, val in params.items():
        if col in df.columns:
            step = config[col]['step']
            tol = step / 2.0 if step > 0 else 1e-7
            mask = mask & (np.abs(df[col] - val) <= tol)

    matches = df[mask]
    if not matches.empty:
        row = matches.loc[matches['Result'].idxmax()]
        matched_params = {k: row[k] for k in params.keys() if k in row}
        return {'Result': float(row['Result']), 'Profit': float(row['Profit']), 'found': True, 'matched_params': matched_params}

    # 2. Fallback: Nearest Neighbor
    total_dist_sq = pd.Series(0.0, index=df.index)
    valid_cols = 0

    for col, val in params.items():
        if col in df.columns:
            cfg = config[col]
            step = cfg['step'] if cfg['step'] > 0 else 1.0
            diff = (df[col] - val) / step
            total_dist_sq += diff ** 2
            valid_cols += 1

    if valid_cols == 0:
         return {'Result': 0, 'Profit': 0, 'found': False}

    best_idx = total_dist_sq.idxmin()
    row = df.loc[best_idx]

    matched_params = {k: row[k] for k in params.keys() if k in row}
    return {'Result': float(row['Result']), 'Profit': float(row['Profit']), 'found': True, 'matched_params': matched_params}

# --- Control Group Model ---

class ControlGroupForecaster:
    def __init__(self, global_config, var_cols):
        self.config = global_config
        self.var_cols = var_cols

    def predict(self, best_vectors_history):
        relevant_history = best_vectors_history[-VECTOR_INPUT:]
        if not relevant_history:
            return None

        valid_history = [bv for bv in relevant_history if bv['Result'] != 0 or bv['params']]
        if not valid_history:
            return None

        sums = [0.0] * len(self.var_cols)
        count = len(valid_history)

        for bv in valid_history:
            for idx, col in enumerate(self.var_cols):
                val = bv['params'].get(col, self.config[col]['min'])
                if isinstance(val, str):
                    try: val = float(val.replace(',', '.'))
                    except: val = 0.0

                cfg = self.config[col]
                min_val = cfg['min']
                if cfg['step'] > 0:
                    step_val = (val - min_val) / cfg['step']
                else:
                    step_val = 0
                sums[idx] += step_val

        pred_params = {}
        for idx, col in enumerate(self.var_cols):
            cfg = self.config[col]
            avg_step = sums[idx] / count
            step_pred = int(round(avg_step))
            val_pred = cfg['min'] + step_pred * cfg['step']
            pred_params[col] = val_pred

        return pred_params

# --- Worker Function ---

def worker_predict_week(task_args):
    start_time = time.time()
    try:
        file_name = task_args['file_name']
        history_best = task_args['history_best']
        global_param_config = task_args['config']
        var_cols = task_args['var_cols']

        results = {'control': None}

        # Control Group Prediction
        try:
            control_model = ControlGroupForecaster(global_param_config, var_cols)
            results['control'] = control_model.predict(history_best)
        except Exception:
            pass

        duration = time.time() - start_time
        return {'file_name': file_name, 'results': results, 'duration': duration}

    except Exception as e:
        return {'file_name': task_args.get('file_name', 'unknown'), 'results': {}, 'error': str(e), 'duration': time.time() - start_time}

# --- Metrics Calculation ---

def calculate_financial_metrics(profits):
    if not profits:
        return {}

    profits_np = np.array(profits)

    # 1. Total Profit
    total_profit = np.sum(profits_np)

    # 2. Drawdown
    # Assume starting equity is 0, we track PnL curve
    equity_curve = np.cumsum(profits_np)
    # To properly calculate drawdown, we need running max of equity curve
    # If equity is always negative, peak is 0 (initial).
    # Let's insert 0 at the start to handle initial state
    equity_with_start = np.insert(equity_curve, 0, 0)
    running_max = np.maximum.accumulate(equity_with_start)
    drawdowns = running_max - equity_with_start

    max_drawdown = np.max(drawdowns)
    avg_drawdown = np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0.0

    # 3. Sharpe Ratio
    # Weekly data assumed
    # Risk free rate = 0
    mean_return = np.mean(profits_np)
    std_return = np.std(profits_np)

    if std_return > 1e-9:
        sharpe = (mean_return / std_return) * np.sqrt(52)
    else:
        sharpe = 0.0

    # 4. Expected Returns
    yearly_return = mean_return * 52
    monthly_return = mean_return * (52.0 / 12.0)

    return {
        'Total Profit': total_profit,
        'Max Drawdown': max_drawdown,
        'Average Drawdown': avg_drawdown,
        'Sharpe Ratio': sharpe,
        'Yearly Expected Return': yearly_return,
        'Monthly Expected Return': monthly_return
    }

# --- Report Generation ---

def generate_html_report(results, output_dir):
    print("Generating Report...")

    data = results.get('control', [])
    metrics = {}

    if data:
        profits = [d['stats']['Profit'] for d in data]
        metrics = calculate_financial_metrics(profits)

        labels = [d['file'] for d in data]
        actual_results = [d['stats']['Result'] for d in data]
        found_flags = [d['stats']['found'] for d in data]

        # Plot 1: Result
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        x = range(len(labels))
        ax1.plot(x, actual_results, marker='o', color='blue', label='Actual Result')

        avg_res = np.mean(actual_results) if actual_results else 0
        ax1.axhline(y=avg_res, color='r', linestyle='--', label=f'Avg: {avg_res:.2f}')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Result')
        ax1.set_title('Control Group: Result Performance')
        ax1.legend()
        ax1.grid(True)

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        img1_b64 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        # Plot 2: Profit
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(x, profits, marker='o', color='green', label='Actual Profit')

        avg_prof = np.mean(profits) if profits else 0
        ax2.axhline(y=avg_prof, color='orange', linestyle='--', label=f'Avg: {avg_prof:.2f}')

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Profit')
        ax2.set_title('Control Group: Profit Performance')
        ax2.legend()
        ax2.grid(True)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        img2_b64 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        hit_rate = np.mean(found_flags) * 100 if found_flags else 0
    else:
        img1_b64 = ""
        img2_b64 = ""
        avg_res = 0
        hit_rate = 0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Predictability Report - Baseline</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 50px; border: 1px solid #ccc; padding: 20px; border-radius: 5px; }}
            h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f8f8f8; }}
            .img-container {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
            .metric {{ font-size: 1.1em; font-weight: bold; margin: 10px 0; }}
            .kpi-box {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .kpi-row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .kpi-item {{ flex: 1; min-width: 200px; }}
        </style>
    </head>
    <body>
        <h1>Strategy Predictability Report (Baseline)</h1>

        <div class="section">
            <h2>Performance Metrics (Control Group)</h2>
            <div class="kpi-box">
                <div class="kpi-row">
                    <div class="kpi-item"><strong>Total Profit:</strong> {metrics.get('Total Profit', 0):.2f}</div>
                    <div class="kpi-item"><strong>Sharpe Ratio:</strong> {metrics.get('Sharpe Ratio', 0):.4f}</div>
                    <div class="kpi-item"><strong>Max Drawdown:</strong> {metrics.get('Max Drawdown', 0):.2f}</div>
                    <div class="kpi-item"><strong>Avg Drawdown:</strong> {metrics.get('Average Drawdown', 0):.2f}</div>
                </div>
                <div class="kpi-row" style="margin-top:10px;">
                    <div class="kpi-item"><strong>Yearly Exp. Return:</strong> {metrics.get('Yearly Expected Return', 0):.2f}</div>
                    <div class="kpi-item"><strong>Monthly Exp. Return:</strong> {metrics.get('Monthly Expected Return', 0):.2f}</div>
                    <div class="kpi-item"><strong>Hit Rate (Params Found):</strong> {hit_rate:.1f}%</div>
                    <div class="kpi-item"><strong>Average Result:</strong> {avg_res:.2f}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Visualizations</h2>
            <div class="img-container">
                <h3>Result Graph</h3>
                <img src="data:image/png;base64,{img1_b64}" />
            </div>
            <div class="img-container">
                <h3>Profit Graph</h3>
                <img src="data:image/png;base64,{img2_b64}" />
            </div>
        </div>

        <div class="section">
            <h2>Detailed Predictions</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Found?</th>
                        <th>Result</th>
                        <th>Profit</th>
                        <th>Params</th>
                    </tr>
                </thead>
                <tbody>
    """

    for d in data:
        params_str = ", ".join([f"{k}={v:.2f}" for k,v in d['pred'].items()])
        color = "green" if d['stats']['found'] else "red"
        html += f"""
                <tr>
                    <td>{d['file']}</td>
                    <td style="color:{color}">{d['stats']['found']}</td>
                    <td>{d['stats']['Result']:.2f}</td>
                    <td>{d['stats']['Profit']:.2f}</td>
                    <td>{params_str}</td>
                </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    out_path = os.path.join(output_dir, "Predictability_Report_Baseline.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to {out_path}")

# --- Main ---

def main():
    print("--- Strategy Predictability Program (Baseline) ---")

    # Target Directory (Defaulting to Dummy Data for testing, user input in production)
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # Default for testing
        target_dir = "Data_Files_Dummy"
        if not os.path.exists(target_dir):
             target_dir = input("Path to CSV folder: ").strip()

    if not target_dir or not os.path.exists(target_dir):
        print(f"Invalid directory: {target_dir}")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    # Sorting
    def sort_key(f):
        base = os.path.basename(f)
        try:
            parts = base.split('.')
            # Robust date finding: Look for 8-digit string starting with 20
            start_date_str = ""
            for p in parts:
                if len(p) == 8 and p.startswith("20") and p.isdigit():
                    start_date_str = p
                    break

            if start_date_str:
                return (int(start_date_str), base)
            else:
                # Fallback to original logic if robust fails (though unlikely if format matches)
                start_date_str = parts[-3]
                return (int(start_date_str), base)
        except:
            return (float('inf'), base)

    csv_files.sort(key=sort_key)
    print(f"Found {len(csv_files)} files.")

    # Load Files
    print("Loading files...")
    files_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(process_file_load, csv_files)
        for res in results:
            if res: files_data.append(res)

    if not files_data:
        print("No valid data loaded.")
        return

    # Scan Global Config
    all_values = {}
    for fd in files_data:
        for col, vals in fd['unique_vals'].items():
            if col not in all_values: all_values[col] = set()
            all_values[col].update(vals)

    global_param_config = {}
    sorted_vars = sorted(list(all_values.keys()))
    for col in sorted_vars:
        vals = sorted(list(all_values[col]))
        step = 0
        if len(vals) >= 2:
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            step = np.min(diffs) if len(diffs) > 0 else 0
        global_param_config[col] = {'min': vals[0], 'step': step, 'max': vals[-1]}

    var_cols = sorted_vars

    # Preprocess (Extract Best Vectors)
    print("Preprocessing...")
    def _pre(fd): return preprocess_file_data(fd, global_param_config)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pre_results = list(executor.map(_pre, files_data))

    for i, res in enumerate(pre_results):
        files_data[i].update(res)

    best_vectors_history = [fd['best_vector'] for fd in files_data]
    results_store = {'control': []}

    # Start Prediction Loop
    start_index = VECTOR_INPUT + 1
    if start_index >= len(files_data):
        print(f"Not enough files. Need at least {start_index + 1}.")
        return

    tasks = []
    for i in range(start_index, len(files_data)):
        target_file_data = files_data[i]
        file_name = os.path.basename(target_file_data['path'])

        window_start = max(0, i - TRAINING_WINDOW) # Not strictly needed for Control but kept for structure
        # Control only needs past VECTOR_INPUT vectors
        history_slice = best_vectors_history[:i]

        task = {
            'file_name': file_name,
            'history_best': history_slice, # Passing full history up to i, model slices last VECTOR_INPUT
            'config': global_param_config,
            'var_cols': var_cols
        }
        tasks.append((i, task))

    print(f"Processing {len(tasks)} files...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(worker_predict_week, t[1]): t[0] for t in tasks}

        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            target_file_data = files_data[i]

            try:
                res = future.result()
                preds_map = res['results']
                file_name = res['file_name']

                if preds_map.get('control'):
                    stats = lookup_stats(target_file_data['df'], preds_map['control'], global_param_config)
                    if stats['found'] and 'matched_params' in stats:
                        preds_map['control'] = stats['matched_params']
                    results_store['control'].append({'file': file_name, 'pred': preds_map['control'], 'stats': stats})

            except Exception as e:
                print(f"Error: {e}")

    # Sort results by file order
    # results_store['control'] is likely out of order due to as_completed
    # We rely on file names or just resorting list if needed, but let's just let it match input order if possible.
    # Actually, as_completed yields out of order. We should re-sort based on file index or name.
    # Quick fix: Sort by file name using the sort key logic
    results_store['control'].sort(key=lambda x: sort_key(x['file']))

    generate_html_report(results_store, target_dir)

if __name__ == "__main__":
    main()
