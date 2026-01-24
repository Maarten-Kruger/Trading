
import pandas as pd
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import io
import base64
import warnings
import logging
from itertools import product
from datetime import datetime, timedelta

# Suppress Warnings and Logging
warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

# Configuration
RESULT_CUTOFF = 25
VECTOR_INPUT = 10  # Lookback window size

# --- Library Imports ---
print("Importing libraries...")
try:
    from darts import TimeSeries
    from darts.models import RandomForest
    HAS_DARTS = True
except ImportError:
    HAS_DARTS = False
    print("Warning: Darts not found.")

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, LSTM
    HAS_NF = True
except ImportError:
    HAS_NF = False
    print("Warning: NeuralForecast not found.")

try:
    from tsai.all import *
    HAS_TSAI = True
except ImportError as e:
    HAS_TSAI = False
    print(f"Warning: tsai not found. Error: {e}")

# --- Helper Functions ---

def read_csv_robust(filepath):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
    except:
        df = pd.read_csv(filepath, sep=';', engine='python')

    # Clean numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                series = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(series)
            except ValueError:
                pass
    return df

def get_coords(row, var_cols, config):
    coord = []
    for col in var_cols:
        val = row[col]
        cfg = config[col]
        if cfg['step'] > 0:
            c_val = int(round((val - cfg['min']) / cfg['step']))
        else:
            c_val = 0
        coord.append(c_val)
    return tuple(coord)

def coords_to_params(coord, var_cols, config):
    params = {}
    for idx, col in enumerate(var_cols):
        cfg = config[col]
        step_val = coord[idx]
        val = cfg['min'] + step_val * cfg['step']
        params[col] = val
    return params

def lookup_stats(df, params, config):
    # Find row in df that matches params (using approximate matching for floats)
    # We reconstruct coords to match exactly what we grouped by

    # Actually, simpler: check dist
    best_dist = float('inf')
    best_row = None

    # This might be slow for large DF.
    # Optimization: Filter by first param?
    # Or just iterate.

    # Let's try to match exactly using the grid logic
    # But df contains raw values.

    # Filter using tolerance
    mask = np.ones(len(df), dtype=bool)
    for col, val in params.items():
        if col in df.columns:
            # Tolerance: step / 2 ?
            step = config[col]['step']
            tol = step / 2.0 if step > 0 else 1e-7
            mask = mask & (np.abs(df[col] - val) <= tol)

    matches = df[mask]
    if matches.empty:
        return {'Result': 0, 'Profit': 0, 'found': False}

    # If multiple matches (shouldn't happen on grid), take max result
    row = matches.loc[matches['Result'].idxmax()]
    return {'Result': row['Result'], 'Profit': row['Profit'], 'found': True}

# --- Model Classes ---

class DartsForecaster:
    def __init__(self, global_config, var_cols):
        self.config = global_config
        self.var_cols = var_cols
        self.model = RandomForest(lags=VECTOR_INPUT, n_estimators=50, random_state=42)

    def predict(self, best_vectors_history):
        # best_vectors_history is list of dicts: {'params': {...}, ...}
        if not HAS_DARTS:
            return None

        data_matrix = []
        for bv in best_vectors_history:
            row_vec = []
            for col in self.var_cols:
                val = bv['params'].get(col, self.config[col]['min'])
                cfg = self.config[col]
                if cfg['step'] > 0:
                    step_val = int(round((val - cfg['min']) / cfg['step']))
                else:
                    step_val = 0
                row_vec.append(step_val)
            data_matrix.append(row_vec)

        data_np = np.array(data_matrix)
        ts = TimeSeries.from_values(data_np)

        self.model.fit(ts)
        pred = self.model.predict(n=1)
        pred_vals = pred.values()[0]

        pred_params = {}
        for idx, col in enumerate(self.var_cols):
            cfg = self.config[col]
            step_pred = int(round(pred_vals[idx]))
            val_pred = cfg['min'] + step_pred * cfg['step']
            pred_params[col] = val_pred

        return pred_params

class NeuralForecastForecaster:
    def __init__(self, global_config, var_cols):
        self.config = global_config
        self.var_cols = var_cols
        # Use NHITS - fast and effective
        self.model = NHITS(h=1, input_size=VECTOR_INPUT, max_steps=100, enable_checkpointing=False, logger=False)
        self.nf = None

    def predict(self, all_vectors_history):
        # all_vectors_history: list of DataFrames (one per file time step)
        if not HAS_NF:
            return None

        # 1. Build Long Format DF
        # unique_id | ds | y
        records = []

        # We need to normalize time. Let's use dummy dates.
        start_date = datetime(2020, 1, 1)

        # Identify all unique coords seen in history
        # To avoid explosion, maybe limit to top N coords per file?
        # No, "entire set". We trust the grid is reasonable.

        # Optimization: Use a dictionary for aggregation first
        # key: coord, value: {time_idx: result}
        history_map = {}

        for t, df in enumerate(all_vectors_history):
            # Group by coordinates to handle duplicates in one file (if any)
            # Efficiently map to coords

            # Vectorized coord calculation?
            # Creating a hashable key for each row
            temp_df = df.copy()

            # We assume active_vars are in df
            # Map values to grid indices
            keys = []
            for col in self.var_cols:
                cfg = self.config[col]
                if cfg['step'] > 0:
                    temp_df[col + '_idx'] = ((temp_df[col] - cfg['min']) / cfg['step']).round().astype(int)
                else:
                    temp_df[col + '_idx'] = 0

            idx_cols = [c + '_idx' for c in self.var_cols]

            # Group by indices and take max Result
            grouped = temp_df.groupby(idx_cols)['Result'].max().reset_index()

            for _, row in grouped.iterrows():
                coord = tuple(row[idx_cols].astype(int).values)
                res = row['Result']
                if coord not in history_map:
                    history_map[coord] = {}
                history_map[coord][t] = res

        # Convert to records
        full_time_indices = range(len(all_vectors_history))

        long_data = []
        unique_ids = []

        for coord, timeline in history_map.items():
            uid = str(coord)
            unique_ids.append((uid, coord))
            for t in full_time_indices:
                val = timeline.get(t, 0.0) # Fill missing with 0
                ds = start_date + timedelta(days=t)
                long_data.append({'unique_id': uid, 'ds': ds, 'y': val})

        Y_df = pd.DataFrame(long_data)

        if Y_df.empty:
            return None

        # 2. Train and Predict
        # We instantiate NF every time to avoid carrying over state from previous growing windows incorrectly?
        # NF is designed to be fitted.
        nf = NeuralForecast(models=[self.model], freq='D')
        nf.fit(df=Y_df)

        future_df = nf.predict()

        # 3. Find Best
        if future_df.empty:
            return None

        # future_df has columns [ds, NHITS]
        # We want the row with max NHITS value
        best_row = future_df.loc[future_df['NHITS'].idxmax()]
        best_uid = best_row.name # index is unique_id usually, or column?
        # Check index
        if 'unique_id' in future_df.columns:
            best_uid = future_df.loc[future_df['NHITS'].idxmax()]['unique_id']
        else:
            best_uid = future_df['NHITS'].idxmax()

        # Find coord back
        best_coord = None
        for uid, coord in unique_ids:
            if uid == best_uid:
                best_coord = coord
                break

        if best_coord:
            return coords_to_params(best_coord, self.var_cols, self.config)
        return None

class TsaiForecaster:
    def __init__(self, global_config, var_cols):
        self.config = global_config
        self.var_cols = var_cols

    def predict(self, all_vectors_history):
        if not HAS_TSAI:
            return None

        # Prepare X: (Samples, Features, Time)
        # Samples: Unique Coords
        # Features: 1 (Result)
        # Time: History Length

        history_map = {}
        for t, df in enumerate(all_vectors_history):
            temp_df = df.copy()
            idx_cols = []
            for col in self.var_cols:
                cfg = self.config[col]
                if cfg['step'] > 0:
                    temp_df[col + '_idx'] = ((temp_df[col] - cfg['min']) / cfg['step']).round().astype(int)
                else:
                    temp_df[col + '_idx'] = 0
                idx_cols.append(col + '_idx')

            grouped = temp_df.groupby(idx_cols)['Result'].max().reset_index()
            for _, row in grouped.iterrows():
                coord = tuple(row[idx_cols].astype(int).values)
                if coord not in history_map:
                    history_map[coord] = {}
                history_map[coord][t] = row['Result']

        coords_list = list(history_map.keys())
        time_steps = len(all_vectors_history)

        if not coords_list:
            return None

        X = np.zeros((len(coords_list), 1, time_steps))

        for i, coord in enumerate(coords_list):
            for t in range(time_steps):
                X[i, 0, t] = history_map[coord].get(t, 0.0)

        # We need to forecast step T+1.
        # Simple approach: Train a regressor on sliding windows of this data?
        # Too slow to train a model per sample.
        # Train a global model: Input(Window) -> Output(Next Step)
        # Prepare training data from X

        # Sliding Window
        # X shape: (Samples, 1, Time)
        # We want to use last VECTOR_INPUT steps to predict next.
        # But we only have 'Time' steps available.

        # If Time < VECTOR_INPUT, we can't do much.
        # Assuming we have enough history.

        # We will use a simple tsai Regressor (e.g., TST or InceptionTime)
        # Data preparation:
        # X_train: [Samples, 1, Window_Size]
        # y_train: [Samples] (Next step value)

        # We extract all possible windows from the history
        w = min(VECTOR_INPUT, time_steps - 1)
        if w < 2:
            return None

        X_train = []
        y_train = []

        # Use all coords as samples
        for i in range(len(coords_list)):
            series = X[i, 0, :]
            # Generate windows
            for t in range(w, len(series)):
                X_train.append(series[t-w:t])
                y_train.append(series[t])

        X_train = np.array(X_train)[:, np.newaxis, :] # (TotalSamples, 1, w)
        y_train = np.array(y_train)

        if len(X_train) == 0:
            return None

        # Model
        # InceptionTime is good
        model = InceptionTime(c_in=1, c_out=1)
        # Use simple Learner
        # We need to wrap in Datasets/DataLoaders
        splits = RandomSplitter()(range(len(X_train)))
        tfms = [None, [TSRegression()]]
        dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, num_workers=0)

        learn = ts_learner(dls, model, metrics=mae, verbose=False)
        learn.fit_one_cycle(5, 1e-3) # Fast training

        # Predict on latest window
        X_test = X[:, :, -w:] # (Samples, 1, w)
        # get_preds expects dl
        test_ds = dsets.valid.new_empty()
        # Hack to create test dl
        # Construct directly
        # tsai is a bit complex with inference without dl
        # Let's create a dl
        test_dls = dls.test_dl(X_test)
        preds, _ = learn.get_preds(dl=test_dls)

        # preds is (Samples, 1)
        preds_np = preds.numpy().flatten()

        best_idx = np.argmax(preds_np)
        best_coord = coords_list[best_idx]

        return coords_to_params(best_coord, self.var_cols, self.config)


# --- Main Execution ---

def main():
    print("--- Strategy Predictability Program (Multi-Model) ---")

    # User Input
    if not sys.stdin.isatty():
        try:
            target_dir = sys.stdin.readline().strip()
        except:
            target_dir = ""
    else:
        target_dir = input("Path to CSV folder: ").strip()

    if not target_dir or not os.path.exists(target_dir):
        print(f"Invalid directory: {target_dir}")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    def sort_key(f):
        base = os.path.basename(f)
        try:
            return (int(base.split('_')[0]), base)
        except:
            return (float('inf'), base)

    csv_files.sort(key=sort_key)
    print(f"Found {len(csv_files)} files.")

    # Global Scan
    print("Scanning global parameters...")
    all_values = {}
    files_data = [] # List of DataFrames

    for f in csv_files:
        try:
            df = read_csv_robust(f)
            if 'Trades' not in df.columns or 'Result' not in df.columns:
                continue

            # Identify variable columns (after Trades)
            trades_idx = df.columns.get_loc("Trades")
            var_cols = df.columns[trades_idx+1:].tolist()

            # Store
            files_data.append({
                'path': f,
                'df': df,
                'var_cols': var_cols
            })

            for col in var_cols:
                if col not in all_values: all_values[col] = set()
                all_values[col].update(df[col].dropna().unique())

        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not files_data:
        print("No valid data.")
        return

    # Config
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

    # Pre-calculate Best Vectors for Darts
    print("Extracting Best Vectors...")
    best_vectors_history = []
    for fdata in files_data:
        df = fdata['df']
        # Simple Max
        if not df.empty:
            best_idx = df['Result'].idxmax()
            best_row = df.loc[best_idx]
            best_vectors_history.append({
                'params': best_row[var_cols].to_dict(),
                'Result': best_row['Result']
            })
        else:
            best_vectors_history.append({'params': {}, 'Result': 0})

    # Prepare Models
    darts_model = DartsForecaster(global_param_config, var_cols)
    nf_model = NeuralForecastForecaster(global_param_config, var_cols)
    tsai_model = TsaiForecaster(global_param_config, var_cols)

    # Predictions Storage
    # Structure: {'darts': [], 'nf': [], 'tsai': []}
    results = {'darts': [], 'nf': [], 'tsai': []}

    start_index = VECTOR_INPUT + 1
    if start_index >= len(files_data):
        print(f"Not enough files. Need > {VECTOR_INPUT + 1}")
        return

    print("Running Forecasts (This may take time)...")

    for i in range(start_index, len(files_data)):
        target_file_data = files_data[i]
        file_name = os.path.basename(target_file_data['path'])
        print(f"Processing {file_name}...")

        # Data Slices
        history_best = best_vectors_history[:i]
        history_all = [fd['df'] for fd in files_data[:i]]

        # 1. Darts
        if HAS_DARTS:
            try:
                pred = darts_model.predict(history_best)
                stats = lookup_stats(target_file_data['df'], pred, global_param_config)
                results['darts'].append({
                    'file': file_name,
                    'pred': pred,
                    'stats': stats
                })
            except Exception as e:
                print(f"  Darts Error: {e}")

        # 2. NeuralForecast
        if HAS_NF:
            try:
                # We should suppress stdout from NF
                with io.capture_output() if 'io.capture_output' in globals() else open(os.devnull, 'w') as devnull: # Simple redirection
                     # Redirect stdout/stderr?
                     # Python logging already handled.
                    pred = nf_model.predict(history_all)

                if pred:
                    stats = lookup_stats(target_file_data['df'], pred, global_param_config)
                    results['nf'].append({
                        'file': file_name,
                        'pred': pred,
                        'stats': stats
                    })
                else:
                     results['nf'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'found': False}})
            except Exception as e:
                print(f"  NF Error: {e}")
                results['nf'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'found': False}})

        # 3. Tsai
        if HAS_TSAI:
            try:
                 pred = tsai_model.predict(history_all)
                 if pred:
                    stats = lookup_stats(target_file_data['df'], pred, global_param_config)
                    results['tsai'].append({
                        'file': file_name,
                        'pred': pred,
                        'stats': stats
                    })
                 else:
                    results['tsai'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'found': False}})
            except Exception as e:
                print(f"  Tsai Error: {e}")
                results['tsai'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'found': False}})

    # Generate Report
    generate_html_report(results, target_dir)

def generate_html_report(results, output_dir):
    print("Generating Report...")

    html_parts = []

    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Predictability Report - Multi-Model</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 50px; border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
            h2 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            table { border-collapse: collapse; width: 100%; margin-top: 15px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f8f8f8; }
            .img-container { text-align: center; margin: 20px 0; }
            img { max-width: 100%; height: auto; border: 1px solid #eee; }
            .metric { font-size: 1.1em; font-weight: bold; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Strategy Predictability Report</h1>
        <p>Comparison of Darts (Vector Trajectory), NeuralForecast (Panel Surface), and Tsai (Panel Surface).</p>
    """)

    models = [('Darts', 'darts'), ('NeuralForecast', 'nf'), ('Tsai', 'tsai')]

    for title, key in models:
        data = results.get(key, [])
        if not data:
            continue

        labels = [d['file'] for d in data]
        actual_results = [d['stats']['Result'] for d in data]
        profits = [d['stats']['Profit'] for d in data] # Actually predicted result stats
        found_flags = [d['stats']['found'] for d in data]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(labels))
        ax.plot(x, actual_results, marker='o', label='Actual Result of Predicted Params')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Result')
        ax.set_title(f'{title} Performance')
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # Calculate Avg
        avg_res = np.mean(actual_results) if actual_results else 0
        hit_rate = np.mean(found_flags) * 100 if found_flags else 0

        html_parts.append(f"""
        <div class="section">
            <h2>{title} Model</h2>
            <div class="metric">Average Result: {avg_res:.2f} | Hit Rate (Params Found): {hit_rate:.1f}%</div>
            <div class="img-container">
                <img src="data:image/png;base64,{img_b64}" />
            </div>
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
        """)

        for d in data:
            params_str = ", ".join([f"{k}={v:.2f}" for k,v in d['pred'].items()])
            color = "green" if d['stats']['found'] else "red"
            html_parts.append(f"""
                <tr>
                    <td>{d['file']}</td>
                    <td style="color:{color}">{d['stats']['found']}</td>
                    <td>{d['stats']['Result']:.2f}</td>
                    <td>{d['stats']['Profit']:.2f}</td>
                    <td>{params_str}</td>
                </tr>
            """)

        html_parts.append("""
                </tbody>
            </table>
        </div>
        """)

    html_parts.append("""
    </body>
    </html>
    """)

    out_path = os.path.join(output_dir, "Predictability_Report_MultiModel.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("".join(html_parts))

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
