
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
import time
import traceback
import concurrent.futures
import multiprocessing
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
from datetime import datetime, timedelta
from collections import deque

# Suppress Warnings and Logging
warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# Configuration
RESULT_CUTOFF = 25
VECTOR_INPUT = 10  # Lookback window size (Model Lags)
TRAINING_WINDOW = 30 # Size of the sliding window used for training (Must be > VECTOR_INPUT)
MAX_WORKERS = 4      # Max parallel processes (Adjust based on VRAM)

# --- Library Availability Flags (Lazy Check) ---
import importlib.util

def check_lib(name):
    return importlib.util.find_spec(name) is not None

HAS_DARTS = check_lib("darts")

# --- Helper Functions ---

def geometric_median(X, eps=1e-5):
    """
    Compute the geometric median of point cloud X using Weiszfeld's algorithm.
    X: (N, D) numpy array
    """
    if len(X) == 0:
        return None
    if len(X) == 1:
        return X[0]

    # Initial estimate: Centroid
    y = np.mean(X, axis=0)

    for _ in range(100): # Max iterations
        D = np.linalg.norm(X - y, axis=1)
        non_zeros = (D > eps)

        if not np.any(non_zeros):
            return y

        Dinv = 1 / D[non_zeros]
        W = Dinv / np.sum(Dinv)

        # Weighted sum
        T = np.sum(W[:, np.newaxis] * X[non_zeros], axis=0)

        # Weiszfeld update
        num_zeros = len(X) - np.sum(non_zeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * np.sum(Dinv)
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if np.linalg.norm(y - y1) < eps:
            return y1

        y = y1

    return y

def read_csv_robust(filepath):
    """
    Optimized CSV reader that sniffs separator and decimal format.
    Uses C engine for speed.
    """
    try:
        # Fast sniffing of the first few lines
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            if not header: return pd.DataFrame() # Empty file

        sep = ';' if ';' in header else ','
        # Assume decimal is ',' if sep is ';' (European), else '.'
        decimal = ',' if sep == ';' else '.'

        # Fast load using C engine
        df = pd.read_csv(filepath, sep=sep, decimal=decimal, engine='c')
    except Exception as e:
        # Fallback to python engine if C engine fails (e.g. bad lines or complex structure)
        try:
             df = pd.read_csv(filepath, sep=None, engine='python')
        except:
             return pd.DataFrame()

    # Post-load cleanup
    # Ensure Result and Profit are numeric
    cols_to_numeric = ['Result', 'Profit', 'Trades']
    for c in cols_to_numeric:
        if c in df.columns and df[c].dtype == 'object':
             df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    # Attempt to convert other object columns that might be numeric (parameters)
    # Only iterate object columns to save time
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        try:
            # fast conversion
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='ignore')
        except:
            pass

    return df

def process_file_load(filepath):
    """
    Helper function for parallel file loading.
    Returns dictionary with df and stats to avoid repeated work.
    """
    try:
        df = read_csv_robust(filepath)
        if df.empty or 'Trades' not in df.columns or 'Result' not in df.columns:
            return None

        # Identify var cols (after Trades)
        if 'Trades' in df.columns:
            trades_idx = df.columns.get_loc("Trades")
            var_cols = df.columns[trades_idx+1:].tolist()
        else:
            var_cols = []

        # Return necessary data
        # We assume var_cols are consistent across files, but we return them for the global scan
        return {
            'path': filepath,
            'df': df,
            'var_cols': var_cols,
            'unique_vals': {col: df[col].dropna().unique() for col in var_cols}
        }
    except Exception as e:
        # print(f"Error loading {filepath}: {e}")
        return None

def preprocess_file_data(fdata, config):
    """
    Extracts optimization surface and best vector for a file.
    """
    df = fdata['df']
    var_cols = fdata['var_cols']

    # 1. Best Vector
    best_vector = {'params': {}, 'Result': 0}
    if not df.empty and 'Result' in df.columns:
        best_idx = df['Result'].idxmax()
        best_row = df.loc[best_idx]
        best_vector = {
            'params': best_row[var_cols].to_dict(),
            'Result': best_row['Result']
        }

    # 2. Grid Summary (Coord -> Result)
    grid_summary = {}
    if not df.empty and 'Result' in df.columns:
        # Optimized grouping
        temp_df = df[var_cols + ['Result']].copy()

        # Convert to steps
        for col in var_cols:
            cfg = config[col]
            if cfg['step'] > 0:
                vals = temp_df[col].values
                # Vectorized operation
                temp_df[col] = np.round((vals - cfg['min']) / cfg['step']).astype(int)
            else:
                temp_df[col] = 0

        # Group by all parameter columns
        grouped = temp_df.groupby(var_cols)['Result'].max()

        # Convert to dict where key is tuple of coords
        if len(var_cols) == 1:
            grid_summary = { (k,): v for k, v in grouped.to_dict().items() }
        else:
            grid_summary = grouped.to_dict()

    return {
        'best_vector': best_vector,
        'grid_summary': grid_summary
    }

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
    # 0. Safety Check
    if df.empty:
        return {'Result': 0, 'Profit': 0, 'found': False}

    # 1. Try Exact/Close Match (within tolerance)
    mask = np.ones(len(df), dtype=bool)
    for col, val in params.items():
        if col in df.columns:
            step = config[col]['step']
            tol = step / 2.0 if step > 0 else 1e-7
            mask = mask & (np.abs(df[col] - val) <= tol)

    matches = df[mask]
    if not matches.empty:
        # If multiple matches, take max result
        row = matches.loc[matches['Result'].idxmax()]
        matched_params = {k: row[k] for k in params.keys() if k in row}
        return {'Result': float(row['Result']), 'Profit': float(row['Profit']), 'found': True, 'matched_params': matched_params}

    # 2. Fallback: Nearest Neighbor in Normalized Step Space
    total_dist_sq = pd.Series(0.0, index=df.index)
    valid_cols = 0

    for col, val in params.items():
        if col in df.columns:
            cfg = config[col]
            step = cfg['step'] if cfg['step'] > 0 else 1.0
            # Normalized difference
            diff = (df[col] - val) / step
            total_dist_sq += diff ** 2
            valid_cols += 1

    if valid_cols == 0:
         return {'Result': 0, 'Profit': 0, 'found': False}

    best_idx = total_dist_sq.idxmin()
    row = df.loc[best_idx]

    matched_params = {k: row[k] for k in params.keys() if k in row}
    return {'Result': float(row['Result']), 'Profit': float(row['Profit']), 'found': True, 'matched_params': matched_params}

# --- Model Classes (Lazy Loading Wrapper) ---

def get_forecasters(flags):
    """
    Returns forecaster classes, importing libraries only when called.
    Used inside worker process.
    """

    classes = {}

    if flags['HAS_DARTS']:
        try:
            from darts import TimeSeries
            from darts.models import RandomForest

            class DartsForecaster:
                def __init__(self, global_config, var_cols):
                    self.config = global_config
                    self.var_cols = var_cols
                    # Optimize: n_jobs=-1 to use all cores
                    self.model = RandomForest(lags=VECTOR_INPUT, n_estimators=50, random_state=42, n_jobs=-1)

                def predict(self, slice_matrix, coords_list):
                    # slice_matrix: (N_Coords, Window_Size)
                    # coords_list: List of coordinate tuples corresponding to rows

                    # 1. Filter: Identify rows with at least one value > RESULT_CUTOFF
                    # Ignore -1000.0 (missing)
                    valid_mask = np.any((slice_matrix > RESULT_CUTOFF), axis=1)

                    if not np.any(valid_mask):
                        # Fallback: Take top 100 vectors by Max Result
                        max_vals = np.max(slice_matrix, axis=1)
                        # Filter out rows that are purely -1000
                        real_max_mask = max_vals > -999
                        if not np.any(real_max_mask):
                             return None # No valid data at all

                        # Indices of valid rows
                        valid_indices_all = np.where(real_max_mask)[0]
                        max_vals_valid = max_vals[valid_indices_all]

                        # Top 100
                        top_n = 100
                        if len(max_vals_valid) > top_n:
                            top_args = np.argsort(max_vals_valid)[-top_n:]
                            valid_indices = valid_indices_all[top_args]
                        else:
                            valid_indices = valid_indices_all
                    else:
                        valid_indices = np.where(valid_mask)[0]

                    # 2. Prepare TimeSeries
                    series_list = []

                    # Create a subset matrix
                    subset_matrix = slice_matrix[valid_indices]

                    # Replace -1000.0 with 0.0 (Neutral/Imputation)
                    # Assuming 0.0 is a reasonable filler for "no result" compared to a very negative number
                    # or forward fill logic could be applied but matrix is simple float
                    subset_matrix[subset_matrix < -900] = 0.0

                    for row in subset_matrix:
                         # Create TimeSeries
                         # Darts needs shape (Time, Components)
                         # Row is (Time,)
                         ts = TimeSeries.from_values(row)
                         series_list.append(ts)

                    if not series_list:
                        return None

                    # 3. Train Global Model
                    # We need at least lags+1 length
                    if subset_matrix.shape[1] < VECTOR_INPUT + 1:
                         return None

                    self.model.fit(series=series_list)

                    # 4. Predict
                    # predict n=1 for all series
                    preds = self.model.predict(n=1, series=series_list)

                    # 5. Select Best Candidates
                    pred_values = []
                    for p in preds:
                        # p is TimeSeries, get last value
                        pred_values.append(p.values()[-1][0])

                    pred_values = np.array(pred_values)

                    # Filter predictions > RESULT_CUTOFF
                    high_profit_mask = pred_values > RESULT_CUTOFF

                    if np.any(high_profit_mask):
                        selected_local_indices = np.where(high_profit_mask)[0]
                    else:
                        # Fallback: Top 10 predicted
                        top_n_pred = min(10, len(pred_values))
                        selected_local_indices = np.argsort(pred_values)[-top_n_pred:]

                    # Map back to global indices
                    selected_global_indices = valid_indices[selected_local_indices]

                    # 6. Consensus: Geometric Median of Parameters
                    # Get parameter coordinates (tuples of steps)
                    selected_coords = [coords_list[i] for i in selected_global_indices]

                    # Convert to numpy array of shape (M, D)
                    X_coords = np.array(selected_coords)

                    # Calculate Geometric Median
                    median_coord = geometric_median(X_coords)

                    # 7. Convert to Params
                    # Round median coords to nearest integer steps
                    median_coord_int = np.round(median_coord).astype(int)

                    pred_params = coords_to_params(median_coord_int, self.var_cols, self.config)

                    return pred_params

            classes['DartsForecaster'] = DartsForecaster
        except Exception as e:
            print(f"Lazy Import Error (Darts): {e}")

    return classes

class ControlGroupForecaster:
    def __init__(self, global_config, var_cols):
        self.config = global_config
        self.var_cols = var_cols

    def predict(self, best_vectors_history):
        # best_vectors_history is list of dicts: {'params': {...}, ...}
        # Use global VECTOR_INPUT

        relevant_history = best_vectors_history[-VECTOR_INPUT:]
        if not relevant_history:
            return None

        # Filter valid
        valid_history = [bv for bv in relevant_history if bv['Result'] != 0 or bv['params']]
        if not valid_history:
            return None

        sums = [0.0] * len(self.var_cols)
        count = len(valid_history)

        for bv in valid_history:
            for idx, col in enumerate(self.var_cols):
                val = bv['params'].get(col, self.config[col]['min'])
                # Ensure val is float
                if isinstance(val, str):
                    try:
                        val = float(val.replace(',', '.'))
                    except:
                        val = 0.0

                cfg = self.config[col]
                min_val = cfg['min']
                if isinstance(min_val, str):
                     try:
                        min_val = float(min_val.replace(',', '.'))
                     except:
                        min_val = 0.0

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


# --- Worker Function for Parallel Processing ---

def worker_predict_week(task_args):
    """
    Worker function to process a single week/file.
    Executed in a separate process.
    """
    start_time = time.time()
    try:
        # Unpack arguments
        file_name = task_args['file_name']

        # Verbose Start
        print(f"  [Worker {os.getpid()}] {file_name}: Processing started...", flush=True)

        history_best = task_args['history_best']
        slice_matrix = task_args['slice_matrix'] # Numpy array

        global_param_config = task_args['config']
        var_cols = task_args['var_cols']
        flags = task_args['flags']

        HAS_DARTS = flags['HAS_DARTS']

        # Lazy load forecaster classes inside the worker
        forecaster_classes = get_forecasters(flags)

        results = {'control': None, 'darts': None}

        # 0. Control Group
        try:
            control_model = ControlGroupForecaster(global_param_config, var_cols)
            results['control'] = control_model.predict(history_best)
        except Exception as e:
            pass

        # 1. Darts
        if HAS_DARTS and 'DartsForecaster' in forecaster_classes:
            try:
                coords_list = task_args.get('coords_list')
                if coords_list is not None and slice_matrix is not None:
                    DartsForecaster = forecaster_classes['DartsForecaster']
                    darts_model = DartsForecaster(global_param_config, var_cols)
                    results['darts'] = darts_model.predict(slice_matrix, coords_list)
            except Exception as e:
                print(f"  [Worker {os.getpid()}] {file_name}: Darts Error: {e}", flush=True)
                traceback.print_exc()

        duration = time.time() - start_time
        return {'file_name': file_name, 'results': results, 'duration': duration}

    except Exception as e:
        return {'file_name': task_args.get('file_name', 'unknown'), 'results': {}, 'error': str(e), 'duration': time.time() - start_time}

# --- Main Execution ---

def main():
    print("--- Strategy Predictability Program (Darts Consensus) ---")

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
            # Format: [version] [Bot name].[currency pair].[Timeframe].[date range start].[date range end].csv
            # Example: 1.0 Simple Crossover.EURUSDm.M30.20240101.20240108.csv
            parts = base.split('.')
            # Start date is the 3rd from the end (before .csv and end_date)
            start_date_str = parts[-3]
            return (int(start_date_str), base)
        except:
            return (float('inf'), base)

    csv_files.sort(key=sort_key)
    print(f"Found {len(csv_files)} files.")

    print("\n=== Definitions ===")
    print(f"VECTOR_INPUT ({VECTOR_INPUT}): Lookback window size (Model Lags).")
    print(f"TRAINING_WINDOW ({TRAINING_WINDOW}): Size of the sliding window used for training.")
    print("-" * 30)
    print("Control Group: Avg of Best Vectors.")
    print("Darts (Global RF Consensus): Global Forecasting on High-Profit Trajectories + Geometric Median Consensus.")
    print("===================\n")

    # Global Scan (Parallel Loading)
    print("Loading files in parallel...")
    start_load = time.time()

    files_data = []

    # Use ProcessPoolExecutor for parallel loading
    with concurrent.futures.ProcessPoolExecutor() as executor:
        try:
            results = executor.map(process_file_load, csv_files)

            for res in results:
                if res is not None:
                    files_data.append(res)
        except Exception as e:
            print(f"Error during parallel loading: {e}")

    load_time = time.time() - start_load
    print(f"Loaded {len(files_data)} files in {load_time:.2f}s")

    if not files_data:
        print("No valid data.")
        return

    # Process stats for global config
    print("Scanning global parameters...")
    all_values = {}
    for fd in files_data:
        for col, vals in fd['unique_vals'].items():
            if col not in all_values: all_values[col] = set()
            all_values[col].update(vals)

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

    # Pre-process Data (Parallel) - Build Per-File Grids
    print("Preprocessing data for models...")
    start_pre = time.time()

    def _pre_wrapper(fd):
        return preprocess_file_data(fd, global_param_config)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        pre_results = list(executor.map(_pre_wrapper, files_data))

    # Merge back
    for i, res in enumerate(pre_results):
        files_data[i].update(res)

    print(f"Preprocessed {len(files_data)} files in {time.time() - start_pre:.2f}s")

    # --- MASTER MATRIX CONSTRUCTION ---
    print("Constructing Master Data Structures (Global Matrix)...")
    start_matrix = time.time()

    # 1. Identify ALL unique coords across history
    all_coords_set = set()
    for fd in files_data:
        all_coords_set.update(fd['grid_summary'].keys())

    coords_list = sorted(list(all_coords_set)) # Consistent ordering
    coord_to_idx = {c: i for i, c in enumerate(coords_list)}
    num_coords = len(coords_list)
    num_files = len(files_data)

    print(f"  > Unique Coordinate Combinations: {num_coords}")
    print(f"  > Time Steps (Files): {num_files}")

    # 2. Build Tsai Master Matrix (NumPy)
    # Shape: (Num_Coords, Num_Files)
    # Init with -1000.0 (a value lower than any valid Result) to ensure missing data is treated as "worst case".
    # This prevents the model from preferring "missing" (0) over "bad" (e.g. -4.33).
    master_matrix = np.full((num_coords, num_files), -1000.0, dtype=np.float32)

    # Fill matrix
    # This might take a moment but it's done ONCE.
    for t, fd in enumerate(files_data):
        grid = fd['grid_summary']
        for coord, res in grid.items():
            if coord in coord_to_idx:
                idx = coord_to_idx[coord]
                master_matrix[idx, t] = res

    print(f"Master Matrix Construction took {time.time() - start_matrix:.2f}s")

    # 4. Extract Best Vectors (Compatibility)
    best_vectors_history = [fd['best_vector'] for fd in files_data]

    # Predictions Storage
    # Structure: {'darts': [], 'nf': [], 'tsai': []}
    results = {'control': [], 'darts': []}

    start_index = TRAINING_WINDOW + 1
    if start_index >= len(files_data):
        # Fallback if we have fewer files than TRAINING_WINDOW but enough to start
        start_index = VECTOR_INPUT + 2

    if start_index >= len(files_data):
        print(f"Not enough files. Need > {start_index}")
        return

    print("Preparing Tasks for Parallel Execution...")

    start_time = time.time()
    recent_completion_times = deque(maxlen=10) # Sliding window for ETA
    files_processed = 0
    total_files_to_process = len(files_data) - start_index

    # Flags for worker
    flags = {'HAS_DARTS': HAS_DARTS}

    # Prepare Tasks List
    tasks = []
    for i in range(start_index, len(files_data)):
        target_file_data = files_data[i]
        file_name = os.path.basename(target_file_data['path'])

        window_start = max(0, i - TRAINING_WINDOW)
        window_end = i

        # Slices
        history_best = best_vectors_history[window_start:window_end]
        slice_matrix = master_matrix[:, window_start:window_end] # (N_Coords, Window)

        task = {
            'file_name': file_name,
            'history_best': history_best,
            'slice_matrix': slice_matrix,
            'config': global_param_config,
            'var_cols': var_cols,
            'coords_list': coords_list, # Passed for static covariances
            'window_start': window_start,
            'window_end': window_end,
            'flags': flags
        }
        tasks.append((i, task))

    print(f"Submitted {len(tasks)} tasks to ProcessPoolExecutor (Spawn, Max Workers={MAX_WORKERS})...")

    try:
        # Use spawn context as requested for independent GPU streams/memory
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(worker_predict_week, t[1]): t[0] for t in tasks}

            print("\n--- Parallel Processing Started ---")

            # Timing variables
            last_batch_time = time.time()
            batch_times = deque(maxlen=10) # Stores duration for each batch of MAX_WORKERS

            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                target_file_data = files_data[i]
                file_name = os.path.basename(target_file_data['path'])

                try:
                    res = future.result()
                    # res has {'file_name', 'results': {'control': ..., ...}, 'error': ...}

                    if 'error' in res:
                        print(f"  [Error] {file_name}: {res['error']}")
                        continue

                    duration = res.get('duration', 0)
                    preds_map = res['results']

                    # --- Verification & Stats (Main Process) ---
                    status_msg = []

                    # Control
                    if preds_map.get('control'):
                        stats = lookup_stats(target_file_data['df'], preds_map['control'], global_param_config)
                        if stats['found'] and 'matched_params' in stats:
                            preds_map['control'] = stats['matched_params']
                        results['control'].append({'file': file_name, 'pred': preds_map['control'], 'stats': stats})
                        status_msg.append(f"C:{'Y' if stats['found'] else 'N'}")

                    # Darts
                    if HAS_DARTS:
                        if preds_map.get('darts'):
                            stats = lookup_stats(target_file_data['df'], preds_map['darts'], global_param_config)
                            if stats['found'] and 'matched_params' in stats:
                                preds_map['darts'] = stats['matched_params']
                            results['darts'].append({'file': file_name, 'pred': preds_map['darts'], 'stats': stats})
                            status_msg.append(f"D:{'Y' if stats['found'] else 'N'}")

                except Exception as e:
                    print(f"Error processing result for {file_name}: {e}")
                    traceback.print_exc()

                # Progress & ETA Calculation
                files_processed += 1
                now = time.time()

                # Check if we should print based on MAX_WORKERS batch or completion
                # We assume MAX_WORKERS files are done when this condition hits (roughly)
                if files_processed % MAX_WORKERS == 0 or files_processed == total_files_to_process:

                    # Calculate time for this batch
                    current_batch_duration = now - last_batch_time
                    last_batch_time = now
                    batch_times.append(current_batch_duration)

                    # Avg time per batch (of MAX_WORKERS files)
                    avg_batch_time = sum(batch_times) / len(batch_times)

                    # ETA = ((files left) * (time per max workers files)) / MAX_WORKERS
                    files_left = total_files_to_process - files_processed
                    eta_seconds = (files_left * avg_batch_time) / MAX_WORKERS

                    total_elapsed = now - start_time
                    print(f"\n\n\n Files Completed: {files_processed}/{total_files_to_process} | Time: {total_elapsed:.0f} sec | ETA: {eta_seconds/60:.2f} min \n\n\n", flush=True)

    except KeyboardInterrupt:
        print("\nProcess cancelled by user. Outputting available results...")

    # Generate Report
    generate_html_report(results, target_dir)

def generate_diagnostics_section(results):
    html = """
    <div class="section">
        <h2>Master Diagnostics</h2>
        <p>Detailed analysis of Ground Truth (Actual File Data) vs Model Internal Predictions.</p>
    """
    html += "</div>"
    return html

def generate_html_report(results, output_dir):
    print("Generating Report...")

    html_parts = []

    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Predictability Report - Darts Consensus</title>
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
    """)

    html_parts.append(f"""
    <body>
        <h1>Strategy Predictability Report (Darts Consensus)</h1>
        <div class="section">
            <h2>Definitions</h2>
            <ul>
                <li><strong>RESULT_CUTOFF ({RESULT_CUTOFF}):</strong> Filter vectors inside each file where the result exceeds a defined threshold.</li>
                <li><strong>VECTOR_INPUT ({VECTOR_INPUT}):</strong> Lookback window size (Model Lags). The number of past time steps (files/vectors) the model looks at to make a prediction.</li>
                <li><strong>TRAINING_WINDOW ({TRAINING_WINDOW}):</strong> Size of the sliding window used for training. The number of recent files included in the training dataset for the model.</li>
                <li><strong>MAX_WORKERS ({MAX_WORKERS}):</strong> Max parallel processes (Adjust based on VRAM).</li>
            </ul>
            <hr>
            <ul>
                <li><strong>Control Group:</strong> Calculates the average of the parameter steps from the last VECTOR_INPUT best vectors.</li>
                <li><strong>Darts (Global RF Consensus):</strong>
                    <ul>
                        <li>Identifies all parameter vectors with at least one historical result > {RESULT_CUTOFF}.</li>
                        <li>Trains a Global Random Forest on all these trajectories simultaneously.</li>
                        <li>Predicts the next result for all valid vectors.</li>
                        <li>Filters predictions > {RESULT_CUTOFF} (or top 10).</li>
                        <li>Calculates the <strong>Geometric Median</strong> of the parameter coordinates of the best predicted vectors to find the stable consensus region.</li>
                    </ul>
                </li>
            </ul>
        </div>
        <p>Comparison of Control Group (Avg Best Vectors) vs Darts (Consensus Region).</p>
    """)

    # --- Diagnostics Section ---
    html_parts.append(generate_diagnostics_section(results))

    models = [('Control Group', 'control'), ('Darts Consensus', 'darts')]

    for title, key in models:
        data = results.get(key, [])
        if not data:
            continue

        labels = [d['file'] for d in data]
        actual_results = [d['stats']['Result'] for d in data]
        profits = [d['stats']['Profit'] for d in data] # Actually predicted result stats
        found_flags = [d['stats']['found'] for d in data]

        # Plot 1: Result
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        x = range(len(labels))
        ax1.plot(x, actual_results, marker='o', color='blue', label='Actual Result')

        actual_results = [float(x) for x in actual_results]
        profits = [float(x) for x in profits]

        avg_res = np.mean(actual_results) if actual_results else 0
        ax1.axhline(y=avg_res, color='r', linestyle='--', label=f'Avg: {avg_res:.2f}')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Result')
        ax1.set_title(f'{title}: Result Performance')
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
        ax2.set_title(f'{title}: Profit Performance')
        ax2.legend()
        ax2.grid(True)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        img2_b64 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        # Calculate Avg
        avg_res = np.mean(actual_results) if actual_results else 0
        hit_rate = np.mean(found_flags) * 100 if found_flags else 0

        html_parts.append(f"""
        <div class="section">
            <h2>{title} Model</h2>
            <div class="metric">Average Result: {avg_res:.2f} | Hit Rate (Params Found): {hit_rate:.1f}%</div>
            <div class="img-container">
                <h3>Result Graph</h3>
                <img src="data:image/png;base64,{img1_b64}" />
            </div>
            <div class="img-container">
                <h3>Profit Graph</h3>
                <img src="data:image/png;base64,{img2_b64}" />
            </div>
            <table>
        """)

        html_parts.append("""
            <thead>
                <tr>
                    <th>File</th>
                    <th>Found?</th>
                    <th>Result</th>
                    <th>Profit</th>
                    <th>Params</th>
                </tr>
            </thead>
        """)

        html_parts.append("<tbody>")

        for d in data:
            params_str = ", ".join([f"{k}={v:.2f}" for k,v in d['pred'].items()])
            color = "green" if d['stats']['found'] else "red"

            meta = d.get('model_meta', {})
            extra_cols = ""

            html_parts.append(f"""
                <tr>
                    <td>{d['file']}</td>
                    <td style="color:{color}">{d['stats']['found']}</td>
                    <td>{d['stats']['Result']:.2f}</td>
                    <td>{d['stats']['Profit']:.2f}</td>
                    <td>{params_str}</td>
                    {extra_cols}
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

    out_path = os.path.join(output_dir, "Predictability_Report_Darts.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("".join(html_parts))

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
