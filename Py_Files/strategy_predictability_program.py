
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

# Suppress Warnings and Logging
warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

# Configuration
RESULT_CUTOFF = 25
VECTOR_INPUT = 10  # Lookback window size (Model Lags)
TRAINING_WINDOW = 30 # Size of the sliding window used for training (Must be > VECTOR_INPUT)

# --- Library Availability Flags (Lazy Check) ---
# We check availability without importing to keep workers fast
import importlib.util

def check_lib(name):
    return importlib.util.find_spec(name) is not None

HAS_DARTS = check_lib("darts")
HAS_NF = check_lib("neuralforecast")
HAS_TSAI = check_lib("tsai")

# --- Helper Functions ---

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

                def predict(self, best_vectors_history):
                    # Filter out empty entries if any (or handle them)
                    valid_history = [bv for bv in best_vectors_history if bv['Result'] != 0 or bv['params']]

                    # We need at least VECTOR_INPUT + 1 data points to fit lags
                    if len(valid_history) < VECTOR_INPUT + 1:
                        return None

                    data_matrix = []
                    for bv in valid_history:
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

            classes['DartsForecaster'] = DartsForecaster
        except Exception as e:
            print(f"Lazy Import Error (Darts): {e}")

    if flags['HAS_NF']:
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NHITS
            import torch
            torch.set_float32_matmul_precision('medium')

            class NeuralForecastForecaster:
                def __init__(self, global_config, var_cols):
                    self.config = global_config
                    self.var_cols = var_cols
                    # Use NHITS - fast and effective
                    # Check for GPU
                    accel = 'gpu' if torch.cuda.is_available() else 'cpu'
                    # print(f"[NeuralForecast] Using accelerator: {accel}")
                    self.model = NHITS(h=1, input_size=VECTOR_INPUT, max_steps=100,
                                       enable_checkpointing=False, logger=False,
                                       accelerator=accel)

                def predict(self, Y_df_global, window_start_idx, window_end_idx):
                    if Y_df_global is None or Y_df_global.empty:
                        return None

                    # Calculate date range for the slice
                    # ds starts at 2020-01-01
                    start_date_base = datetime(2020, 1, 1)

                    slice_start_date = start_date_base + timedelta(days=window_start_idx)
                    slice_end_date = start_date_base + timedelta(days=window_end_idx)

                    # Optimization: use searchsorted if sorted, but boolean mask is okay for millions of rows on modern RAM
                    mask = (Y_df_global['ds'] >= slice_start_date) & (Y_df_global['ds'] < slice_end_date)
                    Y_df_window = Y_df_global.loc[mask]

                    if Y_df_window.empty:
                        return None

                    # 2. Train and Predict
                    nf = NeuralForecast(models=[self.model], freq='D')
                    nf.fit(df=Y_df_window)

                    future_df = nf.predict()

                    # 3. Find Best
                    if future_df.empty:
                        return None

                    # future_df has columns [ds, NHITS]
                    # Determine best_uid
                    # best_uid will be the index (int) or whatever was used as unique_id
                    if 'unique_id' in future_df.columns:
                        best_uid = future_df.loc[future_df['NHITS'].idxmax()]['unique_id']
                    else:
                        best_uid = future_df['NHITS'].idxmax()

                    # Return the identifier (index), let main process resolve it
                    return best_uid

            classes['NeuralForecastForecaster'] = NeuralForecastForecaster
        except Exception as e:
             print(f"Lazy Import Error (NF): {e}")

    if flags['HAS_TSAI']:
        try:
            from tsai.all import InceptionTime, RandomSplitter, TSRegression, TSDatasets, TSDataLoaders, ts_learner, mae
            import torch

            class TsaiForecaster:
                def __init__(self, global_config, var_cols):
                    self.config = global_config
                    self.var_cols = var_cols

                def predict(self, master_matrix, window_start_idx, window_end_idx):
                    # Slice the matrix: O(1)
                    X_slice = master_matrix[:, window_start_idx:window_end_idx]
                    n_coords, win_len = X_slice.shape

                    if win_len < 2: return None

                    # Sliding Window Logic on the slice
                    w = min(VECTOR_INPUT, win_len - 1)
                    if w < 2: return None

                    # Fast strided windowing using sliding_window_view
                    try:
                        if win_len < w + 1: return None
                        windows = sliding_window_view(X_slice, window_shape=w+1, axis=1)
                        X_train = windows[:, :, :-1]
                        y_train = windows[:, :, -1]
                        X_train = X_train.reshape(-1, w)
                        y_train = y_train.flatten()
                    except Exception as e:
                        print(f"Error in vectorized windowing: {e}")
                        return None

                    # Add channel dim: (Total_Samples, 1, w)
                    X_train = X_train[:, np.newaxis, :]
                    if len(X_train) == 0: return None

                    # Model
                    model = InceptionTime(c_in=1, c_out=1)
                    splits = RandomSplitter()(range(len(X_train)))
                    tfms = [None, [TSRegression()]]
                    dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits)

                    # High-performance DataLoaders
                    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=4096, num_workers=8, pin_memory=True)

                    learn = ts_learner(dls, model, metrics=mae, verbose=False)
                    learn.fit_one_cycle(5, 1e-3)

                    # Predict on latest window (the end of the slice)
                    X_test = X_slice[:, -w:]
                    X_test = X_test[:, np.newaxis, :]

                    try:
                        learn.model.eval()
                        input_tensor = torch.from_numpy(X_test).float()
                        device = next(learn.model.parameters()).device
                        input_tensor = input_tensor.to(device)
                        with torch.no_grad():
                            preds = learn.model(input_tensor)
                        preds_np = preds.cpu().numpy().flatten()
                        best_idx = np.argmax(preds_np)
                        return int(best_idx)
                    except Exception as e:
                        print(f"[DEBUG TSAI] Exception during prediction: {e}")
                        traceback.print_exc()
                        return None

            classes['TsaiForecaster'] = TsaiForecaster
        except Exception as e:
            print(f"Lazy Import Error (Tsai): {e}")

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
    try:
        # Unpack arguments
        file_name = task_args['file_name']
        history_best = task_args['history_best']
        slice_matrix = task_args['slice_matrix'] # Numpy array

        global_param_config = task_args['config']
        var_cols = task_args['var_cols']

        flags = task_args['flags']
        HAS_DARTS = flags['HAS_DARTS']
        HAS_NF = flags['HAS_NF']
        HAS_TSAI = flags['HAS_TSAI']

        # Lazy load forecaster classes inside the worker
        forecaster_classes = get_forecasters(flags)

        results = {'control': None, 'darts': None, 'nf': None, 'tsai': None}

        # 0. Control Group
        try:
            control_model = ControlGroupForecaster(global_param_config, var_cols)
            results['control'] = control_model.predict(history_best)
        except Exception as e:
            pass

        # 1. Darts
        if HAS_DARTS and 'DartsForecaster' in forecaster_classes:
            try:
                DartsForecaster = forecaster_classes['DartsForecaster']
                darts_model = DartsForecaster(global_param_config, var_cols)
                results['darts'] = darts_model.predict(history_best)
            except Exception as e:
                pass

        # 2. NeuralForecast
        if HAS_NF and 'NeuralForecastForecaster' in forecaster_classes:
            try:
                # print(f"  [Worker] {file_name}: Starting NeuralForecast...", flush=True) # Verbose

                # Reconstruct DF slice for NF
                nf_matrix = task_args.get('nf_matrix')
                nf_dates = task_args.get('nf_dates')
                # nf_coords removed to reduce payload

                if nf_matrix is not None and nf_dates is not None:
                     # Efficiently create DF
                     # Use integer index as unique_id
                     num_coords = nf_matrix.shape[0]
                     temp_df = pd.DataFrame(nf_matrix, index=np.arange(num_coords), columns=nf_dates)
                     temp_df.index.name = 'unique_id'
                     Y_df_window = temp_df.reset_index().melt(id_vars='unique_id', var_name='ds', value_name='y')
                     Y_df_window['ds'] = pd.to_datetime(Y_df_window['ds'])
                     Y_df_window['y'] = Y_df_window['y'].astype(np.float32)

                     NeuralForecastForecaster = forecaster_classes['NeuralForecastForecaster']
                     nf_model = NeuralForecastForecaster(global_param_config, var_cols)

                     # Suppress output
                     with open(os.devnull, 'w') as devnull:
                         # Pass original window indices so predict calculates correct date filter
                         results['nf'] = nf_model.predict(Y_df_window, task_args['window_start'], task_args['window_end'])
            except Exception as e:
                pass

        # 3. Tsai
        if HAS_TSAI and 'TsaiForecaster' in forecaster_classes:
            try:
                # print(f"  [Worker] {file_name}: Starting Tsai...", flush=True) # Verbose

                TsaiForecaster = forecaster_classes['TsaiForecaster']
                tsai_model = TsaiForecaster(global_param_config, var_cols)
                # slice_matrix is already sliced to [window_start:window_end]
                # Pass 0 and width to predict
                win_len = slice_matrix.shape[1]
                results['tsai'] = tsai_model.predict(slice_matrix, 0, win_len)
            except Exception as e:
                pass

        return {'file_name': file_name, 'results': results}

    except Exception as e:
        return {'file_name': task_args.get('file_name', 'unknown'), 'results': {}, 'error': str(e)}

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

    print("\n=== Definitions ===")
    print(f"VECTOR_INPUT ({VECTOR_INPUT}): Lookback window size (Model Lags). The number of past time steps (files/vectors) the model looks at to make a prediction.")
    print(f"TRAINING_WINDOW ({TRAINING_WINDOW}): Size of the sliding window used for training. The number of recent files included in the training dataset for the model.")
    print("-" * 30)
    print("Control Group: Calculates the average of the parameter steps from the last VECTOR_INPUT best vectors.")
    print("Darts (Random Forest): Trains a Random Forest regressor on the trajectory of the best vectors within the TRAINING_WINDOW to predict the next best vector coordinates.")
    print("NeuralForecast (NHITS): Trains a specialized neural network (NHITS) on the entire result surface history within the TRAINING_WINDOW to forecast the result of every parameter combination.")
    print("Tsai (InceptionTime): Trains a Time Series Transformer/CNN (InceptionTime) on the result history of all coordinates within the TRAINING_WINDOW to predict the next best vector.")
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
    # Init with 0 or NaN? 0 implies bad result, which is safe for maximization logic.
    master_matrix = np.zeros((num_coords, num_files), dtype=np.float32)

    # Fill matrix
    # This might take a moment but it's done ONCE.
    for t, fd in enumerate(files_data):
        grid = fd['grid_summary']
        for coord, res in grid.items():
            if coord in coord_to_idx:
                idx = coord_to_idx[coord]
                master_matrix[idx, t] = res

    # 3. Build NeuralForecast Global DataFrame (Pandas Long Format)
    # Columns: unique_id, ds, y
    # ds = 2020-01-01 + t days

    # We can construct this efficiently from the master matrix using numpy magic
    # or list comprehension.

    # Let's use list comprehension for clarity and safety with strings
    nf_records = []
    base_date = datetime(2020, 1, 1)

    # To optimize: Create a DF directly from the matrix?
    # NeuralForecast needs 'unique_id' column (string of tuple)
    # 'ds' column (datetime)
    # 'y' column (float)

    # Fast approach: Stack the matrix
    # Rows: Coords, Cols: Time
    # We want Long format.

    # Coords column
    # Repeat coords list for each time step?

    # Let's do a semi-vectorized approach
    # Create indices
    t_indices = np.arange(num_files)
    dates = [base_date + timedelta(days=int(t)) for t in t_indices]

    # We need to melt the master matrix
    # DataFrame(master_matrix, index=coords_strs, columns=dates) -> melt

    coords_strs = [str(c) for c in coords_list]

    # Create a DataFrame for easy melting
    # This consumes RAM but is fast.
    # If 860,000 rows * 30 cols = 25M elements. Float32 ~ 100MB. Pandas overhead x5 ~ 500MB.
    # This is fine for "nice computer".

    temp_df = pd.DataFrame(master_matrix, index=coords_strs, columns=dates)
    temp_df.index.name = 'unique_id'

    # Reset index to make unique_id a column
    # Melt
    # id_vars='unique_id', var_name='ds', value_name='y'
    Y_df_global = temp_df.reset_index().melt(id_vars='unique_id', var_name='ds', value_name='y')

    # Ensure types
    Y_df_global['ds'] = pd.to_datetime(Y_df_global['ds'])
    Y_df_global['y'] = Y_df_global['y'].astype(np.float32)

    print(f"Master Matrix Construction took {time.time() - start_matrix:.2f}s")

    # 4. Extract Best Vectors (Compatibility)
    best_vectors_history = [fd['best_vector'] for fd in files_data]

    # Prepare Models (Main process only needs ControlGroup for non-parallel parts if any, but in parallel it's done in worker)
    # We no longer instantiate Darts/NF/Tsai here to avoid importing them in main.

    # Predictions Storage
    # Structure: {'darts': [], 'nf': [], 'tsai': []}
    results = {'control': [], 'darts': [], 'nf': [], 'tsai': []}

    start_index = TRAINING_WINDOW + 1
    if start_index >= len(files_data):
        # Fallback if we have fewer files than TRAINING_WINDOW but enough to start
        start_index = VECTOR_INPUT + 2

    if start_index >= len(files_data):
        print(f"Not enough files. Need > {start_index}")
        return

    print("Preparing Tasks for Parallel Execution...")

    start_time = time.time()
    files_processed = 0
    total_files_to_process = len(files_data) - start_index

    # Flags for worker
    flags = {'HAS_DARTS': HAS_DARTS, 'HAS_NF': HAS_NF, 'HAS_TSAI': HAS_TSAI}

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

        # NF specific: Dates slice
        nf_dates_slice = dates[window_start:window_end]

        task = {
            'file_name': file_name,
            'history_best': history_best,
            'slice_matrix': slice_matrix,
            'nf_matrix': slice_matrix if HAS_NF else None,
            'nf_dates': nf_dates_slice if HAS_NF else None,
            # 'nf_coords': coords_strs if HAS_NF else None, # Removed to reduce payload
            'config': global_param_config,
            'var_cols': var_cols,
            # 'coords_list': coords_list, # Removed to reduce payload
            'window_start': window_start,
            'window_end': window_end,
            'flags': flags
        }
        tasks.append((i, task))

    MAX_WORKERS = 4
    print(f"Submitted {len(tasks)} tasks to ProcessPoolExecutor (Spawn, Max Workers={MAX_WORKERS})...")

    try:
        # Use spawn context as requested for independent GPU streams/memory
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(worker_predict_week, t[1]): t[0] for t in tasks}

            print("\n--- Parallel Processing Started ---")

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

                    preds_map = res['results']

                    # --- Verification (Main Process) ---
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

                    # NF
                    if HAS_NF:
                        nf_res = preds_map.get('nf')
                        if nf_res is not None:
                            # Resolve index to params if int/str
                            if isinstance(nf_res, (int, np.integer, str)):
                                try:
                                    idx = int(nf_res)
                                    if 0 <= idx < len(coords_list):
                                        nf_res = coords_to_params(coords_list[idx], var_cols, global_param_config)
                                    else:
                                        nf_res = {}
                                except:
                                    nf_res = {}

                            stats = lookup_stats(target_file_data['df'], nf_res, global_param_config)
                            if stats['found'] and 'matched_params' in stats:
                                preds_map['nf'] = stats['matched_params']
                            results['nf'].append({'file': file_name, 'pred': preds_map['nf'], 'stats': stats})
                            status_msg.append(f"NF:{'Y' if stats['found'] else 'N'}")
                        else:
                            results['nf'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'Profit': 0, 'found': False}})
                            status_msg.append("NF:-")

                    # Tsai
                    if HAS_TSAI:
                        tsai_res = preds_map.get('tsai')
                        if tsai_res is not None:
                             # Resolve index to params if int
                            if isinstance(tsai_res, (int, np.integer)):
                                try:
                                    idx = int(tsai_res)
                                    if 0 <= idx < len(coords_list):
                                        tsai_res = coords_to_params(coords_list[idx], var_cols, global_param_config)
                                    else:
                                        tsai_res = {}
                                except:
                                    tsai_res = {}

                            stats = lookup_stats(target_file_data['df'], tsai_res, global_param_config)
                            if stats['found'] and 'matched_params' in stats:
                                preds_map['tsai'] = stats['matched_params']
                            results['tsai'].append({'file': file_name, 'pred': preds_map['tsai'], 'stats': stats})
                            status_msg.append(f"Tsai:{'Y' if stats['found'] else 'N'}")
                        else:
                            results['tsai'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'Profit': 0, 'found': False}})
                            status_msg.append("Tsai:-")

                except Exception as e:
                    print(f"Error processing result for {file_name}: {e}")
                    traceback.print_exc()

                # Progress
                files_processed += 1
                elapsed = time.time() - start_time
                avg = elapsed / files_processed
                rem = total_files_to_process - files_processed
                eta = avg * rem

                status_str = " | ".join(status_msg)
                print(f"  > [{files_processed}/{total_files_to_process}] {file_name} | {status_str} | ETA: {eta/60:.2f} min", flush=True)

    except KeyboardInterrupt:
        print("\nProcess cancelled by user. Outputting available results...")

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
        <div class="section">
            <h2>Definitions</h2>
            <ul>
                <li><strong>VECTOR_INPUT ({VECTOR_INPUT}):</strong> Lookback window size (Model Lags). The number of past time steps (files/vectors) the model looks at to make a prediction.</li>
                <li><strong>TRAINING_WINDOW ({TRAINING_WINDOW}):</strong> Size of the sliding window used for training. The number of recent files included in the training dataset for the model.</li>
            </ul>
            <hr>
            <ul>
                <li><strong>Control Group:</strong> Calculates the average of the parameter steps from the last VECTOR_INPUT best vectors.</li>
                <li><strong>Darts (Random Forest):</strong> Trains a Random Forest regressor on the trajectory of the best vectors within the TRAINING_WINDOW to predict the next best vector coordinates.</li>
                <li><strong>NeuralForecast (NHITS):</strong> Trains a specialized neural network (NHITS) on the entire result surface history within the TRAINING_WINDOW to forecast the result of every parameter combination.</li>
                <li><strong>Tsai (InceptionTime):</strong> Trains a Time Series Transformer/CNN (InceptionTime) on the result history of all coordinates within the TRAINING_WINDOW to predict the next best vector.</li>
            </ul>
        </div>
        <p>Comparison of Control Group (Avg Best Vectors), Darts (Vector Trajectory), NeuralForecast (Panel Surface), and Tsai (Panel Surface).</p>
    """)

    models = [('Control Group', 'control'), ('Darts', 'darts'), ('NeuralForecast', 'nf'), ('Tsai', 'tsai')]

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
