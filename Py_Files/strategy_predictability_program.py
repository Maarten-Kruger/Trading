
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
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

# Configuration
RESULT_CUTOFF = 25
VECTOR_INPUT = 10  # Lookback window size (Model Lags)
TRAINING_WINDOW = 30 # Size of the sliding window used for training (Must be > VECTOR_INPUT)
MAX_WORKERS = 4      # Max parallel processes (Adjust based on VRAM)
TSAI_EPOCHS = 5      # Epochs for Tsai InceptionTime model
NF_MAX_STEPS = 1000  # Max steps for NeuralForecast NHITS

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
            from neuralforecast.losses.pytorch import DistributionLoss
            import torch
            torch.set_float32_matmul_precision('medium')

            class NeuralForecastForecaster:
                def __init__(self, global_config, var_cols, max_steps=100):
                    self.config = global_config
                    self.var_cols = var_cols
                    # Use NHITS - fast and effective
                    # Check for GPU
                    accel = 'gpu' if torch.cuda.is_available() else 'cpu'
                    # print(f"[NeuralForecast] Using accelerator: {accel}")
                    # Use Bernoulli Distribution for binary classification (High Profit Probability)
                    # explicitly set scaler_type='identity' to avoid scaling 0/1 targets
                    self.model = NHITS(h=1, input_size=VECTOR_INPUT, max_steps=max_steps,
                                       loss=DistributionLoss(distribution='Bernoulli', return_params=True),
                                       enable_checkpointing=False, logger=False,
                                       accelerator=accel, batch_size=4096,
                                       stat_cat_exog_list=self.var_cols,
                                       scaler_type='identity')

                def predict(self, Y_df_global, window_start_idx, window_end_idx, coords_list):
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

                    # --- Preprocessing (Classification Target) ---
                    # Target is 1.0 if Result > RESULT_CUTOFF, else 0.0
                    Y_df_window = Y_df_window.copy() # Avoid SettingWithCopy
                    Y_df_window['y'] = (Y_df_window['y'] > RESULT_CUTOFF).astype(np.float32)

                    # --- Static Covariances (New Requirement) ---
                    # Build static_df from coords_list
                    # unique_id is the index (int) which maps to coords_list[i]
                    # We need to build a DF: unique_id | param1 | param2 ...

                    static_data = []
                    # Optimization: Only build for unique_ids present in Y_df_window?
                    # NHITS needs static vars for all series it trains on.
                    # Since we train on the window which contains all series (ids), we need all.

                    # Assuming coords_list indices match unique_id 0..N
                    # coords_list contains tuples of steps. Convert to real values.

                    # This might be heavy if coords_list is huge.
                    # But it's necessary for the requirement.

                    # Vectorized approach to build static_df
                    # unique_ids = np.arange(len(coords_list))

                    # Extract params into lists
                    # keys: var_cols

                    # Pre-calculate params for all coords
                    # This is better done outside but we do it here.

                    # Use integer step coordinates directly for static features to ensure normalization
                    static_df = pd.DataFrame(coords_list, columns=self.var_cols)
                    static_df['unique_id'] = np.arange(len(coords_list))

                    # Reorder: unique_id first
                    cols = ['unique_id'] + [c for c in static_df.columns if c != 'unique_id']
                    static_df = static_df[cols]

                    # 2. Train and Predict
                    nf = NeuralForecast(models=[self.model], freq='D')
                    nf.fit(df=Y_df_window, static_df=static_df)

                    future_df = nf.predict(static_df=static_df)

                    # 3. Find Best
                    if future_df.empty:
                        return None

                    # Debug Stats
                    pred_col = 'NHITS'
                    if pred_col in future_df.columns:
                        d_min = future_df[pred_col].min()
                        d_max = future_df[pred_col].max()
                        d_mean = future_df[pred_col].mean()
                        print(f"    [NF DEBUG] Pred Range: Min={d_min:.4f}, Max={d_max:.4f}, Mean={d_mean:.4f}")

                    # Calculate Stats (B) - Model Prediction
                    # Model predicts Probability (0-1)
                    avg_pred = float(future_df[pred_col].mean())
                    count_above = int((future_df[pred_col] > 0.5).sum()) # Check > 50% confidence
                    count_neg = int((future_df[pred_col] <= 0).sum())

                    # Predict high result (regression)
                    best_row_idx = future_df[pred_col].idxmax()
                    best_row = future_df.loc[best_row_idx]

                    if 'unique_id' in future_df.columns:
                        best_uid = best_row['unique_id']
                    else:
                        best_uid = best_row_idx

                    pred_val = float(best_row[pred_col])

                    # Return dict with metadata
                    return {
                        'id': best_uid,
                        'pred_val': pred_val,
                        'avg_pred': avg_pred,
                        'count_above': count_above,
                        'count_neg': count_neg
                    }

            classes['NeuralForecastForecaster'] = NeuralForecastForecaster
        except Exception as e:
             print(f"Lazy Import Error (NF): {e}")

    if flags['HAS_TSAI']:
        try:
            from tsai.all import InceptionTime, RandomSplitter, TSClassifier, TSDatasets, TSDataLoaders, ts_learner, accuracy
            import torch
            import torch.nn as nn

            class TsaiForecaster:
                def __init__(self, global_config, var_cols, epochs=5):
                    self.config = global_config
                    self.var_cols = var_cols
                    self.epochs = epochs

                def predict(self, master_matrix, window_start_idx, window_end_idx, coords_list):
                    # Slice the matrix: O(1)
                    X_slice = master_matrix[:, window_start_idx:window_end_idx]
                    n_coords, win_len = X_slice.shape

                    if win_len < 2:
                        print(f"[DEBUG TSAI] win_len < 2 ({win_len})")
                        return None

                    # Sliding Window Logic on the slice
                    w = min(VECTOR_INPUT, win_len - 1)
                    if w < 2:
                        print(f"[DEBUG TSAI] w < 2 ({w})")
                        return None

                    # Fast strided windowing using sliding_window_view
                    try:
                        if win_len < w + 1:
                            print(f"[DEBUG TSAI] win_len ({win_len}) < w+1 ({w+1})")
                            return None
                        # Shape: (n_coords, num_windows, window_size)
                        windows = sliding_window_view(X_slice, window_shape=w+1, axis=1)
                        # X_train_raw: (n_coords, num_windows, w)
                        X_train_raw = windows[:, :, :-1]
                        # y_train_raw: (n_coords, num_windows) - The target value at end of window
                        y_train_raw = windows[:, :, -1]

                        # Flatten coords and windows together
                        # (Total_Samples, w)
                        X_train_res = X_train_raw.reshape(-1, w)
                        y_train_flat = y_train_raw.flatten()
                    except Exception as e:
                        print(f"Error in vectorized windowing: {e}")
                        return None

                    if len(X_train_res) == 0:
                        print(f"[DEBUG TSAI] X_train_res is empty")
                        return None

                    # --- Static Covariances Integration ---
                    # We need to append static params as constant channels.
                    # 1. Build params array (n_coords, n_params)
                    n_params = len(self.var_cols)
                    params_array = np.zeros((n_coords, n_params), dtype=np.float32)

                    for i, coord in enumerate(coords_list):
                        p_dict = coords_to_params(coord, self.var_cols, self.config)
                        for j, col in enumerate(self.var_cols):
                            params_array[i, j] = p_dict[col]

                    # 2. Expand params to match X_train structure
                    # X_train_raw shape: (n_coords, num_windows, w)
                    num_windows = X_train_raw.shape[1]

                    # Repeat params for each window
                    # (n_coords, num_windows, n_params)
                    params_expanded = np.repeat(params_array[:, np.newaxis, :], num_windows, axis=1)

                    # Flatten to (Total_Samples, n_params)
                    params_flat = params_expanded.reshape(-1, n_params)

                    # 3. Create Constant Channels
                    # X_train_res: (Total, w) -> (Total, 1, w)
                    X_res_chan = X_train_res[:, np.newaxis, :]

                    # Params: (Total, n_params) -> (Total, n_params, w) (Broadcast)
                    params_chan = np.repeat(params_flat[:, :, np.newaxis], w, axis=2)

                    # Concatenate: (Total, 1 + n_params, w)
                    X_final = np.concatenate([X_res_chan, params_chan], axis=1)

                    # --- Classification Targets ---
                    # 0: DNC (y <= 0)
                    # 1: LR (0 < y <= CUTOFF)
                    # 2: HR (y > CUTOFF)
                    y_class = np.zeros_like(y_train_flat, dtype=np.longlong)

                    # Vectorized binning
                    mask_pos = y_train_flat > 0
                    mask_hr = y_train_flat > RESULT_CUTOFF

                    # Default is 0 (DNC)
                    y_class[mask_pos] = 1 # LR
                    y_class[mask_hr] = 2  # HR

                    # Model
                    c_in = 1 + n_params
                    c_out = 3 # 3 classes

                    model = InceptionTime(c_in=c_in, c_out=c_out)
                    splits = RandomSplitter()(range(len(X_final)))

                    # TSDatasets for Classification (y is long)
                    # For classification, we use Categorize if strings, but here y is already int.
                    # We typically don't need explicit transforms for int targets in TSDatasets unless we want one-hot encoding or similar.
                    # TSClassifier is a Learner factory, NOT a transform.
                    # Passing TSClassifier() as a transform was the error.

                    # We just use [None, None] or [None, [Categorize()]] if needed, but for ints None is usually fine.
                    tfms = [None, None]
                    dsets = TSDatasets(X_final, y_class, tfms=tfms, splits=splits)

                    # High-performance DataLoaders
                    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=4096, num_workers=0, pin_memory=True) # Workers=0 for safety in spawn

                    learn = ts_learner(dls, model, metrics=accuracy, loss_func=nn.CrossEntropyLoss(), verbose=False)
                    learn.fit_one_cycle(self.epochs, 1e-3)

                    # --- Prediction ---
                    # Predict on latest window
                    # X_slice: (n_coords, win_len)
                    # Latest window: X_slice[:, -w:]
                    X_test_res = X_slice[:, -w:] # (n_coords, w)

                    # Add channel dim
                    X_test_res = X_test_res[:, np.newaxis, :] # (n_coords, 1, w)

                    # Add params channels
                    # params_array: (n_coords, n_params)
                    # Broadcast to (n_coords, n_params, w)
                    params_test = np.repeat(params_array[:, :, np.newaxis], w, axis=2)

                    # Concat
                    X_test = np.concatenate([X_test_res, params_test], axis=1) # (n_coords, 1+n_params, w)

                    try:
                        learn.model.eval()
                        input_tensor = torch.from_numpy(X_test).float()
                        device = next(learn.model.parameters()).device
                        input_tensor = input_tensor.to(device)

                        with torch.no_grad():
                            logits = learn.model(input_tensor) # (n_coords, 3)
                            probs = torch.softmax(logits, dim=1) # (n_coords, 3)

                        probs_np = probs.cpu().numpy()

                        # Select candidate with highest confidence in HR (Class 2)
                        hr_probs = probs_np[:, 2]
                        best_idx = np.argmax(hr_probs)

                        # Diagnostic Stats
                        confidence = float(hr_probs[best_idx])
                        pred_class = int(np.argmax(probs_np[best_idx]))

                        all_preds = np.argmax(probs_np, axis=1)
                        count_class_0 = int((all_preds == 0).sum())
                        count_class_1 = int((all_preds == 1).sum())
                        count_class_2 = int((all_preds == 2).sum())

                        avg_hr_prob = float(np.mean(hr_probs))

                        return {
                            'id': int(best_idx),
                            'confidence': confidence,
                            'pred_class': pred_class,
                            'count_class_0': count_class_0,
                            'count_class_1': count_class_1,
                            'count_class_2': count_class_2,
                            'avg_hr_prob': avg_hr_prob
                        }
                    except Exception as e:
                        print(f"[DEBUG TSAI] Exception during prediction: {e}")
                        traceback.print_exc()
                        return None
                    except:
                         print(f"[DEBUG TSAI] Unknown error during prediction")
                         traceback.print_exc()
                         return None


            classes['TsaiForecaster'] = TsaiForecaster
        except Exception as e:
            print(f"Lazy Import Error (Tsai): {e}")
            traceback.print_exc()

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
        epochs = task_args.get('epochs', {'tsai': 5, 'nf': 100})

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
                print(f"  [Worker {os.getpid()}] {file_name}: Darts Error: {e}", flush=True)
                traceback.print_exc()

        # 2. NeuralForecast
        if HAS_NF and 'NeuralForecastForecaster' in forecaster_classes:
            try:
                print(f"  [Worker {os.getpid()}] {file_name}: Starting NeuralForecast...", flush=True) # Verbose

                # Reconstruct DF slice for NF
                nf_matrix = task_args.get('nf_matrix')
                nf_dates = task_args.get('nf_dates')
                coords_list = task_args.get('coords_list')

                if nf_matrix is not None and nf_dates is not None and coords_list is not None:
                     # Efficiently create DF
                     # Use integer index as unique_id
                     num_coords = nf_matrix.shape[0]
                     temp_df = pd.DataFrame(nf_matrix, index=np.arange(num_coords), columns=nf_dates)
                     temp_df.index.name = 'unique_id'
                     Y_df_window = temp_df.reset_index().melt(id_vars='unique_id', var_name='ds', value_name='y')
                     Y_df_window['ds'] = pd.to_datetime(Y_df_window['ds'])
                     Y_df_window['y'] = Y_df_window['y'].astype(np.float32)

                     NeuralForecastForecaster = forecaster_classes['NeuralForecastForecaster']
                     nf_model = NeuralForecastForecaster(global_param_config, var_cols, max_steps=epochs['nf'])

                     # Pass original window indices so predict calculates correct date filter
                     results['nf'] = nf_model.predict(Y_df_window, task_args['window_start'], task_args['window_end'], coords_list)
            except Exception as e:
                print(f"  [Worker {os.getpid()}] {file_name}: NeuralForecast Error: {e}", flush=True)
                traceback.print_exc()

        # 3. Tsai
        if HAS_TSAI and 'TsaiForecaster' in forecaster_classes:
            try:
                print(f"  [Worker {os.getpid()}] {file_name}: Starting Tsai...", flush=True) # Verbose

                coords_list = task_args.get('coords_list')
                if coords_list is not None:
                    TsaiForecaster = forecaster_classes['TsaiForecaster']
                    tsai_model = TsaiForecaster(global_param_config, var_cols, epochs=epochs['tsai'])
                    # slice_matrix is already sliced to [window_start:window_end]
                    # Pass 0 and width to predict
                    win_len = slice_matrix.shape[1]
                    results['tsai'] = tsai_model.predict(slice_matrix, 0, win_len, coords_list)
            except Exception as e:
                print(f"  [Worker {os.getpid()}] {file_name}: Tsai Error: {e}", flush=True)
                traceback.print_exc()

        duration = time.time() - start_time
        return {'file_name': file_name, 'results': results, 'duration': duration}

    except Exception as e:
        return {'file_name': task_args.get('file_name', 'unknown'), 'results': {}, 'error': str(e), 'duration': time.time() - start_time}

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
    recent_completion_times = deque(maxlen=10) # Sliding window for ETA
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
            'coords_list': coords_list, # Passed for static covariances
            'window_start': window_start,
            'window_end': window_end,
            'flags': flags,
            'epochs': {'tsai': TSAI_EPOCHS, 'nf': NF_MAX_STEPS}
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

                    # 1. Ground Truth Stats (A)
                    grid_vals = list(target_file_data['grid_summary'].values())
                    gt_avg = np.mean(grid_vals) if grid_vals else 0
                    gt_count = sum(1 for v in grid_vals if v > RESULT_CUTOFF)
                    gt_neg_count = sum(1 for v in grid_vals if v <= 0)
                    gt_total = len(grid_vals)

                    # Store global stats for reporting
                    global_stats = {
                        'avg': gt_avg,
                        'count_above': gt_count,
                        'count_neg': gt_neg_count,
                        'total': gt_total
                    }

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
                        nf_raw = preds_map.get('nf')
                        nf_meta = {}
                        nf_params = {}

                        if nf_raw:
                            nf_meta = nf_raw # It is a dict now
                            nf_id = nf_raw.get('id')

                            # Resolve ID
                            if isinstance(nf_id, (int, np.integer, str)):
                                try:
                                    idx = int(nf_id)
                                    if 0 <= idx < len(coords_list):
                                        nf_params = coords_to_params(coords_list[idx], var_cols, global_param_config)
                                except:
                                    pass

                            stats = lookup_stats(target_file_data['df'], nf_params, global_param_config)
                            if stats['found'] and 'matched_params' in stats:
                                nf_params = stats['matched_params'] # Snap to grid

                            results['nf'].append({'file': file_name, 'pred': nf_params, 'stats': stats, 'model_meta': nf_meta, 'global_stats': global_stats})

                            # Verbose status
                            # Show Predicted Value and Avg Prediction
                            p_val = nf_meta.get('pred_val', 0)
                            p_avg = nf_meta.get('avg_pred', 0)
                            status_msg.append(f"NF[Y, P:{p_val:.1f}, Avg:{p_avg:.1f}]" if stats['found'] else "NF[N]")
                        else:
                            results['nf'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'Profit': 0, 'found': False}, 'model_meta': {}, 'global_stats': global_stats})
                            status_msg.append("NF[-]")

                    # Tsai
                    if HAS_TSAI:
                        tsai_raw = preds_map.get('tsai')
                        tsai_meta = {}
                        tsai_params = {}

                        if tsai_raw:
                            tsai_meta = tsai_raw
                            tsai_id = tsai_raw.get('id')

                             # Resolve index
                            if isinstance(tsai_id, (int, np.integer)):
                                try:
                                    idx = int(tsai_id)
                                    if 0 <= idx < len(coords_list):
                                        tsai_params = coords_to_params(coords_list[idx], var_cols, global_param_config)
                                except:
                                    pass

                            stats = lookup_stats(target_file_data['df'], tsai_params, global_param_config)
                            if stats['found'] and 'matched_params' in stats:
                                tsai_params = stats['matched_params']

                            results['tsai'].append({'file': file_name, 'pred': tsai_params, 'stats': stats, 'model_meta': tsai_meta, 'global_stats': global_stats})

                            # Verbose status
                            conf = tsai_meta.get('confidence', 0)
                            cls = tsai_meta.get('pred_class', 0)
                            status_msg.append(f"Tsai[Y, C:{conf:.2f}, Cls:{cls}]" if stats['found'] else "Tsai[N]")
                        else:
                            results['tsai'].append({'file': file_name, 'pred': {}, 'stats': {'Result': 0, 'Profit': 0, 'found': False}, 'model_meta': {}, 'global_stats': global_stats})
                            status_msg.append("Tsai[-]")

                except Exception as e:
                    print(f"Error processing result for {file_name}: {e}")
                    traceback.print_exc()

                # Progress & ETA Calculation
                files_processed += 1
                now = time.time()
                recent_completion_times.append(now)

                if len(recent_completion_times) > 1:
                    # Calculate speed based on sliding window (ignoring initial warmup if deque is full/partial)
                    # Time delta between oldest and newest in the window
                    window_duration = recent_completion_times[-1] - recent_completion_times[0]
                    # Number of intervals is len - 1
                    avg_sec_per_file = window_duration / (len(recent_completion_times) - 1)
                else:
                    # Fallback to global average if not enough data
                    avg_sec_per_file = (now - start_time) / files_processed

                rem = total_files_to_process - files_processed
                eta_seconds = avg_sec_per_file * rem

                # Check if we should print based on MAX_WORKERS batch or completion
                if files_processed % MAX_WORKERS == 0 or files_processed == total_files_to_process:
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

    # NeuralForecast Diagnostics
    nf_data = results.get('nf', [])
    if nf_data:
        html += """
        <h3>NeuralForecast Diagnostics</h3>
        <p>Model predicts Probability (0-1). Comparing 'Win Rate' (Probability) instead of raw Result.</p>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th style="background-color: #e6f7ff;">Act Win Rate</th>
                    <th style="background-color: #e6f7ff;">Act > 25</th>
                    <th style="background-color: #e6f7ff;">Total</th>
                    <th style="background-color: #fff0f6;">Model Win Rate</th>
                    <th style="background-color: #fff0f6;">Model > 50%</th>
                    <th style="background-color: #f9f0ff;">Optimism Ratio</th>
                </tr>
            </thead>
            <tbody>
        """
        for d in nf_data:
            g_stats = d.get('global_stats', {})
            m_meta = d.get('model_meta', {})

            act_high = g_stats.get('count_above', 0)
            total = g_stats.get('total', 1)
            if total == 0: total = 1
            act_prob = act_high / total

            mod_avg = m_meta.get('avg_pred', 0) # Average Probability
            mod_high = m_meta.get('count_above', 0) # Count > 0.5

            # Optimism: Model Avg Prob / Actual Win Rate
            if act_prob < 1e-5:
                optimism = 0.0
            else:
                optimism = mod_avg / act_prob

            opt_color = "black"
            if optimism > 1.2: opt_color = "red" # Over-optimistic
            elif optimism < 0.8: opt_color = "blue" # Pessimistic

            html += f"""
            <tr>
                <td>{d['file']}</td>
                <td style="background-color: #e6f7ff;">{act_prob:.2%}</td>
                <td style="background-color: #e6f7ff;">{act_high}</td>
                <td style="background-color: #e6f7ff;">{total}</td>
                <td style="background-color: #fff0f6;">{mod_avg:.2%}</td>
                <td style="background-color: #fff0f6;">{mod_high}</td>
                <td style="background-color: #f9f0ff; color: {opt_color}; font-weight: bold;">{optimism:.2f}x</td>
            </tr>
            """
        html += "</tbody></table>"

    # Tsai Diagnostics
    tsai_data = results.get('tsai', [])
    if tsai_data:
        html += """
        <h3>Tsai Diagnostics</h3>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th style="background-color: #e6f7ff;">Act Avg</th>
                    <th style="background-color: #e6f7ff;">Act > 25</th>
                    <th style="background-color: #fff0f6;">Conf Score</th>
                    <th style="background-color: #fff0f6;">Pred Class</th>
                    <th style="background-color: #fff0f6;">Class 0 (Loss)</th>
                    <th style="background-color: #fff0f6;">Class 1 (Profit)</th>
                    <th style="background-color: #fff0f6;">Class 2 (High)</th>
                </tr>
            </thead>
            <tbody>
        """
        for d in tsai_data:
            g_stats = d.get('global_stats', {})
            m_meta = d.get('model_meta', {})

            act_avg = g_stats.get('avg', 0)
            act_high = g_stats.get('count_above', 0)

            conf = m_meta.get('confidence', 0)
            cls = m_meta.get('pred_class', -1)

            c0 = m_meta.get('count_class_0', 0)
            c1 = m_meta.get('count_class_1', 0)
            c2 = m_meta.get('count_class_2', 0)

            # Highlight chosen class
            cls_str = "Unknown"
            if cls == 0: cls_str = "<span style='color:red'>DNC (0)</span>"
            elif cls == 1: cls_str = "<span style='color:orange'>Low (1)</span>"
            elif cls == 2: cls_str = "<span style='color:green'>High (2)</span>"

            html += f"""
            <tr>
                <td>{d['file']}</td>
                <td style="background-color: #e6f7ff;">{act_avg:.2f}</td>
                <td style="background-color: #e6f7ff;">{act_high}</td>
                <td style="background-color: #fff0f6;">{conf:.1%}</td>
                <td style="background-color: #fff0f6;">{cls_str}</td>
                <td style="background-color: #fff0f6;">{c0}</td>
                <td style="background-color: #fff0f6;">{c1}</td>
                <td style="background-color: #fff0f6;">{c2}</td>
            </tr>
            """
        html += "</tbody></table>"

    html += "</div>"
    return html

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
    """)

    html_parts.append(f"""
    <body>
        <h1>Strategy Predictability Report</h1>
        <div class="section">
            <h2>Definitions</h2>
            <ul>
                <li><strong>RESULT_CUTOFF ({RESULT_CUTOFF}):</strong> Filter vectors inside each file where the result exceeds a defined threshold.</li>
                <li><strong>VECTOR_INPUT ({VECTOR_INPUT}):</strong> Lookback window size (Model Lags). The number of past time steps (files/vectors) the model looks at to make a prediction.</li>
                <li><strong>TRAINING_WINDOW ({TRAINING_WINDOW}):</strong> Size of the sliding window used for training. The number of recent files included in the training dataset for the model.</li>
                <li><strong>MAX_WORKERS ({MAX_WORKERS}):</strong> Max parallel processes (Adjust based on VRAM).</li>
                <li><strong>TSAI_EPOCHS ({TSAI_EPOCHS}):</strong> Epochs for Tsai InceptionTime model.</li>
                <li><strong>NF_MAX_STEPS ({NF_MAX_STEPS}):</strong> Max steps for NeuralForecast NHITS.</li>
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

    # --- Diagnostics Section ---
    html_parts.append(generate_diagnostics_section(results))

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
        """)

        # Dynamic Headers
        if key == 'nf':
            html_parts.append("""
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Found?</th>
                        <th>Result</th>
                        <th>Profit</th>
                        <th>Params</th>
                        <th>Confidence</th>
                        <th>Model Avg</th>
                        <th>Count > 50%</th>
                    </tr>
                </thead>
            """)
        elif key == 'tsai':
            html_parts.append("""
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Found?</th>
                        <th>Result</th>
                        <th>Profit</th>
                        <th>Params</th>
                        <th>Confidence</th>
                        <th>Pred Class</th>
                        <th>Count HR</th>
                    </tr>
                </thead>
            """)
        else:
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

            if key == 'nf':
                p_val = meta.get('pred_val', 0)
                p_avg = meta.get('avg_pred', 0)
                cnt = meta.get('count_above', 0)
                extra_cols = f"<td>{p_val:.4f}</td><td>{p_avg:.4f}</td><td>{cnt}</td>"
            elif key == 'tsai':
                conf = meta.get('confidence', 0)
                cls = meta.get('pred_class', 0)
                cnt = meta.get('count_class_2', 0)
                extra_cols = f"<td>{conf:.2%}</td><td>{cls}</td><td>{cnt}</td>"

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

    out_path = os.path.join(output_dir, "Predictability_Report_MultiModel.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("".join(html_parts))

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
