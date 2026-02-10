
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
import random

# Suppress Warnings
warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.ERROR)

# --- Configuration ---
RESULT_CUTOFF = 25
VECTOR_INPUT = 10  # Lookback window size (Model Lags)
TRAINING_WINDOW = 30 # Size of the sliding window used for training
MAX_WORKERS = 4      # Max parallel processes

# --- Generative Optimization Config ---
GEN_EPOCHS = 50       # VAE/Surrogate Training Epochs
GEN_POP_SIZE = 50     # NSGA-II Population Size
GEN_GENERATIONS = 20  # NSGA-II Generations
GEN_LATENT_DIM = 4    # VAE Latent Dimension

# --- Library Availability ---
import importlib.util

def check_lib(name):
    return importlib.util.find_spec(name) is not None

HAS_DARTS = check_lib("darts")
HAS_TORCH = check_lib("torch")
HAS_SKLEARN = check_lib("sklearn")

# --- Helper Functions ---

def read_csv_robust(filepath):
    """
    Optimized CSV reader that sniffs separator and decimal format.
    Uses C engine for speed.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            if not header: return pd.DataFrame()

        sep = ';' if ';' in header else ','
        decimal = ',' if sep == ';' else '.'

        df = pd.read_csv(filepath, sep=sep, decimal=decimal, engine='c')
    except Exception as e:
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
    """Helper function for parallel file loading."""
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
    except Exception as e:
        return None

def preprocess_file_data(fdata, config):
    """Extracts optimization surface and best vector for a file."""
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
        temp_df = df[var_cols + ['Result']].copy()

        # Convert to steps
        for col in var_cols:
            cfg = config[col]
            if cfg['step'] > 0:
                vals = temp_df[col].values
                temp_df[col] = np.round((vals - cfg['min']) / cfg['step']).astype(int)
            else:
                temp_df[col] = 0

        grouped = temp_df.groupby(var_cols)['Result'].max()

        if len(var_cols) == 1:
            grid_summary = { (k,): v for k, v in grouped.to_dict().items() }
        else:
            grid_summary = grouped.to_dict()

    return {
        'best_vector': best_vector,
        'grid_summary': grid_summary
    }

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

    # 2. Fallback: Nearest Neighbor in Normalized Step Space
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

# --- Forecasting Models ---

class ControlGroupForecaster:
    def __init__(self, global_config, var_cols):
        self.config = global_config
        self.var_cols = var_cols

    def predict(self, best_vectors_history):
        relevant_history = best_vectors_history[-VECTOR_INPUT:]
        if not relevant_history: return None
        valid_history = [bv for bv in relevant_history if bv['Result'] != 0 or bv['params']]
        if not valid_history: return None

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

def get_darts_forecaster(flags):
    if flags['HAS_DARTS']:
        try:
            from darts import TimeSeries
            from darts.models import RandomForest

            class DartsForecaster:
                def __init__(self, global_config, var_cols):
                    self.config = global_config
                    self.var_cols = var_cols
                    self.model = RandomForest(lags=VECTOR_INPUT, n_estimators=50, random_state=42, n_jobs=-1)

                def predict(self, best_vectors_history):
                    valid_history = [bv for bv in best_vectors_history if bv['Result'] != 0 or bv['params']]
                    if len(valid_history) < VECTOR_INPUT + 1: return None

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
            return DartsForecaster
        except:
            return None
    return None

# --- Generative Optimization Models ---

def get_generative_optimizer(flags):
    if flags['HAS_TORCH'] and flags['HAS_SKLEARN']:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.neighbors import NearestNeighbors
            from torch.utils.data import DataLoader, TensorDataset

            # Define CHSDM-Net (Surrogate) - Reward Predictor
            class CHSDM_Surrogate(nn.Module):
                def __init__(self, input_dim, hidden_dim=64):
                    super(CHSDM_Surrogate, self).__init__()
                    # Static features (input_dim) -> Hidden -> Output (Reward)
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1) # Predicts 'Mean Result'
                    )
                def forward(self, x):
                    return self.net(x)

            # Define Stability Surrogate - Variance Predictor
            class Stability_Surrogate(nn.Module):
                def __init__(self, input_dim, hidden_dim=64):
                    super(Stability_Surrogate, self).__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1) # Predicts 'Std Dev'
                    )
                def forward(self, x):
                    return torch.abs(self.net(x)) # Variance must be positive

            # Define VAE
            class VAE(nn.Module):
                def __init__(self, input_dim, latent_dim=4, hidden_dim=64):
                    super(VAE, self).__init__()
                    # Encoder
                    self.enc1 = nn.Linear(input_dim, hidden_dim)
                    self.enc2_mu = nn.Linear(hidden_dim, latent_dim)
                    self.enc2_logvar = nn.Linear(hidden_dim, latent_dim)
                    # Decoder
                    self.dec1 = nn.Linear(latent_dim, hidden_dim)
                    self.dec2 = nn.Linear(hidden_dim, input_dim)
                    self.relu = nn.ReLU()

                def encode(self, x):
                    h1 = self.relu(self.enc1(x))
                    return self.enc2_mu(h1), self.enc2_logvar(h1)

                def reparameterize(self, mu, logvar):
                    std = torch.exp(0.5*logvar)
                    eps = torch.randn_like(std)
                    return mu + eps*std

                def decode(self, z):
                    h3 = self.relu(self.dec1(z))
                    return self.dec2(h3) # Linear output (regression)

                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = self.reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar

            class GenerativeOptimizer:
                def __init__(self, global_config, var_cols, epochs=GEN_EPOCHS):
                    self.config = global_config
                    self.var_cols = var_cols
                    self.epochs = epochs
                    self.input_dim = len(var_cols)
                    self.latent_dim = GEN_LATENT_DIM
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                def optimize(self, master_matrix, window_start_idx, window_end_idx, coords_list):
                    # 1. Prepare Data
                    # We need (X, y) pairs where X is static vector params, y is Result
                    # We average results over the window to get a "stable" target for the surrogate
                    # Or we take the max result? Let's take Mean to encourage stability.

                    slice_matrix = master_matrix[:, window_start_idx:window_end_idx]
                    # Identify valid rows (not -1000)
                    valid_mask = np.mean(slice_matrix, axis=1) > -900

                    if not np.any(valid_mask):
                        return None

                    valid_indices = np.where(valid_mask)[0]
                    valid_coords = [coords_list[i] for i in valid_indices]

                    # Target 1: Mean result (Reward)
                    # Target 2: Std Dev (Instability)

                    raw_data = slice_matrix[valid_indices, :]
                    masked_data = np.ma.masked_less(raw_data, -900)

                    targets_mean = masked_data.mean(axis=1).filled(0)
                    targets_std = masked_data.std(axis=1).filled(100) # High variance if undefined

                    # Normalize Targets
                    t_mean_mu = targets_mean.mean()
                    t_mean_sigma = targets_mean.std() + 1e-6
                    y_mean_norm = (targets_mean - t_mean_mu) / t_mean_sigma

                    t_std_mu = targets_std.mean()
                    t_std_sigma = targets_std.std() + 1e-6
                    y_std_norm = (targets_std - t_std_mu) / t_std_sigma

                    # Define X_data for sklearn later
                    X_data = np.array(valid_coords, dtype=np.float32)

                    X_tensor = torch.tensor(X_data).to(self.device)
                    y_mean_tensor = torch.tensor(y_mean_norm, dtype=np.float32).reshape(-1, 1).to(self.device)
                    y_std_tensor = torch.tensor(y_std_norm, dtype=np.float32).reshape(-1, 1).to(self.device)

                    dataset = TensorDataset(X_tensor, y_mean_tensor, y_std_tensor)
                    loader = DataLoader(dataset, batch_size=64, shuffle=True)

                    # 2. Train VAE, Reward Surrogate, Stability Surrogate
                    vae = VAE(self.input_dim, self.latent_dim).to(self.device)
                    surrogate_reward = CHSDM_Surrogate(self.input_dim).to(self.device)
                    surrogate_stability = Stability_Surrogate(self.input_dim).to(self.device)

                    opt_vae = optim.Adam(vae.parameters(), lr=1e-3)
                    opt_reward = optim.Adam(surrogate_reward.parameters(), lr=1e-3)
                    opt_stab = optim.Adam(surrogate_stability.parameters(), lr=1e-3)

                    loss_mse = nn.MSELoss()

                    vae.train()
                    surrogate_reward.train()
                    surrogate_stability.train()

                    for epoch in range(self.epochs):
                        for xb, yb_mean, yb_std in loader:
                            # Train VAE
                            recon, mu, logvar = vae(xb)
                            loss_recon = loss_mse(recon, xb)
                            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss_vae = loss_recon + 0.001 * loss_kl

                            opt_vae.zero_grad()
                            loss_vae.backward()
                            opt_vae.step()

                            # Train Reward Surrogate
                            pred_r = surrogate_reward(xb)
                            loss_r = loss_mse(pred_r, yb_mean)
                            opt_reward.zero_grad()
                            loss_r.backward()
                            opt_reward.step()

                            # Train Stability Surrogate
                            pred_s = surrogate_stability(xb)
                            loss_s = loss_mse(pred_s, yb_std)
                            opt_stab.zero_grad()
                            loss_s.backward()
                            opt_stab.step()

                    # 3. Generative Optimization (Scalarized Multi-Objective)
                    # Objective: Maximize Reward - lambda * Stability
                    # We want high mean, low std.

                    vae.eval()
                    surrogate_reward.eval()
                    surrogate_stability.eval()

                    pop_size = GEN_POP_SIZE
                    z_pop = torch.randn(pop_size, self.latent_dim).to(self.device)

                    best_z = None
                    best_fitness = -float('inf')

                    lambda_stability = 1.0 # Penalty factor for variance

                    for gen in range(GEN_GENERATIONS):
                        with torch.no_grad():
                            decoded = vae.decode(z_pop)
                            pred_mean = surrogate_reward(decoded).squeeze()
                            pred_std = surrogate_stability(decoded).squeeze()

                            # Fitness = Mean - Stability (Lower std is better)
                            # Since inputs are normalized, this is roughly Sharpe-like
                            fitness = pred_mean - lambda_stability * pred_std

                        # Selection
                        sorted_idx = torch.argsort(fitness, descending=True)
                        elite = z_pop[sorted_idx[:pop_size//2]]

                        if fitness[sorted_idx[0]] > best_fitness:
                            best_fitness = fitness[sorted_idx[0]].item()
                            best_z = elite[0].clone()

                        # Mutation
                        offspring = elite.clone()
                        mutation = torch.randn_like(offspring) * 0.1
                        offspring = offspring + mutation

                        z_pop = torch.cat([elite, offspring], dim=0)
                        # Ensure population size stays constant (if odd division)
                        if z_pop.shape[0] > pop_size:
                            z_pop = z_pop[:pop_size]

                    # 4. Decode Best
                    with torch.no_grad():
                        best_vec_approx = vae.decode(best_z.unsqueeze(0)).cpu().numpy()[0]

                    # 5. Map to Nearest Real Vector (HNSW / NearestNeighbor)
                    # We map to the *entire* historical set of coords (coords_list)
                    # because the strategy might allow any valid parameter set,
                    # but usually we want one that we have data for to be "safe" or just snap to grid.
                    # Let's map to the *valid coords for this window* to ensure we pick something realistic.
                    # Or map to *all known coords* if we trust the surrogate extrapolation.
                    # Document says: "Map ... to the nearest real vector available in the dataset."

                    # We use all valid coords from the slice to ensure we pick something that was actually tested recently?
                    # No, the document implies finding the best structural match.
                    # Let's use `X_data` (valid coords in window).

                    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_data)
                    distances, indices = nbrs.kneighbors([best_vec_approx])

                    nearest_idx = indices[0][0]
                    nearest_real_vec = X_data[nearest_idx] # This is a float array of steps

                    # Convert back to params dict
                    # Note: X_data corresponds to valid_coords
                    # We need to map back to original coords_list index or just use the values.
                    # Since we have the values (steps), we can convert directly.

                    pred_params = {}
                    for i, col in enumerate(self.var_cols):
                        cfg = self.config[col]
                        step_val = int(round(nearest_real_vec[i]))
                        val_pred = cfg['min'] + step_val * cfg['step']
                        pred_params[col] = val_pred

                    return {
                        'pred_params': pred_params,
                        'score': best_fitness * t_mean_sigma + t_mean_mu, # Approx Denorm
                        'generations': GEN_GENERATIONS,
                        'latent_norm': float(torch.norm(best_z))
                    }

            return GenerativeOptimizer
        except Exception as e:
            print(f"Error loading Generative Models: {e}")
            return None
    return None

# --- Worker Function ---

def worker_predict_week(task_args):
    start_time = time.time()
    try:
        file_name = task_args['file_name']
        print(f"  [Worker {os.getpid()}] {file_name}: Processing started...", flush=True)

        history_best = task_args['history_best']
        slice_matrix = task_args['slice_matrix']
        global_param_config = task_args['config']
        var_cols = task_args['var_cols']
        flags = task_args['flags']
        coords_list = task_args.get('coords_list')

        results = {'control': None, 'darts': None, 'generative': None}

        # 0. Control Group
        try:
            control_model = ControlGroupForecaster(global_param_config, var_cols)
            results['control'] = control_model.predict(history_best)
        except: pass

        # 1. Darts
        DartsForecaster = get_darts_forecaster(flags)
        if DartsForecaster:
            try:
                darts_model = DartsForecaster(global_param_config, var_cols)
                results['darts'] = darts_model.predict(history_best)
            except Exception as e:
                print(f"  [Worker {os.getpid()}] {file_name}: Darts Error: {e}", flush=True)

        # 2. Generative Optimization
        GenerativeOptimizer = get_generative_optimizer(flags)
        if GenerativeOptimizer and slice_matrix is not None:
            try:
                print(f"  [Worker {os.getpid()}] {file_name}: Starting Generative Optimization...", flush=True)
                gen_model = GenerativeOptimizer(global_param_config, var_cols)
                # Pass full window for training
                win_len = slice_matrix.shape[1]
                results['generative'] = gen_model.optimize(slice_matrix, 0, win_len, coords_list)
            except Exception as e:
                print(f"  [Worker {os.getpid()}] {file_name}: Generative Error: {e}", flush=True)
                traceback.print_exc()

        duration = time.time() - start_time
        return {'file_name': file_name, 'results': results, 'duration': duration}

    except Exception as e:
        return {'file_name': task_args.get('file_name', 'unknown'), 'results': {}, 'error': str(e), 'duration': time.time() - start_time}

# --- Main Execution ---

def main():
    print("--- Strategy Predictability Program (Generative) ---")

    if not sys.stdin.isatty():
        try: target_dir = sys.stdin.readline().strip()
        except: target_dir = ""
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
            parts = base.split('.')
            start_date_str = parts[-3]
            return (int(start_date_str), base)
        except:
            return (float('inf'), base)

    csv_files.sort(key=sort_key)
    print(f"Found {len(csv_files)} files.")

    print("\n=== Definitions ===")
    print("Generative Optimization: Uses VAE + Surrogate Model + NSGA-II to synthesize optimal vectors.")
    print(f"TRAINING_WINDOW ({TRAINING_WINDOW}): Weeks used for training.")
    print("-" * 30)

    # Global Scan
    print("Loading files in parallel...")
    start_load = time.time()
    files_data = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_file_load, csv_files)
        for res in results:
            if res: files_data.append(res)

    print(f"Loaded {len(files_data)} files in {time.time() - start_load:.2f}s")
    if not files_data: return

    # Global Config
    print("Scanning global parameters...")
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

    # Preprocess
    print("Preprocessing data...")
    start_pre = time.time()
    def _pre_wrapper(fd): return preprocess_file_data(fd, global_param_config)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pre_results = list(executor.map(_pre_wrapper, files_data))
    for i, res in enumerate(pre_results): files_data[i].update(res)
    print(f"Preprocessed in {time.time() - start_pre:.2f}s")

    # Master Matrix
    print("Constructing Master Matrix...")
    start_matrix = time.time()

    all_coords_set = set()
    for fd in files_data:
        all_coords_set.update(fd['grid_summary'].keys())

    coords_list = sorted(list(all_coords_set))
    coord_to_idx = {c: i for i, c in enumerate(coords_list)}
    num_coords = len(coords_list)
    num_files = len(files_data)

    master_matrix = np.full((num_coords, num_files), -1000.0, dtype=np.float32)

    for t, fd in enumerate(files_data):
        grid = fd['grid_summary']
        for coord, res in grid.items():
            if coord in coord_to_idx:
                idx = coord_to_idx[coord]
                master_matrix[idx, t] = res

    print(f"Master Matrix: {num_coords} coords x {num_files} files (Took {time.time() - start_matrix:.2f}s)")

    # Prepare Tasks
    best_vectors_history = [fd['best_vector'] for fd in files_data]
    results = {'control': [], 'darts': [], 'generative': []}

    start_index = TRAINING_WINDOW + 1
    if start_index >= len(files_data): start_index = VECTOR_INPUT + 2
    if start_index >= len(files_data):
        print("Not enough files.")
        return

    tasks = []
    for i in range(start_index, len(files_data)):
        target_file_data = files_data[i]
        file_name = os.path.basename(target_file_data['path'])
        window_start = max(0, i - TRAINING_WINDOW)
        window_end = i

        tasks.append((i, {
            'file_name': file_name,
            'history_best': best_vectors_history[window_start:window_end],
            'slice_matrix': master_matrix[:, window_start:window_end],
            'config': global_param_config,
            'var_cols': var_cols,
            'coords_list': coords_list,
            'flags': {'HAS_DARTS': HAS_DARTS, 'HAS_TORCH': HAS_TORCH, 'HAS_SKLEARN': HAS_SKLEARN}
        }))

    print(f"Submitted {len(tasks)} tasks (Max Workers={MAX_WORKERS})...")

    # Execution
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
        future_to_idx = {executor.submit(worker_predict_week, t[1]): t[0] for t in tasks}
        files_processed = 0
        total = len(tasks)

        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            target_file_data = files_data[i]
            file_name = os.path.basename(target_file_data['path'])

            try:
                res = future.result()
                if 'error' in res:
                    print(f"Error {file_name}: {res['error']}")
                    continue

                preds = res['results']

                if preds.get('control'):
                    stats = lookup_stats(target_file_data['df'], preds['control'], global_param_config)
                    results['control'].append({'file': file_name, 'pred': preds['control'], 'stats': stats})

                if HAS_DARTS and preds.get('darts'):
                    stats = lookup_stats(target_file_data['df'], preds['darts'], global_param_config)
                    results['darts'].append({'file': file_name, 'pred': preds['darts'], 'stats': stats})

                gen_res = preds.get('generative')
                if gen_res:
                    pred_params = gen_res['pred_params']
                    stats = lookup_stats(target_file_data['df'], pred_params, global_param_config)
                    results['generative'].append({
                        'file': file_name,
                        'pred': pred_params,
                        'stats': stats,
                        'meta': gen_res
                    })
                else:
                    results['generative'].append({'file': file_name, 'stats': {'Result': 0, 'Profit': 0, 'found': False}, 'pred': {}, 'meta': {}})

                files_processed += 1
                if files_processed % MAX_WORKERS == 0:
                    print(f"Completed {files_processed}/{total} files...", flush=True)

            except Exception as e:
                print(f"Task Exception: {e}")
                traceback.print_exc()

    generate_html_report(results, target_dir)

def generate_diagnostics_section(results):
    html = """
    <div class="section">
        <h2>Master Diagnostics</h2>
        <p>Generative Model Performance vs Ground Truth.</p>
    """
    gen_data = results.get('generative', [])
    if gen_data:
        html += """
        <h3>Generative Optimization Diagnostics</h3>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th style="background-color: #e6f7ff;">Found?</th>
                    <th style="background-color: #e6f7ff;">Predicted Score</th>
                    <th style="background-color: #fff0f6;">Latent Norm</th>
                    <th style="background-color: #fff0f6;">Generations</th>
                    <th style="background-color: #f9f0ff;">Profit (Actual)</th>
                </tr>
            </thead>
            <tbody>
        """
        for d in gen_data:
            m = d.get('meta', {})
            found = d['stats']['found']
            score = m.get('score', 0)
            norm = m.get('latent_norm', 0)
            gens = m.get('generations', 0)
            profit = d['stats']['Profit']
            color = "green" if found else "red"

            html += f"""
            <tr>
                <td>{d['file']}</td>
                <td style="color:{color}; font-weight:bold;">{found}</td>
                <td style="background-color: #e6f7ff;">{score:.2f}</td>
                <td style="background-color: #fff0f6;">{norm:.2f}</td>
                <td style="background-color: #fff0f6;">{gens}</td>
                <td style="background-color: #f9f0ff;">{profit:.2f}</td>
            </tr>
            """
        html += "</tbody></table>"
    html += "</div>"
    return html

def generate_html_report(results, output_dir):
    print("Generating Report...")
    html_parts = ["""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Predictability Report - Generative</title>
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
        <h1>Strategy Predictability Report (Generative)</h1>
        <div class="section">
            <h2>Configuration</h2>
            <ul>
                <li><strong>Generative Optimization:</strong> VAE (Manifold) + Surrogate (Reward) + NSGA-II.</li>
                <li><strong>Population:</strong> """ + str(GEN_POP_SIZE) + """</li>
                <li><strong>Generations:</strong> """ + str(GEN_GENERATIONS) + """</li>
            </ul>
        </div>
    """]

    html_parts.append(generate_diagnostics_section(results))

    models = [('Control Group', 'control'), ('Darts', 'darts'), ('Generative Model', 'generative')]

    for title, key in models:
        data = results.get(key, [])
        if not data: continue

        labels = [d['file'] for d in data]
        actual_results = [d['stats']['Result'] for d in data]
        profits = [d['stats']['Profit'] for d in data]

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(range(len(labels)), actual_results, marker='o', color='blue', label='Actual Result')
        ax1.axhline(y=np.mean(actual_results), color='r', linestyle='--', label=f'Avg: {np.mean(actual_results):.2f}')
        ax1.set_title(f'{title} Result')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True)

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        img1_b64 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(range(len(labels)), profits, marker='o', color='green', label='Actual Profit')
        ax2.axhline(y=np.mean(profits), color='orange', linestyle='--', label=f'Avg: {np.mean(profits):.2f}')
        ax2.set_title(f'{title} Profit')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        img2_b64 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        html_parts.append(f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="metric">Avg Result: {np.mean(actual_results):.2f}</div>
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
                    <tr><th>File</th><th>Found?</th><th>Result</th><th>Profit</th><th>Params</th><th>Meta</th></tr>
                </thead>
                <tbody>
        """)

        for d in data:
            found = d['stats']['found']
            color = "green" if found else "red"
            params = ", ".join([f"{k}={v:.2f}" for k,v in d['pred'].items()])
            meta = ""
            if key == 'generative':
                m = d.get('meta', {})
                score = m.get('score', 0)
                norm = m.get('latent_norm', 0)
                meta = f"Score: {score:.1f}, Norm: {norm:.2f}"

            html_parts.append(f"""
                <tr>
                    <td>{d['file']}</td>
                    <td style="color:{color}">{found}</td>
                    <td>{d['stats']['Result']:.2f}</td>
                    <td>{d['stats']['Profit']:.2f}</td>
                    <td>{params}</td>
                    <td>{meta}</td>
                </tr>
            """)
        html_parts.append("</tbody></table></div>")

    html_parts.append("</body></html>")

    out_path = os.path.join(output_dir, "Predictability_Report_Generative.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("".join(html_parts))
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
