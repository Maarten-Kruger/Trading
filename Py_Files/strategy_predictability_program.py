# To run this, just type in the terminal:
# .\venv\Scripts\activate  (when you are cd C:\trading_bot) and then
# python strategy_predictability_program.py

#or

# .\venv\Scripts\python.exe strategy_predictability_program.py

import pandas as pd
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import io
import base64
import warnings
from itertools import product
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Darts imports
try:
    from darts import TimeSeries
    try:
        from darts.models import RandomForestModel as RandomForest
    except ImportError:
        from darts.models import RandomForest
    from darts.models import LinearRegressionModel, NBEATSModel
except ImportError:
    print("Darts not found. Please install it: pip install darts")
    # sys.exit(1) # Don't exit immediately to allow other imports check

# NeuralForecast imports
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, LSTM
except ImportError:
    print("NeuralForecast not found. Please install it: pip install neuralforecast")

# Configuration Variables
RESULT_CUTOFF = 30
VECTOR_INPUT = 10

def main():
    print("--- Strategy Predictability Program ---")

    # 1. User Input
    if not sys.stdin.isatty():
        try:
            target_dir = sys.stdin.readline().strip()
        except:
            target_dir = ""
    else:
        target_dir = input("Please enter the path to the folder with the CSV files: ").strip()

    if not target_dir:
        print("No directory provided.")
        return
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found.")
        return

    # 2. Load and Sort Files
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {target_dir}")
        return

    # Sort files by integer prefix
    def sort_key(file_path):
        base = os.path.basename(file_path)
        try:
            val = int(base.split('_')[0])
            return (val, base)
        except (ValueError, IndexError):
            return (float('inf'), base)

    csv_files.sort(key=sort_key)
    print(f"Found {len(csv_files)} files.")

    # 3. Global Parameter Analysis & Best Vector finding
    files_data, best_vectors, global_param_config = analyze_and_find_best(csv_files)

    if not best_vectors:
        print("No best vectors found.")
        return

    # 4. Prepare Data for Forecasting
    start_index = VECTOR_INPUT + 1
    if start_index >= len(best_vectors):
        print(f"Not enough data to predict. Need at least {start_index + 1} files (VECTOR_INPUT={VECTOR_INPUT}), have {len(best_vectors)}.")
        return

    all_vars_union = sorted(list(global_param_config.keys()))

    # Build full matrix (List of lists of integers)
    data_matrix = []
    for bv in best_vectors:
        row_vec = []
        for col in all_vars_union:
            val = bv['params'].get(col, global_param_config[col]['min'])
            cfg = global_param_config[col]
            if cfg['step'] > 0:
                step_val = int(round((val - cfg['min']) / cfg['step']))
            else:
                step_val = 0
            row_vec.append(step_val)
        data_matrix.append(row_vec)

    # 5. Define Models
    models_to_run = [
        {
            'name': 'Random Forest (Darts)',
            'type': 'darts',
            'model_cls': RandomForest,
            'kwargs': {'lags': VECTOR_INPUT, 'n_estimators': 100, 'random_state': 42}
        },
        {
            'name': 'Linear Regression (Darts)',
            'type': 'darts',
            'model_cls': LinearRegressionModel,
            'kwargs': {'lags': VECTOR_INPUT}
        },
        {
            'name': 'NBEATS (Darts)',
            'type': 'darts',
            'model_cls': NBEATSModel,
            'kwargs': {'input_chunk_length': VECTOR_INPUT, 'output_chunk_length': 1, 'n_epochs': 20, 'random_state': 42}
        },
        {
            'name': 'NeuralForecast (NHITS)',
            'type': 'neuralforecast',
            'model_cls': NHITS,
            'kwargs': {
                'h': 1,
                'input_size': VECTOR_INPUT,
                'max_steps': 50,
                'enable_checkpointing': False,
                'logger': False
            }
        }
    ]

    all_predictions = {}

    for m in models_to_run:
        print(f"\nRunning Model: {m['name']}...")
        try:
            if m['type'] == 'darts':
                preds = run_darts_forecast(m, data_matrix, best_vectors, files_data, start_index, all_vars_union, global_param_config)
            elif m['type'] == 'neuralforecast':
                preds = run_neuralforecast_loop(m, data_matrix, best_vectors, files_data, start_index, all_vars_union, global_param_config)
            else:
                preds = []

            all_predictions[m['name']] = preds
        except Exception as e:
            print(f"Failed to run {m['name']}: {e}")
            import traceback
            traceback.print_exc()

    # 6. Generate Report
    generate_html_report(all_predictions, target_dir)


def analyze_and_find_best(csv_files):
    print("Scanning files for global parameter ranges...")
    global_param_config = {}
    all_values = {}
    files_data = []

    for file_path in csv_files:
        try:
            df = read_csv_robust(file_path)
            if "Trades" not in df.columns:
                print(f"Skipping {os.path.basename(file_path)}: 'Trades' column not found.")
                continue

            trades_idx = df.columns.get_loc("Trades")
            var_cols = df.columns[trades_idx+1:].tolist()

            files_data.append({
                'path': file_path,
                'df': df,
                'var_cols': var_cols
            })

            for col in var_cols:
                if col not in all_values:
                    all_values[col] = set()
                vals = df[col].dropna().unique()
                all_values[col].update(vals)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not files_data:
        print("No valid data loaded.")
        return [], [], {}

    sorted_vars = sorted(list(all_values.keys()))
    for col in sorted_vars:
        vals = sorted(list(all_values[col]))
        if len(vals) < 2:
            step = 0
        else:
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            step = np.min(diffs) if len(diffs) > 0 else 0

        global_param_config[col] = {
            'min': vals[0],
            'step': step,
            'max': vals[-1],
        }

    # Find Best Vector per File
    print("Finding best vectors...")
    best_vectors = []

    for fdata in files_data:
        df = fdata['df']
        var_cols = fdata['var_cols']
        active_vars = [c for c in var_cols if c in global_param_config and global_param_config[c]['step'] > 0]

        filtered_df = df[df['Result'] >= RESULT_CUTOFF].copy()

        if filtered_df.empty or not active_vars:
            # Fallback
            best_idx = df['Result'].idxmax()
            best_row = df.loc[best_idx]
            best_vectors.append({
                'file': fdata['path'],
                'row': best_row,
                'params': best_row[var_cols].to_dict(),
                'radius': 0
            })
            continue

        # Coordinate Map
        coord_map = {}
        for idx, row in filtered_df.iterrows():
            coord = []
            for col in active_vars:
                val = row[col]
                cfg = global_param_config[col]
                if cfg['step'] > 0:
                    c_val = int(round((val - cfg['min']) / cfg['step']))
                else:
                    c_val = 0
                coord.append(c_val)
            coord_map[tuple(coord)] = idx

        # Find Max Radius
        best_candidate = None
        max_radius = -1
        filtered_df = filtered_df.sort_values('Result', ascending=False)

        for idx, row in filtered_df.iterrows():
            center_coord = []
            for col in active_vars:
                val = row[col]
                cfg = global_param_config[col]
                if cfg['step'] > 0:
                    c_val = int(round((val - cfg['min']) / cfg['step']))
                else:
                    c_val = 0
                center_coord.append(c_val)
            center_coord = tuple(center_coord)

            r = 0
            while r < 20:
                next_r = r + 1
                if check_hypercube(center_coord, next_r, coord_map, len(active_vars)):
                    r = next_r
                else:
                    break

            if r > max_radius:
                max_radius = r
                best_candidate = row

        if best_candidate is None:
             best_candidate = filtered_df.iloc[0]
             max_radius = 0

        best_vectors.append({
            'file': fdata['path'],
            'row': best_candidate,
            'params': best_candidate[var_cols].to_dict(),
            'radius': max_radius
        })

        print(f"  {os.path.basename(fdata['path'])}: Best Vector Radius {max_radius}, Result {best_candidate['Result']:.2f}")

    return files_data, best_vectors, global_param_config


def run_darts_forecast(model_info, data_matrix, best_vectors, files_data, start_index, all_vars_union, global_param_config):
    data_np = np.array(data_matrix)
    ts = TimeSeries.from_values(data_np)

    model_cls = model_info['model_cls']
    kwargs = model_info['kwargs']

    # Initialize model once if it supports incremental training?
    # Actually, for walk forward, we re-train or update.
    # RandomForest in Darts fits on the whole series passed to fit().

    predictions = []

    for i in range(start_index, len(best_vectors)):
        train_series = ts[:i]

        model = model_cls(**kwargs)

        try:
            model.fit(train_series)
            pred = model.predict(n=1)
            pred_vals = pred.values()[0]

            pred_params = {}
            for idx, col in enumerate(all_vars_union):
                cfg = global_param_config[col]
                step_pred = int(round(pred_vals[idx]))
                val_pred = cfg['min'] + step_pred * cfg['step']
                pred_params[col] = val_pred

            stats = lookup_stats(files_data[i]['df'], pred_params)

            predictions.append({
                'file_index': i,
                'file_name': os.path.basename(best_vectors[i]['file']),
                'predicted_params': pred_params,
                'actual_result': stats['Result'],
                'actual_profit': stats['Profit'],
                'found': stats['found']
            })
            print(f"  [{model_info['name']}] Pred for {os.path.basename(best_vectors[i]['file'])}: Found={stats['found']}, Profit={stats['Profit']:.2f}")

        except Exception as e:
            print(f"  Error at {i}: {e}")

    return predictions


def run_neuralforecast_loop(model_info, data_matrix, best_vectors, files_data, start_index, all_vars_union, global_param_config):
    # Convert data_matrix to long format DataFrame
    # unique_id = param_name
    # ds = index (0..T)
    # y = value

    records = []
    for t_idx, row in enumerate(data_matrix):
        for col_idx, val in enumerate(row):
            records.append({
                'unique_id': all_vars_union[col_idx],
                'ds': t_idx,
                'y': float(val)
            })

    full_df = pd.DataFrame(records)

    predictions = []

    for i in range(start_index, len(best_vectors)):
        # Train on 0..i-1
        train_df = full_df[full_df['ds'] < i].copy()

        # Instantiate NF model
        # We need to re-instantiate because NF fits internally
        model_cls = model_info['model_cls']
        kwargs = model_info['kwargs']

        # NF requires a list of models
        nf_model = model_cls(**kwargs)
        nf = NeuralForecast(models=[nf_model], freq=1) # freq=1 integer

        try:
            nf.fit(df=train_df)
            forecast_df = nf.predict() # Defaults to h=1 as set in kwargs or default

            # forecast_df has index unique_id, columns ds, ModelName
            # We want ds = i

            pred_params = {}

            # Reset index to make unique_id a column
            forecast_df = forecast_df.reset_index()

            # Check if 'ds' is in columns, sometimes it's index?
            # NF predict usually returns DataFrame with unique_id as index or column depending on version.
            # In 3.1.4, it typically returns unique_id as index.

            # We want the prediction for ds = i.
            # predict() without 'futr_df' predicts h steps ahead from end of train_df.
            # Since train_df ends at i-1, predict h=1 gives ds=i.

            # Model column name is usually class name, e.g. NHITS
            model_col_name = model_cls.__name__

            for col in all_vars_union:
                # Get prediction for this param
                row = forecast_df[forecast_df['unique_id'] == col]
                if not row.empty:
                    val = row.iloc[0][model_col_name]
                    cfg = global_param_config[col]
                    step_pred = int(round(val))
                    val_pred = cfg['min'] + step_pred * cfg['step']
                    pred_params[col] = val_pred
                else:
                    pred_params[col] = global_param_config[col]['min'] # Fallback

            stats = lookup_stats(files_data[i]['df'], pred_params)

            predictions.append({
                'file_index': i,
                'file_name': os.path.basename(best_vectors[i]['file']),
                'predicted_params': pred_params,
                'actual_result': stats['Result'],
                'actual_profit': stats['Profit'],
                'found': stats['found']
            })
            print(f"  [{model_info['name']}] Pred for {os.path.basename(best_vectors[i]['file'])}: Found={stats['found']}, Profit={stats['Profit']:.2f}")

        except Exception as e:
            print(f"  Error NeuralForecast at {i}: {e}")
            import traceback
            traceback.print_exc()

    return predictions

def read_csv_robust(filepath):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
    except:
        df = pd.read_csv(filepath, sep=';', engine='python')

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                series = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(series)
            except ValueError:
                pass
    return df

def check_hypercube(center, radius, coord_map, dim):
    ranges = [range(c - radius, c + radius + 1) for c in center]
    # Limit number of checks
    count = 0
    limit = 5000
    for point in product(*ranges):
        count += 1
        if count > limit:
            return False
        if point not in coord_map:
            return False
    return True

def lookup_stats(df, params):
    mask = np.ones(len(df), dtype=bool)
    for col, val in params.items():
        if col in df.columns:
            mask = mask & np.isclose(df[col], val, atol=1e-7)
    matches = df[mask]
    if matches.empty:
        return {'Result': 0, 'Profit': 0, 'found': False}
    row = matches.iloc[0]
    return {'Result': row['Result'], 'Profit': row['Profit'], 'found': True}

def generate_html_report(all_predictions, output_dir):
    if not all_predictions:
        print("No predictions to report.")
        return

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Predictability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .model-section {{ margin-bottom: 50px; border-bottom: 2px solid #333; padding-bottom: 20px; }}
            .plot {{ display: inline-block; margin: 10px; border: 1px solid #ddd; padding: 10px; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Strategy Predictability Report</h1>
        <p>Prediction Input Vector Size: {VECTOR_INPUT}</p>
    """

    for model_name, predictions in all_predictions.items():
        if not predictions:
            html += f"<div class='model-section'><h2>{model_name}</h2><p>No predictions generated.</p></div>"
            continue

        labels = [p['file_name'] for p in predictions]
        results = [p['actual_result'] for p in predictions]
        profits = [p['actual_profit'] for p in predictions]
        found_status = [1 if p['found'] else 0 for p in predictions]
        x = range(len(labels))

        # Graph 1: Result
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x, results, marker='o', color='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Result')
        ax1.set_title(f'{model_name}: Result Performance')
        ax1.grid(True)
        for i, found in enumerate(found_status):
            if not found:
                ax1.text(i, results[i], 'Missing', color='red', ha='center', va='bottom')

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        # Graph 2: Profit
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(x, profits, marker='o', color='green')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Profit')
        ax2.set_title(f'{model_name}: Profit Performance')
        ax2.grid(True)
        for i, found in enumerate(found_status):
            if not found:
                ax2.text(i, profits[i], 'Missing', color='red', ha='center', va='bottom')

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        html += f"""
        <div class="model-section">
            <h2>Model: {model_name}</h2>
            <div class="plot">
                <img src="data:image/png;base64,{img1}" />
            </div>
            <div class="plot">
                <img src="data:image/png;base64,{img2}" />
            </div>

            <h3>Prediction Log</h3>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Found in Opt?</th>
                        <th>Actual Result</th>
                        <th>Actual Profit</th>
                        <th>Predicted Params</th>
                    </tr>
                </thead>
                <tbody>
        """

        for p in predictions:
            p_str = ", ".join([f"{k}={v:.2f}" for k,v in p['predicted_params'].items()])
            html += f"""
                    <tr>
                        <td>{p['file_name']}</td>
                        <td style="color: {'green' if p['found'] else 'red'}">{p['found']}</td>
                        <td>{p['actual_result']:.2f}</td>
                        <td>{p['actual_profit']:.2f}</td>
                        <td>{p_str}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

    html += """
    </body>
    </html>
    """

    out_path = os.path.join(output_dir, "Predictability_Report.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
