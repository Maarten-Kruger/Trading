
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
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

# Configuration Variables
RESULT_CUTOFF = 30
VECTOR_INPUT = 10

# --- Data Loading and Processing ---

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
    count = 0
    limit = 5000
    for point in product(*ranges):
        count += 1
        if count > limit:
            return False
        if point not in coord_map:
            return False
    return True

def load_data(target_dir):
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {target_dir}")
        return [], {}

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

    global_param_config = {}
    all_values = {}
    files_data = []

    print("Scanning files for global parameter ranges...")
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
        return [], {}

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

    return files_data, global_param_config

def get_best_vectors(files_data, global_param_config):
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
            # Safety limit
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

    return best_vectors

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

# --- Forecasting Models Interface ---

class ForecastingModel:
    def __init__(self, name):
        self.name = name

    def is_available(self):
        return True

    def run(self, best_vectors, global_param_config, files_data):
        raise NotImplementedError

# --- Darts Implementation ---

class DartsBase(ForecastingModel):
    def __init__(self, name, model_cls, **model_kwargs):
        super().__init__(name)
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.TimeSeries = None
        try:
            from darts import TimeSeries
            self.TimeSeries = TimeSeries
        except ImportError:
            pass

    def is_available(self):
        return self.model_cls is not None and self.TimeSeries is not None

    def run(self, best_vectors, global_param_config, files_data):
        if not self.is_available():
            return []

        start_index = VECTOR_INPUT + 1
        if start_index >= len(best_vectors):
            return []

        all_vars_union = sorted(list(global_param_config.keys()))

        # Build full matrix
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

        data_np = np.array(data_matrix)
        ts = self.TimeSeries.from_values(data_np)

        predictions = []

        print(f"Running {self.name}...")
        for i in range(start_index, len(best_vectors)):
            # Train on 0..i-1
            train_series = ts[:i]
            try:
                # Instantiate per loop to avoid state leakage and simulate strictly sequential prediction
                model = self.model_cls(**self.model_kwargs)
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
            except Exception as e:
                print(f"  Error in {self.name} at {i}: {e}")

        return predictions

class DartsRandomForest(DartsBase):
    def __init__(self):
        model_cls = None
        try:
            try:
                from darts.models import RandomForestModel as RandomForest
            except ImportError:
                from darts.models import RandomForest
            model_cls = RandomForest
        except ImportError:
            pass
        super().__init__("Darts RandomForest", model_cls, lags=VECTOR_INPUT, n_estimators=100, random_state=42)

class DartsExponentialSmoothing(DartsBase):
    def __init__(self):
        model_cls = None
        try:
            from darts.models import ExponentialSmoothing
            model_cls = ExponentialSmoothing
        except ImportError:
            pass
        super().__init__("Darts ExponentialSmoothing", model_cls)

class DartsNBEATS(DartsBase):
    def __init__(self):
        model_cls = None
        try:
            from darts.models import NBEATSModel
            model_cls = NBEATSModel
        except ImportError:
            pass
        # NBEATS requires input_chunk_length and output_chunk_length
        super().__init__("Darts N-BEATS", model_cls, input_chunk_length=VECTOR_INPUT, output_chunk_length=1, n_epochs=10, random_state=42)

# --- NeuralForecast Implementation ---

class NeuralForecastBase(ForecastingModel):
    def __init__(self, name, model_list):
        super().__init__(name)
        self.model_list = model_list
        self.NeuralForecast = None
        try:
            from neuralforecast import NeuralForecast
            self.NeuralForecast = NeuralForecast
        except ImportError:
            pass

    def is_available(self):
        return self.NeuralForecast is not None and self.model_list is not None and len(self.model_list) > 0

    def run(self, best_vectors, global_param_config, files_data):
        if not self.is_available():
            return []

        start_index = VECTOR_INPUT + 1
        if start_index >= len(best_vectors):
            return []

        all_vars_union = sorted(list(global_param_config.keys()))

        # Convert data to long format
        records = []
        for i, bv in enumerate(best_vectors):
            # Using dummy date
            ds_val = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
            for col in all_vars_union:
                val = bv['params'].get(col, global_param_config[col]['min'])
                # Normalize/Stepify
                cfg = global_param_config[col]
                if cfg['step'] > 0:
                    step_val = int(round((val - cfg['min']) / cfg['step']))
                else:
                    step_val = 0

                records.append({
                    'ds': ds_val,
                    'unique_id': col,
                    'y': step_val
                })

        full_df = pd.DataFrame(records)

        predictions = []
        print(f"Running {self.name}...")

        # We need to reuse the class to avoid confusing Python with imports inside methods
        NeuralForecast = self.NeuralForecast

        for i in range(start_index, len(best_vectors)):
            cutoff_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
            train_df = full_df[full_df['ds'] < cutoff_date]

            try:
                # Instantiate NeuralForecast
                # Note: models might need to be fresh or reset?
                # NeuralForecast(models=...) copies them?
                # Let's hope so or that fit resets.
                nf = NeuralForecast(models=self.model_list, freq='D')
                nf.fit(df=train_df)
                fcst = nf.predict()

                # fcst has columns: ds, unique_id, ModelName...
                fcst_cols = [c for c in fcst.columns if c not in ['ds', 'unique_id']]
                if not fcst_cols:
                     # Fallback?
                     raise Exception("No forecast column found")
                res_col = fcst_cols[0]

                pred_params = {}
                for uid in all_vars_union:
                    row = fcst[fcst['unique_id'] == uid]
                    if row.empty:
                        val = 0
                    else:
                        val = row.iloc[0][res_col]

                    # Denormalize
                    cfg = global_param_config[uid]
                    step_pred = int(round(val))
                    val_pred = cfg['min'] + step_pred * cfg['step']
                    pred_params[uid] = val_pred

                stats = lookup_stats(files_data[i]['df'], pred_params)
                predictions.append({
                    'file_index': i,
                    'file_name': os.path.basename(best_vectors[i]['file']),
                    'predicted_params': pred_params,
                    'actual_result': stats['Result'],
                    'actual_profit': stats['Profit'],
                    'found': stats['found']
                })

            except Exception as e:
                print(f"  Error in {self.name} at {i}: {e}")

        return predictions

class NeuralForecastLSTM(NeuralForecastBase):
    def __init__(self):
        models = []
        try:
            from neuralforecast.models import LSTM
            # h=1 for 1 step ahead
            models = [LSTM(h=1, input_size=VECTOR_INPUT, max_steps=100)]
        except ImportError:
            pass
        super().__init__("NeuralForecast LSTM", models)

class NeuralForecastNHITS(NeuralForecastBase):
    def __init__(self):
        models = []
        try:
            from neuralforecast.models import NHITS
            models = [NHITS(h=1, input_size=VECTOR_INPUT, max_steps=100)]
        except ImportError:
            pass
        super().__init__("NeuralForecast NHITS", models)


# --- Report Generation ---

def generate_html_report(results, output_dir):
    if not results:
        print("No results to report.")
        return

    html_parts = []

    html_header = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Predictability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .model-section {{ margin-bottom: 60px; border-top: 2px solid #333; padding-top: 20px; }}
            .plot {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Strategy Predictability Report</h1>
        <p>Prediction Input Vector Size: {VECTOR_INPUT}</p>
    """
    html_parts.append(html_header)

    for model_name, predictions in results.items():
        if not predictions:
            html_parts.append(f"<div class='model-section'><h2>{model_name}</h2><p>No predictions generated (missing dependencies or data).</p></div>")
            continue

        labels = [p['file_name'] for p in predictions]
        results_vals = [p['actual_result'] for p in predictions]
        profits = [p['actual_profit'] for p in predictions]
        found_status = [p['found'] for p in predictions]
        x = range(len(labels))

        # Graph 1: Result
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x, results_vals, marker='o', color='blue', label='Actual Result')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Result')
        ax1.set_title(f'{model_name}: Result Performance')
        ax1.grid(True)
        for i, found in enumerate(found_status):
            if not found:
                ax1.text(i, results_vals[i], 'Missing', color='red', ha='center', va='bottom', fontsize=8)

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        # Graph 2: Profit
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x, profits, marker='o', color='green', label='Actual Profit')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Profit')
        ax2.set_title(f'{model_name}: Profit Performance')
        ax2.grid(True)
        for i, found in enumerate(found_status):
            if not found:
                ax2.text(i, profits[i], 'Missing', color='red', ha='center', va='bottom', fontsize=8)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        # Table
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Found?</th>
                    <th>Actual Result</th>
                    <th>Actual Profit</th>
                    <th>Predicted Params</th>
                </tr>
            </thead>
            <tbody>
        """
        for p in predictions:
            p_str = ", ".join([f"{k}={v:.2f}" for k,v in p['predicted_params'].items()])
            table_html += f"""
                <tr>
                    <td>{p['file_name']}</td>
                    <td style="color: {'green' if p['found'] else 'red'}">{p['found']}</td>
                    <td>{p['actual_result']:.2f}</td>
                    <td>{p['actual_profit']:.2f}</td>
                    <td>{p_str}</td>
                </tr>
            """
        table_html += "</tbody></table>"

        section = f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            <div class="plot">
                <img src="data:image/png;base64,{img1}" />
            </div>
            <div class="plot">
                <img src="data:image/png;base64,{img2}" />
            </div>
            {table_html}
        </div>
        """
        html_parts.append(section)

    html_parts.append("</body></html>")

    out_path = os.path.join(output_dir, "Predictability_Report.html")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    print(f"Report saved to {out_path}")

# --- Main ---

def main():
    print("--- Strategy Predictability Program ---")

    # 1. User Input
    if len(sys.argv) > 1:
         # Support passing dir as argument
         target_dir = sys.argv[1]
    elif not sys.stdin.isatty():
        try:
            target_dir = sys.stdin.readline().strip()
        except:
            target_dir = ""
    else:
        target_dir = input("Please enter the path to the folder with the CSV files: ").strip()

    if not target_dir or not os.path.exists(target_dir):
        print(f"Invalid directory: {target_dir}")
        return

    # 2. Load Data
    files_data, global_param_config = load_data(target_dir)
    if not files_data:
        return

    # 3. Best Vectors
    best_vectors = get_best_vectors(files_data, global_param_config)
    if len(best_vectors) <= VECTOR_INPUT:
        print(f"Not enough data to predict. Need > {VECTOR_INPUT} files.")
        return

    # 4. Run Models
    models = [
        DartsRandomForest(),
        DartsExponentialSmoothing(),
        DartsNBEATS(),
        NeuralForecastLSTM(),
        NeuralForecastNHITS()
    ]

    results = {}
    for model in models:
        if model.is_available():
            results[model.name] = model.run(best_vectors, global_param_config, files_data)
        else:
            print(f"Model {model.name} is not available (dependencies missing).")

    # 5. Report
    generate_html_report(results, target_dir)

if __name__ == "__main__":
    main()
