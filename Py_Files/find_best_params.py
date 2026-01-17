import pandas as pd
import numpy as np
import os
import glob
import math
from itertools import product
import matplotlib.pyplot as plt
import io
import base64

# Configuration
INPUT_DIR = "OptimizationResults"  # Relative to this script, or absolute
REQUIRED_GRID_SIDE = 5  # Corresponds to Radius = 1 (1 step up, 1 step down) -> 3 points total per axis

def main():
    # Determine the directory relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, INPUT_DIR)

    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found.")
        return

    excel_files = glob.glob(os.path.join(target_dir, "*.xlsx"))
    if not excel_files:
        print(f"No Excel files found in {target_dir}")
        return

    # Sort files by integer prefix before processing (or after, but sorting file list is good practice)
    def sort_key(file_path):
        base = os.path.basename(file_path)
        try:
            # Try to extract the first integer
            val = int(base.split('_')[0])
            return (val, base)
        except (ValueError, IndexError):
            # Fallback for files without a prefix: put them at the end
            return (float('inf'), base)

    excel_files.sort(key=sort_key)

    results_table = []

    for file_path in excel_files:
        try:
            process_file(file_path, results_table)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print summary table to console
    print("\n" + "="*120)
    print(f"{'File':<30} | {'Best Result':<11} | {'Profit':<10} | {'Radius':<6} | {'Neigh. Profits':<18} | {'Variables (Value, Step)'}")
    print("-" * 120)
    for res in results_table:
        file_name = os.path.basename(res['file'])
        best_res = f"{res['result']:.2f}"
        profit = f"{res['profit']:.2f}"
        radius = str(res['radius'])

        # Neighbor Profits
        if res['radius'] != "N/A (Fixed)" and res['min_neigh_profit'] is not None:
             neigh_prof = f"{res['min_neigh_profit']:.0f} - {res['max_neigh_profit']:.0f}"
        else:
             neigh_prof = "N/A"

        # Format variables string
        vars_parts = []
        for k, v in res['params'].items():
            step = res['step_sizes'].get(k, 0)
            if step > 0:
                vars_parts.append(f"{k}={v} (±{step:.0f})")
            else:
                vars_parts.append(f"{k}={v}")

        vars_str = ", ".join(vars_parts)
        print(f"{file_name[:29]:<30} | {best_res:<11} | {profit:<10} | {radius:<6} | {neigh_prof:<18} | {vars_str}")
    print("="*120 + "\n")

    export_to_html(results_table)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def process_file(file_path, results_table):
    print(f"Processing {file_path}...")

    try:
        df = pd.read_excel(file_path)
    except Exception:
        df = pd.read_excel(file_path, engine='openpyxl')

    # Convert comma decimals to float for all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Replace comma with dot and convert
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            except ValueError:
                pass

    # Identify variable columns
    try:
        trades_idx = df.columns.get_loc("Trades")
        var_cols = df.columns[trades_idx+1:].tolist()
    except KeyError:
        print("Column 'Trades' not found. Skipping file.")
        return

    if not var_cols:
        print("No variable columns found.")
        return

    # Determine step sizes and identify varying columns
    step_sizes = {}
    varying_cols = []

    for col in var_cols:
        unique_vals = sorted(df[col].unique())
        if len(unique_vals) <= 1:
            step_sizes[col] = 0.0
            continue

        diffs = np.diff(unique_vals)
        diffs = diffs[diffs > 1e-9]

        if len(diffs) == 0:
            step_sizes[col] = 0.0
            continue

        step_val = np.min(diffs)
        step_sizes[col] = step_val
        varying_cols.append(col)

    # Generate Per-File Plots
    scatter_b64 = None
    violin_b64 = None

    if varying_cols:
        num_vars = len(varying_cols)
        cols = min(num_vars, 3)
        rows = math.ceil(num_vars / cols)

        # Scatter Plots
        fig_scatter, axes_scatter = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_vars == 1: axes_scatter = [axes_scatter]
        elif isinstance(axes_scatter, np.ndarray): axes_scatter = axes_scatter.flatten()
        else: axes_scatter = [axes_scatter]

        for i, col in enumerate(varying_cols):
            ax = axes_scatter[i]
            ax.scatter(df[col], df['Profit'], alpha=0.5, s=10, c='royalblue', edgecolors='none')
            ax.set_title(f"Profit vs {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Profit")
            ax.grid(True, linestyle=':', alpha=0.6)

        for j in range(i+1, len(axes_scatter)): axes_scatter[j].axis('off')
        plt.tight_layout()
        scatter_b64 = plot_to_base64(fig_scatter)
        plt.close(fig_scatter)

        # Violin Plots
        fig_violin, axes_violin = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if num_vars == 1: axes_violin = [axes_violin]
        elif isinstance(axes_violin, np.ndarray): axes_violin = axes_violin.flatten()
        else: axes_violin = [axes_violin]

        for i, col in enumerate(varying_cols):
            ax = axes_violin[i]
            unique_vals = sorted(df[col].unique())
            data_to_plot = [df[df[col] == val]['Profit'].values for val in unique_vals]

            parts = ax.violinplot(data_to_plot, showmeans=False, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('indianred')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

            ax.set_xticks(range(1, len(unique_vals) + 1))
            ax.set_xticklabels([str(v) for v in unique_vals], rotation=45, ha='right')
            ax.set_title(f"Profit Dist. by {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Profit")
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)

        for j in range(i+1, len(axes_violin)): axes_violin[j].axis('off')
        plt.tight_layout()
        violin_b64 = plot_to_base64(fig_violin)
        plt.close(fig_violin)


    # Robustness Optimization Logic
    if not varying_cols:
        best_idx = df['Result'].idxmax()
        row = df.loc[best_idx]
        results_table.append({
            'file': file_path,
            'result': row['Result'],
            'profit': row['Profit'],
            'params': row[var_cols].to_dict(),
            'step_sizes': step_sizes,
            'radius': "N/A (Fixed)",
            'min_neigh_profit': None,
            'max_neigh_profit': None,
            'scatter_plot': scatter_b64,
            'violin_plot': violin_b64
        })
        return

    # Normalize coordinates
    coord_map = {}
    min_vals = {col: df[col].min() for col in varying_cols}

    for idx, row in df.iterrows():
        coords = []
        for col in varying_cols:
            val = row[col]
            mn = min_vals[col]
            st = step_sizes[col]
            c = int(round((val - mn) / st))
            coords.append(c)

        coord_tuple = tuple(coords)
        coord_map[coord_tuple] = {
            'Profit': row['Profit'],
            'Result': row['Result'],
            'Index': idx,
            'Row': row
        }

    # Sort candidates by Result descending
    sorted_candidates = sorted(coord_map.items(), key=lambda x: x[1]['Result'], reverse=True)

    target_radius = (REQUIRED_GRID_SIDE - 1) // 2

    best_candidate = None

    # 1. Find Best Result with Radius >= 1
    for coord, data in sorted_candidates:
        if check_radius(coord, target_radius, varying_cols, coord_map):
            max_r = measure_max_radius(coord, varying_cols, coord_map)
            min_p, max_p = get_radius_stats(coord, max_r, varying_cols, coord_map)

            best_candidate = {
                'file': file_path,
                'result': data['Result'],
                'profit': data['Profit'],
                'params': data['Row'][var_cols].to_dict(),
                'step_sizes': step_sizes,
                'radius': max_r,
                'min_neigh_profit': min_p,
                'max_neigh_profit': max_p,
                'scatter_plot': scatter_b64,
                'violin_plot': violin_b64
            }
            break

    if best_candidate:
        results_table.append(best_candidate)
        return

    # 2. Fallback: Find row with Max Radius
    print("  No candidate met target radius. Searching for max radius...")

    max_found_radius = -1
    best_fallback = None

    for coord, data in sorted_candidates:
        r = measure_max_radius(coord, varying_cols, coord_map)

        if r > max_found_radius:
            max_found_radius = r
            min_p, max_p = get_radius_stats(coord, r, varying_cols, coord_map)
            best_fallback = {
                'file': file_path,
                'result': data['Result'],
                'profit': data['Profit'],
                'params': data['Row'][var_cols].to_dict(),
                'step_sizes': step_sizes,
                'radius': r,
                'min_neigh_profit': min_p,
                'max_neigh_profit': max_p,
                'scatter_plot': scatter_b64,
                'violin_plot': violin_b64
            }
        elif r == max_found_radius:
             if best_fallback is None:
                min_p, max_p = get_radius_stats(coord, r, varying_cols, coord_map)
                best_fallback = {
                    'file': file_path,
                    'result': data['Result'],
                    'profit': data['Profit'],
                    'params': data['Row'][var_cols].to_dict(),
                    'step_sizes': step_sizes,
                    'radius': r,
                    'min_neigh_profit': min_p,
                    'max_neigh_profit': max_p,
                    'scatter_plot': scatter_b64,
                    'violin_plot': violin_b64
                }

    if best_fallback:
        results_table.append(best_fallback)
    else:
        print("  No suitable data found.")

def check_radius(center_coord, radius, varying_cols, coord_map):
    if radius == 0:
        return coord_map[center_coord]['Profit'] > 0

    ranges = [range(c - radius, c + radius + 1) for c in center_coord]

    for neighbor in product(*ranges):
        if neighbor not in coord_map:
            return False
        if coord_map[neighbor]['Profit'] <= 0:
            return False
    return True

def measure_max_radius(center_coord, varying_cols, coord_map):
    if center_coord not in coord_map or coord_map[center_coord]['Profit'] <= 0:
        return -1

    radius = 1
    while True:
        if check_radius(center_coord, radius, varying_cols, coord_map):
            radius += 1
        else:
            return radius - 1

def get_radius_stats(center_coord, radius, varying_cols, coord_map):
    if radius < 0: return None, None
    if radius == 0:
        p = coord_map[center_coord]['Profit']
        return p, p
    ranges = [range(c - radius, c + radius + 1) for c in center_coord]
    profits = []
    for neighbor in product(*ranges):
        if neighbor in coord_map:
            profits.append(coord_map[neighbor]['Profit'])
    if not profits: return None, None
    return min(profits), max(profits)

def export_to_html(results_table, filename="Optimization_Report.html"):
    if not results_table:
        print("No results to export.")
        return

    # 1. Generate Evolution Plot (Overall)
    all_vars = set()
    for res in results_table:
        all_vars.update(res['params'].keys())
    all_vars = sorted(list(all_vars))

    evolution_plot_b64 = None
    if all_vars:
        num_vars = len(all_vars)
        cols = 2
        rows = (num_vars + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))
        if num_vars == 1: axes = [axes]
        elif isinstance(axes, np.ndarray): axes = axes.flatten()
        else: axes = [axes]

        file_names = [os.path.basename(res['file']) for res in results_table]
        x_indices = range(len(file_names))

        for i, var in enumerate(all_vars):
            ax = axes[i]
            values, lower_bounds, upper_bounds, valid_indices = [], [], [], []
            for idx, res in enumerate(results_table):
                if var in res['params']:
                    val = res['params'][var]
                    step = res['step_sizes'].get(var, 0)
                    r = res['radius'] if isinstance(res['radius'], (int, float)) else 0
                    values.append(val)
                    lower_bounds.append(val - r * step)
                    upper_bounds.append(val + r * step)
                    valid_indices.append(idx)

            if valid_indices:
                ax.plot(valid_indices, values, marker='o', label='Best Value', color='blue')
                ax.fill_between(valid_indices, lower_bounds, upper_bounds, color='blue', alpha=0.2, label='Robust Region')
                ax.set_xticks(valid_indices)
                ax.set_xticklabels([file_names[j] for j in valid_indices], rotation=45, ha='right', fontsize=8)
                ax.set_title(f"Variable: {var}")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        evolution_plot_b64 = plot_to_base64(fig)
        plt.close(fig)

    # 2. Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Optimization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .plot-box {{ flex: 1 1 45%; min-width: 400px; }}
            .variables {{ white-space: pre-wrap; font-family: monospace; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Optimization Results Summary</h1>

        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Best Result</th>
                    <th>Profit</th>
                    <th>Radius</th>
                    <th>Neighbor Profits</th>
                    <th>Variables (Value, Step)</th>
                </tr>
            </thead>
            <tbody>
    """

    for res in results_table:
        file_name = os.path.basename(res['file'])
        radius = str(res['radius'])
        if res['radius'] != "N/A (Fixed)" and res['min_neigh_profit'] is not None:
             neigh_prof = f"{res['min_neigh_profit']:.0f} - {res['max_neigh_profit']:.0f}"
        else:
             neigh_prof = "N/A"

        vars_parts = []
        for k, v in res['params'].items():
            step = res['step_sizes'].get(k, 0)
            if step > 0:
                vars_parts.append(f"{k}={v} (±{step:.0f})")
            else:
                vars_parts.append(f"{k}={v}")
        vars_str = "\\n".join(vars_parts)

        html_content += f"""
                <tr>
                    <td>{file_name}</td>
                    <td>{res['result']:.2f}</td>
                    <td>{res['profit']:.2f}</td>
                    <td>{radius}</td>
                    <td>{neigh_prof}</td>
                    <td class="variables">{vars_str}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>

        <div class="section">
            <h2>Parameter Evolution (Robustness)</h2>
            <p>Tracking the optimal parameter value (dot) and its robust profitability range (shaded) across files.</p>
    """
    if evolution_plot_b64:
        html_content += f'<img src="data:image/png;base64,{evolution_plot_b64}" alt="Parameter Evolution Plot">'
    else:
        html_content += "<p>No variable data available for evolution plot.</p>"

    html_content += """
        </div>

        <div class="section">
            <h2>Detailed Analysis per File</h2>
    """

    for res in results_table:
        file_name = os.path.basename(res['file'])
        html_content += f"""
            <div class="section">
                <h3>{file_name}</h3>
                <div class="plot-container">
        """
        if res['scatter_plot']:
            html_content += f"""
                    <div class="plot-box">
                        <h4>Profit Correlation (Scatter)</h4>
                        <img src="data:image/png;base64,{res['scatter_plot']}" alt="Scatter Plot for {file_name}">
                    </div>
            """
        if res['violin_plot']:
             html_content += f"""
                    <div class="plot-box">
                        <h4>Profit Distribution (Violin)</h4>
                        <img src="data:image/png;base64,{res['violin_plot']}" alt="Violin Plot for {file_name}">
                    </div>
            """
        html_content += """
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML exported to {filename}")

if __name__ == "__main__":
    main()
