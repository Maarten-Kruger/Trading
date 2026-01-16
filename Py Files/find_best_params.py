import pandas as pd
import numpy as np
import os
import glob
import math
from itertools import product

# Configuration
INPUT_DIR = "OptimizationResults"  # Relative to this script, or absolute
REQUIRED_GRID_SIDE = 3  # Corresponds to Radius = 1 (1 step up, 1 step down) -> 3 points total per axis

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

    results_table = []

    for file_path in excel_files:
        try:
            process_file(file_path, results_table)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print summary table
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
            'max_neigh_profit': None
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
                'max_neigh_profit': max_p
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

        # Optimization: if we already found a candidate with radius X, and the current candidate has result < best_fallback.result,
        # we still need to check if radius > X.

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
                'max_neigh_profit': max_p
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
                    'max_neigh_profit': max_p
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
    """
    Returns (min_profit, max_profit) of all neighbors within the valid radius.
    """
    if radius < 0:
        return None, None

    # If radius is 0, just the center
    if radius == 0:
        p = coord_map[center_coord]['Profit']
        return p, p

    ranges = [range(c - radius, c + radius + 1) for c in center_coord]
    profits = []

    for neighbor in product(*ranges):
        if neighbor in coord_map:
            profits.append(coord_map[neighbor]['Profit'])

    if not profits:
        return None, None

    return min(profits), max(profits)

if __name__ == "__main__":
    main()
