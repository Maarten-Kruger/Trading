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
    print("\n" + "="*80)
    print(f"{'File':<30} | {'Best Result':<12} | {'Max Radius':<10} | {'Variables (Values)'}")
    print("-" * 80)
    for res in results_table:
        file_name = os.path.basename(res['file'])
        best_res = f"{res['result']:.2f}"
        radius = res['radius']
        # Format variables string
        vars_str = ", ".join([f"{k}={v}" for k, v in res['params'].items()])
        print(f"{file_name[:29]:<30} | {best_res:<12} | {radius:<10} | {vars_str}")
    print("="*80 + "\n")

def process_file(file_path, results_table):
    print(f"Processing {file_path}...")

    # Read Excel, handling comma decimals
    # Since we don't know if it's strictly CSV-style in Excel or proper numbers,
    # we read as string first to safely replace commas, then convert.
    # But usually pd.read_excel reads numbers as numbers if they are formatted as such.
    # The prompt implies they might be text like "30,65".
    # We'll try reading normally, if object type, then convert.

    try:
        df = pd.read_excel(file_path)
    except Exception:
        # Fallback for csv masquerading as xlsx? No, user said Excel.
        # Maybe use engine='openpyxl' explicit.
        df = pd.read_excel(file_path, engine='openpyxl')

    # Convert comma decimals to float for all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Replace comma with dot and convert
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            except ValueError:
                # Could be true string column
                pass

    # Identify variable columns
    # Standard columns end at 'Trades'.
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
            # Constant parameter, treated as satisfied for robustness check (or ignored)
            # We assign step size 0 to indicate it doesn't move.
            step_sizes[col] = 0.0
            continue

        # Calculate differences
        diffs = np.diff(unique_vals)
        # Filter small diffs (floating point noise)
        diffs = diffs[diffs > 1e-9]

        if len(diffs) == 0:
            step_sizes[col] = 0.0
            continue

        # Use GCD-like approach or Min diff.
        # In optimization, usually steps are uniform. Min diff is safe.
        # But if we have 10, 30 (step 10), min is 20? No.
        # 10, 30 -> diff 20. If 20 was missing, we might assume step is 20.
        # But if user says "integer amount step sizes", let's assume min diff is the unit.
        step_val = np.min(diffs)
        step_sizes[col] = step_val
        varying_cols.append(col)

    # If no columns vary, robustness is trivial (radius infinite? or 0?)
    # If parameters don't vary, we can't "increment".
    # But we can consider "increment" impossible, so radius is effectively 0?
    # Or, the condition "can increment... and still be profitable" fails if we can't increment.
    # However, if I have a fixed parameter, do I fail?
    # Let's assume constant params are "neutral" and don't block robustness.
    # Only VARYING params define the grid.

    if not varying_cols:
        # Only one setup exists or all fixed.
        # Find best result.
        best_idx = df['Result'].idxmax()
        row = df.loc[best_idx]
        results_table.append({
            'file': file_path,
            'result': row['Result'],
            'params': row[var_cols].to_dict(),
            'radius': "N/A (Fixed)"
        })
        return

    # Normalize coordinates for varying columns
    # coord = round( (val - min) / step )
    # Store in a dictionary: tuple(coords) -> (Profit, Result, Index)

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
        # Store data. If duplicate coords (shouldn't happen in opt result usually), take better profit?
        # But opt results usually unique params.
        coord_map[coord_tuple] = {
            'Profit': row['Profit'],
            'Result': row['Result'],
            'Index': idx,
            'Row': row
        }

    # Sort candidates by Result descending
    sorted_candidates = sorted(coord_map.items(), key=lambda x: x[1]['Result'], reverse=True)

    # Target radius
    target_radius = (REQUIRED_GRID_SIDE - 1) // 2 # 3 -> 1

    best_candidate_radius_1 = None

    # 1. Find Best Result with Radius >= 1
    for coord, data in sorted_candidates:
        if check_radius(coord, target_radius, varying_cols, coord_map):
            # Found the best result that satisfies condition
            # Now calculate its ACTUAL max radius
            max_r = measure_max_radius(coord, varying_cols, coord_map)
            best_candidate_radius_1 = {
                'file': file_path,
                'result': data['Result'],
                'params': data['Row'][var_cols].to_dict(),
                'radius': max_r
            }
            break

    if best_candidate_radius_1:
        results_table.append(best_candidate_radius_1)
        return

    # 2. Fallback: Find row with Max Radius
    # We need to iterate all (or top N?) to find max radius.
    # Brute force on all might be slow if 10k rows. But Python is reasonably fast for 10k.
    # Optimization: If we find a large radius, we can skip checks for candidates that are "on the edge" of the parameter space?
    # Simple approach: Check all.

    print("  No candidate met target radius. Searching for max radius...")

    max_found_radius = -1
    best_candidate_fallback = None

    for coord, data in sorted_candidates:
        r = measure_max_radius(coord, varying_cols, coord_map)
        if r > max_found_radius:
            max_found_radius = r
            best_candidate_fallback = {
                'file': file_path,
                'result': data['Result'],
                'params': data['Row'][var_cols].to_dict(),
                'radius': r
            }
        elif r == max_found_radius:
            # Tie breaker: Result (already sorted by result, but we are iterating in order)
            # If we find same radius later, it has lower Result, so keep existing.
            if best_candidate_fallback is None:
                best_candidate_fallback = {
                    'file': file_path,
                    'result': data['Result'],
                    'params': data['Row'][var_cols].to_dict(),
                    'radius': r
                }

    if best_candidate_fallback:
        results_table.append(best_candidate_fallback)
    else:
        # Should not happen if data exists
        print("  No suitable data found.")

def check_radius(center_coord, radius, varying_cols, coord_map):
    """
    Checks if all neighbors within radius exist and have Profit > 0.
    """
    if radius == 0:
        return coord_map[center_coord]['Profit'] > 0

    ranges = [range(c - radius, c + radius + 1) for c in center_coord]

    # Generate all neighbors
    for neighbor in product(*ranges):
        if neighbor not in coord_map:
            return False
        if coord_map[neighbor]['Profit'] <= 0:
            return False

    return True

def measure_max_radius(center_coord, varying_cols, coord_map):
    """
    Determines the maximum radius K such that all neighbors <= K are valid.
    """
    # Verify center first
    if center_coord not in coord_map or coord_map[center_coord]['Profit'] <= 0:
        return -1 # Invalid center

    radius = 1
    while True:
        if check_radius(center_coord, radius, varying_cols, coord_map):
            radius += 1
        else:
            return radius - 1

if __name__ == "__main__":
    main()
