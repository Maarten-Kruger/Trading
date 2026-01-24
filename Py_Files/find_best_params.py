import pandas as pd
import numpy as np
import os
import glob
import math
from itertools import product, combinations
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import io
import base64

# Configuration
REQUIRED_GRID_SIDE = 5  # Corresponds to Radius = 1 (1 step up, 1 step down) -> 3 points total per axis

def main():
    # Ask for the path to the folder with the CSV files
    target_dir = input("Please enter the path to the folder with the CSV files: ").strip()

    if not target_dir:
        print("No directory provided.")
        return

    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found. Please check configuration.")
        return

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

    # PHASE 1: Load all files and determine Global Parameter Ranges
    files_data = []
    global_param_values = {} # {col_name: sorted_unique_values_list}

    print("Loading files and scanning parameter ranges...")
    for file_path in csv_files:
        try:
            # Load Data
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Clean numeric columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                    except ValueError:
                        pass

            # Identify Variables
            try:
                trades_idx = df.columns.get_loc("Trades")
                var_cols = df.columns[trades_idx+1:].tolist()
            except KeyError:
                print(f"Skipping {os.path.basename(file_path)}: 'Trades' column not found.")
                continue

            # Update Global Params
            for col in var_cols:
                unique_vals = df[col].dropna().unique()
                if col not in global_param_values:
                    global_param_values[col] = set()
                global_param_values[col].update(unique_vals)

            files_data.append({
                'path': file_path,
                'df': df,
                'var_cols': var_cols
            })

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Sort global param values
    for col in global_param_values:
        global_param_values[col] = sorted(list(global_param_values[col]))

    # PHASE 2: Process each file using Global Ranges
    results_table = []

    for fdata in files_data:
        try:
            process_file_data(fdata, global_param_values, results_table)
        except Exception as e:
            print(f"Error processing {fdata['path']}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table to console
    print("\n" + "="*120)
    print(f"{'File':<30} | {'Best Result':<11} | {'Profit':<10} | {'Inc/Dec':<8} | {'Neigh. Profits':<18} | {'Variables (Value, Step)'}")
    print("-" * 120)
    for res in results_table:
        file_name = os.path.basename(res['file'])
        best_res = f"{res['result']:.2f}"
        profit = f"{res['profit']:.2f}"

        if res.get('radius_inc') != "N/A":
             inc_dec = f"+{res['radius_inc']}/-{res['radius_dec']}"
        else:
             inc_dec = "N/A"

        # Neighbor Profits
        if res['min_neigh_profit'] is not None:
             neigh_prof = f"{res['min_neigh_profit']:.0f} - {res['max_neigh_profit']:.0f}"
        else:
             neigh_prof = "N/A"

        # Format variables string
        vars_parts = []
        for k, v in res['params'].items():
            step = res['step_sizes'].get(k, 0)
            if step > 0:
                vars_parts.append(f"{k}={v}")
            else:
                vars_parts.append(f"{k}={v}")

        vars_str = ", ".join(vars_parts)
        print(f"{file_name[:29]:<30} | {best_res:<11} | {profit:<10} | {inc_dec:<8} | {neigh_prof:<18} | {vars_str}")
    print("="*120 + "\n")

    export_to_html(results_table, global_param_values, filename=os.path.join(target_dir, "Optimization_Report.html"))

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def check_square_region(center_coord, active_indices, coord_map, radius, direction, condition_type):
    """
    Checks if the defined 'bubble' (square region) meets the condition.
    range: [0, R] for inc, [-R, 0] for dec. Excludes origin.
    active_indices: list of indices in center_coord that are varying.
    """
    full_dim = len(center_coord)
    ranges = []
    active_set = set(active_indices)

    for i in range(full_dim):
        if i in active_set:
            if direction == 'inc':
                ranges.append(range(0, radius + 1))
            else:
                ranges.append(range(-radius, 1))
        else:
            ranges.append(range(0, 1))

    points_in_bubble = []

    # Iterate all combinations in the box
    for steps in product(*ranges):
        # Exclude origin (all zeros means we are at center)
        if all(s == 0 for s in steps):
            continue

        # Calculate coordinate
        test_coord = list(center_coord)
        valid_point = True
        for i, step in enumerate(steps):
            test_coord[i] += step

        test_coord = tuple(test_coord)

        if test_coord not in coord_map:
            # Missing point implies the square is broken
            return False

        points_in_bubble.append(coord_map[test_coord])

    if not points_in_bubble:
        # Should not happen if R >= 1 and active_indices is not empty
        return True

    # Check Condition
    if condition_type == 'loss':
        # No Losing Sets -> All Profit >= 0
        for p in points_in_bubble:
            if p['Profit'] < 0:
                return False
    elif condition_type == 'avg':
        # Avg Profit > 0 (Condition applies to the Bubble Set)
        avg_profit = np.mean([p['Profit'] for p in points_in_bubble])
        if avg_profit <= 0:
            return False
    elif condition_type == 'result':
        # Result >= 25 (User Requirement)
        for p in points_in_bubble:
            if p['Result'] < 25:
                return False

    return True

def find_max_radius(center_coord, active_indices, coord_map, direction, condition_type):
    radius = 0
    # Safety limit to prevent infinite loops
    MAX_SEARCH_RADIUS = 50

    while radius < MAX_SEARCH_RADIUS:
        next_r = radius + 1
        if check_square_region(center_coord, active_indices, coord_map, next_r, direction, condition_type):
            radius = next_r
        else:
            break

    return radius

def calculate_1d_robustness(center_coord, varying_cols, coord_map):
    """
    Calculates the max valid increment and decrement for each variable independently (1D slice),
    while keeping other variables fixed at the center_coord values.
    Returns a dict: {col_name: {'inc': int, 'dec': int}}
    """
    robustness = {}

    for i, col in enumerate(varying_cols):
        # Current variable index is i
        # We vary coord[i], keep others fixed
        base_coord = list(center_coord)

        # Check Increment
        inc = 0
        while True:
            test_coord = list(base_coord)
            test_coord[i] += (inc + 1)
            if tuple(test_coord) in coord_map and coord_map[tuple(test_coord)]['Profit'] > 0:
                inc += 1
            else:
                break

        # Check Decrement
        dec = 0
        while True:
            test_coord = list(base_coord)
            test_coord[i] -= (dec + 1)
            if tuple(test_coord) in coord_map and coord_map[tuple(test_coord)]['Profit'] > 0:
                dec += 1
            else:
                break

        robustness[col] = {'inc': inc, 'dec': dec}

    return robustness

def calculate_robustness_stats(varying_cols, coord_map):
    """
    Calculates robustness based on the new logic:
    1. Filter Positive Vectors (Result > 25)
    2. Separate Inc/Dec bubbles.
    3. Rank Top 100 by Inc Radius.
    4. Find Max Sum (Inc + Dec) for the winner.
    Returns: { dim: { 'loss': max_sum, 'avg': max_sum, 'result': max_sum } }
    """
    num_vars = len(varying_cols)
    robustness_results = {} # { d: { 'loss': val, 'avg': val, 'result': val } }

    # Pre-calculate combinations
    indices = range(num_vars)
    dim_combinations = {d: list(combinations(indices, d)) for d in range(1, num_vars + 1)}

    # Filter Positive Vectors
    positive_vectors = []
    for coord, data in coord_map.items():
        if data['Result'] > 25:
             positive_vectors.append(coord)

    if not positive_vectors:
        return {d: {'loss': 0, 'avg': 0, 'result': 0} for d in range(1, num_vars + 1)}

    conditions = ['loss', 'avg', 'result']

    for d in range(1, num_vars + 1):
        robustness_results[d] = {}

        for cond in conditions:
            # We need to find the Best Vector for this (Dimension, Condition) pair.
            # We must check ALL combinations for this dimensionality and find the global max.

            current_dim_max_sum = 0

            for active_idxs in dim_combinations[d]:
                candidates = []
                # List of tuples: (inc_radius, result, coord, active_idxs)

                for coord in positive_vectors:
                    # Calculate Inc Radius
                    r_inc = find_max_radius(coord, active_idxs, coord_map, 'inc', cond)

                    # Store candidate
                    # We store Result for tie-breaking
                    res_val = coord_map[coord]['Result']
                    candidates.append({
                        'r_inc': r_inc,
                        'result': res_val,
                        'coord': coord,
                        'active_idxs': active_idxs
                    })

                # Rank: Sort by r_inc DESC, then result DESC
                candidates.sort(key=lambda x: (x['r_inc'], x['result']), reverse=True)

                # Keep Top 100 per combination
                top_100 = candidates[:100]

                # Find Winner (Max Sum) for this combination
                combo_max_sum = 0

                for cand in top_100:
                    coord = cand['coord']
                    active_idxs = cand['active_idxs']
                    r_inc = cand['r_inc']

                    # Calculate Dec Radius
                    r_dec = find_max_radius(coord, active_idxs, coord_map, 'dec', cond)

                    total_sum = r_inc + r_dec
                    if total_sum > combo_max_sum:
                        combo_max_sum = total_sum

                if combo_max_sum > current_dim_max_sum:
                    current_dim_max_sum = combo_max_sum

            robustness_results[d][cond] = current_dim_max_sum

    return robustness_results

def process_file_data(file_data, global_param_values, results_table):
    file_path = file_data['path']
    df = file_data['df']
    var_cols = file_data['var_cols']

    print(f"Processing {os.path.basename(file_path)}...")

    if not var_cols:
        print("No variable columns found.")
        return

    # Determine step sizes and identify varying columns (local check)
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

    # 1. Calculation Logic: Find Best Candidate First
    best_candidate = None
    coord_map = {} # Defined here to be available for plotting later if needed

    if not varying_cols:
        best_idx = df['Result'].idxmax()
        row = df.loc[best_idx]
        best_candidate = {
            'file': file_path,
            'result': row['Result'],
            'profit': row['Profit'],
            'params': row[var_cols].to_dict(),
            'step_sizes': step_sizes,
            'radius_inc': "N/A",
            'radius_dec': "N/A",
            'min_neigh_profit': None,
            'max_neigh_profit': None,
            'robustness_1d': {},
            'multidim_robustness': {},
            'coord_map': {},
            'center_coord': (),
            'varying_cols': varying_cols
        }
    else:
        # Normalize coordinates
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

        sorted_candidates = sorted(coord_map.items(), key=lambda x: x[1]['Result'], reverse=True)
        target_radius = (REQUIRED_GRID_SIDE - 1) // 2

        # Primary Search
        for coord, data in sorted_candidates:
            # We want min(inc, dec) >= target_radius
            inc, dec = measure_max_box(coord, varying_cols, coord_map)

            if min(inc, dec) >= target_radius:
                min_p, max_p = get_box_stats(coord, inc, dec, coord_map)
                robust_1d = calculate_1d_robustness(coord, varying_cols, coord_map)

                best_candidate = {
                    'file': file_path,
                    'result': data['Result'],
                    'profit': data['Profit'],
                    'params': data['Row'][var_cols].to_dict(),
                    'step_sizes': step_sizes,
                    'radius_inc': inc,
                    'radius_dec': dec,
                    'min_neigh_profit': min_p,
                    'max_neigh_profit': max_p,
                    'robustness_1d': robust_1d,
                    # multidim_robustness calculated globally below
                    'coord_map': coord_map,
                    'center_coord': coord,
                    'varying_cols': varying_cols
                }
                break

        # Fallback Search
        if not best_candidate:
            print("  No candidate met target radius. Searching for max box...")
            max_found_min_dim = -1
            best_fallback = None

            for coord, data in sorted_candidates:
                inc, dec = measure_max_box(coord, varying_cols, coord_map)
                min_dim = min(inc, dec)

                if min_dim > max_found_min_dim:
                    max_found_min_dim = min_dim
                    min_p, max_p = get_box_stats(coord, inc, dec, coord_map)
                    robust_1d = calculate_1d_robustness(coord, varying_cols, coord_map)

                    best_fallback = {
                        'file': file_path,
                        'result': data['Result'],
                        'profit': data['Profit'],
                        'params': data['Row'][var_cols].to_dict(),
                        'step_sizes': step_sizes,
                        'radius_inc': inc,
                        'radius_dec': dec,
                        'min_neigh_profit': min_p,
                        'max_neigh_profit': max_p,
                        'robustness_1d': robust_1d,
                        'coord_map': coord_map,
                        'center_coord': coord,
                        'varying_cols': varying_cols
                    }
                elif min_dim == max_found_min_dim and best_fallback is None:
                     # Keep first one found (highest result)
                    min_p, max_p = get_box_stats(coord, inc, dec, coord_map)
                    robust_1d = calculate_1d_robustness(coord, varying_cols, coord_map)

                    best_fallback = {
                        'file': file_path,
                        'result': data['Result'],
                        'profit': data['Profit'],
                        'params': data['Row'][var_cols].to_dict(),
                        'step_sizes': step_sizes,
                        'radius_inc': inc,
                        'radius_dec': dec,
                        'min_neigh_profit': min_p,
                        'max_neigh_profit': max_p,
                        'robustness_1d': robust_1d,
                        'coord_map': coord_map,
                        'center_coord': coord,
                        'varying_cols': varying_cols
                    }
            best_candidate = best_fallback

    if not best_candidate:
        print("  No suitable data found.")
        return

    # Calculate Global Robustness Stats (New Logic)
    print("  Calculating Multi-Dim Robustness Stats...")
    best_candidate['multidim_robustness'] = calculate_robustness_stats(varying_cols, coord_map)

    # 2. Plotting: Heatmaps (Using Global Ranges)
    heatmaps_dict = {}
    result_heatmaps_dict = {}

    if varying_cols and len(varying_cols) >= 2:
        pairs = list(combinations(varying_cols, 2))

        for col1, col2 in pairs:
            # --- PROFIT HEATMAP ---
            # Create a dedicated figure for each heatmap
            fig, ax = plt.subplots(figsize=(5, 4))

            # Filter DF: other cols fixed to best params
            mask = np.ones(len(df), dtype=bool)
            for other_col in varying_cols:
                if other_col in [col1, col2]: continue
                target_val = best_candidate['params'][other_col]
                mask = mask & (np.isclose(df[other_col], target_val))

            slice_df = df[mask]

            if not slice_df.empty:
                try:
                    # Global Axis Ranges
                    x_vals_global = global_param_values.get(col1, sorted(df[col1].unique()))
                    y_vals_global = global_param_values.get(col2, sorted(df[col2].unique()))

                    # Create Pivot Table
                    pivot = slice_df.pivot_table(index=col2, columns=col1, values='Profit', aggfunc='mean')

                    # REINDEX to Global Range
                    pivot = pivot.reindex(index=y_vals_global, columns=x_vals_global)

                    # Determine norm for Profit (center=0)
                    data_min = np.nanmin(pivot.values)
                    data_max = np.nanmax(pivot.values)
                    center = 0

                    # Ensure vmin < center < vmax
                    vmin = min(data_min, center - 1e-9)
                    vmax = max(data_max, center + 1e-9)

                    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

                    # Plot Heatmap
                    im = ax.imshow(pivot.values, cmap='RdYlGn', origin='lower', aspect='auto', norm=norm)

                    # Set ticks and labels
                    ax.set_xticks(np.arange(len(x_vals_global)))
                    ax.set_yticks(np.arange(len(y_vals_global)))
                    ax.set_xticklabels([f"{v:g}" for v in x_vals_global], rotation=45, ha='right')
                    ax.set_yticklabels([f"{v:g}" for v in y_vals_global])

                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title(f"Profit Heatmap: {col1} vs {col2}")

                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Profit')

                    # Highlight Best Point
                    best_x = best_candidate['params'][col1]
                    best_y = best_candidate['params'][col2]

                    try:
                        # Find indices in the GLOBAL list
                        x_idx = x_vals_global.index(best_x) if best_x in x_vals_global else -1
                        y_idx = y_vals_global.index(best_y) if best_y in y_vals_global else -1

                        if x_idx != -1 and y_idx != -1:
                            ax.plot(x_idx, y_idx, marker='*', color='blue', markersize=10, markeredgecolor='white')
                    except ValueError:
                        pass

                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No Data for Slice", ha='center', va='center')

            plt.tight_layout()
            heatmaps_dict[(col1, col2)] = plot_to_base64(fig)
            plt.close(fig)

            # --- RESULT HEATMAP ---
            fig, ax = plt.subplots(figsize=(5, 4))

            # (We can reuse slice_df from above as the filter logic is identical)

            if not slice_df.empty:
                try:
                    # Global Axis Ranges (Same as above)
                    x_vals_global = global_param_values.get(col1, sorted(df[col1].unique()))
                    y_vals_global = global_param_values.get(col2, sorted(df[col2].unique()))

                    # Create Pivot Table for RESULT
                    pivot = slice_df.pivot_table(index=col2, columns=col1, values='Result', aggfunc='mean')

                    # REINDEX to Global Range
                    pivot = pivot.reindex(index=y_vals_global, columns=x_vals_global)

                    # Determine norm for Result (center=25)
                    data_min = np.nanmin(pivot.values)
                    data_max = np.nanmax(pivot.values)
                    center = 25

                    # Ensure vmin < center < vmax
                    vmin = min(data_min, center - 1e-9)
                    vmax = max(data_max, center + 1e-9)

                    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

                    # Plot Heatmap
                    im = ax.imshow(pivot.values, cmap='RdYlGn', origin='lower', aspect='auto', norm=norm)

                    # Set ticks and labels
                    ax.set_xticks(np.arange(len(x_vals_global)))
                    ax.set_yticks(np.arange(len(y_vals_global)))
                    ax.set_xticklabels([f"{v:g}" for v in x_vals_global], rotation=45, ha='right')
                    ax.set_yticklabels([f"{v:g}" for v in y_vals_global])

                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title(f"Result Heatmap: {col1} vs {col2}")

                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Result')

                    # Highlight Best Point (Same as above)
                    best_x = best_candidate['params'][col1]
                    best_y = best_candidate['params'][col2]

                    try:
                        x_idx = x_vals_global.index(best_x) if best_x in x_vals_global else -1
                        y_idx = y_vals_global.index(best_y) if best_y in y_vals_global else -1

                        if x_idx != -1 and y_idx != -1:
                            ax.plot(x_idx, y_idx, marker='*', color='blue', markersize=10, markeredgecolor='white')
                    except ValueError:
                        pass

                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No Data for Slice", ha='center', va='center')

            plt.tight_layout()
            result_heatmaps_dict[(col1, col2)] = plot_to_base64(fig)
            plt.close(fig)


    best_candidate['heatmaps_dict'] = heatmaps_dict
    best_candidate['result_heatmaps_dict'] = result_heatmaps_dict

    # Violin Plot (Individual Plots per Variable)
    violin_plots_dict = {}
    result_violin_plots_dict = {}

    if varying_cols:
        for col in varying_cols:
            # --- PROFIT VIOLIN ---
            fig, ax = plt.subplots(figsize=(5, 4))
            try:
                global_vals = global_param_values.get(col, sorted(df[col].unique()))
                data_to_plot = []
                for val in global_vals:
                    subset = df[df[col] == val]['Profit'].values
                    data_to_plot.append(subset if len(subset) > 0 else [])

                valid_data = []
                valid_positions = []
                for idx, d in enumerate(data_to_plot):
                    if len(d) > 0:
                        valid_data.append(d)
                        valid_positions.append(idx + 1)

                if valid_data:
                    parts = ax.violinplot(valid_data, positions=valid_positions, showmeans=False, showmedians=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor('indianred')
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)

                ax.set_xticks(range(1, len(global_vals) + 1))
                ax.set_xticklabels([str(v) for v in global_vals], rotation=45, ha='right')
                ax.set_xlim(0.5, len(global_vals) + 0.5)
                ax.set_title(f"Profit Dist. by {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Profit")
                ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            plt.tight_layout()
            violin_plots_dict[col] = plot_to_base64(fig)
            plt.close(fig)

            # --- RESULT VIOLIN ---
            fig, ax = plt.subplots(figsize=(5, 4))
            try:
                global_vals = global_param_values.get(col, sorted(df[col].unique()))
                data_to_plot = []
                for val in global_vals:
                    subset = df[df[col] == val]['Result'].values # <--- Result
                    data_to_plot.append(subset if len(subset) > 0 else [])

                valid_data = []
                valid_positions = []
                for idx, d in enumerate(data_to_plot):
                    if len(d) > 0:
                        valid_data.append(d)
                        valid_positions.append(idx + 1)

                if valid_data:
                    parts = ax.violinplot(valid_data, positions=valid_positions, showmeans=False, showmedians=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor('cornflowerblue') # Change color for Results to distinguish?
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)

                ax.set_xticks(range(1, len(global_vals) + 1))
                ax.set_xticklabels([str(v) for v in global_vals], rotation=45, ha='right')
                ax.set_xlim(0.5, len(global_vals) + 0.5)
                ax.set_title(f"Result Dist. by {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Result")
                ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            plt.tight_layout()
            result_violin_plots_dict[col] = plot_to_base64(fig)
            plt.close(fig)

    best_candidate['violin_plots_dict'] = violin_plots_dict
    best_candidate['result_violin_plots_dict'] = result_violin_plots_dict

    # Remove large objects
    if 'coord_map' in best_candidate: del best_candidate['coord_map']

    results_table.append(best_candidate)

def check_box(center_coord, inc, dec, coord_map):
    ranges = [range(c - dec, c + inc + 1) for c in center_coord]
    for neighbor in product(*ranges):
        if neighbor not in coord_map:
            return False
        if coord_map[neighbor]['Profit'] <= 0:
            return False
    return True

def measure_max_box(center_coord, varying_cols, coord_map):
    if center_coord not in coord_map or coord_map[center_coord]['Profit'] <= 0:
        return 0, 0

    inc = 0
    dec = 0

    # Greedily expand both
    while True:
        can_inc = check_box(center_coord, inc + 1, dec, coord_map)
        can_dec = check_box(center_coord, inc, dec + 1, coord_map)

        if can_inc and can_dec:
            inc += 1
            dec += 1
        elif can_inc:
            inc += 1
        elif can_dec:
            dec += 1
        else:
            break

    return inc, dec

def get_box_stats(center_coord, inc, dec, coord_map):
    ranges = [range(c - dec, c + inc + 1) for c in center_coord]
    profits = []
    for neighbor in product(*ranges):
        if neighbor in coord_map:
            profits.append(coord_map[neighbor]['Profit'])
    if not profits: return None, None
    return min(profits), max(profits)

def export_to_html(results_table, global_param_values, filename="Optimization_Report.html"):
    if not results_table:
        print("No results to export.")
        return

    # Generate the Three Big Graphs for Multi-Dim Robustness
    graph_loss_b64 = None
    graph_avg_b64 = None
    graph_result_b64 = None

    file_names = [os.path.basename(res['file']) for res in results_table]
    x_indices = range(len(file_names))

    # Collect max dimensions across all files to define consistent lines
    max_dims = 0
    for res in results_table:
        if 'multidim_robustness' in res:
             max_dims = max(max_dims, max(res['multidim_robustness'].keys(), default=0))

    if max_dims > 0:
        # Common plot settings
        def create_robustness_graph(metric_key, title):
            fig, ax = plt.subplots(figsize=(12, 8))

            for d in range(1, max_dims + 1):
                y_values = []
                for res in results_table:
                    val = 0
                    if 'multidim_robustness' in res and d in res['multidim_robustness']:
                        val = res['multidim_robustness'][d][metric_key]
                    y_values.append(val)

                label = f"{d} Variable{'s' if d>1 else ''}"
                ax.plot(x_indices, y_values, marker='o', label=label)

            ax.set_xticks(x_indices)
            ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=8)
            ax.set_title(title)
            ax.set_ylabel("Max Sum (Inc + Dec Radius)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            return plot_to_base64(fig)

        graph_loss_b64 = create_robustness_graph('loss', "Robustness Limit: No Loss (Max Sum Inc+Dec)")
        graph_avg_b64 = create_robustness_graph('avg', "Robustness Limit: Average Profit > 0 (Max Sum Inc+Dec)")
        graph_result_b64 = create_robustness_graph('result', "Robustness Limit: Result >= 25 (Max Sum Inc+Dec)")

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
            .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }}
            .plot-box {{ flex: 0 0 auto; margin-bottom: 20px; border: 1px solid #eee; padding: 10px; }}
            .plot-box h4 {{ text-align: center; margin: 5px 0; font-size: 0.9em; }}
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
                    <th>Inc/Dec</th>
                    <th>Neighbor Profits</th>
                    <th>Variables (Value, Step)</th>
                </tr>
            </thead>
            <tbody>
    """

    for res in results_table:
        file_name = os.path.basename(res['file'])

        if res.get('radius_inc') != "N/A":
             inc_dec = f"+{res['radius_inc']}/-{res['radius_dec']}"
        else:
             inc_dec = "N/A"

        if res['min_neigh_profit'] is not None:
             neigh_prof = f"{res['min_neigh_profit']:.0f} - {res['max_neigh_profit']:.0f}"
        else:
             neigh_prof = "N/A"

        vars_parts = []
        for k, v in res['params'].items():
            step = res['step_sizes'].get(k, 0)
            if step > 0:
                vars_parts.append(f"{k}={v}")
            else:
                vars_parts.append(f"{k}={v}")
        vars_str = "\\n".join(vars_parts)

        html_content += f"""
                <tr>
                    <td>{file_name}</td>
                    <td>{res['result']:.2f}</td>
                    <td>{res['profit']:.2f}</td>
                    <td>{inc_dec}</td>
                    <td>{neigh_prof}</td>
                    <td class="variables">{vars_str}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>

        <div class="section">
            <h2>Multi-Dimensional Robustness Analysis</h2>
            <p>These graphs show the maximum "Step Radius" for each dimensionality group (1 variable changing, 2 variables changing simultaneously, etc.) where the robustness condition holds.</p>
    """

    if graph_loss_b64:
        html_content += f"""
            <h3>1. Loss Limit (No Losing Sets)</h3>
            <p>Maximum radius where NO combination of parameter changes results in a loss.</p>
            <img src="data:image/png;base64,{graph_loss_b64}" alt="Loss Robustness Graph">

            <h3>2. Average Profit Limit</h3>
            <p>Maximum radius where the average profit of all tested combinations remains positive.</p>
            <img src="data:image/png;base64,{graph_avg_b64}" alt="Avg Robustness Graph">

            <h3>3. Result Limit (Result >= 30)</h3>
            <p>Maximum radius where all tested combinations yield a 'Result' score of at least 30.</p>
            <img src="data:image/png;base64,{graph_result_b64}" alt="Result Robustness Graph">
        """
    else:
        html_content += "<p>No multi-dimensional robustness data available.</p>"

    html_content += """
        </div>

        <div class="section">
            <h2>Detailed Analysis: Profit</h2>
    """

    # SECTION 3: Variable Distributions (Violin Plots - PROFIT)
    # Group by Variable
    all_vars = sorted(list(global_param_values.keys()))
    html_content += "<h3>Distribution Analysis (Profit Violin Plots)</h3>"

    for var in all_vars:
        html_content += f"<h4>Variable: {var}</h4>"
        html_content += '<div class="plot-container">'

        has_plots = False
        for res in results_table:
            file_name = os.path.basename(res['file'])
            # Check if this file has a plot for this variable
            if 'violin_plots_dict' in res and var in res['violin_plots_dict']:
                img_b64 = res['violin_plots_dict'][var]
                html_content += f"""
                    <div class="plot-box">
                        <h4>{file_name}</h4>
                        <img src="data:image/png;base64,{img_b64}" alt="Profit Violin {var} {file_name}">
                    </div>
                """
                has_plots = True

        if not has_plots:
            html_content += "<p>No distribution data for this variable.</p>"

        html_content += '</div>' # End plot-container

    # SECTION 4: 2D Heatmaps (PROFIT)
    # Collect all pairs found across all files
    all_pairs = set()
    for res in results_table:
        if 'heatmaps_dict' in res:
            all_pairs.update(res['heatmaps_dict'].keys())

    # Sort pairs for consistent order
    sorted_pairs = sorted(list(all_pairs))

    if sorted_pairs:
        html_content += "<h3>Interaction Analysis (Profit 2D Heatmaps)</h3>"

        for pair in sorted_pairs:
            var1, var2 = pair
            html_content += f"<h4>Interaction: {var1} vs {var2}</h4>"
            html_content += '<div class="plot-container">'

            has_plots = False
            for res in results_table:
                file_name = os.path.basename(res['file'])
                if 'heatmaps_dict' in res and pair in res['heatmaps_dict']:
                    img_b64 = res['heatmaps_dict'][pair]
                    html_content += f"""
                        <div class="plot-box">
                            <h4>{file_name}</h4>
                            <img src="data:image/png;base64,{img_b64}" alt="Profit Heatmap {var1}v{var2} {file_name}">
                        </div>
                    """
                    has_plots = True

            if not has_plots:
                html_content += "<p>No interaction data for this pair.</p>"

            html_content += '</div>'

    html_content += """
        </div>

        <div class="section">
            <h2>Detailed Analysis: Results</h2>
    """

    # SECTION 5: Variable Distributions (Violin Plots - RESULTS)
    html_content += "<h3>Distribution Analysis (Result Violin Plots)</h3>"

    for var in all_vars:
        html_content += f"<h4>Variable: {var}</h4>"
        html_content += '<div class="plot-container">'

        has_plots = False
        for res in results_table:
            file_name = os.path.basename(res['file'])
            if 'result_violin_plots_dict' in res and var in res['result_violin_plots_dict']:
                img_b64 = res['result_violin_plots_dict'][var]
                html_content += f"""
                    <div class="plot-box">
                        <h4>{file_name}</h4>
                        <img src="data:image/png;base64,{img_b64}" alt="Result Violin {var} {file_name}">
                    </div>
                """
                has_plots = True

        if not has_plots:
            html_content += "<p>No distribution data for this variable.</p>"

        html_content += '</div>'

    # SECTION 6: 2D Heatmaps (RESULTS)
    # We can reuse sorted_pairs as the pairs are the same (based on variables)
    if sorted_pairs:
        html_content += "<h3>Interaction Analysis (Result 2D Heatmaps)</h3>"

        for pair in sorted_pairs:
            var1, var2 = pair
            html_content += f"<h4>Interaction: {var1} vs {var2}</h4>"
            html_content += '<div class="plot-container">'

            has_plots = False
            for res in results_table:
                file_name = os.path.basename(res['file'])
                if 'result_heatmaps_dict' in res and pair in res['result_heatmaps_dict']:
                    img_b64 = res['result_heatmaps_dict'][pair]
                    html_content += f"""
                        <div class="plot-box">
                            <h4>{file_name}</h4>
                            <img src="data:image/png;base64,{img_b64}" alt="Result Heatmap {var1}v{var2} {file_name}">
                        </div>
                    """
                    has_plots = True

            if not has_plots:
                html_content += "<p>No interaction data for this pair.</p>"

            html_content += '</div>'

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
