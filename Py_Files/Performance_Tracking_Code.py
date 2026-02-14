import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import re
import glob
import random
import tkinter as tk
from tkinter import filedialog

def generate_performance_report():
    # 1. Ask for folder path
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Bring window to front
    folder_path = filedialog.askdirectory(title="Select Folder Containing MT5 Optimization CSVs")
    
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    # 2. Identify and Sort Files
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not files:
        print(f"No CSV files found in: {folder_path}")
        return

    # Configuration
    param_cols = ['InpMA1Period', 'InpMA2Period', 'InpADXPeriod', 'InpADXThreshold', 'InpTPPoints', 'InpSLPoints']
    metric_cols = ['Profit', 'Result']
    
    print(f"Found {len(files)} files. Analyzing data...")

    # 3. Choose 100 random parameter sets from the first file
    try:
        # MT5 uses ';' separator and ',' decimal
        first_df = pd.read_csv(files[0], sep=';', decimal=',')
        unique_sets = first_df[param_cols].drop_duplicates()
        
        sample_size = min(100, len(unique_sets))
        selected_params = unique_sets.sample(n=sample_size)
        param_values = selected_params.values
        
        # Initialize storage for metrics
        perf_data = {metric: {tuple(row): [] for row in param_values} for metric in metric_cols}
        week_labels = []

        # 4. Extract performance across all weeks
        for file_path in files:
            # Extract dates from filename for the X-axis (e.g., 20240101.20240108)
            match = re.search(r'(\d{8}\.\d{8})', os.path.basename(file_path))
            label = match.group(0) if match else os.path.basename(file_path)
            week_labels.append(label)
            
            df = pd.read_csv(file_path, sep=';', decimal=',')
            for metric in metric_cols:
                df_grouped = df.groupby(param_cols)[metric].first()
                for p_tuple in perf_data[metric].keys():
                    val = df_grouped.get(p_tuple, 0) # Default to 0 if set didn't trade that week
                    perf_data[metric][p_tuple].append(val)

        # 5. Build the HTML Report
        print("Generating HTML report with 200 graphs...")
        html_content = f"""
        <html>
        <head>
            <title>MT5 Parameter Performance Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; background-color: #f4f7f6; }}
                .set-card {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 30px; 
                             border-radius: 10px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .params {{ font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>MT5 Strategy Tracking Report</h1>
            <p>Tracking 100 random parameter sets across {len(files)} optimization periods.</p>
        """

        for i, params in enumerate(param_values):
            p_tuple = tuple(params)
            param_desc = " | ".join([f"{col}: {val}" for col, val in zip(param_cols, params)])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Profit
            ax1.plot(week_labels, perf_data['Profit'][p_tuple], marker='o', color='#27ae60')
            ax1.set_title(f'Profit Over Time')
            ax1.grid(True, linestyle='--', alpha=0.6)
            plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
            
            # Plot 2: Result
            ax2.plot(week_labels, perf_data['Result'][p_tuple], marker='s', color='#2980b9')
            ax2.set_title(f'Result Over Time')
            ax2.grid(True, linestyle='--', alpha=0.6)
            plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
            
            plt.tight_layout()
            
            # Embed image in HTML
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            html_content += f"""
            <div class="set-card">
                <div class="params">Set {i+1}: {param_desc}</div>
                <img src="data:image/png;base64,{img_b64}">
            </div>
            """

        html_content += "</body></html>"
        
        output_path = os.path.join(folder_path, "parameter_tracking_report.html")
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"\nSuccess! Report saved to:\n{output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_performance_report()