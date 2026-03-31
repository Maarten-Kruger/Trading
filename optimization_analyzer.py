import pandas as pd
import plotly.graph_objects as go
import os

# Filter variables
MIN_PROFIT = None
MIN_TRADES = None

def process_csv(csv_path, min_profit, min_trades):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Initial filtering if arguments provided
    if min_profit is not None:
        df = df[df['Profit'] >= min_profit]
    if min_trades is not None:
        df = df[df['Trades'] >= min_trades]

    # Get variables after the "Trades" column
    cols = df.columns.tolist()
    if 'Trades' not in cols:
        raise ValueError("Column 'Trades' not found in the CSV file.")

    trades_idx = cols.index('Trades')
    variable_cols = cols[trades_idx + 1:]

    print(f"Found variables: {variable_cols}")

    # Generate HTML string
    html_content = f"""<html>
<head>
    <title>Optimization Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>Optimization Analysis for {os.path.basename(csv_path)}</h1>
"""

    for var_col in variable_cols:
        print(f"Generating graph for variable: {var_col}")
        other_vars = [c for c in variable_cols if c != var_col]

        fig = go.Figure()

        # Get all unique, sorted values for the x-axis variable to ensure gaps are shown properly
        x_steps = sorted(df[var_col].unique())

        # Group by all other variables
        if not other_vars:
            # If there's only one variable overall
            group_data = df.sort_values(var_col)
            fig.add_trace(go.Scatter(x=group_data[var_col], y=group_data['Profit'], mode='lines+markers', name="All"))
        else:
            grouped = df.groupby(other_vars)

            for group_keys, group_df in grouped:
                # Name of the line based on the combination of other variables
                if len(other_vars) == 1:
                    name_str = f"{other_vars[0]}={group_keys}"
                else:
                    name_parts = [f"{k}={v}" for k, v in zip(other_vars, group_keys)]
                    name_str = ", ".join(name_parts)

                # Create a complete series based on all possible x steps to maintain gaps
                # We can do this by setting index to var_col and reindexing to x_steps
                series_data = group_df.set_index(var_col)['Profit'].reindex(x_steps)

                fig.add_trace(go.Scatter(
                    x=series_data.index,
                    y=series_data.values,
                    mode='lines+markers',
                    name=name_str,
                    connectgaps=False # Ensure missing data creates gaps
                ))

        fig.update_layout(
            title=f"Profit vs {var_col}",
            xaxis_title=var_col,
            yaxis_title="Profit",
            hovermode="x unified"
        )

        # Convert figure to HTML block and append
        html_content += fig.to_html(full_html=False, include_plotlyjs=False)

    html_content += "</body></html>"

    # Save output to the same directory
    output_filename = os.path.splitext(csv_path)[0] + "_analysis.html"
    print(f"Saving output to {output_filename}...")
    with open(output_filename, 'w') as f:
        f.write(html_content)

    print("Done!")

if __name__ == "__main__":
    csv_file = input("Enter the path to the CSV file: ").strip(' "\'')
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
    else:
        process_csv(csv_file, MIN_PROFIT, MIN_TRADES)
