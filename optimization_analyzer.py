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
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 40px;
            padding: 20px;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <h1>Optimization Analysis for {os.path.basename(csv_path)}</h1>
"""

    for var_col in variable_cols:
        print(f"Generating graph for variable: {var_col}")
        other_vars = [c for c in variable_cols if c != var_col]

        fig = go.Figure()

        # Get all unique, sorted values for the x-axis variable to ensure gaps are shown properly
        x_steps = sorted(df[var_col].dropna().unique())

        # Group by all other variables
        if not other_vars:
            # If there's only one variable overall
            # Group by var_col and mean the profit to avoid duplicate indices
            group_data = df.groupby(var_col)['Profit'].mean().reset_index()
            group_data = group_data.sort_values(var_col)
            fig.add_trace(go.Scatter(
                x=group_data[var_col],
                y=group_data['Profit'],
                mode='lines+markers',
                name="All"
            ))
        else:
            # Group by all variables including var_col and average the profit
            # This fixes the "cannot reindex on an axis with duplicate labels" error
            avg_df = df.groupby([var_col] + other_vars)['Profit'].mean().reset_index()

            # Now group by the other variables to create the lines
            grouped = avg_df.groupby(other_vars)

            # To limit visual mess, if there are many groups, we can adjust visibility
            # But the user wants them all. We'll improve the layout instead.

            for group_keys, group_df in grouped:
                # Name of the line based on the combination of other variables
                if len(other_vars) == 1:
                    name_str = f"{other_vars[0]}={group_keys}"
                else:
                    if isinstance(group_keys, tuple):
                        name_parts = [f"{k}={v}" for k, v in zip(other_vars, group_keys)]
                    else:
                        name_parts = [f"{other_vars[0]}={group_keys}"]
                    name_str = ", ".join(name_parts)

                # Create a complete series based on all possible x steps to maintain gaps
                series_data = group_df.set_index(var_col)['Profit'].reindex(x_steps)

                fig.add_trace(go.Scatter(
                    x=series_data.index,
                    y=series_data.values,
                    mode='lines+markers',
                    name=name_str,
                    connectgaps=False, # Ensure missing data creates gaps
                    opacity=0.8,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title=dict(
                text=f"<b>Profit vs {var_col}</b>",
                font=dict(size=22, color="#2c3e50"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=dict(text=f"<b>{var_col} Steps</b>", font=dict(size=14)),
            yaxis_title=dict(text="<b>Profit</b>", font=dict(size=14)),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                title=dict(text="<b>Combinations</b>", font=dict(size=12)),
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=11),
                itemsizing='constant',
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60), # Right margin for legend
            height=650,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Segoe UI"
            )
        )

        # Add a light grid and zero line to y-axis for better visibility
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')

        # Convert figure to HTML block and append, wrapping in a nice container
        # We only want to include the plotly.js library once
        include_js = 'cdn' if var_col == variable_cols[0] else False

        html_content += f'<div class="chart-container">\n'
        html_content += fig.to_html(full_html=False, include_plotlyjs=include_js)
        html_content += f'\n</div>\n'

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
