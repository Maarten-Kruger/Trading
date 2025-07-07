import pandas as pd
import numpy as np
import plotly.graph_objects as go

SAMPLE_SIZE = 50
CSV_FILE = "EURUSD_M30_Data.csv"


def main() -> None:
    df = pd.read_csv(CSV_FILE)
    df["Time"] = pd.to_datetime(df["Time"])
    sample = df.head(SAMPLE_SIZE)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=sample["Time"],
                open=sample["Open"],
                high=sample["High"],
                low=sample["Low"],
                close=sample["Close"],
            )
        ]
    )
    fig.update_layout(title=f"First {SAMPLE_SIZE} EURUSD 30m bars")
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    table_html = sample.to_html(index=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>EURUSD Sample Candles</title>
</head>
<body>
    <h1>EURUSD {SAMPLE_SIZE} Bar Sample</h1>
    {chart_html}
    <h2>Sample Data</h2>
    {table_html}
</body>
</html>
"""

    with open("sample_candles.html", "w") as f:
        f.write(html)
    print("Created sample_candles.html")


if __name__ == "__main__":
    main()
