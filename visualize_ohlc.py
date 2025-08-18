#!/usr/bin/env python3
"""Simple OHLC candlestick viewer with scroll navigation.

Usage:
    python visualize_ohlc.py [path/to/data.csv]

If no path is supplied, the script prompts for one on startup. Scroll up to
move forward through candles and scroll down to go backward. The CSV file must
contain columns: ``Open``, ``High``, ``Low`` and ``Close`` (case-sensitive).
"""

from __future__ import annotations

import argparse
from typing import Set

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


class OHLCViewer:
    """Interactive candlestick simulator using matplotlib."""

    def __init__(self, df: pd.DataFrame, window: int = 50) -> None:
        self.df = df.reset_index(drop=True)
        self.window = window
        self.start = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.draw()

    def draw(self) -> None:
        """Draw the current window of candles."""
        self.ax.clear()
        segment = self.df.iloc[self.start : self.start + self.window]

        for i, row in segment.iterrows():
            idx = i - self.start
            color = "green" if row.Close >= row.Open else "red"
            self.ax.add_line(Line2D([idx, idx], [row.Low, row.High], color=color))
            rect = Rectangle(
                (idx - 0.3, min(row.Open, row.Close)),
                0.6,
                abs(row.Close - row.Open),
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            self.ax.add_patch(rect)

        self.ax.set_xlim(-1, self.window)
        segment_high = segment.High.max()
        segment_low = segment.Low.min()
        pad = (segment_high - segment_low) * 0.05
        self.ax.set_ylim(segment_low - pad, segment_high + pad)
        self.ax.set_title("OHLC Candlestick Viewer")
        self.ax.set_xlabel("Candle")
        self.ax.set_ylabel("Price")
        self.fig.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        """Handle scroll events to navigate data."""
        if event.button == "up":
            self.start = min(len(self.df) - self.window, self.start + 1)
        elif event.button == "down":
            self.start = max(0, self.start - 1)
        self.draw()

    def show(self) -> None:
        plt.show()


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required: Set[str] = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive OHLC viewer")
    parser.add_argument("csv", nargs="?", help="Path to OHLC CSV file")
    parser.add_argument(
        "--window", type=int, default=50, help="Number of candles to display"
    )
    args = parser.parse_args()

    csv_path = args.csv or input("Path to OHLC CSV file: ").strip()
    df = load_csv(csv_path)
    viewer = OHLCViewer(df, window=args.window)
    viewer.show()


if __name__ == "__main__":
    main()
