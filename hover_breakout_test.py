import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os

# Parameters for the strategy
PARAMS = {
    'lookback_candles': 10,   # number of candles used to check the range
    'range_pips': 12,          # allowed high-low range in pips during hover
    'distance_pips': 40,        # stop loss distance in pips
    'rr': 3,              # risk to reward ratio
    'lookahead_candles': 30,   # how many candles forward to look for TP/SL
    'risk_percent': 0.01,     # percent of starting equity risked per trade
    'spread_pips': 2        # assumed spread in pips
}

START_EQUITY = 10000
PIP = 0.0001  # EURUSD pip size
PIP_VALUE = 10  # value per pip for one standard lot


def backtest(df, params):
    equity = START_EQUITY
    risk_amount = START_EQUITY * params['risk_percent']
    equity_curve = []
    equity_time = []
    trades = []

    i = params['lookback_candles']
    while i < len(df) - params['lookahead_candles']:
        # check if previous candles hovered in range
        range_high = df['High'].iloc[i - params['lookback_candles']:i].max()
        range_low = df['Low'].iloc[i - params['lookback_candles']:i].min()
        if range_high - range_low <= params['range_pips'] * PIP:
            direction = None
            if df['Close'].iloc[i] > range_high:
                direction = 'long'
            elif df['Close'].iloc[i] < range_low:
                direction = 'short'

            if direction:
                entry_time = df['Time'].iloc[i]
                entry_price = df['Close'].iloc[i]
                if direction == 'long':
                    entry_price += params['spread_pips'] * PIP
                    sl = entry_price - params['distance_pips'] * PIP
                    tp = entry_price + params['distance_pips'] * params['rr'] * PIP
                else:
                    entry_price -= params['spread_pips'] * PIP
                    sl = entry_price + params['distance_pips'] * PIP
                    tp = entry_price - params['distance_pips'] * params['rr'] * PIP

                pos_size = risk_amount / (params['distance_pips'] * PIP_VALUE)
                exit_price = None
                exit_time = None
                result = 'partial'
                for j in range(i + 1, i + params['lookahead_candles'] + 1):
                    high = df['High'].iloc[j]
                    low = df['Low'].iloc[j]
                    if direction == 'long':
                        if low <= sl:
                            exit_price = sl
                            exit_time = df['Time'].iloc[j]
                            result = 'loss'
                            break
                        if high >= tp:
                            exit_price = tp
                            exit_time = df['Time'].iloc[j]
                            result = 'win'
                            break
                    else:
                        if high >= sl:
                            exit_price = sl
                            exit_time = df['Time'].iloc[j]
                            result = 'loss'
                            break
                        if low <= tp:
                            exit_price = tp
                            exit_time = df['Time'].iloc[j]
                            result = 'win'
                            break
                if exit_price is None:
                    exit_price = df['Close'].iloc[i + params['lookahead_candles']]
                    exit_time = df['Time'].iloc[i + params['lookahead_candles']]
                profit_pips = (exit_price - entry_price) / PIP
                if direction == 'short':
                    profit_pips *= -1
                profit = profit_pips * PIP_VALUE * pos_size
                equity += profit
                trades.append({
                    'Time Open': entry_time,
                    'Open Price': entry_price,
                    'Time Close': exit_time,
                    'Close Price': exit_price,
                    'Take Profit Price': tp,
                    'Stop Loss Price': sl,
                    'Profit': profit
                })
                equity_curve.append(equity)
                equity_time.append(exit_time)
                i = i + params['lookahead_candles'] + 1
                continue
        equity_curve.append(equity)
        equity_time.append(df['Time'].iloc[i])
        i += 1

    return trades, equity_time, equity_curve


def compute_stats(trades):
    total = len(trades)
    wins = [t for t in trades if t['Profit'] > 0]
    losses = [t for t in trades if t['Profit'] < 0]
    win_rate = len(wins) / total * 100 if total else 0
    avg_win = np.mean([t['Profit'] for t in wins]) / START_EQUITY * 100 if wins else 0
    avg_loss = -np.mean([t['Profit'] for t in losses]) / START_EQUITY * 100 if losses else 0
    expectancy = (avg_win * (win_rate / 100)) - (avg_loss * (1 - win_rate / 100))

    # compute equity curve to get max drawdown
    equity = START_EQUITY
    peak = START_EQUITY
    max_dd = 0
    for t in trades:
        equity += t['Profit']
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / START_EQUITY * 100
        if drawdown > max_dd:
            max_dd = drawdown

    return {
        'Total Trades': total,
        'Win Rate': win_rate,
        'Max Drawdown': max_dd,
        'Expectancy': expectancy,
        'Average Win %': avg_win,
        'Average Loss %': avg_loss
    }


def plot_equity(times, equity, filename):
    plt.figure(figsize=(10,4))
    plt.plot(pd.to_datetime(times), equity)
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.title('Equity Curve')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def create_pdf(params, stats, equity_image, trades, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    text_y = 750
    c.setFont('Helvetica-Bold', 14)
    c.drawString(50, text_y, 'Hover Breakout Strategy Backtest')
    text_y -= 20
    c.setFont('Helvetica', 10)
    c.drawString(50, text_y, 'Parameters:')
    text_y -= 15
    for k, v in params.items():
        c.drawString(60, text_y, f'{k}: {v}')
        text_y -= 12
    text_y -= 10
    c.drawString(50, text_y, 'Results:')
    text_y -= 15
    for k, v in stats.items():
        c.drawString(60, text_y, f'{k}: {v:.2f}')
        text_y -= 12
    text_y -= 10
    c.drawImage(equity_image, 50, text_y-200, width=500, height=200)
    c.showPage()
    c.save()


def main():
    df = pd.read_csv('EURUSD_M30_Data.csv')
    trades, times, equity = backtest(df, PARAMS)
    stats = compute_stats(trades)

    equity_img = 'equity_curve.png'
    plot_equity(times, equity, equity_img)

    create_pdf(PARAMS, stats, equity_img, trades, 'hover_breakout_report.pdf')

    trade_df = pd.DataFrame(trades)
    trade_df.to_csv('tradelog_hover_breakout.csv', index=False)
    if os.path.exists(equity_img):
        os.remove(equity_img)


if __name__ == '__main__':
    main()
