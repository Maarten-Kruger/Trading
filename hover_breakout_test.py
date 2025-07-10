import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

# Strategy parameters (all values in pips or bars)
LOOKBACK = 6              # number of candles to check for tight range
RANGE_THRESHOLD_PIPS = 38 # maximum high-low range to qualify as hovering
STOP_LOSS_PIPS = 9        # stop loss distance
TAKE_PROFIT_PIPS = 25     # take profit distance
HOLD_PERIOD = 12          # number of candles to hold trade if TP/SL not hit
SPREAD_PIPS = 2           # assumed spread cost per trade

# Account parameters
RISK_PER_TRADE = 0.01     # fraction of equity to risk per trade
INITIAL_EQUITY = 10000.0  # starting demo account

PIP_SIZE = 0.0001         # EURUSD pip size
PIP_VALUE_PER_LOT = 10    # USD per pip for 1 standard lot


STRATEGY_PARAMS = {
    'Lookback': LOOKBACK,
    'Range Threshold (pips)': RANGE_THRESHOLD_PIPS,
    'Stop Loss (pips)': STOP_LOSS_PIPS,
    'Take Profit (pips)': TAKE_PROFIT_PIPS,
    'Hold Period (bars)': HOLD_PERIOD,
    'Spread (pips)': SPREAD_PIPS,
    'Risk Per Trade': RISK_PER_TRADE,
    'Initial Equity': INITIAL_EQUITY
}


def load_data(path):
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def simulate_strategy(df):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    times = [df['Time'].iloc[0]]
    trades = []

    for i in range(LOOKBACK, len(df) - HOLD_PERIOD - 1):
        range_high = df['High'].iloc[i-LOOKBACK:i].max()
        range_low = df['Low'].iloc[i-LOOKBACK:i].min()
        if range_high - range_low <= RANGE_THRESHOLD_PIPS * PIP_SIZE:
            current_close = df['Close'].iloc[i]
            breakout = None
            if current_close > range_high:
                breakout = 'long'
            elif current_close < range_low:
                breakout = 'short'
            if breakout:

                entry_price = df['Open'].iloc[i + 1]
                entry_time = df['Time'].iloc[i + 1]
                sl = entry_price - STOP_LOSS_PIPS * PIP_SIZE if breakout == 'long' else entry_price + STOP_LOSS_PIPS * PIP_SIZE
                tp = entry_price + TAKE_PROFIT_PIPS * PIP_SIZE if breakout == 'long' else entry_price - TAKE_PROFIT_PIPS * PIP_SIZE

                risk_amount = equity * RISK_PER_TRADE
                lot_size = risk_amount / (STOP_LOSS_PIPS * PIP_VALUE_PER_LOT)
                trade_equity_start = equity
                result_pips = None
                exit_price = None
                exit_time = None

                for j in range(i+1, i+1+HOLD_PERIOD):
                    candle = df.iloc[j]
                    if breakout == 'long':
                        if candle['Low'] <= sl:
                            result_pips = -STOP_LOSS_PIPS
                            exit_price = sl
                            exit_time = candle['Time']
                            break
                        if candle['High'] >= tp:
                            result_pips = TAKE_PROFIT_PIPS
                            exit_price = tp
                            exit_time = candle['Time']
                            break
                    else:
                        if candle['High'] >= sl:
                            result_pips = -STOP_LOSS_PIPS
                            exit_price = sl
                            exit_time = candle['Time']
                            break
                        if candle['Low'] <= tp:
                            result_pips = TAKE_PROFIT_PIPS
                            exit_price = tp
                            exit_time = candle['Time']
                            break

                if result_pips is None:
                    candle = df.iloc[i+HOLD_PERIOD]
                    exit_price = candle['Close']
                    exit_time = candle['Time']
                    if breakout == 'long':
                        result_pips = (exit_price - entry_price) / PIP_SIZE
                    else:
                        result_pips = (entry_price - exit_price) / PIP_SIZE

                result_pips -= SPREAD_PIPS
                profit = result_pips * PIP_VALUE_PER_LOT * lot_size
                equity += profit

                trades.append({
                    'Time Open': entry_time,
                    'Open Price': entry_price,
                    'Time Close': exit_time,
                    'Close Price': exit_price,
                    'Take Profit Price': tp,
                    'Stop Loss Price': sl,
                    'Profit': profit,
                    'Trade Impact': profit / trade_equity_start,
                    'Account Size': equity
                })
                equity_curve.append(equity)
                times.append(exit_time)

    trade_df = pd.DataFrame(trades)
    return trade_df, equity_curve, times


def calculate_metrics(trade_df, equity_curve):
    total_trades = len(trade_df)
    wins = trade_df[trade_df['Profit'] > 0]
    win_rate = len(wins) / total_trades if total_trades else 0
    avg_win = wins['Trade Impact'].mean() if not wins.empty else 0
    losses = trade_df[trade_df['Profit'] <= 0]
    avg_loss = losses['Trade Impact'].mean() if not losses.empty else 0
    expectancy = (avg_win * win_rate) + (avg_loss * (1 - win_rate))

    peak = equity_curve[0]
    drawdowns = []
    max_drawdown = 0
    for bal in equity_curve:
        if bal > peak:
            peak = bal
        dd = (bal - peak) / peak
        drawdowns.append(dd)
        if dd < max_drawdown:
            max_drawdown = dd

    metrics = {
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Expectancy': expectancy,
        'Average Win Size': avg_win,
        'Average Loss Size': avg_loss
    }
    return metrics


def plot_equity_curve(times, equity_curve, path='equity_curve.png'):
    plt.figure(figsize=(10,5))
    plt.plot(times, equity_curve)
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.title('Demo Account Equity Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def generate_report(metrics, params, path_img, output_pdf):

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    elements = []
    elements.append(Paragraph('Hover Breakout Strategy Report', styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph('Strategy Parameters', styles['Heading2']))
    for key, value in params.items():
        elements.append(Paragraph(f"{key}: {value}", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph('Performance Metrics', styles['Heading2']))

    for key, value in metrics.items():
        if key == 'Win Rate' or 'Expectancy' in key or 'Drawdown' in key or 'Size' in key:
            text = f"{key}: {value*100:.2f}%"
        else:
            text = f"{key}: {value}"
        elements.append(Paragraph(text, styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Image(path_img, width=500, height=300))
    doc.build(elements)


def main():
    df = load_data('EURUSD_M30_Data.csv')
    trade_df, equity_curve, times = simulate_strategy(df)
    metrics = calculate_metrics(trade_df, equity_curve)
    plot_equity_curve(times, equity_curve)


    trade_df.to_csv('tradelog_Hover_Breakout_Strategy.csv', index=False)


if __name__ == '__main__':
    main()
