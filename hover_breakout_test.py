import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

# Strategy parameters (all values in pips or bars)
LOOKBACK = 10             # number of 30 minute periods to check for tight range
RANGE_THRESHOLD_PIPS = 20 # maximum high-low range to qualify as hovering
STOP_LOSS_PIPS = 8        # stop loss distance
TAKE_PROFIT_PIPS = 50     # take profit distance
HOLD_PERIOD = 12          # number of 30 minute periods to hold trade if TP/SL not hit
SPREAD_PIPS = 2           # assumed spread cost per trade

# Account parameters
RISK_PER_TRADE = 0.03     # fraction of equity to risk per trade
INITIAL_EQUITY = 10000.0  # starting demo account

PIP_SIZE = 0.0001         # EURUSD pip size
PIP_VALUE_PER_LOT = 10    # USD per pip for 1 standard lot
BARS_PER_CHECK = 30       # number of 1 minute candles per 30 minute period


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



def simulate_strategy(df, lookback, range_threshold_pips, stop_loss_pips,
                      take_profit_pips, hold_period, spread_pips,
                      risk_per_trade, initial_equity, bars_per_check=BARS_PER_CHECK):
    equity = initial_equity

    # Use a fixed risk amount based on the starting equity
    fixed_risk_amount = initial_equity * risk_per_trade

    equity_curve = [equity]
    times = [df['Time'].iloc[0]]
    trades = []


    half_spread = (spread_pips / 2) * PIP_SIZE

    step = bars_per_check
    lookback_bars = lookback * step
    hold_bars = hold_period * step

    for i in range(lookback_bars, len(df) - hold_bars, step):
        range_high = df['High'].iloc[i-lookback_bars:i].max()
        range_low = df['Low'].iloc[i-lookback_bars:i].min()
        if range_high - range_low <= range_threshold_pips * PIP_SIZE:
            current_close = df['Close'].iloc[i]
            breakout = None
            if current_close > range_high:
                breakout = 'long'
            elif current_close < range_low:
                breakout = 'short'
            if breakout:

                # Open the trade on the same candle that triggered the breakout
                entry_index = i
                direction = 1 if breakout == 'long' else -1
                entry_price_raw = df['Close'].iloc[entry_index]
                entry_price = entry_price_raw + half_spread * direction
                entry_time = df['Time'].iloc[entry_index]
                sl = entry_price - stop_loss_pips * PIP_SIZE if breakout == 'long' else entry_price + stop_loss_pips * PIP_SIZE
                tp = entry_price + take_profit_pips * PIP_SIZE if breakout == 'long' else entry_price - take_profit_pips * PIP_SIZE

                # Position size is based on the fixed risk amount so each trade
                # risks the same dollar value regardless of account growth.
                lot_size = fixed_risk_amount / (stop_loss_pips * PIP_VALUE_PER_LOT)

                trade_equity_start = equity
                result_pips = None
                exit_price = None
                exit_time = None

                for j in range(entry_index, min(entry_index + hold_bars + 1, len(df))):
                    candle = df.iloc[j]
                    if breakout == 'long':
                        if candle['Low'] <= sl:
                            exit_raw = sl
                            exit_time = candle['Time']
                            exit_price = exit_raw - half_spread
                            result_pips = (exit_price - entry_price) / PIP_SIZE
                            break
                        if candle['High'] >= tp:
                            exit_raw = tp
                            exit_time = candle['Time']
                            exit_price = exit_raw - half_spread
                            result_pips = (exit_price - entry_price) / PIP_SIZE
                            break
                    else:
                        if candle['High'] >= sl:
                            exit_raw = sl
                            exit_time = candle['Time']
                            exit_price = exit_raw + half_spread
                            result_pips = (entry_price - exit_price) / PIP_SIZE
                            break
                        if candle['Low'] <= tp:
                            exit_raw = tp
                            exit_time = candle['Time']
                            exit_price = exit_raw + half_spread
                            result_pips = (entry_price - exit_price) / PIP_SIZE
                            break

                if result_pips is None:
                    end_index = min(entry_index + hold_bars, len(df) - 1)
                    candle = df.iloc[end_index]
                    exit_raw = candle['Close']
                    exit_time = candle['Time']
                    if breakout == 'long':
                        exit_price = exit_raw - half_spread
                        result_pips = (exit_price - entry_price) / PIP_SIZE
                    else:
                        exit_price = exit_raw + half_spread
                        result_pips = (entry_price - exit_price) / PIP_SIZE
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
    df = load_data('EURUSD_M1_Data.csv')
    trade_df, equity_curve, times = simulate_strategy(
        df,
        LOOKBACK,
        RANGE_THRESHOLD_PIPS,
        STOP_LOSS_PIPS,
        TAKE_PROFIT_PIPS,
        HOLD_PERIOD,
        SPREAD_PIPS,
        RISK_PER_TRADE,
        INITIAL_EQUITY,
        BARS_PER_CHECK,
    )
    metrics = calculate_metrics(trade_df, equity_curve)
    plot_equity_curve(times, equity_curve)
    generate_report(metrics, STRATEGY_PARAMS, 'equity_curve.png', 'Hover_Breakout_Strategy_Report.pdf')
    trade_df.to_csv('tradelog_Hover_Breakout_Strategy.csv', index=False)


if __name__ == '__main__':
    main()
