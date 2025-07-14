import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Strategy Parameters
BACK_CANDLES = 10          # number of candles to look back for range
RANGE_PIPS = 40            # tight range maximum size in pips
TP_PIPS = 50               # distance to take profit in pips
SL_PIPS = 8                # distance to stop loss in pips
FUTURE_CANDLES = 12        # how many candles to look forward for TP/SL
SPREAD = 0.0002            # 2 pips spread

RISK_PERCENT = 0.03       # risk 1% of starting equity per trade
STARTING_EQUITY = 10000   # account starts with $10,000

DATA_FILE = 'EURUSD_M30_Data.csv'


def backtest():
    df = pd.read_csv(DATA_FILE, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)

    equity = STARTING_EQUITY
    risk_amount = STARTING_EQUITY * RISK_PERCENT
    equity_curve = []
    trade_log = []

    for idx in range(BACK_CANDLES, len(df) - FUTURE_CANDLES):
        window = df.iloc[idx - BACK_CANDLES:idx]
        if (window['High'].max() - window['Low'].min()) <= RANGE_PIPS / 10000:
            current_close = df['Close'].iloc[idx]
            range_high = window['High'].max()
            range_low = window['Low'].min()
            direction = 0
            if current_close > range_high:
                direction = 1
            elif current_close < range_low:
                direction = -1
            if direction == 0:
                continue

            entry_time = df['Time'].iloc[idx]
            entry_price = current_close + direction * (SPREAD / 2)
            tp_price = entry_price + direction * TP_PIPS / 10000
            sl_price = entry_price - direction * SL_PIPS / 10000
            exit_time = df['Time'].iloc[idx + FUTURE_CANDLES]
            close_price = df['Close'].iloc[idx + FUTURE_CANDLES] - direction * (SPREAD / 2)
            outcome = 'partial'

            for j in range(1, FUTURE_CANDLES + 1):
                bar_high = df['High'].iloc[idx + j]
                bar_low = df['Low'].iloc[idx + j]
                if direction == 1:
                    if bar_high >= tp_price:
                        close_price = tp_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'tp'
                        break
                    if bar_low <= sl_price:
                        close_price = sl_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'sl'
                        break
                else:
                    if bar_low <= tp_price:
                        close_price = tp_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'tp'
                        break
                    if bar_high >= sl_price:
                        close_price = sl_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'sl'
                        break

            pnl_pips = (close_price - entry_price) * direction * 10000
            pnl_risk_multiple = pnl_pips / SL_PIPS
            pnl_money = pnl_risk_multiple * risk_amount
            equity += pnl_money
            trade_log.append({
                'Time Open': entry_time,
                'Open Price': entry_price,
                'Time Close': exit_time,
                'Close Price': close_price,
                'Take Profit Price': tp_price,
                'Stop Loss Price': sl_price,
                'Profit/Loss': pnl_money
            })
            equity_curve.append((exit_time, equity))

    # === Statistics ===
    total_trades = len(trade_log)
    wins = [t for t in trade_log if t['Profit/Loss'] > 0]
    profits = [t['Profit/Loss'] for t in trade_log]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    expectancy = np.mean(profits) / STARTING_EQUITY * 100 if profits else 0
    avg_win = np.mean([p for p in profits if p > 0]) / STARTING_EQUITY * 100 if wins else 0
    avg_loss = np.mean([abs(p) for p in profits if p < 0]) / STARTING_EQUITY * 100 if len(profits) > len(wins) else 0

    eq_values = [e for _, e in equity_curve]
    if eq_values:
        peaks = np.maximum.accumulate(eq_values)
        drawdowns = 100 * (peaks - eq_values) / STARTING_EQUITY
        max_drawdown = np.max(drawdowns)
    else:
        max_drawdown = 0

    # === Plot equity curve ===
    if equity_curve:
        times = [t for t, _ in equity_curve]
        plt.figure(figsize=(10, 4))
        plt.plot(times, eq_values)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.tight_layout()
        plt.savefig('equity_curve.png')
    else:
        plt.figure()
        plt.savefig('equity_curve.png')

    # === PDF report ===
    c = canvas.Canvas('hover_breakout_results.pdf', pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(40, y, 'Hover Breakout Strategy Results')
    y -= 20
    c.drawString(40, y, f'Starting Equity: ${STARTING_EQUITY}')
    y -= 20
    c.drawString(40, y, f'Total Trades: {total_trades}')
    y -= 20
    c.drawString(40, y, f'Win Rate: {win_rate:.2f}% - percent of trades profitable')
    y -= 20
    c.drawString(40, y, f'Max Drawdown: {max_drawdown:.2f}% - worst equity loss')
    y -= 20
    c.drawString(40, y, f'Expectancy: {expectancy:.2f}% - average gain per trade')
    y -= 20
    c.drawString(40, y, f'Average Win Size: {avg_win:.2f}% - average winning trade')
    y -= 20
    c.drawString(40, y, f'Average Loss Size: {avg_loss:.2f}% - average losing trade')
    y -= 40
    c.drawImage('equity_curve.png', 40, y - 300, width=500, height=300)
    c.save()

    pd.DataFrame(trade_log).to_csv('tradelog_HoverBreakout.csv', index=False)


if __name__ == '__main__':
    backtest()
