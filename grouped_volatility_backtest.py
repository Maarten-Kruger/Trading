import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Strategy Parameters
BACK_CANDLES = 22          # look back this many candles for previous big candle
CANDLE_SIZE_PIPS = 50      # minimum size for a "large" candle
TP_PIPS = 50               # take profit distance
SL_PIPS = 50               # stop loss distance
FUTURE_CANDLES = 20        # how many candles ahead to check for TP/SL
SPREAD = 0.0002            # 2 pips spread
FOLLOW_DIRECTION = False   # follow candle direction, False for opposite

RISK_PERCENT = 0.02        # percent of starting equity risked per trade
STARTING_EQUITY = 10000

DATA_FILE = 'EURUSD_M30_Data.csv'


def backtest(
    back_candles: int = BACK_CANDLES,
    candle_size_pips: int = CANDLE_SIZE_PIPS,
    tp_pips: int = TP_PIPS,
    sl_pips: int = SL_PIPS,
    future_candles: int = FUTURE_CANDLES,
    follow_direction: bool = FOLLOW_DIRECTION,
    generate_files: bool = True,
):
    df = pd.read_csv(DATA_FILE, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)

    equity = STARTING_EQUITY
    risk_amount = STARTING_EQUITY * RISK_PERCENT
    equity_curve = []
    trade_log = []

    size_threshold = candle_size_pips / 10000

    for idx in range(back_candles, len(df) - future_candles):
        window = df.iloc[idx - back_candles:idx]
        # look for most recent large candle within lookback window
        large_candles = [i for i in range(len(window))
                         if window['High'].iloc[i] - window['Low'].iloc[i] >= size_threshold]
        if not large_candles:
            continue
        last_large_index = large_candles[-1]
        candle_open = window['Open'].iloc[last_large_index]
        candle_close = window['Close'].iloc[last_large_index]
        direction = 1 if candle_close > candle_open else -1
        if not follow_direction:
            direction *= -1

        entry_time = df['Time'].iloc[idx]
        entry_price = df['Open'].iloc[idx] + direction * (SPREAD / 2)
        tp_price = entry_price + direction * tp_pips / 10000
        sl_price = entry_price - direction * sl_pips / 10000

        exit_time = df['Time'].iloc[idx + future_candles]
        close_price = df['Close'].iloc[idx + future_candles] - direction * (SPREAD / 2)
        outcome = 'partial'

        for j in range(0, future_candles + 1):
            high = df['High'].iloc[idx + j]
            low = df['Low'].iloc[idx + j]

            # check for both TP and SL in same bar and assume SL hit first
            if direction == 1:
                if high >= tp_price and low <= sl_price:
                    close_price = sl_price
                    exit_time = df['Time'].iloc[idx + j]
                    outcome = 'sl'
                    break
                if low <= sl_price:
                    close_price = sl_price
                    exit_time = df['Time'].iloc[idx + j]
                    outcome = 'sl'
                    break
                if high >= tp_price:
                    close_price = tp_price
                    exit_time = df['Time'].iloc[idx + j]
                    outcome = 'tp'
                    break
            else:
                if low <= tp_price and high >= sl_price:
                    close_price = sl_price
                    exit_time = df['Time'].iloc[idx + j]
                    outcome = 'sl'
                    break
                if high >= sl_price:
                    close_price = sl_price
                    exit_time = df['Time'].iloc[idx + j]
                    outcome = 'sl'
                    break
                if low <= tp_price:
                    close_price = tp_price
                    exit_time = df['Time'].iloc[idx + j]
                    outcome = 'tp'
                    break

        pnl_pips = (close_price - entry_price) * direction * 10000
        pnl_risk_multiple = pnl_pips / sl_pips
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

    total_trades = len(trade_log)
    wins = [t for t in trade_log if t['Profit/Loss'] > 0]
    profits = [t['Profit/Loss'] for t in trade_log]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    expectancy = np.mean(profits) / STARTING_EQUITY * 100 if profits else 0
    avg_win = np.mean([p for p in profits if p > 0]) / STARTING_EQUITY * 100 if wins else 0
    avg_loss = np.mean([abs(p) for p in profits if p < 0]) / STARTING_EQUITY * 100 if len(profits) > len(wins) else 0

    if equity_curve:
        eq_values = [e for _, e in equity_curve]
        peaks = np.maximum.accumulate(eq_values)
        drawdowns = 100 * (peaks - eq_values) / STARTING_EQUITY
        max_drawdown = np.max(drawdowns)
    else:
        max_drawdown = 0
        eq_values = []

    results = {
        'Final Equity': equity,
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Average Win Size': avg_win,
        'Average Loss Size': avg_loss,
        'Max Drawdown': max_drawdown,
    }

    if generate_files:
        if equity_curve:
            times = [t for t, _ in equity_curve]
            plt.figure(figsize=(10, 4))
            plt.plot(times, eq_values)
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity ($)')
            plt.tight_layout()
            plt.savefig('equity_curve_grouped.png')
            plt.close()
        else:
            plt.figure()
            plt.savefig('equity_curve_grouped.png')
            plt.close()

        c = canvas.Canvas('grouped_volatility_results.pdf', pagesize=letter)
        width, height = letter
        y = height - 40
        c.drawString(40, y, 'Grouped Volatility Strategy Results')
        y -= 20
        c.drawString(40, y, 'Strategy Parameters:')
        y -= 15
        for label, val in [
            ('BACK_CANDLES', back_candles),
            ('CANDLE_SIZE_PIPS', candle_size_pips),
            ('TP_PIPS', tp_pips),
            ('SL_PIPS', sl_pips),
            ('FUTURE_CANDLES', future_candles),
            ('SPREAD', SPREAD),
            ('FOLLOW_DIRECTION', follow_direction),
            ('RISK_PERCENT', RISK_PERCENT),
            ('STARTING_EQUITY', STARTING_EQUITY),
        ]:
            c.drawString(60, y, f'{label} = {val}')
            y -= 15

        y -= 10
        c.drawString(40, y, f'Total Trades: {total_trades}')
        y -= 20
        c.drawString(40, y, f'Win Rate: {win_rate:.2f}% - percent of trades profitable')
        y -= 20
        c.drawString(40, y, f'Max Drawdown: {max_drawdown:.2f}% - worst equity drop')
        y -= 20
        c.drawString(40, y, f'Expectancy: {expectancy:.2f}% - average gain per trade')
        y -= 20
        c.drawString(40, y, f'Average Win Size: {avg_win:.2f}%')
        y -= 20
        c.drawString(40, y, f'Average Loss Size: {avg_loss:.2f}%')
        y -= 40
        c.drawImage('equity_curve_grouped.png', 40, y - 300, width=500, height=300)
        c.save()

        pd.DataFrame(trade_log).to_csv('tradelog_GroupedVolatility.csv', index=False)

    return results


if __name__ == '__main__':
    backtest()
