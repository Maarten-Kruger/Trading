import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Default Strategy Parameters
BACK_CANDLES = 10     # look back this many candles for tight range
RANGE_PIPS = 8        # range size threshold in pips
TP_PIPS = 12          # take profit distance
SL_PIPS = 20          # stop loss distance
FUTURE_CANDLES = 12   # bars to check in the future for exit
SPREAD = 0.0002       # broker spread in price units

RISK_PERCENT = 0.03
STARTING_EQUITY = 10000

DATA_FILE = 'EURUSD_M30_Data.csv'


def hover_breakout_backtest(
    back_candles: int = BACK_CANDLES,
    range_pips: int = RANGE_PIPS,
    tp_pips: int = TP_PIPS,
    sl_pips: int = SL_PIPS,
    future_candles: int = FUTURE_CANDLES,
    generate_files: bool = True,
):
    """Run the hover breakout strategy backtest."""

    df = pd.read_csv(DATA_FILE, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)

    equity = STARTING_EQUITY
    risk_per_trade = STARTING_EQUITY * RISK_PERCENT
    equity_curve = []
    trade_log = []

    for idx in range(back_candles, len(df) - future_candles):
        window = df.iloc[idx - back_candles:idx]
        if (window['High'].max() - window['Low'].min()) <= range_pips / 10000:
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
            tp_price = entry_price + direction * tp_pips / 10000
            sl_price = entry_price - direction * sl_pips / 10000

            exit_time = df['Time'].iloc[idx + future_candles]
            close_price = df['Close'].iloc[idx + future_candles] - direction * (SPREAD / 2)
            outcome = 'partial'

            for j in range(1, future_candles + 1):
                high = df['High'].iloc[idx + j]
                low = df['Low'].iloc[idx + j]
                if direction == 1:
                    if high >= tp_price:
                        close_price = tp_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'tp'
                        break
                    if low <= sl_price:
                        close_price = sl_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'sl'
                        break
                else:
                    if low <= tp_price:
                        close_price = tp_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'tp'
                        break
                    if high >= sl_price:
                        close_price = sl_price
                        exit_time = df['Time'].iloc[idx + j]
                        outcome = 'sl'
                        break

            pnl_pips = (close_price - entry_price) * direction * 10000
            pnl_multiple = pnl_pips / sl_pips
            profit = pnl_multiple * risk_per_trade
            equity += profit

            trade_log.append({
                'Time Open': entry_time,
                'Open Price': entry_price,
                'Time Close': exit_time,
                'Close Price': close_price,
                'Take Profit Price': tp_price,
                'Stop Loss Price': sl_price,
                'Profit/Loss': profit,
                'Outcome': outcome,
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

    if equity_curve:
        eq_values = [e for _, e in equity_curve]
        peaks = np.maximum.accumulate(eq_values)
        drawdowns = 100 * (peaks - eq_values) / STARTING_EQUITY
        max_drawdown = np.max(drawdowns)
    else:
        max_drawdown = 0

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
            plt.savefig('equity_curve_new.png')
            plt.close()
        else:
            plt.figure()
            plt.savefig('equity_curve_new.png')
            plt.close()

        c = canvas.Canvas('hover_breakout_results_new.pdf', pagesize=letter)
        width, height = letter
        y = height - 40
        c.drawString(40, y, 'Hover Breakout Strategy Results')
        y -= 20
        c.drawString(40, y, 'Strategy Parameters:')
        y -= 15
        for label, val in [
            ('BACK_CANDLES', back_candles),
            ('RANGE_PIPS', range_pips),
            ('TP_PIPS', tp_pips),
            ('SL_PIPS', sl_pips),
            ('FUTURE_CANDLES', future_candles),
            ('SPREAD', SPREAD),
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
        c.drawString(40, y, f'Expectancy: {expectancy:.2f}% - avg gain per trade')
        y -= 20
        c.drawString(40, y, f'Average Win Size: {avg_win:.2f}%')
        y -= 20
        c.drawString(40, y, f'Average Loss Size: {avg_loss:.2f}%')
        y -= 40
        c.drawImage('equity_curve_new.png', 40, y - 300, width=500, height=300)
        c.save()

        pd.DataFrame(trade_log).to_csv('tradelog_HoverBreakout_new.csv', index=False)

    return results


if __name__ == '__main__':
    hover_breakout_backtest()
