import json
import pandas as pd


def load_market_data(filename: str = 'EURUSD_M30_Data.csv') -> pd.DataFrame:
    """Load and sort market data."""
    df = pd.read_csv(filename, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)
    return df


def hover_breakout_strategy(
    df: pd.DataFrame,
    *,
    back_candles: int = 10,
    range_pips: int = 8,
    tp_pips: int = 12,
    sl_pips: int = 20,
    future_candles: int = 12,
    spread: float = 0.0002,
    save_log: bool = True,
    log_filename: str = "tradelog_HoverBreakout.csv",
) -> pd.DataFrame:
    """Execute the Hover Breakout strategy and return trades as DataFrame."""
    trades = []
    pip = 0.0001

    for idx in range(back_candles, len(df) - future_candles):
        window = df.iloc[idx - back_candles:idx]
        if (window['High'].max() - window['Low'].min()) <= range_pips * pip:
            range_high = window['High'].max()
            range_low = window['Low'].min()
            close = df['Close'].iloc[idx]
            direction = 0
            if close > range_high:
                direction = 1
            elif close < range_low:
                direction = -1
            if direction == 0:
                continue

            entry_time = df['Time'].iloc[idx]
            entry_price = close + direction * (spread / 2)
            tp_price = entry_price + direction * tp_pips * pip
            sl_price = entry_price - direction * sl_pips * pip

            close_time = df['Time'].iloc[idx + future_candles]
            close_price = df['Close'].iloc[idx + future_candles] - direction * (spread / 2)
            status = 'partial'

            for j in range(1, future_candles + 1):
                high = df['High'].iloc[idx + j]
                low = df['Low'].iloc[idx + j]
                if direction == 1:
                    if high >= tp_price:
                        close_price = tp_price
                        close_time = df['Time'].iloc[idx + j]
                        status = 'tp'
                        break
                    if low <= sl_price:
                        close_price = sl_price
                        close_time = df['Time'].iloc[idx + j]
                        status = 'sl'
                        break
                else:
                    if low <= tp_price:
                        close_price = tp_price
                        close_time = df['Time'].iloc[idx + j]
                        status = 'tp'
                        break
                    if high >= sl_price:
                        close_price = sl_price
                        close_time = df['Time'].iloc[idx + j]
                        status = 'sl'
                        break

            pip_diff = (close_price - entry_price) * direction / pip

            trades.append({
                'Time Open': entry_time,
                'Open Price': entry_price,
                'Time Close': close_time,
                'Close Price': close_price,
                'Pip PnL': pip_diff,
                'Status': status,
                'SL': sl_price,
                'TP': tp_price,
            })

    df_trades = pd.DataFrame(trades)
    if save_log:
        df_trades.to_csv(log_filename, index=False)
    return df_trades


if __name__ == '__main__':
    df = load_market_data()
    params = {
        'Back Candles': 10,
        'Range Pips': 8,
        'TP Pips': 12,
        'SL Pips': 20,
        'Future Candles': 12,
        'Spread': 0.0002,
    }
    hover_breakout_strategy(
        df,
        back_candles=params['Back Candles'],
        range_pips=params['Range Pips'],
        tp_pips=params['TP Pips'],
        sl_pips=params['SL Pips'],
        future_candles=params['Future Candles'],
        spread=params['Spread'],
        save_log=True,
        log_filename='tradelog_HoverBreakout.csv',
    )
    with open('strategy_params.json', 'w') as f:
        json.dump(params, f)
