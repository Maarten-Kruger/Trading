import json
import pandas as pd


def load_market_data(filename: str = 'EURUSD_M30_Data.csv') -> pd.DataFrame:
    """Load and sort market data."""
    df = pd.read_csv(filename, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)
    return df


def grouped_volatility_strategy(
    df: pd.DataFrame,
    *,
    back_candles: int = 20,
    candle_size_pips: int = 30,
    tp_pips: int = 30,
    sl_pips: int = 30,
    future_candles: int = 20,
    follow_direction: bool = True,
    spread: float = 0.0002,
    save_log: bool = True,
    log_filename: str = 'tradelog_GroupedVolatility.csv',
) -> pd.DataFrame:
    """Execute the Grouped Volatility strategy and return trades as DataFrame."""

    trades = []
    pip = 0.0001
    size_threshold = candle_size_pips * pip

    for idx in range(back_candles, len(df) - future_candles):
        window = df.iloc[idx - back_candles:idx]
        large_indices = [
            i for i in range(len(window))
            if abs(window['Close'].iloc[i] - window['Open'].iloc[i]) >= size_threshold
        ]
        if not large_indices:
            continue
        last_large = large_indices[-1]
        candle_open = window['Open'].iloc[last_large]
        candle_close = window['Close'].iloc[last_large]
        direction = 1 if candle_close > candle_open else -1
        if not follow_direction:
            direction *= -1

        entry_time = df['Time'].iloc[idx]
        entry_price = df['Open'].iloc[idx] + direction * (spread / 2)
        tp_price = entry_price + direction * tp_pips * pip
        sl_price = entry_price - direction * sl_pips * pip

        close_time = df['Time'].iloc[idx + future_candles]
        close_price = df['Close'].iloc[idx + future_candles] - direction * (spread / 2)
        status = 'partial'

        for j in range(1, future_candles + 1):
            high = df['High'].iloc[idx + j]
            low = df['Low'].iloc[idx + j]
            if direction == 1:
                if high >= tp_price and low <= sl_price:
                    close_price = sl_price
                    close_time = df['Time'].iloc[idx + j]
                    status = 'sl'
                    break
                if low <= sl_price:
                    close_price = sl_price
                    close_time = df['Time'].iloc[idx + j]
                    status = 'sl'
                    break
                if high >= tp_price:
                    close_price = tp_price
                    close_time = df['Time'].iloc[idx + j]
                    status = 'tp'
                    break
            else:
                if low <= tp_price and high >= sl_price:
                    close_price = sl_price
                    close_time = df['Time'].iloc[idx + j]
                    status = 'sl'
                    break
                if high >= sl_price:
                    close_price = sl_price
                    close_time = df['Time'].iloc[idx + j]
                    status = 'sl'
                    break
                if low <= tp_price:
                    close_price = tp_price
                    close_time = df['Time'].iloc[idx + j]
                    status = 'tp'
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

    columns = [
        'Time Open',
        'Open Price',
        'Time Close',
        'Close Price',
        'Pip PnL',
        'Status',
        'SL',
        'TP',
    ]
    df_trades = pd.DataFrame(trades, columns=columns)

    if save_log:
        df_trades.to_csv(log_filename, index=False)
    return df_trades


if __name__ == '__main__':
    df = load_market_data()
    params = {
        'Back Candles': 20,
        'Candle Size Pips': 30,
        'TP Pips': 40,
        'SL Pips': 25,
        'Future Candles': 30,
        'Follow Direction': False,
        'Spread': 0.0002,
    }
    grouped_volatility_strategy(
        df,
        back_candles=params['Back Candles'],
        candle_size_pips=params['Candle Size Pips'],
        tp_pips=params['TP Pips'],
        sl_pips=params['SL Pips'],
        future_candles=params['Future Candles'],
        follow_direction=params['Follow Direction'],
        spread=params['Spread'],
        save_log=True,
        log_filename='tradelog_GroupedVolatility.csv',
    )
    with open('strategy_params.json', 'w') as f:
        json.dump(params, f)
