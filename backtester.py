import pandas as pd


def load_market_data(filename: str = 'data.csv') -> pd.DataFrame:
    """Read price data and return a sorted DataFrame."""
    df = pd.read_csv(filename, parse_dates=['Time'])
    df.sort_values('Time', inplace=True)
    return df


def example_strategy(
    df: pd.DataFrame,
    sl_pips: int = 20,
    tp_pips: int = 20,
) -> list:
    """Simple placeholder strategy recording trades."""
    trades = []
    for i in range(len(df) - 1):
        open_time = df['Time'].iloc[i]
        open_price = df['Open'].iloc[i]
        close_time = df['Time'].iloc[i + 1]
        close_price = df['Close'].iloc[i + 1]
        direction = 1
        pip_diff = (close_price - open_price) * 10000 * direction
        status = 'partial'
        if pip_diff >= tp_pips:
            status = 'tp'
            pip_diff = tp_pips
            close_price = open_price + tp_pips / 10000 * direction
        elif pip_diff <= -sl_pips:
            status = 'sl'
            pip_diff = -sl_pips
            close_price = open_price - sl_pips / 10000 * direction
        trades.append({
            'Time Open': open_time,
            'Open Price': open_price,
            'Time Close': close_time,
            'Close Price': close_price,
            'Pip Difference': pip_diff,
            'Status': status,
            'SL': open_price - sl_pips / 10000 * direction,
            'TP': open_price + tp_pips / 10000 * direction,
        })
    pd.DataFrame(trades).to_csv('tradelog.csv', index=False)
    return trades


if __name__ == '__main__':
    df = load_market_data()
    example_strategy(df)
