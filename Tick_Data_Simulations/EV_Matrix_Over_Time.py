import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# --- CONFIGURATION ---
FILE_PATH = 'EURUSD_1.6_Tick Data.csv'
OUTPUT_FILE = 'EV_Matrix_Over_Time.csv'

STEP_SECONDS = 60         # Step forward by 60 seconds each time
LOOKBACK_SECONDS = 3600   # 3600 seconds (1 hour) lookback
TICK_DENSITY = 100        # Every 100th tick
PIP_VALUE = 0.00001       # E.g., 0.00001 for EURUSD, 0.01 for USDJPY, 1.0 for BTCUSD
SPREAD_THRESHOLD = 5 * PIP_VALUE # Only trade if spread is below 5 points (scaled by PIP_VALUE)

TRADE_START_HOUR = 8      # Start trading at 8:00
TRADE_END_HOUR = 17       # Stop trading at 17:00 (5 PM)

RR_LEVELS = np.array([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], dtype=np.float64)
R_SIZES = np.array([20, 50, 75, 100], dtype=np.float64)

@njit(cache=True)
def get_hour_from_unix(timestamp_sec):
    # UNIX time starts at 1970-01-01 00:00:00 UTC.
    # Total seconds % 86400 gives seconds in the current day.
    # Divide by 3600 to get the hour.
    return (timestamp_sec % 86400) // 3600

@njit(cache=True)
def calculate_window_ev(bid_slice, ask_slice, times_slice, r_sizes, rr_levels, tick_density, spread_threshold, start_hour, end_hour, pip_value):
    """
    Calculates EV for a specific window of ticks.
    """
    num_r = len(r_sizes)
    num_rr = len(rr_levels)
    ev_matrix = np.full((num_r, num_rr), np.nan)

    # We enter at every TICK_DENSITY-th tick
    entry_indices = np.arange(0, len(bid_slice), tick_density)

    for i_r in range(num_r):
        r_size = r_sizes[i_r]
        for i_rr in range(num_rr):
            rr = rr_levels[i_rr]

            r_value = r_size * pip_value
            tp_value = r_value * rr

            wins = 0
            losses = 0

            for i in entry_indices:
                tick_time = times_slice[i]
                hour = get_hour_from_unix(tick_time)

                # Skip if outside trading hours
                if hour < start_hour or hour >= end_hour:
                    continue

                spread = ask_slice[i] - bid_slice[i]

                # Only take the trade if spread is below the threshold
                if spread < spread_threshold:
                    # UP trade logic
                    entry_up = ask_slice[i]
                    sl_up = entry_up - r_value
                    tp_up = entry_up + tp_value

                    # DOWN trade logic
                    entry_down = bid_slice[i]
                    sl_down = entry_down + r_value
                    tp_down = entry_down - tp_value

                    up_resolved = False
                    down_resolved = False

                    # Scan forward to find the outcome
                    for j in range(i + 1, len(bid_slice)):
                        # Evaluate UP trade (Exit at BID)
                        if not up_resolved:
                            if bid_slice[j] <= sl_up:
                                losses += 1
                                up_resolved = True
                            elif bid_slice[j] >= tp_up:
                                wins += 1
                                up_resolved = True

                        # Evaluate DOWN trade (Exit at ASK)
                        if not down_resolved:
                            if ask_slice[j] >= sl_down:
                                losses += 1
                                down_resolved = True
                            elif ask_slice[j] <= tp_down:
                                wins += 1
                                down_resolved = True

                        # Break early if both directions hit an exit
                        if up_resolved and down_resolved:
                            break

            total_trades = wins + losses

            if total_trades > 0:
                win_rate = wins / total_trades
                loss_rate = losses / total_trades
                ev = (win_rate * rr) - (loss_rate * 1)
            else:
                ev = np.nan

            ev_matrix[i_r, i_rr] = ev

    return ev_matrix

# Global variables for worker processes to avoid pickling overhead
_g_bid = None
_g_ask = None
_g_times = None

def init_worker(bid_arr, ask_arr, times_arr):
    global _g_bid, _g_ask, _g_times
    _g_bid = bid_arr
    _g_ask = ask_arr
    _g_times = times_arr

def process_chunk(args):
    """
    Process a chunk of time steps.
    """
    (timestamps_chunk, start_times, end_times, r_sizes, rr_levels, tick_density) = args
    results = []

    for t_step, start_t, end_t in zip(timestamps_chunk, start_times, end_times):
        # Find indices within the lookback window [start_t, end_t]
        # Since times is sorted, we can use searchsorted
        idx_start = np.searchsorted(_g_times, start_t, side='left')
        idx_end = np.searchsorted(_g_times, end_t, side='right')

        if idx_start < idx_end:
            bid_slice = _g_bid[idx_start:idx_end]
            ask_slice = _g_ask[idx_start:idx_end]
            times_slice = _g_times[idx_start:idx_end]

            ev_matrix = calculate_window_ev(bid_slice, ask_slice, times_slice, r_sizes, rr_levels, tick_density, SPREAD_THRESHOLD, TRADE_START_HOUR, TRADE_END_HOUR, PIP_VALUE)
            results.append((t_step, ev_matrix))
        else:
            # Empty window
            results.append((t_step, np.full((len(r_sizes), len(rr_levels)), np.nan)))

    return results

def main():
    print("Loading data...")
    df = pd.read_csv(FILE_PATH, sep='\t')

    # Forward fill and backward fill missing BID/ASK just like original
    df['<BID>'] = df['<BID>'].ffill().bfill()
    df['<ASK>'] = df['<ASK>'].ffill().bfill()

    # Combine date and time
    print("Parsing timestamps...")
    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S.%f')

    # Convert everything to numpy arrays for numba/multiprocessing
    # Using astype('datetime64[s]') is robust across pandas versions to get seconds directly
    times_sec = df['datetime'].values.astype('datetime64[s]').astype(np.int64)

    bid = df['<BID>'].values.astype(np.float64)
    ask = df['<ASK>'].values.astype(np.float64)

    min_time = times_sec[0]
    max_time = times_sec[-1]

    # We start steps such that the first step has at least some data.
    # We step forward by STEP_SECONDS.
    # At each step `t`, the lookback is `[t - LOOKBACK_SECONDS, t]`.
    # Let's start `t` at min_time + STEP_SECONDS and go up to max_time

    eval_times = np.arange(min_time + STEP_SECONDS, max_time + STEP_SECONDS, STEP_SECONDS)

    start_times = eval_times - LOOKBACK_SECONDS
    end_times = eval_times

    num_cores = cpu_count()
    print(f"Using {num_cores} cores to process {len(eval_times)} steps...")

    # Split the evaluation times into chunks for multiprocessing
    chunk_size = max(1, len(eval_times) // num_cores)

    chunks = []
    for i in range(0, len(eval_times), chunk_size):
        c_eval = eval_times[i:i + chunk_size]
        c_start = start_times[i:i + chunk_size]
        c_end = end_times[i:i + chunk_size]
        chunks.append((c_eval, c_start, c_end, R_SIZES, RR_LEVELS, TICK_DENSITY))

    start_compute_time = time.time()

    all_results = []
    with Pool(processes=num_cores, initializer=init_worker, initargs=(bid, ask, times_sec)) as pool:
        for chunk_results in tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing time chunks"):
            all_results.extend(chunk_results)

    print(f"Computation finished in {time.time() - start_compute_time:.2f} seconds.")

    # Format results into a list of dictionaries for pandas
    print("Formatting output...")
    output_rows = []

    # The timestamps in times_sec might be relative to unix epoch in nanoseconds incorrectly.
    # Actually, df['datetime'].values.astype(np.int64) gives nanoseconds from 1970.
    # Converting to seconds and back should yield the same date, but pd.to_datetime(t_step, unit='s')
    # assumes UTC timestamp.

    for t_step, ev_matrix in all_results:
        row = {'Timestamp': pd.to_datetime(t_step, unit='s')}

        # Flatten the matrix into the columns
        for i_r, r in enumerate(R_SIZES):
            for i_rr, rr in enumerate(RR_LEVELS):
                col_name = f'R{int(r)}_RR{rr}'
                row[col_name] = ev_matrix[i_r, i_rr]

        output_rows.append(row)

    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
