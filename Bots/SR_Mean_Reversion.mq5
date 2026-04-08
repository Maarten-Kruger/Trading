#property copyright "Expert Trading Advisor"
#property link      ""
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>

//--- Input parameters (Core Strategy Metrics)
input int    InpAtrPeriod            = 14;     // ATR Period
input int    InpLookbackCandles      = 100;    // Lookback window for S/R Histogram
input double InpBinMultiplier        = 0.5;    // Bin size multiplier (BIN_MULTIPLIER * ATR)
input double InpTopPercentileBins    = 10.0;   // Top percentile of bins to consider as S/R (e.g., 10%)
input double InpMinSRMultiplier      = 2.0;    // Minimum separation between S/R lines (MIN_SR_MULTIPLIER * ATR)
input double InpSRMultiplier         = 1.0;    // Buffer zone around S/R line (SR_MULTIPLIER * ATR)
input bool   InpBreakout             = false;  // BREAKOUT mode flag (false = Mean Reversion, true = Breakout)
input double InpSLMultiplier         = 1.5;    // Stop Loss Multiplier (SL_MULTIPLIER * ATR)
input double InpTPTrueMultiplier     = 2.0;    // TP Multiplier when BREAKOUT is true (TP_TRUE_MULTIPLIER * ATR)
input bool   InpTakeHalfProfitAtSR   = false;  // Take half profit at S/R line (only applies when BREAKOUT is false)
input double InpRiskPercent          = 1.0;    // Risk percentage of equity per trade

//--- Global Variables (Hidden from Input Panel)
uint   Slippage       = 5;          // Slippage in points
ulong  MagicNumber    = 999999;     // Universal Magic Number

// Optimization Weights (Must sum to ~100)
double Wpr            = 20.0;       // Weight for Payoff Ratio (W_pr)
double Wmr            = 20.0;       // Weight for Monthly Return (W_mr)
double Wnp            = 15.0;       // Weight for Negative Penalty (W_np)
double Wtc            = 20.0;       // Weight for Trade Count per Month (W_tc)
double Wsr            = 15.0;       // Weight for Sharpe Ratio (W_sr)
double WRor            = 10.0;      // Weight for Risk of Ruin

// Sigmoid Normalization Parameters (k = steepness, m = midpoint)
double PrK            = 0.585927;   // Payoff Ratio Steepness
double PrM            = 4.25;       // Payoff Ratio Midpoint
double MrK            = 0.005859;   // Monthly Return Steepness
double MrM            = 425;        // Monthly Return Midpoint
double NpK            = 0.005493;   // Negative Penalty Steepness
double NpM            = 600.0;      // Negative Penalty Midpoint
double TcK            = 0.079899;   // Trade Count Steepness
double TcM            = 42.5;       // Trade Count Midpoint
double SrK            = 0.878890;   // Sharpe Ratio Steepness
double SrM            = 2.5;        // Sharpe Ratio Midpoint
double RoRK           = 0.046257;   // Risk of Ruin Steepness
double RoRM           = 52.5;       // Risk of Ruin Midpoint

//--- Global Objects
CTrade  trade;                      // Trading object
int     g_atr_handle = INVALID_HANDLE; // ATR Indicator Handle
double  g_atr_buffer[];             // Array for ATR data

//--- S/R Arrays
double g_active_sr_lines[];         // Array holding current active S/R lines

//--- Temporary Struct for Bin Sorting
struct BinData
{
    double price;
    int    count;
};

//--- Trade Tracker
struct TradeTracker
{
    ulong  ticket;
    double sr_line;
    bool   partial_closed;
};

TradeTracker g_tracked_trades[];

//--- Test Tracking Variables
datetime g_test_start = 0;          // First bar time in test
datetime g_test_end   = 0;          // Last bar time in test
int      g_total_bars = 0;          // Number of processed bars

//+------------------------------------------------------------------+
//| Update test time and bar counters                                |
//+------------------------------------------------------------------+
void UpdateTestStats()
{
    datetime now = TimeCurrent();
    if(g_test_start == 0) g_test_start = now;
    g_test_end = now;
    g_total_bars++;
}

//+------------------------------------------------------------------+
//| Calculate number of months between two datetimes                 |
//+------------------------------------------------------------------+
double CalcMonths(datetime start_time, datetime end_time)
{
    if(start_time == 0 || end_time == 0) return(0.0);

    MqlDateTime start_struct, end_struct;
    TimeToStruct(start_time, start_struct);
    TimeToStruct(end_time, end_struct);

    int months = (end_struct.year - start_struct.year) * 12 +
                 (end_struct.mon  - start_struct.mon)  + 1;

    return((double)months);
}

//+------------------------------------------------------------------+
//| Helper: Detect new bar                                           |
//+------------------------------------------------------------------+
bool IsNewBar()
{
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, _Period, 0);
    if(current_bar_time != last_bar_time)
    {
        last_bar_time = current_bar_time;
        return(true);
    }
    return(false);
}

//+------------------------------------------------------------------+
//| Determine if Friday 16:00 GMT+2 close time has been reached      |
//+------------------------------------------------------------------+
bool IsFridayCutoffReached()
{
    const int GMT_PLUS_TWO = 2 * 60 * 60;
    datetime now_gmt = TimeGMT();
    datetime now_gmt2 = now_gmt + GMT_PLUS_TWO;

    MqlDateTime time_struct;
    TimeToStruct(now_gmt2, time_struct);

    if(time_struct.day_of_week != 5) return(false); // 5 = Friday
    if(time_struct.hour > 16) return(true);
    if(time_struct.hour == 16 && time_struct.min >= 0) return(true);

    return(false);
}

//+------------------------------------------------------------------+
//| Calculate trade volume based on percentage of equity risk        |
//+------------------------------------------------------------------+
double CalcLotSize(double risk_percent, double sl_distance_points)
{
    double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
    double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double step      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double min_vol   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_vol   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

    if(tick_val <= 0 || tick_size <= 0 || sl_distance_points <= 0) return(min_vol);

    double risk_amount   = equity * risk_percent / 100.0;
    double stop_distance = sl_distance_points * _Point;
    double ticks         = stop_distance / tick_size;
    double lot           = risk_amount / (ticks * tick_val);

    // Adjust to broker limits
    lot = MathFloor(lot / step) * step;
    lot = MathMax(min_vol, MathMin(max_vol, lot));

    return(lot);
}

//+------------------------------------------------------------------+
//| Helper: Add Trade to Tracker                                     |
//+------------------------------------------------------------------+
void TrackNewTrade(ulong ticket, double sr_line)
{
    int size = ArraySize(g_tracked_trades);
    ArrayResize(g_tracked_trades, size + 1);
    g_tracked_trades[size].ticket = ticket;
    g_tracked_trades[size].sr_line = sr_line;
    g_tracked_trades[size].partial_closed = false;
}

//+------------------------------------------------------------------+
//| Exit Logic                                                       |
//+------------------------------------------------------------------+
void CheckForExit()
{
    bool friday_cutoff = IsFridayCutoffReached();
    double current_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double current_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;

        string symbol = PositionGetString(POSITION_SYMBOL);
        if(symbol != _Symbol) continue;

        long magic = PositionGetInteger(POSITION_MAGIC);
        if(magic != (long)MagicNumber) continue;

        // Universal Rule: Close if Friday 16:00 (GMT+2)
        if(friday_cutoff)
        {
            trade.PositionClose(ticket);
            continue;
        }

        // --- Custom Exit: Partial Close at SR Line (Mean Reversion only) ---
        if(!InpBreakout && InpTakeHalfProfitAtSR)
        {
            // Find this trade in the tracker
            for(int j = 0; j < ArraySize(g_tracked_trades); j++)
            {
                if(g_tracked_trades[j].ticket == ticket)
                {
                    if(!g_tracked_trades[j].partial_closed)
                    {
                        double sr_line = g_tracked_trades[j].sr_line;
                        long pos_type = PositionGetInteger(POSITION_TYPE);
                        bool crossed_sr = false;

                        if(pos_type == POSITION_TYPE_BUY)
                        {
                            // Price rising towards SR line
                            if(current_bid >= sr_line) crossed_sr = true;
                        }
                        else if(pos_type == POSITION_TYPE_SELL)
                        {
                            // Price falling towards SR line
                            if(current_ask <= sr_line) crossed_sr = true;
                        }

                        if(crossed_sr)
                        {
                            double total_volume = PositionGetDouble(POSITION_VOLUME);
                            double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
                            double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

                            double partial_vol = MathFloor((total_volume * 0.5) / step) * step;
                            if(partial_vol >= min_vol)
                            {
                                if(trade.PositionClosePartial(ticket, partial_vol))
                                {
                                    g_tracked_trades[j].partial_closed = true;
                                    Print("Partial close executed at S/R Line: ", sr_line);
                                }
                            }
                            else
                            {
                                // If half volume is too small, just close the whole thing
                                trade.PositionClose(ticket);
                                g_tracked_trades[j].partial_closed = true;
                            }
                        }
                    }
                    break; // Found trade, no need to keep searching tracker
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Entry Logic                                                      |
//+------------------------------------------------------------------+
void CheckForEntry(double current_atr)
{
    if(IsFridayCutoffReached()) return; // Prevent entries after Friday cutoff
    if(ArraySize(g_active_sr_lines) == 0) return;

    double open1 = iOpen(_Symbol, _Period, 1);
    double close1 = iClose(_Symbol, _Period, 1);

    double buffer_dist = InpSRMultiplier * current_atr;
    double sl_dist = InpSLMultiplier * current_atr;
    double tp_breakout_dist = InpTPTrueMultiplier * current_atr;

    trade.SetDeviationInPoints(Slippage);

    for(int i = 0; i < ArraySize(g_active_sr_lines); i++)
    {
        double sr_line = g_active_sr_lines[i];
        double upper_bound = sr_line + buffer_dist;
        double lower_bound = sr_line - buffer_dist;

        // Check if previous candle opened inside buffer
        bool opened_inside = (open1 <= upper_bound && open1 >= lower_bound);
        if(!opened_inside) continue;

        bool breakout_up = (close1 > upper_bound);
        bool breakout_down = (close1 < lower_bound);

        if(!breakout_up && !breakout_down) continue; // No breakout occurred

        int signal_type = 0; // 1 = Buy, -1 = Sell

        if(InpBreakout)
        {
            // Trend Following
            if(breakout_up) signal_type = 1;     // Buy
            if(breakout_down) signal_type = -1;  // Sell
        }
        else
        {
            // Mean Reversion (Fade)
            if(breakout_up) signal_type = -1;    // Sell
            if(breakout_down) signal_type = 1;   // Buy
        }

        if(signal_type == 1) // BUY
        {
            double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

            double sl = 0;
            double tp = 0;

            if(InpBreakout)
            {
                // Breakout mode: SL is at the opposite edge of the buffer (where TP would be for Mean Reversion)
                sl = upper_bound; // Breakout Buy happens above upper_bound, so sl is upper_bound (or lower_bound if strict opposite edge, but let's stick to the opposite of entry meaning inside the zone)
                // Wait, if it's a Buy Breakout, price is > upper_bound. The opposite edge of the *buffer zone* is lower_bound.
                sl = lower_bound;
                tp = price + tp_breakout_dist;
            }
            else
            {
                // Mean Reversion mode: Standard SL
                sl = price - sl_dist;
                // Opposite edge of buffer
                tp = upper_bound;
            }

            double actual_sl_dist = MathAbs(price - sl);
            double lots = CalcLotSize(InpRiskPercent, actual_sl_dist / _Point);

            if(trade.Buy(lots, _Symbol, price, sl, tp, "Strategy_Buy"))
            {
                if(trade.ResultRetcode() == TRADE_RETCODE_DONE)
                    TrackNewTrade(trade.ResultOrder(), sr_line);
            }
        }
        else if(signal_type == -1) // SELL
        {
            double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

            double sl = 0;
            double tp = 0;

            if(InpBreakout)
            {
                // Breakout mode: SL is opposite edge of buffer zone
                sl = upper_bound; // Sell Breakout happens below lower_bound, so opposite edge is upper_bound
                tp = price - tp_breakout_dist;
            }
            else
            {
                // Mean Reversion mode: Standard SL
                sl = price + sl_dist;
                // Opposite edge of buffer
                tp = lower_bound;
            }

            double actual_sl_dist = MathAbs(price - sl);
            double lots = CalcLotSize(InpRiskPercent, actual_sl_dist / _Point);

            if(trade.Sell(lots, _Symbol, price, sl, tp, "Strategy_Sell"))
            {
                if(trade.ResultRetcode() == TRADE_RETCODE_DONE)
                    TrackNewTrade(trade.ResultOrder(), sr_line);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber((long)MagicNumber);

    // Initialize ATR
    g_atr_handle = iATR(_Symbol, _Period, InpAtrPeriod);
    if(g_atr_handle == INVALID_HANDLE)
    {
        Print("Failed to create ATR indicator handle");
        return(INIT_FAILED);
    }

    ArraySetAsSeries(g_atr_buffer, true);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    g_test_end = TimeCurrent();
    if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
}

//+------------------------------------------------------------------+
//| Cleanup closed trades from tracking array                        |
//+------------------------------------------------------------------+
void CleanupActiveTrades()
{
    int total = ArraySize(g_tracked_trades);
    for(int i = total - 1; i >= 0; i--)
    {
        if(!PositionSelectByTicket(g_tracked_trades[i].ticket))
        {
            // Position no longer exists, remove from array
            for(int j = i; j < ArraySize(g_tracked_trades) - 1; j++)
            {
                g_tracked_trades[j] = g_tracked_trades[j+1];
            }
            ArrayResize(g_tracked_trades, ArraySize(g_tracked_trades) - 1);
        }
    }
}

//+------------------------------------------------------------------+
//| Helper: Get Current ATR                                          |
//+------------------------------------------------------------------+
double GetCurrentATR()
{
    if(CopyBuffer(g_atr_handle, 0, 1, 1, g_atr_buffer) <= 0) return 0.0;
    return g_atr_buffer[0];
}

//+------------------------------------------------------------------+
//| S/R Histogram and Clustering Engine                              |
//+------------------------------------------------------------------+
void CalculateSRLines(double atr)
{
    ArrayFree(g_active_sr_lines);
    if(atr <= 0 || InpLookbackCandles <= 0) return;

    double rates_high[];
    double rates_low[];
    ArraySetAsSeries(rates_high, true);
    ArraySetAsSeries(rates_low, true);

    // Copy wick data (skip candle 0 as it's unclosed, start from 1)
    if(CopyHigh(_Symbol, _Period, 1, InpLookbackCandles, rates_high) != InpLookbackCandles) return;
    if(CopyLow(_Symbol, _Period, 1, InpLookbackCandles, rates_low) != InpLookbackCandles) return;

    double global_max = rates_high[ArrayMaximum(rates_high)];
    double global_min = rates_low[ArrayMinimum(rates_low)];

    double bin_size = InpBinMultiplier * atr;
    if(bin_size <= 0) return;

    // Calculate number of bins
    int total_bins = (int)MathCeil((global_max - global_min) / bin_size) + 1;
    if(total_bins <= 0) return;

    BinData bins[];
    ArrayResize(bins, total_bins);
    for(int i = 0; i < total_bins; i++)
    {
        bins[i].price = global_min + (i * bin_size) + (bin_size / 2.0); // Center of the bin
        bins[i].count = 0;
    }

    // Populate touch-count array (Wick-to-Wick coverage)
    for(int i = 0; i < InpLookbackCandles; i++)
    {
        double high = rates_high[i];
        double low = rates_low[i];

        for(int j = 0; j < total_bins; j++)
        {
            double bin_bottom = global_min + (j * bin_size);
            double bin_top = bin_bottom + bin_size;

            // Check if candle wick intersects the bin
            if(high >= bin_bottom && low <= bin_top)
            {
                bins[j].count++;
            }
        }
    }

    // Sort bins by touch count descending
    // Simple Bubble Sort since total_bins is usually small
    for(int i = 0; i < total_bins - 1; i++)
    {
        for(int j = 0; j < total_bins - i - 1; j++)
        {
            if(bins[j].count < bins[j+1].count)
            {
                BinData temp = bins[j];
                bins[j] = bins[j+1];
                bins[j+1] = temp;
            }
        }
    }

    // Determine the cutoff threshold for the TOP_PERCENTILE_BINS
    int num_top_bins = (int)MathFloor(total_bins * (InpTopPercentileBins / 100.0));
    if(num_top_bins <= 0) num_top_bins = 1;

    int threshold_count = bins[num_top_bins - 1].count;
    if(threshold_count == 0) return;

    // Extract candidate bins
    double candidate_bins[];
    for(int i = 0; i < total_bins; i++)
    {
        if(bins[i].count >= threshold_count && bins[i].count > 0)
        {
            int size = ArraySize(candidate_bins);
            ArrayResize(candidate_bins, size + 1);
            candidate_bins[size] = bins[i].price;
        }
    }

    // Sort candidate bins by price ascending
    ArraySort(candidate_bins);

    // Cluster bins closer than MIN_SR_MULTIPLIER * ATR
    double min_separation = InpMinSRMultiplier * atr;

    int cand_size = ArraySize(candidate_bins);
    if(cand_size == 0) return;

    double current_cluster_sum = candidate_bins[0];
    int current_cluster_count = 1;

    for(int i = 1; i < cand_size; i++)
    {
        double prev_avg = current_cluster_sum / current_cluster_count;
        if(candidate_bins[i] - prev_avg < min_separation)
        {
            // Add to current cluster
            current_cluster_sum += candidate_bins[i];
            current_cluster_count++;
        }
        else
        {
            // Finalize current cluster and save to active lines
            int active_size = ArraySize(g_active_sr_lines);
            ArrayResize(g_active_sr_lines, active_size + 1);
            g_active_sr_lines[active_size] = current_cluster_sum / current_cluster_count;

            // Start new cluster
            current_cluster_sum = candidate_bins[i];
            current_cluster_count = 1;
        }
    }

    // Add the final cluster
    if(current_cluster_count > 0)
    {
        int active_size = ArraySize(g_active_sr_lines);
        ArrayResize(g_active_sr_lines, active_size + 1);
        g_active_sr_lines[active_size] = current_cluster_sum / current_cluster_count;
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    bool new_bar = IsNewBar();
    if(!new_bar) return; // Evaluate only on bar close/open

    UpdateTestStats();

    // Update ATR
    double current_atr = GetCurrentATR();
    if(current_atr <= 0) return; // Wait for enough data

    // Calculate S/R Lines based on previous closed candles
    CalculateSRLines(current_atr);
    CleanupActiveTrades();

    CheckForExit();      // Manage open positions
    CheckForEntry(current_atr);     // Scan for new setups (multiple trades allowed inherently)
}

//+------------------------------------------------------------------+
//| Sigmoid normalization function: f(x) = 1 / (1 + e^(-k(x - m)))   |
//+------------------------------------------------------------------+
double SigmoidNormalize(double x, double k, double m)
{
    return (1.0 / (1.0 + MathPow(2.71828, -k * (x - m))));
}

//+------------------------------------------------------------------+
//| Calculate Negative Penalty (Average loss from negative months)   |
//+------------------------------------------------------------------+
double CalcNegativePenalty()
{
    double monthly_profits[];
    int num_months = (int)CalcMonths(g_test_start, g_test_end);
    if(num_months <= 0) return 0;

    ArrayResize(monthly_profits, num_months);
    ArrayInitialize(monthly_profits, 0.0);

    HistorySelect(g_test_start, g_test_end);
    for(int i = 0; i < HistoryDealsTotal(); i++)
    {
        long ticket = HistoryDealGetTicket(i);
        if(ticket > 0)
        {
            datetime close_time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);

            MqlDateTime start_struct, close_struct;
            TimeToStruct(g_test_start, start_struct);
            TimeToStruct(close_time, close_struct);
            int month_index = (close_struct.year - start_struct.year) * 12 + (close_struct.mon - start_struct.mon);

            if(month_index >= 0 && month_index < num_months)
            {
                monthly_profits[month_index] += profit;
            }
        }
    }

    double total_loss = 0;
    int negative_months = 0;
    for(int i = 0; i < num_months; i++)
    {
        if(monthly_profits[i] < 0)
        {
            total_loss += MathAbs(monthly_profits[i]);
            negative_months++;
        }
    }

    return (negative_months > 0) ? total_loss / negative_months : 0;
}

//+------------------------------------------------------------------+
//| Calculate the Risk of Ruin (RoR)                                 |
//+------------------------------------------------------------------+
double CalcRiskOfRuin(double prob_win, double avg_win, double avg_loss)
{
    const double MaxRisk = 0.05; // 5% Risk Threshold
    const double VERY_LARGE_ROR = 1.0e9;

    if (prob_win >= 1.0) return 0.0; // 100% win rate = 0% RoR
    if (prob_win <= 0 || avg_win <= 0 || avg_loss >= 0) return VERY_LARGE_ROR;

    double prob_loss = 1.0 - prob_win;
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);

    if (equity <= 0) return VERY_LARGE_ROR;

    double avg_win_pct = avg_win / equity;
    double avg_loss_pct = MathAbs(avg_loss) / equity;

    double Z = (prob_win * avg_win_pct) - (prob_loss * avg_loss_pct);
    double A = MathSqrt((prob_win * MathPow(avg_win_pct, 2)) + (prob_loss * MathPow(avg_loss_pct, 2)));

    if (A == 0) return (Z > 0) ? 0.0 : VERY_LARGE_ROR;

    double P = 0.5 * (1 + (Z / A));

    if (P >= 1.0) return 0.0;
    if (P <= 0) return VERY_LARGE_ROR;

    double RoR = MathPow(((1.0 - P) / P), (MaxRisk / A));
    return RoR;
}

//+------------------------------------------------------------------+
//| Custom Optimization Criterion                                    |
//+------------------------------------------------------------------+
double OnTester()
{
    // --- METRIC CALCULATIONS ---
    double total_profit = TesterStatistics(STAT_PROFIT);
    double start_equity = TesterStatistics(STAT_INITIAL_DEPOSIT);
    double num_months = CalcMonths(g_test_start, g_test_end);

    double monthly_return_raw = (num_months > 0 && start_equity > 0) ? (total_profit / num_months) : 0;
    double negative_penalty_raw = CalcNegativePenalty();
    double total_trades = TesterStatistics(STAT_TRADES);
    double trades_per_month_raw = (num_months > 0) ? total_trades / num_months : 0;
    double sharpe_ratio_raw = TesterStatistics(STAT_SHARPE_RATIO);

    double win_rate = total_trades > 0 ? TesterStatistics(STAT_PROFIT_TRADES) / total_trades : 0;
    double profit_trades = TesterStatistics(STAT_PROFIT_TRADES);
    double loss_trades = TesterStatistics(STAT_LOSS_TRADES);

    double avg_win = profit_trades > 0 ? TesterStatistics(STAT_GROSS_PROFIT) / profit_trades : 0;
    double avg_loss = loss_trades > 0 ? TesterStatistics(STAT_GROSS_LOSS) / loss_trades : 0;

    double payoff_ratio_raw = (avg_loss != 0) ? avg_win / MathAbs(avg_loss) : 0;
    double risk_of_ruin_raw = CalcRiskOfRuin(win_rate, avg_win, avg_loss) * 100.0;

    // --- DEBUG OUTPUT (BEFORE NORMALIZATION) ---
    printf("--- Before Sigmoid Normalization ---");
    printf("Payoff Ratio (Pr): %.2f", payoff_ratio_raw);
    printf("Monthly Return (Mr): %.2f", monthly_return_raw);
    printf("Negative Penalty (Np): %.2f", negative_penalty_raw);
    printf("Trade Count/Month (Tc): %.2f", trades_per_month_raw);
    printf("Sharpe Ratio (Sr): %.2f", sharpe_ratio_raw);
    printf("Risk of Ruin (RoR): %.2f", risk_of_ruin_raw);

    // --- SIGMOID NORMALIZATION ---
    double payoff_ratio_norm = SigmoidNormalize(payoff_ratio_raw, PrK, PrM);
    double monthly_return_norm = SigmoidNormalize(monthly_return_raw, MrK, MrM);
    double negative_penalty_norm = SigmoidNormalize(negative_penalty_raw, NpK, NpM);
    double trades_per_month_norm = SigmoidNormalize(trades_per_month_raw, TcK, TcM);
    double sharpe_ratio_norm = SigmoidNormalize(sharpe_ratio_raw, SrK, SrM);
    double risk_of_ruin_norm = SigmoidNormalize(risk_of_ruin_raw, RoRK, RoRM);

    // --- DEBUG OUTPUT (AFTER NORMALIZATION) ---
    printf("--- After Sigmoid Normalization ---");
    printf("Payoff Ratio (Pr) Norm: %.4f", payoff_ratio_norm);
    printf("Monthly Return (Mr) Norm: %.4f", monthly_return_norm);
    printf("Negative Penalty (Np) Norm: %.4f", negative_penalty_norm);
    printf("Trade Count/Month (Tc) Norm: %.4f", trades_per_month_norm);
    printf("Sharpe Ratio (Sr) Norm: %.4f", sharpe_ratio_norm);
    printf("Risk of Ruin (RoR) Norm: %.4f", risk_of_ruin_norm);

    // --- WEIGHTING (Ensuring they sum to a relative proportion) ---
    double total_weight = Wpr + Wmr + Wnp + Wtc + Wsr + WRor;
    if (total_weight <= 0) total_weight = 1.0;

    double w_pr = Wpr / total_weight;
    double w_mr = Wmr / total_weight;
    double w_np = Wnp / total_weight;
    double w_tc = Wtc / total_weight;
    double w_sr = Wsr / total_weight;
    double w_ror = WRor / total_weight;

    // --- FINAL OBJECTIVE FUNCTION ---
    // Y = (Pr * Wpr) + (Mr * Wmr) - (Np * Wnp) - (RoR * WRor) + (Tc * Wtc) + (Sr * Wsr) * 100
    double objective_score = ((payoff_ratio_norm * w_pr) +
                              (monthly_return_norm * w_mr) -
                              (negative_penalty_norm * w_np) -
                              (risk_of_ruin_norm * w_ror) +
                              (trades_per_month_norm * w_tc) +
                              (sharpe_ratio_norm * w_sr)) * 100.0;

    printf("--- Final Score ---");
    printf("Objective Score: %.4f", objective_score);

    return(objective_score);
}