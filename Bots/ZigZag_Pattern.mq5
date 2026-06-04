#property copyright "Expert Trading Advisor"
#property link      ""
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>

//--- Enum for Pattern Selection
enum ENUM_PATTERN_MODE
{
   PATTERN_ENGULFING_ONLY = 0, // Engulfing Only
   PATTERN_CONTINUATION_ONLY = 1, // Continuation Only
   PATTERN_BOTH = 2 // Both Patterns
};

//--- Input parameters (Core Strategy Metrics)
input double InpRiskPercent    = 1.0;  // Risk percentage of equity per trade
input ENUM_PATTERN_MODE InpPatternMode = PATTERN_BOTH; // Pattern Mode Selection
input int    InpTPBufferPoints = 50;   // Take profit buffer in points
input int    InpSLBufferPoints = 50;   // Stop loss buffer in points
input double InpMinRR          = 1.0;  // Minimum Risk:Reward ratio (0 to disable)

//--- ZigZag Indicator Parameters
input int    InpZigZagDepth    = 12;   // ZigZag Depth
input int    InpZigZagDev      = 5;    // ZigZag Deviation
input int    InpZigZagBackstep = 3;    // ZigZag Backstep

//--- Global Variables (Hidden from Input Panel)
uint   Slippage       = 5;          // Slippage in points
ulong  MagicNumber    = 999999;     // Universal Magic Number
int    g_zigzag_handle;             // Handle for the ZigZag indicator

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
//| Exit Logic                                                       |
//+------------------------------------------------------------------+
void CheckForExit()
{
    bool friday_cutoff = IsFridayCutoffReached();

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;

        string symbol = PositionGetString(POSITION_SYMBOL);
        if(symbol != _Symbol) continue;

        long magic = PositionGetInteger(POSITION_MAGIC);
        if(magic != (long)MagicNumber) continue;

        // --- CUSTOM EXIT LOGIC HERE ---
        bool strategy_exit_condition = false;

        // Universal Rule: Close if Friday 16:00 (GMT+2) or strategy dictates
        if(friday_cutoff || strategy_exit_condition)
        {
            trade.PositionClose(ticket);
        }
    }
}

//+------------------------------------------------------------------+
//| Entry Logic                                                      |
//+------------------------------------------------------------------+
void CheckForEntry()
{
    if(IsFridayCutoffReached()) return; // Prevent entries after Friday cutoff

    double prices[];
    datetime times[];

    // We need 5 points: i (unconfirmed), i-1, i-2, i-3, i-4
    if(GetZigZagPoints(5, prices, times) < 5) return;

    // The most recent point [0] is the unconfirmed leg 'i' that is currently repainting.
    // However, the moment a new unconfirmed leg 'i' starts, the point 'i-1' becomes established.
    // We track the time of point [1] (which is i-1) to detect when a new leg is initiated.
    static datetime last_i1_time = 0;
    datetime current_i1_time = times[1];

    // If the time of i-1 hasn't changed, we haven't initiated a new leg. Wait.
    if(current_i1_time == last_i1_time) return;

    // A new unconfirmed leg has initiated!
    last_i1_time = current_i1_time;

    // Let's alias the points for clarity based on user description:
    // times[0] / prices[0] = i (unconfirmed, currently forming)
    // times[1] / prices[1] = i-1 (last confirmed point)
    // times[2] / prices[2] = i-2
    // times[3] / prices[3] = i-3
    // times[4] / prices[4] = i-4

    double i1 = prices[1];
    double i2 = prices[2];
    double i3 = prices[3];
    double i4 = prices[4];

    // Determine the direction of the new unconfirmed leg 'i'
    // If i is below i-1, the leg is going DOWN. If i is above i-1, the leg is going UP.
    // Or simpler: ZigZag alternates. We can check the relationship between i-1 and i-2.
    bool i1_is_high = (i1 > i2); // If i-1 is a High, the current leg 'i' is going DOWN.
    bool i1_is_low  = (i1 < i2); // If i-1 is a Low, the current leg 'i' is going UP.

    bool buy_condition = false;
    bool sell_condition = false;
    double tp_level = 0.0;
    double sl_level = 0.0;

    // --- Pattern Recognition ---

    // ENGULFING (Sell): HL -> LH ending in LL.
    // Leg 'i' going DOWN (i-1 is High).
    // i-4(Low), i-3(High), i-2(Higher Low), i-1(Lower High)
    bool is_engulfing_sell = i1_is_high && (i2 > i4) && (i1 < i3);

    // ENGULFING (Buy): LH -> HL ending in HH.
    // Leg 'i' going UP (i-1 is Low).
    // i-4(High), i-3(Low), i-2(Lower High), i-1(Higher Low)
    bool is_engulfing_buy  = i1_is_low && (i2 < i4) && (i1 > i3);

    // CONTINUATION (Sell): LL -> LH ending in LL.
    // Leg 'i' going DOWN (i-1 is High).
    // i-4(Low), i-3(High), i-2(Lower Low), i-1(Lower High)
    bool is_continuation_sell = i1_is_high && (i2 < i4) && (i1 < i3);

    // CONTINUATION (Buy): HH -> HL ending in HH.
    // Leg 'i' going UP (i-1 is Low).
    // i-4(High), i-3(Low), i-2(Higher High), i-1(Higher Low)
    bool is_continuation_buy  = i1_is_low && (i2 > i4) && (i1 > i3);


    // Evaluate based on Pattern Mode
    if(InpPatternMode == PATTERN_ENGULFING_ONLY || InpPatternMode == PATTERN_BOTH)
    {
        if(is_engulfing_buy)  { buy_condition = true;  tp_level = i2; sl_level = i1; }
        if(is_engulfing_sell) { sell_condition = true; tp_level = i2; sl_level = i1; }
    }

    if(InpPatternMode == PATTERN_CONTINUATION_ONLY || InpPatternMode == PATTERN_BOTH)
    {
        // Don't overwrite if an engulfing condition was already met (though they are mutually exclusive structurally)
        if(is_continuation_buy && !buy_condition)   { buy_condition = true;  tp_level = i2; sl_level = i1; }
        if(is_continuation_sell && !sell_condition) { sell_condition = true; tp_level = i2; sl_level = i1; }
    }

    if(!buy_condition && !sell_condition) return;

    trade.SetDeviationInPoints(Slippage);
    double entry_price = 0.0;
    double final_sl = 0.0;
    double final_tp = 0.0;

    if(buy_condition)
    {
        entry_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        // SL at i-1. Positive buffer moves SL further away (lower)
        final_sl = sl_level - (InpSLBufferPoints * _Point);

        // TP at i-2. Positive buffer moves TP further away (higher)
        final_tp = tp_level + (InpTPBufferPoints * _Point);

        // If buffer was 0, it simply stays at the exact level
    }
    else if(sell_condition)
    {
        entry_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

        // SL at i-1. Positive buffer moves SL further away (higher)
        final_sl = sl_level + (InpSLBufferPoints * _Point);

        // TP at i-2. Positive buffer moves TP further away (lower)
        final_tp = tp_level - (InpTPBufferPoints * _Point);
    }

    // --- Minimum Risk:Reward Rule ---
    double sl_dist_points = MathAbs(entry_price - final_sl) / _Point;
    double tp_dist_points = MathAbs(entry_price - final_tp) / _Point;

    if(InpMinRR > 0.0 && sl_dist_points > 0.0)
    {
        double rr_ratio = tp_dist_points / sl_dist_points;
        if(rr_ratio < InpMinRR) return; // Discard trade if RR is too low
    }

    // Calculate dynamic lot size based on calculated SL distance
    double lots = CalcLotSize(InpRiskPercent, sl_dist_points);
    if(lots <= 0) return;

    if(buy_condition)
    {
        trade.Buy(lots, _Symbol, entry_price, final_sl, final_tp, "Strategy_Buy");
    }
    else if(sell_condition)
    {
        trade.Sell(lots, _Symbol, entry_price, final_sl, final_tp, "Strategy_Sell");
    }
}

//+------------------------------------------------------------------+
//| Helper: Get the last N ZigZag swing points                       |
//+------------------------------------------------------------------+
// Returns the number of points found. Fills the prices and times arrays
// with the most recent swing point at index 0 (which is i-1),
// i-2 at index 1, i-3 at index 2, etc.
int GetZigZagPoints(int count, double &prices[], datetime &times[])
{
    double zz_buffer[];
    datetime time_buffer[];

    // We need to fetch enough bars to find 'count' swing points.
    // 1000 bars is usually enough, but could be adjusted if needed.
    int bars_to_copy = 1000;

    ArraySetAsSeries(zz_buffer, true);
    ArraySetAsSeries(time_buffer, true);

    if(CopyBuffer(g_zigzag_handle, 0, 0, bars_to_copy, zz_buffer) <= 0) return 0;
    if(CopyTime(_Symbol, _Period, 0, bars_to_copy, time_buffer) <= 0) return 0;

    ArrayResize(prices, count);
    ArrayResize(times, count);

    int points_found = 0;

    // Start from index 1 to ignore the unconfirmed current leg (which would be at index 0 if it existed and matched)
    // Actually, ZigZag repaints the most recent point. The point at the 0th non-empty value
    // might be the currently forming (unconfirmed) leg.
    // We will collect the points and let the calling function decide what to do with them.
    for(int i = 0; i < bars_to_copy && points_found < count; i++)
    {
        if(zz_buffer[i] > 0.0) // Valid swing point
        {
            prices[points_found] = zz_buffer[i];
            times[points_found] = time_buffer[i];
            points_found++;
        }
    }

    return points_found;
}

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber((long)MagicNumber);

    // Initialize ZigZag Indicator
    g_zigzag_handle = iCustom(_Symbol, _Period, "Examples\\ZigZag", InpZigZagDepth, InpZigZagDev, InpZigZagBackstep);
    if(g_zigzag_handle == INVALID_HANDLE)
    {
        Print("Failed to create ZigZag indicator handle.");
        return(INIT_FAILED);
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    g_test_end = TimeCurrent();
    if(g_zigzag_handle != INVALID_HANDLE) IndicatorRelease(g_zigzag_handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Evaluate on every tick to catch the exact moment a leg initiates
    UpdateTestStats();
    CheckForExit();      // Manage open positions
    CheckForEntry();     // Scan for new setups (multiple trades allowed inherently)
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