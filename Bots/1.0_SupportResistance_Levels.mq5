#property copyright "Expert Trading Advisor"
#property link      ""
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>

//--- Input parameters (Core Strategy Metrics)
input int    InpLookback       = 1000; // LOOKBACK: Number of candles to search for peaks/troughs
input double InpTPPoints       = 400;  // TP: Take profit distance in points
input double InpSLPoints       = 200;  // SL: Stop loss distance in points
input double InpBufferPoints   = 50;   // BUFFER: Zone around the line in points
input int    InpExtremity      = 10;   // EXTREMITY: Candles to left/right for peak/trough validation
input double InpThresholdPts   = 100;  // THRESHOLD: Max points between levels to cluster them
input int    InpMinGroup       = 3;    // MIN_GROUP: Minimum peaks/troughs to form a valid line
input bool   InpFallingKnife   = true; // FALLINGKNIFE: True to wait for candle close rejection
input double InpRiskPercent    = 1.0;  // Risk percentage of equity per trade

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

//--- Test Tracking Variables
datetime g_test_start = 0;          // First bar time in test
datetime g_test_end   = 0;          // Last bar time in test
int      g_total_bars = 0;          // Number of processed bars

//--- Strategy Variables
double   g_significance_lines[];    // Array storing calculated support/resistance lines
int      g_num_lines = 0;           // Number of valid lines
bool     g_in_buy_zone = false;     // True if price entered support zone
bool     g_in_sell_zone = false;    // True if price entered resistance zone
double   g_active_buy_line = 0;     // The support line currently being tested
double   g_active_sell_line = 0;    // The resistance line currently being tested

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
double CalcLotSize(double risk_percent)
{
    double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
    double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double step      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double min_vol   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_vol   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

    if(tick_val <= 0 || tick_size <= 0 || InpSLPoints <= 0) return(min_vol);
    
    double risk_amount   = equity * risk_percent / 100.0;
    double stop_distance = InpSLPoints * _Point;
    double ticks         = stop_distance / tick_size;
    double lot           = risk_amount / (ticks * tick_val);
    
    // Adjust to broker limits
    lot = MathFloor(lot / step) * step;
    lot = MathMax(min_vol, MathMin(max_vol, lot));

    return(lot);
}

//+------------------------------------------------------------------+
//| Calculate Significance Lines (Support & Resistance)              |
//+------------------------------------------------------------------+
void CalculateSignificanceLines()
{
    g_num_lines = 0;
    ArrayResize(g_significance_lines, 0);

    // Arrays to hold all detected peaks and troughs
    double extremums[];
    int ext_count = 0;

    int total_bars = iBars(_Symbol, _Period);
    if(total_bars < InpLookback) return;

    // Loop through LOOKBACK period, leaving EXTREMITY margins on both sides
    for(int i = InpExtremity; i < InpLookback - InpExtremity; i++)
    {
        double current_high = iHigh(_Symbol, _Period, i);
        double current_low = iLow(_Symbol, _Period, i);

        bool is_peak = true;
        bool is_trough = true;

        // Check EXTREMITY bars to the left (older) and right (newer)
        for(int j = 1; j <= InpExtremity; j++)
        {
            if(current_high <= iHigh(_Symbol, _Period, i + j) || current_high <= iHigh(_Symbol, _Period, i - j))
                is_peak = false;
            
            if(current_low >= iLow(_Symbol, _Period, i + j) || current_low >= iLow(_Symbol, _Period, i - j))
                is_trough = false;
        }

        if(is_peak)
        {
            ext_count++;
            ArrayResize(extremums, ext_count);
            extremums[ext_count - 1] = current_high;
        }
        
        if(is_trough)
        {
            ext_count++;
            ArrayResize(extremums, ext_count);
            extremums[ext_count - 1] = current_low;
        }
    }

    if(ext_count == 0) return;

    // Sort ascending for clustering
    ArraySort(extremums);

    // Simple Distance Threshold Clustering
    double cluster_sum = extremums[0];
    int cluster_count = 1;
    double threshold_price = InpThresholdPts * _Point;

    for(int i = 1; i < ext_count; i++)
    {
        if((extremums[i] - extremums[i-1]) <= threshold_price)
        {
            // Point belongs to the active cluster
            cluster_sum += extremums[i];
            cluster_count++;
        }
        else
        {
            // Gap exceeds threshold, close active cluster
            if(cluster_count >= InpMinGroup)
            {
                g_num_lines++;
                ArrayResize(g_significance_lines, g_num_lines);
                g_significance_lines[g_num_lines - 1] = cluster_sum / cluster_count;
            }
            
            // Start a new cluster
            cluster_sum = extremums[i];
            cluster_count = 1;
        }
    }

    // Check the last cluster
    if(cluster_count >= InpMinGroup)
    {
        g_num_lines++;
        ArrayResize(g_significance_lines, g_num_lines);
        g_significance_lines[g_num_lines - 1] = cluster_sum / cluster_count;
    }
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
void CheckForEntry(bool new_bar)
{
    if(IsFridayCutoffReached()) return; // Prevent entries after Friday cutoff
    if(PositionsTotal() > 0) return;    // Maximum 1 trade at a time
    if(g_num_lines == 0) return;        // No lines calculated yet

    double lots = CalcLotSize(InpRiskPercent);
    trade.SetDeviationInPoints(Slippage);

    // Get current prices
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double buffer_price = InpBufferPoints * _Point;

    bool buy_condition = false;
    bool sell_condition = false;

    // Evaluate if price is currently in any zone
    bool currently_in_buy_zone = false;
    bool currently_in_sell_zone = false;

    for(int i = 0; i < g_num_lines; i++)
    {
        double line = g_significance_lines[i];

        // Buy Zone: Bid price dropped into buffer sitting right on top of a support level
        if(bid <= line + buffer_price && bid >= line)
        {
            currently_in_buy_zone = true;
            g_active_buy_line = line;
        }

        // Sell Zone: Ask price pushed up into buffer just below a resistance level
        if(ask >= line - buffer_price && ask <= line)
        {
            currently_in_sell_zone = true;
            g_active_sell_line = line;
        }
    }

    if(InpFallingKnife)
    {
        // 1. Update State if entering zones
        if(currently_in_buy_zone) g_in_buy_zone = true;
        if(currently_in_sell_zone) g_in_sell_zone = true;

        // 2. Check for triggers on new bar
        if(new_bar)
        {
            double prev_close = iClose(_Symbol, _Period, 1);

            // Buy condition: was in buy zone, now closed above the buffer
            if(g_in_buy_zone && prev_close > (g_active_buy_line + buffer_price))
            {
                buy_condition = true;
            }
            // Sell condition: was in sell zone, now closed below the buffer
            else if(g_in_sell_zone && prev_close < (g_active_sell_line - buffer_price))
            {
                sell_condition = true;
            }

            // Reset states on new bar since we evaluated the close
            g_in_buy_zone = false;
            g_in_sell_zone = false;
        }
    }
    else
    {
        // Execute immediately if in zone
        if(currently_in_buy_zone) buy_condition = true;
        if(currently_in_sell_zone) sell_condition = true;
    }

    if(buy_condition)
    {
        double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double sl = price - InpSLPoints * _Point;
        double tp = price + InpTPPoints * _Point;
        trade.Buy(lots, _Symbol, price, sl, tp, "Strategy_Buy");
        g_in_buy_zone = false;
        g_in_sell_zone = false;
    }
    else if(sell_condition)
    {
        double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double sl = price + InpSLPoints * _Point;
        double tp = price - InpTPPoints * _Point;
        trade.Sell(lots, _Symbol, price, sl, tp, "Strategy_Sell");
        g_in_buy_zone = false;
        g_in_sell_zone = false;
    }
}

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber((long)MagicNumber);
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    g_test_end = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    bool new_bar = IsNewBar();

    if(new_bar)
    {
        UpdateTestStats();
        CalculateSignificanceLines(); // Refresh support/resistance lines every new bar
    }

    CheckForExit();          // Manage open positions
    CheckForEntry(new_bar);  // Scan for new setups on every tick
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