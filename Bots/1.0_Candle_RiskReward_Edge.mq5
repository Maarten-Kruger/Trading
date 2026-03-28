#property copyright "Expert Trading Advisor"
#property description "1.0 Candle RiskReward Edge Strategy"
#property description "Operates by analyzing EMA direction every Period# M1 candles."
#property description "If an active trade exists, it rolls over the trade by modifying SL/TP."

#property link      ""
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>

//--- Input parameters (Core Strategy Metrics)
input int    InpPeriod         = 15;   // Period in M1 candles (how often logic runs)
input double InpRiskReward     = 2.0;  // Risk/Reward Ratio
input int    InpEmaPeriod      = 50;   // EMA Period
input double InpSLPoints       = 200;  // Stop loss distance in points
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
double WRor           = 10.0;       // Weight for Risk of Ruin

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
int     ema_handle;                 // EMA indicator handle
int     candle_counter = 0;         // Counter for M1 candles passed

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
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber((long)MagicNumber);

    // Initialize EMA indicator
    ema_handle = iMA(_Symbol, PERIOD_M1, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
    if(ema_handle == INVALID_HANDLE)
    {
        Print("Failed to create EMA indicator handle");
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
    if(ema_handle != INVALID_HANDLE)
        IndicatorRelease(ema_handle);
}

//+------------------------------------------------------------------+
//| Exit Logic (Friday Close Only)                                   |

//+------------------------------------------------------------------+
void CheckForExit()
{
    bool friday_cutoff = IsFridayCutoffReached();

    if(!friday_cutoff) return; // Only Friday exit is universal here

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;

        string symbol = PositionGetString(POSITION_SYMBOL);
        if(symbol != _Symbol) continue;

        long magic = PositionGetInteger(POSITION_MAGIC);
        if(magic != (long)MagicNumber) continue;

        trade.PositionClose(ticket);
    }
}

//+------------------------------------------------------------------+
//| Process Strategy Logic (Runs every InpPeriod M1 candles)         |
//+------------------------------------------------------------------+
void ProcessStrategy()
{
    if(IsFridayCutoffReached()) return; // Prevent entries/modifications after Friday cutoff

    double current_spread_points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    Print("Current Spread: ", current_spread_points);

    // Calculate Stop Loss and Take Profit in points (ignoring spread for calculations)
    double stop_loss_d = InpSLPoints;
    double take_profit_d = (InpRiskReward * InpSLPoints);

    // Check if there is an active trade for this symbol and magic number
    ulong active_ticket = 0;
    int position_type = -1;

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;

        if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
           PositionGetInteger(POSITION_MAGIC) == (long)MagicNumber)
        {
            active_ticket = ticket;
            position_type = (int)PositionGetInteger(POSITION_TYPE);
            break; // Found the active trade
        }
    }

    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double point = _Point;

    if(active_ticket > 0)
    {
        if(InpRiskReward == 0.0)
        {
            // If RiskReward is 0, we do not trail/modify SL/TP. Instead, we just close the trade when checked.
            trade.PositionClose(active_ticket);
        }
        else
        {
            // Active trade exists: Move SL and TP based on current price
            double new_sl = 0;
            double new_tp = 0;

            if(position_type == POSITION_TYPE_BUY)
            {
                new_sl = bid - (stop_loss_d * point);
                new_tp = bid + (take_profit_d * point);
            }
            else if(position_type == POSITION_TYPE_SELL)
            {
                new_sl = ask + (stop_loss_d * point);
                new_tp = ask - (take_profit_d * point);
            }

            // Modify the position
            trade.PositionModify(active_ticket, new_sl, new_tp);
        }
    }
    else
    {
        // No active trade: Check EMA direction and enter a new trade
        double ema_array[];
        ArraySetAsSeries(ema_array, true);

        if(CopyBuffer(ema_handle, 0, 0, 2, ema_array) != 2)
        {
            Print("Failed to copy EMA buffer");
            return;
        }

        double ema_current = ema_array[0];
        double ema_prev = ema_array[1];

        double lots = CalcLotSize(InpRiskPercent);
        trade.SetDeviationInPoints(Slippage);

        if(ema_current > ema_prev) // EMA pointing UP -> BUY
        {
            double sl = ask - (stop_loss_d * point);
            double tp = (InpRiskReward == 0.0) ? 0.0 : ask + (take_profit_d * point);
            trade.Buy(lots, _Symbol, ask, sl, tp, "Strategy_Buy");
        }
        else if(ema_current < ema_prev) // EMA pointing DOWN -> SELL
        {
            double sl = bid + (stop_loss_d * point);
            double tp = (InpRiskReward == 0.0) ? 0.0 : bid - (take_profit_d * point);
            trade.Sell(lots, _Symbol, bid, sl, tp, "Strategy_Sell");
        }
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    UpdateTestStats();

    // Only check for universal exit dynamically on every tick
    CheckForExit();

    // Core logic relies on M1 bar closures
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, PERIOD_M1, 0);

    if(current_bar_time != last_bar_time)
    {
        last_bar_time = current_bar_time;

        candle_counter++;
        if(candle_counter >= InpPeriod)
        {
            ProcessStrategy();
            candle_counter = 0; // Reset counter after execution
        }
    }
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