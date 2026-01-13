//+------------------------------------------------------------------+
//|                                     AdaptiveMomentumTrend.mq5    |
//|                                     Copyright 2024, Jules        |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Jules"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- Input parameters for Strategy Logic
input int    InpSMAFastPeriod = 50;      // Fast SMA Period (variable_x)
input int    InpSMASlowPeriod = 200;     // Slow SMA Period (variable_y)
input int    InpStochK        = 14;      // Stochastic K Period
input int    InpStochD        = 3;       // Stochastic D Period
input int    InpStochSlowing  = 3;       // Stochastic Slowing
input int    InpStochOverbought = 80;    // Stochastic Overbought Level
input int    InpStochOversold   = 20;    // Stochastic Oversold Level
input int    InpLWMAPeriod    = 10;      // LWMA Period (variable_z)
input double InpRiskPercent   = 1.0;     // Risk percentage of equity per trade
input int    InpSLPoints      = 500;     // Stop Loss in Points
input int    InpTPPoints      = 1000;    // Take Profit in Points

//--- Optimization Parameters (Normal variables as requested)
double Slippage = 5;       // Slippage in points
ulong  MagicNumber = 123456; // Magic Number

//--- Weights for Optimization (Must sum to 100 approx)
double Wpr = 20.0;   // Weight for Payoff Ratio
double Wmr = 20.0;   // Weight for Monthly Return
double Wnp = 15.0;   // Weight for Negative Penalty
double Wtc = 20.0;   // Weight for Trade Count per Month
double Wsr = 15.0;   // Weight for Sharpe Ratio
double Wemd = 10.0;  // Weight for Risk of Ruin (Le)

//--- Sigmoid Parameters (Steepness k, Midpoint m)
double PrK = 0.5;    // Payoff Ratio Steepness
double PrM = 2.0;    // Payoff Ratio Midpoint
double MrK = 0.01;   // Monthly Return Steepness
double MrM = 500.0;  // Monthly Return Midpoint
double NpK = 0.005;  // Negative Penalty Steepness
double NpM = 500.0;  // Negative Penalty Midpoint
double TcK = 0.1;    // Trade Count Steepness
double TcM = 20.0;   // Trade Count Midpoint
double SrK = 0.5;    // Sharpe Ratio Steepness
double SrM = 1.0;    // Sharpe Ratio Midpoint
double LeK = 0.1;    // Risk of Ruin Steepness
double LeM = 10.0;   // Risk of Ruin Midpoint

//--- Global Objects and Variables
CTrade trade;
int    handle_sma_fast;
int    handle_sma_slow;
int    handle_stoch;
int    handle_lwma;

//--- Test tracking variables
datetime g_test_start = 0;
datetime g_test_end   = 0;

//+------------------------------------------------------------------+
//| Helper: Calculate number of months between two datetimes         |
//+------------------------------------------------------------------+
double CalcMonths(datetime start_time, datetime end_time)
  {
   if(start_time == 0 || end_time == 0)
      return(0.0);

   MqlDateTime start_struct, end_struct;
   TimeToStruct(start_time, start_struct);
   TimeToStruct(end_time,   end_struct);

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
//| Helper: Check if Friday 16:00 GMT+2 cutoff is reached            |
//+------------------------------------------------------------------+
bool IsFridayCutoffReached()
  {
   const int   GMT_PLUS_TWO = 2 * 60 * 60;
   datetime    now_gmt      = TimeGMT();
   datetime    now_gmt2     = now_gmt + GMT_PLUS_TWO;
   MqlDateTime time_struct;
   TimeToStruct(now_gmt2, time_struct);

   // 0 = Sunday ... 5 = Friday
   if(time_struct.day_of_week != 5)
      return(false);

   if(time_struct.hour > 16)
      return(true);

   if(time_struct.hour == 16 && time_struct.min >= 0)
      return(true);

   return(false);
  }

//+------------------------------------------------------------------+
//| Helper: Calculate Lot Size based on Risk %                       |
//+------------------------------------------------------------------+
double CalcLotSize(double risk_percent)
  {
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double step      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_vol   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_vol   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(tick_val <= 0 || tick_size <= 0 || InpSLPoints <= 0)
      return(min_vol);

   double risk_amount   = equity * risk_percent / 100.0;
   double stop_distance = InpSLPoints * _Point;
   double ticks         = stop_distance / tick_size;

   if(ticks <= 0) return min_vol;

   double lot = risk_amount / (ticks * tick_val);

   lot = MathFloor(lot / step) * step;
   lot = MathMax(min_vol, MathMin(max_vol, lot));

   return(lot);
  }

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialize Trade
   trade.SetExpertMagicNumber((long)MagicNumber);
   trade.SetDeviationInPoints((ulong)Slippage);

   // Initialize Indicators
   handle_sma_fast = iMA(_Symbol, _Period, InpSMAFastPeriod, 0, MODE_SMA, PRICE_CLOSE);
   handle_sma_slow = iMA(_Symbol, _Period, InpSMASlowPeriod, 0, MODE_SMA, PRICE_CLOSE);
   handle_stoch    = iStochastic(_Symbol, _Period, InpStochK, InpStochD, InpStochSlowing, MODE_SMA, STO_LOWHIGH);
   handle_lwma     = iMA(_Symbol, _Period, InpLWMAPeriod, 0, MODE_LWMA, PRICE_CLOSE);

   if(handle_sma_fast == INVALID_HANDLE || handle_sma_slow == INVALID_HANDLE ||
      handle_stoch == INVALID_HANDLE || handle_lwma == INVALID_HANDLE)
     {
      Print("Error creating indicators");
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(handle_sma_fast);
   IndicatorRelease(handle_sma_slow);
   IndicatorRelease(handle_stoch);
   IndicatorRelease(handle_lwma);

   g_test_end = TimeCurrent();
  }

//+------------------------------------------------------------------+
//| Close all trades on Friday cutoff                                |
//+------------------------------------------------------------------+
void CheckFridayClose()
  {
   if(IsFridayCutoffReached())
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket))
           {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
               trade.PositionClose(ticket);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Main Strategy Logic                                              |
//+------------------------------------------------------------------+
void CheckEntryLogic()
  {
   // Buffers for indicator values
   double sma_fast[], sma_slow[], stoch_main[], lwma[], close[];

   ArraySetAsSeries(sma_fast, true);
   ArraySetAsSeries(sma_slow, true);
   ArraySetAsSeries(stoch_main, true);
   ArraySetAsSeries(lwma, true);
   ArraySetAsSeries(close, true);

   // Copy data for the last finished bar (index 1)
   if(CopyBuffer(handle_sma_fast, 0, 1, 1, sma_fast) < 1) return;
   if(CopyBuffer(handle_sma_slow, 0, 1, 1, sma_slow) < 1) return;
   if(CopyBuffer(handle_stoch, 0, 1, 1, stoch_main) < 1) return;
   if(CopyBuffer(handle_lwma, 0, 1, 1, lwma) < 1) return;
   if(CopyClose(_Symbol, _Period, 1, 1, close) < 1) return;

   double fSMA = sma_fast[0];
   double sSMA = sma_slow[0];
   double stoch = stoch_main[0];
   double lWMA = lwma[0];
   double cls = close[0];

   // --- Trend Filter ---
   bool isBullish = (fSMA > sSMA);
   bool isBearish = (fSMA < sSMA);

   // --- Value Setup & Momentum Check ---
   // Buy: Bullish Trend + Stoch < 20 (Oversold) + Close > LWMA
   if(isBullish && stoch < InpStochOversold && cls > lWMA)
     {
      // Open Buy
      double lots = CalcLotSize(InpRiskPercent);
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - InpSLPoints * _Point;
      double tp = price + InpTPPoints * _Point;

      trade.Buy(lots, _Symbol, price, sl, tp, "AdaptiveMomentumTrend Buy");
     }
   // Sell: Bearish Trend + Stoch > 80 (Overbought) + Close < LWMA
   else if(isBearish && stoch > InpStochOverbought && cls < lWMA)
     {
      // Open Sell
      double lots = CalcLotSize(InpRiskPercent);
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + InpSLPoints * _Point;
      double tp = price - InpTPPoints * _Point;

      trade.Sell(lots, _Symbol, price, sl, tp, "AdaptiveMomentumTrend Sell");
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Update Test Stats
   if(g_test_start == 0) g_test_start = TimeCurrent();
   g_test_end = TimeCurrent();

   // Friday Close Rule
   CheckFridayClose();
   if(IsFridayCutoffReached()) return;

   // Logic only on new bar
   if(!IsNewBar()) return;

   CheckEntryLogic();
  }

//+------------------------------------------------------------------+
//| Sigmoid normalization function                                   |
//+------------------------------------------------------------------+
double SigmoidNormalize(double x, double k, double m)
{
    return (1.0 / (1.0 + MathPow(2.71828, -k * (x - m))));
}

//+------------------------------------------------------------------+
//| Calculate the average sum of absolute losses from negative months|
//+------------------------------------------------------------------+
double CalcNegativePenalty()
{
    double monthly_profits[];
    int num_months = (int)CalcMonths(g_test_start, g_test_end);
    if(num_months <= 0) return 0;

    ArrayResize(monthly_profits, num_months);
    ArrayInitialize(monthly_profits, 0.0);

    if(HistorySelect(g_test_start, g_test_end))
    {
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
    const double MaxRisk = 0.05; // 5%
    const double VERY_LARGE_ROR = 1.0e9; // A large number to signify certain ruin

    // --- Edge Case Handling ---
    if (prob_win >= 1.0) return 0.0; // 100% win rate = 0% RoR

    if (prob_win <= 0 || avg_win <= 0 || avg_loss >= 0)
    {
        // This print might be noisy in optimization, but requested for debugging purposes
        // printf("RoR Error: Invalid strategy parameters. prob_win=%.2f, avg_win=%.2f, avg_loss=%.2f", prob_win, avg_win, avg_loss);
        return VERY_LARGE_ROR;
    }

    double prob_loss = 1.0 - prob_win;

    // --- Formula Calculation ---
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    // Use initial equity or current? Usually RoR is based on current state or average stats relative to a base.
    // The memory implies calculating stats based on the test run.
    // However, if the account blew up, equity is small.
    // Use Tester Initial Deposit or Current Equity?
    // Let's use TesterStatistics(STAT_INITIAL_DEPOSIT) + Profit, which is current equity.

    if (equity <= 0) return VERY_LARGE_ROR;

    double avg_win_pct = avg_win / equity;
    double avg_loss_pct = MathAbs(avg_loss) / equity;

    double Z = (prob_win * avg_win_pct) - (prob_loss * avg_loss_pct);
    double A = MathSqrt((prob_win * MathPow(avg_win_pct, 2)) + (prob_loss * MathPow(avg_loss_pct, 2)));

    if (A == 0) return (Z > 0) ? 0.0 : VERY_LARGE_ROR;

    double P = 0.5 * (1 + (Z / A));

    // --- Final RoR Calculation ---
    if (P >= 1.0) return 0.0; // 100% chance of positive step = 0% RoR
    if (P <= 0) return VERY_LARGE_ROR;   // 0% chance of positive step = certain ruin

    double RoR = MathPow(((1.0 - P) / P), (MaxRisk / A));

    return RoR;
}

//+------------------------------------------------------------------+
//| Custom optimization criterion                                    |
//+------------------------------------------------------------------+
double OnTester()
{
    // --- METRIC CALCULATIONS ---
    double total_profit = TesterStatistics(STAT_PROFIT);
    double start_equity = TesterStatistics(STAT_INITIAL_DEPOSIT);
    double num_months = CalcMonths(g_test_start, g_test_end);
    double monthly_return_raw = (num_months > 0 && start_equity > 0) ? (total_profit / num_months) : 0; // In dollars (User requested: Total Profit (dollars) / Number of Months)
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

    // Risk of Ruin
    double risk_of_ruin_raw = CalcRiskOfRuin(win_rate, avg_win, avg_loss) * 100.0; // Normalized to % (e.g. 10.0 for 10%)

    // --- DEBUG OUTPUT (BEFORE NORMALIZATION) ---
    printf("--- Before Sigmoid Normalization ---");
    printf("Payoff Ratio (Pr): %.5f", payoff_ratio_raw);
    printf("Monthly Return (Mr): %.5f", monthly_return_raw);
    printf("Negative Penalty (Np): %.5f", negative_penalty_raw);
    printf("Trade Count/Month (Tc): %.5f", trades_per_month_raw);
    printf("Sharpe Ratio (Sr): %.5f", sharpe_ratio_raw);
    printf("Risk of Ruin (RoR): %.5f", risk_of_ruin_raw);

    // --- SIGMOID NORMALIZATION ---
    double payoff_ratio_norm = SigmoidNormalize(payoff_ratio_raw, PrK, PrM);
    double monthly_return_norm = SigmoidNormalize(monthly_return_raw, MrK, MrM);
    double negative_penalty_norm = SigmoidNormalize(negative_penalty_raw, NpK, NpM);
    double trades_per_month_norm = SigmoidNormalize(trades_per_month_raw, TcK, TcM);
    double sharpe_ratio_norm = SigmoidNormalize(sharpe_ratio_raw, SrK, SrM);
    double risk_of_ruin_norm = SigmoidNormalize(risk_of_ruin_raw, LeK, LeM);

    // --- DEBUG OUTPUT (AFTER NORMALIZATION) ---
    printf("--- After Sigmoid Normalization ---");
    printf("Payoff Ratio (Pr) Norm: %.5f", payoff_ratio_norm);
    printf("Monthly Return (Mr) Norm: %.5f", monthly_return_norm);
    printf("Negative Penalty (Np) Norm: %.5f", negative_penalty_norm);
    printf("Trade Count/Month (Tc) Norm: %.5f", trades_per_month_norm);
    printf("Sharpe Ratio (Sr) Norm: %.5f", sharpe_ratio_norm);
    printf("Risk of Ruin (RoR) Norm: %.5f", risk_of_ruin_norm);

    // --- WEIGHTING ---
    // User formula: Y = (Pr * Wpr) + (Mr * Wmr) - (Np * Wnp) - (Le * Wemd) + (Tc * Wtc) + (Sr * Wsr) * 100
    // Wait, the formula in prompt: Y = (Pr * Wpr) + (Mr * Wmr) - (Np * Wnp) - (Le * Wemd) + (Tc * Wtc) + (Sr * Wsr) * 100
    // Actually the parentheses are ambiguous in the prompt: ... + (Sr * Wsr)x100.
    // "The final objective function to be maximized, with weights (Wx) for each factor, is: Y = (Pr * Wpr) + ... + (Sr * Wsr)x100"
    // Usually it means the whole sum multiplied by 100 or just scaling.
    // The template has: double objective_score = ((payoff_ratio_norm * Wpr) ... ) * 100;
    // I will assume the sum is multiplied by 100.

    // Normalize weights if they don't sum to 1. But user said "Where all Wx weights sum to 100%".
    // If user inputs 20, 20, 15, etc., they sum to 100.
    // If I use them directly: (0.5 * 20) = 10.
    // If I normalize them to 1.0 (div by 100): (0.5 * 0.2) = 0.1.
    // The template divides by total_weight.
    // I will follow the template logic of normalizing weights to sum to 1.0.

    double total_weight = Wpr + Wmr + Wnp + Wtc + Wsr + Wemd;
    if (total_weight <= 0) total_weight = 1.0;

    double nWpr = Wpr / total_weight;
    double nWmr = Wmr / total_weight;
    double nWnp = Wnp / total_weight;
    double nWtc = Wtc / total_weight;
    double nWsr = Wsr / total_weight;
    double nWemd = Wemd / total_weight;

    double objective_score = ( (payoff_ratio_norm * nWpr) +
                               (monthly_return_norm * nWmr) -
                               (negative_penalty_norm * nWnp) -
                               (risk_of_ruin_norm * nWemd) +
                               (trades_per_month_norm * nWtc) +
                               (sharpe_ratio_norm * nWsr) ) * 100.0;

    printf("--- Final Score ---");
    printf("Objective Score: %.5f", objective_score);

    return(objective_score);
}
