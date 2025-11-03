#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.5"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input int    InpRangeBars      = 10;     // Number of bars to define range (excluding breakout bar)
input double InpRangePoints    = 200;    // Maximum range size in points
input double InpTPPoints       = 400;    // Take profit distance in points
input double InpSLPoints       = 200;    // Stop loss distance in points
input double InpRiskPercent    = 1.0;    // Risk percentage of equity per trade
input int    InpMaxBarsOpen    = 5;      // Maximum bars to keep position open
uint   InpSlippage       = 5;      // Slippage in points
double InpWpr            = 20.0;   // Weight for Payoff Ratio
double InpWmr            = 20.0;   // Weight for Monthly Return
double InpWnp            = 15.0;   // Weight for Negative Penalty
double InpWtc            = 20.0;   // Weight for Trade Count per Month
double InpWsr            = 15.0;   // Weight for Sharpe Ratio
double InpWRor           = 10.0;   // Weight for Risk of Ruin
double InpPrK            = 0.585927;    // Payoff Ratio Steepness (k)
double InpPrM            = 4.25;    // Payoff Ratio Midpoint (m)
double InpMrK            = 0.005859;    // Monthly Return Steepness (k)
double InpMrM            = 425;   // Monthly Return Midpoint (m)
double InpNpK            = 0.005493;  // Negative Penalty Steepness (k)
double InpNpM            = 600.0;  // Negative Penalty Midpoint (m)
double InpTcK            = 0.079899;    // Trade Count Steepness (k)
double InpTcM            = 42.5;   // Trade Count Midpoint (m)
double InpSrK            = 0.878890;    // Sharpe Ratio Steepness (k)
double InpSrM            = 2.5;    // Sharpe Ratio Midpoint (m)
double InpRoRK           = 0.046257;   // Risk of Ruin Steepness (k)
double InpRoRM           = 52.5;   // Risk of Ruin Midpoint (m)
ulong  InpMagicNumber    = 1300001;// Magic number for HoverBreakout trades

//--- global objects
CTrade  trade;               // trading object

//--- test tracking variables
datetime g_test_start = 0;   // first bar time in test
datetime g_test_end   = 0;   // last bar time in test
int      g_total_bars = 0;   // number of processed bars

//+------------------------------------------------------------------+
//| Update test time and bar counters                                |
//+------------------------------------------------------------------+
void UpdateTestStats()
  {
   datetime now = TimeCurrent();
   if(g_test_start == 0)
      g_test_start = now;
   g_test_end = now;
   g_total_bars++;
  }

//+------------------------------------------------------------------+
//| Calculate number of months between two datetimes                 |
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
//| Helper: detect new bar                                          |
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
//| Calculate hover range from past bars                            |
//| Returns true if bars stayed within specified range               |
//+------------------------------------------------------------------+
bool CalcRange(int bars_back, double &range_high, double &range_low)
  {
   if(bars_back < 2)
      return(false);

// initialise with second previous bar because bar1 is the breakout bar
   range_high = iHigh(_Symbol, _Period, 2);
   range_low  = iLow(_Symbol,  _Period, 2);

   for(int i = 3; i <= bars_back + 1; i++)
     {
      double h = iHigh(_Symbol, _Period, i);
      double l = iLow(_Symbol,  _Period, i);
      if(h > range_high)
         range_high = h;
      if(l < range_low)
         range_low  = l;
     }
   return((range_high - range_low) <= InpRangePoints * _Point);
  }

//+------------------------------------------------------------------+
//| Determine if Friday 16:00 GMT+2 close time has been reached      |
//+------------------------------------------------------------------+
bool IsFridayCutoffReached()
  {
   const int   GMT_PLUS_TWO = 2 * 60 * 60;
   datetime    now_gmt      = TimeGMT();
   datetime    now_gmt2     = now_gmt + GMT_PLUS_TWO;
   MqlDateTime time_struct;
   TimeToStruct(now_gmt2, time_struct);

   if(time_struct.day_of_week != 5) // 0 = Sunday ... 5 = Friday
      return(false);

   if(time_struct.hour > 16)
      return(true);

   if(time_struct.hour == 16 && time_struct.min >= 0)
      return(true);

   return(false);
  }

//+------------------------------------------------------------------+
//| Calculate trade volume based on equity risk                     |
//+------------------------------------------------------------------+
double CalcLotSize(double risk_percent)
  {
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size= SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double step     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_vol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

// prevent division by zero
   if(tick_val <= 0 || tick_size <= 0 || InpSLPoints <= 0)
      return(min_vol);

   double risk_amount   = equity * risk_percent / 100.0;
   double stop_distance = InpSLPoints * _Point;
   double ticks         = stop_distance / tick_size;
   double lot           = risk_amount / (ticks * tick_val);

// adjust to broker limits
   lot = MathFloor(lot / step) * step;
   lot = MathMax(min_vol, MathMin(max_vol, lot));

   return(lot);
  }

//+------------------------------------------------------------------+
//| Close open position after a number of bars                       |
//+------------------------------------------------------------------+
void CheckForExit()
  {
// Iterate through all open positions and close those that exceed the
// maximum number of bars specified in InpMaxBarsOpen.
   bool friday_cutoff = IsFridayCutoffReached();

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      if(symbol != _Symbol)
         continue;

      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic != (long)InpMagicNumber)
         continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      int bars_open = iBarShift(_Symbol, _Period, open_time);
      bool close_due_to_bars = (bars_open >= InpMaxBarsOpen && bars_open >= 0);

      if(friday_cutoff || close_due_to_bars)
         trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
//| Entry logic: trade on breakout from tight range                 |
//+------------------------------------------------------------------+
void CheckForEntry()
  {
   double high, low;
   if(!CalcRange(InpRangeBars, high, low))
      return; // Range condition not met

   if(IsFridayCutoffReached())
      return; // Do not open new trades after the cutoff time

   double last_close = iClose(_Symbol, _Period, 1);

   trade.SetDeviationInPoints(InpSlippage);

   double lots = CalcLotSize(InpRiskPercent);

// Breakout above the range -> buy
   if(last_close > high)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - InpSLPoints * _Point;
      double tp = price + InpTPPoints * _Point;
      trade.Buy(lots, _Symbol, price, sl, tp, "HoverBreakout");
     }
// Breakout below the range -> sell
   else
      if(last_close < low)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + InpSLPoints * _Point;
         double tp = price - InpTPPoints * _Point;
         trade.Sell(lots, _Symbol, price, sl, tp, "HoverBreakout");
        }
  }

//+------------------------------------------------------------------+
//| Expert initialization                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber((long)InpMagicNumber);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   g_test_end = TimeCurrent();
  }

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
// Only run logic once per new bar, loops otherwise
   bool new_bar = IsNewBar();
   if(!new_bar)
      return;

   UpdateTestStats();

   CheckForExit();   // manage existing position
   CheckForEntry();  // look for new opportunity
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
    const double MaxRisk = 0.05; // 5%
    const double VERY_LARGE_ROR = 1.0e9; // A large number to signify certain ruin

    // --- Edge Case Handling ---
    if (prob_win >= 1.0) return 0.0; // 100% win rate = 0% RoR

    if (prob_win <= 0 || avg_win <= 0 || avg_loss >= 0)
    {
        printf("RoR Error: Invalid strategy parameters. prob_win=%.2f, avg_win=%.2f, avg_loss=%.2f", prob_win, avg_win, avg_loss);
        return VERY_LARGE_ROR;
    }

    double prob_loss = 1.0 - prob_win;

    // --- Formula Calculation ---
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    if (equity <= 0)
    {
        printf("RoR Error: Cannot calculate with zero or negative equity.");
        return VERY_LARGE_ROR;
    }

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
    double monthly_return_raw = (num_months > 0 && start_equity > 0) ? (total_profit / num_months) : 0; // In dollars
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
    double risk_of_ruin_raw = CalcRiskOfRuin(win_rate, avg_win, avg_loss) * 100.0; // Normalized to %

    // --- DEBUG OUTPUT (BEFORE NORMALIZATION) ---
    printf("--- Before Sigmoid Normalization ---");
    printf("Payoff Ratio (Pr): %.2f", payoff_ratio_raw);
    printf("Monthly Return (Mr): %.2f", monthly_return_raw);
    printf("Negative Penalty (Np): %.2f", negative_penalty_raw);
    printf("Trade Count/Month (Tc): %.2f", trades_per_month_raw);
    printf("Sharpe Ratio (Sr): %.2f", sharpe_ratio_raw);
    printf("Risk of Ruin (RoR): %.2f", risk_of_ruin_raw);

    // --- SIGMOID NORMALIZATION ---
    double payoff_ratio_norm = SigmoidNormalize(payoff_ratio_raw, InpPrK, InpPrM);
    double monthly_return_norm = SigmoidNormalize(monthly_return_raw, InpMrK, InpMrM);
    double negative_penalty_norm = SigmoidNormalize(negative_penalty_raw, InpNpK, InpNpM);
    double trades_per_month_norm = SigmoidNormalize(trades_per_month_raw, InpTcK, InpTcM);
    double sharpe_ratio_norm = SigmoidNormalize(sharpe_ratio_raw, InpSrK, InpSrM);
    double risk_of_ruin_norm = SigmoidNormalize(risk_of_ruin_raw, InpRoRK, InpRoRM);

    // --- DEBUG OUTPUT (AFTER NORMALIZATION) ---
    printf("--- After Sigmoid Normalization ---");
    printf("Payoff Ratio (Pr) Norm: %.4f", payoff_ratio_norm);
    printf("Monthly Return (Mr) Norm: %.4f", monthly_return_norm);
    printf("Negative Penalty (Np) Norm: %.4f", negative_penalty_norm);
    printf("Trade Count/Month (Tc) Norm: %.4f", trades_per_month_norm);
    printf("Sharpe Ratio (Sr) Norm: %.4f", sharpe_ratio_norm);
    printf("Risk of Ruin (RoR) Norm: %.4f", risk_of_ruin_norm);

    // --- WEIGHTING ---
    double total_weight = InpWpr + InpWmr + InpWnp + InpWtc + InpWsr + InpWRor;
    if (total_weight <= 0) total_weight = 1.0; // Avoid division by zero

    double Wpr = InpWpr / total_weight;
    double Wmr = InpWmr / total_weight;
    double Wnp = InpWnp / total_weight;
    double Wtc = InpWtc / total_weight;
    double Wsr = InpWsr / total_weight;
    double WRor = InpWRor / total_weight;

    // --- FINAL OBJECTIVE FUNCTION ---
    double objective_score = ((payoff_ratio_norm * Wpr) +
                             (monthly_return_norm * Wmr) -
                             (negative_penalty_norm * Wnp) +
                             (trades_per_month_norm * Wtc) +
                             (sharpe_ratio_norm * Wsr) -
                             (risk_of_ruin_norm * WRor))*100;

    printf("--- Final Score ---");
    printf("Objective Score: %.4f", objective_score);

    return(objective_score);
}

