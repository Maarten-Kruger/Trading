#property copyright "Jules"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>

//--- Input parameters
input double InpRiskPercent      = 1.0;      // Risk percentage of equity per trade
input double InpPreSessionHours  = 2.0;      // Hours before London Open to monitor for band
input long   InpVolumeThreshold  = 100;      // Minimum volume for initial breakout
input int    InpMaxTimeStopMins  = 120;      // Maximum trade duration in minutes
input int    InpSLPoints         = 200;      // Stop Loss in points
input int    InpTPPoints         = 400;      // Take Profit in points

//--- Hardcoded Strategy Parameters
double Slippage       = 5.0;      // Slippage in points
ulong  MagicNumber    = 240001;   // Magic number

//--- Hardcoded Optimization Parameters
double Wpr            = 20.0;     // Weight for Payoff Ratio
double Wmr            = 20.0;     // Weight for Monthly Return
double Wnp            = 15.0;     // Weight for Negative Penalty
double Wtc            = 20.0;     // Weight for Trade Count per Month
double Wsr            = 15.0;     // Weight for Sharpe Ratio
double WRor           = 10.0;     // Weight for Risk of Ruin

double PrK            = 0.585927; // Payoff Ratio Steepness (k)
double PrM            = 4.25;     // Payoff Ratio Midpoint (m)
double MrK            = 0.005859; // Monthly Return Steepness (k)
double MrM            = 425.0;    // Monthly Return Midpoint (m)
double NpK            = 0.005493; // Negative Penalty Steepness (k)
double NpM            = 600.0;    // Negative Penalty Midpoint (m)
double TcK            = 0.079899; // Trade Count Steepness (k)
double TcM            = 42.5;     // Trade Count Midpoint (m)
double SrK            = 0.878890; // Sharpe Ratio Steepness (k)
double SrM            = 2.5;      // Sharpe Ratio Midpoint (m)
double RoRK           = 0.046257; // Risk of Ruin Steepness (k)
double RoRM           = 52.5;     // Risk of Ruin Midpoint (m)

//--- Global Objects
CTrade  trade;

//--- Test Tracking Variables
datetime g_test_start = 0;
datetime g_test_end   = 0;
int      g_total_bars = 0;

//--- Strategy State Variables
double   g_band_high        = -1.0;
double   g_band_low         = -1.0;
bool     g_has_broken_out   = false;
bool     g_momentum_stopped = false;
int      g_breakout_dir     = 0;     // 1 for Buy, -1 for Sell
long     g_last_volume      = 0;
datetime g_last_trade_time  = 0;

//+------------------------------------------------------------------+
//| Helper: Calculate number of months                               |
//+------------------------------------------------------------------+
double CalcMonths(datetime start_time, datetime end_time)
{
   if(start_time == 0 || end_time == 0) return 0.0;
   MqlDateTime start_struct, end_struct;
   TimeToStruct(start_time, start_struct);
   TimeToStruct(end_time,   end_struct);
   int months = (end_struct.year - start_struct.year) * 12 + (end_struct.mon - start_struct.mon) + 1;
   return (double)months;
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
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Helper: Check if it is South African Winter (Oct - Mar)          |
//+------------------------------------------------------------------+
bool IsWinter(datetime time)
{
   MqlDateTime dt;
   TimeToStruct(time, dt);
   // October (10) to March (3)
   if(dt.mon >= 10 || dt.mon <= 3) return true;
   return false;
}

//+------------------------------------------------------------------+
//| Get session times based on season                                |
//+------------------------------------------------------------------+
void GetSessionTimes(datetime time, int &london_start, int &london_end, int &lunch_start, int &lunch_end, int &restart_start, int &restart_end)
{
   bool winter = IsWinter(time);
   int shift = winter ? 1 : 0;

   // Base times (Summer)
   // London Open: 09:00 - 12:00
   // Lunch Lull:  12:00 - 14:30
   // Restart:     14:30 - 18:00

   london_start  = 9 + shift;
   london_end    = 12 + shift;
   lunch_start   = 12 + shift;
   lunch_end     = 14 + shift; // Hour part
   restart_start = 14 + shift; // Hour part
   restart_end   = 18 + shift;
}

//+------------------------------------------------------------------+
//| Check if current time is within trading windows                  |
//+------------------------------------------------------------------+
bool IsTradingSession(datetime time)
{
   int l_start, l_end, lunch_start, lunch_end, r_start, r_end;
   GetSessionTimes(time, l_start, l_end, lunch_start, lunch_end, r_start, r_end);

   MqlDateTime dt;
   TimeToStruct(time, dt);

   // Convert everything to minutes for easier comparison
   int current_mins = dt.hour * 60 + dt.min;

   int session1_start = l_start * 60;
   int session1_end   = l_end * 60; // 12:00 / 13:00

   int session2_start = r_start * 60 + 30; // 14:30 / 15:30
   int session2_end   = r_end * 60;        // 18:00 / 19:00

   if (current_mins >= session1_start && current_mins < session1_end) return true;
   if (current_mins >= session2_start && current_mins < session2_end) return true;

   return false;
}

//+------------------------------------------------------------------+
//| Check if we are in the monitoring phase (pre-London)             |
//+------------------------------------------------------------------+
bool IsMonitoringPhase(datetime time)
{
   int l_start, l_end, lunch_start, lunch_end, r_start, r_end;
   GetSessionTimes(time, l_start, l_end, lunch_start, lunch_end, r_start, r_end);

   MqlDateTime dt;
   TimeToStruct(time, dt);
   int current_mins = dt.hour * 60 + dt.min;

   int session_start = l_start * 60;
   int monitor_start = session_start - (int)(InpPreSessionHours * 60);

   if (current_mins >= monitor_start && current_mins < session_start) return true;

   // Also check pre-Restart? Prompt says "Again observe the lowest and highest closes before the New York open".
   // "From 12:00 to 14:30 SAST, you must stop all new entries and observe the Lunch Lull."
   // "Again observe the lowest and highest closes before the New York open at 14:30."
   // So Lunch Lull IS the monitoring phase for the afternoon session.
   int lunch_start_mins = lunch_start * 60;
   int lunch_end_mins   = lunch_start_mins + 150; // 2h 30m

   if (current_mins >= lunch_start_mins && current_mins < lunch_end_mins) return true;

   return false;
}

//+------------------------------------------------------------------+
//| Reset strategy state for a new session                           |
//+------------------------------------------------------------------+
void ResetSessionState()
{
   g_band_high = -1.0;
   g_band_low  = -1.0;
   g_has_broken_out = false;
   g_momentum_stopped = false;
   g_breakout_dir = 0;
   g_last_volume = 0;
}

//+------------------------------------------------------------------+
//| Update breakout band (High/Low of Closes)                        |
//+------------------------------------------------------------------+
void UpdateBands()
{
   double close = iClose(_Symbol, _Period, 1);

   if (g_band_high == -1.0) // Init
   {
      g_band_high = close;
      g_band_low = close;
   }
   else
   {
      if (close > g_band_high) g_band_high = close;
      if (close < g_band_low) g_band_low = close;
   }
}

//+------------------------------------------------------------------+
//| Friday Close Logic                                               |
//+------------------------------------------------------------------+
bool IsFridayCutoff()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.day_of_week == 5 && dt.hour >= 16) return true; // Friday 16:00
   return false;
}

//+------------------------------------------------------------------+
//| Calculate Lot Size                                               |
//+------------------------------------------------------------------+
double CalcLotSize(double risk_p)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if (tick_val <= 0 || tick_size <= 0 || InpSLPoints <= 0) return min_vol;

   double risk_amt = equity * risk_p / 100.0;
   double sl_dist = InpSLPoints * _Point;
   double ticks = sl_dist / tick_size;
   double lot = risk_amt / (ticks * tick_val);

   lot = MathFloor(lot / step) * step;
   return MathMax(min_vol, MathMin(max_vol, lot));
}

//+------------------------------------------------------------------+
//| Manage Exits (Time Stop & Friday)                                |
//+------------------------------------------------------------------+
void ManageExits()
{
   bool friday = IsFridayCutoff();

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;

      bool close = false;

      // Time Stop
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      long duration_sec = TimeCurrent() - open_time;
      if (duration_sec > InpMaxTimeStopMins * 60) close = true;

      if (friday) close = true;

      if (close) trade.PositionClose(ticket);
   }
}

//+------------------------------------------------------------------+
//| Entry Logic                                                      |
//+------------------------------------------------------------------+
void CheckEntry()
{
   datetime now = TimeCurrent();

   // 1. Monitor Phase: Update Bands
   if (IsMonitoringPhase(now))
   {
      // If we were just in a session, reset state logic is needed.
      // But simplest is to reset if we detect start of monitoring?
      // Or simply: If entering monitoring, bands reset automatically if we handle it right.
      // Let's reset bands if they are stale (from previous session).
      // A simple way: If we are in monitoring, we update. If we transition FROM trading TO monitoring, we must reset.

      // Check if we need to reset: if g_has_broken_out is true, we haven't reset yet.
      if (g_has_broken_out || g_band_high == -1.0)
      {
         ResetSessionState();
      }

      UpdateBands();
      return;
   }

   // 2. Trading Session Phase
   if (IsTradingSession(now))
   {
      if (IsFridayCutoff()) return;

      double close = iClose(_Symbol, _Period, 1);
      long vol = iVolume(_Symbol, _Period, 1);

      // Phase A: Initial Breakout
      if (!g_has_broken_out)
      {
         if (g_band_high != -1.0 && vol > InpVolumeThreshold)
         {
            double lot = CalcLotSize(InpRiskPercent);
            trade.SetDeviationInPoints((ulong)Slippage);

            if (close > g_band_high)
            {
               double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               double sl = price - InpSLPoints * _Point;
               double tp = price + InpTPPoints * _Point;
               if (trade.Buy(lot, _Symbol, price, sl, tp, "LondonBreakout"))
               {
                  g_has_broken_out = true;
                  g_breakout_dir = 1;
                  g_last_volume = vol;
                  g_momentum_stopped = false;
               }
            }
            else if (close < g_band_low)
            {
               double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
               double sl = price + InpSLPoints * _Point; // Sell SL is above
               double tp = price - InpTPPoints * _Point; // Sell TP is below
               if (trade.Sell(lot, _Symbol, price, sl, tp, "LondonBreakout"))
               {
                  g_has_broken_out = true;
                  g_breakout_dir = -1;
                  g_last_volume = vol;
                  g_momentum_stopped = false;
               }
            }
         }
      }
      // Phase B: Momentum Follow-up
      else if (g_has_broken_out && !g_momentum_stopped)
      {
         // "If the volume is still going up/growing, then it makes another entry."
         // "Until the volume starts decreasing"

         if (vol > g_last_volume)
         {
            // Add entry
            double lot = CalcLotSize(InpRiskPercent);
            double price = (g_breakout_dir == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double sl = (g_breakout_dir == 1) ? (price - InpSLPoints*_Point) : (price + InpSLPoints*_Point);
            double tp = (g_breakout_dir == 1) ? (price + InpTPPoints*_Point) : (price - InpTPPoints*_Point);

            bool res = false;
            if (g_breakout_dir == 1) res = trade.Buy(lot, _Symbol, price, sl, tp, "LondonBreakout Add");
            else res = trade.Sell(lot, _Symbol, price, sl, tp, "LondonBreakout Add");

            if (res)
            {
               g_last_volume = vol; // Update reference volume
            }
         }
         else
         {
            // Volume stopped growing
            g_momentum_stopped = true;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Deinitialization                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_test_end = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Tick Function                                                    |
//+------------------------------------------------------------------+
void OnTick()
{
   if (!IsNewBar()) return;

   if (g_test_start == 0) g_test_start = TimeCurrent();
   g_test_end = TimeCurrent();
   g_total_bars++;

   ManageExits();
   CheckEntry();
}

//+------------------------------------------------------------------+
//| Sigmoid Normalization                                            |
//+------------------------------------------------------------------+
double SigmoidNormalize(double x, double k, double m)
{
   return (1.0 / (1.0 + MathPow(2.71828, -k * (x - m))));
}

//+------------------------------------------------------------------+
//| Calculate Negative Penalty                                       |
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
//| Calculate Risk of Ruin                                           |
//+------------------------------------------------------------------+
double CalcRiskOfRuin(double prob_win, double avg_win, double avg_loss)
{
   const double MaxRisk = 0.05;
   const double VERY_LARGE_ROR = 1.0e9;

   if (prob_win >= 1.0) return 0.0;
   if (prob_win <= 0 || avg_win <= 0 || avg_loss >= 0)
   {
      printf("RoR Error: Invalid params. win=%.2f, avg_win=%.2f, avg_loss=%.2f", prob_win, avg_win, avg_loss);
      return VERY_LARGE_ROR;
   }

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
//| Custom Tester Objective                                          |
//+------------------------------------------------------------------+
double OnTester()
{
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

   printf("--- Before Sigmoid Normalization ---");
   printf("Payoff Ratio (Pr): %.2f", payoff_ratio_raw);
   printf("Monthly Return (Mr): %.2f", monthly_return_raw);
   printf("Negative Penalty (Np): %.2f", negative_penalty_raw);
   printf("Trade Count/Month (Tc): %.2f", trades_per_month_raw);
   printf("Sharpe Ratio (Sr): %.2f", sharpe_ratio_raw);
   printf("Risk of Ruin (RoR): %.2f", risk_of_ruin_raw);

   double payoff_ratio_norm = SigmoidNormalize(payoff_ratio_raw, PrK, PrM);
   double monthly_return_norm = SigmoidNormalize(monthly_return_raw, MrK, MrM);
   double negative_penalty_norm = SigmoidNormalize(negative_penalty_raw, NpK, NpM);
   double trades_per_month_norm = SigmoidNormalize(trades_per_month_raw, TcK, TcM);
   double sharpe_ratio_norm = SigmoidNormalize(sharpe_ratio_raw, SrK, SrM);
   double risk_of_ruin_norm = SigmoidNormalize(risk_of_ruin_raw, RoRK, RoRM);

   printf("--- After Sigmoid Normalization ---");
   printf("Payoff Ratio (Pr) Norm: %.4f", payoff_ratio_norm);
   printf("Monthly Return (Mr) Norm: %.4f", monthly_return_norm);
   printf("Negative Penalty (Np) Norm: %.4f", negative_penalty_norm);
   printf("Trade Count/Month (Tc) Norm: %.4f", trades_per_month_norm);
   printf("Sharpe Ratio (Sr) Norm: %.4f", sharpe_ratio_norm);
   printf("Risk of Ruin (RoR) Norm: %.4f", risk_of_ruin_norm);

   // Weights are hardcoded, but we must ensure they sum to 100 or normalize them?
   // Prompt: "Where all Wx weights sum to 100%."
   // Logic in reference code normalizes them. I will do the same to be safe.
   double total_weight = Wpr + Wmr + Wnp + Wtc + Wsr + WRor;
   if (total_weight <= 0) total_weight = 100.0;

   double nWpr = Wpr / total_weight;
   double nWmr = Wmr / total_weight;
   double nWnp = Wnp / total_weight;
   double nWtc = Wtc / total_weight;
   double nWsr = Wsr / total_weight;
   double nWRor = WRor / total_weight;

   // Formula: Y = (Pr*Wpr) + (Mr*Wmr) - (Np*Wnp) - (RoR*WRor) + (Tc*Wtc) + (Sr*Wsr) * 100
   // Note: The reference code multiplied (RoR * WRor) was subtracted?
   // Prompt says: "- (RoR * WRor)" (Risk of Ruin $L_e$).
   // And "- (Np * Wnp)".

   double objective_score = ((payoff_ratio_norm * nWpr) +
                             (monthly_return_norm * nWmr) -
                             (negative_penalty_norm * nWnp) +
                             (trades_per_month_norm * nWtc) +
                             (sharpe_ratio_norm * nWsr) -
                             (risk_of_ruin_norm * nWRor)) * 100.0;

   printf("--- Final Score ---");
   printf("Objective Score: %.4f", objective_score);

   return objective_score;
}
