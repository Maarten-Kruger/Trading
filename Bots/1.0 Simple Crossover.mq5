//+------------------------------------------------------------------+
//|                                        1.0 Simple Crossover.mq5 |
//|                                  Copyright 2024, Maarten Kruger |
//|                               https://github.com/Maarten-Kruger |
//+------------------------------------------------------------------+
#property copyright "Maarten Kruger"
#property link      "https://github.com/Maarten-Kruger"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- Enums
enum ENUM_MA_TYPE_CUSTOM
  {
   MA_SMA = 0, // Simple Moving Average
   MA_EMA = 1  // Exponential Moving Average
  };

enum ENUM_MA_COUNT
  {
   MA_COUNT_2 = 2, // 2 Moving Averages
   MA_COUNT_3 = 3, // 3 Moving Averages
   MA_COUNT_4 = 4  // 4 Moving Averages
  };

//--- Input Parameters
input group "Moving Averages"
input ENUM_MA_TYPE_CUSTOM InpMAType = MA_SMA;       // MA Type
input ENUM_MA_COUNT       InpMACount = MA_COUNT_2;  // MA Count
input int                 InpMA1Period = 10;        // MA 1 Period (Fastest)
input int                 InpMA2Period = 20;        // MA 2 Period
input int                 InpMA3Period = 50;        // MA 3 Period
input int                 InpMA4Period = 200;       // MA 4 Period (Slowest)

input group "Trend Filter"
input int                 InpADXPeriod = 14;        // ADX Period
input double              InpADXThreshold = 25.0;   // ADX Threshold

input group "Risk Management"
input double              InpRiskPercent = 1.0;     // Risk Percent per Trade
input int                 InpTPPoints = 500;        // Take Profit (Points)
input int                 InpSLPoints = 250;        // Stop Loss (Points)
input int                 InpMaxBarsOpen = 100;     // Max Bars Open

input group "Exits"
input bool                InpUseTrailing = true;    // Use Trailing Stop
input int                 InpTrailTrigger = 100;    // Trailing Start (Points profit)
input int                 InpTrailDist = 50;        // Trailing Distance (Points)
input bool                InpUseReverseExit = true; // Use Reverse Crossover Exit

//--- Global Optimization Variables (Not Inputs)
double Wpr = 20.0;          // Weight for Payoff Ratio
double Wmr = 20.0;          // Weight for Monthly Return
double Wnp = 15.0;          // Weight for Negative Penalty
double Wtc = 20.0;          // Weight for Trade Count
double Wsr = 15.0;          // Weight for Sharpe Ratio
double Wemd = 10.0;         // Weight for Risk of Ruin (Le)

double PrK = 0.585927;           // Payoff Ratio Steepness
double PrM = 4.25;           // Payoff Ratio Midpoint
double MrK = 0.005859;          // Monthly Return Steepness
double MrM = 425;         // Monthly Return Midpoint
double NpK = 0.005493;          // Negative Penalty Steepness
double NpM = 600.0;         // Negative Penalty Midpoint
double TcK = 0.079899;           // Trade Count Steepness
double TcM = 42.5;          // Trade Count Midpoint
double SrK = 0.878890;           // Sharpe Ratio Steepness
double SrM = 2.5;           // Sharpe Ratio Midpoint
double LeK = 0.046257;           // Risk of Ruin Steepness
double LeM = 52.5;          // Risk of Ruin Midpoint

double Slippage = 5.0;      // Slippage in points
ulong  MagicNumber = 123456;// Magic Number

//--- Global Objects
CTrade trade;

//--- Indicators
int hMA1, hMA2, hMA3, hMA4;
int hADX;

//--- Test Tracking
datetime g_test_start = 0;
datetime g_test_end = 0;
int      g_total_bars = 0;

//+------------------------------------------------------------------+
//| Helper: Get MA Method                                            |
//+------------------------------------------------------------------+
ENUM_MA_METHOD GetMAMethod()
  {
   return (InpMAType == MA_SMA) ? MODE_SMA : MODE_EMA;
  }

//+------------------------------------------------------------------+
//| Initialization function                                          |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints((ulong)Slippage);

   // Create MA Handles
   hMA1 = iMA(_Symbol, _Period, InpMA1Period, 0, GetMAMethod(), PRICE_CLOSE);
   hMA2 = iMA(_Symbol, _Period, InpMA2Period, 0, GetMAMethod(), PRICE_CLOSE);

   if(InpMACount >= MA_COUNT_3)
      hMA3 = iMA(_Symbol, _Period, InpMA3Period, 0, GetMAMethod(), PRICE_CLOSE);

   if(InpMACount == MA_COUNT_4)
      hMA4 = iMA(_Symbol, _Period, InpMA4Period, 0, GetMAMethod(), PRICE_CLOSE);

   // Create ADX Handle
   hADX = iADX(_Symbol, _Period, InpADXPeriod);

   if(hMA1 == INVALID_HANDLE || hMA2 == INVALID_HANDLE || hADX == INVALID_HANDLE ||
      (InpMACount >= MA_COUNT_3 && hMA3 == INVALID_HANDLE) ||
      (InpMACount == MA_COUNT_4 && hMA4 == INVALID_HANDLE))
     {
      Print("Error creating indicator handles");
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Deinitialization function                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(hMA1);
   IndicatorRelease(hMA2);
   if(InpMACount >= MA_COUNT_3) IndicatorRelease(hMA3);
   if(InpMACount == MA_COUNT_4) IndicatorRelease(hMA4);
   IndicatorRelease(hADX);
  }

//+------------------------------------------------------------------+
//| Update test stats                                                |
//+------------------------------------------------------------------+
void UpdateTestStats()
  {
   datetime now = TimeCurrent();
   if(g_test_start == 0) g_test_start = now;
   g_test_end = now;
   g_total_bars++;
  }

//+------------------------------------------------------------------+
//| Calculate Months                                                 |
//+------------------------------------------------------------------+
double CalcMonths(datetime start, datetime end)
  {
   if(start == 0 || end == 0) return 0.0;
   MqlDateTime dt1, dt2;
   TimeToStruct(start, dt1);
   TimeToStruct(end, dt2);
   return ((dt2.year - dt1.year) * 12 + (dt2.mon - dt1.mon) + 1);
  }

//+------------------------------------------------------------------+
//| Check Friday 16:00 Close (GMT+2)                                 |
//+------------------------------------------------------------------+
bool IsFridayCutoffReached()
  {
   datetime now = TimeGMT();
   // Strategy operates on GMT+2
   datetime now_gmt2 = now + 7200; // 2 * 3600

   MqlDateTime dt;
   TimeToStruct(now_gmt2, dt);

   if(dt.day_of_week == 5) // Friday
     {
      if(dt.hour > 16) return true;
      if(dt.hour == 16 && dt.min >= 0) return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Calculate Lot Size                                               |
//+------------------------------------------------------------------+
double CalcLotSize()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(tick_val == 0 || tick_size == 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   double risk_money = equity * InpRiskPercent / 100.0;
   double sl_money = InpSLPoints * tick_val; // Value of SL per 1 lot

   if(sl_money == 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   double lots = risk_money / sl_money;
   lots = MathFloor(lots / lot_step) * lot_step;

   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(lots < min_lot) lots = min_lot;
   if(lots > max_lot) lots = max_lot;

   return lots;
  }

//+------------------------------------------------------------------+
//| Main OnTick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   UpdateTestStats();

   //--- 1. Check Friday Cutoff
   if(IsFridayCutoffReached())
     {
      if(PositionsTotal() > 0)
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
      return; // Stop processing
     }

   //--- Data Gathering
   double ma1[], ma2[], ma3[], ma4[], adx[];
   ArraySetAsSeries(ma1, true);
   ArraySetAsSeries(ma2, true);
   ArraySetAsSeries(ma3, true);
   ArraySetAsSeries(ma4, true);
   ArraySetAsSeries(adx, true);

   CopyBuffer(hMA1, 0, 0, 3, ma1);
   CopyBuffer(hMA2, 0, 0, 3, ma2);
   if(InpMACount >= MA_COUNT_3) CopyBuffer(hMA3, 0, 0, 3, ma3);
   if(InpMACount == MA_COUNT_4) CopyBuffer(hMA4, 0, 0, 3, ma4);
   CopyBuffer(hADX, 0, 0, 2, adx);

   double close[];
   ArraySetAsSeries(close, true);
   CopyClose(_Symbol, _Period, 0, 3, close);

   //--- 2. Manage Open Positions
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      long type = PositionGetInteger(POSITION_TYPE);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);

      // A. Max Bars Exit
      int bars_open = iBarShift(_Symbol, _Period, open_time);
      if(bars_open >= InpMaxBarsOpen)
        {
         trade.PositionClose(ticket);
         continue;
        }

      // B. Reverse Crossover Exit
      if(InpUseReverseExit)
        {
         bool reverse_signal = false;
         // Use MA1 crossing MA2 as the trigger for reversal
         if(type == POSITION_TYPE_BUY && ma1[1] < ma2[1]) reverse_signal = true;
         if(type == POSITION_TYPE_SELL && ma1[1] > ma2[1]) reverse_signal = true;

         if(reverse_signal)
           {
            trade.PositionClose(ticket);
            continue;
           }
        }

      // C. Trailing Stop
      if(InpUseTrailing)
        {
         double profit_points = 0;
         if(type == POSITION_TYPE_BUY)
            profit_points = (current_price - open_price) / _Point;
         else
            profit_points = (open_price - current_price) / _Point;

         if(profit_points > InpTrailTrigger)
           {
            double new_sl = 0;
            if(type == POSITION_TYPE_BUY)
              {
               new_sl = current_price - InpTrailDist * _Point;
               if(new_sl > current_sl && new_sl < current_price)
                  trade.PositionModify(ticket, new_sl, current_tp);
              }
            else
              {
               new_sl = current_price + InpTrailDist * _Point;
               if((new_sl < current_sl || current_sl == 0) && new_sl > current_price)
                  trade.PositionModify(ticket, new_sl, current_tp);
              }
           }
        }
     }

   //--- 3. Entry Logic

   // Trend Filter
   if(adx[0] <= InpADXThreshold) return;

   bool buy_signal = false;
   bool sell_signal = false;

   if(InpMACount == MA_COUNT_2)
     {
      // Fast (MA1) Crosses Above Slow (MA2)
      if(ma1[1] > ma2[1] && ma1[2] <= ma2[2]) buy_signal = true;
      if(ma1[1] < ma2[1] && ma1[2] >= ma2[2]) sell_signal = true;
     }
   else
     {
      // 3 or 4 MA Stack
      bool stack_buy = false;
      bool stack_sell = false;

      // Check current stack
      if(close[1] > ma1[1] && ma1[1] > ma2[1] && ma2[1] > ma3[1])
        {
         if(InpMACount == MA_COUNT_3) stack_buy = true;
         else if(ma3[1] > ma4[1]) stack_buy = true;
        }

      if(close[1] < ma1[1] && ma1[1] < ma2[1] && ma2[1] < ma3[1])
        {
         if(InpMACount == MA_COUNT_3) stack_sell = true;
         else if(ma3[1] < ma4[1]) stack_sell = true;
        }

      // Check previous stack to detect NEW formation
      bool prev_stack_buy = false;
      bool prev_stack_sell = false;

      if(close[2] > ma1[2] && ma1[2] > ma2[2] && ma2[2] > ma3[2])
        {
         if(InpMACount == MA_COUNT_3) prev_stack_buy = true;
         else if(ma3[2] > ma4[2]) prev_stack_buy = true;
        }

      if(close[2] < ma1[2] && ma1[2] < ma2[2] && ma2[2] < ma3[2])
        {
         if(InpMACount == MA_COUNT_3) prev_stack_sell = true;
         else if(ma3[2] < ma4[2]) prev_stack_sell = true;
        }

      if(stack_buy && !prev_stack_buy) buy_signal = true;
      if(stack_sell && !prev_stack_sell) sell_signal = true;
     }

   // Execution
   if(buy_signal)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - InpSLPoints * _Point;
      double tp = price + InpTPPoints * _Point;
      trade.Buy(CalcLotSize(), _Symbol, price, sl, tp, "1.0 Simple Crossover");
     }
   else if(sell_signal)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + InpSLPoints * _Point;
      double tp = price - InpTPPoints * _Point;
      trade.Sell(CalcLotSize(), _Symbol, price, sl, tp, "1.0 Simple Crossover");
     }
  }

//+------------------------------------------------------------------+
//| Sigmoid Normalization                                            |
//+------------------------------------------------------------------+
double SigmoidNormalize(double x, double k, double m)
  {
   return (1.0 / (1.0 + MathExp(-k * (x - m))));
  }

//+------------------------------------------------------------------+
//| Calculate Negative Penalty                                       |
//+------------------------------------------------------------------+
double CalcNegativePenalty()
  {
   int num_months = (int)CalcMonths(g_test_start, g_test_end);
   if(num_months <= 0) return 0;

   double monthly_profits[];
   ArrayResize(monthly_profits, num_months);
   ArrayInitialize(monthly_profits, 0.0);

   HistorySelect(g_test_start, g_test_end);
   for(int i = 0; i < HistoryDealsTotal(); i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0)
        {
         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         datetime time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);

         MqlDateTime t1, t2;
         TimeToStruct(g_test_start, t1);
         TimeToStruct(time, t2);

         int m_idx = (t2.year - t1.year) * 12 + (t2.mon - t1.mon);
         if(m_idx >= 0 && m_idx < num_months)
            monthly_profits[m_idx] += profit;
        }
     }

   double sum_loss = 0;
   int count = 0;
   for(int i = 0; i < num_months; i++)
     {
      if(monthly_profits[i] < 0)
        {
         sum_loss += MathAbs(monthly_profits[i]);
         count++;
        }
     }

   return (count > 0) ? sum_loss / count : 0.0;
  }

//+------------------------------------------------------------------+
//| Calculate Risk of Ruin                                           |
//+------------------------------------------------------------------+
double CalcRiskOfRuin(double prob_win, double avg_win, double avg_loss)
  {
   double MaxRisk = 0.05;
   if(prob_win >= 1.0) return 0.0;
   if(prob_win <= 0 || avg_win <= 0 || avg_loss >= 0) return 100.0; // Fail state

   double prob_loss = 1.0 - prob_win;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY); // Current equity
   // Ideally should be initial equity or average equity, but usually current or initial is used.
   // Using Initial Deposit for consistency with % metrics if possible, but formula usually uses current unit size.
   // Logic from memory uses 'avg_win%' which is avg_win/equity.

   if(equity <= 0) return 100.0;

   double win_pct = avg_win / equity;
   double loss_pct = MathAbs(avg_loss) / equity;

   double Z = (prob_win * win_pct) - (prob_loss * loss_pct);
   double A = MathSqrt((prob_win * MathPow(win_pct, 2)) + (prob_loss * MathPow(loss_pct, 2)));

   if(A == 0) return 100.0;

   double P = 0.5 * (1 + Z/A);
   if(P >= 1.0) return 0.0;
   if(P <= 0) return 100.0; // 100% RoR

   double RoR = MathPow((1 - P)/P, MaxRisk/A);
   return RoR * 100.0; // Return as percent
  }

//+------------------------------------------------------------------+
//| OnTester - Optimization Logic                                    |
//+------------------------------------------------------------------+
double OnTester()
  {
   double total_profit = TesterStatistics(STAT_PROFIT);
   double total_trades = TesterStatistics(STAT_TRADES);
   double sharpe = TesterStatistics(STAT_SHARPE_RATIO);
   double profit_trades = TesterStatistics(STAT_PROFIT_TRADES);
   double loss_trades = TesterStatistics(STAT_LOSS_TRADES);
   double gross_profit = TesterStatistics(STAT_GROSS_PROFIT);
   double gross_loss = TesterStatistics(STAT_GROSS_LOSS);

   double num_months = CalcMonths(g_test_start, g_test_end);
   if(num_months < 1) num_months = 1;

   // 1. Payoff Ratio
   double avg_win = (profit_trades > 0) ? gross_profit / profit_trades : 0;
   double avg_loss = (loss_trades > 0) ? gross_loss / loss_trades : 0; // Negative value
   double payoff = (avg_loss != 0) ? avg_win / MathAbs(avg_loss) : 0;

   // 2. Monthly Return
   double monthly_return = total_profit / num_months;

   // 3. Negative Penalty
   double penalty = CalcNegativePenalty();

   // 4. Trade Count
   double trades_per_month = total_trades / num_months;

   // 5. Sharpe Ratio (already got)

   // 6. Risk of Ruin
   double win_rate = (total_trades > 0) ? profit_trades / total_trades : 0;
   double ror = CalcRiskOfRuin(win_rate, avg_win, avg_loss);

   // Debug Prints (Pre-Norm)
   printf("--- Pre-Norm ---");
   printf("Pr: %.4f, Mr: %.4f, Np: %.4f, Tc: %.4f, Sr: %.4f, Le: %.4f",
          payoff, monthly_return, penalty, trades_per_month, sharpe, ror);

   // Normalization
   double n_pr = SigmoidNormalize(payoff, PrK, PrM);
   double n_mr = SigmoidNormalize(monthly_return, MrK, MrM);
   double n_np = SigmoidNormalize(penalty, NpK, NpM);
   double n_tc = SigmoidNormalize(trades_per_month, TcK, TcM);
   double n_sr = SigmoidNormalize(sharpe, SrK, SrM);
   double n_le = SigmoidNormalize(ror, LeK, LeM);

   // Debug Prints (Post-Norm)
   printf("--- Post-Norm ---");
   printf("n_Pr: %.4f, n_Mr: %.4f, n_Np: %.4f, n_Tc: %.4f, n_Sr: %.4f, n_Le: %.4f",
          n_pr, n_mr, n_np, n_tc, n_sr, n_le);

   // Final Score
   // Y = (Pr * Wpr) + (Mr * Wmr) - (Np * Wnp) - (Le * Wemd) + (Tc * Wtc) + (Sr * Wsr) * 100
   // Note: The formula description in prompt says: "... + (Sr * Wsr) * 100".
   // It also says "Where all Wx weights sum to 100%".
   // Usually the *100 is applied to the whole sum or the weights are percentages.
   // The prompt formula: Y = (Pr*Wpr) + (Mr*Wmr) - (Np*Wnp) - (Le*Wemd) + (Tc*Wtc) + (Sr*Wsr) * 100
   // Order of operations: Only Sr term is multiplied by 100?
   // Or is it (Sum) * 100?
   // Looking at 1.6 HoverBreakout logic provided in memory:
   // double objective_score = ((payoff_ratio_norm * Wpr) + ... ) * 100;
   // I will assume the sum is multiplied by 100.

   double sum_weights = Wpr + Wmr + Wnp + Wtc + Wsr + Wemd;
   // Normalize weights to 1 (if they are percentages like 20, 20 etc)
   // But if I strictly follow the formula "Where all Wx weights sum to 100%" implies they are already scaled.
   // I will just use them as is if they sum to 100.
   // But for safety, I will normalize them so the total score is 0-100 range.

   double w_pr = Wpr / sum_weights;
   double w_mr = Wmr / sum_weights;
   double w_np = Wnp / sum_weights;
   double w_tc = Wtc / sum_weights;
   double w_sr = Wsr / sum_weights;
   double w_le = Wemd / sum_weights;

   double score = (n_pr * w_pr) + (n_mr * w_mr) - (n_np * w_np) - (n_le * w_le) + (n_tc * w_tc) + (n_sr * w_sr);
   score *= 100.0;

   printf("Final Score: %.4f", score);

   return score;
  }
