#property copyright ""
#property link      ""
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input double InpRiskPerTradePercent = 1.0;       // Risk per trade in percent of equity
input int    InpLookbackCandles     = 2;         // Number of consecutive candles required
input double InpBodyThresholdPips   = 20.0;      // Minimum candle body size in pips
input double InpStopLossPips        = 15.0;      // Stop loss distance in pips
input double InpRiskRewardRatio     = 2.0;       // Risk-to-reward ratio for take profit
input bool   InpInvertDirection     = false;     // Trade opposite to candle direction
input int    InpMaxTradeCandles     = 16;        // Maximum number of candles to keep a position open (0 disables)
input ulong  InpMagicNumber         = 880088;    // Expert advisor magic number

//--- optimization weights (sum should be 100)
input double InpWeightTrades        = 40.0;      // Weight for trade density component (Wt)
input double InpWeightProfit        = 40.0;      // Weight for profitability per month (Wp)
input double InpWeightDrawdown      = 20.0;      // Weight for drawdown penalty (Wd)

//--- trading engine
CTrade trade;

datetime   g_lastBarTime   = 0;     // Tracks the time of the last processed bar
datetime   g_testStartTime = 0;     // Time stamp of the first processed bar during the test
datetime   g_testEndTime   = 0;     // Time stamp of the last processed bar during the test
uint       g_processedBars = 0;     // Number of closed bars evaluated by the strategy

//+------------------------------------------------------------------+
//| Helper: pip size for current symbol                              |
//+------------------------------------------------------------------+
double PipValue()
{
   if(_Digits == 3 || _Digits == 5)
      return(_Point * 10.0);
   return(_Point);
}

//+------------------------------------------------------------------+
//| Helper: normalize price                                          |
//+------------------------------------------------------------------+
double NormalizePrice(double price)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(step <= 0.0)
      step = _Point;

   price = MathRound(price / step) * step;
   return(NormalizeDouble(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)));
}

//+------------------------------------------------------------------+
//| Helper: calculate trading volume based on risk percent           |
//+------------------------------------------------------------------+
double CalculateVolume(double stop_distance_points)
{
   if(stop_distance_points <= 0.0)
      return(0.0);

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_amount = equity * InpRiskPerTradePercent / 100.0;

   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tick_value <= 0.0 || tick_size <= 0.0)
      return(0.0);

   double loss_per_lot = (stop_distance_points / tick_size) * tick_value;
   if(loss_per_lot <= 0.0)
      return(0.0);

   double volume = risk_amount / loss_per_lot;

   double min_volume   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(volume_step <= 0.0)
      volume_step = min_volume;

   volume = MathMax(min_volume, MathMin(volume, max_volume));
   volume = MathFloor(volume / volume_step) * volume_step;

   if(volume < min_volume)
      return(0.0);

   return(volume);
}

//+------------------------------------------------------------------+
//| Detect new bar                                                    |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(current_time == g_lastBarTime)
      return(false);

   g_lastBarTime = current_time;
   return(true);
}

//+------------------------------------------------------------------+
//| Evaluate signal based on candle bodies                           |
//+------------------------------------------------------------------+
bool CheckLargeCandles(const MqlRates &rates_array[], int rates_count, bool &isBullish)
{
   if(rates_count <= 0)
      return(false);

   double pip = PipValue();
   if(pip <= 0.0)
      return(false);

   for(int i = 0; i < rates_count; ++i)
   {
      double body_points = MathAbs(rates_array[i].close - rates_array[i].open);
      double body_pips   = body_points / pip;

      if(body_pips < InpBodyThresholdPips)
         return(false);
   }

   isBullish = (rates_array[0].close > rates_array[0].open);
   if(InpInvertDirection)
      isBullish = !isBullish;

   return(true);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(InpLookbackCandles <= 0)
   {
      Print("Invalid lookback setting. It must be greater than zero.");
      return(INIT_PARAMETERS_INCORRECT);
   }

   if(InpStopLossPips <= 0.0)
   {
      Print("Stop loss distance must be positive.");
      return(INIT_PARAMETERS_INCORRECT);
   }

   if(InpMaxTradeCandles < 0)
   {
      Print("Maximum trade candle limit cannot be negative.");
      return(INIT_PARAMETERS_INCORRECT);
   }

   trade.SetExpertMagicNumber((int)InpMagicNumber);
   trade.SetTypeFillingBySymbol(_Symbol);

   g_lastBarTime   = 0;
   g_testStartTime = 0;
   g_testEndTime   = 0;
   g_processedBars = 0;

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // No resources to release. Function intentionally left blank.
}

//+------------------------------------------------------------------+
//| Helper: close positions that exceeded max candle lifetime        |
//+------------------------------------------------------------------+
void CloseExpiredPositions()
{
   if(InpMaxTradeCandles == 0)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      string position_symbol = PositionGetString(POSITION_SYMBOL);
      if(position_symbol != _Symbol)
         continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(open_time == 0)
         continue;

      int open_shift = iBarShift(_Symbol, _Period, open_time, true);
      if(open_shift < 0)
         continue;

      if(open_shift < InpMaxTradeCandles)
         continue;

      if(!trade.PositionClose(ticket))
         PrintFormat("Failed to close position %I64u after exceeding max candles. Error %d", ticket, GetLastError());
      else
         PrintFormat("Closed position %I64u after exceeding max candles of %d.", ticket, InpMaxTradeCandles);
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   CloseExpiredPositions();

   if(!IsNewBar())
      return;

   MqlRates rates[];
   int count = CopyRates(_Symbol, _Period, 1, InpLookbackCandles, rates);
   if(count != InpLookbackCandles)
      return;

   bool isBullish = true;
   g_processedBars++;

   if(g_testStartTime == 0)
      g_testStartTime = rates[count - 1].time;

   g_testEndTime = rates[0].time;

   if(!CheckLargeCandles(rates, count, isBullish))
      return;

   double pip = PipValue();
   double stop_distance_points = InpStopLossPips * pip;
   double take_distance_points = stop_distance_points * InpRiskRewardRatio;

   double volume = CalculateVolume(stop_distance_points);
   if(volume <= 0.0)
   {
      Print("Calculated volume is zero. Trade skipped.");
      return;
   }

   double price = 0.0;
   double sl    = 0.0;
   double tp    = 0.0;

   if(isBullish)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl    = NormalizePrice(price - stop_distance_points);
      tp    = NormalizePrice(price + take_distance_points);

      if(!trade.Buy(volume, _Symbol, price, sl, tp, "Volatility"))
         PrintFormat("Buy order failed. Error %d", GetLastError());
   }
   else
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl    = NormalizePrice(price + stop_distance_points);
      tp    = NormalizePrice(price - take_distance_points);

      if(!trade.Sell(volume, _Symbol, price, sl, tp, "Volatility"))
         PrintFormat("Sell order failed. Error %d", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Helper: compute number of months between two timestamps          |
//+------------------------------------------------------------------+
int MonthsBetween(datetime start_time, datetime end_time)
{
   if(start_time == 0 || end_time == 0 || end_time < start_time)
      return(0);

   MqlDateTime start_dt, end_dt;
   TimeToStruct(start_time, start_dt);
   TimeToStruct(end_time, end_dt);

   int months = (end_dt.year - start_dt.year) * 12 + (end_dt.mon - start_dt.mon);

   if(end_dt.day >= start_dt.day)
      months += 1;

   if(months <= 0)
      months = 1;

   return(months);
}

//+------------------------------------------------------------------+
//| Tester function for custom optimization objective                 |
//+------------------------------------------------------------------+
double OnTester()
{
   double total_trades = TesterStatistics(STAT_TRADES);
   double total_profit = TesterStatistics(STAT_PROFIT);
   double initial_deposit = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double drawdown_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT) / 100.0;

   if(g_processedBars == 0)
      return(0.0);

   double trade_density = total_trades / (double)g_processedBars;

   int months = MonthsBetween(g_testStartTime, g_testEndTime);
   double monthly_profit = 0.0;
   if(initial_deposit > 0.0 && months > 0)
      monthly_profit = ((total_profit / initial_deposit) / (double)months);

   double weighted_score = trade_density * InpWeightTrades + monthly_profit * InpWeightProfit - drawdown_percent * InpWeightDrawdown;

   return(weighted_score * 100.0);
}

//+------------------------------------------------------------------+
