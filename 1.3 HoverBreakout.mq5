#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.3"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input int    InpRangeDataPoints = 10;    // Number of data points to define range (excluding breakout point)
input double InpRangePoints     = 200;   // Maximum range size in points
input double InpTPPoints        = 400;   // Take profit distance in points
input double InpSLPoints        = 200;   // Stop loss distance in points
input double InpRiskPercent     = 1.0;   // Risk percentage of equity per trade
input int    InpMaxPointsOpen   = 5;     // Maximum data points to keep position open
input uint   InpIntervalSeconds = 60;    // Seconds between price measurements
input uint   InpSlippage        = 5;     // Slippage in points
input double InpWt              = 40.0;  // Weight % for trade density
input double InpWp              = 40.0;  // Weight % for monthly consistency
input double InpWd              = 20.0;  // Weight % for drawdown

//--- global objects
CTrade  trade;               // trading object

//--- collected price data
double  g_prices[];          // recorded prices at custom intervals

//--- test tracking variables
datetime g_test_start = 0;   // first data point time in test
datetime g_test_end   = 0;   // last data point time in test
int      g_total_points = 0; // number of processed data points

//+------------------------------------------------------------------+
//| Update test time and data point counters                         |
//+------------------------------------------------------------------+
void UpdateTestStats()
  {
   datetime now = TimeCurrent();
   if(g_test_start == 0)
      g_test_start = now;
   g_test_end = now;
   g_total_points++;
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
//| Helper: detect new custom interval                               |
//+------------------------------------------------------------------+
bool IsNewInterval()
  {
   static datetime last_time = 0;
   datetime now = TimeCurrent();
   if(now - last_time >= (int)InpIntervalSeconds)
     {
      last_time = now;
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Record current price at the custom interval                      |
//+------------------------------------------------------------------+
void RecordPrice()
  {
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int size = ArraySize(g_prices);
   ArrayResize(g_prices, size + 1);
   g_prices[size] = price;

   int max_size = InpRangeDataPoints + 2;
   if(ArraySize(g_prices) > max_size)
     {
      // keep only the most recent elements
      int start = ArraySize(g_prices) - max_size;
      for(int i = 0; i < max_size; i++)
         g_prices[i] = g_prices[start + i];
      ArrayResize(g_prices, max_size);
     }
  }

//+------------------------------------------------------------------+
//| Calculate hover range from past data points                     |
//| Returns true if points stayed within specified range            |
//+------------------------------------------------------------------+
bool CalcRange(int points_back, double &range_high, double &range_low)
  {
   int size = ArraySize(g_prices);
   if(points_back < 1 || size < points_back + 2)
      return(false);

   int start = size - 2 - points_back;
   if(start < 0)
      return(false);

   range_high = g_prices[start];
   range_low  = g_prices[start];

   for(int i = start + 1; i <= size - 3; i++)
     {
      double p = g_prices[i];
      if(p > range_high)
         range_high = p;
      if(p < range_low)
         range_low  = p;
     }
   return((range_high - range_low) <= InpRangePoints * _Point);
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
//| Close open position after a number of data points                |
//+------------------------------------------------------------------+
void CheckForExit()
  {
// Iterate through all open positions and close those that exceed the
// maximum number of data points specified in InpMaxPointsOpen.
   if(InpIntervalSeconds == 0)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      int points_open = (int)((TimeCurrent() - open_time) / (int)InpIntervalSeconds);

      if(points_open >= InpMaxPointsOpen)
         trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
//| Entry logic: trade on breakout from tight range                 |
//+------------------------------------------------------------------+
void CheckForEntry()
  {
   double high, low;
   if(!CalcRange(InpRangeDataPoints, high, low))
      return; // Range condition not met

   int size = ArraySize(g_prices);
   if(size < 2)
      return;

   double last_close = g_prices[size - 2];

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
   ArrayResize(g_prices, 0);
   g_test_start = 0;
   g_test_end = 0;
   g_total_points = 0;
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
   if(!IsNewInterval())
      return;

   RecordPrice();
   UpdateTestStats();

   CheckForExit();   // manage existing position
   CheckForEntry();  // look for new opportunity
  }

//+------------------------------------------------------------------+
//| Custom optimization criterion                                    |
//| y = T*Wt + P*Wp - D*Wd                                           |
//|   T : trade density = total trades / total data points           |
//|   P : monthly profit ratio = (total profit / starting equity) / months |
//|   D : relative drawdown percent from tester statistics           |
//+------------------------------------------------------------------+
double OnTester()
  {
   // Retrieve base statistics from the strategy tester
   double trades        = TesterStatistics(STAT_TRADES);                 // total number of trades
   double points        = (double)g_total_points;                        // total number of data points processed
   double profit        = TesterStatistics(STAT_PROFIT);                 // total net profit
   double startEquity   = TesterStatistics(STAT_INITIAL_DEPOSIT);        // starting equity
   double months        = CalcMonths(g_test_start, g_test_end);          // test length in months
   double drawdownPct   = TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100;   // relative drawdown

   double tradeDensity = 0.0;
   if(points > 0.0)
      tradeDensity = trades / points;

   double monthlyProfit = 0.0;
   if(months > 0.0 && startEquity > 0.0)
      monthlyProfit = (profit / startEquity) / months;

   // Normalise weights so they sum to 1.0 even if inputs don't add to 100
   double weightSum = InpWt + InpWp + InpWd;
   if(weightSum <= 0.0)
      weightSum = 1.0;
   double wt = InpWt / weightSum;
   double wp = InpWp / weightSum;
   double wd = InpWd / weightSum;


   // Objective value to maximise during optimisation
   double score = (tradeDensity * wt + monthlyProfit * wp - drawdownPct * wd)*100;
   return(score);
  }

