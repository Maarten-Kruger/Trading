#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.3"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input int    InpRangeBars      = 10;     // Number of bars to define range (excluding breakout bar)
input double InpRangePoints    = 200;    // Maximum range size in points
input double InpTPPoints       = 400;    // Take profit distance in points
input double InpSLPoints       = 200;    // Stop loss distance in points
input double InpRiskPercent    = 1.0;    // Risk percentage of equity per trade
input int    InpMaxBarsOpen    = 5;      // Maximum bars to keep position open
input uint   InpSlippage       = 5;      // Slippage in points
input double InpWt             = 40.0;   // Weight % for trade density
input double InpWp             = 40.0;   // Weight % for monthly consistency
input double InpWd             = 20.0;   // Weight % for drawdown

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
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      if(symbol != _Symbol)
         continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      int bars_open = iBarShift(_Symbol, _Period, open_time);

      if(bars_open >= InpMaxBarsOpen)
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
//| Custom optimization criterion                                    |
//| y = T*Wt + P*Wp - D*Wd                                           |
//|   T : trade density = total trades / total bars                  |
//|   P : monthly profit ratio = (total profit / starting equity) / months |
//|   D : relative drawdown percent from tester statistics           |
//+------------------------------------------------------------------+
double OnTester()
  {
   // Retrieve base statistics from the strategy tester
   double trades        = TesterStatistics(STAT_TRADES);                 // total number of trades
   double bars          = (double)g_total_bars;                          // total number of bars processed
   double profit        = TesterStatistics(STAT_PROFIT);                 // total net profit
   double startEquity   = TesterStatistics(STAT_INITIAL_DEPOSIT);        // starting equity
   double months        = CalcMonths(g_test_start, g_test_end);          // test length in months
   double drawdownPct   = TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100;   // relative drawdown

   double tradeDensity = 0.0;
   if(bars > 0.0)
      tradeDensity = trades / bars;

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
   double score = tradeDensity * wt + monthlyProfit * wp - drawdownPct * wd;
   printf("Trades percentage number thingy = " + tradeDensity);
   printf("Monthly Profit percentage number thingy = " + monthlyProfit);
   printf("Drawdown percentage number thingy = " + drawdownPct);
   printf("Bars percentage number thingy = " + bars);
   printf("Months percentage number thingy = " + months);
   
   
   return(score);
  }

