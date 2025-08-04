#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input int    InpRangeBars      = 10;     // Number of bars to define range
input double InpRangePoints    = 200;    // Maximum range size in points
input double InpTPPoints       = 400;    // Take profit distance in points
input double InpSLPoints       = 200;    // Stop loss distance in points
input double InpLots           = 0.10;   // Lot size
input int    InpMaxBarsOpen    = 5;      // Maximum bars to keep position open
input uint   InpSlippage       = 5;      // Slippage in points

//--- global objects
CTrade  trade;               // trading object

//+------------------------------------------------------------------+
//| Calculate hover range from past bars                            |
//| Returns true if bars stayed within specified range               |
//+------------------------------------------------------------------+
bool CalcRange(int bars_back, double &range_high, double &range_low)
{
   if(bars_back < 1)
      return(false);

   // initialise with most recently closed bar
   range_high = iHigh(_Symbol, _Period, 1);
   range_low  = iLow(_Symbol,  _Period, 1);

   for(int i = 2; i <= bars_back; i++)
   {
      double h = iHigh(_Symbol, _Period, i);
      double l = iLow(_Symbol,  _Period, i);
      if(h > range_high) range_high = h;
      if(l < range_low)  range_low  = l;
   }
   return((range_high - range_low) <= InpRangePoints * _Point);
}

//+------------------------------------------------------------------+
//| Close open position after a number of bars                       |
//+------------------------------------------------------------------+
void CheckForExit()
{
   if(!PositionSelect(_Symbol))
      return;

   datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
   int bars_open = iBarShift(_Symbol, _Period, open_time);

   // Close the position if it has been open for too many bars
   if(bars_open >= InpMaxBarsOpen)
      trade.PositionClose(_Symbol);
}

//+------------------------------------------------------------------+
//| Entry logic: trade on breakout from tight range                 |
//+------------------------------------------------------------------+
void CheckForEntry()
{
   if(PositionSelect(_Symbol))
      return; // Only one position at a time

   double high, low;
   if(!CalcRange(InpRangeBars, high, low))
      return; // Range condition not met


   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   trade.SetDeviationInPoints(InpSlippage);

   // Breakout above the range -> buy immediately
   if(ask > high)
   {
      double sl = ask - InpSLPoints * _Point;
      double tp = ask + InpTPPoints * _Point;
      trade.Buy(InpLots, _Symbol, ask, sl, tp, "HoverBreakout");
   }
   // Breakout below the range -> sell immediately
   else if(bid < low)
   {
      double sl = bid + InpSLPoints * _Point;
      double tp = bid - InpTPPoints * _Point;
      trade.Sell(InpLots, _Symbol, bid, sl, tp, "HoverBreakout");
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
}

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   CheckForExit();   // manage existing position
   CheckForEntry();  // look for new opportunity
}