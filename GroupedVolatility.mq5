#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input double InpLots            = 0.10;   // Lot size
input uint   InpSlippage        = 5;      // Slippage in points
input int    InpLookback        = 2;      // Number of candles to group
input double InpMinPipSize      = 10;     // Minimum candle size in pips
input bool   InpReverse         = false;  // Trade opposite direction
input double InpStopLossPips    = 15;     // Stop loss distance in pips
input double InpRiskReward      = 2.0;    // Risk to reward ratio

CTrade trade;                         // trading object

//+------------------------------------------------------------------+
//| Helper: convert pips to price distance                            |
//+------------------------------------------------------------------+
double PipsToPrice(const double pips)
  {
   double pip = _Point;
   if(_Digits == 3 || _Digits == 5)
      pip *= 10.0;
   return(pips * pip);
  }

//+------------------------------------------------------------------+
//| Helper: detect new bar                                           |
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
//| Helper: check if recent candles are large                        |
//+------------------------------------------------------------------+
bool AreGroupedCandlesLarge(int lookback, double minPips)
  {
   for(int i=1; i<=lookback; i++)
     {
      double high = iHigh(_Symbol, _Period, i);
      double low  = iLow(_Symbol, _Period, i);
      double rangePips = (high - low) / PipsToPrice(1.0);
      if(rangePips < minPips)
         return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar())
      return;                          // work only on new bar

   if(PositionSelect(_Symbol))
      return;                          // only one position at a time

   if(!AreGroupedCandlesLarge(InpLookback, InpMinPipSize))
      return;                          // recent candles not large enough

   bool isBull = iClose(_Symbol, _Period, 1) > iOpen(_Symbol, _Period, 1);
   if(InpReverse)
      isBull = !isBull;                // optionally flip trade direction

   double price    = isBull ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl_dist  = PipsToPrice(InpStopLossPips);
   double tp_dist  = sl_dist * InpRiskReward;
   double sl       = isBull ? price - sl_dist : price + sl_dist;
   double tp       = isBull ? price + tp_dist : price - tp_dist;

   trade.SetDeviationInPoints(InpSlippage);

   if(isBull)
      trade.Buy(InpLots, _Symbol, price, sl, tp);
   else
      trade.Sell(InpLots, _Symbol, price, sl, tp);
  }
