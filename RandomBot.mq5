#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input double InpLots      = 0.10;  // Lot size
input uint   InpSlippage  = 5;     // Slippage in points

CTrade  trade;               // trading object

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
//| Expert initialization                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   MathSrand((uint)TimeLocal());
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
   if(!IsNewBar())
      return;

   trade.SetDeviationInPoints(InpSlippage);

   if(PositionSelect(_Symbol))
      trade.PositionClose(_Symbol);

   if(MathRand() % 2 == 0)
      trade.Buy(InpLots, _Symbol);
   else
      trade.Sell(InpLots, _Symbol);
  }
