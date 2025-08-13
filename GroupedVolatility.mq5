#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input double InpRiskPercent     = 1.0;    // Percent of equity to risk
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
//| Helper: calculate trade volume based on equity risk               |
//+------------------------------------------------------------------+
double VolumeForRisk(double stopLossPips, double riskPercent)
  {
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money  = equity * riskPercent / 100.0;

   double tick_value  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pip_value   = tick_value * PipsToPrice(1.0) / tick_size; // value of one pip per lot
   double risk_per_lot = stopLossPips * pip_value;
   if(risk_per_lot <= 0.0)
      return(0.0);

   double volume = risk_money / risk_per_lot;

   double minVol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxVol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   volume = MathFloor(volume / stepVol) * stepVol; // align with volume step
   volume = MathMax(minVol, MathMin(maxVol, volume));

   return(volume);
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
   double volume   = VolumeForRisk(InpStopLossPips, InpRiskPercent);

   if(volume <= 0.0)
      return;                          // volume could not be calculated

   trade.SetDeviationInPoints(InpSlippage);

   if(isBull)
      trade.Buy(volume, _Symbol, price, sl, tp);
   else
      trade.Sell(volume, _Symbol, price, sl, tp);
  }
