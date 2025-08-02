#property strict
#property description "Hover Breakout Expert Advisor"

#include <Trade/Trade.mqh>

input int    BackCandles    = 10;   // number of candles to define range
input int    RangePips      = 50;    // maximum range size in pips
input int    TPPips         = 12;   // take profit in pips
input int    SLPips         = 20;   // stop loss in pips
input int    FutureCandles  = 12;   // bars to hold the position
input double LotSize        = 0.1;  // trading volume
input uint   Slippage       = 10;   // slippage in points

CTrade trade;
datetime lastBarTime = 0;

void OnTick()
{
   datetime currentBar = iTime(_Symbol, _Period, 0);
   if(currentBar != lastBarTime)
   {
      lastBarTime = currentBar;
      CloseExpiredPositions();
      if(!PositionSelect(_Symbol))
         CheckForEntry();
   }
}

void CheckForEntry()
{
   double close = iClose(_Symbol, _Period, 1);
   double rangeHigh = iHigh(_Symbol, _Period, iHighest(_Symbol, _Period, MODE_HIGH, BackCandles, 2));
   double rangeLow  = iLow(_Symbol, _Period, iLowest(_Symbol, _Period, MODE_LOW, BackCandles, 2));

   if(rangeHigh - rangeLow > RangePips * _Point)
      return;

   if(close > rangeHigh)
      OpenPosition(ORDER_TYPE_BUY);
   else if(close < rangeLow)
      OpenPosition(ORDER_TYPE_SELL);
}

void OpenPosition(ENUM_ORDER_TYPE type)
{
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                           : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl, tp;

   if(type == ORDER_TYPE_BUY)
   {
      sl = price - SLPips * _Point;
      tp = price + TPPips * _Point;
      trade.SetDeviationInPoints(Slippage);
      trade.Buy(LotSize, _Symbol, price, sl, tp, "HoverBreakout");
      Print("this is a buy");
   }
   else
   {
      sl = price + SLPips * _Point;
      tp = price - TPPips * _Point;
      trade.SetDeviationInPoints(Slippage);
      trade.Sell(LotSize, _Symbol, price, sl, tp, "HoverBreakout");
      Print("this is a sell");
   }
}

void CloseExpiredPositions()
{
   if(PositionSelect(_Symbol))
   {
      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      int shift = iBarShift(_Symbol, _Period, openTime, true);
      if(shift >= FutureCandles)
         trade.PositionClose(_Symbol);
   }
}
