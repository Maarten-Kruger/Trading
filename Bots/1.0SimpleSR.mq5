//+------------------------------------------------------------------+
//|                                              1.0SimpleSR.mq5     |
//|                                  Copyright 2026, AI Collaborator |
//|                                             https://www.mql5.com |
//|                                                                  |
//| Description: A straightforward Support and Resistance bounce     |
//| EA. Identifies recent local extrema to place bounce trades       |
//| with dynamic risk management. Closes positions before the        |
//| weekend gaps.                                                    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI Collaborator"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input Parameters
input int      InpLookback      = 100;    // Bars to scan for S/R levels
input double   InpRiskPercent   = 1.0;    // Risk per trade (% of Account Balance)
input int      InpStopLoss      = 150;    // Stop Loss in Points (15 pips)
input int      InpTakeProfit    = 300;    // Take Profit in Points (30 pips)
input int      InpBuffer        = 20;     // Sensitivity Buffer in Points (2 pips)

//--- Global Variables
CTrade         trade;
double         supportLevels[];
double         resistanceLevels[];
long           MagicNumber      = 123456; // Standard global variable for Magic Number

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Configure the trade object with the universal Magic Number
   trade.SetExpertMagicNumber(MagicNumber);

   Print("SR Bounce Bot Initialized. Building foundation on the rock...");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Enforces the universal rule to close all trades on Friday 16:00  |
//| (GMT+2) and prevents new trades from opening.                    |
//+------------------------------------------------------------------+
bool CheckFridayClose()
{
   MqlDateTime dt;
   TimeGMT(dt);

   // Apply GMT+2 Offset
   dt.hour += 2;
   if(dt.hour >= 24)
   {
      dt.hour -= 24;
      dt.day_of_week++;
      if(dt.day_of_week > 6) dt.day_of_week = 0; // Wrap around to Sunday
   }

   // If it's Friday (day 5) and time is >= 16:00
   if(dt.day_of_week == 5 && dt.hour >= 16)
   {
      // Close all active positions for this EA
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            trade.PositionClose(ticket);
            Print("Friday 16:00 GMT+2 Reached. Closed position: ", ticket);
         }
      }
      return true; // Trading should be blocked
   }

   return false; // Safe to trade
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Universal Rule: Close all trades on Friday 16:00 (GMT+2) and stop trading
   if(CheckFridayClose()) return;

   // Check if we already have an open position
   if(PositionsTotal() > 0) return;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   static datetime lastBarTime = 0;
   static double currentSupport = 0;
   static double currentResistance = 0;

   // Identify the most recent major S/R levels only once per candle
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime != lastBarTime)
   {
      currentSupport = FindLocalExtrema(false); // Lowest low in lookback
      currentResistance = FindLocalExtrema(true); // Highest high in lookback
      lastBarTime = currentBarTime;
   }

   // Strategy Logic: Bounce off Support
   if(bid <= currentSupport + (InpBuffer * _Point) && bid >= currentSupport)
   {
      double sl = ask - (InpStopLoss * _Point);
      double tp = ask + (InpTakeProfit * _Point);
      double lotSize = CalculateLotSize(InpStopLoss);

      if(trade.Buy(lotSize, _Symbol, ask, sl, tp, "Support Bounce"))
         Print("Buy order placed at support: ", currentSupport, " | Lot Size: ", lotSize);
   }

   // Strategy Logic: Bounce off Resistance
   if(ask >= currentResistance - (InpBuffer * _Point) && ask <= currentResistance)
   {
      double sl = bid + (InpStopLoss * _Point);
      double tp = bid - (InpTakeProfit * _Point);
      double lotSize = CalculateLotSize(InpStopLoss);

      if(trade.Sell(lotSize, _Symbol, bid, sl, tp, "Resistance Bounce"))
         Print("Sell order placed at resistance: ", currentResistance, " | Lot Size: ", lotSize);
   }
}

//+------------------------------------------------------------------+
//| Find the highest high or lowest low in the lookback period       |
//+------------------------------------------------------------------+
double FindLocalExtrema(bool findMax)
{
   int highestBar = iHighest(_Symbol, _Period, MODE_HIGH, InpLookback, 1);
   int lowestBar  = iLowest(_Symbol, _Period, MODE_LOW, InpLookback, 1);

   if(findMax) return iHigh(_Symbol, _Period, highestBar);
   else        return iLow(_Symbol, _Period, lowestBar);
}

//+------------------------------------------------------------------+
//| Calculates the appropriate lot size based on account risk        |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopLossPoints)
{
   if(stopLossPoints <= 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   // Calculate risk amount based on percentage of balance
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (InpRiskPercent / 100.0);

   // Get tick value for calculating point value
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tickSize == 0 || tickValue == 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   // Calculate lot size: Risk Amount / (Stop Loss Points * Point Value)
   // Note: point value is tick_value * (_Point / tick_size)
   double pointValue = tickValue * (_Point / tickSize);
   double calculatedLot = riskAmount / (stopLossPoints * pointValue);

   // Constrain lot size to broker limits
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   // Round to nearest valid step
   calculatedLot = MathRound(calculatedLot / stepLot) * stepLot;

   if(calculatedLot < minLot) calculatedLot = minLot;
   if(calculatedLot > maxLot) calculatedLot = maxLot;

   return calculatedLot;
}