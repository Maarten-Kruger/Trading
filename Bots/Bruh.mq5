//+------------------------------------------------------------------+
//|                                              SR_Bounce_Bot.mq5   |
//|                                  Copyright 2026, AI Collaborator |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI Collaborator"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input Parameters
input int      InpLookback      = 100;    // Bars to scan for S/R levels
input double   InpLotSize       = 0.1;    // Trading Lot Size
input int      InpStopLoss      = 150;    // Stop Loss in Points (15 pips)
input int      InpTakeProfit    = 300;    // Take Profit in Points (30 pips)
input int      InpBuffer        = 20;     // Sensitivity Buffer in Points (2 pips)

//--- Global Variables
CTrade         trade;
double         supportLevels[];
double         resistanceLevels[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("SR Bounce Bot Initialized. Building foundation on the rock...");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if we already have an open position
   if(PositionsTotal() > 0) return;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Identify the most recent major S/R levels
   double currentSupport = FindLocalExtrema(false); // Lowest low in lookback
   double currentResistance = FindLocalExtrema(true); // Highest high in lookback

   // Strategy Logic: Bounce off Support
   if(bid <= currentSupport + (InpBuffer * _Point) && bid >= currentSupport)
   {
      double sl = ask - (InpStopLoss * _Point);
      double tp = ask + (InpTakeProfit * _Point);
      
      if(trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "Support Bounce"))
         Print("Buy order placed at support: ", currentSupport);
   }

   // Strategy Logic: Bounce off Resistance
   if(ask >= currentResistance - (InpBuffer * _Point) && ask <= currentResistance)
   {
      double sl = bid + (InpStopLoss * _Point);
      double tp = bid - (InpTakeProfit * _Point);
      
      if(trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "Resistance Bounce"))
         Print("Sell order placed at resistance: ", currentResistance);
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