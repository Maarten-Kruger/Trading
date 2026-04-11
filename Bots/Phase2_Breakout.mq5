//+------------------------------------------------------------------+
//|                                              Phase2_Breakout.mq5 |
//|                                      US500 Long-Only Breakout EA |
//+------------------------------------------------------------------+
#property copyright "Custom Build"
#property link      ""
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- Input Parameters
input group "Strategy Settings"
input int    InpLookbackCandles = 200;      // Lookback Candles
input int    InpBinSizePoints   = 10;       // Bin Size (in Points)
input int    InpMinBinTouches   = 5;        // Minimum Bin Touches (Hits)
input int    InpATRPeriod       = 14;       // ATR Period
input double InpATRMultiplier   = 1.5;      // ATR Multiplier for Buffer Zone

input group "Trade Settings"
input double InpLotSize         = 0.1;      // Trade Lot Size
input double InpRiskReward      = 2.0;      // Risk:Reward Ratio

//--- Global Variables
CTrade         trade;
int            atrHandle;
datetime       lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialize ATR Indicator
   atrHandle = iATR(_Symbol, _Period, InpATRPeriod);
   if(atrHandle == INVALID_HANDLE)
     {
      Print("Error creating ATR indicator handle!");
      return(INIT_FAILED);
     }
     
   trade.SetExpertMagicNumber(123456);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(atrHandle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // 1. Only execute once per new bar
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime == lastBarTime) return; 

   // We have a new bar, which means Candle 1 just closed
   if(!CheckForBreakout()) return;
   
   lastBarTime = currentBarTime; // Update time only if we successfully checked
  }

//+------------------------------------------------------------------+
//| Core Logic: Map Zones & Check Breakout                           |
//+------------------------------------------------------------------+
bool CheckForBreakout()
  {
   // Array setup for historical data
   double high[], low[], open[], close[], atr[];
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(atr, true);

   // Fetch data (Lookback + 2 to ensure we have Candle 1)
   int copiedH = CopyHigh(_Symbol, _Period, 1, InpLookbackCandles, high);
   int copiedL = CopyLow(_Symbol, _Period, 1, InpLookbackCandles, low);
   int copiedO = CopyOpen(_Symbol, _Period, 1, 2, open);
   int copiedC = CopyClose(_Symbol, _Period, 1, 2, close);
   int copiedATR = CopyBuffer(atrHandle, 0, 1, 1, atr);

   if(copiedH < InpLookbackCandles || copiedL < InpLookbackCandles || copiedATR <= 0) 
       return false; // Not enough data yet

   // 1. Calculate Histogram Min/Max Bounds over Lookback
   double minPrice = low[0];
   double maxPrice = high[0];
   for(int i = 0; i < InpLookbackCandles; i++)
     {
      if(low[i] < minPrice) minPrice = low[i];
      if(high[i] > maxPrice) maxPrice = high[i];
     }

   // 2. Build the Histogram Bins
   double binSize = InpBinSizePoints * _Point; 
   if(binSize <= 0) return false;
   
   int numBins = (int)MathCeil((maxPrice - minPrice) / binSize) + 1;
   int hits[];
   ArrayResize(hits, numBins);
   ArrayInitialize(hits, 0);

   // 3. Populate Hits (Wick-to-Wick filling, similar to Python's searchsorted logic)
   for(int i = 0; i < InpLookbackCandles; i++)
     {
      int startIdx = (int)MathFloor((low[i] - minPrice) / binSize);
      int endIdx   = (int)MathFloor((high[i] - minPrice) / binSize);
      
      startIdx = MathMax(0, startIdx);
      endIdx   = MathMin(numBins - 1, endIdx);
      
      for(int j = startIdx; j <= endIdx; j++)
        {
         hits[j]++;
        }
     }

   // 4. Calculate Current Buffer Value based on ATR
   double bufferVal = atr[0] * InpATRMultiplier;

   // Get Candle 1's Open and Close
   double c1_open = open[0];
   double c1_close = close[0];

   // 5. Scan established SR lines for a Breakout Trigger
   for(int k = 0; k < numBins; k++)
     {
      if(hits[k] >= InpMinBinTouches)
        {
         // Calculate the center of this bin
         double srLine = minPrice + (k * binSize) + (binSize / 2.0);
         
         double zoneTop = srLine + bufferVal;
         double zoneBottom = srLine - bufferVal;

         // BREAKOUT CONDITION:
         // 1. The candle opened INSIDE the zone
         bool openedInZone = (c1_open >= zoneBottom && c1_open <= zoneTop);
         
         // 2. The candle closed ABOVE the zone top
         bool closedAboveZone = (c1_close > zoneTop);

         if(openedInZone && closedAboveZone)
           {
            // Trigger Found! Execute Long Trade
            ExecuteBuyTrade(zoneBottom);
            return true; // Exit after one setup
           }
        }
     }
     
   return true; // Scan completed, no trigger
  }

//+------------------------------------------------------------------+
//| Execute Trade with Dynamic Risk:Reward                           |
//+------------------------------------------------------------------+
void ExecuteBuyTrade(double oppositeZoneLine)
  {
   // Check if we already have an open position to prevent spamming
   if(PositionsTotal() > 0) return; 

   double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // 1. SL is at the opposite side of the zone (Zone Bottom)
   double sl = NormalizeDouble(oppositeZoneLine, _Digits);
   
   // Safety check: ensure SL is actually below current price
   if(sl >= askPrice) 
   {
      Print("Invalid SL calculation. Skipping trade.");
      return;
   }

   // 2. Calculate Risk in Price terms
   double riskPoints = askPrice - sl;
   
   // 3. Calculate TP based on RR
   double tp = NormalizeDouble(askPrice + (riskPoints * InpRiskReward), _Digits);

   // 4. Send Order
   trade.Buy(InpLotSize, _Symbol, askPrice, sl, tp, "Breakout Buy");
   
   PrintFormat("LONG Triggered! Entry: %f, SL: %f, TP: %f", askPrice, sl, tp);
  }
//+------------------------------------------------------------------+