//+------------------------------------------------------------------+
//|                  QuantLab_Executable_Lab_V6.mq5                  |
//|                            Copyright 2026, Trading Lab           |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMS
enum ENUM_TRIGGER_MODE {
   TRIG_MACD,     // 1. MACD Bullish
   TRIG_SMA2,     // 2. SMA 10 > 20
   TRIG_SMA3,     // 3. SMA 3-Way Stacked
   TRIG_STOCH,    // 4. Stoch Over 50
   TRIG_BANDS,    // 5. Price > Mid Band
   TRIG_RANDOM    // 6. Random (100% Chance)
};

enum ENUM_RISK_MODE {
   RISK_STANDARD,               // 1. Standard (Max Candles + SL)
   RISK_WEEKDAYS_ALL,           // 2. Weekdays: Close ALL Friday 16:00
   RISK_WEEKDAYS_HOLD_NEG,      // 3. Weekdays: Close Winners Friday, Hold Losers
   RISK_WEEKDAYS_HOLD_POS,      // 4. Weekdays: Close Losers Friday, Hold Winners
   RISK_WEEKDAYS_ATR_TRAIL,     // 5. Weekdays: ATR Trailing Stop (+ Close ALL Friday)
   RISK_WEEKDAYS_HEDGE,         // 6. Weekdays: Dynamic MA Hedge (+ Close ALL Friday)
   RISK_WEEKDAYS_HOLD_NEG_DD    // 7. Weekdays: Close Winners Friday, Pause on 5% DD
};
//--- INPUT PARAMETERS
input ENUM_TRIGGER_MODE InpTriggerMode   = TRIG_MACD;           // Trigger Mechanism
input ENUM_RISK_MODE    InpRiskMode      = RISK_WEEKDAYS_HOLD_NEG; // Risk Management Strategy

input double            InpTargetPercentReward = 2.0;           // Target Profit % of Balance
input double            InpBin3TargetPercent   = 10.0;          // Bin 3 Target Profit % for OnTester
input int               InpTargetPips    = 100;                 // Target (Pips)
input int               InpMaxOpenTrades = 5;                   // Max concurrent main trades
input long              InpMagicNum      = 987654;              // EA Magic Number

// Standard Mode Specifics
input int               InpMaxCandles    = 500;                 // Max holding bars (Standard Mode)
input double            InpRiskReward    = 2.0; // Risk:Reward Ratio (e.g., 2.0 means TP is 2x SL)

// Hedge Mode Specifics
input int               InpHedgePips     = 1000;                // Pips down to trigger Hedge
input int               InpMAPeriod      = 14;                  // MA Period for Hedge

// ATR Trailing Mode Specifics
input int               InpATRPeriod     = 14;                  // ATR Period
input double            InpATRMult       = 2.0;                 // ATR Multiplier for Trail Distance

// Base Filter
input int               InpTrendPeriod   = 200;                 // Downtrend Filter EMA

// Drawdown Pause Mode Specifics
input double            InpMaxDrawdownPercent = 5.0; // Max Drawdown % before pausing

//--- GLOBALS
double PipPoint;
CTrade trade;

// Handles
int hMACD, hSMA10, hSMA20, hSMA50, hTrend, hStoch, hBands, hHedgeMA, hATR;


// Stats
int TotalOpened = 0;
int TotalTPHits = 0;
int TotalClosedByRisk = 0;

// Optimization History (In-Memory Array)
struct EquityRecord {
   datetime time;
   double   equity;
};
EquityRecord EquityHistory[];
int HistoryCount = 0;


//+------------------------------------------------------------------+
//| Helper: Find if a hedge exists for a specific main ticket        |
//+------------------------------------------------------------------+
ulong GetHedgeTicket(ulong main_ticket) {
   string target_comment = "H_" + IntegerToString(main_ticket);
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)) {
         if(PositionGetInteger(POSITION_MAGIC) == InpMagicNum + 1) {
            string comment = PositionGetString(POSITION_COMMENT);
            if(StringFind(comment, target_comment) >= 0) return ticket;
         }
      }
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   PipPoint = (digits == 3 || digits == 5) ? 10 * point : point;

   trade.SetExpertMagicNumber(InpMagicNum);

   hMACD    = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);
   hSMA10   = iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE);
   hSMA20   = iMA(_Symbol, _Period, 20, 0, MODE_SMA, PRICE_CLOSE);
   hSMA50   = iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE);
   hTrend   = iMA(_Symbol, _Period, InpTrendPeriod, 0, MODE_EMA, PRICE_CLOSE);
   hStoch   = iStochastic(_Symbol, _Period, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   hBands   = iBands(_Symbol, _Period, 20, 0, 2.0, PRICE_CLOSE);

   if(InpRiskMode == RISK_WEEKDAYS_HEDGE) {
      hHedgeMA = iMA(_Symbol, _Period, InpMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   }
   if(InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
      hATR = iATR(_Symbol, _Period, InpATRPeriod);
   }


   Print("--- Executable Lab V6 Started ---");
   PrintFormat("Testing: %s with %s", EnumToString(InpTriggerMode), EnumToString(InpRiskMode));

   // Initialize Timer for minute-by-minute equity tracking (every 60 seconds)
   EventSetTimer(60);

   return(INIT_SUCCEEDED);

}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   double curPrice = iClose(_Symbol, _Period, 0);

   // ==========================================
   // 1. RISK MANAGEMENT MODULE (EXITS & HEDGES)
   // ==========================================

   // A. Friday Expirations (All Weekdays Modes)
   bool isFridayClose = (dt.day_of_week == 5 && dt.hour >= 16);
   if(isFridayClose && InpRiskMode != RISK_STANDARD) {
      for(int i = PositionsTotal() - 1; i >= 0; i--) {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket)) {
            long magic = PositionGetInteger(POSITION_MAGIC);

            // Only evaluate our Main Trades (and Hedge trades if they exist)
            if(magic == InpMagicNum || magic == (InpMagicNum + 1)) {
               double profit = PositionGetDouble(POSITION_PROFIT);
               bool shouldClose = false;

               // Determine if this specific trade should be closed based on the selected Risk Mode
               if(InpRiskMode == RISK_WEEKDAYS_ALL || InpRiskMode == RISK_WEEKDAYS_HEDGE || InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
                  shouldClose = true;
               }
               else if((InpRiskMode == RISK_WEEKDAYS_HOLD_NEG || InpRiskMode == RISK_WEEKDAYS_HOLD_NEG_DD) && profit > 0) {
                  shouldClose = true; // Close the winners, let losers run
               }
               else if(InpRiskMode == RISK_WEEKDAYS_HOLD_POS && profit < 0) {
                  shouldClose = true; // Close the losers, let winners run
               }

               if(shouldClose) {
                  if(trade.PositionClose(ticket)) {
                     if(magic == InpMagicNum) TotalClosedByRisk++;
                  }
               }
            }
         }
      }
      return; // Skip entry logic during the Friday close window
   }

   // B. Max Candles Expiry (Universal for all modes)
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum) {
         datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
         int barsPassed = iBarShift(_Symbol, _Period, openTime);
         if(barsPassed >= InpMaxCandles) {
            if(trade.PositionClose(ticket)) TotalClosedByRisk++;
         }
      }
   }

   // C. Dynamic Hedging Engine
   if(InpRiskMode == RISK_WEEKDAYS_HEDGE) {
      double maVal[]; ArraySetAsSeries(maVal, true);
      if(CopyBuffer(hHedgeMA, 0, 0, 1, maVal) > 0) {
         double currentMA = maVal[0];

         for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {

               double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
               double volume = PositionGetDouble(POSITION_VOLUME);
               double hedgeThreshold = openPrice - (InpHedgePips * PipPoint);
               ulong existingHedge = GetHedgeTicket(ticket);

               // Trigger Hedge
               if(currentMA <= hedgeThreshold && existingHedge == 0) {
                  trade.SetExpertMagicNumber(InpMagicNum + 1);
                  double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                  trade.Sell(volume, _Symbol, bid, 0, 0, "H_" + IntegerToString(ticket));
                  trade.SetExpertMagicNumber(InpMagicNum);
               }
               // Remove Hedge
               else if(currentMA > hedgeThreshold && existingHedge > 0) {
                  trade.PositionClose(existingHedge);
               }
            }
         }
      }
   }

   // D. ATR Trailing Stop
   if(InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
      double atrVal[]; ArraySetAsSeries(atrVal, true);
      if(CopyBuffer(hATR, 0, 0, 1, atrVal) > 0) {
         double currentATR = atrVal[0];
         for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum) {
               double currentSL = PositionGetDouble(POSITION_SL);
               double currentTP = PositionGetDouble(POSITION_TP);
               double proposedSL = curPrice - (currentATR * InpATRMult);

               // Only move the Stop Loss UP (trailing), never down.
               // Adding a 5 pip buffer to prevent server spam for micro-movements.
               if(proposedSL > currentSL + (5 * PipPoint)) {
                  trade.PositionModify(ticket, proposedSL, currentTP);
               }
            }
         }
      }
   }


   // ==========================================
   // 2. TRIGGER MODULE (ENTRIES)
   // ==========================================
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, _Period, 0);

   if(currentBar != lastBar) {
      lastBar = currentBar;

      // Downtrend Filter
      double ema[2]; CopyBuffer(hTrend, 0, 0, 2, ema);
      if(curPrice < ema[0] && ema[0] < ema[1]) return;

      // Day Filter (Only Standard mode can open on any day)
      if(InpRiskMode != RISK_STANDARD) {
         if(dt.day_of_week != 1 && dt.day_of_week != 2) return;
      }

      // Drawdown Pause & Recovery Logic
      static bool isPausedForDD = false;
      if(InpRiskMode == RISK_WEEKDAYS_HOLD_NEG_DD) {
         double balance = AccountInfoDouble(ACCOUNT_BALANCE);
         double equity = AccountInfoDouble(ACCOUNT_EQUITY);

         // Calculate floating drawdown percentage
         double ddPercent = (balance > 0) ? ((balance - equity) / balance * 100.0) : 0;

         // Lock out new trades if DD exceeds the user-defined threshold
         if(ddPercent >= InpMaxDrawdownPercent) {
            isPausedForDD = true;
         }
         // Un-pause only when equity fully recovers to balance (0% DD)
         else if(ddPercent <= 0.0) {
            isPausedForDD = false;
         }

         // Abort entry evaluation if currently in a timeout state
         if(isPausedForDD) return;
      }

      // Count Main Trades
      int mainTradesCount = 0;
      for(int i = 0; i < PositionsTotal(); i++) {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum)
            mainTradesCount++;
      }

      // Evaluate Entry
      if(mainTradesCount < InpMaxOpenTrades) {
         bool triggerMet = false;

         switch(InpTriggerMode) {
            case TRIG_MACD: {
               double m[1], s[1]; CopyBuffer(hMACD, 0, 0, 1, m); CopyBuffer(hMACD, 1, 0, 1, s);
               triggerMet = (m[0] > s[0]); break;
            }
            case TRIG_SMA2: {
               double f[1], s[1]; CopyBuffer(hSMA10, 0, 0, 1, f); CopyBuffer(hSMA20, 0, 0, 1, s);
               triggerMet = (f[0] > s[0]); break;
            }
            case TRIG_SMA3: {
               double f[1], m[1], s[1]; CopyBuffer(hSMA10, 0, 0, 1, f); CopyBuffer(hSMA20, 0, 0, 1, m); CopyBuffer(hSMA50, 0, 0, 1, s);
               triggerMet = (f[0] > m[0] && m[0] > s[0]); break;
            }
            case TRIG_STOCH: {
               double k[1]; CopyBuffer(hStoch, 0, 0, 1, k);
               triggerMet = (k[0] > 50); break;
            }
            case TRIG_BANDS: {
               double mid[1]; CopyBuffer(hBands, 0, 0, 1, mid);
               triggerMet = (curPrice > mid[0]); break;
            }
            case TRIG_RANDOM: {
               // triggerMet = (MathRand() % 100 < 50); break;
               triggerMet = true; break;
            }
         }

         if(triggerMet) {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double tp = ask + (InpTargetPips * PipPoint);
            double sl = 0;

            // Calculate initial Stop Loss based on Risk Mode

            if(InpRiskMode == RISK_STANDARD) {
               // If RR is 0 (or negative), disable the Stop Loss entirely
               if(InpRiskReward <= 0.0) {
                  sl = 0.0;
               }
               else {
                  // Calculate Risk in pips based on the Reward (InpTargetPips) and the RR ratio
                  double slPips = InpTargetPips / InpRiskReward;
                  sl = ask - (slPips * PipPoint);
               }
            }
            else if(InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
               double atrVal[]; ArraySetAsSeries(atrVal, true);
               if(CopyBuffer(hATR, 0, 0, 1, atrVal) > 0) {
                  sl = ask - (atrVal[0] * InpATRMult); // Initial ATR Stop
               }
            }

            // Calculate dynamic lot size based on target profit percentage of balance
            double balance = AccountInfoDouble(ACCOUNT_BALANCE);
            double targetProfitValue = balance * (InpTargetPercentReward / 100.0);

            double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
            double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);

            double targetPipsRaw = (tp - ask); // in price units
            double targetTicks = targetPipsRaw / tickSize;

            double calculatedLots = 0.0;
            if (targetTicks > 0 && tickValue > 0) {
                calculatedLots = targetProfitValue / (targetTicks * tickValue);
            }

            // Constrain lot size
            double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
            double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

            if (calculatedLots < minLot) calculatedLots = minLot;
            if (calculatedLots > maxLot) calculatedLots = maxLot;

            // Round lot size to step
            int steps = (int)MathRound(calculatedLots / lotStep);
            double lots = steps * lotStep;

            if(trade.Buy(lots, _Symbol, ask, sl, tp, EnumToString(InpTriggerMode))) {
               TotalOpened++;
            }

         }
      }
   }
}


//+------------------------------------------------------------------+
//| Timer function for recording Minute-by-Minute Equity             |
//+------------------------------------------------------------------+
void OnTimer() {
   // Resize array in chunks to avoid constant memory reallocation
   if (HistoryCount >= ArraySize(EquityHistory)) {
      ArrayResize(EquityHistory, ArraySize(EquityHistory) + 10000);
   }

   EquityHistory[HistoryCount].time = TimeCurrent();
   EquityHistory[HistoryCount].equity = AccountInfoDouble(ACCOUNT_EQUITY);
   HistoryCount++;

   /* --- ALTERNATIVE CSV EXPORT METHOD (Commented out per request) ---
   int handle = FileOpen("EquityHistory.csv", FILE_WRITE|FILE_CSV|FILE_READ, ',');
   if(handle != INVALID_HANDLE) {
      FileSeek(handle, 0, SEEK_END);
      FileWrite(handle, TimeToString(TimeCurrent()), AccountInfoDouble(ACCOUNT_EQUITY));
      FileClose(handle);
   }
   ------------------------------------------------------------------ */
}

//+------------------------------------------------------------------+
//| Trade Transaction (Track TP Wins & SL Hits)                      |

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      if(HistoryDealSelect(trans.deal)) {
         long magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         long reason = HistoryDealGetInteger(trans.deal, DEAL_REASON);

         if(magic == InpMagicNum && entry == DEAL_ENTRY_OUT) {
            if(reason == DEAL_REASON_TP) {
               TotalTPHits++;
            }
            else if(reason == DEAL_REASON_SL && (InpRiskMode == RISK_STANDARD || InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL)) {
               TotalClosedByRisk++; // Track Stop Loss hits correctly
            }
         }
      }
   }
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();

   int completed = TotalTPHits + TotalClosedByRisk;
   double winRate = (completed > 0) ? ((double)TotalTPHits / completed) * 100.0 : 0;

   Print("===============================================");
   PrintFormat("TRIGGER: %s | RISK: %s", EnumToString(InpTriggerMode), EnumToString(InpRiskMode));
   Print("===============================================");
   PrintFormat("Main Trades Opened: %d", TotalOpened);
   PrintFormat("Hits TP (Wins):     %d", TotalTPHits);
   PrintFormat("Closed by Risk:     %d", TotalClosedByRisk);
   PrintFormat("Actual Win Rate:    %.2f%%", winRate);
   Print("===============================================");
}
//+------------------------------------------------------------------+
//| Tester function (Custom Optimization Score)                      |
//+------------------------------------------------------------------+
double OnTester() {
   int bin1Count = 0; // Dropped 5% within 24 hours
   int bin2Count = 0; // Dropped 10% overall
   int bin3Count = 0; // Gained Target% overall

   /* --- ALTERNATIVE CSV READ METHOD (Commented out) ---
   // If you used the CSV write method in OnTimer, you would read it here
   int handle = FileOpen("EquityHistory.csv", FILE_READ|FILE_CSV, ',');
   if(handle != INVALID_HANDLE) {
      while(!FileIsEnding(handle)) {
         string tStr = FileReadString(handle);
         double eq = FileReadNumber(handle);
         // Process logic...
      }
      FileClose(handle);
   }
   ------------------------------------------------------ */

   // We simulate a $10,000 account at each recorded point.
   // This means our simulated starting equity is 10,000.
   // - Bin 1: Equity drops below 9,500 (-5%) within 24 hours (86400 seconds).
   // - Bin 2: Equity drops below 9,000 (-10%) at any point.
   // - Bin 3: Equity rises above (10,000 * (1 + InpBin3TargetPercent/100)) at any point.

   double startEquitySim = 10000.0;
   double bin1Threshold = startEquitySim * 0.95;
   double bin2Threshold = startEquitySim * 0.90;
   double bin3Threshold = startEquitySim * (1.0 + (InpBin3TargetPercent / 100.0));

   for(int i = 0; i < HistoryCount; i++) {
      datetime startTime = EquityHistory[i].time;
      double startRealEquity = EquityHistory[i].equity;

      // Look ahead in history to find the outcome
      for(int j = i + 1; j < HistoryCount; j++) {
         datetime curTime = EquityHistory[j].time;
         double curRealEquity = EquityHistory[j].equity;

         // Calculate simulated equity relative to the starting point
         double relativeChange = curRealEquity / startRealEquity;
         double simulatedEquity = startEquitySim * relativeChange;

         // Check Bin 1 (-5% within 24 hours)
         if(simulatedEquity <= bin1Threshold && (curTime - startTime) <= 86400) {
            bin1Count++;
            break; // Stop looking ahead for this point
         }

         // Check Bin 2 (-10% overall)
         if(simulatedEquity <= bin2Threshold) {
            bin2Count++;
            break; // Stop looking ahead for this point
         }

         // Check Bin 3 (+Target% overall)
         if(simulatedEquity >= bin3Threshold) {
            bin3Count++;
            break; // Stop looking ahead for this point
         }
      }
   }

   int totalBins = bin1Count + bin2Count + bin3Count;

   if(totalBins == 0) return 0.0; // Avoid division by zero

   // The final score is the percentage of points that successfully hit Bin 3
   double resultScore = ((double)bin3Count / totalBins) * 100.0;

   PrintFormat("OnTester Results: Bin1 (-5%% in 24h) = %d, Bin2 (-10%%) = %d, Bin3 (+%.2f%%) = %d | Score = %.2f%%",
               bin1Count, bin2Count, InpBin3TargetPercent, bin3Count, resultScore);

   return resultScore;
}
